//! When the homepage snapshot is (re)built, and how far a build has got.
//!
//! The snapshot used to be built lazily by the first request after a server
//! start, then rebuilt every 15 minutes whether or not anything had changed.
//! Now:
//!
//! - [`start_background_refresh`] builds it as soon as the server starts,
//!   then checks for changes every [`CHECK_INTERVAL`];
//! - a check compares a cheap [`DataMarker`] read from the database with the
//!   one the served snapshot was built from ([`needs_rebuild`]), and only
//!   rebuilds on a difference;
//! - the Refresh controls run the same check on demand, or force a rebuild,
//!   through [`request_refresh`];
//! - every build step reports its completion ([`tracked`]), so [`status`]
//!   can tell the client how far the build has got, or why it failed.
//!
//! Server-only, like [`crate::homepage::snapshot`].

use std::future::Future;
use std::sync::{Mutex, MutexGuard, Once, PoisonError};
use std::time::{Duration, Instant};

use crate::get_pool;
use crate::homepage::refresh_status::{BuildProgress, RefreshOutcome, RefreshPhase, RefreshStatus};
use crate::homepage::snapshot::{self, DataMarker, SnapshotBuildError};

/// How often the background task checks the database for changes. The check
/// reads two catalog values and one statistics view, so it is cheap enough
/// to run every minute.
pub const CHECK_INTERVAL: Duration = Duration::from_secs(60);

/// One step of [`snapshot::build`]. The queries run concurrently, so steps
/// complete in any order; their declaration order is only the order the
/// "current step" label is picked in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildStep {
    DataMarker,
    Branches,
    HypothesisCount,
    ArchivedCount,
    Eligibility,
    LatestFits,
    FitFailures,
    CrossMatches,
    BestOrbits,
    Indexing,
}

impl BuildStep {
    /// Every step, in label order.
    pub const ALL: [BuildStep; 10] = [
        BuildStep::DataMarker,
        BuildStep::Branches,
        BuildStep::HypothesisCount,
        BuildStep::ArchivedCount,
        BuildStep::Eligibility,
        BuildStep::LatestFits,
        BuildStep::FitFailures,
        BuildStep::CrossMatches,
        BuildStep::BestOrbits,
        BuildStep::Indexing,
    ];

    /// Label shown to the user while this step is the first one pending.
    pub fn label(self) -> &'static str {
        match self {
            BuildStep::DataMarker => "Checking the data version",
            BuildStep::Branches => "Reading branches",
            BuildStep::HypothesisCount => "Counting hypotheses",
            BuildStep::ArchivedCount => "Counting archived trajectories",
            BuildStep::Eligibility => "Checking fit eligibility",
            BuildStep::LatestFits => "Reading the latest orbit fits",
            BuildStep::FitFailures => "Reading fit failures",
            BuildStep::CrossMatches => "Reading cross-matches",
            BuildStep::BestOrbits => "Reading the best orbits",
            BuildStep::Indexing => "Indexing and encoding",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// A build in progress.
struct RunningBuild {
    started_at: Instant,
    done: [bool; BuildStep::ALL.len()],
}

/// The single build slot: at most one build runs at a time.
struct BuildState {
    running: Option<RunningBuild>,
    /// Message of the last failed check or build, cleared when a build starts.
    last_error: Option<String>,
}

static BUILD_STATE: Mutex<BuildState> = Mutex::new(BuildState {
    running: None,
    last_error: None,
});

/// Locks [`BUILD_STATE`]. A panic while holding the lock cannot leave it
/// inconsistent (every critical section is a few field assignments), so a
/// poisoned lock is simply recovered.
fn lock_state() -> MutexGuard<'static, BuildState> {
    BUILD_STATE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Runs one build step and records its completion.
///
/// # Arguments
/// * `step` — the step `future` performs.
/// * `future` — the step itself.
///
/// # Return
/// Whatever `future` returns; the step counts as done even if it failed,
/// since the whole build then fails anyway.
pub async fn tracked<T>(step: BuildStep, future: impl Future<Output = T>) -> T {
    let output = future.await;
    if let Some(running) = lock_state().running.as_mut() {
        running.done[step.index()] = true;
    }
    output
}

/// Progress of a running build, for the client.
///
/// # Arguments
/// * `done` — completion flag of each step, indexed like [`BuildStep::ALL`].
/// * `elapsed` — time since the build started.
///
/// # Return
/// The [`BuildProgress`], labelled with the first pending step (or the last
/// step once all are done and the snapshot is being published).
fn progress_of(done: &[bool; BuildStep::ALL.len()], elapsed: Duration) -> BuildProgress {
    let step = BuildStep::ALL
        .iter()
        .find(|step| !done[step.index()])
        .unwrap_or(&BuildStep::Indexing);
    BuildProgress {
        done: done.iter().filter(|done| **done).count() as u32,
        total: BuildStep::ALL.len() as u32,
        step: step.label().to_string(),
        elapsed_ms: u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
    }
}

/// Current snapshot version and build state.
///
/// # Return
/// The [`RefreshStatus`] reported to the client. No database access.
pub fn status() -> RefreshStatus {
    let version = snapshot::snapshot().map(|snap| snap.version);
    let state = lock_state();
    let phase = match (&state.running, &state.last_error) {
        (Some(running), _) => {
            RefreshPhase::Building(progress_of(&running.done, running.started_at.elapsed()))
        }
        (None, Some(message)) => RefreshPhase::Failed {
            message: message.clone(),
        },
        (None, None) => RefreshPhase::Idle,
    };
    RefreshStatus { version, phase }
}

/// Whether the database changed since the served snapshot was built.
///
/// # Arguments
/// * `served` — marker the served snapshot was built from, `None` if no
///   snapshot has been built yet.
/// * `latest` — marker just read from the database.
///
/// # Return
/// `true` if a rebuild is needed: no snapshot yet, or any marker field
/// differs.
pub fn needs_rebuild(served: Option<&DataMarker>, latest: &DataMarker) -> bool {
    served != Some(latest)
}

/// Checks for changes and rebuilds if needed, or rebuilds unconditionally.
///
/// # Arguments
/// * `force` — skip the change check and always rebuild.
///
/// # Return
/// What was done, see [`RefreshOutcome`]. Returns as soon as the build is
/// started; [`status`] follows it. An [`RefreshOutcome::UpToDate`] answer
/// also clears any error recorded earlier.
///
/// # Errors
/// [`SnapshotBuildError`] if reading the [`DataMarker`] fails; the message
/// is also recorded for [`status`].
pub async fn request_refresh(force: bool) -> Result<RefreshOutcome, SnapshotBuildError> {
    if lock_state().running.is_some() {
        return Ok(RefreshOutcome::AlreadyRunning);
    }

    if !force {
        let latest = match snapshot::read_data_marker(get_pool().await).await {
            Ok(marker) => marker,
            Err(e) => {
                lock_state().last_error = Some(e.to_string());
                return Err(e.into());
            }
        };
        let served = snapshot::snapshot();
        if !needs_rebuild(served.as_ref().map(|snap| &snap.marker), &latest) {
            // The served snapshot matches the database, so an error left by
            // an earlier check or build no longer describes the current state.
            lock_state().last_error = None;
            return Ok(RefreshOutcome::UpToDate);
        }
    }

    Ok(if start_build() {
        RefreshOutcome::Started
    } else {
        RefreshOutcome::AlreadyRunning
    })
}

/// Frees the build slot when the build task ends, including by panic, so a
/// failed build can never block every later one.
struct BuildGuard;

impl Drop for BuildGuard {
    fn drop(&mut self) {
        let mut state = lock_state();
        state.running = None;
        if std::thread::panicking() {
            state.last_error = Some("the snapshot build panicked".to_string());
        }
    }
}

/// Claims the build slot and spawns a build.
///
/// # Return
/// `false` if a build was already running (nothing is spawned).
fn start_build() -> bool {
    {
        let mut state = lock_state();
        if state.running.is_some() {
            return false;
        }
        state.running = Some(RunningBuild {
            started_at: Instant::now(),
            done: [false; BuildStep::ALL.len()],
        });
        state.last_error = None;
    }

    tokio::spawn(async {
        let _guard = BuildGuard;
        let started_at = Instant::now();
        match snapshot::build().await {
            Ok(snap) => {
                tracing::info!(
                    "homepage snapshot v{} built in {:.1} s",
                    snap.version,
                    started_at.elapsed().as_secs_f64()
                );
                snapshot::publish(snap);
            }
            Err(e) => {
                // The previous snapshot, if any, stays served.
                tracing::error!("homepage snapshot build failed: {e}");
                lock_state().last_error = Some(e.to_string());
            }
        }
    });
    true
}

/// Starts the background task that builds the snapshot right away, then
/// rebuilds it whenever the database changes. Also starts loading the 3D
/// view's planet ephemerides, so both are warm before the first visitor.
///
/// Idempotent: only the first call does anything (the Dioxus dev server may
/// rebuild the router, and so call this, several times).
///
/// # Notes
/// Must be called from within the Tokio runtime.
pub fn start_background_refresh() {
    static STARTED: Once = Once::new();
    STARTED.call_once(|| {
        crate::orbit3d::server_fns::warm_up_planets();
        tokio::spawn(async {
            loop {
                if let Err(e) = request_refresh(false).await {
                    tracing::warn!("homepage snapshot change check failed: {e}");
                }
                tokio::time::sleep(CHECK_INTERVAL).await;
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn marker(branches: i64, overlay: i64) -> DataMarker {
        DataMarker {
            branches_filenode: branches,
            best_state_filenode: 2,
            overlay_changes: overlay,
        }
    }

    #[test]
    fn first_build_is_always_needed() {
        assert!(needs_rebuild(None, &marker(1, 0)));
    }

    #[test]
    fn unchanged_marker_needs_no_rebuild() {
        assert!(!needs_rebuild(Some(&marker(1, 5)), &marker(1, 5)));
    }

    #[test]
    fn a_convert_or_a_new_fit_triggers_a_rebuild() {
        // `TRUNCATE` by `fink-fat convert` gives `branches` a new filenode.
        assert!(needs_rebuild(Some(&marker(1, 5)), &marker(9, 5)));
        // A new orbit fit / failure / cross-match bumps the overlay counter.
        assert!(needs_rebuild(Some(&marker(1, 5)), &marker(1, 6)));
    }

    #[test]
    fn step_indices_follow_declaration_order() {
        for (i, step) in BuildStep::ALL.iter().enumerate() {
            assert_eq!(step.index(), i);
        }
    }

    #[test]
    fn progress_labels_the_first_pending_step() {
        let mut done = [false; BuildStep::ALL.len()];
        done[BuildStep::DataMarker.index()] = true;
        done[BuildStep::HypothesisCount.index()] = true;
        let progress = progress_of(&done, Duration::from_millis(1500));
        assert_eq!(progress.done, 2);
        assert_eq!(progress.total, BuildStep::ALL.len() as u32);
        assert_eq!(progress.step, BuildStep::Branches.label());
        assert_eq!(progress.elapsed_ms, 1500);
    }

    #[test]
    fn progress_once_every_step_is_done() {
        let done = [true; BuildStep::ALL.len()];
        let progress = progress_of(&done, Duration::ZERO);
        assert_eq!(progress.percent(), 100);
        assert_eq!(progress.step, BuildStep::Indexing.label());
    }
}
