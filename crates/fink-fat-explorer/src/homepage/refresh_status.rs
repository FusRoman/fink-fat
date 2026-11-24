//! What the homepage's Refresh controls show while the in-RAM snapshot is
//! checked and rebuilt.
//!
//! Compiled for both targets: the serializable status types cross the
//! `#[server]` boundary, the two server functions are declared on both sides
//! (the macro keeps the real body server-only), and the pure display logic
//! ([`interpret_status`], [`RefreshUi`]'s accessors) runs in the browser. The
//! server-side lifecycle that produces a [`RefreshStatus`] lives in
//! `homepage::snapshot_refresh`.

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

/// Progress of a running snapshot build.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BuildProgress {
    /// Build steps finished so far.
    pub done: u32,
    /// Total number of build steps.
    pub total: u32,
    /// Label of the first step still pending, e.g. "Reading branches".
    pub step: String,
    /// Time since the build started, in milliseconds.
    pub elapsed_ms: u64,
}

impl BuildProgress {
    /// Completion as a whole percentage.
    ///
    /// # Return
    /// `done / total` in `0..=100`; `0` when `total` is `0`.
    pub fn percent(&self) -> u32 {
        (self.done.min(self.total) * 100)
            .checked_div(self.total)
            .unwrap_or(0)
    }
}

/// Server-side state of the snapshot build.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RefreshPhase {
    /// No build running, and the last one (if any) succeeded.
    Idle,
    /// A build is running.
    Building(BuildProgress),
    /// The last build, or the change check before it, failed; the previous
    /// snapshot (if any) is still served.
    Failed { message: String },
}

/// What [`get_refresh_status`] reports.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefreshStatus {
    /// Version of the snapshot currently served, `None` before the first
    /// build lands. Changes exactly when a new snapshot is published.
    pub version: Option<u64>,
    pub phase: RefreshPhase,
}

/// What a refresh request did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RefreshOutcome {
    /// A build was started.
    Started,
    /// A build was already running (e.g. started by the background change
    /// check); the caller can follow that one.
    AlreadyRunning,
    /// Nothing changed in the database since the served snapshot was built.
    UpToDate,
}

/// Asks the server for a snapshot rebuild.
///
/// # Arguments
/// * `force` — `false` rebuilds only if the database changed since the
///   served snapshot was built (cheap check); `true` always rebuilds.
///
/// # Return
/// What the request did, see [`RefreshOutcome`]. Returns as soon as the
/// build is started: follow it with [`get_refresh_status`].
///
/// # Errors
/// The change check failing (database unreachable, schema predating the
/// current `fink-fat convert`).
#[server]
pub async fn refresh_snapshot(force: bool) -> Result<RefreshOutcome, ServerFnError> {
    crate::homepage::snapshot_refresh::request_refresh(force)
        .await
        .map_err(|e| ServerFnError::new(e.to_string()))
}

/// Current version and build state of the homepage snapshot.
///
/// # Return
/// The [`RefreshStatus`]; cheap (no database access), meant to be polled.
#[server]
pub async fn get_refresh_status() -> Result<RefreshStatus, ServerFnError> {
    Ok(crate::homepage::snapshot_refresh::status())
}

/// Client-side state of the Refresh controls.
#[derive(Debug, Clone, PartialEq)]
pub enum RefreshUi {
    /// Nothing in progress.
    Idle,
    /// Request sent, no build progress reported yet.
    Checking,
    /// A build is running.
    Building(BuildProgress),
    /// The check found nothing to rebuild.
    UpToDate,
    /// A new snapshot landed.
    Updated,
    /// The check or the build failed.
    Failed(String),
}

impl RefreshUi {
    /// Whether a refresh is in flight, i.e. the buttons must stay disabled.
    pub fn is_busy(&self) -> bool {
        matches!(self, Self::Checking | Self::Building(_))
    }

    /// Fill level of the progress gauge drawn behind the buttons.
    ///
    /// # Return
    /// The build percentage while building, `None` otherwise (no gauge).
    pub fn gauge_percent(&self) -> Option<u32> {
        match self {
            Self::Building(progress) => Some(progress.percent()),
            _ => None,
        }
    }

    /// Text of the main Refresh button.
    pub fn button_label(&self) -> String {
        match self {
            Self::Idle => "Refresh".to_string(),
            Self::Checking => "Checking…".to_string(),
            Self::Building(progress) => format!("Refreshing {}%", progress.percent()),
            Self::UpToDate => "✓ Up to date".to_string(),
            Self::Updated => "✓ Updated".to_string(),
            Self::Failed(_) => "Refresh failed".to_string(),
        }
    }

    /// One-line detail shown under the buttons.
    ///
    /// # Return
    /// The current step and elapsed time while building, the error message
    /// on failure, a short confirmation when nothing changed; `None` when
    /// the button label says it all.
    pub fn detail(&self) -> Option<String> {
        match self {
            Self::Building(progress) => Some(format!(
                "{} · step {}/{} · {} s",
                progress.step,
                (progress.done + 1).min(progress.total),
                progress.total,
                progress.elapsed_ms / 1000
            )),
            Self::UpToDate => Some("No change in the database since the last build".to_string()),
            Self::Failed(message) => Some(message.clone()),
            Self::Idle | Self::Checking | Self::Updated => None,
        }
    }
}

/// What the client does after one status poll.
#[derive(Debug, Clone, PartialEq)]
pub enum PollVerdict {
    /// Keep polling, showing this state meanwhile.
    Continue(RefreshUi),
    /// Stop polling and show this state; `new_snapshot` tells whether the
    /// page should refetch its data.
    Finished { ui: RefreshUi, new_snapshot: bool },
}

/// Interprets one [`RefreshStatus`] poll made after a refresh request.
///
/// # Arguments
/// * `status` — the polled status.
/// * `version_before` — the snapshot version read before the request.
///
/// # Return
/// [`PollVerdict::Finished`] once a new snapshot is served or the build
/// failed, [`PollVerdict::Continue`] while it is still running or not yet
/// visible.
pub fn interpret_status(status: &RefreshStatus, version_before: Option<u64>) -> PollVerdict {
    match &status.phase {
        RefreshPhase::Building(progress) => {
            PollVerdict::Continue(RefreshUi::Building(progress.clone()))
        }
        RefreshPhase::Failed { message } => PollVerdict::Finished {
            ui: RefreshUi::Failed(message.clone()),
            new_snapshot: false,
        },
        RefreshPhase::Idle if status.version.is_some() && status.version != version_before => {
            PollVerdict::Finished {
                ui: RefreshUi::Updated,
                new_snapshot: true,
            }
        }
        RefreshPhase::Idle => PollVerdict::Continue(RefreshUi::Checking),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn progress(done: u32, total: u32) -> BuildProgress {
        BuildProgress {
            done,
            total,
            step: "Reading branches".to_string(),
            elapsed_ms: 12_500,
        }
    }

    fn status(version: Option<u64>, phase: RefreshPhase) -> RefreshStatus {
        RefreshStatus { version, phase }
    }

    #[test]
    fn percent_is_bounded_and_safe_on_zero_total() {
        assert_eq!(progress(0, 10).percent(), 0);
        assert_eq!(progress(5, 10).percent(), 50);
        assert_eq!(progress(10, 10).percent(), 100);
        assert_eq!(progress(12, 10).percent(), 100);
        assert_eq!(progress(0, 0).percent(), 0);
    }

    #[test]
    fn building_keeps_polling_with_progress() {
        let verdict = interpret_status(
            &status(Some(3), RefreshPhase::Building(progress(2, 10))),
            Some(3),
        );
        assert_eq!(
            verdict,
            PollVerdict::Continue(RefreshUi::Building(progress(2, 10)))
        );
    }

    #[test]
    fn new_version_finishes_with_a_refetch() {
        let verdict = interpret_status(&status(Some(4), RefreshPhase::Idle), Some(3));
        assert_eq!(
            verdict,
            PollVerdict::Finished {
                ui: RefreshUi::Updated,
                new_snapshot: true
            }
        );
        // First build ever: no version before, one now.
        let verdict = interpret_status(&status(Some(1), RefreshPhase::Idle), None);
        assert!(matches!(
            verdict,
            PollVerdict::Finished {
                new_snapshot: true,
                ..
            }
        ));
    }

    #[test]
    fn idle_with_unchanged_version_waits_for_the_build_to_show_up() {
        let verdict = interpret_status(&status(Some(3), RefreshPhase::Idle), Some(3));
        assert_eq!(verdict, PollVerdict::Continue(RefreshUi::Checking));
        let verdict = interpret_status(&status(None, RefreshPhase::Idle), None);
        assert_eq!(verdict, PollVerdict::Continue(RefreshUi::Checking));
    }

    #[test]
    fn failure_finishes_without_a_refetch() {
        let verdict = interpret_status(
            &status(
                Some(3),
                RefreshPhase::Failed {
                    message: "boom".to_string(),
                },
            ),
            Some(3),
        );
        assert_eq!(
            verdict,
            PollVerdict::Finished {
                ui: RefreshUi::Failed("boom".to_string()),
                new_snapshot: false
            }
        );
    }

    #[test]
    fn only_checking_and_building_are_busy() {
        assert!(RefreshUi::Checking.is_busy());
        assert!(RefreshUi::Building(progress(1, 2)).is_busy());
        for ui in [
            RefreshUi::Idle,
            RefreshUi::UpToDate,
            RefreshUi::Updated,
            RefreshUi::Failed(String::new()),
        ] {
            assert!(!ui.is_busy(), "{ui:?}");
        }
    }

    #[test]
    fn gauge_only_while_building() {
        assert_eq!(
            RefreshUi::Building(progress(3, 4)).gauge_percent(),
            Some(75)
        );
        assert_eq!(RefreshUi::Checking.gauge_percent(), None);
        assert_eq!(RefreshUi::Updated.gauge_percent(), None);
    }

    #[test]
    fn detail_shows_the_running_step_one_based() {
        let detail = RefreshUi::Building(progress(2, 10)).detail();
        assert_eq!(
            detail.as_deref(),
            Some("Reading branches · step 3/10 · 12 s")
        );
        // All steps done but the build not yet published: never "11/10".
        let detail = RefreshUi::Building(progress(10, 10)).detail();
        assert!(detail.is_some_and(|d| d.contains("step 10/10")));
    }
}
