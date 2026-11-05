//! Night-to-night container for the whole set of live branches — the single
//! entry point for running fink-fat's per-night loop.
//!
//! # Integration point
//!
//! Night 0 and every subsequent night are the **same call**:
//! [`BranchCollection::advance_one_night`] skips the update phase when
//! `self` is empty (nothing to propagate yet) and runs it otherwise. A
//! caller never special-cases "first night" vs. "later night":
//!
//! ```ignore
//! let mut collection = BranchCollection::empty();
//! for night_obs in &nights_in_order {
//!     let step = collection.current_step;
//!     collection = collection.advance_one_night(
//!         night_obs, &obs_dataset, &engine_config, &kalman_context, step,
//!     )?;
//! }
//! ```
//!
//! A streaming caller that receives a night visit by visit drives the same
//! pipeline through a [`NightSession`] instead.

use std::collections::HashSet;

use camino::Utf8Path;
use photom::observation_dataset::{ObsDataset, ObsId, observation::Observation};

use crate::{
    engine_config::{kalman_context::KalmanContext, main_config::EngineConfig},
    error::{EngineError, FinkFatError},
    spacetime_bucket::healpix_binner::HealpixBinner,
    topocentric_kf::branching::{
        ArchivedTrajectory, Branch, BranchSnapshot,
        night_session::NightSession,
        orchestrate::{GateRecord, OrchestrateEvent},
        visit::group_observations_into_visits,
    },
};

/// Filename the [`BranchCollectionSnapshot`] is written to under
/// [`EngineConfig::storage_path_buf`] — shared by `fink-fat` (the real
/// engine binary, which writes it after the last night of a run, and every
/// `--snapshot-every N` nights when that flag is set) and `fink-fat-eval`
/// (which only reads it, for `tracking_analysis --from-snapshot`).
pub const SNAPSHOT_FILENAME: &str = "branch_collection.rkyv";

/// The full set of live branches tracked across nights — one entry per
/// surviving candidate association history, possibly several per original
/// bank once ambiguity has forced branching.
#[derive(Default)]
pub struct BranchCollection<'state_lf, 'bank_config> {
    pub branches: Vec<Branch<'state_lf, 'bank_config>>,
    /// Exposed for observability, not just diagnostics: observations
    /// consumed by a candidate extension of an existing lineage during the
    /// night just advanced, where that candidate branch was itself pruned
    /// before surviving to `branches` — see
    /// [`NightSession::close`]. These were already given back to that same
    /// night's discovery
    /// step this same night (see that method's doc), so a caller does not
    /// need to do anything with this field to benefit from that — it's
    /// informational only at this point (e.g. for reporting/telemetry).
    /// Empty before the first `advance_one_night` call and on nights where
    /// `self` started empty (nothing to advance/prune).
    pub last_night_consumed_then_pruned_ids: HashSet<ObsId>,
    /// Per-lineage gate records from the night just advanced — passive
    /// diagnostics for gate-selectivity analysis (see
    /// [`GateRecord`]). Not persisted in the
    /// snapshot (ephemeral, one night's worth); empty before the first
    /// `advance_one_night` and on nights that started empty.
    pub last_night_gate_records: Vec<GateRecord>,
    /// Night index this collection was last advanced to — the `current_step`
    /// a caller should pass into the next [`Self::advance_one_night`] call.
    /// `0` for a fresh/empty collection.
    pub current_step: usize,
    /// Reconstructions of the lineages that stopped being propagated
    /// **during the night just advanced** — see [`ArchivedTrajectory`] and
    /// [`purge_stale_lineages`](super::pruning::purge_stale_lineages).
    ///
    /// **Not cumulative** during live tracking: [`Self::advance_one_night`]
    /// returns only this night's fresh batch here (like `last_night_*`), not
    /// the run's full archive — the caller (`fink-fat track`'s loop) is
    /// expected to append it to the on-disk log
    /// ([`super::write_archived_batch`]) and drop it before the next call,
    /// rather than let it accumulate in RAM for the whole run (that
    /// accumulation, plus a full clone of it every night, is exactly what
    /// this field's contract used to cost before archived trajectories
    /// moved to a dedicated append-only log — see
    /// [`super::read_archived_log`]).
    ///
    /// Offline consumers (`fink-fat convert`, `tracking_analysis`) that need
    /// the *full* run's archive should instead load it explicitly with
    /// [`super::read_archived_log`] and assign it to this field themselves
    /// after loading a snapshot — at that point it *is* the cumulative
    /// history, since nothing else populates it. This dual meaning
    /// (per-night delta during tracking vs. explicit cumulative load
    /// offline) is a deliberate trade-off to avoid a signature change
    /// rippling through every offline consumer of `.archived`.
    pub archived: Vec<ArchivedTrajectory>,
}

/// Owned, borrow-free snapshot of a [`BranchCollection`], for persisting the
/// pipeline's state to disk at the end of a night and reloading it at the
/// start of the next one (see [`BranchCollection::to_snapshot`]).
///
/// Neither the live [`KalmanContext`]
/// nor the [`EngineConfig`] are part of the snapshot: both are cheap for the
/// caller to hold onto (or rebuild) across a process restart, and are
/// re-supplied to [`BranchCollection::from_snapshot`].
#[derive(Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct BranchCollectionSnapshot {
    pub branches: Vec<BranchSnapshot>,
    pub last_night_consumed_then_pruned_ids: HashSet<ObsId>,
    pub current_step: usize,
}

use crate::logging::LogTarget;

/// Structured log events for the per-night `BranchCollection` pipeline
/// entry point. See [`crate::logging`] for the `.emit()` pattern.
pub enum CollectionEvent {
    Start,
    InputSummary {
        n_observations: usize,
        current_step: usize,
    },
    SkipPropagation,
    Advancing,
    SeedingNewLineages,
    End {
        n_branches: usize,
    },
}

crate::impl_log_target!(
    CollectionEvent,
    "collection",
    "Per-night BranchCollection pipeline entry point (advance_one_night)",
    [tracing::Level::INFO, tracing::Level::DEBUG]
);

impl CollectionEvent {
    pub fn emit(&self) {
        use CollectionEvent::*;
        match self {
            Start => {
                tracing::info!(target: CollectionEvent::TARGET, "Start of the advance one night pipeline")
            }
            InputSummary {
                n_observations,
                current_step,
            } => tracing::debug!(
                target: CollectionEvent::TARGET, n_observations, current_step, "Advance one night: input summary"
            ),
            SkipPropagation => tracing::info!(
                target: CollectionEvent::TARGET, "Branches is empty, skip the kalman propagation"
            ),
            Advancing => tracing::info!(
                target: CollectionEvent::TARGET, "Find previous branches, perform kalman one night advance"
            ),
            SeedingNewLineages => {
                tracing::info!(target: CollectionEvent::TARGET, "Start new seeds generation")
            }
            End { n_branches } => tracing::info!(
                target: CollectionEvent::TARGET, n_branches, "End of the advance one night pipeline"
            ),
        }
    }

    pub fn span() -> tracing::Span {
        tracing::info_span!(target: CollectionEvent::TARGET, "Advance one night")
    }
}

impl<'state_lf, 'bank_config> BranchCollection<'state_lf, 'bank_config> {
    /// Convert to an owned, borrow-free snapshot suitable for on-disk
    /// persistence (see [`BranchCollectionSnapshot`]). The engine does not
    /// perform the actual file I/O — the caller is responsible for encoding
    /// and writing (e.g. with `rkyv::to_bytes`) the returned snapshot.
    pub fn to_snapshot(&self) -> BranchCollectionSnapshot {
        BranchCollectionSnapshot {
            branches: self.branches.iter().map(Branch::to_snapshot).collect(),
            last_night_consumed_then_pruned_ids: self.last_night_consumed_then_pruned_ids.clone(),
            current_step: self.current_step,
        }
    }

    /// Rebuild a full [`BranchCollection`] from a snapshot previously
    /// produced by [`Self::to_snapshot`], reattaching the live
    /// `kalman_context`/`engine_config` the caller already holds (the same
    /// arguments it would pass to [`Self::advance_one_night`]).
    pub fn from_snapshot(
        snapshot: BranchCollectionSnapshot,
        kalman_context: &'state_lf KalmanContext,
        engine_config: &'bank_config EngineConfig,
    ) -> Self {
        Self {
            branches: snapshot
                .branches
                .into_iter()
                .map(|b| Branch::from_snapshot(b, kalman_context, &engine_config.kfbank_config))
                .collect(),
            last_night_consumed_then_pruned_ids: snapshot.last_night_consumed_then_pruned_ids,
            last_night_gate_records: Vec::new(),
            current_step: snapshot.current_step,
            // Archived trajectories no longer round-trip through the
            // snapshot (see `BranchCollection::archived`'s doc) — a run
            // resuming from this snapshot has no need for its prior
            // archive, which lives entirely in the separate append-only log
            // instead.
            archived: Vec::new(),
        }
    }

    /// Read+deserialize+reconstruct a [`BranchCollection`] from the
    /// `.rkyv` file at `path` (see [`Self::to_snapshot`]/
    /// [`Self::from_snapshot`]). Returns `Err` if the file is missing,
    /// unreadable, or fails to deserialize as a [`BranchCollectionSnapshot`].
    ///
    /// The file is memory-mapped rather than read into a heap-allocated
    /// `Vec<u8>`: at the scale this snapshot can reach, that would be an
    /// extra full-size copy of the file sitting in memory for the duration
    /// of the (already memory-heavy) deserialization step.
    pub fn load_snapshot_from_disk(
        path: &Utf8Path,
        kalman_context: &'state_lf KalmanContext,
        engine_config: &'bank_config EngineConfig,
    ) -> Result<Self, EngineError> {
        let file = std::fs::File::open(path).map_err(FinkFatError::Io)?;
        // SAFETY: `path` is fink-fat's own snapshot file, not modified by any
        // other process while a run holds it open, so the mapped bytes stay
        // valid and stable for the lifetime of this read-only mapping.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(FinkFatError::Io)?;
        let snapshot = rkyv::from_bytes::<BranchCollectionSnapshot, rkyv::rancor::Error>(&mmap)
            .map_err(|e| FinkFatError::Message(e.to_string()))?;
        Ok(Self::from_snapshot(snapshot, kalman_context, engine_config))
    }

    pub fn empty() -> Self {
        BranchCollection {
            branches: Vec::new(),
            last_night_consumed_then_pruned_ids: HashSet::new(),
            last_night_gate_records: Vec::new(),
            current_step: 0,
            archived: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.branches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.branches.is_empty()
    }

    /// Advance the collection by one night — the batch driver of a
    /// [`NightSession`].
    ///
    /// Groups `night_obs` into visits
    /// ([`group_observations_into_visits`]), feeds them in chronological
    /// order to [`NightSession::advance_visit`], then closes the night
    /// ([`NightSession::close`]: N-scan pruning, stale purge, discovery of new
    /// lineages from the unclaimed observations). When `self` is empty (night
    /// 0, or no lineage survived) the visit loop is skipped entirely and every
    /// observation goes to discovery. A caller never special-cases "first
    /// night" vs. "later night".
    ///
    /// # Arguments
    /// * `night_obs` – This night's observations.
    /// * `obs_dataset`, `kalman_context` – Shared ephemeris/observation
    ///   context, threaded through to both phases.
    /// * `engine_config` – Supplies the branching/pruning parameters
    ///   (`advance_params`), the discovery parameters and the HEALPix depth.
    /// * `current_step` – Current night index, for N-scan bookkeeping.
    ///
    /// # Returns
    /// The next `BranchCollection`, or `Err` if the discovery phase's
    /// underlying bank-building fails (see
    /// [`build_kf_bank_collection_from_observations`](crate::topocentric_kf::kalman_bank::from_seeds::build_kf_bank_collection_from_observations)).
    pub fn advance_one_night(
        &self,
        night_obs: &[&Observation],
        obs_dataset: &ObsDataset,
        engine_config: &'bank_config EngineConfig,
        kalman_context: &'state_lf KalmanContext,
        current_step: usize,
    ) -> Result<Self, EngineError> {
        let _enter = CollectionEvent::span().entered();

        CollectionEvent::Start.emit();
        CollectionEvent::InputSummary {
            n_observations: night_obs.len(),
            current_step,
        }
        .emit();

        let spatial_binner = HealpixBinner::new(engine_config.healpix_depth);
        let params = &engine_config.advance_params;
        let mut session = NightSession::begin(self, current_step);

        if self.branches.is_empty() {
            CollectionEvent::SkipPropagation.emit();
        } else {
            CollectionEvent::Advancing.emit();
            let _advance = OrchestrateEvent::span().entered();
            let visits =
                group_observations_into_visits(night_obs, params.visit_epoch_tolerance_days);
            let n_visits = visits.len();
            OrchestrateEvent::VisitSummary { n_visits }.emit();
            let visit_progress_modulo = (n_visits / 100).max(1);
            for (visit_index, visit) in visits.iter().enumerate() {
                if visit_index % visit_progress_modulo == 0 {
                    OrchestrateEvent::VisitProgress {
                        visit_index,
                        n_visits,
                    }
                    .emit();
                }
                session.advance_visit(visit, obs_dataset, kalman_context, params, &spatial_binner);
            }
        }

        CollectionEvent::SeedingNewLineages.emit();
        let next = session.close(
            night_obs,
            obs_dataset,
            kalman_context,
            engine_config,
            &spatial_binner,
        )?;

        CollectionEvent::End {
            n_branches: next.branches.len(),
        }
        .emit();
        Ok(next)
    }
}
