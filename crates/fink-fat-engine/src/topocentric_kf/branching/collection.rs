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
//! let mut collection = BranchCollection::default();
//! for (step, night_obs) in nights_in_order.iter().enumerate() {
//!     collection = collection.advance_one_night(
//!         night_obs, &obs_dataset, &kalman_context,
//!         &advance_params, &discovery_params, step,
//!     )?;
//! }
//! ```

use std::collections::HashSet;

use camino::Utf8Path;
use photom::observation_dataset::{ObsDataset, ObsId, observation::Observation};

use crate::{
    engine_config::{kalman_context::KalmanContext, main_config::EngineConfig},
    error::{EngineError, FinkFatError},
    spacetime_bucket::healpix_binner::HealpixBinner,
    topocentric_kf::branching::{
        Branch, BranchSnapshot, discovery::seed_new_lineages_from_leftovers,
        orchestrate::advance_bank_collection_one_night,
    },
};

/// Filename the [`BranchCollectionSnapshot`] is written to under
/// [`EngineConfig::storage_path_buf`] — shared by `fink-fat` (the real
/// engine binary, which writes it after every night) and `fink-fat-eval`
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
    /// [`NightAdvanceOutcome::consumed_then_pruned_ids`](super::orchestrate::NightAdvanceOutcome::consumed_then_pruned_ids).
    /// These were already given back to `advance_one_night`'s own discovery
    /// step this same night (see that method's doc), so a caller does not
    /// need to do anything with this field to benefit from that — it's
    /// informational only at this point (e.g. for reporting/telemetry).
    /// Empty before the first `advance_one_night` call and on nights where
    /// `self` started empty (nothing to advance/prune).
    pub last_night_consumed_then_pruned_ids: HashSet<ObsId>,
    /// Night index this collection was last advanced to — the `current_step`
    /// a caller should pass into the next [`Self::advance_one_night`] call.
    /// `0` for a fresh/empty collection.
    pub current_step: usize,
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
            current_step: snapshot.current_step,
        }
    }

    /// Read+deserialize+reconstruct a [`BranchCollection`] from the
    /// `.rkyv` file at `path` (see [`Self::to_snapshot`]/
    /// [`Self::from_snapshot`]). Returns `Err` if the file is missing,
    /// unreadable, or fails to deserialize as a [`BranchCollectionSnapshot`].
    pub fn load_snapshot_from_disk(
        path: &Utf8Path,
        kalman_context: &'state_lf KalmanContext,
        engine_config: &'bank_config EngineConfig,
    ) -> Result<Self, EngineError> {
        let bytes = std::fs::read(path).map_err(FinkFatError::Io)?;
        let snapshot = rkyv::from_bytes::<BranchCollectionSnapshot, rkyv::rancor::Error>(&bytes)
            .map_err(|e| FinkFatError::Message(e.to_string()))?;
        Ok(Self::from_snapshot(snapshot, kalman_context, engine_config))
    }

    pub fn empty() -> Self {
        BranchCollection {
            branches: Vec::new(),
            last_night_consumed_then_pruned_ids: HashSet::new(),
            current_step: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.branches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.branches.is_empty()
    }

    /// Advance the collection by one night.
    ///
    /// When `self` is empty (night 0, or simply no live lineage survived),
    /// the update phase is skipped entirely — there is nothing to propagate
    /// or branch — and every observation is available for discovery.
    /// Otherwise, existing lineages are propagated/branched/scored/pruned
    /// first ([`advance_bank_collection_one_night`]), and only the
    /// observations they didn't claim feed the discovery phase
    /// ([`seed_new_lineages_from_leftovers`]). Either way the result is one
    /// `BranchCollection` ready for the next night.
    ///
    /// # Arguments
    /// * `night_obs` – This night's observations.
    /// * `obs_dataset`, `kalman_context` – Shared ephemeris/observation
    ///   context, threaded through to both phases.
    /// * `advance_params` – MHT branching/pruning tuning for the update
    ///   phase (see [`crate::engine_config::night_advance_params::NightAdvanceParams`]).
    /// * `discovery_params` – Pairing/seeding tuning for the discovery phase
    ///   (see `BankBuildParams`).
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

        // `advance_bank_collection_one_night` groups `night_obs` into
        // visits internally and builds one bucket index per visit (see its
        // module doc for why a single per-night index would be wrong at
        // LSST cadence) — nothing to build here.
        let (mut branches, consumed_observation_ids, consumed_then_pruned_ids) =
            if self.branches.is_empty() {
                CollectionEvent::SkipPropagation.emit();
                (Vec::new(), HashSet::new(), HashSet::new())
            } else {
                CollectionEvent::Advancing.emit();
                let outcome = advance_bank_collection_one_night(
                    &self.branches,
                    night_obs,
                    obs_dataset,
                    kalman_context,
                    &engine_config.advance_params,
                    &spatial_binner,
                    current_step,
                );
                (
                    outcome.branches,
                    outcome.consumed_observation_ids,
                    outcome.consumed_then_pruned_ids,
                )
            };

        // Must not reuse an id already held by a branch that got fully
        // pruned this night, so the max is taken over the pre-update set
        // (`self.branches`) too, not just the survivors.
        let mut next_lineage_id = self
            .branches
            .iter()
            .chain(branches.iter())
            .map(|branch| branch.lineage_id)
            .max()
            .map_or(0, |id| id + 1);

        CollectionEvent::SeedingNewLineages.emit();

        // Observations an existing lineage's candidate extension merely
        // *tried* this night, but that didn't survive this same night's
        // pruning (`cap_top_b_per_lineage`/`apply_n_scan_pruning`), are not
        // truly claimed by anything: give them back to the leftover pool so
        // a real object doesn't lose its only same-night seeding chance to
        // an unrelated candidate that ultimately went nowhere. See
        // `NightAdvanceOutcome::consumed_then_pruned_ids`'s doc — this is
        // always resolvable within this same night (an observation only
        // ever appears in one night's `night_obs`, so there is no later
        // night where it could be revisited).
        let truly_claimed_ids: HashSet<ObsId> = consumed_observation_ids
            .difference(&consumed_then_pruned_ids)
            .copied()
            .collect();

        // Separate, differently-binned index — built once inside
        // `build_kf_bank_collection_from_observations` (see `discovery`
        // module docs).
        let new_lineages = seed_new_lineages_from_leftovers(
            night_obs,
            &truly_claimed_ids,
            obs_dataset,
            kalman_context,
            engine_config,
            &spatial_binner,
            &mut next_lineage_id,
        )?;
        branches.extend(new_lineages);

        CollectionEvent::End {
            n_branches: branches.len(),
        }
        .emit();

        Ok(Self {
            branches,
            last_night_consumed_then_pruned_ids: consumed_then_pruned_ids,
            current_step: current_step + 1,
        })
    }
}
