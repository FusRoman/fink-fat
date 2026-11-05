//! Seed brand-new lineages from observations no existing lineage claimed.
//!
//! [`orchestrate`](super::orchestrate) only ever *advances* lineages that
//! already exist — it has no way to notice a previously untracked object.
//! This module closes that gap: night-0 seeding
//! ([`build_kf_bank_collection`](crate::topocentric_kf::kalman_bank::from_seeds::build_kf_bank_collection))
//! and this "discovery" step are the same intra-night pairing algorithm
//! ([`build_kf_bank_collection_from_observations`]), applied to a different
//! observation subset — all of `night_obs` at night 0, only the leftover
//! (unclaimed) observations afterward.

use std::collections::HashSet;

use photom::observation_dataset::{ObsDataset, ObsId, observation::Observation};

use crate::{
    engine_config::{kalman_context::KalmanContext, main_config::EngineConfig},
    error::EngineError,
    spacetime_bucket::healpix_binner::HealpixBinner,
    topocentric_kf::{
        branching::Branch, kalman_bank::from_seeds::build_kf_bank_collection_from_observations,
    },
};

use crate::logging::LogTarget;

/// Structured log events for new-lineage discovery from unclaimed
/// observations. See [`crate::logging`] for the `.emit()` pattern.
pub enum DiscoveryEvent {
    Summary {
        n_leftover_observations: usize,
        n_banks_built: usize,
        n_dead_banks: usize,
        n_new_lineages: usize,
    },
}

crate::impl_log_target!(
    DiscoveryEvent,
    "discovery",
    "Seeding brand-new lineages from observations no existing lineage claimed",
    [tracing::Level::DEBUG]
);

impl DiscoveryEvent {
    pub fn emit(&self) {
        use DiscoveryEvent::*;
        match self {
            Summary {
                n_leftover_observations,
                n_banks_built,
                n_dead_banks,
                n_new_lineages,
            } => tracing::debug!(
                target: DiscoveryEvent::TARGET, n_leftover_observations, n_banks_built, n_dead_banks, n_new_lineages,
                "Discovery: new-lineage seeding summary"
            ),
        }
    }
}

/// Build brand-new lineages from the observations this night's existing
/// lineages did not claim.
///
/// # Arguments
/// * `night_obs` – This night's full observation set.
/// * `consumed_observation_ids` – Ids claimed by an existing lineage this
///   night (see
///   [`NightAdvanceOutcome::consumed_observation_ids`](super::orchestrate::NightAdvanceOutcome::consumed_observation_ids)).
///   Empty at night 0, when every observation is up for grabs.
/// * `next_lineage_id` – Monotonic counter; advanced by one per new lineage
///   created. Caller seeds it above the highest `lineage_id` already in use
///   (including lineages that got fully pruned this night, so ids are never
///   reused).
/// * `next_branch_id` – Monotonic counter for the `branch_id` of each new
///   lineage's root branch. **Deliberately separate from `next_lineage_id`**:
///   [`advance_bank_collection_one_night`](super::orchestrate::advance_bank_collection_one_night)
///   allocates `branch_id`s from its own counter (bounded by the highest
///   `branch_id` among the branches it advanced), which grows independently
///   of — and much faster than — `next_lineage_id` (one new value per
///   branching event vs. one per newly discovered object). Reusing
///   `next_lineage_id`'s value as a `branch_id` here would let a freshly
///   seeded lineage collide with a `branch_id` orchestration already handed
///   out this same night. Caller seeds it above the highest `branch_id`
///   among the branches surviving this night's advance step.
/// * `current_step` – Current night index; recorded as each new lineage's
///   `last_real_update_step` (birth counts as a real update, so a freshly
///   seeded lineage starts at staleness age zero).
///
/// # Returns
/// One [`Branch::seed`] per new bank built from the leftover observations.
#[allow(clippy::too_many_arguments)]
pub fn seed_new_lineages_from_leftovers<'state_lf, 'bank_config>(
    night_obs: &[&Observation],
    consumed_observation_ids: &HashSet<ObsId>,
    obs_dataset: &ObsDataset,
    kalman_context: &'state_lf KalmanContext,
    engine_config: &'bank_config EngineConfig,
    spatial_binner: &HealpixBinner,
    next_lineage_id: &mut u64,
    next_branch_id: &mut u64,
    current_step: usize,
) -> Result<Vec<Branch<'state_lf, 'bank_config>>, EngineError> {
    let leftover_obs: Vec<&Observation> = night_obs
        .iter()
        .copied()
        .filter(|obs| !consumed_observation_ids.contains(obs.id()))
        .collect();

    let new_banks = build_kf_bank_collection_from_observations(
        obs_dataset,
        &leftover_obs,
        kalman_context,
        engine_config,
        spatial_binner,
    )?;

    let n_before_filter = new_banks.len();
    let live_banks: Vec<_> = new_banks
        .into_iter()
        .filter(|bank| bank.is_alive())
        .collect();
    let n_dead = n_before_filter - live_banks.len();

    let new_lineages: Vec<_> = live_banks
        .into_iter()
        .map(|bank| {
            let lineage_id = *next_lineage_id;
            *next_lineage_id += 1;
            let branch_id = *next_branch_id;
            *next_branch_id += 1;
            Branch::seed(bank, lineage_id, branch_id, current_step)
        })
        .collect();

    DiscoveryEvent::Summary {
        n_leftover_observations: leftover_obs.len(),
        n_banks_built: n_before_filter,
        n_dead_banks: n_dead,
        n_new_lineages: new_lineages.len(),
    }
    .emit();

    Ok(new_lineages)
}
