//! Seed brand-new lineages from observations no existing lineage claimed.
//!
//! [`orchestrate`](super::orchestrate) only ever *advances* lineages that
//! already exist — it has no way to notice a previously untracked object.
//! This module closes that gap: night-0 seeding
//! ([`build_kf_bank_collection`](crate::topocentric_kf::bank_collection::build_kf_bank_collection))
//! and this "discovery" step are the same intra-night pairing algorithm
//! ([`build_kf_bank_collection_from_observations`]), applied to a different
//! observation subset — all of `night_obs` at night 0, only the leftover
//! (unclaimed) observations afterward.

use std::collections::HashSet;

use photom::observation_dataset::{ObsDataset, ObsId, observation::Observation};

use crate::{
    error::EngineError,
    topocentric_kf::{
        bank_collection::{BankBuildParams, build_kf_bank_collection_from_observations},
        branching::Branch,
        single_kalman::context::KalmanContext,
    },
};

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
///
/// # Returns
/// One [`Branch::seed`] per new bank built from the leftover observations.
pub fn seed_new_lineages_from_leftovers<'state_lf>(
    night_obs: &[&Observation],
    consumed_observation_ids: &HashSet<ObsId>,
    obs_dataset: &ObsDataset,
    kalman_context: &'state_lf KalmanContext,
    bank_build_params: &BankBuildParams,
    next_lineage_id: &mut u64,
) -> Result<Vec<Branch<'state_lf>>, EngineError> {
    let leftover_obs: Vec<&Observation> = night_obs
        .iter()
        .copied()
        .filter(|obs| !consumed_observation_ids.contains(obs.id()))
        .collect();

    let new_banks = build_kf_bank_collection_from_observations(
        obs_dataset,
        &leftover_obs,
        kalman_context,
        bank_build_params,
    )?;

    Ok(new_banks
        .banks
        .into_iter()
        .map(|bank| {
            let id = *next_lineage_id;
            *next_lineage_id += 1;
            Branch::seed(bank, id, id)
        })
        .collect())
}
