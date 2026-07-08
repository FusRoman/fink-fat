//! Top-level intra-night pair → [`KFBank`] construction pipeline.
//!
//! Entry point: [`build_kf_bank_collection`]. For each night present in an
//! [`ObsDataset`] (via `iter_night_id`/`iter_night_observations`), links
//! intra-night observations into tracklets with [`link_tracklets`] (one
//! maximal-baseline pair per multi-detection object — see that module's
//! docs for why this replaces a plain all-pairs enumeration), then converts
//! each surviving pair into a [`KFBank`] via `KFBank::from_grid`. Pairs are
//! never formed across nights.

use photom::{
    NightId,
    observation_dataset::{ObsDataset, observation::Observation},
};

use crate::{
    engine_config::{kalman_context::KalmanContext, main_config::EngineConfig},
    error::EngineError,
    seeding::tracklet_linker::link_tracklets,
    spacetime_bucket::healpix_binner::HealpixBinner,
    topocentric_kf::kalman_bank::KFBank,
};

pub type KFBankCollection<'state_lf> = Vec<KFBank<'state_lf>>;

/// Build one [`KFBank`] per admissible intra-night observation pair across
/// every night in `obs_dataset`.
///
/// Thin wrapper around [`build_kf_bank_collection_from_observations`]:
/// resolves `night_id` to its observations, then delegates. Nights without a
/// night index (`iter_night_observations` returning `None`) are silently
/// skipped.
pub fn build_kf_bank_collection<'state_lf>(
    obs_dataset: &ObsDataset,
    night_id: &NightId,
    kalman_context: &'state_lf KalmanContext,
    params: &EngineConfig,
) -> Result<KFBankCollection<'state_lf>, EngineError> {
    let Some(obs_iter) = obs_dataset.iter_night_observations(night_id) else {
        return Ok(KFBankCollection::default());
    };
    let night_obs: Vec<&Observation> = obs_iter.collect();

    let spatial_binner = HealpixBinner::new(params.healpix_depth);

    build_kf_bank_collection_from_observations(
        obs_dataset,
        &night_obs,
        kalman_context,
        params,
        &spatial_binner,
    )
}

/// Build one [`KFBank`] per admissible intra-night pair found in an
/// arbitrary slice of observations.
///
/// This is the reusable core behind both night-0 seeding
/// ([`build_kf_bank_collection`]) and the per-night "discovery" step that
/// seeds brand-new lineages from observations no existing bank claimed (see
/// [`seed_new_lineages_from_leftovers`](crate::topocentric_kf::branching::discovery::seed_new_lineages_from_leftovers)) —
/// both need "pair up this set of observations and build banks," differing
/// only in which observations they pass in.
///
/// Pipeline:
/// 1. [`link_tracklets`] — links intra-night observations into per-object
///    tracklets (time/angular-speed/magnitude gated, reusing the same
///    spatial bucketing as `generate_pairs`) and emits one pair per
///    multi-detection object.
/// 2. `KFBank::from_grid` per surviving pair.
///
/// A `KFBank::from_grid` failure for one pair (e.g. unresolvable observer, no
/// admissible (ρ, ρ̇) region) is logged and the pair is skipped — it does not
/// abort the whole run, consistent with `generate_pairs` being a permissive
/// pre-filter (see `pairs` module docs).
pub fn build_kf_bank_collection_from_observations<'state_lf>(
    obs_dataset: &ObsDataset,
    night_obs: &[&Observation],
    kalman_context: &'state_lf KalmanContext,
    params: &EngineConfig,
    spatial_binner: &HealpixBinner,
) -> Result<KFBankCollection<'state_lf>, EngineError> {
    if night_obs.is_empty() {
        return Ok(KFBankCollection::default());
    }

    let pairs = link_tracklets(night_obs, spatial_binner, &params.pairs);

    tracing::debug!(
        n_obs = night_obs.len(),
        n_pairs = pairs.len(),
        "pairs generated from observation slice"
    );

    let mut banks = Vec::new();
    for pair in &pairs {
        match KFBank::from_grid(
            obs_dataset,
            pair.a,
            pair.b,
            kalman_context,
            &params.seeding_grid_config,
            params.kfbank_config.clone(),
        ) {
            Ok(bank) => banks.push(bank),
            Err(err) => {
                tracing::debug!(
                    first = *pair.a.id(),
                    second = *pair.b.id(),
                    %err,
                    "KFBank::from_grid failed for pair, skipping"
                );
            }
        }
    }

    Ok(banks)
}
