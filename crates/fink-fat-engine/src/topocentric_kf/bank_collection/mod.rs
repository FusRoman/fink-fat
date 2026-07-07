//! Top-level intra-night pair → [`KFBank`] construction pipeline.
//!
//! Entry point: [`build_kf_bank_collection`]. For each night present in an
//! [`ObsDataset`] (via `iter_night_id`/`iter_night_observations`), generates
//! intra-night observation pairs with [`generate_pairs`] (reusing the
//! existing gating: max_dt, angular-speed dot-product test, magnitude gate,
//! `BucketIndex`/`SpatialBinner`/`TimeBinner`), then converts each surviving
//! pair into a [`KFBank`] via `KFBank::from_grid`. Pairs are never formed
//! across nights.

use photom::{
    MJDTT, NightId,
    observation_dataset::{ObsDataset, observation::Observation},
};

use crate::{
    engine_config::pair_config::PairConfig,
    error::EngineError,
    seeding::pairs::generate_pairs,
    spacetime_bucket::{
        bucket::build_alert_bucket_index, healpix_binner::HealpixBinner,
        uniform_time_binner::UniformTimeBinner,
    },
    topocentric_kf::{
        kalman_bank::{KFBank, config::KFBankConfig, seed_grid::GridConfig},
        single_kalman::context::KalmanContext,
    },
};

/// Flat container of all [`KFBank`]s built from an [`ObsDataset`].
#[derive(Default)]
pub struct KFBankCollection<'state_lf> {
    pub banks: Vec<KFBank<'state_lf>>,
}

/// Tuning parameters for pairing a set of observations and building a
/// [`KFBank`] per surviving pair: spatial binning, intra-night pairing
/// gates, seed-grid construction, and bank configuration.
///
/// Grouped into one struct so [`build_kf_bank_collection`] and
/// [`build_kf_bank_collection_from_observations`] stay under clippy's
/// argument-count threshold, and so callers building banks from more than
/// one observation subset (e.g. the per-night "discovery" step in
/// `topocentric_kf::branching::discovery`) can reuse the same parameter set
/// without repeating five individual arguments at every call site.
pub struct BankBuildParams<'a> {
    pub spatial_binner: &'a HealpixBinner,
    pub time_binner_width: MJDTT,
    pub pair_config: &'a PairConfig,
    pub grid_config: &'a GridConfig,
    pub bank_config: &'a KFBankConfig,
}

impl<'state_lf> KFBankCollection<'state_lf> {
    pub fn len(&self) -> usize {
        self.banks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.banks.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &KFBank<'state_lf>> {
        self.banks.iter()
    }
}

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
    params: &BankBuildParams,
) -> Result<KFBankCollection<'state_lf>, EngineError> {
    let Some(obs_iter) = obs_dataset.iter_night_observations(night_id) else {
        return Ok(KFBankCollection::default());
    };
    let night_obs: Vec<&Observation> = obs_iter.collect();

    build_kf_bank_collection_from_observations(obs_dataset, &night_obs, kalman_context, params)
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
/// 1. `build_alert_bucket_index` + `generate_pairs(..., pair_config)` — reused
///    gating logic (max_dt / angular speed / magnitude / bucket dedup+sort).
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
    params: &BankBuildParams,
) -> Result<KFBankCollection<'state_lf>, EngineError> {
    if night_obs.is_empty() {
        return Ok(KFBankCollection::default());
    }

    let t0 = night_obs
        .iter()
        .map(|o| o.mjd_tt())
        .fold(f64::INFINITY, f64::min);
    let time_binner = UniformTimeBinner::new(t0, params.time_binner_width);

    let bucket_index = build_alert_bucket_index(
        night_obs.iter().copied(),
        params.spatial_binner,
        &time_binner,
    );

    let pairs = generate_pairs(
        &bucket_index,
        params.spatial_binner,
        &time_binner,
        params.pair_config,
    );

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
            params.grid_config,
            params.bank_config.clone(),
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

    Ok(KFBankCollection { banks })
}
