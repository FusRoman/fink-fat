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
    observation_dataset::{ObsDataset, ObsId, observation::Observation},
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
        KalmanContext,
        kalman_bank::{KFBank, config::KFBankConfig, seed_grid::GridConfig},
    },
};

/// One [`KFBank`] built from a single intra-night observation pair, plus the
/// pair's identity for traceability.
pub struct KFPairBank<'state_lf> {
    pub night_id: NightId,
    pub first_obs_id: ObsId,
    pub second_obs_id: ObsId,
    pub bank: KFBank<'state_lf>,
}

/// Flat container of all [`KFBank`]s built from an [`ObsDataset`].
#[derive(Default)]
pub struct KFBankCollection<'state_lf> {
    pub banks: Vec<KFPairBank<'state_lf>>,
}

impl<'state_lf> KFBankCollection<'state_lf> {
    pub fn len(&self) -> usize {
        self.banks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.banks.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &KFPairBank<'state_lf>> {
        self.banks.iter()
    }
}

/// Build one [`KFBank`] per admissible intra-night observation pair across
/// every night in `obs_dataset`.
///
/// Pipeline per night:
/// 1. `iter_night_observations(night_id)` — night-scoped observations only.
/// 2. `build_alert_bucket_index` + `generate_pairs(..., pair_config)` — reused
///    gating logic (max_dt / angular speed / magnitude / bucket dedup+sort).
/// 3. `KFBank::from_grid` per surviving pair.
///
/// A `KFBank::from_grid` failure for one pair (e.g. unresolvable observer, no
/// admissible (ρ, ρ̇) region) is logged and the pair is skipped — it does not
/// abort the whole run, consistent with `generate_pairs` being a permissive
/// pre-filter (see `pairs` module docs).
///
/// Nights without a night index (`iter_night_id`/`iter_night_observations`
/// returning `None`) are silently skipped.
pub fn build_kf_bank_collection<'state_lf>(
    obs_dataset: &ObsDataset,
    kalman_context: &'state_lf KalmanContext,
    spatial_binner: &HealpixBinner,
    time_binner_width: MJDTT,
    pair_config: &PairConfig,
    grid_config: &GridConfig,
    bank_config: &KFBankConfig,
) -> Result<KFBankCollection<'state_lf>, EngineError> {
    let Some(night_ids_iter) = obs_dataset.iter_night_id() else {
        return Ok(KFBankCollection::default());
    };
    let mut night_ids: Vec<NightId> = night_ids_iter.copied().collect();
    night_ids.sort_unstable();

    let mut banks = Vec::new();

    for night_id in night_ids {
        let Some(obs_iter) = obs_dataset.iter_night_observations(&night_id) else {
            continue;
        };
        let night_obs: Vec<&Observation> = obs_iter.collect();
        if night_obs.is_empty() {
            continue;
        }

        let t0 = night_obs
            .iter()
            .map(|o| o.mjd_tt())
            .fold(f64::INFINITY, f64::min);
        let time_binner = UniformTimeBinner::new(t0, time_binner_width);

        let bucket_index =
            build_alert_bucket_index(night_obs.iter().copied(), spatial_binner, &time_binner);

        let pairs = generate_pairs(&bucket_index, spatial_binner, &time_binner, pair_config);

        tracing::debug!(night = %night_id, n_pairs = pairs.len(), "night pairs generated");

        for pair in &pairs {
            match KFBank::from_grid(
                obs_dataset,
                pair.a,
                pair.b,
                kalman_context,
                grid_config,
                bank_config.clone(),
            ) {
                Ok(bank) => banks.push(KFPairBank {
                    night_id,
                    first_obs_id: *pair.a.id(),
                    second_obs_id: *pair.b.id(),
                    bank,
                }),
                Err(err) => {
                    tracing::debug!(
                        night = %night_id,
                        first = *pair.a.id(),
                        second = *pair.b.id(),
                        %err,
                        "KFBank::from_grid failed for pair, skipping"
                    );
                }
            }
        }
    }

    Ok(KFBankCollection { banks })
}
