//! Kalman "error box" sizing: bank-level predictive search-region radius
//! (reusing the exact same computation the engine uses for candidate
//! gating, [`KFBank::predict_search_region`]) and per-hypothesis sky-plane
//! uncertainty ellipses.

use nalgebra::Vector3;
use photom::observation_dataset::{ObsDataset, observation::Observation};

use fink_fat_engine::{
    engine_config::{EngineConfig, kalman_context::KalmanContext},
    spacetime_bucket::{
        bucket::{BucketIndex, build_alert_bucket_index},
        healpix_binner::HealpixBinner,
    },
    topocentric_kf::{
        branching::candidate_search::{SingleBinTimeBinner, find_candidates_for_bank},
        kalman_bank::{KFBank, ellipse_region_finder::radius_strategy::largest_eigenvalue_2x2},
        observer_state::get_observer,
    },
};

/// Resolved observer state and spatial index for the *next* night, built
/// once per night and shared across every bank's predictive error-box
/// computation (mirrors `orchestrate::resolve_observer_state` — reproduced
/// here since that helper is private to the engine crate, but built purely
/// from already-public engine APIs).
pub struct NextNightContext<'obs> {
    bucket_index: BucketIndex<&'obs Observation>,
    epoch: f64,
    r_obs: Vector3<f64>,
    v_obs: Vector3<f64>,
}

/// Build the next-night context, or `None` if there is no next night, it has
/// no observations, or the observer state fails to resolve at its epoch
/// (logged at `debug`, treated as "no predictive metrics this night" rather
/// than aborting the run).
pub fn build_next_night_context<'obs>(
    obs_dataset: &ObsDataset,
    kalman_context: &KalmanContext,
    next_night_obs: &[&'obs Observation],
    spatial_binner: &HealpixBinner,
) -> Option<NextNightContext<'obs>> {
    let representative_obs = *next_night_obs.first()?;
    let epoch = representative_obs.mjd_tt();

    let observer = get_observer(obs_dataset, representative_obs).ok()?;
    let helio_state = kalman_context
        .get_ephem()
        .helio_observer_state(observer, epoch)
        .ok()?;

    let bucket_index = build_alert_bucket_index(
        next_night_obs.iter().copied(),
        spatial_binner,
        &SingleBinTimeBinner,
    );

    Some(NextNightContext {
        bucket_index,
        epoch,
        r_obs: helio_state.helio_cart_pos,
        v_obs: helio_state.helio_cart_vel,
    })
}

/// One bank's predicted error box for the next night: bounding radius and
/// how many of the next night's observations actually fall inside it.
pub struct BankErrorBox {
    pub radius_arcsec: f64,
    pub n_observations_in_box: usize,
}

/// Predict `bank`'s search region at `next_night`'s epoch (exactly the
/// computation [`KFBank::predict_search_region`] performs for real
/// candidate gating) and count how many of the next night's observations
/// land inside it. `None` if the bank fails to propagate (e.g. singular
/// Jacobian) — logged at `debug` by the underlying engine call, dropped
/// from the night's aggregate rather than treated as an error.
pub fn bank_predictive_error_box(
    bank: &KFBank,
    next_night: &NextNightContext,
    spatial_binner: &HealpixBinner,
    engine_config: &EngineConfig,
) -> Option<BankErrorBox> {
    let advance_params = &engine_config.advance_params;

    let region = bank
        .predict_search_region(
            next_night.epoch,
            next_night.r_obs,
            next_night.v_obs,
            advance_params.obs_noise.into(),
            advance_params.top_k,
            advance_params.radius_strategy,
        )
        .ok()?;

    let radius_arcsec = region.radius_rad.to_degrees() * 3600.0;
    if !radius_arcsec.is_finite() {
        return None;
    }

    let candidates = find_candidates_for_bank(
        &region,
        bank.track_ids().to_vec(),
        &next_night.bucket_index,
        spatial_binner,
        engine_config.kfbank_config.gate_chi2,
        advance_params.likelihood_threshold,
    );

    Some(BankErrorBox {
        radius_arcsec,
        n_observations_in_box: candidates.matches.len(),
    })
}

/// Same ceiling as the engine's own `RadiusStrategy::Clamped` safety net
/// (`NightAdvanceParams`'s default, 30 arcmin) — a per-hypothesis radius
/// beyond this is not a real physical uncertainty, it's the numerically
/// ill-conditioned covariance a long-surviving lineage's drifting range
/// estimate can produce (see `propagate_covariance`'s doc in
/// `fink_fat_engine::topocentric_kf::single_kalman::propagate`). Filtered
/// out entirely rather than clamped to this value: a 1e14 arcsec reading
/// is not "30 arcmin degraded", it's an invalid measurement that would
/// still corrupt mean/median if merely clamped.
pub const MAX_HYPOTHESIS_RADIUS_ARCSEC: f64 = 1800.0;

/// Per-hypothesis 1σ sky-plane bounding radius (arcsec), for every live
/// hypothesis in `bank`. Hypotheses whose sky covariance is unavailable
/// (degenerate Jacobian) are skipped, as are hypotheses whose radius
/// exceeds [`MAX_HYPOTHESIS_RADIUS_ARCSEC`] (numerically ill-conditioned,
/// not a real uncertainty).
///
/// Returns the valid radii plus how many hypotheses were excluded for
/// exceeding the ceiling — the latter is a transparency signal: it should
/// be 0 once the engine-side root cause (drifting range estimate on a
/// long-surviving lineage) is fixed, so a nonzero count here is worth
/// investigating rather than silently dropping.
pub fn hypothesis_error_box_radii_arcsec(bank: &KFBank) -> (Vec<f64>, usize) {
    let mut n_excessive = 0;
    let radii = bank
        .hypotheses()
        .iter()
        .filter_map(|h| h.kf.sky_covariance().ok())
        .filter(|cov| cov.iter().all(|x| x.is_finite()))
        .map(|cov| largest_eigenvalue_2x2(&cov).max(0.0).sqrt().to_degrees() * 3600.0)
        .filter(|&r| {
            let ok = r <= MAX_HYPOTHESIS_RADIUS_ARCSEC;
            if !ok {
                n_excessive += 1;
            }
            ok
        })
        .collect();
    (radii, n_excessive)
}
