//! Admissible-region grid generator for the multi-hypothesis filter bank.
//!
//! A single tracklet constrains the two sky angles and the two angular rates,
//! leaving the topocentric range `ρ` and range-rate `ρ̇` undetermined. The set
//! of `(ρ, ρ̇)` that yield a physically plausible (Sun-bound) orbit is the
//! **admissible region** (Milani & Gronchi). This module tiles that region
//! with a grid of [`KFState`] seeds, each weighted by a prior over known
//! small-body populations.
//!
//! Relationship to [`crate::topocentric_kf::init::init_kf_state`]
//! ----------------------------------------------------------------
//! [`init_kf_state`] commits to a *single* range guess, resolved from a
//! Keplerian circular-orbit prior. [`admissible_region_grid`] instead tiles
//! the *whole* admissible region with many [`KFState`] hypotheses, so that a
//! [`crate::topocentric_kf::bank::KFBank`] can let the data — rather than a
//! prior assumption — decide which range is correct.
//!
//! Every resulting [`KFState`] is expressed in the same **attributable
//! coordinates** $(\alpha, \delta, \dot\alpha, \dot\delta, \rho, \dot\rho)$
//! used everywhere else in `topocentric_kf` (see the module-level
//! documentation of [`crate::topocentric_kf::KFState`]). The two angular
//! components and their rates come straight from the tracklet and are
//! identical for every node; only `(ρ, ρ̇)` — and their associated
//! variances — vary from node to node. This module therefore reuses
//! [`init`]'s elementary building blocks (tracklet geometry, observer state,
//! angular covariance blocks) instead of recomputing them, and only adds the
//! grid-specific pieces: range tiling, the bound-orbit `ρ̇` interval, and
//! population weighting.
//!
//! Energy boundary
//! ---------------
//! For a fixed `ρ`, with `r = ‖r_obs + ρ û‖`, the heliocentric velocity is
//! `v = v_T + ρ̇ û` where `v_T = v_obs + ρ ẇ` (`û = los`, `ẇ = los_dot`).
//! Requiring a bound orbit (`ε = ½‖v‖² − k²/r ≤ 0`) gives a quadratic in `ρ̇`:
//!
//! ```text
//! ρ̇² + 2(v_T·û) ρ̇ + (‖v_T‖² − 2k²/r) ≤ 0
//! ```
//!
//! whose two roots bracket the admissible `ρ̇` interval. Ranges where the
//! discriminant is negative admit no bound orbit and are skipped.
//!
//! Per-node covariance
//! --------------------
//! The angular position/rate blocks $(\alpha, \delta, \dot\alpha,
//! \dot\delta)$ are shared by every node — they only depend on the
//! tracklet, never on `(ρ, ρ̇)`. The range and range-rate blocks are
//! node-specific: crucially, their σ are set to **half the local grid-cell
//! width**, not to the full range ambiguity. Adjacent nodes tile the
//! `(ρ, ρ̇)` plane, so the *bank* — not any single covariance — carries the
//! ambiguity. This keeps every hypothesis well-conditioned.

use nalgebra::Vector6;
use outfit::constants::GAUSS_GRAV_SQUARED;
use photom::observation_dataset::{ObsDataset, observation::Observation};

use tracing::{trace, trace_span};

use crate::{
    engine_config::{
        grid_population::{GridConfig, Population},
        kalman_context::KalmanContext,
    },
    error::EngineError,
    topocentric_kf::{
        observer_state::HelioObsState,
        single_kalman::{
            KFState,
            init::{
                TrackletGeometry, angular_position_variances, angular_rate_variances,
                assemble_diagonal_covariance, init_observer_state, pair_midpoint_epoch,
                tracklet_geometry,
            },
        },
    },
};

/// Gaussian-mixture prior density over semi-major axis.
fn population_weight(a: f64, populations: &[Population]) -> f64 {
    populations
        .iter()
        .map(|p| {
            let z = (a - p.a_center) / p.a_sigma;
            p.weight * (-0.5 * z * z).exp()
        })
        .sum()
}

/// Half-width of the log-spaced range cell around `rho` (used as the range σ
/// so that adjacent cells tile the range axis).
fn rho_cell_halfwidth(rho: f64, config: &GridConfig) -> f64 {
    if config.n_rho <= 1 {
        return 0.5 * (config.rho_max - config.rho_min);
    }
    let ratio = (config.rho_max / config.rho_min).powf(1.0 / (config.n_rho - 1) as f64);
    // ≈ ¼·ρ·(ratio − 1/ratio): half the mean linear spacing to the neighbours.
    0.25 * rho * (ratio - 1.0 / ratio)
}

/// Build the admissible-region seed grid for one observation pair.
///
/// Arguments mirror [`crate::topocentric_kf::init::init_kf_state`]: the
/// observation dataset, the pair of observations, and the ephemeris `state`
/// used to resolve the observer's heliocentric position/velocity. Every
/// resulting [`KFState`] carries the same `state` reference, so each
/// hypothesis can later be propagated and updated exactly like a
/// single-guess [`KFState`].
///
/// Returns `(state, weight)` seeds ready for `KFBank::from_seeds`. Every seed
/// is a Sun-bound orbit; hyperbolic `(ρ, ρ̇)` combinations are excluded by
/// construction.
///
/// # Errors
///
/// Propagates any failure to resolve the observer's heliocentric state (e.g.
/// an unknown observatory code), exactly like [`init_kf_state`].
pub fn admissible_region_grid<'state_lf>(
    obs_dataset: &ObsDataset,
    first_obs: &Observation,
    second_obs: &Observation,
    state: &'state_lf KalmanContext,
    config: &GridConfig,
) -> Result<Vec<(KFState<'state_lf>, f64)>, EngineError> {
    let t_mid = pair_midpoint_epoch(first_obs, second_obs);

    let span = trace_span!(
        "admissible_region_grid",
        t_mid = t_mid,
        n_rho = config.n_rho,
        n_rho_dot = config.n_rho_dot,
        rho_min = config.rho_min,
        rho_max = config.rho_max,
    );
    let _enter = span.enter();

    trace!(
        obs1_epoch = first_obs.mjd_tt(),
        obs2_epoch = second_obs.mjd_tt(),
        "Starting grid generation from tracklet pair"
    );

    // Observer heliocentric position/velocity at the pair's midpoint epoch —
    // the same helper used by `init_kf_state`'s single-guess strategy.
    let HelioObsState {
        helio_cart_pos: r_obs,
        helio_cart_vel: v_obs,
    } = init_observer_state(obs_dataset, first_obs, second_obs, state.get_ephem())?;

    // Tracklet geometry (midpoint angles/rates and line-of-sight vectors),
    // identical to what `init_kf_state` uses for its single best-guess range.
    let TrackletGeometry {
        mid_point,
        mid_speed,
        los,
        los_dot,
    } = tracklet_geometry(first_obs, second_obs);

    // The angular position/rate covariance blocks depend only on the
    // tracklet astrometry, never on (rho, rho_dot): every node in the grid
    // shares exactly the same values, so compute them once outside the loop.
    let (var_ra, var_dec) = angular_position_variances(&mid_point);
    let (var_ra_dot, var_dec_dot) = angular_rate_variances(&mid_speed);

    let k2 = GAUSS_GRAV_SQUARED;
    let r_obs_norm2 = r_obs.norm_squared();
    let r_obs_dot_u = r_obs.dot(&los);

    let log_rho_min = config.rho_min.ln();
    let log_rho_max = config.rho_max.ln();

    let mut seeds: Vec<(KFState<'state_lf>, f64)> = Vec::new();

    for i in 0..config.n_rho {
        let frac = if config.n_rho == 1 {
            0.5
        } else {
            i as f64 / (config.n_rho - 1) as f64
        };
        let rho = (log_rho_min + frac * (log_rho_max - log_rho_min)).exp();

        // Heliocentric distance at this range (law of cosines).
        let r_helio = (r_obs_norm2 + 2.0 * rho * r_obs_dot_u + rho * rho).sqrt();

        // rho_dot-independent part of the heliocentric velocity, and the
        // bound (Sun-bound) rho_dot interval at this range.
        let v_transverse = v_obs + rho * los_dot;
        let v_transverse_dot_u = v_transverse.dot(&los);
        let discriminant = v_transverse_dot_u * v_transverse_dot_u
            - (v_transverse.norm_squared() - 2.0 * k2 / r_helio);
        if discriminant <= 0.0 {
            trace!(
                i_rho = i,
                rho_au = rho,
                r_helio_au = r_helio,
                discriminant,
                "Skipping range: no bound orbit (discriminant <= 0)"
            );
            continue; // No Sun-bound orbit is possible at this range.
        }
        let sqrt_discriminant = discriminant.sqrt();
        let rho_dot_lo = -v_transverse_dot_u - sqrt_discriminant;
        let rho_dot_hi = -v_transverse_dot_u + sqrt_discriminant;
        let rho_dot_interval = rho_dot_hi - rho_dot_lo;

        trace!(
            i_rho = i,
            rho_au = rho,
            r_helio_au = r_helio,
            rho_dot_min_au_per_day = rho_dot_lo,
            rho_dot_max_au_per_day = rho_dot_hi,
            interval_au_per_day = rho_dot_interval,
            "Valid range: bound-orbit rho_dot interval found"
        );

        // Per-node range/range-rate covariance: half the local grid-cell
        // width (range axis) and half the local bound-orbit interval per
        // sampled node (range-rate axis) — see the module-level doc comment.
        let sigma_rho = rho_cell_halfwidth(rho, config).max(config.sigma_pos_au_floor);
        let sigma_rho_dot =
            (0.5 * rho_dot_interval / config.n_rho_dot as f64).max(config.sigma_rho_dot_floor);
        let var_rho = sigma_rho * sigma_rho;
        let var_rho_dot = sigma_rho_dot * sigma_rho_dot;

        for j in 0..config.n_rho_dot {
            // Midpoint sampling keeps every node strictly inside the
            // interval (the endpoints are parabolic, a → ∞ there).
            let frac_j = (j as f64 + 0.5) / config.n_rho_dot as f64;
            let rho_dot = rho_dot_lo + frac_j * rho_dot_interval;

            // Heliocentric velocity for this (rho, rho_dot) node, used only
            // to evaluate the orbit's energy/semi-major axis for population
            // weighting. The KFState seed itself stays in attributable
            // coordinates: this Cartesian velocity is a local, throwaway
            // quantity, never stored.
            let velocity = v_transverse + rho_dot * los;
            let energy = 0.5 * velocity.norm_squared() - k2 / r_helio;
            if energy >= 0.0 {
                trace!(
                    i_rho = i,
                    j_rho_dot = j,
                    rho_au = rho,
                    rho_dot_au_per_day = rho_dot,
                    energy,
                    "Skipping node: energy >= 0 (near boundary, numerical safety)"
                );
                continue; // Numerical safety margin near the bound-orbit boundary.
            }
            let semi_major_axis = -k2 / (2.0 * energy);

            let weight = population_weight(semi_major_axis, &config.populations);
            if weight < config.weight_floor {
                trace!(
                    i_rho = i,
                    j_rho_dot = j,
                    rho_au = rho,
                    rho_dot_au_per_day = rho_dot,
                    a_au = semi_major_axis,
                    weight,
                    weight_floor = config.weight_floor,
                    "Skipping node: weight below floor (not in populations)"
                );
                continue;
            }

            trace!(
                i_rho = i,
                j_rho_dot = j,
                rho_au = rho,
                rho_dot_au_per_day = rho_dot,
                a_au = semi_major_axis,
                weight,
                "Node accepted: bound Sun-bound orbit"
            );

            // Attributable state vector for this node: the angular position
            // and angular rate are shared with every other node (they come
            // straight from the tracklet); rho and rho_dot are this node's
            // own grid coordinates.
            let state_vector = Vector6::new(
                mid_point.ra,
                mid_point.dec,
                mid_speed.0.ra,
                mid_speed.0.dec,
                rho,
                rho_dot,
            );
            let covariance = assemble_diagonal_covariance(
                var_ra,
                var_dec,
                var_ra_dot,
                var_dec_dot,
                var_rho,
                var_rho_dot,
            );

            seeds.push((
                KFState {
                    state: state_vector,
                    covariance,
                    epoch: t_mid,

                    r_obs,
                    v_obs,
                    universal_anomaly: None,

                    kalman_gain: None,
                    nis_ema: None,

                    shared_ctx: state,
                },
                weight,
            ));
        }
    }

    trace!(
        n_seeds = seeds.len(),
        n_rho_samples = config.n_rho,
        n_rho_dot_samples = config.n_rho_dot,
        max_possible = config.n_rho * config.n_rho_dot,
        "Grid generation complete"
    );

    Ok(seeds)
}
