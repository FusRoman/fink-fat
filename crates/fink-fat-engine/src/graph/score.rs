//! Inter-night candidate scoring between `SeedNode`s.
//!
//! # Overview
//! This module computes **gated, additive costs** for directed edges `i → j`
//! between seed candidates across revisits (inter-night or intra-night).
//! Scores are designed to be **interpretable** and **sparse-friendly** so they
//! can feed bipartite solvers (e.g., Hungarian/Auction) or a global
//! **min-cost flow** after Top-K pruning.
//!
//! ## Pipeline (conceptual)
//! For a source seed `i` at revisit `R_k` and a candidate seed `j` at
//! revisit `R_{k+Δ}`:
//!
//! 1. **Predict** `i` to `t_j = epoch_mid(j)` on `i`’s **own tangent plane**,
//! 2. **Project** `j` to the **same plane** and build the position residual,
//! 3. Form a **diagonal** plane covariance `S = Σ̂_i(t_j) + Σ_pos(j)`,
//! 4. Compute **Mahalanobis** distance `d²_pos = Δpᵀ S⁻¹ Δp` and apply a **hard gate**,
//! 5. Add **kinematics** (direction / speed) using `j`’s finite-difference velocity
//!    in `i`’s plane and gate extreme mismatches,
//! 6. Add **photometry** via a robust z-score with a noise **floor**,
//! 7. Add a **gap penalty** if `Δ > 1` revisits,
//! 8. Sum weighted components into a final **cost** (lower is better).
//!
//! ## Design choices
//! - Geometry and kinematics are evaluated on the **tangent plane of `i`** to avoid
//!   mixing frames and to keep the covariance combination simple (diagonal model).
//! - The plane covariance is **diagonal**, consistent with extraction heuristics;
//!   this yields a simple, interpretable `d²_pos` while remaining robust.
//! - `j`’s plane velocity is estimated by a **symmetric finite difference** around `t_j`,
//!   making it robust to small projection offsets and to mild propagation errors.
//! - Photometry uses a **pooled σ** plus a **floor** to prevent over-weighting.
//!
//! ## Units & conventions
//! - Angles: **radians**,
//! - Times: **days** (MJD TT),
//! - Speeds: **radians/day**,
//! - Fluxes: **nJy**,
//! - Costs: **dimensionless**, additive (lower is better).
//!
//! ## Determinism
//! Given the same inputs and configuration, the scoring is deterministic. Any
//! stochasticity must come from upstream seed building or noise realizations.
//!
//! ## Gating vs. weighting
//! - **Hard gates** (`ScoreGates`) immediately **reject** an edge (return `None`).
//! - **Weights** (`ScoreWeights`) control **relative influence** of accepted terms.
//!
//! ## Tuning strategy
//! - Start with conservative position gating (`max_d2_pos ≈ 9.0`, about 3σ in 2D),
//! - Enable velocity **direction** first (`theta0` a few degrees), then **speed**,
//! - Calibrate photometry with a realistic `flux_sigma_floor`,
//! - Set `gap_rho` near 1 for linear penalties in missed revisits, increase to
//!   penalize long gaps more aggressively.
//!
//! ## Failure modes & guards
//! - **Zero/near-zero** variances lead to infinite weights; `d²_pos` will exceed gate,
//! - Degenerate velocities (‖v‖ ~ 0) disable kinematic terms by design,
//! - Non-finite numbers are treated as hard-gate failures.
//!
//! ## See also
//! - `crate::link::solvers` — bipartite assignment backends,
//! - `crate::propagation::features` — seed representation and propagation,
//! - Future work: trajectory-level smoothing / flow constraints.

/* ---------------------------- Score outputs --------------------------- */

use crate::{
    astro_math::{l2_norm, radec_to_tangent},
    engine_config::score_config::{
        GapScoreConfig, InterNightScoreConfig, NumericConfig, PhotometryScoreConfig,
        PositionScoreConfig, PredictConfig, VelocityScoreConfig,
    },
    seeding::{seed_id::SeedId, seed_node::SeedNode},
};

/// Decomposed score for a single directed edge `i → j`.
///
/// Contains both the **total cost** and **component-level diagnostics** for
/// interpretability and tuning.
#[derive(Clone, Debug)]
pub struct ScoredEdge {
    /// Source/target seed identifiers.
    pub from: SeedId,
    pub to: SeedId,
    /// Time gap (**days**): `t_j - t_i`.
    pub dt_days: f64,
    /// Additive total cost after gating and weighting (lower is better).
    pub cost: f64,
    /// Component-wise diagnostics useful for calibration and debugging.
    pub components: ScoreComponents,
}

/// Individual penalty components (pre-weights).
///
/// Missing/undefined components (e.g., degenerate velocity) are encoded as `None`
/// and contribute `0` to the final cost.
#[derive(Clone, Debug, Default)]
pub struct ScoreComponents {
    /// Plane-position Mahalanobis distance **squared**.
    pub d2_pos: f64,
    /// Velocity **direction** mismatch (**radians**); `None` if ‖v‖ ~ 0 for either seed.
    pub vel_angle_rad: Option<f64>,
    /// Velocity **speed** mismatch (**radians/day**); `None` if ‖v‖ ~ 0 for either seed.
    pub vel_speed_diff: Option<f64>,
    /// Photometry z-score: `|ΔF| / σ_pool`.
    pub z_flux: Option<f64>,
    /// Gap penalty `(Δ - 1)^rho` (0 if `Δ ≤ 1`).
    pub gap_penalty: f64,
    /// True if bands differ (`band(i) != band(j)`).
    pub band_mismatch: bool,
}

/* ----------------------------- Public API ----------------------------- */

impl ScoredEdge {
    /// Compute the **gated** score for the directed edge `i → j`.
    ///
    /// Returns
    /// -------
    /// * `Some(ScoredEdge)` if all hard gates pass,
    /// * `None` if any gate is violated or a non-finite value is encountered.
    ///
    /// Method
    /// ------
    /// 1. Predict `i` to `t_j` on `i`’s plane and combine covariances (diagonal),
    /// 2. Compute plane residual and Mahalanobis `d²_pos` (position gate),
    /// 3. Estimate `j`’s plane velocity by symmetric finite difference and
    ///    compare to `i`’s predicted velocity (direction/speed gates),
    /// 4. Add photometric z-score and band penalty (optional),
    /// 5. Add gap penalty for `Δ > 1`,
    /// 6. Sum weighted components into `cost`.
    ///
    /// Units
    /// -----
    /// - Plane positions in **radians**,
    /// - Speeds in **radians/day**,
    /// - Fluxes in **nJy**,
    /// - Time in **days** (MJD TT).
    pub fn score(
        i: &SeedNode,
        j: &SeedNode,
        cfg: &InterNightScoreConfig,
        delta_revisit: u32,
    ) -> Option<Self> {
        // 1) Position term
        let d2_pos = compute_position_term(i, j, &cfg.predict, &cfg.position, &cfg.numeric)?;

        // 2) Time gap
        let dt_days = j.plane.epoch_mid - i.plane.epoch_mid;

        // 3) Velocity terms
        let (vel_angle_rad, vel_speed_diff) = compute_velocity_terms(i, j, dt_days, &cfg.velocity)?;

        // 4) Photometry
        let z_flux = compute_flux_z(i, j, &cfg.photometry);

        // 5) Gap penalty
        let gap_penalty = compute_gap_penalty(delta_revisit, &cfg.gap);

        // 6) Band mismatch
        let band_mismatch = i.photom.band != j.photom.band;

        // 7) Compose
        let components = ScoreComponents {
            d2_pos,
            vel_angle_rad,
            vel_speed_diff,
            z_flux,
            gap_penalty,
            band_mismatch,
        };

        let cost = compose_cost(&components, cfg);

        Some(Self {
            from: i.seed_id,
            to: j.seed_id,
            dt_days,
            cost,
            components,
        })
    }
}

/* ------------------------------ Helpers ------------------------------ */

/// Compute the position term `d²_pos` on the tangent plane of `i`,
/// including the hard gate on `max_d2_pos`.
///
/// Returns `None` if the Mahalanobis distance is non-finite or exceeds
/// the configured gate.
fn compute_position_term(
    i: &SeedNode,
    j: &SeedNode,
    predict: &PredictConfig,
    cfg: &PositionScoreConfig,
    numeric: &NumericConfig,
) -> Option<f64> {
    let t_j = j.plane.epoch_mid;

    // Predict i to t_j on its own tangent plane.
    let (p_hat, cov_i) = i.plane.predict_on_plane(t_j, &predict.noise);
    let (px, py) = (p_hat[0], p_hat[1]);

    // Project j to i's plane.
    let p_j = radec_to_tangent(
        j.plane.ra_mid,
        j.plane.dec_mid,
        i.plane.center.ra0,
        i.plane.center.dec0,
    );
    let (dx, dy) = (p_j[0] - px, p_j[1] - py);

    // Diagonal covariance S = Σ̂_i(tj) + Σ_pos(j).
    let mut sxx = (cov_i[0][0] + j.plane.cov_pos[0][0]).max(0.0);
    let mut syy = (cov_i[1][1] + j.plane.cov_pos[1][1]).max(0.0);

    // Optional numeric floor for stability.
    if numeric.min_variance > 0.0 {
        sxx = sxx.max(numeric.min_variance);
        syy = syy.max(numeric.min_variance);
    }

    let inv_sxx = 1.0 / sxx;
    let inv_syy = 1.0 / syy;

    let d2_pos = dx * dx * inv_sxx + dy * dy * inv_syy;

    if !d2_pos.is_finite() || d2_pos > cfg.gate.max_d2_pos {
        return None;
    }
    Some(d2_pos)
}

/// Compute velocity-related diagnostics (direction and speed mismatch)
/// evaluated on `i`'s plane at `t_j = j.epoch_mid`.
///
/// This function enforces the velocity gates:
/// - angular gate via `cos_max_theta_vel`,
/// - speed gate via `max_speed_diff`.
///
/// Returns
/// -------
/// * `None` if any velocity gate is violated,
/// * `Some((vel_angle_rad, vel_speed_diff))` otherwise, with `None` entries
///   when the corresponding weight is zero or velocities are degenerate.
fn compute_velocity_terms(
    i: &SeedNode,
    j: &SeedNode,
    dt_days: f64,
    cfg: &VelocityScoreConfig,
) -> Option<(Option<f64>, Option<f64>)> {
    // Predict velocity of i at t_j on its plane.
    let vi = if let Some(a) = i.plane.acc_xy {
        [
            i.plane.vel_xy[0] + a[0] * dt_days,
            i.plane.vel_xy[1] + a[1] * dt_days,
        ]
    } else {
        i.plane.vel_xy
    };

    // Symmetric finite difference for j, measured in i's plane.
    let t_j = j.plane.epoch_mid;
    let eps = cfg.scale.vel_eps_days;
    let inv_2eps = 1.0 / (2.0 * eps);

    let (ra_p, dec_p) = j.predict_radec(t_j + eps);
    let (ra_m, dec_m) = j.predict_radec(t_j - eps);

    let p_plus = i.plane.radec_to_tangent_precomp(ra_p, dec_p);
    let p_minus = i.plane.radec_to_tangent_precomp(ra_m, dec_m);

    let vj = [
        (p_plus[0] - p_minus[0]) * inv_2eps,
        (p_plus[1] - p_minus[1]) * inv_2eps,
    ];

    let norm_vi = l2_norm(vi[0], vi[1]);
    let norm_vj = l2_norm(vj[0], vj[1]);

    // Degenerate => disable kinematics (no gating).
    if norm_vi == 0.0 || norm_vj == 0.0 {
        return Some((None, None));
    }

    let inv_norms = 1.0 / (norm_vi * norm_vj);
    let cosang = ((vi[0] * vj[0] + vi[1] * vj[1]) * inv_norms).clamp(-1.0, 1.0);

    // Direction gate
    if cosang < cfg.gate.cos_max_theta_vel() {
        return None;
    }

    // Speed gate
    let dv = (norm_vi - norm_vj).abs();
    if dv > cfg.gate.max_speed_diff {
        return None;
    }

    // Diagnostics only if weighted
    let vel_angle_rad = (cfg.weight.w_vel_dir > 0.0).then(|| cosang.acos());
    let vel_speed_diff = (cfg.weight.w_vel_norm > 0.0).then(|| dv);

    Some((vel_angle_rad, vel_speed_diff))
}

/// Compute the photometric z-score `z_flux = |ΔF| / σ_pool`.
///
/// No hard gate is applied: if the term cannot be evaluated reliably
/// (non-finite or zero variance), this function returns `None` and the
/// photometric term simply does not contribute to the cost.
fn compute_flux_z(i: &SeedNode, j: &SeedNode, cfg: &PhotometryScoreConfig) -> Option<f64> {
    if cfg.weight.w_flux <= 0.0 {
        return None;
    }

    let df = (j.photom.flux_mean as f64) - (i.photom.flux_mean as f64);
    let s_i = i.photom.flux_std as f64;
    let s_j = j.photom.flux_std as f64;

    let sigma_floor = cfg.scale.flux_sigma_floor;
    let sigma_sq = s_i * s_i + s_j * s_j + sigma_floor * sigma_floor;

    if !sigma_sq.is_finite() || sigma_sq <= 0.0 {
        return None;
    }

    let z = df.abs() / sigma_sq.sqrt();
    z.is_finite().then_some(z)
}

/// Compute the gap penalty `(Δ - 1)^rho` for `Δ > 1`, or `0` otherwise.
fn compute_gap_penalty(delta_revisit: u32, cfg: &GapScoreConfig) -> f64 {
    if delta_revisit > 1 {
        let delta = (delta_revisit as f64) - 1.0;
        delta.powf(cfg.scale.rho)
    } else {
        0.0
    }
}

/// Compose the final scalar cost from the decomposed components and
/// the scoring configuration.
///
/// This function is pure: it does not perform any gating, it simply
/// applies the configured weights.
fn compose_cost(components: &ScoreComponents, cfg: &InterNightScoreConfig) -> f64 {
    let mut cost = 0.0;

    // Position
    cost += cfg.position.weight.w_pos * components.d2_pos;

    // Velocity
    if let Some(theta) = components.vel_angle_rad {
        cost += cfg.velocity.weight.w_vel_dir * (theta / cfg.velocity.scale.theta0);
    }
    if let Some(dv) = components.vel_speed_diff {
        cost += cfg.velocity.weight.w_vel_norm * (dv / cfg.velocity.scale.v0);
    }

    // Photometry
    if let Some(z) = components.z_flux {
        cost += cfg.photometry.weight.w_flux * z;
    }

    // Gap
    cost += cfg.gap.weight.w_gap * components.gap_penalty;

    // Band mismatch
    if components.band_mismatch {
        cost += cfg.band.weight.w_band_mismatch;
    }

    cost
}
