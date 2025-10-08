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

use crate::propagation::features::{radec_to_tangent, ModelNoise, SeedId, SeedNode};

/* ------------------------------ Weights ------------------------------ */

/// Additive weights for the edge **cost**.
///
/// The final cost (schematic) is:
///
/// `cost = w_pos * d2_pos
///       + w_vel_dir  * (theta / theta0)
///       + w_vel_norm * (| |v_i| - |v_j| | / v0)
///       + w_flux     * z_flux
///       + w_gap      * gap(Δ)
///       + band_mismatch * w_band_mismatch`
///
/// where missing (uninformative) components contribute `0`.
#[derive(Clone, Copy, Debug)]
pub struct ScoreWeights {
    /// Position term (Mahalanobis d² on the plane). Strongly discriminative.
    pub w_pos: f64,
    /// Direction consistency (angle between `v_i` and `v_j`, normalized by `theta0`).
    pub w_vel_dir: f64,
    /// Speed consistency (absolute speed difference, normalized by `v0`).
    pub w_vel_norm: f64,
    /// Photometry penalty (flux z-score).
    pub w_flux: f64,
    /// Gap penalty weight for Δ>1 revisits.
    pub w_gap: f64,
    /// Small additive penalty when `band(i) != band(j)`. Set to `0.0` to disable.
    pub w_band_mismatch: f64,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            w_pos: 0.5,           // strong — geometry dominates
            w_vel_dir: 0.3,       // moderate — useful when geometry is ambiguous
            w_vel_norm: 0.3,      // moderate — speed mismatch as consistency check
            w_flux: 0.5,          // moderate — helps against confusions
            w_gap: 0.3,           // mild to moderate — encourages short gaps
            w_band_mismatch: 0.2, // small — cross-band matches are possible but less likely
        }
    }
}

/* ------------------------------- Gates ------------------------------- */

/// **Hard** gates: if any is violated, the edge is rejected (`None`).
#[derive(Clone, Copy, Debug)]
pub struct ScoreGates {
    /// Max Mahalanobis `d²_pos` (e.g., `~9.0` ≈ 3σ in 2D).
    pub max_d2_pos: f64,
    /// Max angle between velocity directions (**radians**).
    pub max_theta_vel: f64,
    /// Max **absolute** speed difference (**radians/day**).
    pub max_speed_diff: f64,
}

impl Default for ScoreGates {
    fn default() -> Self {
        Self {
            max_d2_pos: 9.0,                      // ≈ 3σ in 2D
            max_theta_vel: 10.0_f64.to_radians(), // ~10°
            max_speed_diff: f64::INFINITY,        // disabled by default
        }
    }
}

/* ------------------------------- Scales ------------------------------ */

/// Scaling constants and numerical knobs for the score components.
///
/// These parameters set the **natural scales** for the normalized penalties and
/// control the finite-difference step for kinematics.
#[derive(Clone, Copy, Debug)]
pub struct ScoreScales {
    /// Direction reference angle (**radians**) used to scale `theta`.
    pub theta0: f64,
    /// Speed reference (**radians/day**) used to scale `| |v_i| - |v_j| |`.
    pub v0: f64,
    /// Photometry σ floor (**nJy**) for robust pooling:
    /// `σ_pool = sqrt(σ_i² + σ_j² + σ_floor²)`.
    pub flux_sigma_floor: f64,
    /// Gap exponent: `gap(Δ) = (Δ - 1)^rho` if `Δ > 1`, else `0`.
    pub gap_rho: f64,
    /// Symmetric finite-difference step (**days**) for `j`’s plane velocity.
    pub vel_eps_days: f64,
}

impl Default for ScoreScales {
    fn default() -> Self {
        Self {
            theta0: 5.0_f64.to_radians(), // a few degrees
            v0: 0.005,                    // ~0.29 deg/day in rad/day
            flux_sigma_floor: 50.0,       // tune to the survey noise model
            gap_rho: 1.0,                 // linear penalty in (Δ - 1)
            vel_eps_days: 1e-3,           // ~86.4 s; small but safely > integration jitter
        }
    }
}

/* ------------------------------- Config ------------------------------ */

/// Complete configuration to score edges.
///
/// - `noise` feeds the prediction covariance for `i` at `t_j`,
/// - `weights` shape the additive objective,
/// - `gates` prune implausible candidates early,
/// - `scales` normalize penalties and set numerical steps.
#[derive(Clone, Copy, Debug)]
pub struct ScoreConfig {
    /// Model noise for `i`’s prediction covariance at `t_j`.
    pub noise: ModelNoise,
    /// Additive weights for the cost.
    pub weights: ScoreWeights,
    /// Hard gates for pruning.
    pub gates: ScoreGates,
    /// Scaling constants and numerical knobs.
    pub scales: ScoreScales,
}

/* ---------------------------- Score outputs --------------------------- */

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
    pub fn score(i: &SeedNode, j: &SeedNode, cfg: ScoreConfig, delta_revisit: u32) -> Option<Self> {
        // --- 1) Predict i at t_j (mean & plane covariance) --------------------
        let t_j = j.epoch_mid;
        let (p_hat, s_i) = i.predict_on_plane(t_j, cfg.noise);
        let (px, py) = (p_hat[0], p_hat[1]);

        // --- 2) Project j to i's plane ---------------------------------------
        let p_j = radec_to_tangent(j.ra_mid, j.dec_mid, i.center_ra, i.center_dec);
        let dp = [p_j[0] - px, p_j[1] - py];

        // --- 3) Merge covariance: S = Σ̂_i(tj) + Σ_pos(j) (diagonal) ----------
        let sxx = (s_i[0][0] + j.cov_pos[0][0]).max(0.0);
        let syy = (s_i[1][1] + j.cov_pos[1][1]).max(0.0);

        // Invert diagonal (if zero, the corresponding term tends to +∞ and gets gated)
        let inv_sxx = if sxx > 0.0 { 1.0 / sxx } else { f64::INFINITY };
        let inv_syy = if syy > 0.0 { 1.0 / syy } else { f64::INFINITY };

        // Mahalanobis d² (diagonal)
        let d2_pos = dp[0] * dp[0] * inv_sxx + dp[1] * dp[1] * inv_syy;

        // Hard gate #1: position
        if !(d2_pos.is_finite()) || d2_pos > cfg.gates.max_d2_pos {
            return None;
        }

        // --- 4) Velocity consistency in i's plane -----------------------------
        // Predict velocity of i at t_j on its own plane (constant or quadratic)
        let dt = t_j - i.epoch_mid;
        let vi = if let Some(a) = i.acc_xy {
            [i.vel_xy[0] + a[0] * dt, i.vel_xy[1] + a[1] * dt]
        } else {
            i.vel_xy
        };

        // Estimate v_j in i's plane via symmetric finite difference of (RA,Dec)->plane.
        let eps = cfg.scales.vel_eps_days;
        let (ra_p, dec_p) = j.predict_radec(t_j + eps);
        let (ra_m, dec_m) = j.predict_radec(t_j - eps);
        let p_plus = radec_to_tangent(ra_p, dec_p, i.center_ra, i.center_dec);
        let p_minus = radec_to_tangent(ra_m, dec_m, i.center_ra, i.center_dec);
        let vj = [
            (p_plus[0] - p_minus[0]) / (2.0 * eps),
            (p_plus[1] - p_minus[1]) / (2.0 * eps),
        ];

        let norm_vi = l2_norm(vi[0], vi[1]);
        let norm_vj = l2_norm(vj[0], vj[1]);

        // If one norm is ~0, skip velocity terms (they're not informative).
        let (mut vel_angle, mut vel_speed_diff) = (None, None);
        if norm_vi > 0.0 && norm_vj > 0.0 {
            let cosang = ((vi[0] * vj[0] + vi[1] * vj[1]) / (norm_vi * norm_vj)).clamp(-1.0, 1.0);
            let theta = cosang.acos(); // radians
            let dv = (norm_vi - norm_vj).abs();

            // Hard gates #2-3
            if theta > cfg.gates.max_theta_vel || dv > cfg.gates.max_speed_diff {
                return None;
            }

            vel_angle = Some(theta);
            vel_speed_diff = Some(dv);
        }

        // --- 5) Photometry ----------------------------------------------------
        let mut z_flux = None;
        if cfg.weights.w_flux > 0.0 {
            let df = (j.flux_mean as f64) - (i.flux_mean as f64);
            let s_i = i.flux_std as f64;
            let s_j = j.flux_std as f64;
            let sigma =
                (s_i * s_i + s_j * s_j + cfg.scales.flux_sigma_floor * cfg.scales.flux_sigma_floor)
                    .sqrt();
            if sigma.is_finite() && sigma > 0.0 {
                z_flux = Some(df.abs() / sigma);
            }
        }

        // --- 6) Gap penalty ---------------------------------------------------
        let gap_penalty = if delta_revisit > 1 {
            ((delta_revisit as f64) - 1.0).powf(cfg.scales.gap_rho)
        } else {
            0.0
        };

        // --- 7) Compose weighted cost ----------------------------------------
        let comps = ScoreComponents {
            d2_pos,
            vel_angle_rad: vel_angle,
            vel_speed_diff,
            z_flux,
            gap_penalty,
            band_mismatch: i.band != j.band,
        };

        let cost = cfg.weights.w_pos * comps.d2_pos
            + comps
                .vel_angle_rad
                .map_or(0.0, |th| cfg.weights.w_vel_dir * (th / cfg.scales.theta0))
            + comps
                .vel_speed_diff
                .map_or(0.0, |dv| cfg.weights.w_vel_norm * (dv / cfg.scales.v0))
            + comps.z_flux.map_or(0.0, |z| cfg.weights.w_flux * z)
            + cfg.weights.w_gap * comps.gap_penalty
            + if comps.band_mismatch {
                cfg.weights.w_band_mismatch
            } else {
                0.0
            };

        Some(Self {
            from: i.seed_id,
            to: j.seed_id,
            dt_days: dt,
            cost,
            components: comps,
        })
    }
}

/* ------------------------------ Helpers ------------------------------ */

/// Euclidean L2 norm of a 2D vector.
///
/// Uses `f64::hypot(x, y)` for better numerical stability than `sqrt(x*x + y*y)`.
#[inline]
fn l2_norm(x: f64, y: f64) -> f64 {
    x.hypot(y)
}

/* --------------------------- Tests --------------------------- */

#[cfg(test)]
mod scoring_tests {
    use std::f64::consts::PI;

    use super::*;
    use proptest::prelude::*;

    // Local copy of inverse gnomonic for building consistent seeds in tests.
    #[inline]
    fn tangent_to_radec(x: f64, y: f64, ra0: f64, dec0: f64) -> (f64, f64) {
        const TWO_PI: f64 = std::f64::consts::PI * 2.0;
        let rho2 = x * x + y * y;
        if rho2 < 1e-24 {
            return (ra0.rem_euclid(TWO_PI), dec0);
        }
        let rho = rho2.sqrt();
        let c = rho.atan();
        let (sc, cc) = c.sin_cos();
        let (s0, c0) = dec0.sin_cos();
        let dec = (cc * s0 + (y * sc * c0) / rho).asin();
        let denom = rho * c0 * cc - y * s0 * sc;
        let ra = ra0 + (x * sc).atan2(denom);
        (ra.rem_euclid(TWO_PI), dec)
    }

    fn mk_seed(
        seed_id: u64,
        ra0: f64,
        dec0: f64,
        pos_xy: [f64; 2],
        vel_xy: [f64; 2],
        flux_mean: f32,
        band: u8,
    ) -> SeedNode {
        let (ra_mid, dec_mid) = tangent_to_radec(pos_xy[0], pos_xy[1], ra0, dec0);
        SeedNode {
            seed_id,
            night_id: 0,
            epoch_mid: 59000.0,
            pos_xy,
            vel_xy,
            cov_pos: [
                [(0.2_f64.to_radians() / 3600.0).powi(2), 0.0],
                [0.0, (0.2_f64.to_radians() / 3600.0).powi(2)],
            ],
            cov_vel: [
                [(0.02_f64.to_radians() / 3600.0).powi(2), 0.0],
                [0.0, (0.02_f64.to_radians() / 3600.0).powi(2)],
            ],
            acc_xy: None,
            flux_mean,
            flux_std: 50.0,
            band,
            n_obs: 2,
            members: vec![],
            center_ra: ra0,
            center_dec: dec0,
            ra_mid,
            dec_mid,
        }
    }

    fn default_cfg() -> ScoreConfig {
        ScoreConfig {
            noise: ModelNoise {
                q0: 1e-12,
                q1: 0.0,
                q2: 5e-14,
            },
            weights: ScoreWeights::default(),
            gates: ScoreGates::default(),
            scales: ScoreScales::default(),
        }
    }

    #[test]
    fn score_identity_is_low() {
        let ra0 = 1.0;
        let dec0 = 0.2;
        let i = mk_seed(1, ra0, dec0, [0.0, 0.0], [1e-3, 0.0], 1000.0, 1);
        let j = mk_seed(2, ra0, dec0, [0.0, 0.0], [1e-3, 0.0], 1000.0, 1);
        let cfg = default_cfg();

        let e = ScoredEdge::score(&i, &j, cfg, 1).expect("should pass gates");
        assert!(
            e.components.d2_pos < 1e-6,
            "position residual should be tiny"
        );
        assert!(e.cost >= 0.0);
        assert!(
            e.cost < 1e-3,
            "identity-like match should have near-zero cost"
        );
    }

    #[test]
    fn gate_direction_rejects_large_angle() {
        let ra0 = 1.0;
        let dec0 = 0.2;
        // vi along +x, vj along +y (90° apart)
        let i = mk_seed(1, ra0, dec0, [0.0, 0.0], [1e-3, 0.0], 1000.0, 1);
        let j = mk_seed(2, ra0, dec0, [0.0, 0.0], [0.0, 1e-3], 1000.0, 1);

        let mut cfg = default_cfg();
        cfg.gates.max_theta_vel = 5.0_f64.to_radians(); // 5° tolerance
        let e = ScoredEdge::score(&i, &j, cfg, 1);
        assert!(e.is_none(), "90° should be gated out");
    }

    #[test]
    fn gate_speed_diff_rejects_large_delta_speed() {
        let ra0 = 1.0;
        let dec0 = 0.2;
        let i = mk_seed(1, ra0, dec0, [0.0, 0.0], [1e-3, 0.0], 1000.0, 1);
        let j = mk_seed(2, ra0, dec0, [0.0, 0.0], [5e-3, 0.0], 1000.0, 1);

        let mut cfg = default_cfg();
        cfg.gates.max_speed_diff = 1e-3; // 0 tolerance beyond ~equal speed
        let e = ScoredEdge::score(&i, &j, cfg, 1);
        assert!(e.is_none(), "too different speeds should be gated out");
    }

    #[test]
    fn flux_penalty_increases_cost() {
        let ra0 = 1.0;
        let dec0 = 0.2;
        let i = mk_seed(1, ra0, dec0, [0.0, 0.0], [1e-3, 0.0], 1000.0, 1);
        let j_same = mk_seed(2, ra0, dec0, [0.0, 0.0], [1e-3, 0.0], 1000.0, 1);
        let j_diff = mk_seed(3, ra0, dec0, [0.0, 0.0], [1e-3, 0.0], 1500.0, 1);

        let mut cfg = default_cfg();
        cfg.weights.w_flux = 1.0;
        cfg.scales.flux_sigma_floor = 10.0;

        let e_same = ScoredEdge::score(&i, &j_same, cfg, 1).unwrap();
        let e_diff = ScoredEdge::score(&i, &j_diff, cfg, 1).unwrap();
        assert!(
            e_diff.cost > e_same.cost,
            "flux mismatch should raise the cost"
        );
    }

    #[test]
    fn gap_penalty_increases_cost() {
        let ra0 = 1.0;
        let dec0 = 0.2;
        let i = mk_seed(1, ra0, dec0, [0.0, 0.0], [5e-4, 2e-4], 1000.0, 1);
        let j = mk_seed(2, ra0, dec0, [0.0, 0.0], [5e-4, 2e-4], 1000.0, 1);

        let cfg = default_cfg();
        let e1 = ScoredEdge::score(&i, &j, cfg, 1).unwrap();
        let e3 = ScoredEdge::score(&i, &j, cfg, 3).unwrap();
        assert!(e3.cost > e1.cost, "larger revisit gap should increase cost");
    }

    /* ---------------------- Property-based tests ---------------------- */

    proptest! {

        /// Draw (dx,dy) strictly inside the 3σ ellipse implied by the current config.
        #[test]
        fn prop_small_perturbations_pass_adaptive(
            u   in 0.0f64..1.0,                   // radius factor
            ang in 0.0f64..(2.0 * PI),            // angle
            dvx in -2e-4f64..2e-4,
            dvy in -2e-4f64..2e-4,
            dflux in -100f32..100f32
        ) {
            let ra0 = 1.0;
            let dec0 = 0.2;

            // Seeds & config
            let i  = mk_seed(1, ra0, dec0, [0.0, 0.0], [1e-3, 0.0], 1000.0, 1);
            let j0 = mk_seed(2, ra0, dec0, [0.0, 0.0], [1e-3, 0.0], 1000.0, 1);
            let mut cfg = default_cfg();
            cfg.gates.max_theta_vel = 20.0_f64.to_radians();
            cfg.gates.max_speed_diff = 2e-3;

            // Combined plane variance S at dt=0 (same recipe as scorer)
            let (_p_hat, s_i) = i.predict_on_plane(j0.epoch_mid, cfg.noise);
            let sxx = (s_i[0][0] + j0.cov_pos[0][0]).max(0.0);
            let syy = (s_i[1][1] + j0.cov_pos[1][1]).max(0.0);

            // Use worst axis for a conservative circular bound to the ellipse.
            let sigma_worst = sxx.max(syy).sqrt();       // 1σ on worst axis
            let r_max = 0.95 * 3.0 * sigma_worst;        // strictly inside 3σ

            // Sample in the disc of radius r_max
            let r  = r_max * u;
            let dx = r * ang.cos();
            let dy = r * ang.sin();

            let j = mk_seed(2, ra0, dec0, [dx, dy], [1e-3 + dvx, dvy], 1000.0 + dflux, 1);

            let e = ScoredEdge::score(&i, &j, cfg, 1);
            prop_assert!(e.is_some(), "inside the 3σ (slightly shrunk) circle should pass");
            let e = e.unwrap();
            prop_assert!(e.components.d2_pos <= 9.0 + 1e-6);
        }

        // If the velocity direction mismatch is very large, a tight gate should reject.
        #[test]
        fn prop_large_angle_rejected(angle_deg in 60_u32..120_u32) {
            let angle = (angle_deg as f64).to_radians();
            let ra0 = 1.0;
            let dec0 = 0.2;

            let vi = [1e-3, 0.0];
            let vj = [ 1e-3 * angle.cos(), 1e-3 * angle.sin() ];

            let i = mk_seed(1, ra0, dec0, [0.0, 0.0], vi, 1000.0, 1);
            let j = mk_seed(2, ra0, dec0, [0.0, 0.0], vj, 1000.0, 1);

            let mut cfg = default_cfg();
            cfg.gates.max_theta_vel = 30.0_f64.to_radians();
            let e = ScoredEdge::score(&i, &j, cfg, 1);
            prop_assert!(e.is_none());
        }
    }
}
