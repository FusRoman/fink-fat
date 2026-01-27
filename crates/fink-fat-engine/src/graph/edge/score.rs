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
//! - **Hard gates** immediately **reject** an edge (return `None`).
//! - **Weights** control **relative influence** of accepted terms.
//!
//! ## Tuning strategy
//! - Start with conservative position gating (`max_d2 ≈ 9.0`, about 3σ in 2D),
//! - Enable velocity **direction** first (`theta0` a few degrees), then **speed**,
//! - Calibrate photometry with a realistic `flux_sigma_floor`,
//! - Set `rho` near 1 for linear penalties in missed revisits, increase to
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
        GapScore, NumericConfig, PhotometryScore, PositionScore, PredictConfig, ScoreConfig,
        VelocityScore,
    },
    seeding::{seed_id::SeedId, seed_node::SeedNode},
};

/// Decomposed score for a single directed edge `i → j`.
///
/// A `ScoredEdge` represents a **candidate linkage** from a source seed `i` to a
/// target seed `j` (directed edge) together with:
/// - a final **scalar cost** (`cost`) suitable for ranking or graph solvers,
/// - a structured set of **component-level diagnostics** (`components`) used for
///   interpretability, calibration, and debugging.
///
/// The edge is considered valid only if all **hard gates** (e.g. positional and
/// velocity consistency thresholds) were satisfied during scoring. When a gate
/// fails, edge creation is aborted (see [`ScoredEdge::score`]).
///
/// Fields
/// ------
/// - `from`, `to` encode the directed linkage `from → to`.
/// - `dt_days` is the signed time separation between the two seeds.
/// - `cost` is an additive scalar where **lower is better**.
/// - `components` stores the raw per-term diagnostics (pre-weights or
///   optionally absent).
#[derive(Clone, Debug)]
pub struct ScoredEdge {
    /// Source/target seed identifiers.
    pub from: SeedId,
    pub to: SeedId,
    /// Time gap (**days**): `t_j - t_i`.
    ///
    /// This is computed from the seeds’ mid-epochs and is used for:
    /// - predicting kinematics to a common epoch,
    /// - reporting the temporal separation of the candidate linkage.
    pub dt_days: f64,
    /// Additive total cost after gating and weighting (lower is better).
    ///
    /// This value is produced by combining [`ScoreComponents`] with the
    /// configured weights/normalizations (see `compose_cost`). It is intended
    /// to be consumed directly by assignment / flow / shortest-path solvers.
    pub cost: f64,
    /// Component-wise diagnostics useful for calibration and debugging.
    ///
    /// These diagnostics are the “explainable” part of the score and make it
    /// possible to:
    /// - plot distributions (same vs different asteroid),
    /// - tune thresholds and weights,
    /// - detect numerical pathologies (degenerate velocities, etc.).
    pub components: ScoreComponents,
}

/// Individual score components (diagnostics) used to build the final edge cost.
///
/// `ScoreComponents` stores the per-term quantities computed during scoring
/// *before* they are scaled by weights/normalizations in the final `cost`.
///
/// Some components may be absent (`None`) when they are undefined or intentionally
/// disabled:
/// - **degenerate kinematics** (‖v‖ ≈ 0) yields `None` velocity diagnostics,
/// - a disabled weight (e.g. `w_flux <= 0`) yields `None` photometry diagnostics,
/// - band mismatch disables photometry by construction.
///
/// Missing components are treated as “no contribution” by the cost composition
/// logic (i.e. they contribute `0` to the final sum).
#[derive(Clone, Debug, Default)]
pub struct ScoreComponents {
    /// Plane-position Mahalanobis distance **squared**.
    ///
    /// This term measures the positional consistency of `j` with the predicted
    /// position of `i` at `t_j`, expressed on the tangent plane of `i`.
    /// It is typically the primary geometric discriminator and is subject to a
    /// hard gate upstream (e.g. `max_d2`).
    pub d2_pos: f64,

    /// Velocity **direction** mismatch (**radians**).
    ///
    /// This is the angular difference `θ = arccos(cosang)` between the velocity
    /// vectors of `i` and `j` expressed on `i`’s tangent plane at `t_j`.
    ///
    /// `None` is used when the direction diagnostic is undefined or disabled:
    /// - either seed has near-zero velocity norm (degenerate case),
    /// - or the corresponding scoring weight is zero (diagnostic not needed).
    pub vel_angle_rad: Option<f64>,

    /// Velocity **speed** mismatch (**radians/day**).
    ///
    /// This is the absolute difference in speed norms:
    /// `Δv = ||v_i| − |v_j||`, where both velocities are expressed on `i`’s plane.
    ///
    /// `None` is used when the speed diagnostic is undefined or disabled:
    /// - either seed has near-zero velocity norm (degenerate case),
    /// - or the corresponding scoring weight is zero (diagnostic not needed).
    pub vel_speed_diff: Option<f64>,

    /// Photometry z-score: `|ΔF| / σ_pool`.
    ///
    /// This is a pooled-variance z-score comparing mean fluxes of the two seeds.
    /// It is a **soft consistency term** (no hard gate). If unavailable or
    /// unreliable (non-finite / zero variance), it is `None`.
    ///
    /// Note: photometry is typically disabled when there is a band mismatch.
    pub z_flux: Option<f64>,

    /// Temporal gap penalty `(Δ - 1)^rho` (0 if `Δ ≤ 1`).
    ///
    /// This term encodes a soft prior preferring short revisit separations.
    /// It depends only on the revisit index separation `Δ` (not on astrometry).
    pub gap_penalty: f64,

    /// True if seed band sets do not overlap (no common filter).
    ///
    /// When `true`, photometric comparison is generally not meaningful, so the
    /// photometry term is disabled (`z_flux = None`) and an explicit mismatch
    /// penalty may be applied in the final cost.
    pub band_mismatch: bool,
}

/* ----------------------------- Public API ----------------------------- */

impl ScoredEdge {
    /// Compute the **gated** score for the directed edge `i → j`.
    ///
    /// This is the main entry point for turning two [`SeedNode`] candidates
    /// into a scored, interpretable edge. It applies:
    /// - **hard gating** (rejecting incompatible candidates early),
    /// - **diagnostic extraction** (component-level quantities),
    /// - **final cost composition** (weighted additive scalar).
    ///
    /// Arguments
    /// ---------
    /// * `i` – Source seed (edge tail). Provides:
    ///   - the reference tangent plane used for projection,
    ///   - the kinematic state to be predicted to `t_j`,
    ///   - the source identifier stored in `from`.
    /// * `j` – Target seed (edge head). Provides:
    ///   - the target epoch `t_j` (mid-epoch),
    ///   - the observed position to be projected onto `i`’s plane,
    ///   - the target identifier stored in `to`.
    /// * `cfg` – Scoring configuration controlling both:
    ///   - **hard gates** (e.g. `position.max_d2`, `velocity.max_speed_diff`),
    ///   - **weights/normalizations** used to compose the final scalar `cost`,
    ///   - predictor and numeric-stability knobs (e.g. `predict.noise`,
    ///     `numeric.min_variance`).
    /// * `delta_revisit` – Integer separation in revisit index between the
    ///   two seeds (`Δ`). Used only for the temporal gap penalty:
    ///   - `Δ = 1` means consecutive revisits → no gap penalty,
    ///   - `Δ > 1` increases the penalty as `(Δ − 1)^rho`.
    ///
    /// Returns
    /// -------
    /// * `Some(ScoredEdge)` if all hard gates pass and all required quantities
    ///   remain finite.
    /// * `None` if any hard gate is violated or a non-finite value is produced.
    ///
    /// Method
    /// ------
    /// 1. **Position (gate)**:
    ///    - Predict `i` to `t_j` on the tangent plane of `i`,
    ///    - Project `j` onto the same plane,
    ///    - Combine covariances (diagonal) and compute Mahalanobis `d²_pos`,
    ///    - Reject if `d²_pos` is non-finite or exceeds `position.max_d2`.
    ///
    /// 2. **Time alignment**:
    ///    - Compute `dt_days = t_j − t_i` from mid-epochs.
    ///
    /// 3. **Velocity (gates)**:
    ///    - Predict / extrapolate `i`’s plane velocity to `t_j` (optional acceleration),
    ///    - Estimate `j`’s plane velocity via symmetric finite difference using
    ///      `velocity.vel_eps_days`,
    ///    - Apply speed gate (`velocity.max_speed_diff`).
    ///
    /// 4. **Band consistency**:
    ///    - Determine whether the two seeds share at least one band.
    ///    - If there is no overlap, flag `band_mismatch` and disable photometry.
    ///
    /// 5. **Photometry (soft)**:
    ///    - If bands overlap and enabled, compute the pooled-variance flux z-score
    ///      `z_flux`. No hard gate is applied; failures yield `None`.
    ///
    /// 6. **Temporal gap (soft)**:
    ///    - Compute the gap penalty from `delta_revisit` using `gap.rho`.
    ///
    /// 7. **Compose final cost**:
    ///    - Combine all components using configured weights and normalizations
    ///      into a single additive scalar `cost`.
    ///
    /// Units
    /// -----
    /// - Plane positions in **radians** (tangent-plane coordinates),
    /// - Speeds in **radians/day**,
    /// - Fluxes in **nJy** (or consistent linear flux unit),
    /// - Time in **days** (based on the seeds’ epochs; typically MJD TT).
    ///
    /// Notes
    /// -----
    /// - This function is directional (`i → j`), because prediction and
    ///   projection are performed on the tangent plane of `i` at `t_j`.
    /// - Optional diagnostics are encoded as `Option` to preserve interpretability:
    ///   missing components mean “not defined / not used”, not “zero”.
    pub fn score(
        i: &SeedNode,
        j: &SeedNode,
        cfg: &ScoreConfig,
        delta_revisit: u32,
    ) -> Option<Self> {
        // 1) Position term
        let d2_pos = compute_position_term(i, j, &cfg.predict, &cfg.position, &cfg.numeric)?;

        // 2) Time gap
        let dt_days = j.plane.epoch_mid - i.plane.epoch_mid;

        // 3) Velocity terms
        let (vel_angle_rad, vel_speed_diff) = compute_velocity_terms(i, j, dt_days, &cfg.velocity)?;

        // 5) Gap penalty
        let gap_penalty = compute_gap_penalty(delta_revisit, &cfg.gap);

        // 6) Band overlap / mismatch
        let band_mismatch = !i.photom.shares_any_band(&j.photom);

        // 4) Photometry (only if bands overlap)
        let z_flux = if band_mismatch {
            None
        } else {
            compute_flux_z(i, j, &cfg.photometry)
        };

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

/// Compute the positional consistency term `d²_pos` between two seeds on a
/// common tangent plane, with hard gating.
///
/// This function evaluates how well a candidate seed `j` matches the
/// *predicted position* of a source seed `i` at the epoch of `j`,
/// using a **Mahalanobis distance on the tangent plane of `i`**.
///
/// Conceptually, it answers the question:
/// *"If seed `i` corresponds to a real moving object, how compatible is
/// the observed position of seed `j` with the predicted position of `i`
/// at the same time?"*
///
/// The computation proceeds as follows:
///
/// 1. **Time alignment**
///    - The target epoch `t_j` is taken as the mid-epoch of seed `j`.
///
/// 2. **Prediction on the tangent plane**
///    - Seed `i` is propagated to epoch `t_j` on *its own tangent plane*
///      using the kinematic predictor and its associated process noise.
///    - This yields:
///        - `p̂ = (x̂, ŷ)`: the predicted 2D position on the plane,
///        - `Σ̂_i(t_j)`: the predicted positional covariance.
///
/// 3. **Projection of the candidate**
///    - The observed sky position `(RA, Dec)` of seed `j` is projected onto
///      the same tangent plane as `i`.
///    - A residual vector `Δp = (dx, dy)` is formed between the projected
///      position of `j` and the predicted position of `i`.
///
/// 4. **Covariance model**
///    - A **diagonal covariance matrix** is constructed:
///
///        S = Σ̂_i(t_j) + Σ_pos(j)
///
///      where:
///        - `Σ̂_i(t_j)` is the predicted covariance of `i`,
///        - `Σ_pos(j)` is the positional measurement covariance of `j`.
///
///    - Only the diagonal terms `(σ²_x, σ²_y)` are used, assuming no
///      cross-correlation on the plane.
///    - An optional numeric variance floor (`numeric.min_variance`) is applied
///      to each axis to ensure numerical stability and prevent division by zero.
///
/// 5. **Mahalanobis distance**
///    - The squared Mahalanobis distance is computed as:
///
///        d²_pos = dx² / σ²_x + dy² / σ²_y
///
/// 6. **Hard gating**
///    - If `d²_pos` is non-finite or exceeds the configured gate
///      `cfg.max_d2`, the candidate is rejected and `None` is returned.
///
/// This term is typically used as:
/// - a **hard gate** to prune implausible inter-night edges early,
/// - a **positional cost component** in additive scoring models.
///
/// Arguments
/// ---------
/// * `i` – Source [`SeedNode`] providing the reference tangent plane and
///   the kinematic model used for prediction.
/// * `j` – Candidate [`SeedNode`] whose observed position is tested
///   against the prediction of `i`.
/// * `predict` – Predictor configuration, including the additive
///   process-noise model used during propagation.
/// * `cfg` – Position scoring configuration containing the hard gate
///   `max_d2`.
/// * `numeric` – Numeric stability configuration (e.g. minimum variance
///   floor applied to covariance terms).
///
/// Return
/// ------
/// * `Some(d2_pos)` if the positional Mahalanobis distance is finite and
///   satisfies `d²_pos ≤ cfg.max_d2`.
/// * `None` if:
///   - the distance is non-finite (NaN or infinite),
///   - or the hard positional gate is violated.
///
/// Notes
/// -----
/// - All computations are performed **on the tangent plane of `i`**,
///   ensuring local linearity even for moderately large angular separations.
/// - The use of a diagonal covariance is a deliberate simplification that
///   favors speed and robustness over full covariance propagation.
/// - This function is purely geometric/kinematic and does **not** depend
///   on photometry, band information, or temporal gap penalties.
fn compute_position_term(
    i: &SeedNode,
    j: &SeedNode,
    predict: &PredictConfig,
    cfg: &PositionScore,
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

    if !d2_pos.is_finite() || d2_pos > cfg.max_d2 {
        return None;
    }
    Some(d2_pos)
}

/// Compute velocity-consistency diagnostics between two seeds on a common
/// tangent plane, with hard kinematic gating.
///
/// This function evaluates whether the *apparent motion* of a candidate
/// seed `j` is compatible with the predicted motion of a source seed `i`,
/// both expressed **on the tangent plane of `i` at the epoch of `j`**.
///
/// Two complementary velocity diagnostics are considered:
/// - **direction mismatch** (angular difference between velocity vectors),
/// - **speed mismatch** (absolute difference of velocity norms).
///
/// These diagnostics are primarily used as:
/// - **hard gates** to reject kinematically inconsistent inter-night edges,
/// - optional **scoring components** when the corresponding weights are non-zero.
///
/// The computation proceeds as follows:
///
/// 1. **Velocity of the source seed (`i`)**
///    - The velocity of `i` is evaluated on its tangent plane at `t_j`,
///      the mid-epoch of seed `j`.
///    - If an acceleration term is available (`acc_xy` from triplets),
///      a linear extrapolation is applied:
///
///        vᵢ(t_j) = vᵢ + aᵢ · Δt
///
///      Otherwise, the instantaneous velocity stored in the seed is used.
///
/// 2. **Velocity of the candidate seed (`j`)**
///    - The velocity of `j` is estimated numerically using a **symmetric
///      finite difference** in time:
///
///        vⱼ ≈ [p(t_j + ε) − p(t_j − ε)] / (2ε)
///
///      where:
///        - `ε = cfg.vel_eps_days`,
///        - positions are projected onto the tangent plane of `i`.
///
///    - This ensures that both velocity vectors are expressed in the
///      **same local coordinate system**.
///
/// 3. **Degenerate cases**
///    - If either velocity has zero norm, the kinematic information is
///      considered unusable.
///    - In this case, **no velocity gating is applied**, and the function
///      returns `Some((None, None))`.
///
/// 4. **Direction consistency**
///    - The cosine of the angle between the two velocity vectors is computed:
///
///        cos(θ) = (vᵢ · vⱼ) / (|vᵢ| |vⱼ|)
///
/// 5. **Speed consistency**
///    - The absolute difference in speed is computed:
///
///        Δv = ||vᵢ| − |vⱼ||
///
///    - A hard speed gate is enforced via `cfg.max_speed_diff`.
///
/// 6. **Diagnostic outputs**
///    - The angular mismatch `θ` (in radians) is returned **only if**
///      the corresponding weight `cfg.w_dir > 0`.
///    - The speed mismatch `Δv` is returned **only if**
///      the corresponding weight `cfg.w_norm > 0`.
///
///    This allows the caller to distinguish between:
///    - *gating-only usage* (weights set to zero),
///    - *gating + scoring usage*.
///
/// Arguments
/// ---------
/// * `i` – Source [`SeedNode`] providing the reference tangent plane and
///   predicted kinematics.
/// * `j` – Candidate [`SeedNode`] whose apparent motion is tested.
/// * `dt_days` – Time offset (in days) between the reference epoch of `i`
///   and the epoch `t_j`.
/// * `cfg` – Velocity scoring configuration, including:
///   - speed gate (`max_speed_diff`),
///   - finite-difference step (`vel_eps_days`),
///   - scoring weights (`w_dir`, `w_norm`).
///
/// Return
/// ------
/// * `None` if:
///   - a non-finite value is produced,
///   - the direction gate is violated,
///   - or the speed gate is violated.
/// * `Some((vel_angle_rad, vel_speed_diff))` otherwise, where:
///   - `vel_angle_rad` is `Some(θ)` in radians if `w_dir > 0`, else `None`,
///   - `vel_speed_diff` is `Some(Δv)` if `w_norm > 0`, else `None`.
///
/// Notes
/// -----
/// - All kinematic quantities are evaluated **on the tangent plane of `i`**,
///   ensuring consistent local geometry.
/// - The symmetric finite difference used for `j` provides a robust estimate
///   of apparent angular velocity from astrometric predictions.
/// - This function deliberately separates **gating logic** from **scoring
///   diagnostics**, allowing clean reuse in optimization and evaluation
///   pipelines.
fn compute_velocity_terms(
    i: &SeedNode,
    j: &SeedNode,
    dt_days: f64,
    cfg: &VelocityScore,
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
    let eps = cfg.vel_eps_days;
    if !eps.is_finite() || eps <= 0.0 {
        return None;
    }
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

    if !cosang.is_finite() {
        return None;
    }

    // Speed gate
    let dv = (norm_vi - norm_vj).abs();
    if !dv.is_finite() || dv > cfg.max_speed_diff {
        return None;
    }

    // Diagnostics only if weighted
    let vel_angle_rad = (cfg.w_dir > 0.0).then(|| cosang.acos());
    let vel_speed_diff = (cfg.w_norm > 0.0).then(|| dv);

    Some((vel_angle_rad, vel_speed_diff))
}

/// Compute the photometric consistency term `z_flux` between two seeds.
///
/// This function measures how compatible the mean fluxes of two seeds
/// `i` and `j` are, using a **pooled-variance z-score**:
///
///     z_flux = |F_j − F_i| / σ_pool
///
/// where `σ_pool` combines the photometric uncertainties of both seeds
/// and an optional variance floor.
///
/// Unlike positional or kinematic terms, **no hard gate is applied** here.
/// If the photometric information cannot be evaluated reliably, the
/// function simply returns `None` and the photometric term does not
/// contribute to the total score.
///
/// The computation proceeds as follows:
///
/// 1. **Weight check**
///    - If the photometric weight `cfg.w_flux ≤ 0`, the term is disabled
///      entirely and `None` is returned.
///
/// 2. **Flux difference**
///    - The difference in mean flux between the two seeds is computed:
///
///        ΔF = F_j − F_i
///
///      where each mean flux typically represents an average over the
///      detections composing the seed.
///
/// 3. **Pooled variance**
///    - The combined variance is modeled as:
///
///        σ²_pool = σ²_i + σ²_j + σ²_floor
///
///      where:
///        - `σ_i`, `σ_j` are the empirical flux standard deviations of
///          seeds `i` and `j`,
///        - `σ_floor` is a configurable variance floor that prevents
///          unrealistically small uncertainties.
///
/// 4. **Validity checks**
///    - If the pooled variance is non-finite or non-positive, the term
///      is discarded.
///
/// 5. **Z-score evaluation**
///    - The absolute z-score is computed and returned if finite.
///
/// Arguments
/// ---------
/// * `i` – Source [`SeedNode`] providing the reference mean flux and
///   photometric uncertainty.
/// * `j` – Candidate [`SeedNode`] providing the comparison mean flux and
///   photometric uncertainty.
/// * `cfg` – Photometry scoring configuration, including:
///   - photometric weight (`w_flux`),
///   - variance floor (`flux_sigma_floor`).
///
/// Return
/// ------
/// * `Some(z_flux)` if the pooled variance is valid and the resulting
///   z-score is finite.
/// * `None` if:
///   - the photometric weight is zero or negative,
///   - the pooled variance is non-finite or non-positive,
///   - or the z-score is non-finite.
///
/// Notes
/// -----
/// - This term acts as a **soft consistency penalty** rather than a hard
///   rejection criterion.
/// - Large values of `z_flux` indicate significant photometric
///   inconsistency between the two seeds.
/// - The variance floor is essential to prevent over-weighting nearly
///   identical fluxes with unrealistically small reported uncertainties.
/// - This function is agnostic to band information; band consistency, if
///   required, must be handled separately.
fn compute_flux_z(i: &SeedNode, j: &SeedNode, cfg: &PhotometryScore) -> Option<f64> {
    if cfg.w_flux <= 0.0 {
        return None;
    }

    let df = (j.photom.flux_mean as f64) - (i.photom.flux_mean as f64);
    let s_i = i.photom.flux_std as f64;
    let s_j = j.photom.flux_std as f64;

    let sigma_floor = cfg.flux_sigma_floor;
    let sigma_sq = s_i * s_i + s_j * s_j + sigma_floor * sigma_floor;

    if !sigma_sq.is_finite() || sigma_sq <= 0.0 {
        return None;
    }

    let z = df.abs() / sigma_sq.sqrt();
    z.is_finite().then_some(z)
}

/// Compute the temporal gap penalty between two seeds based on revisit separation.
///
/// This function applies a **soft penalty** that increases with the number of
/// revisits separating two linked seeds. It is designed to discourage edges
/// spanning large temporal gaps, while still allowing them when other
/// consistency terms (position, velocity, photometry) strongly agree.
///
/// The penalty is defined as:
///
///     gap_penalty = (Δ − 1)^ρ        for Δ > 1
///                   0               otherwise
///
/// where:
/// - `Δ` is the integer difference in revisit indices (`delta_revisit`),
/// - `ρ` (`cfg.rho`) controls how fast the penalty grows with increasing gap.
///
/// Interpretation:
/// - `Δ = 1` (consecutive revisits) → no penalty,
/// - `Δ > 1` → increasing penalty as the temporal gap widens.
///
/// Unlike positional or kinematic terms, **this function does not perform
/// any hard gating**. The returned value is always finite and non-negative,
/// and is meant to be combined additively with other score components.
///
/// Arguments
/// ---------
/// * `delta_revisit` – Difference in revisit indices between two seeds.
///   A value of `1` corresponds to consecutive revisits.
/// * `cfg` – Gap scoring configuration containing the exponent `rho`
///   controlling the strength of the penalty.
///
/// Return
/// ------
/// * A non-negative gap penalty:
///   - `0.0` if `delta_revisit ≤ 1`,
///   - `(delta_revisit − 1)^rho` otherwise.
///
/// Notes
/// -----
/// - This term encodes a **prior preference for short temporal gaps**
///   without strictly forbidding longer gaps.
/// - Setting `rho = 0` yields a constant penalty of `1` for all `Δ > 1`,
///   while larger values of `rho` increasingly suppress long-gap links.
/// - This penalty is purely temporal and independent of astrometry,
///   kinematics, or photometry.
/// - Typical usage is as a weighted additive cost in inter-night scoring
///   pipelines.
fn compute_gap_penalty(delta_revisit: u32, cfg: &GapScore) -> f64 {
    if delta_revisit > 1 {
        let delta = (delta_revisit as f64) - 1.0;
        delta.powf(cfg.rho)
    } else {
        0.0
    }
}

/// Compose the final scalar cost from individual score components.
///
/// This function aggregates all previously computed **per-component
/// diagnostics** (position, velocity, photometry, temporal gap, and band
/// consistency) into a **single scalar cost** using the weights and
/// normalizations defined in the [`ScoreConfig`].
///
/// The function is deliberately **pure and side-effect free**:
/// - it does **not** perform any gating,
/// - it assumes all hard decisions (rejections) have already been applied,
/// - it only combines existing components linearly.
///
/// The resulting cost is designed to be:
/// - **interpretable** (each term has a clear physical meaning),
/// - **additive** (compatible with shortest-path, min-cost flow, or
///   assignment solvers),
/// - **sparse-friendly** (components may be absent without special casing).
///
/// The aggregation follows this structure:
///
/// 1. **Position term**
///    - The squared Mahalanobis distance `d²_pos` is added directly:
///
///        w_pos · d²_pos
///
/// 2. **Velocity terms**
///    - If available, the **direction mismatch** contributes as:
///
///        w_dir · (θ / θ₀)
///
///      where `θ` is the angular difference (radians) and `θ₀` is a
///      normalization scale.
///    - If available, the **speed mismatch** contributes as:
///
///        w_norm · (Δv / v₀)
///
///      where `Δv` is the absolute speed difference and `v₀` is a
///      normalization scale.
///
/// 3. **Photometric term**
///    - If available, the photometric z-score contributes linearly:
///
///        w_flux · z_flux
///
/// 4. **Temporal gap penalty**
///    - The soft gap penalty is always included:
///
///        w_gap · gap_penalty
///
/// 5. **Band mismatch penalty**
///    - If the two seeds originate from different photometric bands,
///      a fixed penalty is added:
///
///        w_band_mismatch
///
/// Arguments
/// ---------
/// * `components` – Decomposed score components for a candidate edge,
///   as produced by the individual `compute_*` routines.
/// * `cfg` – Global scoring configuration defining:
///   - per-term weights,
///   - normalization scales for velocity diagnostics.
///
/// Return
/// ------
/// * The final scalar cost (non-negative), suitable for direct use in
///   graph-based solvers or ranking.
///
/// Notes
/// -----
/// - Components wrapped in `Option` are included **only if present**,
///   allowing upstream logic to disable specific terms via weights or
///   degeneracy handling.
/// - Normalization constants (`theta0`, `v0`) ensure that heterogeneous
///   physical quantities contribute on comparable scales.
/// - Lower costs correspond to **higher physical compatibility** between
///   the two seeds.
/// - This function makes no assumptions about optimality; tuning of the
///   weights and normalizations is expected to be performed offline.
fn compose_cost(components: &ScoreComponents, cfg: &ScoreConfig) -> f64 {
    let mut cost = 0.0;

    // Position
    cost += cfg.position.w_pos * components.d2_pos;

    // Velocity
    if let Some(theta) = components.vel_angle_rad {
        cost += cfg.velocity.w_dir * (theta / cfg.velocity.theta0);
    }
    if let Some(dv) = components.vel_speed_diff {
        cost += cfg.velocity.w_norm * (dv / cfg.velocity.v0);
    }

    // Photometry
    if let Some(z) = components.z_flux {
        cost += cfg.photometry.w_flux * z;
    }

    // Gap
    cost += cfg.gap.w_gap * components.gap_penalty;

    // Band mismatch
    if components.band_mismatch {
        cost += cfg.band.w_band_mismatch;
    }

    cost
}
