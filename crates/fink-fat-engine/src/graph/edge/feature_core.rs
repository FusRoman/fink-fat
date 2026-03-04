//! Feature core (shared intermediates)
//!
//! This module defines `FeatureCore`, an internal helper used to compute "core"
//! intermediate quantities once per edge (`from`, `to`) and reuse them across
//! multiple public feature families (position, velocity, photometry, etc.).
//!
//! Why this exists
//! ---------------
//! Edge feature computation often needs the same expensive building blocks:
//! - propagation from `from` epoch to `to` epoch on the tangent plane,
//! - tangent-plane projection,
//! - innovation vector r and innovation covariance S,
//! - robust normalization / whitening,
//! - kinematic consistency checks in velocity space.
//!
//! `FeatureCore` consolidates these computations to:
//! - avoid duplicated work,
//! - keep the public feature mapping code short (`EdgeFeatures { ... }`),
//! - centralize numerical guards (floors, eps, NaN handling).
//!
//! Numerical robustness philosophy
//! ------------------------------
//! This code is used in ML pipelines at LSST scale. We prioritize:
//! - determinism and stability (avoid NaNs/Infs in datasets),
//! - fast, branch-light computations,
//! - well-defined fallbacks for degenerate cases (singular covariances, dt<=0).
//!
//! In practice we:
//! - floor covariance diagonals (`FLOOR`) to avoid singularities,
//! - clamp dot/angle derived metrics (e.g., cos in [-1, 1]),
//! - map non-finite results to 0.0 (`finite_or_zero`).
//!
//! Units & frames
//! --------------
//! - Positions and velocities are expressed on the tangent plane of `from`
//!   (a small-angle local linearization).
//! - Coordinates returned by `radec_to_tangent_precomp` are in radians on that plane.
//! - `dt` is in days.
//! - Covariances are 2×2 matrices consistent with the tangent-plane coordinate system.
//!
//! -----------------------------------------------------------------------------

use crate::{
    astro_math::{
        cholesky_lower_sym_2x2, clamp_unit, dot2, invert_sym_2x2, l2_norm, mat_vec2, safe_ln,
    },
    seeding::SeedNode,
};

/// Shared intermediate computations reused across feature sub-sets.
///
/// This internal struct:
/// - computes once the expensive/central quantities (innovation $\mathbf{r}$,
///   covariance $\mathbf{S}$, predicted velocity direction, etc.),
/// - stores the final scalar values reused in multiple feature families,
/// - keeps high-level feature mapping (`EdgeFeatures { ... }`) short and readable.
///
/// Stability contract
/// ------------------
/// All stored scalars are intended to be **finite** in normal operation. When the
/// underlying computations encounter invalid inputs (NaNs/Infs), singular matrices,
/// or degenerate kinematics (e.g., $|\mathbf{v}| \approx 0$), we fall back to
/// stable defaults and/or sanitize outputs via [`FeatureCore::finite_or_zero`].
///
/// Implementation note
/// -------------------
/// The public structs are meant to be stable API for ML export.
/// `FeatureCore` is intentionally `pub(crate)` so we can refactor internals
/// without breaking external code.
///
/// Attributes
/// ----------
/// **Position family**
///
/// * `chi2_pos` – Mahalanobis innovation distance:
///   $\chi^2\_{\mathrm{pos}} = \mathbf{r}^\top \mathbf{S}^{-1} \mathbf{r}$.
/// * `log_chi2_pos` – $\ln(\chi^2\_{\mathrm{pos}} + \varepsilon)$ for dynamic
///   range compression.
/// * `z_dx` – Diagonal z-score for $x$ residual:
///   $z\_{\Delta x} = r\_x \,/\, \sqrt{S\_{xx}}$.
/// * `z_dy` – Diagonal z-score for $y$ residual:
///   $z\_{\Delta y} = r\_y \,/\, \sqrt{S\_{yy}}$.
/// * `z_resid_norm` – $\sqrt{z\_{\Delta x}^2 + z\_{\Delta y}^2}$ (magnitude
///   proxy in sigma units).
/// * `z_along` – Along-track z-score (projected on predicted velocity direction).
/// * `z_cross` – Cross-track z-score (perpendicular to predicted direction).
///
/// **Whitening family (Cholesky)**
///
/// Given the Cholesky factorization $\mathbf{S} = \mathbf{L}\,\mathbf{L}^\top$
/// and solving $\mathbf{L}\,\mathbf{z} = \mathbf{r}$:
///
/// * `chol_z1` – First whitened component $z\_1$.
/// * `chol_z2` – Second whitened component $z\_2$.
/// * `chol_z_norm` – $\|\mathbf{z}\| = \sqrt{z\_1^2 + z\_2^2} \approx \sqrt{\chi^2\_{\mathrm{pos}}}$
///   when $\mathbf{S}$ is SPD.
///
/// **Velocity family**
///
/// * `cos_dtheta_v` – $\cos \Delta\theta\_v = \hat{\mathbf{v}}\_{\mathrm{pred}} \cdot \hat{\mathbf{v}}\_{\mathrm{to}}$.
/// * `rel_speed_diff` – Scale-free speed mismatch:
///   $\frac{| \|\mathbf{v}\_{\mathrm{to}}\| - \|\mathbf{v}\_{\mathrm{pred}}\| |}{\|\mathbf{v}\_{\mathrm{to}}\| + \|\mathbf{v}\_{\mathrm{pred}}\| + \varepsilon}$.
/// * `innov_speed_ratio` – $\frac{\|\mathbf{r}\| / \Delta t}{\|\mathbf{v}\_{\mathrm{pred}}\| + \varepsilon}$.
/// * `chi2_vel` – Velocity-space Mahalanobis distance:
///   $\chi^2\_{\mathrm{vel}} = \delta\mathbf{v}^\top \mathbf{S}\_{\mathrm{vel}}^{-1} \delta\mathbf{v}$.
/// * `log_chi2_vel` – $\ln(\chi^2\_{\mathrm{vel}} + \varepsilon)$.
pub(crate) struct FeatureCore {
    // ----------------------------- Position features -----------------------------
    /// Mahalanobis innovation distance:
    /// $\chi^2\_{\mathrm{pos}} = \mathbf{r}^\top \mathbf{S}^{-1} \mathbf{r}$.
    pub(crate) chi2_pos: f64,
    /// Log-compressed position $\chi^2$:
    /// $\ln(\chi^2\_{\mathrm{pos}} + \varepsilon)$.
    pub(crate) log_chi2_pos: f64,
    /// Diagonal z-score for $x$ residual:
    /// $z\_{\Delta x} = r\_x / \sqrt{S\_{xx}}$.
    pub(crate) z_dx: f64,
    /// Diagonal z-score for $y$ residual:
    /// $z\_{\Delta y} = r\_y / \sqrt{S\_{yy}}$.
    pub(crate) z_dy: f64,
    /// Norm of diagonal z-scores:
    /// $\sqrt{z\_{\Delta x}^2 + z\_{\Delta y}^2}$.
    pub(crate) z_resid_norm: f64,
    /// Along-track z-score:
    /// $z\_\parallel = (\mathbf{r} \cdot \hat{\mathbf{u}}) / \sqrt{\hat{\mathbf{u}}^\top \mathbf{S} \hat{\mathbf{u}}}$.
    pub(crate) z_along: f64,
    /// Cross-track z-score:
    /// $z\_\perp = (\mathbf{r} \cdot \hat{\mathbf{n}}) / \sqrt{\hat{\mathbf{n}}^\top \mathbf{S} \hat{\mathbf{n}}}$.
    pub(crate) z_cross: f64,

    // ----------------------- Whitening (Cholesky) features -----------------------
    /// First whitened residual component $z\_1 = r\_x / L\_{00}$,
    /// from solving $\mathbf{L}\,\mathbf{z} = \mathbf{r}$.
    pub(crate) chol_z1: f64,
    /// Second whitened residual component
    /// $z\_2 = (r\_y - L\_{10}\,z\_1) / L\_{11}$.
    pub(crate) chol_z2: f64,
    /// Euclidean norm of the whitened residuals:
    /// $\|\mathbf{z}\| = \sqrt{z\_1^2 + z\_2^2} \approx \sqrt{\chi^2\_{\mathrm{pos}}}$.
    pub(crate) chol_z_norm: f64,

    // ----------------------------- Velocity features -----------------------------
    /// Cosine of the angle between predicted and target velocities:
    /// $\cos \Delta\theta\_v = \hat{\mathbf{v}}\_{\mathrm{pred}} \cdot \hat{\mathbf{v}}\_{\mathrm{to}}$.
    pub(crate) cos_dtheta_v: f64,
    /// Relative speed difference:
    /// $\frac{|\,\|\mathbf{v}\_{\mathrm{to}}\| - \|\mathbf{v}\_{\mathrm{pred}}\|\,|}{\|\mathbf{v}\_{\mathrm{to}}\| + \|\mathbf{v}\_{\mathrm{pred}}\| + \varepsilon}$.
    pub(crate) rel_speed_diff: f64,
    /// Innovation-induced speed ratio:
    /// $\frac{\|\mathbf{r}\| / \Delta t}{\|\mathbf{v}\_{\mathrm{pred}}\| + \varepsilon}$.
    pub(crate) innov_speed_ratio: f64,

    /// Velocity innovation Mahalanobis distance:
    /// $\chi^2\_{\mathrm{vel}} = \delta\mathbf{v}^\top \mathbf{S}\_{\mathrm{vel}}^{-1} \delta\mathbf{v}$,
    /// where $\delta\mathbf{v} = \mathbf{v}\_{\mathrm{to}} - \mathbf{v}\_{\mathrm{pred}}$.
    pub(crate) chi2_vel: f64,
    /// Log-compressed velocity $\chi^2$:
    /// $\ln(\chi^2\_{\mathrm{vel}} + \varepsilon)$.
    pub(crate) log_chi2_vel: f64,
    // --- Cached intermediates reused by `compute_cost` -----------------------
    /// Time gap $\Delta t$ from `from` to `to` (days).
    pub(crate) dt: f64,
    /// Precomputed $\Delta t^2$ (days²).
    pub(crate) dt_sq: f64,
    /// Position innovation $\mathbf{r} = \mathbf{p}\_{{\mathrm{to}}} - \mathbf{p}\_{{\mathrm{pred}}}$
    /// on the `from` tangent plane (radians).
    pub(crate) r_pos: [f64; 2],
    /// Velocity innovation $\delta\mathbf{v} = \mathbf{v}\_{{\mathrm{to}}} - \mathbf{v}\_{{\mathrm{pred}}}$
    /// (rad/day).
    pub(crate) dv: [f64; 2],
}

impl FeatureCore {
    /// Numerical floor used across feature computations.
    ///
    /// This constant is applied to:
    /// - stabilize covariance diagonals (avoid singular matrices),
    /// - guard denominators in normalized quantities (`sqrt(S_xx)`),
    /// - make 2×2 inversions robust via [`invert_sym_2x2`],
    /// - prevent whitening from failing due to tiny/negative eigenvalues.
    ///
    /// Design choice
    /// -------------
    /// This is intentionally *very small* to avoid biasing well-behaved cases,
    /// but large enough to avoid division-by-zero and catastrophic numeric blowups.
    pub(crate) const FLOOR: f64 = 1e-20;

    /// Generic epsilon used in ratios and small denominators.
    ///
    /// This is used when a denominator could be very small but non-zero, to
    /// avoid huge spikes in feature values.
    pub(crate) const EPS: f64 = 1e-16;

    /// Build all shared intermediates from a directed edge `(from -> to)`.
    ///
    /// Overview
    /// --------
    /// 1. Compute the time separation $\Delta t$ between seeds (days).
    /// 2. Propagate the `from` seed on its tangent plane to the epoch of `to`.
    /// 3. Project the `to` seed position onto the `from` tangent plane.
    /// 4. Build the innovation vector $\mathbf{r} = \mathbf{p}\_{\mathrm{to}} - \mathbf{p}\_{\mathrm{pred}}$.
    /// 5. Build the innovation covariance $\mathbf{S}$ in position space.
    /// 6. Compute normalized innovation metrics (Mahalanobis $\chi^2$,
    ///    z-scores, Cholesky whitening).
    /// 7. Compute kinematic consistency metrics (velocity angle/speed mismatch).
    /// 8. Compute velocity-space Mahalanobis metrics using $\mathbf{S}\_{\mathrm{vel}}$.
    ///
    /// Arguments
    /// ---------
    /// * `from` – Source seed node (older epoch).
    /// * `to` – Target seed node (newer epoch).
    ///
    /// Return
    /// ------
    /// A fully-populated [`FeatureCore`] containing scalar intermediates reused
    /// by the public feature structs.
    ///
    /// Notes
    /// -----
    /// - This function is meant to be called **once per edge**.
    /// - All outputs are sanitized to be finite where possible.
    /// - When $\Delta t$ is invalid (non-finite or $\leq 0$), metrics that
    ///   depend on $\Delta t$ fall back to $0$.
    ///
    /// Performance
    /// -----------
    /// Most operations are small fixed-size linear algebra ($2 \times 2$),
    /// optimized for hot loops over many candidate edges.
    #[inline]
    pub(crate) fn from_nodes(from: &SeedNode, to: &SeedNode) -> Self {
        // ---------------------------------------------------------------------
        // 1) Time separation
        // ---------------------------------------------------------------------
        // Time separation in days (TT). Used for propagation and covariance growth.
        let dt = to.delta_days(from);
        let dt_sq = dt * dt;

        // Whether dt is usable as a positive finite number.
        let dt_ok = dt.is_finite() && dt > 0.0;

        // ---------------------------------------------------------------------
        // 2) Propagate `from` to the epoch of `to`
        // ---------------------------------------------------------------------
        // Propagate `from` seed state to the epoch of `to` (on the `from` tangent plane).
        //
        // Returns:
        // - `p_pred`: predicted position (x,y) on tangent plane at `to` epoch
        // - `v_pred`: predicted velocity (vx,vy) on same plane/epoch
        // - `_`: optional extra outputs (ignored here)
        let (p_pred, v_pred, _) = from.propagate_from(dt, dt_sq);

        // ---------------------------------------------------------------------
        // 3) Project `to` onto `from` tangent plane
        // ---------------------------------------------------------------------
        // Project the target seed position onto the tangent plane of `from`.
        let p_to = Self::project_to_on_from(from, to);

        // ---------------------------------------------------------------------
        // 4) Innovation vector r = observed - predicted
        // ---------------------------------------------------------------------
        // Innovation / residual on the tangent plane: r = observed - predicted.
        let r = [p_to[0] - p_pred[0], p_to[1] - p_pred[1]];

        // ---------------------------------------------------------------------
        // 5) Innovation covariance S
        // ---------------------------------------------------------------------
        // Innovation covariance S: accounts for prediction uncertainty and target uncertainty.
        let s = Self::innovation_cov(from, to, dt_sq);

        // Robust inverse of S (with flooring and fallback).
        // We assume `invert_sym_2x2` returns a usable matrix even if S is near-singular.
        let s_inv = invert_sym_2x2(s, Self::FLOOR);

        // ---------------------------------------------------------------------
        // 6) Position-space metrics (Mahalanobis, z-scores, whitening)
        // ---------------------------------------------------------------------
        // Mahalanobis distance: chi2_pos = rᵀ S⁻¹ r.
        // `max(0.0)` avoids negative values caused by floating-point noise.
        let chi2_pos = Self::finite_or_zero(dot2(r, mat_vec2(s_inv, r)).max(0.0));

        // Log-transform (improves dynamic range and ML behavior).
        // We add a tiny constant to avoid ln(0).
        let log_chi2_pos = safe_ln(chi2_pos + 1e-16);

        // Diagonal z-scores: cheap, robust proxies that ignore correlation.
        let (z_dx, z_dy, z_resid_norm) = Self::z_diag(r, s);

        // Along/cross decomposition using predicted velocity direction.
        let (z_along, z_cross, v_norm) = Self::z_along_cross(r, s, v_pred);

        // Whitened innovation via Cholesky decomposition:
        // - factorize S into L·Lᵀ,
        // - solve L·z = r for z.
        //
        // If S is not SPD (or factorization fails), whitened features fall back to 0.
        let (chol_z1, chol_z2, chol_z_norm) = match cholesky_lower_sym_2x2(s, Self::FLOOR) {
            Some(l) => {
                // Extract lower triangular elements.
                let l00 = l[0][0];
                let l10 = l[1][0];
                let l11 = l[1][1];

                // Guard against invalid / degenerate factors (should be > 0 for SPD).
                if l00.is_finite() && l00 > 0.0 && l11.is_finite() && l11 > 0.0 {
                    // Forward substitution for 2×2 lower-triangular system:
                    // z1 = r0 / l00
                    // z2 = (r1 - l10*z1) / l11
                    let z1 = r[0] / l00;
                    let z2 = (r[1] - l10 * z1) / l11;
                    let norm = l2_norm(z1, z2);
                    (
                        Self::finite_or_zero(z1),
                        Self::finite_or_zero(z2),
                        Self::finite_or_zero(norm),
                    )
                } else {
                    (0.0, 0.0, 0.0)
                }
            }
            None => (0.0, 0.0, 0.0),
        };

        // ---------------------------------------------------------------------
        // 7) Velocity consistency metrics
        // ---------------------------------------------------------------------
        // Velocity estimated at `to` (already in the same tangent-plane frame).
        let v_to = [to.plane.vel_xy[0], to.plane.vel_xy[1]];
        let v_to_norm = l2_norm(v_to[0], v_to[1]);

        // Directional agreement between predicted and target velocities.
        let cos_dtheta_v = Self::cos_between(v_pred, v_norm, v_to, v_to_norm);

        // Relative speed mismatch (scale-free).
        let rel_speed_diff = Self::rel_speed_diff(v_norm, v_to_norm);

        // Innovation-induced speed ratio: (|r|/dt) / |v_pred|.
        //
        // Intuition:
        // - |r|/dt is the speed "suggested" by the observed discrepancy,
        // - dividing by |v_pred| measures how large that discrepancy is relative
        //   to predicted motion.
        let r_norm = l2_norm(r[0], r[1]);
        let innov_speed_ratio = if dt_ok && v_norm.is_finite() && v_norm > 0.0 {
            (r_norm / dt) / (v_norm + Self::EPS)
        } else {
            0.0
        };

        // ---------------------------------------------------------------------
        // 8) Velocity-space Mahalanobis metrics
        // ---------------------------------------------------------------------
        // Velocity innovation: dv = v_to - v_pred (same epoch as `to`).
        // Note: this assumes both velocities are expressed in (approximately) the same tangent frame.
        let dv = [v_to[0] - v_pred[0], v_to[1] - v_pred[1]];

        // Innovation covariance in velocity space.
        let s_vel = Self::innovation_cov_vel(from, to);
        let s_vel_inv = invert_sym_2x2(s_vel, Self::FLOOR);

        // Mahalanobis distance in velocity space.
        let chi2_vel = Self::finite_or_zero(dot2(dv, mat_vec2(s_vel_inv, dv)).max(0.0));
        let log_chi2_vel = safe_ln(chi2_vel + 1e-16);

        // ---------------------------------------------------------------------
        // Pack outputs (sanitizing where appropriate)
        // ---------------------------------------------------------------------
        Self {
            chi2_pos,
            log_chi2_pos: Self::finite_or_zero(log_chi2_pos),

            z_dx,
            z_dy,
            z_resid_norm,
            z_along,
            z_cross,

            chol_z1,
            chol_z2,
            chol_z_norm,

            cos_dtheta_v,
            rel_speed_diff,
            innov_speed_ratio: Self::finite_or_zero(innov_speed_ratio),

            chi2_vel,
            log_chi2_vel: Self::finite_or_zero(log_chi2_vel),

            // Cached for reuse in compute_cost (avoids re-propagation/re-projection).
            dt,
            dt_sq,
            r_pos: r,
            dv,
        }
    }

    // -------------------------------------------------------------------------
    // Small inline building blocks
    // -------------------------------------------------------------------------

    /// Project the target seed position onto the source tangent plane.
    ///
    /// This uses the optimized precomputed tangent-plane transform stored in
    /// the `from` seed (`radec_to_tangent_precomp`), which avoids rebuilding
    /// rotation/projection matrices for every candidate edge.
    ///
    /// Arguments
    /// ---------
    /// * `from` – Source seed providing the tangent plane definition.
    /// * `to` – Target seed providing the sky position (RA, Dec).
    ///
    /// Return
    /// ------
    /// Target position `[x, y]` on the tangent plane of `from` (radians).
    ///
    /// Notes
    /// -----
    /// This is a purely geometric operation; no uncertainties are used here.
    #[inline]
    pub(crate) fn project_to_on_from(from: &SeedNode, to: &SeedNode) -> [f64; 2] {
        // Project target RA/Dec onto the precomputed tangent plane of `from`.
        from.plane
            .radec_to_tangent_precomp(to.plane.ra_mid, to.plane.dec_mid)
    }

    /// Build the innovation covariance matrix $\mathbf{S}$ in position space.
    ///
    /// Definition
    /// ----------
    /// We use a simple constant-velocity propagation of uncertainty:
    ///
    /// $$\mathbf{C}\_{\mathrm{pred}} = \mathbf{C}^{\mathrm{pos}}\_{\mathrm{from}} + \Delta t^{2} \mathbf{C}^{\mathrm{vel}}\_{\mathrm{from}}$$
    ///
    /// $$\mathbf{S} = \mathbf{C}\_{\mathrm{pred}} + \mathbf{C}^{\mathrm{pos}}\_{\mathrm{to}} + \varepsilon\_f \mathbf{I}$$
    ///
    /// where $\varepsilon\_f$ is a small diagonal floor.
    ///
    /// Interpretation
    /// --------------
    /// - $\mathbf{C}^{\mathrm{pos}}\_{\mathrm{from}}$: position uncertainty at the `from` epoch.
    /// - $\mathbf{C}^{\mathrm{vel}}\_{\mathrm{from}}$: velocity uncertainty at the `from` epoch.
    /// - Multiplying by $\Delta t^{2}$ approximates how velocity uncertainty
    ///   grows into position uncertainty over the time gap $\Delta t$.
    /// - Adding $\mathbf{C}^{\mathrm{pos}}\_{\mathrm{to}}$ accounts for measurement
    ///   uncertainty at the `to` epoch.
    ///
    /// Arguments
    /// ---------
    /// * `from` – Source seed node providing position/velocity covariances.
    /// * `to` – Target seed node providing position covariance at its epoch.
    /// * `dt_sq` – Precomputed $\Delta t^{2}$ (days²).
    ///
    /// Return
    /// ------
    /// Innovation covariance matrix $\mathbf{S}$ ($2 \times 2$).
    ///
    /// Notes
    /// -----
    /// A small diagonal floor $\varepsilon\_f \mathbf{I}$ is added to prevent:
    /// - singular matrices,
    /// - $\sqrt{0}$ in z-score computations,
    /// - unstable inversion in Mahalanobis distance,
    /// - Cholesky failures due to borderline numerical PSD-ness.
    #[inline]
    pub(crate) fn innovation_cov(from: &SeedNode, to: &SeedNode, dt_sq: f64) -> [[f64; 2]; 2] {
        // Extract covariances in tangent-plane coordinates (x,y).
        let cpos_from = from.plane.cov_pos;
        let cvel_from = from.plane.cov_vel;
        let cpos_to = to.plane.cov_pos;

        // Prediction covariance: position uncertainty + (dt^2)*velocity uncertainty.
        //
        // This is the usual "constant-velocity" uncertainty growth model on a plane.
        let cov_pred = [
            [
                cpos_from[0][0] + dt_sq * cvel_from[0][0],
                cpos_from[0][1] + dt_sq * cvel_from[0][1],
            ],
            [
                cpos_from[1][0] + dt_sq * cvel_from[1][0],
                cpos_from[1][1] + dt_sq * cvel_from[1][1],
            ],
        ];

        // Innovation covariance: predicted covariance + target position covariance + floor*I.
        [
            [
                cov_pred[0][0] + cpos_to[0][0] + Self::FLOOR,
                cov_pred[0][1] + cpos_to[0][1],
            ],
            [
                cov_pred[1][0] + cpos_to[1][0],
                cov_pred[1][1] + cpos_to[1][1] + Self::FLOOR,
            ],
        ]
    }

    /// Build the innovation covariance matrix $\mathbf{S}\_{\mathrm{vel}}$ in velocity space.
    ///
    /// Definition
    /// ----------
    /// $$\mathbf{S}\_{\mathrm{vel}} = \mathbf{C}^{\mathrm{vel}}\_{\mathrm{from}} + \mathbf{C}^{\mathrm{vel}}\_{\mathrm{to}} + \varepsilon\_f \mathbf{I}$$
    ///
    /// Arguments
    /// ---------
    /// * `from` – Source seed providing velocity covariance.
    /// * `to` – Target seed providing velocity covariance.
    ///
    /// Return
    /// ------
    /// Innovation covariance matrix in velocity space $\mathbf{S}\_{\mathrm{vel}}$
    /// ($2 \times 2$).
    ///
    /// Notes
    /// -----
    /// This is symmetric up to floating error and is floored on the diagonal
    /// for stability.
    #[inline]
    fn innovation_cov_vel(from: &SeedNode, to: &SeedNode) -> [[f64; 2]; 2] {
        let c1 = from.plane.cov_vel;
        let c2 = to.plane.cov_vel;

        [
            [c1[0][0] + c2[0][0] + Self::FLOOR, c1[0][1] + c2[0][1]],
            [c1[1][0] + c2[1][0], c1[1][1] + c2[1][1] + Self::FLOOR],
        ]
    }

    /// Build the positional innovation covariance with optional CWNA (Continuous White Noise Acceleration) process noise.
    ///
    /// Extends [`innovation_cov`] by adding the Singer/CWNA diagonal term:
    ///
    /// $$\mathbf{S}\_\text{pos} \mathrel{+}= \sigma_q^2 \cdot \frac{\Delta t^3}{3} \cdot \mathbf{I}$$
    ///
    /// When `sigma_q == 0.0` the result is identical to [`innovation_cov`].
    ///
    /// Arguments
    /// ---------
    /// * `from` – Source seed.
    /// * `to` – Target seed.
    /// * `dt` – Time gap $\Delta t$ (days).
    /// * `dt_sq` – Precomputed $\Delta t^2$ (days²).
    /// * `sigma_q` – CWNA spectral density (rad · day^(−3/2)), set `0.0` to disable.
    ///
    /// Return
    /// ------
    /// Positional innovation covariance $\mathbf{S}$ ($2 \times 2$).
    #[inline]
    pub(crate) fn innovation_cov_cwna(
        from: &SeedNode,
        to: &SeedNode,
        dt: f64,
        dt_sq: f64,
        sigma_q: f64,
    ) -> [[f64; 2]; 2] {
        let mut s = Self::innovation_cov(from, to, dt_sq);
        if sigma_q != 0.0 {
            // Q_pos = σ_q² · dt³/3  (scalar, same for x and y)
            let q = sigma_q * sigma_q * dt * dt_sq / 3.0;
            s[0][0] += q;
            s[1][1] += q;
        }
        s
    }

    /// Build the velocity innovation covariance with optional CWNA process noise.
    ///
    /// Extends [`innovation_cov_vel`] by adding the Singer/CWNA diagonal term:
    ///
    /// $$\mathbf{S}\_\text{vel} \mathrel{+}= \sigma_q^2 \cdot \Delta t \cdot \mathbf{I}$$
    ///
    /// When `sigma_q == 0.0` the result is identical to [`innovation_cov_vel`].
    ///
    /// Arguments
    /// ---------
    /// * `from` – Source seed.
    /// * `to` – Target seed.
    /// * `dt` – Time gap $\Delta t$ (days).
    /// * `sigma_q` – CWNA spectral density (rad · day^(−3/2)), set `0.0` to disable.
    ///
    /// Return
    /// ------
    /// Velocity innovation covariance $\mathbf{S}\_{\mathrm{vel}}$ ($2 \times 2$).
    #[inline]
    pub(crate) fn innovation_cov_vel_cwna(
        from: &SeedNode,
        to: &SeedNode,
        dt: f64,
        sigma_q: f64,
    ) -> [[f64; 2]; 2] {
        let mut s = Self::innovation_cov_vel(from, to);
        if sigma_q != 0.0 {
            // Q_vel = σ_q² · dt  (scalar, same for vx and vy)
            let q = sigma_q * sigma_q * dt;
            s[0][0] += q;
            s[1][1] += q;
        }
        s
    }

    /// Compute cheap diagonal-based z-scores for the innovation $\mathbf{r}$.
    ///
    /// Definition
    /// ----------
    /// We use only the diagonal terms of $\mathbf{S}$:
    ///
    /// $$z\_{\Delta x} = \frac{r\_x}{\sqrt{S\_{xx}}} ,\quad z\_{\Delta y} = \frac{r\_y}{\sqrt{S\_{yy}}} ,\quad z\_{\mathrm{norm}} = \sqrt{z\_{\Delta x}^2 + z\_{\Delta y}^2}$$
    ///
    /// Arguments
    /// ---------
    /// * `r` – Innovation vector $[r\_x,\, r\_y]$ (tangent-plane radians).
    /// * `s` – Innovation covariance matrix $\mathbf{S}$ ($2 \times 2$).
    ///
    /// Return
    /// ------
    /// $(z\_{\Delta x},\; z\_{\Delta y},\; z\_{\mathrm{norm}})$ – diagonal z-scores
    /// and their Euclidean norm.
    ///
    /// Notes
    /// -----
    /// This approximation:
    /// - ignores correlation between $x$ and $y$,
    /// - is robust and cheap,
    /// - is useful as an ML feature even when whitening fails.
    #[inline]
    pub(crate) fn z_diag(r: [f64; 2], s: [[f64; 2]; 2]) -> (f64, f64, f64) {
        // Guard diagonal terms to avoid sqrt(0) or sqrt(negative).
        let s_xx = s[0][0].max(Self::FLOOR);
        let s_yy = s[1][1].max(Self::FLOOR);

        // Normalize each axis independently.
        let z_dx = r[0] / s_xx.sqrt();
        let z_dy = r[1] / s_yy.sqrt();

        // Scalar proxy for innovation magnitude in sigma units.
        let z_norm = l2_norm(z_dx, z_dy);

        (
            Self::finite_or_zero(z_dx),
            Self::finite_or_zero(z_dy),
            Self::finite_or_zero(z_norm),
        )
    }

    /// Compute along-track and cross-track z-scores based on the predicted
    /// velocity direction.
    ///
    /// Definitions
    /// -----------
    /// Let $\hat{\mathbf{u}}$ be the unit vector along predicted velocity
    /// $\mathbf{v}\_{\mathrm{pred}}$:
    ///
    /// $$\hat{\mathbf{u}} = \frac{\mathbf{v}\_{\mathrm{pred}}}{\|\mathbf{v}\_{\mathrm{pred}}\|}$$
    ///
    /// Let $\hat{\mathbf{n}}$ be its perpendicular:
    /// $\hat{\mathbf{n}} = (-\hat{u}\_y,\; \hat{u}\_x)$.
    ///
    /// Then:
    ///
    /// $$z\_\parallel = \frac{\mathbf{r} \cdot \hat{\mathbf{u}}}{\sqrt{\hat{\mathbf{u}}^\top \mathbf{S} \hat{\mathbf{u}}}} ,\qquad z\_\perp = \frac{\mathbf{r} \cdot \hat{\mathbf{n}}}{\sqrt{\hat{\mathbf{n}}^\top \mathbf{S} \hat{\mathbf{n}}}}$$
    ///
    /// Arguments
    /// ---------
    /// * `r` – Innovation vector $[r\_x,\, r\_y]$.
    /// * `s` – Innovation covariance matrix $\mathbf{S}$ ($2 \times 2$).
    /// * `v_pred` – Predicted velocity vector at the target epoch.
    ///
    /// Return
    /// ------
    /// $(z\_\parallel,\; z\_\perp,\; \|\mathbf{v}\_{\mathrm{pred}}\|)$.
    ///
    /// Notes
    /// -----
    /// If $\|\mathbf{v}\_{\mathrm{pred}}\|$ is invalid or zero, we fall back to a
    /// fixed orthonormal basis: $\hat{\mathbf{u}} = (1,0)$,
    /// $\hat{\mathbf{n}} = (0,1)$.
    ///
    /// This makes the feature well-defined even for seeds with poorly constrained
    /// motion (e.g., too few detections, or numerical artifacts).
    #[inline]
    pub(crate) fn z_along_cross(
        r: [f64; 2],
        s: [[f64; 2]; 2],
        v_pred: [f64; 2],
    ) -> (f64, f64, f64) {
        // Speed of predicted motion.
        let v_norm = l2_norm(v_pred[0], v_pred[1]);

        // Build an orthonormal basis (u along-track, n cross-track).
        let (u, n) = if v_norm.is_finite() && v_norm > 0.0 {
            let u = [v_pred[0] / v_norm, v_pred[1] / v_norm];
            let n = [-u[1], u[0]];
            (u, n)
        } else {
            // Fallback basis if velocity is degenerate.
            ([1.0, 0.0], [0.0, 1.0])
        };

        // Denominator is the standard deviation along direction u: sqrt(uᵀ S u).
        let su = dot2(u, mat_vec2(s, u)).max(Self::FLOOR);
        let z_along = dot2(r, u) / su.sqrt();

        // Denominator is the standard deviation along direction n: sqrt(nᵀ S n).
        let sn = dot2(n, mat_vec2(s, n)).max(Self::FLOOR);
        let z_cross = dot2(r, n) / sn.sqrt();

        (
            Self::finite_or_zero(z_along),
            Self::finite_or_zero(z_cross),
            v_norm,
        )
    }

    /// Compute $\cos(\theta)$ between two vectors given their norms.
    ///
    /// $$\cos \theta = \operatorname{clamp}\_{[-1, 1]}\!\left(\frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\;\|\mathbf{b}\|}\right)$$
    ///
    /// Arguments
    /// ---------
    /// * `a` – First vector.
    /// * `a_norm` – Precomputed $\|\mathbf{a}\|$.
    /// * `b` – Second vector.
    /// * `b_norm` – Precomputed $\|\mathbf{b}\|$.
    ///
    /// Return
    /// ------
    /// Cosine of the angle in $[-1, 1]$ (clamped and finite), or $0$ if invalid.
    ///
    /// Notes
    /// -----
    /// - Using precomputed norms avoids recomputing square roots in hot paths.
    /// - Clamping avoids tiny numerical drift outside $[-1, 1]$ that could occur
    ///   due to floating-point rounding.
    #[inline]
    pub(crate) fn cos_between(a: [f64; 2], a_norm: f64, b: [f64; 2], b_norm: f64) -> f64 {
        if a_norm.is_finite() && b_norm.is_finite() && a_norm > 0.0 && b_norm > 0.0 {
            // dot(a,b) / (|a||b|), clamped to [-1,1] to avoid numeric drift.
            clamp_unit(dot2(a, b) / (a_norm * b_norm))
        } else {
            0.0
        }
    }

    /// Relative speed difference:
    /// $\frac{|a - b|}{a + b + \varepsilon}$.
    ///
    /// This is a scale-free measure of mismatch between two speeds:
    /// - $0$ means same speed,
    /// - larger values indicate increasing mismatch,
    /// - the denominator keeps the value bounded and stable when speeds grow.
    ///
    /// Arguments
    /// ---------
    /// * `a` – First speed (non-negative).
    /// * `b` – Second speed (non-negative).
    ///
    /// Return
    /// ------
    /// Relative speed mismatch (finite), or $0$ if invalid.
    ///
    /// Notes
    /// -----
    /// We include $\varepsilon$ to avoid division by zero in edge cases.
    #[inline]
    pub(crate) fn rel_speed_diff(a: f64, b: f64) -> f64 {
        if (a + b).is_finite() && (a + b) > 0.0 {
            (a - b).abs() / (a + b + Self::EPS)
        } else {
            0.0
        }
    }

    /// Safe division helper.
    ///
    /// This helper is convenient when building features from ratios where:
    /// - the denominator might become 0,
    /// - either operand might be non-finite.
    ///
    /// Arguments
    /// ---------
    /// * `num` – Numerator.
    /// * `denom` – Denominator.
    ///
    /// Return
    /// ------
    /// `num / denom` if both are finite and `denom != 0`, otherwise `0.0`.
    ///
    /// Notes
    /// -----
    /// This is intentionally conservative (returns 0.0 rather than NaN/Inf)
    /// to keep ML datasets stable.
    #[inline]
    pub(crate) fn safe_div(num: f64, denom: f64) -> f64 {
        if num.is_finite() && denom.is_finite() && denom != 0.0 {
            num / denom
        } else {
            0.0
        }
    }

    /// Map non-finite values to `0.0`.
    ///
    /// Rationale
    /// ---------
    /// In large-scale pipelines, a single NaN can poison:
    /// - Parquet datasets (and downstream readers),
    /// - ONNX runtime tensors,
    /// - ML training code that does not expect NaNs.
    ///
    /// This function implements a simple policy:
    /// - if it is finite, keep it,
    /// - else replace by `0.0`.
    ///
    /// Arguments
    /// ---------
    /// * `x` – Input scalar.
    ///
    /// Return
    /// ------
    /// `x` if finite, otherwise `0.0`.
    #[inline]
    pub(crate) fn finite_or_zero(x: f64) -> f64 {
        if x.is_finite() { x } else { 0.0 }
    }
}
