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

use photom::coordinates::{cov2::Cov2, gnomonic_projection::TangentVec};

use crate::{astro_math::safe_ln, seeding::SeedNode};

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
#[derive(Clone, Copy)]
pub(crate) struct FeatureCore {
    // ----------------------------- Position features -----------------------------
    /// Mahalanobis innovation distance:
    /// $\chi^2\_{\mathrm{pos}} = \mathbf{r}^\top \mathbf{S}^{-1} \mathbf{r}$.
    pub(crate) chi2_pos: f64,
    /// Log-compressed position $\chi^2$:
    /// $\ln(\chi^2\_{\mathrm{pos}} + \varepsilon)$.
    pub(crate) log_chi2_pos: f64,

    /// Diagonal z-score for $x$ and $y$ residuals:
    /// $z\_{\Delta x} = r\_x / \sqrt{S\_{xx}}$.
    /// $z\_{\Delta y} = r\_y / \sqrt{S\_{yy}}$.
    pub(crate) z_score: TangentVec,
    /// Norm of diagonal z-scores:
    /// $\sqrt{z\_{\Delta x}^2 + z\_{\Delta y}^2}$.
    pub(crate) z_resid_norm: f64,

    /// Along-Cross-track z-score:
    /// $z\_\parallel = (\mathbf{r} \cdot \hat{\mathbf{u}}) / \sqrt{\hat{\mathbf{u}}^\top \mathbf{S} \hat{\mathbf{u}}}$.
    /// $z\_\perp = (\mathbf{r} \cdot \hat{\mathbf{n}}) / \sqrt{\hat{\mathbf{n}}^\top \mathbf{S} \hat{\mathbf{n}}}$.
    pub(crate) z_along_cross: TangentVec,

    // ----------------------- Whitening (Cholesky) features -----------------------
    /// First whitened residual component $z\_1 = r\_x / L\_{00}$,
    /// from solving $\mathbf{L}\,\mathbf{z} = \mathbf{r}$.
    /// $z\_2 = (r\_y - L\_{10}\,z\_1) / L\_{11}$.
    pub(crate) chol_z: TangentVec,
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
    ///
    /// Notes
    /// -----
    /// Not used by the current cost-function path, which relies on the spherical
    /// residual [`r_sph`](Self::r_sph) instead.  Retained for potential debug use
    /// and to keep the struct self-describing.
    #[allow(dead_code)]
    pub(crate) r_pos: TangentVec,
    /// Velocity innovation $\delta\mathbf{v} = \mathbf{v}\_{{\mathrm{to}}} - \mathbf{v}\_{{\mathrm{pred}}}$
    /// (rad/day).
    pub(crate) dv: TangentVec,
    /// Great-circle angular distance between the sky-back-projected predicted position
    /// and `to.plane.ra_mid` / `to.plane.dec_mid` (radians, in $[0, \pi]$).
    ///
    /// Computed by:
    /// 1. Inverting the gnomonic projection of `p_pred` via
    ///    [`tangent_to_radec`](crate::astro_math::tangent_to_radec),
    /// 2. Measuring the great-circle distance to `to` via
    ///    [`ang_sep`](crate::astro_math::ang_sep).
    ///
    /// Unlike the tangent-plane Cartesian residual $\mathbf{r}$, this quantity is
    /// bounded to $[0, \pi]$ and is stable for any angular separation between
    /// the seed centres.  It is used **exclusively** in the cost-function path
    /// to replace the 2-D Mahalanobis $\chi^2\_\mathrm{pos}$ which diverges when
    /// the gnomonic projection denominator $\cos c \to 0$.
    pub(crate) r_sph: f64,
    /// Scalar positional innovation variance for the cost path (baseline, no CWNA).
    ///
    /// Defined as:
    ///
    /// $$S\_\mathrm{scalar} = \frac{S\_{xx} + S\_{yy}}{2} = \frac{\operatorname{tr}(\mathbf{S}\_\mathrm{pos})}{2}$$
    ///
    /// This isotropic approximation serves as the denominator in the
    /// spherical positional $\chi^2$:
    ///
    /// $$\chi^2\_{\mathrm{pos,sph}} = \frac{d^{2}}{S\_\mathrm{scalar}}$$
    ///
    /// where $d$ is [`r_sph`](Self::r_sph).  Using the trace-half instead of
    /// the full 2-D Mahalanobis is well-motivated when the residual is already
    /// collapsed to a scalar great-circle distance.
    pub(crate) s_pos_scalar: f64,
}

impl FeatureCore {
    /// Numerical floor used across feature computations.
    ///
    /// This constant is applied to:
    /// - stabilize covariance diagonals (avoid singular matrices),
    /// - guard denominators in normalized quantities (`sqrt(S_xx)`),
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
        let (p_pred, v_pred, _) = from.propagate_from(dt);

        // Back-project the predicted tangent-plane position to sky coordinates,
        // then measure the great-circle separation to the target position.
        // This is stable for any angular separation between seed centres;
        // it avoids the gnomonic denominator blow-up that affects `r` below.
        let p_pred_equ = p_pred.unproject();
        let r_sph = p_pred_equ.angular_separation(&to.tangent_seed_center());

        // ---------------------------------------------------------------------
        // 3) Project `to` onto `from` tangent plane
        // ---------------------------------------------------------------------
        // Project the target seed position onto the tangent plane of `from`.
        let p_to = to.project_onto(from);

        // ---------------------------------------------------------------------
        // 4) Innovation vector r = observed - predicted
        // ---------------------------------------------------------------------
        // Innovation / residual on the tangent plane: r = observed - predicted.
        let r_innovation_vector = p_to - p_pred;

        // ---------------------------------------------------------------------
        // 5) Innovation covariance S
        // ---------------------------------------------------------------------
        // Innovation covariance S: accounts for prediction uncertainty and target uncertainty.
        let s = from.innovation_cov_pos(to, dt_sq);

        // Scalar baseline positional variance for the cost path:
        // S_scalar = tr(S) / 2 — isotropic proxy used with the spherical residual r_sph.
        let s_pos_scalar = s.trace().max(Self::FLOOR) / 2.0;

        // ---------------------------------------------------------------------
        // 6) Position-space metrics (Mahalanobis, z-scores, whitening)
        // ---------------------------------------------------------------------

        // Diagonal z-scores: cheap, robust proxies that ignore correlation.
        let (z_score_vector, z_resid_norm) = Self::z_diag(r_innovation_vector, s);

        // Along/cross decomposition using predicted velocity direction.
        let (z_along_cross, v_norm) = Self::z_along_cross(r_innovation_vector, s, v_pred);

        // Full whitening via Cholesky: S = L Lᵀ, z = L⁻¹ r.
        // Yields χ² = ‖z‖² consistent with the whitened components.
        let (chol_z, chi2_pos_raw) = s
            .whiten_cholesky(r_innovation_vector, Self::FLOOR)
            .unwrap_or((TangentVec { dx: 0.0, dy: 0.0 }, 0.0));
        let chol_z_norm = Self::finite_or_zero(chi2_pos_raw.sqrt());

        // chi2_pos = ‖L⁻¹ r‖² already computed above via `whiten_cholesky`.
        // Log-transform improves dynamic range and ML behavior; the +1e-16
        // offset avoids ln(0) when the innovation is exactly zero.
        let chi2_pos = Self::finite_or_zero(chi2_pos_raw);
        // Log-transform (improves dynamic range and ML behavior).
        // We add a tiny constant to avoid ln(0).
        let log_chi2_pos = safe_ln(chi2_pos + 1e-16);

        // ---------------------------------------------------------------------
        // 7) Velocity consistency metrics
        // ---------------------------------------------------------------------
        // Velocity estimated at `to` (already in the same tangent-plane frame).
        let v_to = to.plane_model.vel.v;
        let v_to_norm = v_to.norm();

        // Directional agreement between predicted and target velocities.
        let cos_dtheta_v = v_pred.dot(v_to) / ((v_norm * v_to_norm).max(Self::EPS));

        // Relative speed mismatch (scale-free).
        let rel_speed_diff = Self::rel_speed_diff(v_norm, v_to_norm);

        // Innovation-induced speed ratio: (|r|/dt) / |v_pred|.
        //
        // Intuition:
        // - |r|/dt is the speed "suggested" by the observed discrepancy,
        // - dividing by |v_pred| measures how large that discrepancy is relative
        //   to predicted motion.
        let r_innov_norm = r_innovation_vector.norm();
        let innov_speed_ratio = if dt_ok && v_norm.is_finite() && v_norm > 0.0 {
            (r_innov_norm / dt) / (v_norm + Self::EPS)
        } else {
            0.0
        };

        // ---------------------------------------------------------------------
        // 8) Velocity-space Mahalanobis metrics
        // ---------------------------------------------------------------------
        // Velocity innovation: dv = v_to - v_pred (same epoch as `to`).
        // Note: this assumes both velocities are expressed in (approximately) the same tangent frame.
        let dv = v_to - v_pred;

        // Innovation covariance in velocity space.
        let v_to_cov = to.plane_model.vel.cov;
        let v_from_cov = from.plane_model.vel.cov;
        let s_vel = (v_from_cov + v_to_cov).inflate_isotropic(Self::FLOOR);

        // Mahalanobis distance in velocity space via Cholesky whitening:
        //   S_vel = L Lᵀ,  z = L⁻¹ dv,  χ²_vel = ‖z‖².
        // On SPD failure (near-singular or non-finite covariance), fall back to 0
        // (neutral for downstream ML/cost).
        let chi2_vel = s_vel
            .whiten_cholesky(dv, Self::FLOOR)
            .map(|(_, chi2)| Self::finite_or_zero(chi2))
            .unwrap_or(0.0);

        let log_chi2_vel = Self::finite_or_zero(safe_ln(chi2_vel + 1e-16));

        // ---------------------------------------------------------------------
        // Pack outputs (sanitizing where appropriate)
        // ---------------------------------------------------------------------
        Self {
            chi2_pos,
            log_chi2_pos: Self::finite_or_zero(log_chi2_pos),

            z_score: z_score_vector,
            z_resid_norm,
            z_along_cross,

            chol_z,
            chol_z_norm,

            cos_dtheta_v,
            rel_speed_diff,
            innov_speed_ratio: Self::finite_or_zero(innov_speed_ratio),

            chi2_vel,
            log_chi2_vel: Self::finite_or_zero(log_chi2_vel),

            // Cached for reuse in compute_cost (avoids re-propagation/re-projection).
            dt,
            dt_sq,
            r_pos: r_innovation_vector,
            dv,
            r_sph,
            s_pos_scalar,
        }
    }

    // -------------------------------------------------------------------------
    // Small inline building blocks
    // -------------------------------------------------------------------------

    /// Compute cheap diagonal-based z-scores for the innovation $\mathbf{r}$.
    ///
    /// Definition
    /// ----------
    /// Only the diagonal terms of $\mathbf{S}$ are used:
    ///
    /// $$z_{\Delta x} = \frac{r_x}{\sqrt{S_{xx}}} ,\quad z_{\Delta y} = \frac{r_y}{\sqrt{S_{yy}}} ,\quad z_{\mathrm{norm}} = \sqrt{z_{\Delta x}^2 + z_{\Delta y}^2}$$
    ///
    /// # Arguments
    ///
    /// - `r` — Innovation vector $\mathbf{r} = (r_x, r_y)$ on the tangent plane
    ///   (radians).
    /// - `s` — Innovation covariance $\mathbf{S}$ ($2 \times 2$, symmetric).
    ///
    /// # Returns
    ///
    /// `(f64, f64, f64)` — Tuple $(z_{\Delta x},\, z_{\Delta y},\, z_{\mathrm{norm}})$
    /// containing the per-axis diagonal z-scores and their Euclidean norm.
    ///
    /// # Notes
    ///
    /// This approximation:
    /// - ignores the off-diagonal correlation $S_{xy}$,
    /// - is robust and cheap to evaluate,
    /// - remains useful as an ML feature even when full whitening (Cholesky of
    ///   $\mathbf{S}$) fails due to near-singularity.
    #[inline]
    pub(crate) fn z_diag(r: TangentVec, s: Cov2) -> (TangentVec, f64) {
        let z = s.whiten_diag(r, Self::FLOOR);
        let z_norm = Self::finite_or_zero(z.norm());
        (z, z_norm)
    }

    /// Compute along-track and cross-track z-scores based on the predicted
    /// velocity direction.
    ///
    /// Definitions
    /// -----------
    /// Let $\hat{\mathbf{u}}$ be the unit vector along predicted velocity
    /// $\mathbf{v}_{\mathrm{pred}}$:
    ///
    /// $$\hat{\mathbf{u}} = \frac{\mathbf{v}_{\mathrm{pred}}}{\|\mathbf{v}_{\mathrm{pred}}\|}$$
    ///
    /// Let $\hat{\mathbf{n}}$ be its perpendicular:
    /// $\hat{\mathbf{n}} = (-\hat{u}_y,\; \hat{u}_x)$.
    ///
    /// Then:
    ///
    /// $$z_\parallel = \frac{\mathbf{r} \cdot \hat{\mathbf{u}}}{\sqrt{\hat{\mathbf{u}}^\top \mathbf{S} \hat{\mathbf{u}}}}, \qquad z_\perp = \frac{\mathbf{r} \cdot \hat{\mathbf{n}}}{\sqrt{\hat{\mathbf{n}}^\top \mathbf{S} \hat{\mathbf{n}}}}$$
    ///
    /// Arguments
    /// ---------
    /// * `r` – Innovation vector on the tangent plane.
    /// * `s` – Innovation covariance $\mathbf{S}$ in the same tangent frame.
    /// * `v_pred` – Predicted velocity vector at the target epoch, in the same frame.
    ///
    /// Return
    /// ------
    /// * `TangentVec` – Whitened innovation in the along/cross basis:
    ///   `dx = z_parallel`, `dy = z_perp`.
    /// * `f64` – Speed $\|\mathbf{v}_{\mathrm{pred}}\|$ (reused by callers for
    ///   other kinematic features).
    ///
    /// Notes
    /// -----
    /// - When $\|\mathbf{v}_{\mathrm{pred}}\|$ is not finite or is zero, a fallback
    ///   orthonormal basis $\hat{\mathbf{u}} = (1,0)$, $\hat{\mathbf{n}} = (0,1)$ is
    ///   used. This keeps the feature well-defined for seeds with poorly
    ///   constrained motion (too few detections, numerical artifacts).
    /// - Denominators are floored at [`Self::FLOOR`] before taking the square root
    ///   to guard against near-singular covariances.
    #[inline]
    pub(crate) fn z_along_cross(r: TangentVec, s: Cov2, v_pred: TangentVec) -> (TangentVec, f64) {
        let v_norm = v_pred.norm();

        let (u, n) = if v_norm.is_finite() && v_norm > 0.0 {
            let inv = 1.0 / v_norm;
            let u = TangentVec {
                dx: v_pred.dx * inv,
                dy: v_pred.dy * inv,
            };
            let n = TangentVec {
                dx: -u.dy,
                dy: u.dx,
            };
            (u, n)
        } else {
            (
                TangentVec { dx: 1.0, dy: 0.0 },
                TangentVec { dx: 0.0, dy: 1.0 },
            )
        };

        let su = s.quad_form(u).max(Self::FLOOR);
        let sn = s.quad_form(n).max(Self::FLOOR);

        let z = TangentVec {
            dx: Self::finite_or_zero(r.dot(u) / su.sqrt()),
            dy: Self::finite_or_zero(r.dot(n) / sn.sqrt()),
        };

        (z, v_norm)
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
