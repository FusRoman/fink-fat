// -----------------------------------------------------------------------------
// Cadence-robust features (structured)
// -----------------------------------------------------------------------------
//
// This module defines a cadence-robust feature set for **inter-night edges**
// (links) between two `SeedNode` objects.
//
// Why "cadence-robust"?
// ---------------------
// Survey cadence affects the distribution of `dt` (time gaps) between nights.
// Any feature that depends strongly on `dt` (e.g. raw residuals or distances)
// will not generalize well from one cadence to another (ZTF → LSST, rolling
// cadence, weather gaps, etc.).
//
// The feature design here favors:
// - normalized innovations (Mahalanobis / z-scores) rather than raw distances,
// - along-track / cross-track decomposition rather than axis-aligned residuals,
// - angular and relative quantities (dimensionless ratios),
// - robust numerical guards (floors, epsilons, finite checks).
//
// All features are stored as `f64` to keep ML export simple (Parquet / ONNX).

use crate::{
    astro_math::{
        cholesky_lower_sym_2x2, clamp_unit, dot2, invert_sym_2x2, l2_norm, mat_vec2, safe_ln,
    },
    graph::edge::{
        photometry_features::EdgePhotometryFeatures, position_features::EdgePositionFeatures,
        uncertainty_features::EdgeUncertaintyFeatures, velocity_features::EdgeVelocityFeatures,
    },
    seeding::seed_node::SeedNode,
};

/// Cadence-robust edge feature set.
///
/// This is the **public**, ML-friendly container returned by the feature
/// computation pipeline.
///
/// Design
/// ------
/// The feature vector is intentionally **decomposed** into specialized
/// sub-structures:
/// - [`EdgePositionFeatures`] for innovation geometry on the tangent plane,
/// - [`EdgeVelocityFeatures`] for relative kinematic consistency,
/// - [`EdgeUncertaintyFeatures`] for uncertainty/quality ratios (dimensionless),
/// - [`EdgePhotometryFeatures`] for (mostly cadence-invariant) photometry,
///
/// This provides:
/// - better readability at call sites,
/// - easier iteration and ablation studies per feature family,
/// - cleaner code generation for downstream bindings (PyO3, serde, etc.).
///
/// Notes
/// -----
/// All fields are `f64` for easy storage and export (Parquet / Arrow / ONNX).
#[derive(Clone, Debug)]
pub struct EdgeFeatures {
    /// Innovation geometry on the tangent plane (Mahalanobis, along/cross residuals).
    pub position: EdgePositionFeatures,
    /// Relative velocity consistency (angle, speed ratios).
    pub velocity: EdgeVelocityFeatures,
    /// Uncertainty/quality ratios (covariance trace ratios, anisotropy proxies).
    pub uncertainty: EdgeUncertaintyFeatures,
    /// Photometry consistency (normalized flux differences, band sharing).
    pub photometry: EdgePhotometryFeatures,
}

impl EdgeFeatures {
    // -------------------------------------------------------------------------
    // Feature API (high-level)
    // -------------------------------------------------------------------------

    /// Compute the full cadence-robust feature set.
    #[inline]
    pub fn compute_features(from: &SeedNode, to: &SeedNode) -> Self {
        // Build shared intermediate quantities once.
        let core = FeatureCore::from_nodes(from, to);

        EdgeFeatures {
            position: EdgePositionFeatures::position_features(&core),
            velocity: EdgeVelocityFeatures::velocity_features(&core),
            uncertainty: EdgeUncertaintyFeatures::uncertainty_features(from, to),
            photometry: EdgePhotometryFeatures::photometry_features(from, to),
        }
    }

    /// Return an additive negative-log-likelihood-like cost for graph solvers.
    ///
    /// The cost is designed to be:
    /// - physically motivated (Gaussian innovations on the tangent plane),
    /// - mostly parameter-free (only numerical epsilons),
    /// - additive along a trajectory (sum of per-edge costs).
    ///
    /// Definition (up to additive constants)
    /// ------------------------------------
    /// cost =
    ///   0.5 * chi2_pos
    /// + 0.5 * chi2_vel
    /// + 0.5 * z_flux^2
    /// + 0.5 * log(flux_std_ratio)^2
    /// - log(eps_band + band_shared)
    ///
    /// Notes
    /// -----
    /// - `band_shared` is expected to be 0 or 1.
    /// - `flux_std_ratio` can be 0 if undefined; we guard with an epsilon.
    #[inline]
    pub fn kinematic_log_likelihood_cost(&self) -> f64 {
        // Numerical epsilons (not tunable model parameters).
        let eps = 1e-12_f64;
        let eps_band = 1e-3_f64;

        let chi2_pos = self.position.chi2_pos.max(0.0);
        let chi2_vel = self.velocity.chi2_vel.max(0.0);

        let z_flux = self.photometry.z_flux;
        let ln_flux_std_ratio = safe_ln(self.photometry.flux_std_ratio.abs() + eps);

        let band_shared = self.photometry.band_shared.clamp(0.0, 1.0);
        let band_term = -safe_ln(eps_band + band_shared);

        let cost = 0.5 * (chi2_pos + chi2_vel)
            + 0.5 * (z_flux * z_flux)
            + 0.5 * (ln_flux_std_ratio * ln_flux_std_ratio)
            + band_term;

        FeatureCore::finite_or_zero(cost)
    }

    /// Return the total number of scalar leaf features.
    ///
    /// This value is constant and corresponds to the length of
    /// [`EDGE_FEATURE_KEYS`].
    #[inline]
    pub const fn len_flat() -> usize {
        EDGE_FEATURE_KEYS.len()
    }

    /// Return an iterator over all canonical feature names.
    ///
    /// This is allocation-free and preserves the canonical ordering.
    #[inline]
    pub fn flat_names() -> impl Iterator<Item = &'static str> {
        EDGE_FEATURE_KEYS.into_iter().map(|k| k.path())
    }

    /// Retrieve a feature value using a type-safe [`EdgeFeatureKey`].
    ///
    /// This is the **preferred access method** in core Rust code.
    #[inline]
    pub fn get(&self, key: EdgeFeatureKey) -> f64 {
        match key {
            EdgeFeatureKey::PositionChi2Pos => self.position.chi2_pos,
            EdgeFeatureKey::PositionLogChi2Pos => self.position.log_chi2_pos,
            EdgeFeatureKey::PositionZDx => self.position.z_dx,
            EdgeFeatureKey::PositionZDy => self.position.z_dy,
            EdgeFeatureKey::PositionZResidNorm => self.position.z_resid_norm,
            EdgeFeatureKey::PositionZAlong => self.position.z_along,
            EdgeFeatureKey::PositionZCross => self.position.z_cross,
            EdgeFeatureKey::PositionCholZ1 => self.position.chol_z1,
            EdgeFeatureKey::PositionCholZ2 => self.position.chol_z2,
            EdgeFeatureKey::PositionCholZNorm => self.position.chol_z_norm,
            EdgeFeatureKey::VelocityCosDthetaV => self.velocity.cos_dtheta_v,
            EdgeFeatureKey::VelocityRelSpeedDiff => self.velocity.rel_speed_diff,
            EdgeFeatureKey::VelocityInnovSpeedRatio => self.velocity.innov_speed_ratio,
            EdgeFeatureKey::UncertaintyCovVelRatio => self.uncertainty.cov_vel_ratio(),
            EdgeFeatureKey::PhotometryZFlux => self.photometry.z_flux,
            EdgeFeatureKey::PhotometryFluxStdRatio => self.photometry.flux_std_ratio,
            EdgeFeatureKey::PhotometryBandShared => self.photometry.band_shared,
        }
    }

    /// Return the `(name, value)` pair of the `idx`-th feature.
    ///
    /// The index refers to the canonical flat ordering.
    #[inline]
    pub fn flat_at_with_name(&self, idx: usize) -> Option<(&'static str, f64)> {
        let key = *EDGE_FEATURE_KEYS.get(idx)?;
        Some((key.path(), self.get(key)))
    }

    /// Return the value of the `idx`-th feature in canonical order.
    #[inline]
    pub fn flat_at(&self, idx: usize) -> Option<f64> {
        let key = *EDGE_FEATURE_KEYS.get(idx)?;
        Some(self.get(key))
    }

    /// Iterate over all `(name, value)` feature pairs.
    ///
    /// This is the most convenient iterator for plotting and diagnostics.
    #[inline]
    pub fn iter_flat_with_name(&self) -> EdgeFeaturesFlatNameIter<'_> {
        EdgeFeaturesFlatNameIter { f: self, i: 0 }
    }

    /// Iterate over all feature values in canonical order.
    ///
    /// This is the preferred iterator for ML export paths.
    #[inline]
    pub fn iter_flat(&self) -> EdgeFeaturesIter<'_> {
        EdgeFeaturesIter { f: self, i: 0 }
    }

    /// Collect all feature values into a `Vec<f64>` (canonical order).
    ///
    /// Avoid in hot paths.
    #[inline]
    pub fn flat_values_vec(&self) -> Vec<f64> {
        EDGE_FEATURE_KEYS.iter().map(|&k| self.get(k)).collect()
    }

    /// Collect all `(name, value)` pairs into a `Vec`.
    ///
    /// Avoid in hot paths.
    #[inline]
    pub fn flat_named_vec(&self) -> Vec<(&'static str, f64)> {
        EDGE_FEATURE_KEYS
            .iter()
            .map(|&k| (k.path(), self.get(k)))
            .collect()
    }
}

// -----------------------------------------------------------------------------
// Feature core (shared intermediates)
// -----------------------------------------------------------------------------

/// Shared intermediate computations reused across feature sub-sets.
///
/// This internal struct:
/// - computes once the expensive/central quantities (innovation `r`, covariance `S`,
///   predicted velocity direction, etc.),
/// - stores the final scalar values reused in multiple feature families,
/// - keeps high-level feature mapping (`EdgeFeatures { ... }`) short and readable.
///
/// Implementation note
/// -------------------
/// The public structs are meant to be stable API for ML export.
/// `FeatureCore` is intentionally `pub(crate)` so we can refactor internals
/// without breaking external code.
pub(crate) struct FeatureCore {
    // ----------------------------- Position features -----------------------------
    /// Mahalanobis innovation distance `rᵀ S⁻¹ r`.
    pub(crate) chi2_pos: f64,
    /// Log transform of `chi2_pos`.
    pub(crate) log_chi2_pos: f64,
    /// Diagonal z-score for x residual.
    pub(crate) z_dx: f64,
    /// Diagonal z-score for y residual.
    pub(crate) z_dy: f64,
    /// Norm of diagonal z-scores.
    pub(crate) z_resid_norm: f64,
    /// Along-track z-score.
    pub(crate) z_along: f64,
    /// Cross-track z-score.
    pub(crate) z_cross: f64,

    // ----------------------- Whitening (Cholesky) features -----------------------
    /// Whitened (Cholesky) residual component along the first axis.
    pub(crate) chol_z1: f64,
    /// Whitened (Cholesky) residual component along the second axis.
    pub(crate) chol_z2: f64,
    /// Euclidean norm of the whitened residuals (sqrt of chi2).
    pub(crate) chol_z_norm: f64,

    // ----------------------------- Velocity features -----------------------------
    /// Cosine of the angle between predicted and target velocities.
    pub(crate) cos_dtheta_v: f64,
    /// Relative speed difference.
    pub(crate) rel_speed_diff: f64,
    /// Innovation-induced speed ratio.
    pub(crate) innov_speed_ratio: f64,

    /// Velocity innovation Mahalanobis distance `dvᵀ S_vel⁻¹ dv`.
    pub(crate) chi2_vel: f64,
    /// Log transform of `chi2_vel`.
    pub(crate) log_chi2_vel: f64,
}

impl FeatureCore {
    /// Numerical floor used across feature computations.
    ///
    /// Used to:
    /// - stabilize covariance diagonals,
    /// - guard denominators in normalized quantities,
    /// - avoid division by zero / sqrt(0) / singular inversions.
    const FLOOR: f64 = 1e-20;

    /// Generic epsilon used in ratios and small denominators.
    pub(crate) const EPS: f64 = 1e-16;
    /// Build all shared intermediates from an edge.
    ///
    /// Overview
    /// --------
    /// 1. Propagate the "from" seed on its tangent plane to the epoch of "to".
    /// 2. Project the "to" seed position onto the "from" tangent plane.
    /// 3. Build the innovation vector `r = p_to - p_pred`.
    /// 4. Build the innovation covariance `S`.
    /// 5. Compute normalized innovation metrics and kinematic consistency metrics.
    ///
    /// Arguments
    /// ---------
    /// * `e` – Directed edge between two seeds (from older to newer).
    ///
    /// Return
    /// ------
    /// A populated [`FeatureCore`] structure containing all intermediate scalars
    /// reused by the public feature structs.
    ///
    /// Notes
    /// -----
    /// This function is designed to be called once per edge and then reused to
    /// construct multiple feature families.
    #[inline]
    pub(crate) fn from_nodes(from: &SeedNode, to: &SeedNode) -> Self {
        // Time separation in days (TT). Used for propagation and covariance growth.
        let dt = to.delta_days(from);
        let dt_sq = dt * dt;

        // Whether dt is usable as a positive finite number.
        let dt_ok = dt.is_finite() && dt > 0.0;

        // Propagate `from` seed state to the epoch of `to` (on the `from` tangent plane).
        let (p_pred, v_pred, _) = from.propagate_from(dt, dt_sq);

        // Project the target seed position onto the tangent plane of `from`.
        let p_to = Self::project_to_on_from(from, to);

        // Innovation / residual on the tangent plane: r = observed - predicted.
        let r = [p_to[0] - p_pred[0], p_to[1] - p_pred[1]];

        // Innovation covariance S: accounts for prediction uncertainty and target uncertainty.
        let s = Self::innovation_cov(from, to, dt_sq);

        // Robust inverse of S (with flooring and fallback).
        let s_inv = invert_sym_2x2(s, Self::FLOOR);

        // Mahalanobis distance: chi2_pos = rᵀ S⁻¹ r.
        let chi2_pos = Self::finite_or_zero(dot2(r, mat_vec2(s_inv, r)).max(0.0));

        // Log-transform (improves dynamic range and ML behavior).
        let log_chi2_pos = safe_ln(chi2_pos + 1e-16);

        // Diagonal z-scores: cheap, robust proxies.
        let (z_dx, z_dy, z_resid_norm) = Self::z_diag(r, s);

        // Along/cross decomposition using predicted velocity direction.
        let (z_along, z_cross, v_norm) = Self::z_along_cross(r, s, v_pred);

        // Whitened innovation via Cholesky decomposition.
        // Factorize S into L·Lᵀ and solve L·z = r for z.
        let (chol_z1, chol_z2, chol_z_norm) = match cholesky_lower_sym_2x2(s, Self::FLOOR) {
            Some(l) => {
                // Extract lower triangular elements.
                let l00 = l[0][0];
                let l10 = l[1][0];
                let l11 = l[1][1];
                if l00.is_finite() && l00 > 0.0 && l11.is_finite() && l11 > 0.0 {
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

        // Velocity estimated at `to` (already in the same tangent-plane frame).
        let v_to = [to.plane.vel_xy[0], to.plane.vel_xy[1]];
        let v_to_norm = l2_norm(v_to[0], v_to[1]);

        // Directional agreement between predicted and target velocities.
        let cos_dtheta_v = Self::cos_between(v_pred, v_norm, v_to, v_to_norm);

        // Relative speed mismatch.
        let rel_speed_diff = Self::rel_speed_diff(v_norm, v_to_norm);

        // Innovation-induced speed ratio: (|r|/dt) / |v_pred|.
        let r_norm = l2_norm(r[0], r[1]);
        let innov_speed_ratio = if dt_ok && v_norm.is_finite() && v_norm > 0.0 {
            (r_norm / dt) / (v_norm + Self::EPS)
        } else {
            0.0
        };

        // Velocity innovation: dv = v_to - v_pred (same epoch as `to`).
        // Note: this assumes both velocities are expressed in (approximately) the same tangent frame.
        let dv = [v_to[0] - v_pred[0], v_to[1] - v_pred[1]];

        // Innovation covariance in velocity space.
        let s_vel = Self::innovation_cov_vel(from, to);
        let s_vel_inv = invert_sym_2x2(s_vel, Self::FLOOR);

        // Mahalanobis distance in velocity space.
        let chi2_vel = Self::finite_or_zero(dot2(dv, mat_vec2(s_vel_inv, dv)).max(0.0));
        let log_chi2_vel = safe_ln(chi2_vel + 1e-16);

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
        }
    }
    // -------------------------------------------------------------------------
    // Small inline building blocks
    // -------------------------------------------------------------------------

    /// Project the target seed position onto the source tangent plane.
    ///
    /// This uses the optimized precomputed tangent-plane transform stored in
    /// the `from` seed (`radec_to_tangent_precomp`).
    ///
    /// Arguments
    /// ---------
    /// * `from` – Source seed providing the tangent plane definition.
    /// * `to` – Target seed providing the sky position (RA, Dec).
    ///
    /// Return
    /// ------
    /// Target position `[x, y]` on the tangent plane of `from` (radians).
    #[inline]
    pub(crate) fn project_to_on_from(from: &SeedNode, to: &SeedNode) -> [f64; 2] {
        // Project target RA/Dec onto the precomputed tangent plane of `from`.
        from.plane
            .radec_to_tangent_precomp(to.plane.ra_mid, to.plane.dec_mid)
    }

    /// Build the innovation covariance matrix `S`.
    ///
    /// Definition
    /// ----------
    /// We use a simple propagation of uncertainty:
    /// ```text
    /// C_pred ≈ Cpos_from + dt^2 · Cvel_from
    /// S      = C_pred + Cpos_to + floor·I
    /// ```
    ///
    /// This is a pragmatic model:
    /// - keeps the feature definition simple and fast,
    /// - captures uncertainty growth with time gap,
    /// - remains robust under varying cadence through normalization.
    ///
    /// Arguments
    /// ---------
    /// * `from` – Source seed node providing position/velocity covariances.
    /// * `to` – Target seed node providing position covariance at its epoch.
    /// * `dt_sq` – Precomputed `dt²`.
    ///
    /// Return
    /// ------
    /// Innovation covariance matrix `S` (2×2, symmetric up to floating error).
    ///
    /// Notes
    /// -----
    /// A small diagonal floor is added to prevent:
    /// - singular matrices,
    /// - `sqrt(0)` in z-score computations,
    /// - unstable inversion in Mahalanobis distance.
    #[inline]
    pub(crate) fn innovation_cov(from: &SeedNode, to: &SeedNode, dt_sq: f64) -> [[f64; 2]; 2] {
        let cpos_from = from.plane.cov_pos;
        let cvel_from = from.plane.cov_vel;
        let cpos_to = to.plane.cov_pos;

        // Prediction covariance: position uncertainty + (dt^2)*velocity uncertainty.
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

    /// Innovation covariance in velocity space:
    /// S_vel = C_vel(from) + C_vel(to) + floor·I.
    #[inline]
    fn innovation_cov_vel(from: &SeedNode, to: &SeedNode) -> [[f64; 2]; 2] {
        let c1 = from.plane.cov_vel;
        let c2 = to.plane.cov_vel;

        [
            [c1[0][0] + c2[0][0] + Self::FLOOR, c1[0][1] + c2[0][1]],
            [c1[1][0] + c2[1][0], c1[1][1] + c2[1][1] + Self::FLOOR],
        ]
    }

    /// Compute cheap diagonal-based z-scores for the innovation `r`.
    ///
    /// Definition
    /// ----------
    /// We use only the diagonal terms of `S`:
    /// - `z_dx = r_x / sqrt(S_xx)`
    /// - `z_dy = r_y / sqrt(S_yy)`
    /// - `z_norm = hypot(z_dx, z_dy)`
    ///
    /// Arguments
    /// ---------
    /// * `r` – Innovation vector `[dx, dy]` (tangent-plane radians).
    /// * `s` – Innovation covariance matrix `S`.
    ///
    /// Return
    /// ------
    /// `(z_dx, z_dy, z_norm)` – diagonal z-scores and their Euclidean norm.
    ///
    /// Notes
    /// -----
    /// This is intentionally an approximation:
    /// - robust,
    /// - cheap,
    /// - avoids requiring a full whitening transform.
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

    /// Compute along-track and cross-track z-scores based on the predicted velocity direction.
    ///
    /// Definitions
    /// -----------
    /// Let `u` be the unit vector along predicted velocity `v_pred`:
    /// `u = v_pred / |v_pred|`.
    ///
    /// Let `n` be its perpendicular unit vector:
    /// `n = (-u_y, u_x)`.
    ///
    /// Then:
    /// - `z_along = (r·u) / sqrt(uᵀ S u)`
    /// - `z_cross = (r·n) / sqrt(nᵀ S n)`
    ///
    /// Arguments
    /// ---------
    /// * `r` – Innovation vector `[dx, dy]`.
    /// * `s` – Innovation covariance matrix `S`.
    /// * `v_pred` – Predicted velocity vector at the target epoch.
    ///
    /// Return
    /// ------
    /// `(z_along, z_cross, v_norm)` where:
    /// * `z_along` – along-track z-score,
    /// * `z_cross` – cross-track z-score,
    /// * `v_norm` – speed `|v_pred|`.
    ///
    /// Notes
    /// -----
    /// If `|v_pred|` is invalid or zero, we fall back to a fixed orthonormal basis:
    /// `u=(1,0)`, `n=(0,1)`.
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

    /// Compute `cos(angle)` between two vectors given their norms.
    ///
    /// Arguments
    /// ---------
    /// * `a` – First vector.
    /// * `a_norm` – Precomputed `|a|`.
    /// * `b` – Second vector.
    /// * `b_norm` – Precomputed `|b|`.
    ///
    /// Return
    /// ------
    /// Cosine of the angle in `[-1, 1]` (clamped and finite), or `0.0` if invalid.
    ///
    /// Notes
    /// -----
    /// Using precomputed norms avoids recomputing square roots in hot paths.
    #[inline]
    pub(crate) fn cos_between(a: [f64; 2], a_norm: f64, b: [f64; 2], b_norm: f64) -> f64 {
        if a_norm.is_finite() && b_norm.is_finite() && a_norm > 0.0 && b_norm > 0.0 {
            // dot(a,b) / (|a||b|), clamped to [-1,1] to avoid numeric drift.
            clamp_unit(dot2(a, b) / (a_norm * b_norm))
        } else {
            0.0
        }
    }

    /// Relative speed difference: `|a - b| / (a + b + eps)`.
    ///
    /// This is a scale-free measure of mismatch between two speeds.
    ///
    /// Arguments
    /// ---------
    /// * `a` – First speed (non-negative).
    /// * `b` – Second speed (non-negative).
    ///
    /// Return
    /// ------
    /// Relative speed mismatch in `[0, 1]` (approximately), or `0.0` if invalid.
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
    /// Returns `num/denom` if both are finite and `denom != 0`, otherwise `0.0`.
    ///
    /// Arguments
    /// ---------
    /// * `num` – Numerator.
    /// * `denom` – Denominator.
    ///
    /// Return
    /// ------
    /// Safe quotient, or `0.0` if invalid.
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
    /// This is intentionally used to keep ML features stable and avoid propagating
    /// NaNs/Infs into Parquet datasets or ONNX runtimes.
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

// -----------------------------------------------------------------------------
// Flat feature access helpers (field paths + iterator)
// -----------------------------------------------------------------------------
//
// This section provides ergonomic and *stable* access to all scalar leaf fields
// of [`EdgeFeatures`].
//
// Design goals
// ------------
// - Provide **type-safe** access (via [`EdgeFeatureKey`]) for core logic.
// - Provide **string-path** access (via `get_by_path`) for configuration,
//   CLI tools, plotting scripts and interactive analysis.
// - Provide a **flat, canonical ordering** of all features for:
//   - ML dataset export,
//   - feature ablation studies,
//   - consistent plotting and diagnostics.
//
// The flat view is guaranteed to be:
// - allocation-free when iterating,
// - stable across versions unless explicitly changed,
// - consistent between keys, paths, and iterators.

/// Type-safe identifier for every scalar leaf feature in [`EdgeFeatures`].
///
/// Each variant corresponds to **exactly one scalar value** (a "leaf") in the
/// nested [`EdgeFeatures`] structure.
///
/// Rationale
/// ---------
/// This enum avoids *stringly-typed* feature access in core logic and ensures:
/// - compiler-checked exhaustiveness,
/// - safe refactors (renames break at compile time),
/// - a single source of truth for feature identity.
///
/// The enum is intentionally verbose: clarity and robustness are preferred
/// over conciseness here.
///
/// Guarantees
/// ----------
/// - Each variant maps to:
///   - a unique flat index ([`index`](Self::index)),
///   - a unique canonical string path ([`path`](Self::path)).
/// - The mapping is stable and consistent with [`EDGE_FEATURE_KEYS`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum EdgeFeatureKey {
    // ---------------------------------------------------------------------
    // Position-related features
    // ---------------------------------------------------------------------
    /// χ² of the positional residual in the tangent plane.
    PositionChi2Pos,

    /// Log-transformed positional χ² (numerical stabilization).
    PositionLogChi2Pos,

    /// Normalized residual along the RA-like axis.
    PositionZDx,

    /// Normalized residual along the DEC-like axis.
    PositionZDy,

    /// Norm of the normalized residual vector.
    PositionZResidNorm,

    /// Residual projected along the motion direction.
    PositionZAlong,

    /// Residual projected orthogonally to the motion direction.
    PositionZCross,

    /// First Cholesky-whitened residual component.
    PositionCholZ1,

    /// Second Cholesky-whitened residual component.
    PositionCholZ2,

    /// Norm of the Cholesky-whitened residual.
    PositionCholZNorm,

    // ---------------------------------------------------------------------
    // Velocity-related features
    // ---------------------------------------------------------------------
    /// Cosine of the angle between predicted and observed velocity vectors.
    VelocityCosDthetaV,

    /// Relative speed difference between the two nodes.
    VelocityRelSpeedDiff,

    /// Innovation-to-expected speed ratio.
    VelocityInnovSpeedRatio,

    // ---------------------------------------------------------------------
    // Uncertainty-related features
    // ---------------------------------------------------------------------
    /// Ratio of velocity covariance determinants.
    UncertaintyCovVelRatio,

    // ---------------------------------------------------------------------
    // Photometry-related features
    // ---------------------------------------------------------------------
    /// Flux difference expressed as a Z-score.
    PhotometryZFlux,

    /// Ratio of flux standard deviations.
    PhotometryFluxStdRatio,

    /// Indicator of shared photometric band.
    PhotometryBandShared,
}

impl EdgeFeatureKey {
    /// Return the **flat index** of this feature in canonical order.
    ///
    /// This index is used by:
    /// - flat iterators,
    /// - dense ML array exports,
    /// - column alignment between Rust and Python.
    ///
    /// Guarantees
    /// ----------
    /// - Indices are contiguous in `[0, len_flat())`.
    /// - Ordering matches [`EDGE_FEATURE_KEYS`].
    #[inline]
    pub const fn index(self) -> usize {
        match self {
            // Position (0..10)
            Self::PositionChi2Pos => 0,
            Self::PositionLogChi2Pos => 1,
            Self::PositionZDx => 2,
            Self::PositionZDy => 3,
            Self::PositionZResidNorm => 4,
            Self::PositionZAlong => 5,
            Self::PositionZCross => 6,
            Self::PositionCholZ1 => 7,
            Self::PositionCholZ2 => 8,
            Self::PositionCholZNorm => 9,

            // Velocity (10..13)
            Self::VelocityCosDthetaV => 10,
            Self::VelocityRelSpeedDiff => 11,
            Self::VelocityInnovSpeedRatio => 12,

            // Uncertainty (13..17)
            Self::UncertaintyCovVelRatio => 13,

            // Photometry (17..21)
            Self::PhotometryZFlux => 14,
            Self::PhotometryFluxStdRatio => 15,
            Self::PhotometryBandShared => 16,
        }
    }

    /// Return the **canonical string path** of this feature.
    ///
    /// These paths are intended for:
    /// - column names in exported datasets,
    /// - plotting labels,
    /// - CLI / YAML configuration.
    ///
    /// The format is `<group>.<field>` and is guaranteed to be stable.
    #[inline]
    pub const fn path(self) -> &'static str {
        match self {
            // Position
            Self::PositionChi2Pos => "position.chi2_pos",
            Self::PositionLogChi2Pos => "position.log_chi2_pos",
            Self::PositionZDx => "position.z_dx",
            Self::PositionZDy => "position.z_dy",
            Self::PositionZResidNorm => "position.z_resid_norm",
            Self::PositionZAlong => "position.z_along",
            Self::PositionZCross => "position.z_cross",
            Self::PositionCholZ1 => "position.chol_z1",
            Self::PositionCholZ2 => "position.chol_z2",
            Self::PositionCholZNorm => "position.chol_z_norm",

            // Velocity
            Self::VelocityCosDthetaV => "velocity.cos_dtheta_v",
            Self::VelocityRelSpeedDiff => "velocity.rel_speed_diff",
            Self::VelocityInnovSpeedRatio => "velocity.innov_speed_ratio",

            // Uncertainty
            Self::UncertaintyCovVelRatio => "uncertainty.cov_vel_ratio",

            // Photometry
            Self::PhotometryZFlux => "photometry.z_flux",
            Self::PhotometryFluxStdRatio => "photometry.flux_std_ratio",
            Self::PhotometryBandShared => "photometry.band_shared",
        }
    }
}

/// Canonical ordered list of all scalar leaf features.
///
/// This constant defines the **single authoritative ordering** used by:
/// - flat iterators,
/// - index-based access,
/// - ML feature vectors,
/// - exported column names.
///
/// Changing this ordering is a **breaking change** for downstream consumers.
pub const EDGE_FEATURE_KEYS: [EdgeFeatureKey; 17] = [
    // Position
    EdgeFeatureKey::PositionChi2Pos,
    EdgeFeatureKey::PositionLogChi2Pos,
    EdgeFeatureKey::PositionZDx,
    EdgeFeatureKey::PositionZDy,
    EdgeFeatureKey::PositionZResidNorm,
    EdgeFeatureKey::PositionZAlong,
    EdgeFeatureKey::PositionZCross,
    EdgeFeatureKey::PositionCholZ1,
    EdgeFeatureKey::PositionCholZ2,
    EdgeFeatureKey::PositionCholZNorm,
    // Velocity
    EdgeFeatureKey::VelocityCosDthetaV,
    EdgeFeatureKey::VelocityRelSpeedDiff,
    EdgeFeatureKey::VelocityInnovSpeedRatio,
    // Uncertainty
    EdgeFeatureKey::UncertaintyCovVelRatio,
    // Photometry
    EdgeFeatureKey::PhotometryZFlux,
    EdgeFeatureKey::PhotometryFluxStdRatio,
    EdgeFeatureKey::PhotometryBandShared,
];

/// Iterator over `(name, value)` pairs of an [`EdgeFeatures`] instance.
///
/// This iterator:
/// - is exact-sized,
/// - preserves canonical ordering,
/// - performs no allocations.
#[derive(Clone, Debug)]
pub struct EdgeFeaturesFlatNameIter<'a> {
    f: &'a EdgeFeatures,
    i: usize,
}

impl<'a> Iterator for EdgeFeaturesFlatNameIter<'a> {
    type Item = (&'static str, f64);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let out = self.f.flat_at_with_name(self.i)?;
        self.i += 1;
        Some(out)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = EdgeFeatures::len_flat().saturating_sub(self.i);
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for EdgeFeaturesFlatNameIter<'a> {}

/// Iterator over feature values of an [`EdgeFeatures`] instance.
///
/// This iterator:
/// - is exact-sized,
/// - allocation-free,
/// - suitable for dense ML pipelines.
#[derive(Clone, Debug)]
pub struct EdgeFeaturesIter<'a> {
    f: &'a EdgeFeatures,
    i: usize,
}

impl<'a> Iterator for EdgeFeaturesIter<'a> {
    type Item = f64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let out = self.f.flat_at(self.i)?;
        self.i += 1;
        Some(out)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = EdgeFeatures::len_flat().saturating_sub(self.i);
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for EdgeFeaturesIter<'a> {}
