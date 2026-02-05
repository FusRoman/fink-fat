// -----------------------------------------------------------------------------
// Edge position / innovation features
// -----------------------------------------------------------------------------
//
// This module defines `EdgePositionFeatures`, a compact, ML-friendly set of
// dimensionless metrics that describe how well a candidate edge `(from -> to)`
// matches a simple local kinematic model on the tangent plane.
//
// Conceptual picture
// ------------------
// For an edge from seed i ("from") to seed j ("to"):
// 1) propagate seed i to the epoch of seed j on the tangent plane of i,
// 2) project seed j onto the same tangent plane,
// 3) compute the innovation (residual) r = p_to - p_pred,
// 4) compute an innovation covariance S (prediction + measurement uncertainty),
// 5) normalize r using S to obtain dimensionless "surprise" metrics.
//
// Why so many normalizations?
// --------------------------
// Raw residuals depend on cadence (dt), seeing, and astrometric uncertainties.
// Normalizing by S makes the features comparable across nights and observing
// conditions, which helps ML models generalize.
//
// Implementation note
// -------------------
// The heavy lifting is done in `FeatureCore::from_nodes(...)`. This struct is
// a thin, stable wrapper that exposes a subset of those core values as a public
// feature family.
//
// -----------------------------------------------------------------------------
//
// Dependencies:
// - `FeatureCore`: precomputed, sanitized scalar intermediates.
//

use crate::graph::edge::feature_core::FeatureCore;

/// Position/innovation consistency features (dimensionless).
///
/// Definitions
/// -----------
/// Consider an edge from seed `i` ("from") to seed `j` ("to").
///
/// We build:
/// - a predicted position on the tangent plane of `i` propagated to the epoch of `j`,
/// - an innovation (residual) `r = p_to - p_pred`,
/// - an innovation covariance matrix `S`,
/// - its inverse `S⁻¹` (robustly inverted with numerical guards).
///
/// Then we derive:
/// - `chi2_pos = rᵀ S⁻¹ r`
/// - diagonal-based z-scores `z_dx`, `z_dy`
/// - along/cross track z-scores using the predicted motion direction
/// - a fully whitened residual using a Cholesky factorization of `S`
///
/// All features are **dimensionless**.
///
/// Attributes
/// ----------
/// * `chi2_pos` – Position-space Mahalanobis distance (squared).
/// * `log_chi2_pos` – Log-compressed version of `chi2_pos`.
/// * `z_dx`, `z_dy` – Diagonal z-scores (cheap approximation).
/// * `z_resid_norm` – Norm of diagonal z-scores.
/// * `z_along`, `z_cross` – Directional z-scores aligned with predicted motion.
/// * `chol_z1`, `chol_z2` – Whitened residual components (decorrelated).
/// * `chol_z_norm` – Norm of whitened residual (≈ sqrt(chi2_pos) when consistent).
#[derive(Clone, Debug)]
pub struct EdgePositionFeatures {
    /// Mahalanobis squared distance of the position innovation: `rᵀ S⁻¹ r`.
    ///
    /// Interpretation
    /// --------------
    /// - Small values: position residual is compatible with the uncertainties.
    /// - Large values: residual is surprising given the uncertainties (likely false link).
    ///
    /// Statistical intuition
    /// ---------------------
    /// This behaves like a χ² statistic with ~2 degrees of freedom when:
    /// - the model is correct,
    /// - the covariance is meaningful,
    /// - residuals are Gaussian on the tangent plane.
    pub chi2_pos: f64,

    /// `log(chi2_pos + eps)` for numerical stability and better dynamic range.
    ///
    /// Why log?
    /// --------
    /// `chi2_pos` is typically heavy-tailed (a few catastrophic mismatches dominate).
    /// Log-transform compresses large outliers and makes the feature easier to use
    /// for linear/logistic models and tree splits.
    pub log_chi2_pos: f64,

    /// Normalized x residual using diagonal scaling: `dx / sqrt(S_xx)`.
    ///
    /// Notes
    /// -----
    /// This is a cheap, robust z-score approximation that ignores correlations
    /// (off-diagonal terms of `S`).
    pub z_dx: f64,

    /// Normalized y residual using diagonal scaling: `dy / sqrt(S_yy)`.
    ///
    /// Notes
    /// -----
    /// Same approximation as `z_dx`, but on the y axis.
    pub z_dy: f64,

    /// Euclidean norm of `(z_dx, z_dy)`.
    ///
    /// This is a scalar proxy for "how many sigmas away" the innovation is,
    /// ignoring x/y correlations.
    pub z_resid_norm: f64,

    /// Along-track normalized residual.
    ///
    /// Definition
    /// ----------
    /// Let `u` be the unit vector along predicted velocity. Then:
    /// `z_along = (r·u) / sqrt(uᵀ S u)`.
    ///
    /// Interpretation
    /// --------------
    /// Captures whether the innovation is consistent **along the motion direction**.
    /// Many false links drift strongly along-track if the cadence model is wrong.
    pub z_along: f64,

    /// Cross-track normalized residual.
    ///
    /// Definition
    /// ----------
    /// Let `n` be the unit vector orthogonal to the predicted velocity direction.
    /// Then: `z_cross = (r·n) / sqrt(nᵀ S n)`.
    ///
    /// Interpretation
    /// --------------
    /// Cross-track errors often separate true links from spurious ones, because
    /// random associations tend to miss the track direction by a large angle.
    pub z_cross: f64,

    /// Whitened (Cholesky) residual component along the first axis.
    ///
    /// Definition
    /// ----------
    /// We factorize the innovation covariance `S = L·Lᵀ` (L lower-triangular) and
    /// solve `L · z = r`.
    ///
    /// For a 2×2 `L`, the first component is:
    /// `chol_z1 = r_x / L₀₀`.
    ///
    /// Notes
    /// -----
    /// Whitening removes correlations and expresses residuals in sigma units
    /// under the assumed covariance.
    pub chol_z1: f64,

    /// Whitened (Cholesky) residual component along the second axis.
    ///
    /// Definition
    /// ----------
    /// Continuing the solve `L · z = r`:
    /// `chol_z2 = (r_y − L₁₀ · chol_z1) / L₁₁`.
    ///
    /// Interpretation
    /// --------------
    /// Together, `(chol_z1, chol_z2)` form a decorrelated residual vector whose
    /// components are comparable and can be used directly by ML models.
    pub chol_z2: f64,

    /// Euclidean norm of the whitened residuals.
    ///
    /// Property
    /// --------
    /// If whitening succeeds, this should satisfy:
    /// `chol_z_norm = sqrt(chol_z1^2 + chol_z2^2) = sqrt(chi2_pos)`
    /// up to numerical precision and small floors.
    pub chol_z_norm: f64,
}

impl EdgePositionFeatures {
    /// Construct [`EdgePositionFeatures`] from a precomputed [`FeatureCore`].
    ///
    /// Arguments
    /// ---------
    /// * `core` – Shared intermediate computations produced by
    ///   [`FeatureCore::from_nodes`].
    ///
    /// Return
    /// ------
    /// Position/innovation feature family for the edge that generated `core`.
    ///
    /// Notes
    /// -----
    /// This function is intentionally a "pure mapping":
    /// - no recomputation,
    /// - no additional numerical guards,
    /// - all stability policy is handled upstream in `FeatureCore`.
    #[inline]
    pub(crate) fn position_features(core: &FeatureCore) -> Self {
        // Direct field mapping keeps this function predictable and cheap.
        // Any change in the definition of these features should happen in `FeatureCore`.
        Self {
            chi2_pos: core.chi2_pos,
            log_chi2_pos: core.log_chi2_pos,
            z_dx: core.z_dx,
            z_dy: core.z_dy,
            z_resid_norm: core.z_resid_norm,
            z_along: core.z_along,
            z_cross: core.z_cross,
            chol_z1: core.chol_z1,
            chol_z2: core.chol_z2,
            chol_z_norm: core.chol_z_norm,
        }
    }
}
