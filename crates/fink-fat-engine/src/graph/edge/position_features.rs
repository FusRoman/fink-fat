use crate::graph::edge::edge_features::FeatureCore;

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
/// Then:
/// - `chi2_pos = rᵀ S⁻¹ r`
/// - diagonal-based z-scores `z_dx`, `z_dy`
/// - along/cross track z-scores using the predicted motion direction.
///
/// All features are **dimensionless**.
#[derive(Clone, Debug)]
pub struct EdgePositionFeatures {
    /// Mahalanobis squared distance of the position innovation: `rᵀ S⁻¹ r`.
    ///
    /// This behaves like a χ² statistic with ~2 degrees of freedom when:
    /// - the model is correct,
    /// - the covariance is meaningful,
    /// - residuals are Gaussian on the tangent plane.
    pub chi2_pos: f64,

    /// `log(chi2_pos + eps)` for numerical stability and better dynamic range.
    ///
    /// Log-transform is useful for ML since `chi2_pos` is heavy-tailed.
    pub log_chi2_pos: f64,

    /// Normalized x residual using diagonal scaling: `dx / sqrt(S_xx)`.
    ///
    /// This is a cheap, robust z-score approximation that ignores correlations.
    pub z_dx: f64,

    /// Normalized y residual using diagonal scaling: `dy / sqrt(S_yy)`.
    ///
    /// This is a cheap, robust z-score approximation that ignores correlations.
    pub z_dy: f64,

    /// Euclidean norm of `(z_dx, z_dy)`.
    ///
    /// Useful as a scalar "how surprising is the innovation" proxy.
    pub z_resid_norm: f64,

    /// Along-track normalized residual.
    ///
    /// Let `u` be the unit vector along predicted velocity. Then:
    /// `z_along = (r·u) / sqrt(uᵀ S u)`.
    ///
    /// This captures whether the residual is consistent **along motion**.
    pub z_along: f64,

    /// Cross-track normalized residual.
    ///
    /// Let `n` be the unit vector orthogonal to the predicted velocity direction.
    /// Then: `z_cross = (r·n) / sqrt(nᵀ S n)`.
    ///
    /// Cross-track errors often separate true links from spurious ones.
    pub z_cross: f64,

    /// Whitened (Cholesky) residual component along the first axis.
    ///
    /// This is obtained by factorizing the innovation covariance `S = L·Lᵀ` and
    /// solving `L · z = r`, where `r` is the innovation vector.
    /// In practice:
    /// `chol_z1 = r_x / L₀₀`.
    pub chol_z1: f64,

    /// Whitened (Cholesky) residual component along the second axis.
    ///
    /// Using the same factorization `S = L·Lᵀ`, the second component is
    /// `chol_z2 = (r_y − L₁₀ · chol_z1) / L₁₁`.
    ///
    /// Together, `(chol_z1, chol_z2)` are *fully decorrelated* and expressed in
    /// units of sigma.
    pub chol_z2: f64,

    /// Euclidean norm of the whitened residuals.
    ///
    /// This is exactly `sqrt(chi2_pos)`, since `chi2_pos = zᵀ z` for the
    /// whitened residual `z`.
    pub chol_z_norm: f64,
}

impl EdgePositionFeatures {
    /// Compute only position/innovation features.
    #[inline]
    pub(crate) fn position_features(core: &FeatureCore) -> Self {
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
