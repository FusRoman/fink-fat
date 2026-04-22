//! Edge position / innovation features
//!
//! This module defines `EdgePositionFeatures`, a compact, ML-friendly set of
//! dimensionless metrics that describe how well a candidate edge `(from -> to)`
//! matches a simple local kinematic model on the tangent plane.
//!
//! Conceptual picture
//! ------------------
//! For an edge from seed i ("from") to seed j ("to"):
//! 1) propagate seed i to the epoch of seed j on the tangent plane of i,
//! 2) project seed j onto the same tangent plane,
//! 3) compute the innovation (residual) r = p_to - p_pred,
//! 4) compute an innovation covariance S (prediction + measurement uncertainty),
//! 5) normalize r using S to obtain dimensionless "surprise" metrics.
//!
//! Why so many normalizations?
//! --------------------------
//! Raw residuals depend on cadence (dt), seeing, and astrometric uncertainties.
//! Normalizing by S makes the features comparable across nights and observing
//! conditions, which helps ML models generalize.
//!
//! Implementation note
//! -------------------
//! The heavy lifting is done in `FeatureCore::from_nodes(...)`. This struct is
//! a thin, stable wrapper that exposes a subset of those core values as a public
//! feature family.
//!
//! -----------------------------------------------------------------------------
//!
//! Dependencies:
//! - `FeatureCore`: precomputed, sanitized scalar intermediates.
//!

use crate::graph::edge::edge_features::feature_core::FeatureCore;

/// Position/innovation consistency features (dimensionless).
///
/// Definitions
/// -----------
/// Consider an edge from seed $i$ (`from`) to seed $j$ (`to`).
///
/// We build:
/// - a predicted position on the tangent plane of $i$ propagated to the epoch of $j$,
/// - an innovation (residual) $\mathbf{r} = \mathbf{p}\_{\mathrm{to}} - \mathbf{p}\_{\mathrm{pred}}$,
/// - an innovation covariance matrix $\mathbf{S}$,
/// - its inverse $\mathbf{S}^{-1}$ (robustly inverted with numerical guards).
///
/// Then we derive:
///
/// - $\chi^2\_{\mathrm{pos}} = \mathbf{r}^\top \mathbf{S}^{-1} \mathbf{r}$
/// - diagonal-based z-scores $z\_{\Delta x}$, $z\_{\Delta y}$
/// - along/cross track z-scores using the predicted motion direction
/// - a fully whitened residual using a Cholesky factorization of $\mathbf{S}$
///
/// All features are **dimensionless**.
///
/// Attributes
/// ----------
/// * `chi2_pos` – Position-space Mahalanobis distance (squared), $\chi^2\_{\mathrm{pos}}$.
/// * `log_chi2_pos` – Log-compressed version: $\ln(\chi^2\_{\mathrm{pos}} + \varepsilon)$.
/// * `z_dx`, `z_dy` – Diagonal z-scores (cheap approximation).
/// * `z_resid_norm` – $\sqrt{z\_{\Delta x}^2 + z\_{\Delta y}^2}$.
/// * `z_along`, `z_cross` – Directional z-scores aligned with predicted motion.
/// * `chol_z1`, `chol_z2` – Whitened residual components (decorrelated).
/// * `chol_z_norm` – $\|\mathbf{z}\| \approx \sqrt{\chi^2\_{\mathrm{pos}}}$ when consistent.
#[derive(Clone, Debug)]
pub struct EdgePositionFeatures {
    /// Mahalanobis squared distance of the position innovation:
    /// $\chi^2\_{\mathrm{pos}} = \mathbf{r}^\top \mathbf{S}^{-1} \mathbf{r}$.
    ///
    /// Interpretation
    /// --------------
    /// - Small values: position residual is compatible with the uncertainties.
    /// - Large values: residual is surprising given the uncertainties (likely false link).
    ///
    /// Statistical intuition
    /// ---------------------
    /// Under a correct model with Gaussian residuals on the tangent plane,
    /// $\chi^2\_{\mathrm{pos}} \sim \chi^2(2)$ (two degrees of freedom).
    pub chi2_pos: f64,

    /// Log-compressed position $\chi^2$:
    /// $\ln(\chi^2\_{\mathrm{pos}} + \varepsilon)$.
    ///
    /// Why log?
    /// --------
    /// $\chi^2\_{\mathrm{pos}}$ is typically heavy-tailed (a few catastrophic
    /// mismatches dominate). Log-transform compresses large outliers and makes
    /// the feature easier to use for linear/logistic models and tree splits.
    pub log_chi2_pos: f64,

    /// Normalized $x$ residual using diagonal scaling:
    /// $z\_{\Delta x} = r\_x \,/\, \sqrt{S\_{xx}}$.
    ///
    /// Notes
    /// -----
    /// This is a cheap, robust z-score approximation that ignores correlations
    /// (off-diagonal terms of $\mathbf{S}$).
    pub z_dx: f64,

    /// Normalized $y$ residual using diagonal scaling:
    /// $z\_{\Delta y} = r\_y \,/\, \sqrt{S\_{yy}}$.
    ///
    /// Notes
    /// -----
    /// Same approximation as `z_dx`, but on the $y$ axis.
    pub z_dy: f64,

    /// Euclidean norm of the diagonal z-scores:
    /// $\sqrt{z\_{\Delta x}^2 + z\_{\Delta y}^2}$.
    ///
    /// This is a scalar proxy for "how many sigmas away" the innovation is,
    /// ignoring $x$/$y$ correlations.
    pub z_resid_norm: f64,

    /// Along-track normalized residual.
    ///
    /// Definition
    /// ----------
    /// Let $\hat{\mathbf{u}}$ be the unit vector along predicted velocity. Then:
    ///
    /// $$z\_\parallel = \frac{\mathbf{r} \cdot \hat{\mathbf{u}}}{\sqrt{\hat{\mathbf{u}}^\top \mathbf{S} \hat{\mathbf{u}}}}$$
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
    /// Let $\hat{\mathbf{n}} = (-\hat{u}\_y,\; \hat{u}\_x)$ be the unit vector
    /// perpendicular to the predicted velocity direction. Then:
    ///
    /// $$z\_\perp = \frac{\mathbf{r} \cdot \hat{\mathbf{n}}}{\sqrt{\hat{\mathbf{n}}^\top \mathbf{S} \hat{\mathbf{n}}}}$$
    ///
    /// Interpretation
    /// --------------
    /// Cross-track errors often separate true links from spurious ones, because
    /// random associations tend to miss the track direction by a large angle.
    pub z_cross: f64,

    /// First whitened (Cholesky) residual component.
    ///
    /// Definition
    /// ----------
    /// We factorize the innovation covariance
    /// $\mathbf{S} = \mathbf{L}\,\mathbf{L}^\top$ ($\mathbf{L}$ lower-triangular)
    /// and solve $\mathbf{L}\,\mathbf{z} = \mathbf{r}$.
    ///
    /// For a $2 \times 2$ system:
    /// $z\_1 = r\_x / L\_{00}$.
    ///
    /// Notes
    /// -----
    /// Whitening removes correlations and expresses residuals in sigma units
    /// under the assumed covariance.
    pub chol_z1: f64,

    /// Second whitened (Cholesky) residual component.
    ///
    /// Definition
    /// ----------
    /// $z\_2 = (r\_y - L\_{10} \, z\_1) / L\_{11}$.
    ///
    /// Interpretation
    /// --------------
    /// Together, $(z\_1, z\_2)$ form a decorrelated residual vector whose
    /// components are comparable and can be used directly by ML models.
    pub chol_z2: f64,

    /// Euclidean norm of the whitened residuals.
    ///
    /// Property
    /// --------
    /// If whitening succeeds, this should satisfy:
    ///
    /// $$\|\mathbf{z}\| = \sqrt{z\_1^2 + z\_2^2} = \sqrt{\chi^2\_{\mathrm{pos}}}$$
    ///
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
            z_dx: core.z_score.dx,
            z_dy: core.z_score.dy,
            z_resid_norm: core.z_resid_norm,
            z_along: core.z_along_cross.dx,
            z_cross: core.z_along_cross.dy,
            chol_z1: core.chol_z.dx,
            chol_z2: core.chol_z.dy,
            chol_z_norm: core.chol_z_norm,
        }
    }
}
