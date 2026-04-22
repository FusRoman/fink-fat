//! Edge uncertainty / quality features
//!
//! This module defines `EdgeUncertaintyFeatures`, a tiny feature family focused on
//! uncertainty evolution between two seeds (`from -> to`).
//!
//! Rationale
//! ---------
//! In large-scale linking, many false candidates come from poorly constrained
//! seed states (high covariance), or from transitions where the uncertainty
//! grows/shrinks abruptly (change of observing conditions, few points, bad fit).
//!
//! We expose a simple, dimensionless proxy:
//! - the ratio of velocity covariance traces between `to` and `from`.
//!
//! Why the trace?
//! --------------
//! For a 2×2 covariance matrix C, tr(C) = C_xx + C_yy is a cheap scalar summary of
//! total variance (sum of marginal variances). It ignores correlation and
//! anisotropy but is:
//! - fast,
//! - robust,
//! - easy for ML models to consume.
//!
//! Numerical stability
//! -------------------
//! - We clamp traces to be non-negative (defensive against tiny negative values
//!   from numerical noise).
//! - We use `FeatureCore::safe_div` and `FeatureCore::EPS` to avoid division by 0
//!   or NaNs/Infs.
//!
//! -----------------------------------------------------------------------------

use crate::{graph::edge::edge_features::feature_core::FeatureCore, seeding::SeedNode};

/// Uncertainty/quality ratios (dimensionless).
///
/// These features quantify how the uncertainty evolves between the two seeds,
/// and provide a simple scalar proxy for the "size" of the local velocity
/// covariance.
///
/// Current definition
/// ------------------
/// This is a single-value wrapper storing:
///
/// $$r\_{\mathrm{cov}} = \frac{\operatorname{tr}(\mathbf{C}^{\mathrm{vel}}\_{\mathrm{to}})}{\operatorname{tr}(\mathbf{C}^{\mathrm{vel}}\_{\mathrm{from}}) + \varepsilon}$$
///
/// where $\operatorname{tr}(\mathbf{C}) = C\_{xx} + C\_{yy}$ is the trace of the
/// $2 \times 2$ velocity covariance matrix.
///
/// Attributes
/// ----------
/// * `0` – Stored scalar $r\_{\mathrm{cov}}$ (see above).
///
/// Notes
/// -----
/// - $r\_{\mathrm{cov}} > 1$ suggests the target seed has a larger velocity
///   uncertainty than the source seed.
/// - $r\_{\mathrm{cov}} < 1$ suggests the target seed is better constrained
///   in velocity.
/// - Interpreting this physically depends on how covariances are estimated in
///   the seeding stage (number of points, fit model, etc.).
#[derive(Clone, Debug)]
pub struct EdgeUncertaintyFeatures(pub f64);

impl EdgeUncertaintyFeatures {
    /// Return the velocity covariance trace ratio stored in this struct.
    ///
    /// This is a small convenience accessor to avoid exposing the tuple field
    /// directly in downstream code.
    ///
    /// Return
    /// ------
    /// Velocity covariance trace ratio $r\_{\mathrm{cov}}$:
    ///
    /// $$r\_{\mathrm{cov}} = \frac{\operatorname{tr}(\mathbf{C}^{\mathrm{vel}}\_{\mathrm{to}})}{\operatorname{tr}(\mathbf{C}^{\mathrm{vel}}\_{\mathrm{from}}) + \varepsilon}$$
    #[inline]
    pub fn cov_vel_ratio(&self) -> f64 {
        self.0
    }

    /// Compute uncertainty/quality ratio features for an edge `(from -> to)`.
    ///
    /// Overview
    /// --------
    /// 1. Extract the $2 \times 2$ velocity covariance matrices for both seeds.
    /// 2. Reduce each covariance to a scalar using the trace:
    ///    $\operatorname{tr}(\mathbf{C}) = C\_{xx} + C\_{yy}$.
    /// 3. Form a stabilized ratio:
    ///    $r\_{\mathrm{cov}} = \operatorname{tr}(\mathbf{C}^{\mathrm{vel}}\_{\mathrm{to}}) / (\operatorname{tr}(\mathbf{C}^{\mathrm{vel}}\_{\mathrm{from}}) + \varepsilon)$.
    ///
    /// Arguments
    /// ---------
    /// * `from` – Source seed node (older epoch).
    /// * `to` – Target seed node (newer epoch).
    ///
    /// Return
    /// ------
    /// [`EdgeUncertaintyFeatures`] containing a single dimensionless scalar:
    /// the velocity covariance trace ratio $r\_{\mathrm{cov}}$.
    ///
    /// Notes
    /// -----
    /// - Each trace is clamped with `.max(0.0)` to guard against small negative
    ///   values caused by floating-point noise.
    /// - The ratio uses $\varepsilon$ and `safe_div` to remain stable when
    ///   $\operatorname{tr}(\mathbf{C}^{\mathrm{vel}}\_{\mathrm{from}})$ is near zero.
    #[inline]
    pub fn uncertainty_features(from: &SeedNode, to: &SeedNode) -> Self {
        // Small epsilon used to stabilize ratios.
        let eps = FeatureCore::EPS;

        // Extract velocity covariance matrices (2×2, tangent-plane frame).
        let cvel_from = from.plane_model.vel.cov;
        let cvel_to = to.plane_model.vel.cov;

        // Reduce each covariance to a scalar "total variance" proxy.
        // `.max(0.0)` prevents negative traces from numerical noise.
        let tr_vel_from = cvel_from.trace().max(0.0);
        let tr_vel_to = cvel_to.trace().max(0.0);

        // Stabilized ratio: tr_to / (tr_from + eps)
        let cov_vel_ratio = FeatureCore::safe_div(tr_vel_to, tr_vel_from + eps);

        // Store the scalar in the tuple struct.
        Self(cov_vel_ratio)
    }
}
