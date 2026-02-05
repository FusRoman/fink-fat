use crate::{
    astro_math::trace_2x2, graph::edge::edge_features::FeatureCore, seeding::seed_node::SeedNode,
};

/// Uncertainty/quality ratios (dimensionless).
///
/// These features quantify how the uncertainty evolves between the two seeds,
/// and provide simple condition / anisotropy proxies for the local covariance.
/// Trace ratio of velocity covariance:
/// `tr(Cvel_to) / (tr(Cvel_from) + eps)`.
#[derive(Clone, Debug)]
pub struct EdgeUncertaintyFeatures(pub f64);

impl EdgeUncertaintyFeatures {
    #[inline]
    pub fn cov_vel_ratio(&self) -> f64 {
        self.0
    }

    /// Compute only uncertainty/quality ratio features.
    #[inline]
    pub fn uncertainty_features(from: &SeedNode, to: &SeedNode) -> Self {
        let eps = FeatureCore::EPS;

        let cvel_from = from.plane.cov_vel;
        let cvel_to = to.plane.cov_vel;
        let tr_vel_from = trace_2x2(cvel_from).max(0.0);
        let tr_vel_to = trace_2x2(cvel_to).max(0.0);
        let cov_vel_ratio = FeatureCore::safe_div(tr_vel_to, tr_vel_from + eps);

        Self(cov_vel_ratio)
    }
}
