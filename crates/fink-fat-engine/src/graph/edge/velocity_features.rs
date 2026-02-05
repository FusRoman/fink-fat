use crate::graph::edge::edge_features::FeatureCore;

/// Velocity/kinematic consistency features (dimensionless).
///
/// These features aim to measure whether the kinematics inferred from the
/// "from" seed and the "to" seed are compatible in a cadence-invariant way.
#[derive(Clone, Debug)]
pub struct EdgeVelocityFeatures {
    /// Cosine of the angle between:
    /// - predicted velocity at the target epoch (propagated from `from`),
    /// - velocity estimated at `to`.
    ///
    /// Values near:
    /// - `1` indicate aligned directions,
    /// - `0` indicate orthogonal motion,
    /// - `-1` indicate opposite motion.
    pub cos_dtheta_v: f64,

    /// Relative speed difference:
    /// `| |v_to| - |v_pred| | / (|v_to| + |v_pred| + eps)`.
    ///
    /// This is dimensionless and reduces sensitivity to cadence variations.
    pub rel_speed_diff: f64,

    /// Innovation-induced speed ratio:
    /// `( |r| / dt ) / |v_pred|`.
    ///
    /// Intuition:
    /// - `|r|/dt` is the *effective* velocity implied by the position mismatch,
    /// - dividing by `|v_pred|` normalizes by the expected motion scale.
    pub innov_speed_ratio: f64,

    /// Velocity innovation Mahalanobis distance:
    /// `dvᵀ S_vel⁻¹ dv`, where `dv = v_to - v_pred` and
    /// `S_vel = cov_vel_pred`.
    ///
    /// This behaves like a χ² statistic with ~2 degrees of freedom when:
    /// - velocities are expressed in the same tangent-plane frame,
    /// - covariance estimates are meaningful.
    pub chi2_vel: f64,

    /// `log(chi2_vel + eps)` for numerical stability and better dynamic range.
    pub log_chi2_vel: f64,
}

impl EdgeVelocityFeatures {
    /// Compute only velocity/kinematic features.
    #[inline]
    pub(crate) fn velocity_features(core: &FeatureCore) -> Self {
        Self {
            cos_dtheta_v: core.cos_dtheta_v,
            rel_speed_diff: core.rel_speed_diff,
            innov_speed_ratio: core.innov_speed_ratio,
            chi2_vel: core.chi2_vel,
            log_chi2_vel: core.log_chi2_vel,
        }
    }
}
