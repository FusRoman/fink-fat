// -----------------------------------------------------------------------------
// Edge velocity / kinematic consistency features
// -----------------------------------------------------------------------------
//
// This module defines `EdgeVelocityFeatures`, a feature family that focuses on
// kinematic compatibility between two seeds (`from -> to`) independently of
// absolute cadence or geometric scale.
//
// Conceptual goal
// ---------------
// Even if two detections are close in position, their *motions* must also be
// consistent to form a physically plausible link. These features compare:
// - the velocity predicted by propagating the `from` seed,
// - the velocity independently estimated at the `to` seed,
// - the velocity implied by the observed position mismatch.
//
// All quantities are normalized to be dimensionless and relatively invariant
// to cadence (time separation).
//
// Implementation note
// -------------------
// All heavy computations (propagation, innovation, covariance handling) are
// performed in `FeatureCore`. This module is a thin, stable projection of those
// values into a public feature struct.
//
// -----------------------------------------------------------------------------
//
// Dependencies:
// - `FeatureCore`: precomputed, sanitized scalar intermediates.
//

use crate::graph::edge::feature_core::FeatureCore;

/// Velocity/kinematic consistency features (dimensionless).
///
/// These features aim to measure whether the kinematics inferred from the
/// "from" seed and the "to" seed are compatible in a cadence-invariant way.
///
/// Attributes
/// ----------
/// * `cos_dtheta_v` – Directional alignment of velocities.
/// * `rel_speed_diff` – Relative mismatch in speed magnitudes.
/// * `innov_speed_ratio` – Speed implied by position mismatch vs predicted speed.
/// * `chi2_vel` – Velocity-space Mahalanobis distance.
/// * `log_chi2_vel` – Log-compressed version of `chi2_vel`.
#[derive(Clone, Debug)]
pub struct EdgeVelocityFeatures {
    /// Cosine of the angle between:
    /// - predicted velocity at the target epoch (propagated from `from`),
    /// - velocity estimated at `to`.
    ///
    /// Interpretation
    /// --------------
    /// - Values near `+1` indicate strongly aligned directions (good kinematic match).
    /// - Values near `0` indicate orthogonal motion (unlikely physical link).
    /// - Values near `-1` indicate opposite motion (almost certainly false link).
    ///
    /// Notes
    /// -----
    /// This metric is insensitive to the absolute speed scale and focuses purely
    /// on directional consistency.
    pub cos_dtheta_v: f64,

    /// Relative speed difference:
    /// `| |v_to| - |v_pred| | / (|v_to| + |v_pred| + eps)`.
    ///
    /// Interpretation
    /// --------------
    /// - `0` means identical speeds,
    /// - larger values indicate increasing mismatch.
    ///
    /// Cadence invariance
    /// ------------------
    /// Because both numerator and denominator scale with speed, this ratio is
    /// less sensitive to the actual time separation between seeds.
    pub rel_speed_diff: f64,

    /// Innovation-induced speed ratio:
    /// `( |r| / dt ) / |v_pred|`.
    ///
    /// Interpretation
    /// --------------
    /// - Values ≪ 1: position mismatch is small compared to expected motion.
    /// - Values ~ 1: mismatch is comparable to predicted motion.
    /// - Values ≫ 1: position mismatch implies an implausibly large speed.
    ///
    /// This feature connects *geometric inconsistency* with *kinematic scale*.
    pub innov_speed_ratio: f64,

    /// Velocity innovation Mahalanobis distance:
    /// `dvᵀ S_vel⁻¹ dv`, where `dv = v_to - v_pred`.
    ///
    /// Statistical intuition
    /// ---------------------
    /// This behaves like a χ² statistic with ~2 degrees of freedom when:
    /// - velocities are expressed in the same tangent-plane frame,
    /// - velocity covariance estimates are meaningful and comparable.
    ///
    /// Large values indicate kinematic incompatibility.
    pub chi2_vel: f64,

    /// `log(chi2_vel + eps)` for numerical stability and better dynamic range.
    ///
    /// As with position features, the log-transform:
    /// - compresses heavy tails,
    /// - improves robustness for ML models,
    /// - prevents extreme values from dominating training.
    pub log_chi2_vel: f64,
}

impl EdgeVelocityFeatures {
    /// Construct [`EdgeVelocityFeatures`] from a precomputed [`FeatureCore`].
    ///
    /// Arguments
    /// ---------
    /// * `core` – Shared intermediate computations produced by
    ///   [`FeatureCore::from_nodes`].
    ///
    /// Return
    /// ------
    /// Velocity/kinematic feature family for the edge that generated `core`.
    ///
    /// Notes
    /// -----
    /// This function performs a direct field mapping:
    /// - no recomputation,
    /// - no additional numerical guards,
    /// - all stability policy lives in `FeatureCore`.
    ///
    /// This keeps the feature definition transparent and easy to audit.
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
