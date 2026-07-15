/// Speed of light in astronomical units per day.
///
/// Consistent with the IAU 2009 / DE440 value (`c = 299_792.458 km/s`,
/// `1 AU = 149_597_870.7 km`, `1 day = 86_400 s`). Used to convert the
/// topocentric range `ρ` into a light-time delay `τ = ρ / c`.
pub const C_AU_PER_DAY: f64 = 173.144_632_674_240_57;

/// χ²(2) upper 95 % quantile — the consistency reference for NIS-driven
/// covariance inflation.
///
/// Used as a **dead-zone** threshold: as long as the smoothed NIS stays below
/// this value the filter is deemed statistically consistent and its covariance
/// is transported unchanged (`λ = 1`). Inflation only engages on genuine
/// inconsistency, so a well-behaved filter is never perturbed.
pub const CHI2_2DOF_95: f64 = 5.991;

/// Maximum per-step covariance inflation factor.
///
/// Caps how aggressively a single propagation may re-open the covariance. A
/// catastrophic NIS (e.g. 10³) would otherwise inflate `P` by a huge factor in
/// one step (an outlier over-reaction); clamping to `5×` per step spreads the
/// recovery over a few predictions, keeping the transport smooth while still
/// converging quickly back into the consistency dead-zone.
pub const MAX_INFLATION: f64 = 5.0;

/// Sane lower bound on a hypothesis's topocentric range estimate ρ (AU),
/// used to reject covariance propagation before `jacobian_attr_to_cart`
/// becomes numerically ill-conditioned.
///
/// Four of that Jacobian's six columns scale linearly with ρ, so as ρ
/// drifts toward 0 the matrix approaches rank-2 without ever being
/// *exactly* singular — `try_inverse()` still succeeds but returns hugely
/// amplified entries, silently inflating the propagated covariance by many
/// orders of magnitude. Wide guard-rail, not a tight physical bound: SSOs
/// in a typical survey's tracked population range roughly 0.5-10 AU, so
/// `1e-4` AU is already many orders of magnitude below any real target and
/// only ever triggers on numerically drifted hypotheses.
pub const MIN_RHO_AU: f64 = 1e-4;

/// Sane upper bound on ρ (AU) — same rationale as [`MIN_RHO_AU`], the other
/// side of the same ill-conditioning mechanism.
pub const MAX_RHO_AU: f64 = 1e6;

/// Ceiling on `det(S)` (rad⁴) before taking its logarithm — mirrors
/// `llr_score::MAX_MIXTURE_LIKELIHOOD`'s rationale: an extreme determinant
/// (from a near-singular/ill-conditioned innovation covariance, e.g. a
/// long-surviving lineage's drifting range estimate — see
/// `single_kalman::propagate::PropagateError::IllConditionedRange`) must
/// not produce an unbounded `log_weight`, which would otherwise corrupt
/// downstream `KFBank::effective_sample_size()`. `1e20` rad⁴ is already
/// many orders of magnitude above a healthy covariance (~1e-18 rad⁴ at
/// arcsec scale), so this only ever engages on already-pathological
/// hypotheses.
pub const MAX_INNOVATION_DET: f64 = 1e20;
