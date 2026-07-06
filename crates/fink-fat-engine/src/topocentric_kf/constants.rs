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
