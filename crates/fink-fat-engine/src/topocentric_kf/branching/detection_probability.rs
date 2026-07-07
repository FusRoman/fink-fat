//! Absolute-magnitude-based detection probability for the null branch.
//!
//! The null branch's LLR contribution is `log(1 − P_D)` (see [`llr_score`]),
//! where `P_D` is the probability the survey would have detected this object
//! on this night. No photometric model (absolute magnitude `H`, phase-angle
//! term, survey depth) exists anywhere in the crate today, so this module
//! builds the minimal one needed to make `P_D` a function of real geometry
//! and photometry rather than a bare constant:
//!
//! 1. Every time an observation is associated to a track, its apparent
//!    magnitude plus the geometry at that epoch (heliocentric distance `r`,
//!    topocentric range `Δ`) implies an absolute magnitude
//!    ([`implied_absolute_magnitude`]), averaged into a running estimate
//!    ([`update_running_magnitude_estimate`]).
//! 2. At a future epoch, that running estimate plus the *predicted* `r`/`Δ`
//!    gives a predicted apparent magnitude ([`predicted_apparent_magnitude`]).
//! 3. Comparing the predicted magnitude to the survey's limiting magnitude
//!    through a logistic completeness curve gives `P_D`
//!    ([`detection_probability`]).
//!
//! [`llr_score`]: super::llr_score

/// Absolute magnitude `H` implied by one apparent-magnitude observation.
///
/// $$H = m - 5 \log_{10}(r \cdot \Delta)$$
///
/// The phase-angle (Sun–object–observer) term of the standard H-G
/// photometric law is deliberately omitted: no phase-angle geometry is
/// threaded through [`KFState`](crate::topocentric_kf::single_kalman::KFState)
/// today. This is a known simplification — refine with a real G-law term
/// once that geometry is available.
///
/// # Arguments
/// * `apparent_magnitude` – Observed apparent magnitude `m`.
/// * `r_helio_au` – Heliocentric distance of the object (AU).
/// * `delta_topocentric_au` – Topocentric range `Δ` (AU).
///
/// # Returns
/// The implied absolute magnitude `H`.
pub fn implied_absolute_magnitude(
    apparent_magnitude: f64,
    r_helio_au: f64,
    delta_topocentric_au: f64,
) -> f64 {
    apparent_magnitude - 5.0 * (r_helio_au * delta_topocentric_au).log10()
}

/// Fold one new implied-`H` sample into a running mean estimate.
///
/// A plain incremental mean is enough here: the point is to smooth out
/// per-observation photometric noise, not to track its variance.
///
/// # Arguments
/// * `previous_mean` – Running mean so far, `None` if this is the first sample.
/// * `previous_count` – Number of samples folded into `previous_mean`.
/// * `new_implied_h` – New sample from [`implied_absolute_magnitude`].
///
/// # Returns
/// `(updated_mean, updated_count)`.
pub fn update_running_magnitude_estimate(
    previous_mean: Option<f64>,
    previous_count: u32,
    new_implied_h: f64,
) -> (f64, u32) {
    let updated_count = previous_count + 1;
    let updated_mean = match previous_mean {
        Some(mean) => mean + (new_implied_h - mean) / f64::from(updated_count),
        None => new_implied_h,
    };
    (updated_mean, updated_count)
}

/// Predicted apparent magnitude at a future epoch, from a running `H`
/// estimate and the predicted geometry — the inverse of
/// [`implied_absolute_magnitude`].
///
/// # Arguments
/// * `absolute_magnitude_estimate` – Running `H` estimate (see
///   [`update_running_magnitude_estimate`]).
/// * `r_helio_au` – Predicted heliocentric distance (AU).
/// * `delta_topocentric_au` – Predicted topocentric range `Δ` (AU).
///
/// # Returns
/// The predicted apparent magnitude at the target epoch.
pub fn predicted_apparent_magnitude(
    absolute_magnitude_estimate: f64,
    r_helio_au: f64,
    delta_topocentric_au: f64,
) -> f64 {
    absolute_magnitude_estimate + 5.0 * (r_helio_au * delta_topocentric_au).log10()
}

/// Survey completeness at a predicted apparent magnitude — the detection
/// probability `P_D` consumed by the null branch's LLR term.
///
/// A logistic roll-off is used instead of a hard cutoff because real survey
/// completeness degrades smoothly around the limiting magnitude rather than
/// dropping instantly to zero:
///
/// $$P_D = \frac{1}{1 + \exp\!\left(\frac{m_{pred} - m_{lim}}{w}\right)}$$
///
/// # Arguments
/// * `predicted_magnitude` – Predicted apparent magnitude (see
///   [`predicted_apparent_magnitude`]).
/// * `limiting_magnitude` – Survey/field limiting magnitude `m_lim`.
/// * `completeness_width_mag` – Roll-off width `w` (mag), must be `> 0`;
///   typical surveys roll off over roughly 0.3–0.5 mag.
///
/// # Returns
/// `P_D` in `[0, 1]`.
pub fn detection_probability(
    predicted_magnitude: f64,
    limiting_magnitude: f64,
    completeness_width_mag: f64,
) -> f64 {
    debug_assert!(
        completeness_width_mag > 0.0,
        "completeness_width_mag must be strictly positive"
    );
    1.0 / (1.0 + ((predicted_magnitude - limiting_magnitude) / completeness_width_mag).exp())
}

#[cfg(test)]
mod detection_proba_tests {
    use super::*;

    #[test]
    fn implied_and_predicted_magnitude_round_trip() {
        let (r_helio_au, delta_au) = (2.3, 1.4);
        let apparent_magnitude = 19.5;

        let h = implied_absolute_magnitude(apparent_magnitude, r_helio_au, delta_au);
        let round_tripped = predicted_apparent_magnitude(h, r_helio_au, delta_au);

        assert!((round_tripped - apparent_magnitude).abs() < 1e-12);
    }

    #[test]
    fn running_magnitude_estimate_converges_to_sample_mean() {
        let samples = [18.0, 18.4, 17.8, 18.2];

        let (mean, count) = samples.iter().fold((None, 0u32), |(mean, count), &sample| {
            let (updated_mean, updated_count) =
                update_running_magnitude_estimate(mean, count, sample);
            (Some(updated_mean), updated_count)
        });

        let expected_mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert_eq!(count, samples.len() as u32);
        assert!((mean.unwrap() - expected_mean).abs() < 1e-12);
    }

    #[test]
    fn detection_probability_is_half_at_the_limiting_magnitude() {
        let p = detection_probability(21.5, 21.5, 0.4);
        assert!((p - 0.5).abs() < 1e-12);
    }

    #[test]
    fn detection_probability_decreases_monotonically_with_predicted_magnitude() {
        let limiting_magnitude = 21.5;
        let width = 0.4;

        let brighter = detection_probability(20.0, limiting_magnitude, width);
        let at_limit = detection_probability(21.5, limiting_magnitude, width);
        let fainter = detection_probability(23.0, limiting_magnitude, width);

        assert!(brighter > at_limit);
        assert!(at_limit > fainter);
    }
}
