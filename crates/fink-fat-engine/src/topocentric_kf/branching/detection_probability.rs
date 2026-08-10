//! Absolute-magnitude-based detection probability for the null branch.
//!
//! The null branch's LLR contribution is `log(1 − P_D)` (see [`llr_score`]),
//! where `P_D` is the probability the survey would have detected this object
//! on this night. This module holds the crate's photometric model — absolute
//! magnitude `H`, the H-G phase term, survey depth — which makes `P_D` a
//! function of real geometry and photometry rather than a bare constant:
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

use crate::logging::LogTarget;

/// Structured log events for the absolute-magnitude-based detection
/// probability estimate. See [`crate::logging`] for the `.emit()` pattern.
/// Deliberately `trace`-only: called once per null-branch candidate,
/// potentially thousands of times a night.
pub enum DetectionProbabilityEvent {
    Estimate {
        predicted_magnitude: f64,
        limiting_magnitude: f64,
        p_detection: f64,
    },
}

crate::impl_log_target!(
    DetectionProbabilityEvent,
    "detection_probability",
    "Absolute-magnitude-based detection probability (P_D) for the null branch",
    [tracing::Level::TRACE]
);

impl DetectionProbabilityEvent {
    pub fn emit(&self) {
        match self {
            DetectionProbabilityEvent::Estimate {
                predicted_magnitude,
                limiting_magnitude,
                p_detection,
            } => tracing::trace!(
                target: DetectionProbabilityEvent::TARGET, predicted_magnitude, limiting_magnitude, p_detection,
                "Detection probability estimate"
            ),
        }
    }
}

/// Default IAU slope parameter `G` of the H-G photometric system.
///
/// `0.15` is the standard value assumed for an asteroid whose slope has not
/// been measured — which is every object here, since `G` needs a densely
/// sampled phase curve to fit.
pub const DEFAULT_SLOPE_PARAMETER_G: f64 = 0.15;

/// Solar phase angle α: the Sun–object–observer angle, in radians.
///
/// Everything needed is already carried by a
/// [`KFState`](crate::topocentric_kf::single_kalman::KFState): the object's
/// heliocentric position (via `to_cartesian()`) and the observer's
/// (`r_obs`). The Sun sits at the origin of both, so from the object the
/// direction to the Sun is `−r_helio` and the direction to the observer is
/// `r_obs − r_helio`.
///
/// Returns `None` if either direction degenerates (object at the Sun, or
/// observer coincident with the object).
///
/// # Arguments
/// * `r_helio_au` – Object heliocentric position (AU).
/// * `r_obs_au` – Observer heliocentric position (AU).
pub fn solar_phase_angle(
    r_helio_au: &nalgebra::Vector3<f64>,
    r_obs_au: &nalgebra::Vector3<f64>,
) -> Option<f64> {
    let to_sun = -r_helio_au;
    let to_observer = r_obs_au - r_helio_au;

    // Positive-form guard so a NaN coordinate yields `None` rather than
    // reaching the division below.
    let (n_sun, n_obs) = (to_sun.norm(), to_observer.norm());
    if !(n_sun.is_finite() && n_sun > 0.0 && n_obs.is_finite() && n_obs > 0.0) {
        return None;
    }

    let cos_alpha = (to_sun.dot(&to_observer) / (n_sun * n_obs)).clamp(-1.0, 1.0);
    let alpha = cos_alpha.acos();
    alpha.is_finite().then_some(alpha)
}

/// Brightness lost to the phase angle under the IAU H-G law, in magnitudes.
///
/// $$-2.5 \log_{10}\!\left[(1-G)\,\Phi_1(\alpha) + G\,\Phi_2(\alpha)\right],
///   \qquad \Phi_i(\alpha) = \exp\!\left(-A_i \tan^{B_i}(\alpha/2)\right)$$
///
/// with $A_1 = 3.33, B_1 = 0.63, A_2 = 1.87, B_2 = 1.22$.
///
/// Always `>= 0`: an object is brightest at opposition (α = 0, where this
/// returns exactly 0) and fades as the phase angle opens. For main-belt
/// asteroids observed between 0 and 25°, the term spans roughly 0 to 0.8 mag
/// — which is precisely the systematic that, left unmodelled, forces a wide
/// tolerance on any comparison of absolute magnitudes across two arcs
/// observed at different geometries.
///
/// # Arguments
/// * `phase_angle_rad` – Solar phase angle α (radians), from
///   [`solar_phase_angle`].
/// * `slope_parameter_g` – The `G` of the H-G system; see
///   [`DEFAULT_SLOPE_PARAMETER_G`].
pub fn hg_phase_correction(phase_angle_rad: f64, slope_parameter_g: f64) -> f64 {
    const A1: f64 = 3.33;
    const B1: f64 = 0.63;
    const A2: f64 = 1.87;
    const B2: f64 = 1.22;

    // The law is defined on [0, π); tan(α/2) diverges as α → π, which no real
    // observing geometry reaches.
    let half_tan = (phase_angle_rad.clamp(0.0, std::f64::consts::PI - 1e-9) / 2.0).tan();
    if !half_tan.is_finite() || half_tan < 0.0 {
        return 0.0;
    }

    let phi1 = (-A1 * half_tan.powf(B1)).exp();
    let phi2 = (-A2 * half_tan.powf(B2)).exp();
    let g = slope_parameter_g.clamp(0.0, 1.0);
    let phi = (1.0 - g) * phi1 + g * phi2;

    if phi > 0.0 { -2.5 * phi.log10() } else { 0.0 }
}

/// H-G phase term for a filter state, in magnitudes — `0.0` when the geometry
/// degenerates or the correction is disabled.
///
/// One shared helper so the two transformations below cannot drift apart: the
/// inward one ([`implied_absolute_magnitude`]) and the outward one
/// ([`predicted_apparent_magnitude`]) must always see the same value for the
/// round trip to hold.
///
/// # Arguments
/// * `r_helio_au` – Object heliocentric position (AU), i.e. `kf.to_cartesian().pos`.
/// * `r_obs_au` – Observer heliocentric position (AU), i.e. `kf.r_obs`.
/// * `slope_parameter_g` – `None` disables the correction entirely; `Some(g)`
///   applies the H-G law with that slope (see [`DEFAULT_SLOPE_PARAMETER_G`]).
pub fn phase_correction(
    r_helio_au: &nalgebra::Vector3<f64>,
    r_obs_au: &nalgebra::Vector3<f64>,
    slope_parameter_g: Option<f64>,
) -> f64 {
    let Some(g) = slope_parameter_g else {
        return 0.0;
    };
    solar_phase_angle(r_helio_au, r_obs_au)
        .map(|alpha| hg_phase_correction(alpha, g))
        .unwrap_or(0.0)
}

/// Absolute magnitude `H` implied by one apparent-magnitude observation.
///
/// $$H = m - 5 \log_{10}(r \cdot \Delta) - \Phi(\alpha)$$
///
/// # The sign of the phase term
///
/// The H-G law reads `V = H + 5·log10(r·Δ) + Φ(α)` with
/// `Φ = hg_phase_correction(α, G) ≥ 0` — an object is *fainter* away from
/// opposition. Inverting it therefore **subtracts** `Φ`. Adding it instead
/// would double the systematic rather than remove it, and the result would
/// look like "the phase model made things worse", so the direction is pinned
/// by a round-trip test rather than left to the reader.
///
/// Without this term the returned `H` carries a systematic offset of up to a
/// few tenths of a magnitude that varies with observing geometry — harmless
/// while every consumer compares magnitudes measured at similar geometry, and
/// wrong as soon as one compares two arcs observed months apart, which is
/// exactly what fragment linkage does.
///
/// # Arguments
/// * `apparent_magnitude` – Observed apparent magnitude `m`.
/// * `r_helio_au` – Heliocentric distance of the object (AU).
/// * `delta_topocentric_au` – Topocentric range `Δ` (AU).
/// * `phase_correction_mag` – `Φ(α)` from [`phase_correction`]; `0.0` restores
///   the uncorrected behaviour exactly.
///
/// # Returns
/// The implied absolute magnitude `H`.
pub fn implied_absolute_magnitude(
    apparent_magnitude: f64,
    r_helio_au: f64,
    delta_topocentric_au: f64,
    phase_correction_mag: f64,
) -> f64 {
    apparent_magnitude - 5.0 * (r_helio_au * delta_topocentric_au).log10() - phase_correction_mag
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
/// The phase term is **added** here, being subtracted on the way in — the two
/// must always be fed the same `Φ` (see [`phase_correction`]), otherwise the
/// running `H` and the magnitudes predicted from it drift apart.
///
/// # Arguments
/// * `absolute_magnitude_estimate` – Running `H` estimate (see
///   [`update_running_magnitude_estimate`]).
/// * `r_helio_au` – Predicted heliocentric distance (AU).
/// * `delta_topocentric_au` – Predicted topocentric range `Δ` (AU).
/// * `phase_correction_mag` – `Φ(α)` at the predicted geometry; `0.0` restores
///   the uncorrected behaviour exactly.
///
/// # Returns
/// The predicted apparent magnitude at the target epoch.
pub fn predicted_apparent_magnitude(
    absolute_magnitude_estimate: f64,
    r_helio_au: f64,
    delta_topocentric_au: f64,
    phase_correction_mag: f64,
) -> f64 {
    absolute_magnitude_estimate
        + 5.0 * (r_helio_au * delta_topocentric_au).log10()
        + phase_correction_mag
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
    let p_detection =
        1.0 / (1.0 + ((predicted_magnitude - limiting_magnitude) / completeness_width_mag).exp());
    DetectionProbabilityEvent::Estimate {
        predicted_magnitude,
        limiting_magnitude,
        p_detection,
    }
    .emit();
    p_detection
}

#[cfg(test)]
mod detection_proba_tests {
    use super::*;

    use nalgebra::Vector3;

    /// Apparent magnitude an object of absolute magnitude `h` would show at a
    /// given geometry, straight from the H-G law — the forward direction, used
    /// as the oracle for the inverse below.
    fn apparent_from_hg(h: f64, r_helio_au: f64, delta_au: f64, phase_deg: f64) -> f64 {
        h + 5.0 * (r_helio_au * delta_au).log10()
            + hg_phase_correction(phase_deg.to_radians(), DEFAULT_SLOPE_PARAMETER_G)
    }

    /// The whole point of the phase term: the same object seen at different
    /// phase angles must yield the **same** absolute magnitude.
    ///
    /// This pins the sign. `H = m − 5·log10(r·Δ) − Φ` is the correct
    /// inversion; adding `Φ` instead would double the systematic rather than
    /// cancel it, and the module documentation asserted exactly that until
    /// this test was written. A geometry-dependent `H` is precisely what makes
    /// photometry useless for comparing two arcs observed months apart.
    #[test]
    fn phase_corrected_absolute_magnitude_is_geometry_independent() {
        let (h_true, r, delta) = (15.5, 2.7, 1.9);

        let recovered = |phase_deg: f64| {
            let m = apparent_from_hg(h_true, r, delta, phase_deg);
            let phi = hg_phase_correction(phase_deg.to_radians(), DEFAULT_SLOPE_PARAMETER_G);
            implied_absolute_magnitude(m, r, delta, phi)
        };

        for phase_deg in [0.0f64, 5.0, 12.0, 25.0] {
            let h = recovered(phase_deg);
            assert!(
                (h - h_true).abs() < 1e-9,
                "phase {phase_deg} deg recovered H={h}, expected {h_true}"
            );
        }
    }

    /// Without the correction, the recovered `H` drifts with geometry — the
    /// systematic this round exists to remove. Quantified rather than
    /// asserted, so the size of the bias is on record.
    #[test]
    fn uncorrected_absolute_magnitude_drifts_with_phase_angle() {
        let (h_true, r, delta) = (15.5, 2.7, 1.9);
        let at_opposition =
            implied_absolute_magnitude(apparent_from_hg(h_true, r, delta, 0.0), r, delta, 0.0);
        let at_25_deg =
            implied_absolute_magnitude(apparent_from_hg(h_true, r, delta, 25.0), r, delta, 0.0);

        let drift = at_25_deg - at_opposition;
        assert!(
            drift > 0.3,
            "expected a several-tenths systematic across 0-25 deg, got {drift}"
        );
    }

    /// The two transformations must invert each other exactly, whatever `Φ`
    /// is. If they ever drift apart, the running `H` and the magnitudes
    /// predicted from it stop being about the same object.
    #[test]
    fn implied_and_predicted_magnitudes_are_inverses() {
        let (r, delta) = (3.1, 2.2);
        for phi in [0.0f64, 0.2, 0.8] {
            let m = 20.25;
            let h = implied_absolute_magnitude(m, r, delta, phi);
            let back = predicted_apparent_magnitude(h, r, delta, phi);
            assert!((back - m).abs() < 1e-12, "phi={phi} broke the round trip");
        }
    }

    #[test]
    fn disabled_phase_correction_is_exactly_zero() {
        let object = Vector3::new(3.0, 0.5, 0.1);
        let observer = Vector3::new(1.0, 0.0, 0.0);
        assert_eq!(phase_correction(&object, &observer, None), 0.0);
        assert!(phase_correction(&object, &observer, Some(DEFAULT_SLOPE_PARAMETER_G)) > 0.0);
    }

    #[test]
    fn degenerate_geometry_yields_no_correction() {
        // Object at the Sun: the direction to the Sun is undefined.
        let at_sun = Vector3::zeros();
        let observer = Vector3::new(1.0, 0.0, 0.0);
        assert_eq!(
            phase_correction(&at_sun, &observer, Some(DEFAULT_SLOPE_PARAMETER_G)),
            0.0
        );
    }

    #[test]
    fn phase_correction_vanishes_at_opposition() {
        assert!(hg_phase_correction(0.0, DEFAULT_SLOPE_PARAMETER_G).abs() < 1e-12);
    }

    #[test]
    fn phase_correction_grows_with_phase_angle() {
        let g = DEFAULT_SLOPE_PARAMETER_G;
        let at = |deg: f64| hg_phase_correction(deg.to_radians(), g);
        assert!(at(0.0) < at(5.0));
        assert!(at(5.0) < at(15.0));
        assert!(at(15.0) < at(25.0));
    }

    #[test]
    fn phase_correction_is_a_dimming_of_plausible_size() {
        // Main-belt geometries: a few tenths of a magnitude, never negative.
        let g = DEFAULT_SLOPE_PARAMETER_G;
        for deg in [0.0f64, 5.0, 10.0, 20.0, 25.0] {
            let correction = hg_phase_correction(deg.to_radians(), g);
            assert!(correction >= 0.0, "phase term must never brighten");
            assert!(
                correction < 1.5,
                "{deg} deg gave an implausible {correction} mag"
            );
        }
    }

    #[test]
    fn phase_angle_is_zero_at_exact_opposition() {
        // Observer directly between Sun and object: Sun and observer lie in
        // the same direction as seen from the object.
        let object = Vector3::new(3.0, 0.0, 0.0);
        let observer = Vector3::new(1.0, 0.0, 0.0);
        let alpha = solar_phase_angle(&object, &observer).expect("well-posed geometry");
        assert!(alpha.abs() < 1e-12);
    }

    #[test]
    fn phase_angle_is_right_angle_for_quadrature() {
        // Object on the x axis, observer offset perpendicular by the same
        // distance as the object's heliocentric range: a 45 deg phase angle.
        let object = Vector3::new(1.0, 0.0, 0.0);
        let observer = Vector3::new(0.0, 0.0, 0.0);
        // Observer at the Sun => direction to observer == direction to Sun.
        assert!(solar_phase_angle(&object, &observer).expect("valid").abs() < 1e-12);

        let observer = Vector3::new(1.0, 1.0, 0.0);
        let alpha = solar_phase_angle(&object, &observer).expect("valid");
        assert!(
            (alpha - std::f64::consts::FRAC_PI_2).abs() < 1e-12,
            "expected 90 deg, got {} deg",
            alpha.to_degrees()
        );
    }

    #[test]
    fn phase_angle_rejects_degenerate_geometry() {
        let origin = Vector3::zeros();
        assert!(solar_phase_angle(&origin, &Vector3::new(1.0, 0.0, 0.0)).is_none());
        let object = Vector3::new(2.0, 0.0, 0.0);
        assert!(solar_phase_angle(&object, &object).is_none());
    }

    #[test]
    fn implied_and_predicted_magnitude_round_trip() {
        let (r_helio_au, delta_au) = (2.3, 1.4);
        let apparent_magnitude = 19.5;

        let h = implied_absolute_magnitude(apparent_magnitude, r_helio_au, delta_au, 0.0);
        let round_tripped = predicted_apparent_magnitude(h, r_helio_au, delta_au, 0.0);

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
