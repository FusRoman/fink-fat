//! Log-likelihood-ratio (LLR) scoring for branch candidates.
//!
//! Every branch spawned when a bank's search region contains ambiguous
//! next-night candidates is scored against a "clutter" background instead of
//! in raw likelihood, so that pruning has a principled threshold ("more
//! likely than clutter?") rather than an arbitrary one. See
//! `kalman_update_instruction.md` for the full rationale.

use crate::logging::LogTarget;

/// Structured log events for LLR scoring against the clutter background.
/// See [`crate::logging`] for the `.emit()` pattern. Deliberately
/// `trace`-only: called once per candidate/branch, potentially thousands of
/// times a night.
pub enum LlrScoreEvent {
    ObservationDelta {
        mixture_likelihood_z: f64,
        clutter_density: f64,
        delta: f64,
    },
    NullDelta {
        p_detection: f64,
        delta: f64,
    },
    PhotometricDelta {
        predicted_magnitude: Option<f64>,
        observed_magnitude: f64,
        delta: f64,
    },
}

crate::impl_log_target!(
    LlrScoreEvent,
    "llr_score",
    "Log-likelihood-ratio scoring of branch candidates against a clutter background",
    [tracing::Level::TRACE]
);

impl LlrScoreEvent {
    pub fn emit(&self) {
        use LlrScoreEvent::*;
        match self {
            ObservationDelta {
                mixture_likelihood_z,
                clutter_density,
                delta,
            } => tracing::trace!(
                target: LlrScoreEvent::TARGET, mixture_likelihood_z, clutter_density, delta, "Observation LLR delta"
            ),
            NullDelta { p_detection, delta } => tracing::trace!(
                target: LlrScoreEvent::TARGET, p_detection, delta, "Null-branch LLR delta"
            ),
            PhotometricDelta {
                predicted_magnitude,
                observed_magnitude,
                delta,
            } => tracing::trace!(
                target: LlrScoreEvent::TARGET, predicted_magnitude, observed_magnitude, delta, "Photometric LLR delta"
            ),
        }
    }
}

/// Floor applied to a clutter density before taking its logarithm, to avoid
/// `+inf` scores in emptied-out fields with no nearby alerts at all.
const MIN_CLUTTER_DENSITY: f64 = 1e-12;

/// Ceiling applied to a mixture likelihood before taking its logarithm, to
/// avoid a `+inf` delta (which can cancel a `-inf` null-branch delta
/// elsewhere in `cumulative_llr` into `NaN`) if a hypothesis's innovation
/// covariance becomes numerically near-singular.
const MAX_MIXTURE_LIKELIHOOD: f64 = 1e300;

/// LLR contribution of associating a candidate observation to a bank:
///
/// $$\log L(z) - \log \lambda_{clutter}$$
///
/// # Arguments
/// * `mixture_likelihood_z` – Predictive mixture likelihood `L(z)` of the
///   candidate observation under the bank's hypotheses, e.g. from
///   [`KFBank::branch_with`](crate::topocentric_kf::kalman_bank::KFBank::branch_with).
/// * `clutter_density` – Local alert density `λ_clutter` (alerts per
///   steradian), e.g. from
///   [`local_clutter_density`](crate::spacetime_bucket::clutter_density::local_clutter_density).
///
/// # Returns
/// The signed LLR delta: positive means the association is more plausible
/// than clutter, negative means clutter is the better explanation.
pub fn observation_llr_delta(mixture_likelihood_z: f64, clutter_density: f64) -> f64 {
    let delta = mixture_likelihood_z.min(MAX_MIXTURE_LIKELIHOOD).ln()
        - clutter_density.max(MIN_CLUTTER_DENSITY).ln();
    LlrScoreEvent::ObservationDelta {
        mixture_likelihood_z,
        clutter_density,
        delta,
    }
    .emit();
    delta
}

/// LLR contribution of comparing a candidate's apparent magnitude to the
/// bank's magnitude-implied prediction — an *additional* term alongside
/// [`observation_llr_delta`]'s astrometric one, not a replacement for it.
///
/// A Gaussian residual penalty, `-0.5 * ((m_obs - m_pred) / sigma_mag)^2`.
/// The Gaussian normalization constant (`-ln(sigma_mag) - 0.5 ln(2π)`) is
/// deliberately omitted: `sigma_mag` is the same for every candidate in a
/// visit, so the constant cancels in every branch comparison that actually
/// matters (`apply_n_scan_pruning`, `cap_top_b_per_lineage`) — the same
/// pragmatic convention [`observation_llr_delta`] already uses by not
/// normalizing its own clutter term.
///
/// # Arguments
/// * `predicted_magnitude` – The bank's predicted apparent magnitude for
///   this candidate's epoch (see
///   [`predicted_apparent_magnitude`](super::detection_probability::predicted_apparent_magnitude)),
///   or `None` if the bank has no magnitude history yet.
/// * `observed_magnitude` – The candidate observation's apparent magnitude.
/// * `sigma_mag` – Assumed 1-sigma spread (mag) of the residual, e.g.
///   `NightAdvanceParams::photometric_sigma_mag`.
///
/// # Returns
/// `0.0` (neutral — no penalty, no bonus) when `predicted_magnitude` is
/// `None`: a lineage with a single observation so far has no photometric
/// prediction to be penalized against, mirroring
/// `null_branch_detection_probability`'s own no-history fallback.
pub fn photometric_llr_delta(
    predicted_magnitude: Option<f64>,
    observed_magnitude: f64,
    sigma_mag: f64,
) -> f64 {
    let delta = match predicted_magnitude {
        Some(predicted_magnitude) => {
            let residual = observed_magnitude - predicted_magnitude;
            -0.5 * (residual / sigma_mag).powi(2)
        }
        None => 0.0,
    };
    LlrScoreEvent::PhotometricDelta {
        predicted_magnitude,
        observed_magnitude,
        delta,
    }
    .emit();
    delta
}

/// LLR contribution of the null (missed-detection) branch: `log(1 − P_D)`.
///
/// # Arguments
/// * `p_detection` – Detection probability `P_D`, e.g. from
///   [`detection_probability`](super::detection_probability::detection_probability).
///
/// # Returns
/// `ln(1 - p_detection)`, or `f64::NEG_INFINITY` when `p_detection >= 1.0`
/// (a certain detection makes the null branch non-viable — correct MHT
/// behavior: an object that was surely seen cannot also have gone
/// undetected).
pub fn null_branch_llr_delta(p_detection: f64) -> f64 {
    let delta = if p_detection >= 1.0 {
        f64::NEG_INFINITY
    } else {
        (1.0 - p_detection).ln()
    };
    LlrScoreEvent::NullDelta { p_detection, delta }.emit();
    delta
}

#[cfg(test)]
mod ll_score_tests {
    use super::*;

    #[test]
    fn observation_llr_delta_is_positive_when_likelihood_exceeds_clutter() {
        let delta = observation_llr_delta(10.0, 1.0);
        assert!(delta > 0.0);
    }

    #[test]
    fn observation_llr_delta_is_negative_when_clutter_exceeds_likelihood() {
        let delta = observation_llr_delta(1.0, 10.0);
        assert!(delta < 0.0);
    }

    #[test]
    fn observation_llr_delta_is_zero_at_equality() {
        let delta = observation_llr_delta(4.0, 4.0);
        assert!(delta.abs() < 1e-12);
    }

    #[test]
    fn observation_llr_delta_floors_clutter_density_to_avoid_infinity() {
        let delta = observation_llr_delta(1.0, 0.0);
        assert!(delta.is_finite());
    }

    #[test]
    fn observation_llr_delta_caps_mixture_likelihood_to_avoid_infinity() {
        let delta = observation_llr_delta(f64::INFINITY, 1.0);
        assert!(delta.is_finite());
        let delta = observation_llr_delta(1e308, 1.0);
        assert!(delta.is_finite());
    }

    #[test]
    fn photometric_llr_delta_is_zero_without_history() {
        assert_eq!(photometric_llr_delta(None, 18.5, 0.35), 0.0);
    }

    #[test]
    fn photometric_llr_delta_is_zero_at_exact_match() {
        let delta = photometric_llr_delta(Some(18.5), 18.5, 0.35);
        assert!(delta.abs() < 1e-12);
    }

    #[test]
    fn photometric_llr_delta_is_negative_and_grows_with_residual() {
        let small_residual = photometric_llr_delta(Some(18.5), 18.6, 0.35);
        let large_residual = photometric_llr_delta(Some(18.5), 19.5, 0.35);
        assert!(small_residual < 0.0);
        assert!(large_residual < small_residual);
    }

    #[test]
    fn null_branch_llr_delta_matches_ln_one_minus_p() {
        let p_detection: f64 = 0.3;
        let expected = (1.0 - p_detection).ln();
        assert!((null_branch_llr_delta(p_detection) - expected).abs() < 1e-12);
    }

    #[test]
    fn null_branch_llr_delta_is_negative_infinity_at_certain_detection() {
        assert_eq!(null_branch_llr_delta(1.0), f64::NEG_INFINITY);
    }
}
