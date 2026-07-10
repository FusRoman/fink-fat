//! Log-likelihood-ratio (LLR) scoring for branch candidates.
//!
//! Every branch spawned when a bank's search region contains ambiguous
//! next-night candidates is scored against a "clutter" background instead of
//! in raw likelihood, so that pruning has a principled threshold ("more
//! likely than clutter?") rather than an arbitrary one. See
//! `kalman_update_instruction.md` for the full rationale.

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
    mixture_likelihood_z.min(MAX_MIXTURE_LIKELIHOOD).ln()
        - clutter_density.max(MIN_CLUTTER_DENSITY).ln()
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
    if p_detection >= 1.0 {
        return f64::NEG_INFINITY;
    }
    (1.0 - p_detection).ln()
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
