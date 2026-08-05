//! Post-gate candidate-filter cascade, applied to the candidates found in a
//! bank's predicted search region *before* any branch is spawned.
//!
//! [`candidate_search::filter_candidates`](super::candidate_search)'s gate
//! (per-component Mahalanobis + mixture-likelihood threshold) is purely
//! astrometric and deliberately generous — see `NightAdvanceParams`'
//! `gate_chi2`/`likelihood_threshold` docs. In real ZTF-cadence fields that
//! leaves genuine contamination: multiple unrelated alerts inside one
//! lineage's error box, each of which would otherwise spawn its own branch.
//!
//! Every stage below was validated offline (`fink-fat-eval`'s
//! `candidate_filters` shadow study, full-dataset run) to remove a large
//! fraction of that contamination for a negligible true-observation loss —
//! see each [`NightAdvanceParams`] field's doc for the measured numbers.
//! Each stage is a no-op when its config field is `0.0`/`0` ("disabled" —
//! the same sentinel convention `NightAdvanceParams` already uses
//! elsewhere), so this module changes nothing for any deployment that
//! hasn't opted in.

use nalgebra::{Matrix2, Vector2};

use crate::engine_config::night_advance_params::NightAdvanceParams;
use crate::topocentric_kf::branching::candidate_search::CandidateMatch;
use crate::topocentric_kf::branching::llr_score::{observation_llr_delta, photometric_llr_delta};
use crate::topocentric_kf::kalman_bank::KFBank;
use crate::topocentric_kf::kalman_bank::ellipse_region_finder::SearchRegion;
use crate::topocentric_kf::single_kalman::update::wrap_angle;

/// Run every enabled stage of the candidate-filter cascade, in order:
/// photometric → cross-track → relative-likelihood → top-K rank-and-cap.
///
/// # Arguments
/// * `region` – The lineage's predicted search region this visit (for the
///   cross-track gate's center/covariances).
/// * `predicted_bank` – The bank already predicted to this visit's epoch
///   (for the cross-track gate's motion direction).
/// * `predicted_magnitude` – The bank's predicted apparent magnitude, or
///   `None` with no photometric history yet — see
///   [`super::orchestrate::predicted_apparent_magnitude_for_bank`].
/// * `clutter_density` – Local alert density at the region's center (alerts
///   per steradian), used only by the top-K stage's ranking score — see
///   [`crate::spacetime_bucket::clutter_density::local_clutter_density`].
///   Deliberately a single per-visit estimate at the region center rather
///   than a per-candidate one, matching what was validated offline.
/// * `candidates` – Candidates that already passed the astrometric gate.
/// * `params` – Cascade thresholds, plus `photometric_sigma_mag` (shared
///   with the downstream branch LLR scoring) for the top-K stage's score.
pub fn apply_candidate_filters<'obs>(
    region: &SearchRegion,
    predicted_bank: &KFBank<'_, '_>,
    predicted_magnitude: Option<f64>,
    clutter_density: f64,
    mut candidates: Vec<CandidateMatch<'obs>>,
    params: &NightAdvanceParams,
) -> Vec<CandidateMatch<'obs>> {
    if params.candidate_photometric_max_delta_mag > 0.0
        && let Some(predicted_magnitude) = predicted_magnitude
    {
        let max_delta = params.candidate_photometric_max_delta_mag;
        candidates.retain(|c| {
            (c.observation.photometry().magnitude - predicted_magnitude).abs() <= max_delta
        });
    }

    if params.candidate_cross_track_k_sigma > 0.0
        && let Some((ux, uy, sigma_cross)) = cross_track_direction_and_sigma(region, predicted_bank)
    {
        let limit = (params.candidate_cross_track_k_sigma * sigma_cross)
            .max(params.candidate_cross_track_floor_arcsec);
        let cos_dec = region.center_dec.cos();
        candidates.retain(|c| {
            let coord = c.observation.equ_coord();
            let dx = wrap_angle(coord.ra - region.center_ra) * cos_dec * RAD_TO_ARCSEC;
            let dy = (coord.dec - region.center_dec) * RAD_TO_ARCSEC;
            let cross = -dx * uy + dy * ux;
            cross.abs() <= limit
        });
    }

    if params.candidate_rel_likelihood_alpha > 0.0 {
        // `f64::max` ignores NaN, so `l_max` is 0.0 (not NaN) on an empty or
        // degenerate likelihood set — the guard below then keeps everything.
        let l_max = candidates.iter().map(|c| c.likelihood).fold(0.0, f64::max);
        if l_max > 0.0 {
            let threshold = params.candidate_rel_likelihood_alpha * l_max;
            candidates.retain(|c| c.likelihood >= threshold);
        }
    }

    if params.max_candidates_per_visit > 0 && candidates.len() > params.max_candidates_per_visit {
        candidates.sort_by(|a, b| {
            let score_of = |c: &CandidateMatch<'obs>| {
                observation_llr_delta(c.likelihood, clutter_density)
                    + photometric_llr_delta(
                        predicted_magnitude,
                        c.observation.photometry().magnitude,
                        params.photometric_sigma_mag,
                    )
            };
            score_of(b).total_cmp(&score_of(a))
        });
        candidates.truncate(params.max_candidates_per_visit);
    }

    candidates
}

const RAD_TO_ARCSEC: f64 = 206264.80624709636;

/// Tangent-plane unit direction of the bank's weighted-mean apparent motion
/// `(ux, uy)`, plus the predicted mixture's 1-sigma cross-track extent
/// (arcsec) — the shared inputs of the cross-track gate.
///
/// `None` when the mixture carries no weight, its motion direction is
/// degenerate (zero norm — e.g. right at bootstrap, before any rate is
/// constrained), or `region` has no components to measure a spread from.
fn cross_track_direction_and_sigma(
    region: &SearchRegion,
    predicted_bank: &KFBank<'_, '_>,
) -> Option<(f64, f64, f64)> {
    let hypotheses = predicted_bank.hypotheses();
    let weights: Vec<f64> = hypotheses.iter().map(|h| h.log_weight.exp()).collect();
    let total_weight: f64 = weights.iter().sum();
    // Positive-form guard (rather than negating `total_weight > 0.0`) so a
    // NaN total weight falls through to `None` here instead of silently
    // reaching the arithmetic below.
    if !total_weight.is_finite() || total_weight <= 0.0 || region.components.is_empty() {
        return None;
    }

    let (mut ra_dot, mut dec_dot, mut dec) = (0.0, 0.0, 0.0);
    for (w, h) in weights.iter().zip(hypotheses) {
        ra_dot += w * h.kf.state[2];
        dec_dot += w * h.kf.state[3];
        dec += w * h.kf.state[1];
    }
    let (ra_dot, dec_dot, mean_dec) = (
        ra_dot / total_weight,
        dec_dot / total_weight,
        dec / total_weight,
    );
    let vx = ra_dot * mean_dec.cos();
    let vy = dec_dot;
    let speed = (vx * vx + vy * vy).sqrt();
    if !speed.is_finite() || speed <= 0.0 {
        return None;
    }
    let (ux, uy) = (vx / speed, vy / speed);

    // Cross-track projection of a tangent-plane displacement is
    // `-dx·uy + dy·ux` with `dx = ΔRA·cosδ, dy = ΔDec`; applied to a *raw*
    // (ΔRA, ΔDec) covariance that is the row vector `a = (-uy·cosδ, ux)`.
    let cos_dec = region.center_dec.cos();
    let a = Vector2::new(-uy * cos_dec, ux);

    let mut comp_total_weight = 0.0;
    let mut var = 0.0;
    for c in &region.components {
        let s: &Matrix2<f64> = &c.s;
        let comp_var = (a.transpose() * s * a)[(0, 0)];
        let cx = wrap_angle(c.center_ra - region.center_ra) * cos_dec;
        let cy = c.center_dec - region.center_dec;
        let comp_cross = -cx * uy + cy * ux;
        var += c.weight * (comp_var + comp_cross * comp_cross);
        comp_total_weight += c.weight;
    }
    if !comp_total_weight.is_finite() || comp_total_weight <= 0.0 || !var.is_finite() || var <= 0.0
    {
        return None;
    }
    let sigma_cross_arcsec = (var / comp_total_weight).sqrt() * RAD_TO_ARCSEC;
    Some((ux, uy, sigma_cross_arcsec))
}
