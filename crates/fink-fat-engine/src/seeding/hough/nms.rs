//! Non-maximum suppression for Hough peaks.
//!
//! This module defines the suppression layer used after Hough peak extraction.
//! Its role is to remove peaks that are effectively redundant because they are
//! both close in Hough parameter space and supported by nearly the same set of
//! alerts.
//!
//! The suppression criterion combines two complementary signals:
//!
//! - geometric proximity in the discrete Hough grid, through
//!   [`AccKey::hough_bins_are_close`](crate::seeding::hough::accumulator::AccKey::hough_bins_are_close),
//! - strong alert-membership overlap, quantified with Jaccard index and a
//!   symmetric containment score.
//!
//! The resulting behavior is a greedy, score-ordered non-maximum suppression
//! pass. It preserves the highest-ranked candidate in a local neighborhood and
//! suppresses later candidates that are too similar to an already accepted one.
//!
//! ## Main items
//!
//! - [`NMS_MAX_BIN_OFFSET`] controls the geometric neighborhood in Hough space.
//! - [`NMS_MIN_JACCARD`] and [`NMS_MIN_CONTAINMENT`] define the overlap test.
//! - [`overlap_metrics`] computes the overlap scores for two memberships.
//! - [`apply_nms`] filters a ranked list of peak candidates.

use crate::seeding::hough::peaks::PeakCandidate;

/// Maximum absolute offset allowed on each Hough-bin axis when comparing peaks.
///
/// Two peaks are considered geometrically adjacent only if the difference in
/// `vel_ix`, `vel_iy`, `alpha_bin`, and `delta_bin` is at most this value.
pub const NMS_MAX_BIN_OFFSET: i32 = 1;

/// Minimum Jaccard index required to consider two peaks near-duplicate.
///
/// For two memberships $A$ and $B$, the Jaccard index is
/// $J = |A \cap B| / |A \cup B|$.
pub const NMS_MIN_JACCARD: f64 = 0.8;

/// Minimum symmetric containment required to consider two peaks near-duplicate.
///
/// Containment is defined as
/// $$\max\left(\frac{|A \cap B|}{|A|}, \frac{|A \cap B|}{|B|}\right)$$
/// for two memberships $A$ and $B$.
pub const NMS_MIN_CONTAINMENT: f64 = 0.9;

/// Compute Jaccard index and symmetric containment for two sorted unique index lists.
///
/// The inputs must be sorted in ascending order and contain no duplicates.
/// Under that contract, the function computes the intersection size in linear
/// time without allocating temporary sets.
///
/// Arguments
/// ---------
/// * `a` - First sorted unique membership list.
/// * `b` - Second sorted unique membership list.
///
/// Return
/// ------
/// * `(f64, f64)` - `(jaccard, containment)` for the two memberships.
pub fn overlap_metrics(a: &[usize], b: &[usize]) -> (f64, f64) {
    if a.is_empty() || b.is_empty() {
        return (0.0, 0.0);
    }

    let mut i = 0usize;
    let mut j = 0usize;
    let mut inter = 0usize;

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                inter += 1;
                i += 1;
                j += 1;
            }
        }
    }

    if inter == 0 {
        return (0.0, 0.0);
    }

    let union = a.len() + b.len() - inter;
    let jaccard = inter as f64 / union as f64;
    let containment = (inter as f64 / a.len() as f64).max(inter as f64 / b.len() as f64);
    (jaccard, containment)
}

/// Suppress near-duplicate peaks using Hough proximity and alert overlap.
///
/// The input is expected to be ordered from strongest to weakest candidate.
/// The algorithm is greedy: each new peak is compared against the already kept
/// peaks, and it is discarded as soon as it is judged redundant with any prior
/// winner.
///
/// A candidate is suppressed only if both conditions are satisfied:
///
/// - the two peaks are geometrically close in Hough space,
/// - their alert memberships overlap strongly according to the thresholds in
///   this module.
///
/// Arguments
/// ---------
/// * `peaks` - Ranked peak candidates, typically sorted by decreasing score.
///
/// Return
/// ------
/// * `Vec<PeakCandidate>` - Peaks that survived the greedy NMS pass, in input order.
pub fn apply_nms(peaks: Vec<PeakCandidate>) -> Vec<PeakCandidate> {
    let mut kept: Vec<PeakCandidate> = Vec::with_capacity(peaks.len());

    'candidate: for candidate in peaks {
        for winner in &kept {
            if candidate.key.hough_bins_are_close(&winner.key)
                && candidate.strong_membership_overlap(winner)
            {
                continue 'candidate;
            }
        }
        kept.push(candidate);
    }

    kept
}
