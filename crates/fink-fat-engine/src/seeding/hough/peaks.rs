//! Peak handling for Hough-transform seeding.
//!
//! This module contains the post-accumulation stage of the Hough seeding
//! pipeline. It converts sparse accumulator cells into ranked peak candidates,
//! removes photometrically inconsistent or near-duplicate peaks, and finally
//! builds [`SeedNode`] values for downstream orbit-linking.
//!
//! The operations are intentionally decomposed into small, composable steps so
//! that each stage of the peak lifecycle remains mechanically interpretable:
//!
//! - [`PeakCandidate`] stores the local Hough-bin key, vote score, and member
//!   alert indices for one accumulator maximum.
//! - [`PeakCandidate::extract_ranked_peaks`] converts sparse accumulator cells
//!   into ranked candidates ordered by vote weight.
//! - [`apply_photometric_filter`] removes peaks whose member alerts are
//!   photometrically incompatible.
//! - [`PeakCandidate::strong_membership_overlap`] and
//!   [`cap_peaks_per_alert`] implement the modular suppression logic used to
//!   control duplicate seeds and alert over-representation.
//! - [`PeakCandidate::build_seed_from_peak`] transforms one surviving peak into
//!   a seed candidate.
//!
//! ## Scientific role
//!
//! A Hough accumulator peak corresponds to a set of alerts that become
//! spatially coherent after back-projection under one angular-velocity
//! hypothesis. The peak ranking therefore approximates a discrete maximum-
//! likelihood search over a velocity-position parameterization, followed by
//! a suppression stage that merges peaks that are nearly equivalent both in
//! Hough space and in alert membership.
//!
//! Near-duplicate suppression uses two complementary criteria:
//!
//! - geometric proximity in Hough parameter space $(v\_\alpha, v\_\delta,
//!   \alpha\_{\mathrm{bin}}, \delta\_{\mathrm{bin}})$,
//! - strong membership overlap, quantified through Jaccard index and symmetric
//!   containment.
//!
//! The resulting peaks bridge the discrete Hough search and the geometric seed
//! objects consumed by downstream orbit-linking stages.

use ahash::AHashMap;

use crate::{
    Alert,
    engine_config::seeding_config::HoughSeedingConfig,
    night_id::NightId,
    seeding::{
        SeedNode,
        hough::{
            accumulator::{AccKey, AccumulatorCell},
            nms::{NMS_MIN_CONTAINMENT, NMS_MIN_JACCARD, overlap_metrics},
        },
        store::SeedStore,
    },
};

/// Peak retained from the sparse Hough accumulator.
///
/// The `key` identifies the accumulator cell that produced the peak, `score`
/// stores the accumulated vote weight, and `alert_indices` stores the unique
/// member-alert indices in the order used by downstream seed construction.
#[derive(Clone, Debug)]
pub struct PeakCandidate {
    /// Hough-space bin key associated with this peak.
    pub key: AccKey,
    /// Total vote weight accumulated by the peak.
    pub score: f64,
    /// Alert indices associated with the peak.
    pub alert_indices: Vec<usize>,
}

impl PeakCandidate {
    /// Collect and time-sort the alerts supporting one peak.
    ///
    /// This resolves the integer indices stored in [`PeakCandidate::alert_indices`]
    /// into references to the corresponding alerts and orders them by
    /// `mjd_tt`.
    ///
    /// Arguments
    /// ---------
    /// * `alerts` - Full night-level alert slice used as the lookup table.
    ///
    /// Return
    /// ------
    /// * `Vec<&Alert>` - Time-ordered alert references supporting this peak.
    pub fn collect_peak_alerts_sorted<'a>(&self, alerts: &'a [Alert]) -> Vec<&'a Alert> {
        let mut peak_alerts: Vec<&Alert> = self
            .alert_indices
            .iter()
            .filter_map(|&idx| alerts.get(idx))
            .collect();
        peak_alerts.sort_by(|a, b| a.mjd_tt.total_cmp(&b.mjd_tt));
        peak_alerts
    }

    /// Convert one filtered peak into a seed if it satisfies emission constraints.
    ///
    /// Peaks with at least three supporting alerts produce a triplet-derived
    /// seed fitted on the first, middle, and last alerts in time order. The full
    /// peak membership is passed to the seed constructor so the stored
    /// membership remains complete. Peaks with exactly two alerts can emit a
    /// pair seed when `triplet_only == false`.
    ///
    /// Arguments
    /// ---------
    /// * `alerts` - Full night-level alert slice used to resolve indices.
    /// * `night_id` - Night identifier assigned to the resulting seed.
    /// * `triplet_only` - If `true`, suppress pair-seed emission from two-alert
    ///   peaks.
    /// * `local_store` - Thread-local seed store used to allocate provisional
    ///   keys.
    ///
    /// Return
    /// ------
    /// * `Some(SeedNode)` - Seed built from this peak.
    /// * `None` - The peak does not satisfy emission constraints, for example
    ///   because it has too few alerts or the three anchor detections are not
    ///   distinct.
    pub fn build_seed_from_peak(
        &self,
        alerts: &[Alert],
        night_id: NightId,
        triplet_only: bool,
        local_store: &mut SeedStore,
    ) -> Option<SeedNode> {
        let peak_alerts = self.collect_peak_alerts_sorted(alerts);

        if peak_alerts.len() >= 3 {
            let a = peak_alerts[0];
            let b = peak_alerts[peak_alerts.len() / 2];
            let c = peak_alerts[peak_alerts.len() - 1];
            if a.key != b.key && b.key != c.key && a.key != c.key {
                return Some(SeedNode::from_triplet_with_members(
                    local_store,
                    night_id,
                    a,
                    b,
                    c,
                    &peak_alerts,
                ));
            }
            return None;
        }

        if triplet_only || peak_alerts.len() < 2 {
            return None;
        }

        let a = peak_alerts[0];
        let b = peak_alerts[peak_alerts.len() - 1];
        SeedNode::from_pair(local_store, night_id, a, b, None)
    }

    /// Return true when two peaks represent near-duplicate memberships.
    ///
    /// The overlap test uses the full sorted alert membership of each peak and
    /// considers the peaks equivalent when either the Jaccard index or the
    /// symmetric containment score exceeds the configured thresholds.
    ///
    /// Arguments
    /// ---------
    /// * `b` - Second peak used for the membership-overlap comparison.
    ///
    /// Return
    /// ------
    /// * `true` - The peaks share a sufficiently strong membership overlap.
    /// * `false` - The memberships are too distinct to be considered duplicates.
    #[inline]
    pub fn strong_membership_overlap(&self, b: &PeakCandidate) -> bool {
        let (jaccard, containment) = overlap_metrics(&self.alert_indices, &b.alert_indices);
        jaccard >= NMS_MIN_JACCARD || containment >= NMS_MIN_CONTAINMENT
    }

    /// Convert the sparse accumulator into ranked peak candidates.
    ///
    /// The accumulator is filtered by the configured minimum number of alerts
    /// per peak, sorted by decreasing vote score, and truncated to
    /// `max_peaks_per_night`.
    ///
    /// Arguments
    /// ---------
    /// * `acc` - Sparse Hough accumulator keyed by velocity and projected sky
    ///   bin.
    /// * `cfg` - Hough seeding configuration.
    ///
    /// Return
    /// ------
    /// * `Vec<PeakCandidate>` - Ranked peak candidates, best score first.
    pub fn extract_ranked_peaks(
        acc: AHashMap<AccKey, AccumulatorCell>,
        cfg: &HoughSeedingConfig,
    ) -> Vec<Self> {
        let mut peaks: Vec<PeakCandidate> = acc
            .into_iter()
            .filter_map(|(key, mut cell)| {
                cell.alert_indices.sort_unstable();
                cell.alert_indices.dedup();
                (cell.alert_indices.len() >= cfg.min_alerts_per_peak).then_some(PeakCandidate {
                    key,
                    score: cell.score,
                    alert_indices: cell.alert_indices,
                })
            })
            .collect();

        peaks.sort_by(|a, b| b.score.total_cmp(&a.score));
        peaks.truncate(cfg.max_peaks_per_night);
        peaks
    }
}

/// Enforce that each alert index can appear in at most `max_per_alert` retained peaks.
///
/// This is a greedy cap applied after ranking and NMS. Peaks are processed in
/// descending score order and are retained only if every member alert still
/// has remaining budget.
///
/// Arguments
/// ---------
/// * `peaks` - Peak list already ordered by score.
/// * `max_per_alert` - Maximum number of retained peaks that may contain the
///   same alert index. A value of `0` disables the cap.
///
/// Return
/// ------
/// * `Vec<PeakCandidate>` - Peaks that satisfy the per-alert participation cap.
pub fn cap_peaks_per_alert(peaks: Vec<PeakCandidate>, max_per_alert: usize) -> Vec<PeakCandidate> {
    if max_per_alert == 0 {
        return peaks;
    }

    let max_alert_idx = peaks
        .iter()
        .flat_map(|peak| peak.alert_indices.iter().copied())
        .max();
    let Some(max_alert_idx) = max_alert_idx else {
        return peaks;
    };

    let mut usage_count = vec![0usize; max_alert_idx + 1];
    let mut kept = Vec::with_capacity(peaks.len());

    'peak: for peak in peaks {
        for &idx in &peak.alert_indices {
            if usage_count[idx] >= max_per_alert {
                continue 'peak;
            }
        }
        for &idx in &peak.alert_indices {
            usage_count[idx] += 1;
        }
        kept.push(peak);
    }

    kept
}

/// Check whether the alerts attached to one peak are photometrically compatible.
///
/// The test is performed independently in each band. For every band with at
/// least two finite magnitude measurements, all pairwise differences are
/// compared against a tolerance of the form
/// $\Delta m\_{\max} + k\sqrt{\sigma_i^2 + \sigma_j^2}$,
/// where the additive term is the configured absolute limit and the second term
/// is the uncertainty-scaled allowance. Bands with fewer than two valid
/// magnitudes do not constrain the peak and are therefore accepted.
///
/// Arguments
/// ---------
/// * `alerts` - Time-ordered peak membership.
/// * `cfg` - Hough seeding configuration providing the photometric thresholds.
///
/// Return
/// ------
/// * `true` - All per-band magnitude pairs are compatible.
/// * `false` - At least one band contains an incompatible magnitude spread.
fn peak_photometry_ok(alerts: &[&Alert], cfg: &HoughSeedingConfig) -> bool {
    let mut by_band: AHashMap<u8, Vec<(f64, f64)>> = AHashMap::new();
    for alert in alerts {
        if !alert.mag.is_finite() || !alert.mag_err.is_finite() || alert.mag_err < 0.0 {
            continue;
        }
        by_band
            .entry(alert.band)
            .or_default()
            .push((alert.mag, alert.mag_err));
    }

    for mags in by_band.values() {
        for i in 0..mags.len() {
            for j in (i + 1)..mags.len() {
                let (mi, si) = mags[i];
                let (mj, sj) = mags[j];
                let tol = cfg.photometric_max_mag_diff
                    + cfg.photometric_sigma_multiplier * (si * si + sj * sj).sqrt();
                if (mi - mj).abs() > tol {
                    return false;
                }
            }
        }
    }
    true
}

/// Keep only peaks that satisfy the optional photometric consistency check.
///
/// When `cfg.photometric_filter` is disabled, the input vector is returned
/// unchanged. Otherwise, each peak is resolved back to its alerts and passed to
/// the internal photometric compatibility test.
///
/// Arguments
/// ---------
/// * `peaks` - Peak list to filter.
/// * `alerts` - Full night-level alert slice used to resolve peak members.
/// * `cfg` - Hough seeding configuration.
///
/// Return
/// ------
/// * `Vec<PeakCandidate>` - Peaks that passed the photometric compatibility test.
pub fn apply_photometric_filter(
    peaks: Vec<PeakCandidate>,
    alerts: &[Alert],
    cfg: &HoughSeedingConfig,
) -> Vec<PeakCandidate> {
    if !cfg.photometric_filter {
        return peaks;
    }

    peaks
        .into_iter()
        .filter(|peak| {
            let peak_alerts = peak.collect_peak_alerts_sorted(alerts);
            peak_photometry_ok(&peak_alerts, cfg)
        })
        .collect()
}
