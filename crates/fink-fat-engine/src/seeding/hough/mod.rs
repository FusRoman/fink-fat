//! Kinematic Hough-transform seeding.
//!
//! This module implements an alternative intra-night seeding strategy based on a
//! discretized angular-velocity search. It detects approximately linear motion in
//! tangent-plane coordinates by scanning a grid of velocity hypotheses, voting in
//! a sparse accumulator, and converting local maxima into [`SeedNode`] candidates.
//!
//! The implementation is split into three submodules:
//!
//! - [`accumulator`] projects alerts, fills
//!   the sparse Hough grid, and stores the member alert indices per cell.
//! - [`peaks`] ranks accumulator maxima, applies
//!   photometric filtering, enforces per-alert participation limits, and turns
//!   retained peaks into seeds.
//! - [`nms`] applies a greedy non-maximum suppression
//!   step based on Hough-space proximity and membership overlap.
//!
//! In tangent-plane form, the projection used by the module is
//!
//! $$\begin{align} \alpha\_0 &= \mathrm{wrap}\_{\pi}(\alpha - v\_\alpha \Delta t) \\\ \delta\_0 &= \delta - v\_\delta \Delta t \end{align}$$
//!
//! where $\Delta t$ is measured relative to the earliest alert of the night.
//! The projected positions are quantized into spatial bins, and the resulting
//! occupied cells are interpreted as local maxima in a discrete motion-space
//! search.
//!
//! ## Scientific interpretation
//!
//! The method is designed to recover objects whose apparent motion remains close
//! to linear over a single night. Each velocity hypothesis defines a different
//! back-projection of the observations; peaks in the accumulator correspond to
//! combinations of alerts that align under one such hypothesis. The ranking is a
//! discrete approximation to a likelihood or vote-count maximum, while NMS and
//! the per-alert cap reduce fragmentation into multiple nearly identical seeds.
//!
//! ## Main items
//!
//! - [`HoughSeedStats`] records the number of evaluated hypotheses, occupied
//!   accumulator bins, retained peaks, and emitted seeds.
//! - [`build_hough_seeds_for_night`] builds [`SeedNode`] values from one night of
//!   alerts.
//!
//! ## Algorithm outline
//!
//! 1. Build a square grid of angular-velocity hypotheses.
//! 2. Back-project each alert to the reference epoch for every hypothesis.
//! 3. Accumulate votes in sparse spatial bins keyed by velocity and projected
//!    position.
//! 4. Rank occupied bins by vote score and keep only the strongest peaks.
//! 5. Optionally reject peaks whose member alerts are photometrically
//!    incompatible.
//! 6. Apply greedy NMS using both Hough-space proximity and strong member
//!    overlap.
//! 7. Optionally cap how many retained peaks each alert can support.
//! 8. Emit pair seeds (2 alerts) or triplet-derived seeds fitted on
//!    first/middle/last alerts, while preserving full peak membership.
//!
//! The resulting Hough seed is therefore a hybrid object:
//!
//! - the motion model is estimated from three anchor alerts when possible,
//! - the stored membership can contain every alert that voted for the peak,
//! - `n_obs` reflects the total stored membership count, not only the fit
//!   anchors.

pub mod accumulator;
pub mod nms;
pub mod peaks;

use crate::{
    Alert,
    engine_config::seeding_config::HoughSeedingConfig,
    night_id::NightId,
    seeding::{
        SeedNode,
        hough::{
            accumulator::build_hough_accumulator,
            nms::apply_nms,
            peaks::{PeakCandidate, apply_photometric_filter, cap_peaks_per_alert},
        },
        store::SeedStore,
    },
};

/// Summary statistics collected while building Hough seeds for one night.
#[derive(Clone, Copy, Debug, Default)]
pub struct HoughSeedStats {
    /// Number of velocity hypotheses evaluated.
    pub n_velocity_hypotheses: u64,
    /// Number of sparse accumulator cells that received at least one vote.
    pub n_accumulator_bins: u64,
    /// Number of peaks retained after accumulator thresholding and ranking.
    pub n_peaks: u64,
    /// Number of peaks that survived the optional photometric consistency test.
    pub n_peaks_after_photometric_filter: u64,
    /// Number of peaks kept after Hough-space and membership-overlap NMS.
    pub n_peaks_after_nms: u64,
    /// Number of peaks kept after applying the per-alert participation cap.
    pub n_peaks_after_alert_cap: u64,
    /// Number of pair seeds emitted from the retained peaks.
    pub n_pair_seeds: u64,
    /// Number of triplet seeds emitted from the retained peaks.
    pub n_triplet_seeds: u64,
}

/// Build the square velocity grid used by the Hough search.
///
/// The grid spans `[-max_angular_speed, max_angular_speed]` in both angular-
/// velocity components and discards hypotheses whose norm falls outside the
/// configured speed interval. The resulting set is a square grid in parameter
/// space with a radial speed cut.
///
/// Arguments
/// ---------
/// * `cfg` - Hough configuration providing the speed bounds and grid resolution.
///
/// Return
/// ------
/// * `Vec<(usize, usize, f64, f64)>` - Velocity hypotheses encoded as grid
///   indices and angular velocities in radians per day.
fn velocity_grid(cfg: &HoughSeedingConfig) -> Vec<(usize, usize, f64, f64)> {
    let n = cfg.velocity_grid_steps;
    let vmax = cfg.max_angular_speed;
    let vmin = cfg.min_angular_speed;

    let mut out = Vec::with_capacity(n * n);
    let denom = (n.saturating_sub(1)).max(1) as f64;
    for ix in 0..n {
        let vx = -vmax + 2.0 * vmax * (ix as f64) / denom;
        for iy in 0..n {
            let vy = -vmax + 2.0 * vmax * (iy as f64) / denom;
            let speed = (vx * vx + vy * vy).sqrt();
            if speed >= vmin && speed <= vmax {
                out.push((ix, iy, vx, vy));
            }
        }
    }
    out
}

/// Compute the vote weight contributed by one alert.
///
/// When photometric uncertainty weighting is enabled, the alert contributes a
/// weight approximately proportional to the inverse variance of its magnitude
/// estimate, with lower and upper clamping to avoid pathological extremes.
///
/// Arguments
/// ---------
/// * `alert` - Alert contributing one vote to the accumulator.
/// * `cfg` - Hough configuration controlling whether uncertainty weighting is
///   enabled.
///
/// Return
/// ------
/// * `f64` - Non-negative vote weight used when updating the sparse accumulator.
fn vote_weight(alert: &Alert, cfg: &HoughSeedingConfig) -> f64 {
    if !cfg.weight_by_photometric_error {
        return 1.0;
    }
    if !alert.mag_err.is_finite() || alert.mag_err < 0.0 {
        return 1.0;
    }
    let sigma = alert.mag_err.max(1e-6);
    let w = 1.0 / (sigma * sigma);
    if w.is_finite() {
        w.clamp(1e-3, 1e6)
    } else {
        1.0
    }
}

/// Return the reference epoch used for Hough projections.
///
/// The earliest finite observation time of the night is used as the common
/// reference epoch. This keeps all projections numerically stable and makes the
/// velocity search independent of the absolute night date.
///
/// Arguments
/// ---------
/// * `alerts` - Alerts belonging to one night.
///
/// Return
/// ------
/// * `Some(f64)` - Earliest finite `mjd_tt` value in the input slice.
/// * `None` - No finite observation time was available.
fn reference_epoch(alerts: &[Alert]) -> Option<f64> {
    let t_ref = alerts
        .iter()
        .map(|a| a.mjd_tt)
        .fold(f64::INFINITY, f64::min);
    t_ref.is_finite().then_some(t_ref)
}

/// Build seeds for one night using a kinematic Hough transform.
///
/// Each retained peak is converted to a seed candidate after the following
/// stages:
///
/// - Hough-space accumulation over a velocity grid,
/// - ranking and truncation of occupied accumulator bins,
/// - optional photometric filtering,
/// - greedy NMS based on Hough proximity and membership overlap,
/// - optional per-alert participation limiting.
///
/// If a surviving peak has at least three alerts, the kinematic fit uses the
/// first, middle, and last alerts in time order. The complete, deduplicated peak
/// membership is attached to the output seed. Peaks with exactly two alerts can
/// produce pair seeds when `triplet_only == false`.
///
/// The peak membership is deduplicated and time-sorted before being attached to
/// the seed, so `members.len()` may exceed 3 even though the fit uses only three
/// anchor detections.
///
/// Arguments
/// ---------
/// * `alerts` - Alerts belonging to the same night.
/// * `night_id` - Identifier assigned to the output seeds.
/// * `cfg` - Hough transform configuration.
/// * `triplet_only` - If `true`, suppress pair seed emission.
///
/// Return
/// ------
/// * `Vec<SeedNode>` - Seeds sorted using the local `SeedNode` ordering. For
///   triplet-derived Hough seeds, `n_obs == members.len()` and may be >3.
/// * `HoughSeedStats` - Per-night diagnostic counters.
pub fn build_hough_seeds_for_night(
    alerts: &[Alert],
    night_id: NightId,
    cfg: &HoughSeedingConfig,
    triplet_only: bool,
) -> (Vec<SeedNode>, HoughSeedStats) {
    let mut stats = HoughSeedStats::default();
    if alerts.len() < 2 {
        return (Vec::new(), stats);
    }

    let Some(t_ref) = reference_epoch(alerts) else {
        return (Vec::new(), stats);
    };

    tracing::trace!(
        %night_id,
        n_alerts = alerts.len(),
        t_ref,
        "building Hough seeds for night"
    );

    let vel_grid = velocity_grid(cfg);
    stats.n_velocity_hypotheses = vel_grid.len() as u64;

    tracing::trace!(
        %night_id,
        n_velocity_hypotheses = stats.n_velocity_hypotheses,
        "accumulating votes in Hough space"
    );

    let acc = build_hough_accumulator(alerts, cfg, &vel_grid, t_ref);

    stats.n_accumulator_bins = acc.len() as u64;

    tracing::trace!(
        %night_id,
        n_accumulator_bins = stats.n_accumulator_bins,
        "filtering and ranking accumulator bins"
    );

    let peaks = PeakCandidate::extract_ranked_peaks(acc, cfg);
    stats.n_peaks = peaks.len() as u64;

    let peaks = apply_photometric_filter(peaks, alerts, cfg);
    stats.n_peaks_after_photometric_filter = peaks.len() as u64;

    let peaks = apply_nms(peaks);
    stats.n_peaks_after_nms = peaks.len() as u64;

    let peaks = cap_peaks_per_alert(peaks, cfg.max_seeds_per_alert);
    stats.n_peaks_after_alert_cap = peaks.len() as u64;

    tracing::trace!(
        %night_id,
        n_peaks = stats.n_peaks,
        "building seeds from Hough peaks"
    );

    let mut out: Vec<SeedNode> = Vec::with_capacity(peaks.len());
    let mut local_store = SeedStore::new();

    for peak in peaks {
        if let Some(seed) =
            peak.build_seed_from_peak(alerts, night_id, triplet_only, &mut local_store)
        {
            if seed.n_obs >= 3 {
                stats.n_triplet_seeds += 1;
            } else {
                stats.n_pair_seeds += 1;
            }
            out.push(seed);
        }
    }

    tracing::trace!(
        %night_id,
        n_pair_seeds = stats.n_pair_seeds,
        n_triplet_seeds = stats.n_triplet_seeds,
        "finished building Hough seeds for night"
    );

    out.sort();
    (out, stats)
}

#[cfg(test)]
mod hough_transform_tests {
    use super::*;

    use crate::{
        AlertKey, astro_math::arcsec_to_rad, night_id::NightId, seeding::hough::accumulator::AccKey,
    };

    /// Create a synthetic alert used by the Hough seeding tests.
    fn mk_alert(
        i: usize,
        ra: f64,
        dec: f64,
        mjd_tt: f64,
        band: u8,
        mag: f64,
        mag_err: f64,
    ) -> Alert {
        Alert {
            key: AlertKey {
                night_id: NightId(42),
                dia_source_id: i as u64,
            },
            ra,
            ra_err: arcsec_to_rad(0.3),
            dec,
            dec_err: arcsec_to_rad(0.3),
            mjd_tt,
            mag,
            mag_err,
            band,
            ..Default::default()
        }
    }

    /// Verify that a near-linear synthetic trajectory produces a triplet seed.
    #[test]
    fn hough_detects_linear_triplet_peak() {
        let t0 = 60000.0;
        let dt = 10.0 / 1440.0;
        let v_ra = arcsec_to_rad(30.0) * 24.0; // 30 arcsec/h
        let v_dec = arcsec_to_rad(12.0) * 24.0;

        let alerts = vec![
            mk_alert(0, 1.0, 0.1, t0, 1, 1000.0, 20.0),
            mk_alert(
                1,
                1.0 + v_ra * dt,
                0.1 + v_dec * dt,
                t0 + dt,
                1,
                995.0,
                22.0,
            ),
            mk_alert(
                2,
                1.0 + v_ra * 2.0 * dt,
                0.1 + v_dec * 2.0 * dt,
                t0 + 2.0 * dt,
                1,
                1002.0,
                21.0,
            ),
            mk_alert(3, 2.0, -0.2, t0 + dt, 1, 4000.0, 100.0),
        ];

        let cfg = HoughSeedingConfig {
            min_angular_speed: 0.0,
            max_angular_speed: arcsec_to_rad(120.0) * 24.0,
            velocity_grid_steps: 31,
            spatial_bin_size: arcsec_to_rad(4.0),
            min_alerts_per_peak: 3,
            max_peaks_per_night: 4_000,
            photometric_filter: true,
            photometric_max_mag_diff: 0.5,
            photometric_sigma_multiplier: 3.0,
            weight_by_photometric_error: true,
            max_seeds_per_alert: 0,
        };

        let (seeds, stats) = build_hough_seeds_for_night(&alerts, NightId(42), &cfg, false);
        assert!(stats.n_peaks >= 1);
        assert!(!seeds.is_empty());
        assert!(seeds.iter().any(|s| s.n_obs == 3));
    }

    /// Verify that the optional photometric filter can reject an otherwise valid peak.
    #[test]
    fn hough_photometric_filter_rejects_incompatible_peak() {
        let t0 = 61000.0;
        let dt = 8.0 / 1440.0;
        let v_ra = arcsec_to_rad(20.0) * 24.0;

        let alerts = vec![
            mk_alert(0, 1.0, 0.2, t0, 2, 5000.0, 15.0),
            mk_alert(1, 1.0 + v_ra * dt, 0.2, t0 + dt, 2, 120.0, 3.0),
            mk_alert(
                2,
                1.0 + v_ra * 2.0 * dt,
                0.2,
                t0 + 2.0 * dt,
                2,
                5100.0,
                15.0,
            ),
        ];

        let cfg = HoughSeedingConfig {
            max_angular_speed: arcsec_to_rad(80.0) * 24.0,
            velocity_grid_steps: 21,
            spatial_bin_size: arcsec_to_rad(4.0),
            min_alerts_per_peak: 3,
            photometric_filter: true,
            photometric_max_mag_diff: 0.2,
            ..HoughSeedingConfig::default()
        };

        let (seeds, stats) = build_hough_seeds_for_night(&alerts, NightId(42), &cfg, false);
        assert!(stats.n_peaks >= 1);
        assert_eq!(stats.n_peaks_after_photometric_filter, 0);
        assert!(seeds.is_empty());
    }

    /// Verify that `triplet_only` suppresses pair emission for two-alert peaks.
    #[test]
    fn hough_triplet_only_blocks_pair_seed() {
        let t0 = 62000.0;
        let dt = 5.0 / 1440.0;
        let v_ra = arcsec_to_rad(40.0) * 24.0;
        let alerts = vec![
            mk_alert(0, 0.8, -0.3, t0, 1, 1000.0, 30.0),
            mk_alert(1, 0.8 + v_ra * dt, -0.3, t0 + dt, 1, 1005.0, 30.0),
        ];

        let cfg = HoughSeedingConfig {
            max_angular_speed: arcsec_to_rad(120.0) * 24.0,
            velocity_grid_steps: 21,
            spatial_bin_size: arcsec_to_rad(5.0),
            min_alerts_per_peak: 2,
            photometric_filter: false,
            ..HoughSeedingConfig::default()
        };

        let (seeds_pair_ok, _) = build_hough_seeds_for_night(&alerts, NightId(42), &cfg, false);
        assert!(!seeds_pair_ok.is_empty());

        let (seeds_triplet_only, _) = build_hough_seeds_for_night(&alerts, NightId(42), &cfg, true);
        assert!(seeds_triplet_only.is_empty());
    }

    /// Verify that Hough seeds keep all peak members while fitting from anchors.
    #[test]
    fn hough_keeps_all_peak_members() {
        let t0 = 63000.0;
        let dt = 4.0 / 1440.0;
        let v_ra = arcsec_to_rad(25.0) * 24.0;
        let alerts = vec![
            mk_alert(0, 1.2, 0.1, t0, 1, 1000.0, 5.0),
            mk_alert(1, 1.2 + v_ra * dt, 0.1, t0 + dt, 1, 1001.0, 5.0),
            mk_alert(2, 1.2 + v_ra * 2.0 * dt, 0.1, t0 + 2.0 * dt, 1, 1002.0, 5.0),
            mk_alert(3, 1.2 + v_ra * 3.0 * dt, 0.1, t0 + 3.0 * dt, 2, 1003.0, 5.0),
            mk_alert(4, 1.2 + v_ra * 4.0 * dt, 0.1, t0 + 4.0 * dt, 3, 1004.0, 5.0),
        ];

        let cfg = HoughSeedingConfig {
            min_angular_speed: 0.0,
            max_angular_speed: arcsec_to_rad(60.0) * 24.0,
            velocity_grid_steps: 31,
            spatial_bin_size: arcsec_to_rad(4.0),
            min_alerts_per_peak: 5,
            photometric_filter: false,
            ..HoughSeedingConfig::default()
        };

        let (seeds, _) = build_hough_seeds_for_night(&alerts, NightId(42), &cfg, true);
        assert!(!seeds.is_empty());
        assert!(seeds.iter().any(|s| s.n_obs == 5 && s.members.len() == 5));
    }

    #[test]
    fn hough_nms_suppresses_close_and_overlapping_peaks() {
        let peaks = vec![
            PeakCandidate {
                key: AccKey {
                    vel_ix: 10,
                    vel_iy: 11,
                    alpha_bin: 100,
                    delta_bin: -30,
                },
                score: 10.0,
                alert_indices: vec![1, 2, 3, 4],
            },
            PeakCandidate {
                key: AccKey {
                    vel_ix: 10,
                    vel_iy: 12,
                    alpha_bin: 101,
                    delta_bin: -30,
                },
                score: 9.0,
                alert_indices: vec![1, 2, 3, 4],
            },
            PeakCandidate {
                key: AccKey {
                    vel_ix: 18,
                    vel_iy: 18,
                    alpha_bin: 300,
                    delta_bin: 200,
                },
                score: 8.0,
                alert_indices: vec![10, 11, 12],
            },
        ];

        let kept = apply_nms(peaks);
        assert_eq!(kept.len(), 2);
        assert!(kept.iter().any(|p| p.alert_indices == vec![1, 2, 3, 4]));
        assert!(kept.iter().any(|p| p.alert_indices == vec![10, 11, 12]));
    }

    #[test]
    fn hough_cap_limits_alert_participation() {
        let peaks = vec![
            PeakCandidate {
                key: AccKey {
                    vel_ix: 0,
                    vel_iy: 0,
                    alpha_bin: 0,
                    delta_bin: 0,
                },
                score: 5.0,
                alert_indices: vec![1, 2, 3],
            },
            PeakCandidate {
                key: AccKey {
                    vel_ix: 1,
                    vel_iy: 1,
                    alpha_bin: 1,
                    delta_bin: 1,
                },
                score: 4.0,
                alert_indices: vec![1, 4, 5],
            },
            PeakCandidate {
                key: AccKey {
                    vel_ix: 2,
                    vel_iy: 2,
                    alpha_bin: 2,
                    delta_bin: 2,
                },
                score: 3.0,
                alert_indices: vec![1, 6, 7],
            },
        ];

        let kept = cap_peaks_per_alert(peaks, 2);
        assert_eq!(kept.len(), 2);
        assert!(kept.iter().all(|p| p.alert_indices.contains(&1)));
    }
}
