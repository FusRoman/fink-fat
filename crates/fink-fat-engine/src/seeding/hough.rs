//! Kinematic Hough-transform seeding.
//!
//! This module implements an alternative intra-night seeding strategy based on a
//! discretized velocity search. For each velocity hypothesis $(v_\alpha, v_\delta)$,
//! alerts are projected back to a common reference epoch and accumulated in a sparse
//! spatial Hough space. Local maxima in that accumulator are then converted into
//! [`SeedNode`] candidates.
//!
//! The method is designed to recover approximately linear apparent motion on the
//! celestial sphere over the time span of a single night. In tangent-plane form,
//! the projection used by this module is
//!
//! $$\begin{align} \alpha_0 &= \mathrm{wrap}_{\pi}(\alpha - v_\alpha \Delta t) \\ \delta_0 &= \delta - v_\delta \Delta t \end{align}$$
//!
//! where $\Delta t$ is measured relative to the first alert of the night.
//!
//! ## Main types
//!
//! - [`HoughSeedStats`] records the number of hypotheses, accumulator bins, peaks,
//!   and emitted seeds.
//! - [`build_hough_seeds_for_night`] builds [`SeedNode`]
//!   values from one night of alerts.
//!
//! ## Algorithm outline
//!
//! 1. Build a square grid of angular-velocity hypotheses.
//! 2. Project each alert to a reference epoch for each hypothesis.
//! 3. Accumulate votes in sparse spatial bins keyed by velocity and position.
//! 4. Keep the strongest peaks and optionally reject photometrically inconsistent
//!    alert groups.
//! 5. Emit pair or triplet seeds from the retained peaks.

use ahash::AHashMap;

use crate::{
    Alert,
    astro_math::wrap_pm_pi,
    engine_config::seeding_config::HoughSeedingConfig,
    night_id::NightId,
    seeding::{SeedNode, store::SeedStore},
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
    /// Number of pair seeds emitted from the retained peaks.
    pub n_pair_seeds: u64,
    /// Number of triplet seeds emitted from the retained peaks.
    pub n_triplet_seeds: u64,
}

/// Sparse accumulator cell attached to one velocity hypothesis and one spatial bin.
#[derive(Clone, Debug, Default)]
struct AccumulatorCell {
    /// Sum of vote weights contributed by the alerts mapped to this cell.
    score: f64,
    /// Indices of the alerts that voted for this cell.
    alert_indices: Vec<usize>,
}

/// Key used to index the sparse Hough accumulator.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct AccKey {
    /// X index in the velocity grid.
    vel_ix: usize,
    /// Y index in the velocity grid.
    vel_iy: usize,
    /// Discretized projected right ascension bin.
    alpha_bin: i32,
    /// Discretized projected declination bin.
    delta_bin: i32,
}

/// Peak retained from the sparse Hough accumulator.
#[derive(Clone, Debug)]
struct PeakCandidate {
    /// Total vote weight accumulated by the peak.
    score: f64,
    /// Alert indices associated with the peak.
    alert_indices: Vec<usize>,
}

/// Build the square velocity grid used by the Hough search.
///
/// The grid spans `[-max_angular_speed, max_angular_speed]` in both angular-velocity
/// components and discards hypotheses whose norm falls outside the configured speed
/// interval.
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
/// When photometric uncertainty weighting is enabled, the alert contributes roughly
/// inversely proportional to the variance of its magnitude estimate.
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

/// Check whether the alerts attached to one peak are photometrically compatible.
///
/// The test is performed per band. Each pair of magnitude measurements is compared
/// against a threshold that combines an absolute limit and a sigma-scaled tolerance.
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

/// Build seeds for one night using a kinematic Hough transform.
///
/// Each retained peak is converted to a seed candidate:
/// - triplet seeds are emitted when at least three alerts support the peak,
/// - pair seeds are emitted only when `triplet_only == false`.
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
/// * `Vec<SeedNode>` - Seeds sorted using the local `SeedNode` ordering.
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

    // Reference all projections to the earliest alert of the night.
    let t_ref = alerts
        .iter()
        .map(|a| a.mjd_tt)
        .fold(f64::INFINITY, f64::min);
    if !t_ref.is_finite() {
        return (Vec::new(), stats);
    }

    tracing::trace!(
        %night_id,
        n_alerts = alerts.len(),
        t_ref,
        "building Hough seeds for night"
    );

    let vel_grid = velocity_grid(cfg);
    stats.n_velocity_hypotheses = vel_grid.len() as u64;

    let mut acc: AHashMap<AccKey, AccumulatorCell> = AHashMap::new();

    tracing::trace!(
        %night_id,
        n_velocity_hypotheses = stats.n_velocity_hypotheses,
        "accumulating votes in Hough space"
    );

    for &(ix, iy, vx, vy) in &vel_grid {
        for (alert_idx, alert) in alerts.iter().enumerate() {
            // Project the alert back to the reference epoch under one velocity model.
            let dt = alert.mjd_tt - t_ref;
            if !dt.is_finite() {
                continue;
            }
            let alpha0 = wrap_pm_pi(alert.ra - vx * dt);
            let delta0 = alert.dec - vy * dt;
            if !alpha0.is_finite() || !delta0.is_finite() {
                continue;
            }
            if delta0.abs() > std::f64::consts::FRAC_PI_2 + 1e-6 {
                continue;
            }

            let alpha_bin = (alpha0 / cfg.spatial_bin_size).floor() as i32;
            let delta_bin = (delta0 / cfg.spatial_bin_size).floor() as i32;
            let key = AccKey {
                vel_ix: ix,
                vel_iy: iy,
                alpha_bin,
                delta_bin,
            };
            let cell = acc.entry(key).or_default();
            cell.score += vote_weight(alert, cfg);
            cell.alert_indices.push(alert_idx);
        }
    }

    stats.n_accumulator_bins = acc.len() as u64;

    tracing::trace!(
        %night_id,
        n_accumulator_bins = stats.n_accumulator_bins,
        "filtering and ranking accumulator bins"
    );

    // Keep only the bins that are sufficiently populated and rank them by score.
    let mut peaks: Vec<PeakCandidate> = acc
        .into_values()
        .filter_map(|mut cell| {
            cell.alert_indices.sort_unstable();
            cell.alert_indices.dedup();
            (cell.alert_indices.len() >= cfg.min_alerts_per_peak).then_some(PeakCandidate {
                score: cell.score,
                alert_indices: cell.alert_indices,
            })
        })
        .collect();

    peaks.sort_by(|a, b| b.score.total_cmp(&a.score));
    if peaks.len() > cfg.max_peaks_per_night {
        peaks.truncate(cfg.max_peaks_per_night);
    }
    stats.n_peaks = peaks.len() as u64;

    tracing::trace!(
        %night_id,
        n_peaks = stats.n_peaks,
        "building seeds from Hough peaks"
    );

    let mut out: Vec<SeedNode> = Vec::with_capacity(peaks.len());
    let mut local_store = SeedStore::new();

    for peak in peaks {
        // Recover the alerts that voted for this peak and sort them in time order.
        let mut peak_alerts: Vec<&Alert> = peak
            .alert_indices
            .iter()
            .filter_map(|&idx| alerts.get(idx))
            .collect();
        peak_alerts.sort_by(|a, b| a.mjd_tt.total_cmp(&b.mjd_tt));

        if cfg.photometric_filter && !peak_photometry_ok(&peak_alerts, cfg) {
            continue;
        }
        stats.n_peaks_after_photometric_filter += 1;

        // Build the strongest seed supported by this peak.
        if peak_alerts.len() >= 3 {
            let a = peak_alerts[0];
            let b = peak_alerts[peak_alerts.len() / 2];
            let c = peak_alerts[peak_alerts.len() - 1];
            if a.key != b.key && b.key != c.key && a.key != c.key {
                out.push(SeedNode::from_triplet(&mut local_store, night_id, a, b, c));
                stats.n_triplet_seeds += 1;
            }
        } else if !triplet_only && peak_alerts.len() >= 2 {
            let a = peak_alerts[0];
            let b = peak_alerts[peak_alerts.len() - 1];
            if let Some(seed) = SeedNode::from_pair(&mut local_store, night_id, a, b, None) {
                out.push(seed);
                stats.n_pair_seeds += 1;
            }
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

    use crate::{AlertKey, astro_math::arcsec_to_rad, night_id::NightId};

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
            max_peaks_per_night: 64,
            photometric_filter: true,
            photometric_max_mag_diff: 0.5,
            photometric_sigma_multiplier: 3.0,
            weight_by_photometric_error: true,
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
}
