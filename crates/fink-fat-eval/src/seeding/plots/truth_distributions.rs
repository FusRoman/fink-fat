//! Collect and plot parameter distributions for **true** intra-night pairs and
//! triplets.
//!
//! For each ground-truth trajectory that has ≥ 2 alerts on a single night the
//! module computes the seeding-relevant metrics (Δt, angular speed, flux
//! difference, predicted residual, …) and writes one three-panel chart
//! (histogram / CDF / percentiles) per metric to `out_dir`.
//!
//! The plots serve to calibrate the cutoff parameters defined in
//! `eval_config.yml` under `pairs:` and `triplets:`.

use std::path::Path;

use ahash::AHashMap;
use anyhow::Result;
use camino::Utf8Path;
use fink_fat_engine::{
    Alert, AlertStore,
    astro_math::{ang_sep, planar_offset_fast},
    engine_config::{pair_config::PairConfig, triplet_config::TripletConfig},
    night_id::NightId,
};

use crate::truth_sso::{TrajId, TruthSSO};

use super::draw_helpers::MetricPlot;

const RAD_TO_ARCMIN: f64 = 60.0 * 180.0 / std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Data containers
// ─────────────────────────────────────────────────────────────────────────────

/// Metrics measured on ground-truth pairs (same night, same trajectory).
#[derive(Default)]
pub struct TruthPairData {
    /// Time separation between the two detections (hours).
    pub dt_hours: Vec<f64>,
    /// Angular speed (arcmin / day).
    pub angular_speed_arcmin_per_day: Vec<f64>,
    /// |flux_a − flux_b| (upstream flux units).
    pub flux_difference: Vec<f64>,
}

/// Metrics measured on ground-truth triplets (same night, same trajectory).
#[derive(Default)]
pub struct TruthTripletData {
    /// max(Δt(a,b), Δt(b,c)) in hours.
    pub max_dt_between_hours: Vec<f64>,
    /// Angular separation for consecutive sub-pairs – both ab and bc – in arcmin.
    pub pair_sep_arcmin: Vec<f64>,
    /// Predicted residual at c using the linear (a→b) model, in arcmin.
    pub predicted_residual_arcmin: Vec<f64>,
    /// Range of flux values within the triplet (max − min).
    pub flux_difference: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Data collection
// ─────────────────────────────────────────────────────────────────────────────

/// Group alerts from a single night by their ground-truth trajectory ID.
///
/// Only alerts present in `truth` are included.  Each group is sorted by
/// ascending MJD.
fn group_night_by_traj<'a>(
    night_alerts: &'a [Alert],
    truth: &TruthSSO,
) -> AHashMap<TrajId, Vec<&'a Alert>> {
    let mut map: AHashMap<TrajId, Vec<&'a Alert>> = AHashMap::new();
    for alert in night_alerts {
        if let Some(traj_id) = truth.get_truth_traj_id(&alert) {
            map.entry(traj_id).or_default().push(alert);
        }
    }
    for alerts in map.values_mut() {
        alerts.sort_by(|a, b| a.mjd_tt.total_cmp(&b.mjd_tt));
    }
    map
}

/// Accumulate pair metrics for every ordered pair (i,j) with i < j in `alerts`.
fn accumulate_pair_metrics(alerts: &[&Alert], data: &mut TruthPairData) {
    let n = alerts.len();
    if n < 2 {
        return;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let a = alerts[i];
            let b = alerts[j];
            let dt_days = b.mjd_tt - a.mjd_tt;
            if dt_days <= 0.0 {
                continue;
            }
            let sep = ang_sep(a.ra, a.dec, b.ra, b.dec);
            data.dt_hours.push(dt_days * 24.0);
            data.angular_speed_arcmin_per_day
                .push(sep / dt_days * RAD_TO_ARCMIN);
            data.flux_difference.push((a.flux - b.flux).abs());
        }
    }
}

/// Accumulate triplet metrics for every ordered triplet (i,j,k) with i<j<k.
fn accumulate_triplet_metrics(alerts: &[&Alert], data: &mut TruthTripletData) {
    let n = alerts.len();
    if n < 3 {
        return;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let a = alerts[i];
                let b = alerts[j];
                let c = alerts[k];

                let dt_ab = b.mjd_tt - a.mjd_tt;
                let dt_bc = c.mjd_tt - b.mjd_tt;
                let dt_ac = c.mjd_tt - a.mjd_tt;

                if dt_ab <= 0.0 || dt_bc <= 0.0 {
                    continue;
                }

                let sep_ab = ang_sep(a.ra, a.dec, b.ra, b.dec);
                let sep_bc = ang_sep(b.ra, b.dec, c.ra, c.dec);

                // Linear prediction: velocity from (a,b), predict c from a.
                let cos_dec_a = a.dec.cos();
                let (dx_ab, dy_ab) = planar_offset_fast(a.ra, a.dec, cos_dec_a, b.ra, b.dec);
                let (dx_ac, dy_ac) = planar_offset_fast(a.ra, a.dec, cos_dec_a, c.ra, c.dec);
                let vx = dx_ab / dt_ab;
                let vy = dy_ab / dt_ab;
                let pred_x = vx * dt_ac;
                let pred_y = vy * dt_ac;
                let residual = ((dx_ac - pred_x).powi(2) + (dy_ac - pred_y).powi(2)).sqrt();

                let fluxes = [a.flux, b.flux, c.flux];
                let f_max = fluxes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let f_min = fluxes.iter().copied().fold(f64::INFINITY, f64::min);

                data.max_dt_between_hours.push(dt_ab.max(dt_bc) * 24.0);
                data.pair_sep_arcmin.push(sep_ab * RAD_TO_ARCMIN);
                data.pair_sep_arcmin.push(sep_bc * RAD_TO_ARCMIN);
                data.predicted_residual_arcmin
                    .push(residual * RAD_TO_ARCMIN);
                data.flux_difference.push(f_max - f_min);
            }
        }
    }
}

/// Collect pair and triplet truth metrics across all nights and all
/// ground-truth trajectories.
pub fn collect_truth_data(
    alert_store: &AlertStore,
    truth: &TruthSSO,
) -> (TruthPairData, TruthTripletData) {
    let mut pairs = TruthPairData::default();
    let mut triplets = TruthTripletData::default();

    let mut nights: Vec<&NightId> = alert_store.nights().collect();
    nights.sort();

    for night_id in nights {
        let Some(night_alerts) = alert_store.get(night_id) else {
            continue;
        };
        let by_traj = group_night_by_traj(night_alerts, truth);
        for alerts in by_traj.values() {
            accumulate_pair_metrics(alerts, &mut pairs);
            accumulate_triplet_metrics(alerts, &mut triplets);
        }
    }
    (pairs, triplets)
}

// ─────────────────────────────────────────────────────────────────────────────
// Sorting helpers
// ─────────────────────────────────────────────────────────────────────────────

fn sort_finite(mut v: Vec<f64>) -> Vec<f64> {
    v.retain(|x| x.is_finite());
    v.sort_by(|a, b| a.total_cmp(b));
    v
}

// ─────────────────────────────────────────────────────────────────────────────
// Plot entry-points
// ─────────────────────────────────────────────────────────────────────────────

/// Write pair-parameter distribution charts to `out_dir`.
///
/// Files produced:
/// - `pairs_dt.png`             – time separation distribution
/// - `pairs_angular_speed.png`  – angular speed distribution
/// - `pairs_flux_diff.png`      – flux difference distribution
pub fn plot_pair_distributions(
    data: TruthPairData,
    pair_cfg: &PairConfig,
    out_dir: &Utf8Path,
) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;

    let TruthPairData {
        dt_hours,
        angular_speed_arcmin_per_day,
        flux_difference,
    } = data;

    plot_metric(
        &sort_finite(dt_hours),
        "True pairs: dt",
        "dt (hours)",
        Some(pair_cfg.max_dt * 24.0),
        false,
        &out_dir.as_std_path().join("pairs_dt.png"),
    )?;

    plot_metric(
        &sort_finite(angular_speed_arcmin_per_day),
        "True pairs: angular speed",
        "angular speed (arcmin/day)",
        Some(pair_cfg.max_angular_speed * RAD_TO_ARCMIN),
        true,
        &out_dir.as_std_path().join("pairs_angular_speed.png"),
    )?;

    plot_metric(
        &sort_finite(flux_difference),
        "True pairs: flux difference",
        "|flux_a - flux_b|",
        Some(pair_cfg.max_flux_difference),
        false,
        &out_dir.as_std_path().join("pairs_flux_diff.png"),
    )?;

    Ok(())
}

/// Write triplet-parameter distribution charts to `out_dir`.
///
/// Files produced:
/// - `triplets_max_dt.png`       – max Δt between consecutive detections
/// - `triplets_pair_sep.png`     – consecutive-pair angular separation
/// - `triplets_residual.png`     – linear-model predicted residual at c
/// - `triplets_flux_diff.png`    – flux range within the triplet
pub fn plot_triplet_distributions(
    data: TruthTripletData,
    triplet_cfg: &TripletConfig,
    out_dir: &Utf8Path,
) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;

    let TruthTripletData {
        max_dt_between_hours,
        pair_sep_arcmin,
        predicted_residual_arcmin,
        flux_difference,
    } = data;

    plot_metric(
        &sort_finite(max_dt_between_hours),
        "True triplets: max dt between",
        "max dt (hours)",
        Some(triplet_cfg.max_dt_between * 24.0),
        false,
        &out_dir.as_std_path().join("triplets_max_dt.png"),
    )?;

    plot_metric(
        &sort_finite(pair_sep_arcmin),
        "True triplets: consecutive pair separation",
        "pair sep (arcmin)",
        Some(triplet_cfg.max_pair_sep * RAD_TO_ARCMIN),
        true,
        &out_dir.as_std_path().join("triplets_pair_sep.png"),
    )?;

    plot_metric(
        &sort_finite(predicted_residual_arcmin),
        "True triplets: predicted residual at c",
        "residual (arcmin)",
        Some(triplet_cfg.max_predicted_residual * RAD_TO_ARCMIN),
        true,
        &out_dir.as_std_path().join("triplets_residual.png"),
    )?;

    plot_metric(
        &sort_finite(flux_difference),
        "True triplets: flux range",
        "flux range (max-min)",
        Some(triplet_cfg.max_flux_difference),
        false,
        &out_dir.as_std_path().join("triplets_flux_diff.png"),
    )?;

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helper
// ─────────────────────────────────────────────────────────────────────────────

fn plot_metric(
    sorted: &[f64],
    title: &str,
    x_label: &str,
    vline: Option<f64>,
    log_x: bool,
    path: &Path,
) -> Result<()> {
    let mut plot = MetricPlot::new(title, x_label, sorted.to_vec());
    if let Some(vl) = vline {
        plot = plot.with_vline(vl);
    }
    if log_x {
        plot = plot.with_log_x();
    }
    plot.save_to(path)?;
    tracing::debug!("wrote {}", path.display());
    Ok(())
}
