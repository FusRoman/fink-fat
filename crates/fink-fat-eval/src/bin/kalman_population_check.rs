use std::{collections::HashMap, f64::consts::PI};

use anyhow::{Context, Result};
use camino::{Utf8Path, Utf8PathBuf};
use clap::Parser;
use fink_fat_engine::{
    ecliptic_state::EclipticState,
    engine_config::{EngineConfig, load_engine_config_validated},
    tracklet::{Tracklet, track_storage::TrackId, tracklet_data::TrackletData},
};
use photom::{
    TrajId,
    coordinates::{ecliptic::EclipticCoordCov, equatorial::EquCoordCov},
    io::polars::{ContiguousChoice, FromPolarsArgs},
    observation_dataset::{ObsDataset, observation::Observation},
    observer::error_model::ObsErrorModel,
};
use polars::{
    frame::DataFrame,
    lazy::frame::{LazyFrame, ScanArgsParquet},
};

use plotters::prelude::*;

// ── CLI ───────────────────────────────────────────────────────────────────────

/// FINK-FAT Kalman recovery diagnostic.
#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Path to the parquet file containing labelled trajectories.
    #[arg(short, long, value_name = "ALERTS_FILE")]
    pub alerts: Utf8PathBuf,

    /// Path to the fink-fat engine configuration file.
    #[arg(short, long, value_name = "CONFIG_FILE")]
    pub config: Utf8PathBuf,
}

// ── Config / data loading ─────────────────────────────────────────────────────

pub fn load_config(config_path: &Utf8Path) -> Result<EngineConfig> {
    load_engine_config_validated(config_path).context("failed to load engine config")
}

pub fn load_data(parquet_path: &Utf8Path) -> Result<(DataFrame, ObsDataset)> {
    let path = parquet_path.as_str();
    let args = ScanArgsParquet {
        rechunk: true,
        ..Default::default()
    };
    let lf = LazyFrame::scan_parquet(path.into(), args).context("scan_parquet failed")?;
    let obs_dataset = ObsDataset::from_lazy(
        lf.clone(),
        FromPolarsArgs {
            do_rechunk: Some(false),
            error_model: Some(ObsErrorModel::FCCT14),
            contiguous_choice: Some(ContiguousChoice::ContiguousNight),
        },
    )
    .context("ObsDataset::from_lazy failed")?;

    let df = lf.collect().context("collect failed")?;
    Ok((df, obs_dataset))
}

// ── Record ────────────────────────────────────────────────────────────────────

/// One prediction record produced for each observation after the seed.
#[derive(Debug)]
struct PredictionRecord {
    traj_id: u32,
    obs_idx: usize,
    mjd: f64,
    /// Time gap since the previous observation (days).
    gap_days: f64,
    /// Predicted ecliptic longitude (rad).
    pred_lon: f64,
    /// Predicted ecliptic latitude (rad).
    pred_lat: f64,
    /// Observed ecliptic longitude (rad).
    obs_lon: f64,
    /// Observed ecliptic latitude (rad).
    obs_lat: f64,
    /// Innovation in longitude (arcsec).
    inn_lon: f64,
    /// Innovation in latitude (arcsec).
    inn_lat: f64,
    /// Innovation 1-sigma from S = HP⁻Hᵀ + R in longitude (arcsec).
    sigma_lon_pred: f64,
    /// Innovation 1-sigma from S = HP⁻Hᵀ + R in latitude (arcsec).
    sigma_lat_pred: f64,
    /// True if |innovation| ≤ 3σ on both axes.
    in_box: bool,
    /// Normalized Innovation Squared: $\text{NIS} = \nu^\top S^{-1} \nu$.
    ///
    /// For a well-calibrated filter, $\mathbb{E}[\text{NIS}] = \dim(z) = 2$.
    nis: f64,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[inline]
fn rad_to_arcsec(r: f64) -> f64 {
    r * 180.0 * 3600.0 / PI
}

/// Wrap a longitude innovation into (−π, π].
#[inline]
fn wrap_lon(d: f64) -> f64 {
    if d > PI {
        d - 2.0 * PI
    } else if d < -PI {
        d + 2.0 * PI
    } else {
        d
    }
}

// ── Seed selection ────────────────────────────────────────────────────────────

/// Threshold for two observations to be considered on the same night (days).
const SAME_NIGHT_THRESHOLD_DAYS: f64 = 1.0;

/// Select the seed observations from a sorted trajectory.
///
/// Strategy
/// --------
/// 1. Try a triplet: the three first observations where all consecutive pairs
///    are separated by less than [`SAME_NIGHT_THRESHOLD_DAYS`].
/// 2. Fall back to the first pair if the third observation is on a different
///    night or does not exist.
///
/// Return
/// ------
/// `(seed_obs, first_prediction_idx)` where `seed_obs` contains 2 or 3
/// observations and `first_prediction_idx` is the index in `obs_vec` of
/// the first observation that should be predicted (i.e. the one after the
/// seed).
fn select_seed<'a>(obs_vec: &[&'a Observation]) -> Option<(Vec<&'a Observation>, usize)> {
    if obs_vec.len() < 2 {
        return None;
    }

    for i in 0..obs_vec.len() - 1 {
        let dt_01 = obs_vec[i + 1].mjd_tt() - obs_vec[i].mjd_tt();
        if dt_01 < SAME_NIGHT_THRESHOLD_DAYS {
            // Found a same-night pair at (i, i+1).
            // Try to extend to a triplet.
            if i + 2 < obs_vec.len() {
                let dt_12 = obs_vec[i + 2].mjd_tt() - obs_vec[i + 1].mjd_tt();
                if dt_12 < SAME_NIGHT_THRESHOLD_DAYS {
                    // Triplet: seed = [i, i+1, i+2], predict from i+3.
                    return Some((vec![obs_vec[i], obs_vec[i + 1], obs_vec[i + 2]], i + 3));
                }
            }
            // Pair fallback: seed = [i, i+1], predict from i+2.
            return Some((vec![obs_vec[i], obs_vec[i + 1]], i + 2));
        }
    }

    None
}

// ── Per-trajectory Kalman loop ────────────────────────────────────────────────

/// Run the predict → flag → update loop on a single sorted trajectory.
///
/// The seed is built from the first same-night pair or triplet
/// (see [`select_seed`]).  Every subsequent observation triggers a
/// prediction, an in-box check, and an unconditional Kalman update.
///
/// Arguments
/// ---------
/// * `traj_id`       – Trajectory identifier (used in records and error messages).
/// * `obs_vec`       – Observations sorted by ascending MJD.
/// * `engine_config` – Engine configuration supplying filter parameters.
///
/// Return
/// ------
/// One [`PredictionRecord`] per observation after the seed, or an empty
/// `Vec` if the trajectory is too short (fewer than 3 observations total).
/// Run the predict → flag → update loop on a single sorted trajectory.
///
/// Returns the prediction records **and** the initialised seed tracklet,
/// so the caller can aggregate seed-state diagnostics independently of
/// the Kalman loop.
///
/// Return
/// ------
/// * `Ok((Vec<PredictionRecord>, Option<TrackletData>))` –
///   records after the seed, plus the seed itself (`None` if the trajectory
///   was too short to produce any prediction).
/// * `Err(...)` – seed construction or Kalman update failed.
fn run_trajectory_with_seed(
    traj_id: u32,
    obs_vec: &[&Observation],
    engine_config: &EngineConfig,
) -> Result<(Vec<PredictionRecord>, Option<TrackletData<EclipticState>>)> {
    // ── Seed selection ────────────────────────────────────────────────────────
    let (seed_obs, first_pred_idx) = match select_seed(obs_vec) {
        Some(s) => s,
        None => return Ok((vec![], None)),
    };

    if first_pred_idx >= obs_vec.len() {
        return Ok((vec![], None));
    }

    // ── Tracklet initialisation ───────────────────────────────────────────────
    let seed_tracklet = match seed_obs.len() {
        3 => TrackletData::from_triplet(
            TrackId(traj_id),
            seed_obs[0],
            seed_obs[1],
            seed_obs[2],
            engine_config.pairs.max_angular_speed,
            engine_config.process_noise_q,
            engine_config.singer_params.clone(),
        )
        .or_else(|| {
            TrackletData::from_pair(
                TrackId(traj_id),
                seed_obs[0],
                seed_obs[1],
                engine_config.pairs.acc_prior_var,
                engine_config.pairs.max_angular_speed,
                engine_config.process_noise_q,
                engine_config.singer_params.clone(),
            )
        }),
        _ => TrackletData::from_pair(
            TrackId(traj_id),
            seed_obs[0],
            seed_obs[1],
            engine_config.pairs.acc_prior_var,
            engine_config.pairs.max_angular_speed,
            engine_config.process_noise_q,
            engine_config.singer_params.clone(),
        ),
    }
    .with_context(|| format!("traj {traj_id}: failed to build seed"))?;

    // Clone the seed data before consuming it into Tracklet::Seed.
    let seed_data_clone = seed_tracklet.clone();

    let wrapped = Tracklet::Seed(seed_tracklet);
    let mut state = wrapped.state().unwrap().clone();
    let process_noise_q = wrapped.ecliptic_data().unwrap().state.process_noise_q;

    // ── Predict / update loop ─────────────────────────────────────────────────
    let mut records = Vec::with_capacity(obs_vec.len() - first_pred_idx);
    let mut prev_mjd = seed_obs.last().unwrap().mjd_tt();

    for (obs_idx, obs) in obs_vec.iter().enumerate().skip(first_pred_idx) {
        let t_obs = obs.mjd_tt();
        let gap_days = t_obs - prev_mjd;
        prev_mjd = t_obs;

        let predicted = match &wrapped.ecliptic_data().unwrap().state.singer {
            Some(singer) => state.propagate_singer(t_obs, singer),
            None => state.propagate(t_obs, process_noise_q),
        };

        let pred_lon = predicted.x[0];
        let pred_lat = predicted.x[3];

        let equ_cov = EquCoordCov::from_equ(*obs.equ_coord());
        let ecl_obs = EclipticCoordCov::from(equ_cov);

        let s00 = predicted.p[(0, 0)] + ecl_obs.cov.xx;
        let s01 = predicted.p[(0, 3)] + ecl_obs.cov.xy;
        let s11 = predicted.p[(3, 3)] + ecl_obs.cov.yy;
        let s_det = s00 * s11 - s01 * s01;

        let sigma_lon_pred = rad_to_arcsec(s00.sqrt());
        let sigma_lat_pred = rad_to_arcsec(s11.sqrt());

        let nu_lon = wrap_lon(ecl_obs.coord.lon - pred_lon);
        let nu_lat = ecl_obs.coord.lat - pred_lat;
        let inn_lon = rad_to_arcsec(nu_lon);
        let inn_lat = rad_to_arcsec(nu_lat);

        let nis = if s_det > 0.0 {
            (nu_lon * nu_lon * s11 - 2.0 * nu_lon * nu_lat * s01 + nu_lat * nu_lat * s00) / s_det
        } else {
            f64::NAN
        };

        const N_SIGMA: f64 = 3.0;
        let in_box =
            inn_lon.abs() <= N_SIGMA * sigma_lon_pred && inn_lat.abs() <= N_SIGMA * sigma_lat_pred;

        records.push(PredictionRecord {
            traj_id,
            obs_idx,
            mjd: t_obs,
            gap_days,
            pred_lon,
            pred_lat,
            obs_lon: ecl_obs.coord.lon,
            obs_lat: ecl_obs.coord.lat,
            inn_lon,
            inn_lat,
            sigma_lon_pred,
            sigma_lat_pred,
            in_box,
            nis,
        });

        state = predicted
            .kalman_update(&ecl_obs)
            .with_context(|| format!("traj {traj_id}: Kalman update singular at step {obs_idx}"))?;
        state.epoch = t_obs;
    }

    Ok((records, Some(seed_data_clone)))
}

// ── Statistics ────────────────────────────────────────────────────────────────

/// Print population-level and per-gap-bucket recovery statistics.
///
/// For each time-gap bucket the following quantities are reported:
///
/// - `count`      – number of prediction records in the bucket.
/// - `in_box`     – number of records where `|ν| ≤ 3σ` on both axes.
/// - `recovery%`  – fraction of records in the box (ideal ≈ 99.7 % for 3σ).
/// - `mean σ_lon` – mean predicted 1-sigma in longitude (arcsec), from $\sqrt{S_{00}}$.
/// - `mean σ_lat` – mean predicted 1-sigma in latitude (arcsec), from $\sqrt{S_{11}}$.
/// - `mean NIS`   – mean Normalized Innovation Squared $\nu^\top S^{-1} \nu$.
///
/// Calibration interpretation
/// --------------------------
/// For a Gaussian, well-calibrated filter the NIS follows a $\chi^2$ distribution
/// with $\dim(z) = 2$ degrees of freedom:
///
/// $$\mathbb{E}[\text{NIS}] = 2$$
///
/// - `mean NIS ≫ 2` → filter is **overconfident** ($S$ too small, $Q$ under-estimated).
/// - `mean NIS ≪ 2` → filter is **underconfident** ($S$ too large, $Q$ over-estimated).
fn print_population_stats(records: &[PredictionRecord]) {
    let total = records.len();
    if total == 0 {
        println!("No prediction records to report.");
        return;
    }
    let in_box = records.iter().filter(|r| r.in_box).count();
    let recovery_rate = 100.0 * in_box as f64 / total as f64;

    // Global mean NIS (skip NaN from degenerate S).
    let (nis_sum, nis_count) = records.iter().fold((0.0_f64, 0usize), |(s, n), r| {
        if r.nis.is_finite() {
            (s + r.nis, n + 1)
        } else {
            (s, n)
        }
    });
    let mean_nis_global = if nis_count > 0 {
        nis_sum / nis_count as f64
    } else {
        f64::NAN
    };

    println!("=== Population Kalman Recovery Statistics ===\n");
    println!("  Total observations (after seed) : {total}");
    println!("  In predicted box (3σ)           : {in_box}");
    println!("  Recovery rate                   : {recovery_rate:.2}%");
    println!("  Mean NIS (global)               : {mean_nis_global:.3}  [ideal = 2.000]\n");

    // ── Per-gap-bucket breakdown ──────────────────────────────────────────────
    let buckets: &[(f64, f64, &str)] = &[
        (0.0, 1.0, "[0, 1)"),
        (1.0, 3.0, "[1, 3)"),
        (3.0, 7.0, "[3, 7)"),
        (7.0, 15.0, "[7, 15)"),
        (15.0, f64::INFINITY, "[15, ∞)"),
    ];

    println!(
        "  {:>15}  {:>8}  {:>8}  {:>10}  {:>12}  {:>12}  {:>10}",
        "gap (days)", "count", "in_box", "recovery%", "mean σ_lon\"", "mean σ_lat\"", "mean NIS"
    );
    println!("  {}", "-".repeat(96));

    for &(lo, hi, label) in buckets {
        let bucket: Vec<&PredictionRecord> = records
            .iter()
            .filter(|r| r.gap_days >= lo && r.gap_days < hi)
            .collect();

        if bucket.is_empty() {
            continue;
        }

        let n = bucket.len();
        let n_in = bucket.iter().filter(|r| r.in_box).count();
        let pct = 100.0 * n_in as f64 / n as f64;
        let mean_sigma_lon = bucket.iter().map(|r| r.sigma_lon_pred).sum::<f64>() / n as f64;
        let mean_sigma_lat = bucket.iter().map(|r| r.sigma_lat_pred).sum::<f64>() / n as f64;

        // Mean NIS, excluding NaN (degenerate S).
        let (nis_sum, nis_n) = bucket.iter().fold((0.0_f64, 0usize), |(s, k), r| {
            if r.nis.is_finite() {
                (s + r.nis, k + 1)
            } else {
                (s, k)
            }
        });
        let mean_nis = if nis_n > 0 {
            nis_sum / nis_n as f64
        } else {
            f64::NAN
        };

        println!(
            "  {:>15}  {:>8}  {:>8}  {:>10.1}  {:>12.3}  {:>12.3}  {:>10.3}",
            label, n, n_in, pct, mean_sigma_lon, mean_sigma_lat, mean_nis
        );
    }

    // ── Worst-recovery trajectories ───────────────────────────────────────────
    println!("\n=== Worst-recovery trajectories (top 10) ===\n");
    println!(
        "  {:>10}  {:>6}  {:>6}  {:>10}  {:>10}",
        "traj_id", "n_obs", "in_box", "recovery%", "mean NIS"
    );
    println!("  {}", "-".repeat(52));

    let mut traj_map: HashMap<u32, (usize, usize, f64, usize)> = HashMap::new();
    for r in records {
        let entry = traj_map.entry(r.traj_id).or_insert((0, 0, 0.0, 0));
        entry.0 += 1;
        if r.in_box {
            entry.1 += 1;
        }
        if r.nis.is_finite() {
            entry.2 += r.nis;
            entry.3 += 1;
        }
    }

    let mut traj_stats: Vec<(u32, usize, usize, f64, f64)> = traj_map
        .into_iter()
        .map(|(tid, (n, n_in, nis_sum, nis_n))| {
            let pct = 100.0 * n_in as f64 / n as f64;
            let mean_nis = if nis_n > 0 {
                nis_sum / nis_n as f64
            } else {
                f64::NAN
            };
            (tid, n, n_in, pct, mean_nis)
        })
        .collect();

    traj_stats.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));

    for (tid, n, n_in, pct, mean_nis) in traj_stats.iter().take(10) {
        println!("  {tid:>10}  {n:>6}  {n_in:>6}  {pct:>10.1}  {mean_nis:>10.3}");
    }
}

/// Print seed state diagnostics: σ_pos, σ_vel in arcsec and arcsec/day,
/// and the extrapolated position uncertainty at Δt = 1 day.
pub fn print_seed_diagnostics(tracklets: &[TrackletData<EclipticState>]) {
    use std::f64::consts::PI;
    const RAD_TO_ARCSEC: f64 = 180.0 * 3600.0 / PI;

    let n = tracklets.len();
    if n == 0 {
        println!("No tracklets.");
        return;
    }

    // Collect per-tracklet diagnostics.
    let mut sigma_pos_lon: Vec<f64> = Vec::with_capacity(n);
    let mut sigma_vel_lon: Vec<f64> = Vec::with_capacity(n);
    let mut sigma_pos_1day: Vec<f64> = Vec::with_capacity(n);

    for t in tracklets {
        let sp = t.state.p[(0, 0)].sqrt() * RAD_TO_ARCSEC;
        let sv = t.state.p[(1, 1)].sqrt() * RAD_TO_ARCSEC;
        // Extrapolated position uncertainty at Δt = 1 day (ignoring Q):
        // σ_pos(1) = √(P₀₀ + 2·P₀₁·Δt + P₁₁·Δt²)  with Δt = 1
        let sp1 = (t.state.p[(0, 0)] + 2.0 * t.state.p[(0, 1)] + t.state.p[(1, 1)])
            .max(0.0)
            .sqrt()
            * RAD_TO_ARCSEC;
        sigma_pos_lon.push(sp);
        sigma_vel_lon.push(sv);
        sigma_pos_1day.push(sp1);
    }

    // Percentiles helper.
    let percentile = |v: &mut Vec<f64>, p: f64| -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((p / 100.0) * (v.len() - 1) as f64).round() as usize;
        v[idx]
    };

    println!("=== Seed State Diagnostics ({n} tracklets) ===\n");
    println!(
        "  {:>30}  {:>10}  {:>10}  {:>10}  {:>10}",
        "metric", "p10", "p50", "p90", "p99"
    );
    println!("  {}", "-".repeat(76));

    for (label, vals) in [
        ("σ_pos_lon (arcsec)", &mut sigma_pos_lon),
        ("σ_vel_lon (arcsec/day)", &mut sigma_vel_lon),
        ("σ_pos_lon @ Δt=1d (arcsec)", &mut sigma_pos_1day),
    ] {
        let p10 = percentile(vals, 10.0);
        let p50 = percentile(vals, 50.0);
        let p90 = percentile(vals, 90.0);
        let p99 = percentile(vals, 99.0);
        println!("  {label:>30}  {p10:>10.3}  {p50:>10.3}  {p90:>10.3}  {p99:>10.3}");
    }
    println!();
}

/// Plot a log-scale histogram of per-trajectory mean NIS.
///
/// Each bar spans one decade on the x-axis (log₁₀ scale).
/// The y-axis shows the fraction of trajectories in each bin.
///
/// Arguments
/// ---------
/// * `records`   – All prediction records from the Kalman loop.
/// * `out_path`  – Output PNG file path.
fn plot_nis_histogram(records: &[PredictionRecord], out_path: &str) -> Result<()> {
    // ── Aggregate mean NIS per trajectory ────────────────────────────────────
    let mut traj_nis: HashMap<u32, (f64, usize)> = HashMap::new();
    for r in records {
        if r.nis.is_finite() {
            let e = traj_nis.entry(r.traj_id).or_insert((0.0, 0));
            e.0 += r.nis;
            e.1 += 1;
        }
    }

    let mean_nis_per_traj: Vec<f64> = traj_nis
        .values()
        .filter(|(_, n)| *n > 0)
        .map(|(s, n)| s / *n as f64)
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();

    let total = mean_nis_per_traj.len();
    if total == 0 {
        return Ok(());
    }

    // ── Build log₁₀ bins: one bin per decade from 10⁻¹ to 10⁸ ───────────────
    // Edges: -1.0, 0.0, 1.0, ..., 8.0  →  9 bins.
    const LOG_MIN: f64 = -1.0;
    const LOG_MAX: f64 = 8.0;
    const N_BINS: usize = 18; // 0.5-decade bins
    let bin_width = (LOG_MAX - LOG_MIN) / N_BINS as f64;

    let mut counts = vec![0usize; N_BINS];
    for v in &mean_nis_per_traj {
        let log_v = v.log10();
        let idx = ((log_v - LOG_MIN) / bin_width).floor() as isize;
        let idx = idx.clamp(0, N_BINS as isize - 1) as usize;
        counts[idx] += 1;
    }

    let fractions: Vec<f64> = counts.iter().map(|&c| c as f64 / total as f64).collect();
    let y_max = fractions.iter().cloned().fold(0.0_f64, f64::max) * 1.15;

    // ── Plotters setup ────────────────────────────────────────────────────────
    let root = BitMapBackend::new(out_path, (900, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Per-trajectory mean NIS distribution", ("sans-serif", 20))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(LOG_MIN..LOG_MAX, 0.0..y_max)?;

    chart
        .configure_mesh()
        .x_desc("log₁₀(mean NIS)")
        .y_desc("Fraction of trajectories")
        .x_label_formatter(&|x| {
            // Show tick labels as powers of 10.
            let exp = *x as i32;
            format!("10^{exp}")
        })
        .draw()?;

    // ── Bars ──────────────────────────────────────────────────────────────────
    let bar_color = RGBColor(70, 130, 180).mix(0.8); // steel blue, semi-transparent
    let gap = bin_width * 0.05; // small visual gap between bars

    chart.draw_series(fractions.iter().enumerate().map(|(i, &frac)| {
        let x0 = LOG_MIN + i as f64 * bin_width + gap;
        let x1 = LOG_MIN + (i + 1) as f64 * bin_width - gap;
        Rectangle::new([(x0, 0.0), (x1, frac)], bar_color.filled())
    }))?;

    // ── Reference line at NIS = 2 (ideal for 2-dof filter) ───────────────────
    let log_nis_ideal = 2.0_f64.log10(); // ≈ 0.301
    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(log_nis_ideal, 0.0), (log_nis_ideal, y_max)],
            RED.stroke_width(2),
        )))?
        .label("NIS = 2 (ideal)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 15, y)], RED.stroke_width(2)));

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.8))
        .draw()?;

    root.present()?;
    println!("NIS histogram written to: {out_path}");
    Ok(())
}

/// Plot the innovation distributions for longitude and latitude.
///
/// Two panels side by side:
/// - Left  : overlaid normalized histograms (PDF) in arcsec.
/// - Right : empirical cumulative distribution functions (CDF).
///
/// A well-calibrated filter should show zero-mean, symmetric distributions.
/// Significant offset from zero indicates a systematic model bias (scenario A).
fn plot_innovation_distributions(records: &[PredictionRecord], out_path: &str) -> Result<()> {
    let inn_lon: Vec<f64> = records.iter().map(|r| r.inn_lon).collect();
    let inn_lat: Vec<f64> = records.iter().map(|r| r.inn_lat).collect();

    // Clip to ±50 arcsec for readability.
    const CLIP: f64 = 50.0;
    const N_BINS: usize = 100;

    let bin_width = 2.0 * CLIP / N_BINS as f64;

    let mut hist_lon = vec![0usize; N_BINS];
    let mut hist_lat = vec![0usize; N_BINS];
    let mut n_lon = 0usize;
    let mut n_lat = 0usize;

    for v in &inn_lon {
        if v.abs() <= CLIP {
            let idx = ((*v + CLIP) / bin_width).floor() as usize;
            hist_lon[idx.min(N_BINS - 1)] += 1;
            n_lon += 1;
        }
    }
    for v in &inn_lat {
        if v.abs() <= CLIP {
            let idx = ((*v + CLIP) / bin_width).floor() as usize;
            hist_lat[idx.min(N_BINS - 1)] += 1;
            n_lat += 1;
        }
    }

    let to_frac = |hist: &[usize], n: usize| -> Vec<f64> {
        hist.iter()
            .map(|&c| c as f64 / (n as f64 * bin_width))
            .collect()
    };

    let frac_lon = to_frac(&hist_lon, n_lon.max(1));
    let frac_lat = to_frac(&hist_lat, n_lat.max(1));
    let y_max = frac_lon
        .iter()
        .chain(frac_lat.iter())
        .cloned()
        .fold(0.0_f64, f64::max)
        * 1.15;

    // -------------------------------------------------------------------------
    // Empirical CDF: sort clipped values and compute cumulative fractions.
    // -------------------------------------------------------------------------
    let build_cdf = |values: &[f64]| -> Vec<(f64, f64)> {
        let mut clipped: Vec<f64> = values.iter().copied().filter(|v| v.abs() <= CLIP).collect();
        clipped.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = clipped.len();
        if n == 0 {
            return vec![(-CLIP, 0.0), (CLIP, 0.0)];
        }
        clipped
            .iter()
            .enumerate()
            .map(|(i, &x)| (x, (i + 1) as f64 / n as f64))
            .collect()
    };

    let cdf_lon = build_cdf(&inn_lon);
    let cdf_lat = build_cdf(&inn_lat);

    // -------------------------------------------------------------------------
    // Canvas – two panels side by side.
    // -------------------------------------------------------------------------
    let root = BitMapBackend::new(out_path, (1800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let (left, right) = root.split_horizontally(900);

    let color_lon = RGBColor(70, 130, 180);
    let color_lat = RGBColor(220, 80, 60);

    // =========================================================================
    // LEFT PANEL – PDF histograms
    // =========================================================================
    let mut chart_pdf = ChartBuilder::on(&left)
        .caption(
            "Innovation distributions ν_λ and ν_β (clipped ±50\")",
            ("sans-serif", 28),
        )
        .margin(25)
        .x_label_area_size(70)
        .y_label_area_size(90)
        .build_cartesian_2d(-CLIP..CLIP, 0.0..y_max)?;

    chart_pdf
        .configure_mesh()
        .x_desc("Innovation (arcsec)")
        .y_desc("Probability density")
        .x_label_style(("sans-serif", 22))
        .y_label_style(("sans-serif", 22))
        .axis_desc_style(("sans-serif", 26))
        .light_line_style(RGBColor(220, 225, 235))
        .draw()?;

    // Longitude bars.
    chart_pdf
        .draw_series(frac_lon.iter().enumerate().map(|(i, &f)| {
            let x0 = -CLIP + i as f64 * bin_width;
            let x1 = x0 + bin_width * 0.9;
            Rectangle::new([(x0, 0.0), (x1, f)], color_lon.mix(0.6).filled())
        }))?
        .label("ν_λ  longitude")
        .legend(|(x, y)| {
            Rectangle::new([(x, y - 8), (x + 22, y + 8)], color_lon.mix(0.7).filled())
        });

    // Latitude bars.
    chart_pdf
        .draw_series(frac_lat.iter().enumerate().map(|(i, &f)| {
            let x0 = -CLIP + i as f64 * bin_width;
            let x1 = x0 + bin_width * 0.9;
            Rectangle::new([(x0, 0.0), (x1, f)], color_lat.mix(0.6).filled())
        }))?
        .label("ν_β  latitude")
        .legend(|(x, y)| {
            Rectangle::new([(x, y - 8), (x + 22, y + 8)], color_lat.mix(0.7).filled())
        });

    // Zero line.
    chart_pdf.draw_series(std::iter::once(PathElement::new(
        vec![(0.0, 0.0), (0.0, y_max)],
        Into::<ShapeStyle>::into(BLACK).stroke_width(2),
    )))?;

    chart_pdf
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .border_style(BLACK)
        .background_style(WHITE.mix(0.85))
        .label_font(("sans-serif", 24))
        .draw()?;

    // =========================================================================
    // RIGHT PANEL – empirical CDF
    // =========================================================================
    let mut chart_cdf = ChartBuilder::on(&right)
        .caption(
            "Empirical CDF of innovations (clipped ±50\")",
            ("sans-serif", 28),
        )
        .margin(25)
        .x_label_area_size(70)
        .y_label_area_size(90)
        .build_cartesian_2d(-CLIP..CLIP, 0.0_f64..1.05)?;

    chart_cdf
        .configure_mesh()
        .x_desc("Innovation (arcsec)")
        .y_desc("Cumulative probability")
        .x_label_style(("sans-serif", 22))
        .y_label_style(("sans-serif", 22))
        .axis_desc_style(("sans-serif", 26))
        .y_label_formatter(&|v| format!("{:.2}", v))
        .light_line_style(RGBColor(220, 225, 235))
        .draw()?;

    // CDF – longitude.
    chart_cdf
        .draw_series(LineSeries::new(
            cdf_lon.iter().copied(),
            Into::<ShapeStyle>::into(color_lon).stroke_width(3),
        ))?
        .label("ν_λ  longitude")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 25, y)],
                Into::<ShapeStyle>::into(color_lon).stroke_width(3),
            )
        });

    // CDF – latitude.
    chart_cdf
        .draw_series(LineSeries::new(
            cdf_lat.iter().copied(),
            Into::<ShapeStyle>::into(color_lat).stroke_width(3),
        ))?
        .label("ν_β  latitude")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 25, y)],
                Into::<ShapeStyle>::into(color_lat).stroke_width(3),
            )
        });

    // Zero line.
    chart_cdf.draw_series(std::iter::once(PathElement::new(
        vec![(0.0, 0.0), (0.0, 1.05)],
        Into::<ShapeStyle>::into(BLACK).stroke_width(2),
    )))?;

    // Horizontal reference at 0.5 (median).
    chart_cdf.draw_series(std::iter::once(PathElement::new(
        vec![(-CLIP, 0.5), (CLIP, 0.5)],
        Into::<ShapeStyle>::into(RGBColor(150, 150, 150)).stroke_width(1),
    )))?;

    chart_cdf
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .border_style(BLACK)
        .background_style(WHITE.mix(0.85))
        .label_font(("sans-serif", 24))
        .draw()?;

    root.present()?;
    println!("Innovation distribution plot written to: {out_path}");
    Ok(())
}

/// Plot median absolute innovation as a function of the time gap.
///
/// If the filter model were perfect, innovations should stay flat.
/// A rising curve with gap indicates unmodelled dynamics (orbital curvature).
fn plot_innovation_vs_gap(records: &[PredictionRecord], out_path: &str) -> Result<()> {
    // Gap buckets in days (right edge).
    let edges: &[f64] = &[0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0];

    let mut buckets_lon: Vec<Vec<f64>> = vec![vec![]; edges.len()];
    let mut buckets_lat: Vec<Vec<f64>> = vec![vec![]; edges.len()];

    for r in records {
        for (i, &edge) in edges.iter().enumerate() {
            if r.gap_days <= edge {
                buckets_lon[i].push(r.inn_lon.abs());
                buckets_lat[i].push(r.inn_lat.abs());
                break;
            }
        }
    }

    let median = |v: &mut Vec<f64>| -> Option<f64> {
        if v.is_empty() {
            return None;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(v[v.len() / 2])
    };

    let points_lon: Vec<(f64, f64)> = edges
        .iter()
        .zip(buckets_lon.iter_mut())
        .filter_map(|(&x, v)| median(v).map(|y| (x, y)))
        .collect();
    let points_lat: Vec<(f64, f64)> = edges
        .iter()
        .zip(buckets_lat.iter_mut())
        .filter_map(|(&x, v)| median(v).map(|y| (x, y)))
        .collect();

    let y_max = points_lon
        .iter()
        .chain(points_lat.iter())
        .map(|(_, y)| *y)
        .fold(0.0_f64, f64::max)
        * 1.2;

    let root = BitMapBackend::new(out_path, (900, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Median |innovation| vs time gap", ("sans-serif", 18))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(0.0f64..*edges.last().unwrap(), 0.0..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Gap Δt (days)")
        .y_desc("Median |ν| (arcsec)")
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            points_lon.clone(),
            RGBColor(70, 130, 180).stroke_width(2),
        ))?
        .label("ν_λ (longitude)")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 15, y)],
                RGBColor(70, 130, 180).stroke_width(2),
            )
        });

    chart
        .draw_series(LineSeries::new(
            points_lat.clone(),
            RGBColor(220, 80, 60).stroke_width(2),
        ))?
        .label("ν_β (latitude)")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 15, y)],
                RGBColor(220, 80, 60).stroke_width(2),
            )
        });

    // Points.
    chart.draw_series(
        points_lon
            .iter()
            .map(|&(x, y)| Circle::new((x, y), 4, RGBColor(70, 130, 180).filled())),
    )?;
    chart.draw_series(
        points_lat
            .iter()
            .map(|&(x, y)| Circle::new((x, y), 4, RGBColor(220, 80, 60).filled())),
    )?;

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.8))
        .draw()?;
    root.present()?;
    println!("Innovation vs gap plot written to: {out_path}");
    Ok(())
}

/// Plot mean NIS as a function of time gap.
///
/// Reveals whether miscalibration is uniform or concentrated on long gaps
/// (which would indicate unmodelled orbital dynamics).
/// The horizontal dashed line at NIS = 2 is the ideal target.
fn plot_nis_vs_gap(records: &[PredictionRecord], out_path: &str) -> Result<()> {
    let edges: &[f64] = &[0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0];

    let mut buckets: Vec<Vec<f64>> = vec![vec![]; edges.len()];
    for r in records {
        if r.nis.is_finite() {
            for (i, &edge) in edges.iter().enumerate() {
                if r.gap_days <= edge {
                    buckets[i].push(r.nis);
                    break;
                }
            }
        }
    }

    let points: Vec<(f64, f64)> = edges
        .iter()
        .zip(buckets.iter())
        .filter_map(|(&x, v)| {
            if v.is_empty() {
                return None;
            }
            Some((x, v.iter().sum::<f64>() / v.len() as f64))
        })
        .collect();

    let y_max = points.iter().map(|(_, y)| *y).fold(0.0_f64, f64::max) * 1.15;
    // Cap for readability — very large NIS compresses the interesting region.
    let y_max = y_max.min(200.0);

    let root = BitMapBackend::new(out_path, (900, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Mean NIS vs time gap (capped at 200)", ("sans-serif", 18))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(0.0f64..*edges.last().unwrap(), 0.0..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Gap Δt (days)")
        .y_desc("Mean NIS")
        .draw()?;

    // NIS = 2 reference line.
    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(0.0, 2.0), (*edges.last().unwrap(), 2.0)],
            RED.stroke_width(2),
        )))?
        .label("NIS = 2 (ideal)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 15, y)], RED.stroke_width(2)));

    chart
        .draw_series(LineSeries::new(
            points.clone(),
            RGBColor(70, 130, 180).stroke_width(2),
        ))?
        .label("Mean NIS")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 15, y)],
                RGBColor(70, 130, 180).stroke_width(2),
            )
        });

    chart.draw_series(
        points
            .iter()
            .map(|&(x, y)| Circle::new((x, y.min(y_max)), 4, RGBColor(70, 130, 180).filled())),
    )?;

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.8))
        .draw()?;
    root.present()?;
    println!("NIS vs gap plot written to: {out_path}");
    Ok(())
}

/// Plot the ratio |ν| / σ_pred vs gap, per axis.
///
/// This is the normalized innovation amplitude (without squaring).
/// A well-calibrated filter should have median |ν|/σ ≈ 0.8 (half-normal).
/// When this ratio grows with Δt, the model bias dominates over measurement noise.
///
/// The horizontal lines mark:
/// - ratio = 1 : innovation equals 1σ prediction
/// - ratio = 3 : the 3σ gate threshold
fn plot_normalized_innovation_vs_gap(records: &[PredictionRecord], out_path: &str) -> Result<()> {
    // Bin edges in days.
    let edges: Vec<f64> = (0..=60).map(|i| i as f64 * 0.5).collect();

    let mut bins_lon: Vec<Vec<f64>> = vec![vec![]; edges.len() - 1];
    let mut bins_lat: Vec<Vec<f64>> = vec![vec![]; edges.len() - 1];

    for r in records {
        if r.sigma_lon_pred > 0.0 && r.sigma_lat_pred > 0.0 {
            let ratio_lon = r.inn_lon.abs() / r.sigma_lon_pred;
            let ratio_lat = r.inn_lat.abs() / r.sigma_lat_pred;

            for (i, w) in edges.windows(2).enumerate() {
                if r.gap_days >= w[0] && r.gap_days < w[1] {
                    bins_lon[i].push(ratio_lon);
                    bins_lat[i].push(ratio_lat);
                    break;
                }
            }
        }
    }

    // Median per bin.
    let median = |v: &mut Vec<f64>| -> Option<f64> {
        if v.is_empty() {
            return None;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(v[v.len() / 2])
    };

    let points_lon: Vec<(f64, f64)> = bins_lon
        .iter_mut()
        .enumerate()
        .filter_map(|(i, v)| median(v).map(|m| (edges[i] + 0.25, m)))
        .collect();

    let points_lat: Vec<(f64, f64)> = bins_lat
        .iter_mut()
        .enumerate()
        .filter_map(|(i, v)| median(v).map(|m| (edges[i] + 0.25, m)))
        .collect();

    let y_max = points_lon.iter().chain(points_lat.iter())
        .map(|(_, y)| *y)
        .fold(0.0_f64, f64::max)
        .min(50.0) // cap for readability
        * 1.15;

    let x_max = *edges.last().unwrap();

    let root = BitMapBackend::new(out_path, (1000, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Median |ν|/σ_pred vs time gap", ("sans-serif", 18))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(0.0f64..x_max, 0.0f64..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Gap Δt (days)")
        .y_desc("|ν| / σ_pred  (median)")
        .draw()?;

    // Reference lines.
    for (thresh, color, label) in [(1.0_f64, &MAGENTA, "1σ gate"), (3.0_f64, &RED, "3σ gate")] {
        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![(0.0, thresh), (x_max, thresh)],
                color.stroke_width(2),
            )))?
            .label(label)
            .legend({
                let color = *color;
                move |(x, y)| PathElement::new(vec![(x, y), (x + 15, y)], color.stroke_width(2))
            });
    }

    // Longitude curve.
    chart
        .draw_series(LineSeries::new(
            points_lon.clone(),
            RGBColor(70, 130, 180).stroke_width(2),
        ))?
        .label("ν_λ / σ_λ")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 15, y)],
                RGBColor(70, 130, 180).stroke_width(2),
            )
        });

    chart.draw_series(
        points_lon
            .iter()
            .map(|&(x, y)| Circle::new((x, y.min(y_max)), 3, RGBColor(70, 130, 180).filled())),
    )?;

    // Latitude curve.
    chart
        .draw_series(LineSeries::new(
            points_lat.clone(),
            RGBColor(200, 80, 50).stroke_width(2),
        ))?
        .label("ν_β / σ_β")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 15, y)],
                RGBColor(200, 80, 50).stroke_width(2),
            )
        });

    chart.draw_series(
        points_lat
            .iter()
            .map(|&(x, y)| Circle::new((x, y.min(y_max)), 3, RGBColor(200, 80, 50).filled())),
    )?;

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.8))
        .draw()?;

    root.present()?;
    println!("Normalized innovation vs gap written to: {out_path}");
    Ok(())
}

/// Plot recovery diagnostics: in-box rate per trajectory.
///
/// Two panels:
/// - Left  : histogram of per-trajectory in-box recovery rate (%).
///           A well-calibrated filter should be peaked near 99.7 %
///           (the expected 3-σ containment for a Gaussian).
/// - Right : scatter plot of recovery rate (%) vs mean NIS per trajectory.
///           Trajectories with low recovery and high NIS are the most
///           miscalibrated; those with low recovery but NIS ≈ 2 suggest
///           a biased mean rather than inflated variance.
///
/// Arguments
/// ---------
/// * `records`  – All prediction records from the Kalman loop.
/// * `out_path` – Output PNG file path.
/// Plot recovery diagnostics: per-trajectory in-box rate and sigma containment vs gap.
///
/// Two panels side by side:
///
/// - Left  : histogram of per-trajectory in-box recovery rate (3-σ gate).
///           A well-calibrated filter should be peaked near 99.7 %
///           (the expected 3-σ containment for a Gaussian).
///
/// - Right : fraction of observations contained within the k-σ prediction
///           ellipse (k = 1, 2, 3) as a function of the time gap Δt.
///           The dashed horizontal references show the theoretical Gaussian
///           containment: 68.3 % (1-σ), 95.4 % (2-σ), 99.7 % (3-σ).
///           Divergence from these references as Δt grows indicates that
///           the predicted uncertainty is no longer representative of the
///           true prediction error.
///
/// Arguments
/// ---------
/// * `records`  – All prediction records from the Kalman loop.
/// * `out_path` – Output PNG file path.
fn plot_recovery_diagnostics(records: &[PredictionRecord], out_path: &str) -> Result<()> {
    // =========================================================================
    // LEFT PANEL data – per-trajectory recovery rate histogram
    // =========================================================================
    let mut traj_map: HashMap<u32, (usize, usize)> = HashMap::new();
    for r in records {
        let e = traj_map.entry(r.traj_id).or_insert((0, 0));
        e.0 += 1;
        if r.in_box {
            e.1 += 1;
        }
    }

    let recovery_pcts: Vec<f64> = traj_map
        .values()
        .filter(|(n, _)| *n > 0)
        .map(|(n, n_in)| 100.0 * *n_in as f64 / *n as f64)
        .collect();

    if recovery_pcts.is_empty() {
        return Ok(());
    }

    let n_traj = recovery_pcts.len();

    const N_BINS: usize = 20;
    const BIN_W: f64 = 5.0;

    let mut hist = vec![0usize; N_BINS];
    for &pct in &recovery_pcts {
        let idx = (pct / BIN_W).floor() as usize;
        hist[idx.min(N_BINS - 1)] += 1;
    }
    let hist_frac: Vec<f64> = hist.iter().map(|&c| c as f64 / n_traj as f64).collect();
    let hist_y_max = hist_frac.iter().cloned().fold(0.0_f64, f64::max) * 1.2;

    let mut sorted_pcts = recovery_pcts.clone();
    sorted_pcts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_pct = sorted_pcts[sorted_pcts.len() / 2];
    let mean_pct = sorted_pcts.iter().sum::<f64>() / n_traj as f64;

    // =========================================================================
    // RIGHT PANEL data – k-σ containment fraction vs time gap
    // =========================================================================
    // Bin edges: 0.5-day bins up to 30 days.
    let edges: Vec<f64> = (0..=60).map(|i| i as f64 * 0.5).collect();
    let n_gap_bins = edges.len() - 1;

    // For each bin: count total records and those within 1-σ, 2-σ, 3-σ on
    // both axes simultaneously.
    let mut bin_total = vec![0usize; n_gap_bins];
    let mut bin_1s = vec![0usize; n_gap_bins];
    let mut bin_2s = vec![0usize; n_gap_bins];
    let mut bin_3s = vec![0usize; n_gap_bins];

    for r in records {
        if r.sigma_lon_pred <= 0.0 || r.sigma_lat_pred <= 0.0 {
            continue;
        }
        // Find the gap bin.
        let bin_idx = edges.windows(2).enumerate().find_map(|(i, w)| {
            if r.gap_days >= w[0] && r.gap_days < w[1] {
                Some(i)
            } else {
                None
            }
        });
        let Some(bi) = bin_idx else { continue };

        // Normalized innovations on each axis.
        let z_lon = r.inn_lon.abs() / r.sigma_lon_pred;
        let z_lat = r.inn_lat.abs() / r.sigma_lat_pred;

        bin_total[bi] += 1;
        if z_lon <= 1.0 && z_lat <= 1.0 {
            bin_1s[bi] += 1;
        }
        if z_lon <= 2.0 && z_lat <= 2.0 {
            bin_2s[bi] += 1;
        }
        if z_lon <= 3.0 && z_lat <= 3.0 {
            bin_3s[bi] += 1;
        }
    }

    // Convert to fraction; keep only bins with enough samples.
    const MIN_SAMPLES: usize = 10;
    let mid_points: Vec<f64> = edges.windows(2).map(|w| (w[0] + w[1]) * 0.5).collect();

    let frac_series = |counts: &[usize]| -> Vec<(f64, f64)> {
        counts
            .iter()
            .enumerate()
            .filter(|(i, _)| bin_total[*i] >= MIN_SAMPLES)
            .map(|(i, &c)| (mid_points[i], 100.0 * c as f64 / bin_total[i] as f64))
            .collect()
    };

    let pts_1s = frac_series(&bin_1s);
    let pts_2s = frac_series(&bin_2s);
    let pts_3s = frac_series(&bin_3s);

    let x_max = pts_1s
        .iter()
        .chain(pts_2s.iter())
        .chain(pts_3s.iter())
        .map(|(x, _)| *x)
        .fold(0.0_f64, f64::max)
        + 0.5;

    // =========================================================================
    // Canvas
    // =========================================================================
    let root = BitMapBackend::new(out_path, (1800, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    let (left, right) = root.split_horizontally(900);

    let color_bar = RGBColor(70, 130, 180);
    let color_ref = RGBColor(180, 60, 60);
    let color_mean = RGBColor(60, 160, 80);

    let color_1s = RGBColor(70, 130, 180); // blue
    let color_2s = RGBColor(60, 160, 80); // green
    let color_3s = RGBColor(200, 80, 50); // red-orange

    // =========================================================================
    // LEFT PANEL – per-trajectory recovery histogram
    // =========================================================================
    let mut chart_hist = ChartBuilder::on(&left)
        .caption(
            "Per-trajectory in-box recovery rate (3-σ gate)",
            ("sans-serif", 30),
        )
        .margin(30)
        .x_label_area_size(75)
        .y_label_area_size(95)
        .build_cartesian_2d(0.0_f64..100.0_f64, 0.0_f64..hist_y_max)?;

    chart_hist
        .configure_mesh()
        .x_desc("Recovery rate (%)")
        .y_desc("Fraction of trajectories")
        .x_label_style(("sans-serif", 22))
        .y_label_style(("sans-serif", 22))
        .axis_desc_style(("sans-serif", 26))
        .x_label_formatter(&|v| format!("{:.0}%", v))
        .y_label_formatter(&|v| format!("{:.2}", v))
        .light_line_style(RGBColor(220, 225, 235))
        .draw()?;

    // Bars.
    chart_hist
        .draw_series(hist_frac.iter().enumerate().map(|(i, &f)| {
            let x0 = i as f64 * BIN_W;
            let x1 = x0 + BIN_W * 0.85;
            Rectangle::new([(x0, 0.0), (x1, f)], color_bar.mix(0.7).filled())
        }))?
        .label(format!("Trajectories  (n = {n_traj})"))
        .legend(|(x, y)| {
            Rectangle::new([(x, y - 9), (x + 24, y + 9)], color_bar.mix(0.7).filled())
        });

    // Reference line at 99.7 %.
    chart_hist
        .draw_series(std::iter::once(PathElement::new(
            vec![(99.7, 0.0), (99.7, hist_y_max)],
            Into::<ShapeStyle>::into(color_ref).stroke_width(3),
        )))?
        .label("3-σ ideal  (99.7 %)")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 28, y)],
                Into::<ShapeStyle>::into(color_ref).stroke_width(3),
            )
        });

    // Median line.
    chart_hist
        .draw_series(std::iter::once(PathElement::new(
            vec![(median_pct, 0.0), (median_pct, hist_y_max)],
            Into::<ShapeStyle>::into(color_mean).stroke_width(2),
        )))?
        .label(format!("Median  {median_pct:.1} %"))
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 28, y)],
                Into::<ShapeStyle>::into(color_mean).stroke_width(2),
            )
        });

    // Mean line (dashed via short segments).
    chart_hist
        .draw_series((0..40).map(|i| {
            let y0 = i as f64 * hist_y_max / 40.0;
            let y1 = y0 + hist_y_max / 80.0;
            PathElement::new(
                vec![(mean_pct, y0), (mean_pct, y1)],
                Into::<ShapeStyle>::into(RGBColor(120, 120, 120)).stroke_width(2),
            )
        }))?
        .label(format!("Mean  {mean_pct:.1} %"))
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 28, y)],
                Into::<ShapeStyle>::into(RGBColor(120, 120, 120)).stroke_width(2),
            )
        });

    chart_hist
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .border_style(BLACK)
        .background_style(WHITE.mix(0.85))
        .label_font(("sans-serif", 24))
        .draw()?;

    // =========================================================================
    // RIGHT PANEL – k-σ containment fraction vs Δt
    // =========================================================================
    let mut chart_sigma = ChartBuilder::on(&right)
        .caption("Containment fraction vs time gap Δt", ("sans-serif", 30))
        .margin(30)
        .x_label_area_size(75)
        .y_label_area_size(95)
        .build_cartesian_2d(0.0_f64..x_max, 0.0_f64..105.0_f64)?;

    chart_sigma
        .configure_mesh()
        .x_desc("Gap Δt (days)")
        .y_desc("Containment fraction (%)")
        .x_label_style(("sans-serif", 22))
        .y_label_style(("sans-serif", 22))
        .axis_desc_style(("sans-serif", 26))
        .y_label_formatter(&|v| format!("{:.0}%", v))
        .light_line_style(RGBColor(220, 225, 235))
        .draw()?;

    // Theoretical Gaussian containment references (dashed via segments).
    let draw_dashed_h = |chart: &mut ChartContext<_, _>, y: f64, color: RGBColor| -> Result<()> {
        let n_seg = 60usize;
        chart.draw_series((0..n_seg).filter_map(|i| {
            if i % 2 == 0 {
                let x0 = i as f64 * x_max / n_seg as f64;
                let x1 = (i + 1) as f64 * x_max / n_seg as f64;
                Some(PathElement::new(
                    vec![(x0, y), (x1, y)],
                    Into::<ShapeStyle>::into(color).stroke_width(2),
                ))
            } else {
                None
            }
        }))?;
        Ok(())
    };

    draw_dashed_h(&mut chart_sigma, 68.27, color_1s)?;
    draw_dashed_h(&mut chart_sigma, 95.45, color_2s)?;
    draw_dashed_h(&mut chart_sigma, 99.73, color_3s)?;

    // 3-σ curve.
    chart_sigma
        .draw_series(LineSeries::new(
            pts_3s.iter().copied(),
            Into::<ShapeStyle>::into(color_3s).stroke_width(3),
        ))?
        .label("3-σ  (ideal 99.7 %)")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 28, y)],
                Into::<ShapeStyle>::into(color_3s).stroke_width(3),
            )
        });

    chart_sigma.draw_series(
        pts_3s
            .iter()
            .map(|&(x, y)| Circle::new((x, y), 4, color_3s.filled())),
    )?;

    // 2-σ curve.
    chart_sigma
        .draw_series(LineSeries::new(
            pts_2s.iter().copied(),
            Into::<ShapeStyle>::into(color_2s).stroke_width(3),
        ))?
        .label("2-σ  (ideal 95.4 %)")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 28, y)],
                Into::<ShapeStyle>::into(color_2s).stroke_width(3),
            )
        });

    chart_sigma.draw_series(
        pts_2s
            .iter()
            .map(|&(x, y)| Circle::new((x, y), 4, color_2s.filled())),
    )?;

    // 1-σ curve.
    chart_sigma
        .draw_series(LineSeries::new(
            pts_1s.iter().copied(),
            Into::<ShapeStyle>::into(color_1s).stroke_width(3),
        ))?
        .label("1-σ  (ideal 68.3 %)")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 28, y)],
                Into::<ShapeStyle>::into(color_1s).stroke_width(3),
            )
        });

    chart_sigma.draw_series(
        pts_1s
            .iter()
            .map(|&(x, y)| Circle::new((x, y), 4, color_1s.filled())),
    )?;

    chart_sigma
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerLeft)
        .border_style(BLACK)
        .background_style(WHITE.mix(0.85))
        .label_font(("sans-serif", 24))
        .draw()?;

    root.present()?;
    println!("Recovery diagnostics plot written to: {out_path}");
    Ok(())
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();

    let engine_config = load_config(&cli.config)?;
    let (df, obs_dataset) = load_data(&cli.alerts)?;

    println!("Schema: {:?}", df.schema());
    println!("First rows:\n{}", df.head(Some(5)));
    println!("Dataframe and obs_dataset loaded");
    println!("ObsDataset: \n{}", obs_dataset);

    // ── Group observations by trajectory via the obs_dataset index ────────────
    println!("groupby trajectory_id");

    let mut traj_ids: Vec<u32> = obs_dataset
        .iter_traj_id()
        .expect("no trajectory index in obs_dataset")
        .filter_map(|tid| match tid {
            TrajId::Int(v) => Some(*v),
            _ => None,
        })
        .collect();

    // Sort for deterministic output.
    traj_ids.sort_unstable();

    let mut total_obs = 0usize;
    let mut loaded = 0usize;
    let mut skipped_load = 0usize;

    // Materialise trajectories eagerly so we can report counts before the Kalman loop.
    let mut trajectories: Vec<(u32, Vec<&Observation>)> = Vec::with_capacity(traj_ids.len());

    for tid in &traj_ids {
        match obs_dataset.materialize_trajectory(*tid) {
            None => {
                skipped_load += 1;
            }
            Some(mem_obs) => {
                let mut obs: Vec<&Observation> = mem_obs.iter().collect();
                obs.sort_by(|a, b| {
                    a.mjd_tt()
                        .partial_cmp(&b.mjd_tt())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                total_obs += obs.len();
                loaded += 1;
                trajectories.push((*tid, obs));
            }
        }
    }

    println!(
        "Loaded {} trajectories ({} total observations)",
        loaded, total_obs
    );
    println!("Skipped trajectories (load errors): {}", skipped_load);

    let size_dist: std::collections::BTreeMap<usize, usize> =
        trajectories
            .iter()
            .fold(std::collections::BTreeMap::new(), |mut acc, (_, obs)| {
                *acc.entry(obs.len()).or_insert(0) += 1;
                acc
            });
    println!("Trajectory size distribution: {:?}", size_dist);

    // ── Filter trajectories too short to seed ────────────────────────────────────
    const MIN_OBS_FOR_SEED: usize = 3; // adjust to match seed builder requirement

    let (valid_trajectories, short_trajectories): (Vec<_>, Vec<_>) = trajectories
        .into_iter()
        .partition(|(_, obs)| obs.len() >= MIN_OBS_FOR_SEED);

    println!(
        "Filtered {} trajectories with fewer than {} observations",
        short_trajectories.len(),
        MIN_OBS_FOR_SEED
    );
    println!(
        "Remaining: {} trajectories ({} total observations)",
        valid_trajectories.len(),
        valid_trajectories
            .iter()
            .map(|(_, o)| o.len())
            .sum::<usize>()
    );

    // ── Run Kalman loop on every trajectory ───────────────────────────────────
    let mut all_records: Vec<PredictionRecord> = Vec::new();
    let mut all_seeds: Vec<TrackletData<EclipticState>> = Vec::new(); // ← add this
    let mut skipped_kalman = 0usize;
    const MAX_PRINT: usize = 20;

    for (traj_id, obs_vec) in &valid_trajectories {
        match run_trajectory_with_seed(*traj_id, obs_vec, &engine_config) {
            Ok((records, seed)) => {
                all_records.extend(records);
                if let Some(s) = seed {
                    all_seeds.push(s); // ← collect seed
                }
            }
            Err(e) => {
                if skipped_kalman < MAX_PRINT {
                    eprintln!("traj {traj_id}: skipped ({e:#})");
                }
                skipped_kalman += 1;
            }
        }
    }

    println!("Skipped trajectories (Kalman errors): {skipped_kalman}\n");

    // ── Seed diagnostics ──────────────────────────────────────────────────────
    print_seed_diagnostics(&all_seeds);

    // ── Population statistics ─────────────────────────────────────────────────
    print_population_stats(&all_records);

    plot_nis_histogram(&all_records, "nis_histogram.png")?;

    plot_innovation_distributions(&all_records, "innovation_distributions.png")?;
    plot_innovation_vs_gap(&all_records, "innovation_vs_gap.png")?;
    plot_nis_vs_gap(&all_records, "nis_vs_gap.png")?;

    plot_normalized_innovation_vs_gap(&all_records, "normalized_ninnovation_vs_gap.png")?;

    plot_recovery_diagnostics(&all_records, "recovery.png")?;

    Ok(())
}
