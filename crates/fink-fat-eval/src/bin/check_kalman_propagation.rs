use anyhow::{Context, Result};
use camino::{Utf8Path, Utf8PathBuf};
use clap::Parser;
use fink_fat_engine::{
    engine_config::{EngineConfig, load_engine_config_validated},
    tracklet::{Tracklet, track_storage::TrackId, tracklet_data::TrackletData},
};
use hifitime::Epoch;
use photom::{
    coordinates::{
        ecliptic::EclipticCoordCov,
        equatorial::{EquCoord, EquCoordCov},
    },
    io::polars::{ContiguousChoice, FromPolarsArgs},
    observation_dataset::{ObsDataset, observation::ObservationInput},
    observer::{dataset::ObserverId, error_model::ObsErrorModel},
    photometry::{Filter, Photometry},
};

use polars::{
    frame::DataFrame,
    lazy::frame::{LazyFrame, ScanArgsParquet},
};
use serde::Deserialize;
use std::f64::consts::PI;
use std::fs;

/// FINK-FAT: Fink Asteroid Tracker
#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Path to the file containing the night's alerts
    #[arg(short, long, value_name = "ALERTS_FILE")]
    pub alerts: Utf8PathBuf,

    /// Path to the fink-fat configuration file
    #[arg(short, long, value_name = "CONFIG_FILE")]
    pub config: Utf8PathBuf,
}

/// Raw alert record as stored in the JSON file.
#[derive(Debug, Deserialize)]
struct ZtfAlert {
    candid: u64,
    ra: f64,
    dec: f64,
    jd: f64,
    magpsf: f64,
    sigmapsf: f64,
    fid: u32,
}

/// Convert degrees to radians.
#[inline]
fn deg_to_rad(deg: f64) -> f64 {
    deg * PI / 180.0
}

/// 1 arcsecond in radians.
const ONE_ARCSEC_RAD: f64 = PI / (180.0 * 3600.0);

/// Parse a ZTF Julian Date (UTC) into a Modified Julian Date in Terrestrial Time (TT).
///
/// ZTF timestamps are given as Julian Dates in UTC.  The conversion chain is:
///
/// $\text{JD}_\text{UTC} \xrightarrow{-2400000.5} \text{MJD}_\text{UTC} \xrightarrow{+\Delta AT + 32.184\,\text{s}} \text{MJD}_\text{TT}$
///
/// where $\Delta AT$ is the current number of UTC leap seconds.
/// `hifitime` handles the leap-second table internally.
///
/// # Arguments
/// - `jd_utc` — Julian Date in UTC (days), as provided by ZTF alerts.
///
/// # Returns
/// Modified Julian Date in Terrestrial Time (days).
fn jd_utc_to_mjd_tt(jd_utc: f64) -> f64 {
    Epoch::from_jde_utc(jd_utc).to_mjd_tt_days()
}

/// ZTF filter id to filter label.
fn ztf_fid_to_filter(fid: u32) -> Filter {
    match fid {
        1 => Filter::String("g".to_string()),
        2 => Filter::String("r".to_string()),
        3 => Filter::String("i".to_string()),
        other => Filter::Int(other),
    }
}

fn build_obs_dataset_from_json(path: &str) -> Result<ObsDataset> {
    let content = fs::read_to_string(path)?;
    let alerts: Vec<ZtfAlert> = serde_json::from_str(&content)?;

    let observer = Some(ObserverId::MpcCode(*b"I41"));

    let inputs: Vec<ObservationInput> = alerts
        .into_iter()
        .map(|alert| {
            let equ_coord = EquCoord::new(
                deg_to_rad(alert.ra),
                ONE_ARCSEC_RAD,
                deg_to_rad(alert.dec),
                ONE_ARCSEC_RAD,
            );
            let photometry = Photometry {
                magnitude: alert.magpsf,
                error: alert.sigmapsf,
                filter: ztf_fid_to_filter(alert.fid),
            };
            let mjd_tt = jd_utc_to_mjd_tt(alert.jd);
            ObservationInput::new(alert.candid, equ_coord, photometry, mjd_tt, observer)
        })
        .collect();

    let (dataset, _indices) = ObsDataset::empty().push_observation(inputs)?;
    Ok(dataset)
}

pub fn load_config(config_path: &Utf8Path) -> Result<EngineConfig> {
    load_engine_config_validated(config_path).context("failed to load engine config")
}

/// One step of the predict-update cycle.
///
/// Prints a summary row with:
/// - observation index,
/// - MJD of the observation,
/// - predicted position $(\lambda, \beta)$ in degrees,
/// - observed position $(\lambda, \beta)$ in degrees,
/// - innovation $(\Delta\lambda, \Delta\beta)$ in arcseconds,
/// - marginal 1-σ uncertainties after the update in arcseconds.
fn print_step_header() {
    println!(
        "{:>4}  {:>12}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "idx",
        "mjd",
        "pred_lon°",
        "pred_lat°",
        "obs_lon°",
        "obs_lat°",
        "inn_lon\"",
        "inn_lat\"",
        "σ_lon\"",
        "σ_lat\"",
    );
    println!("{}", "-".repeat(116));
}

fn rad_to_deg(r: f64) -> f64 {
    r * 180.0 / PI
}

fn rad_to_arcsec(r: f64) -> f64 {
    r * 180.0 * 3600.0 / PI
}

fn plot_innovations(
    innovations: &[(f64, f64, f64)], // (mjd, inn_lon_arcsec, inn_lat_arcsec)
    output_path: &str,
) -> Result<()> {
    use plotters::prelude::*;

    let root = BitMapBackend::new(output_path, (1920, 1080)).into_drawing_area();
    root.fill(&WHITE)?;

    let (upper, lower) = root.split_vertically(350);

    let mjd_min = innovations
        .iter()
        .map(|(t, _, _)| *t)
        .fold(f64::INFINITY, f64::min);
    let mjd_max = innovations
        .iter()
        .map(|(t, _, _)| *t)
        .fold(f64::NEG_INFINITY, f64::max);
    let t_margin = (mjd_max - mjd_min) * 0.02;

    // Floor for log scale: 1e-3 arcsec
    let log_min: f64 = -3.0;

    let inn_lon_log_max = innovations
        .iter()
        .map(|(_, il, _)| il.abs().max(1e-3).log10())
        .fold(f64::NEG_INFINITY, f64::max)
        * 1.15;
    let inn_lat_log_max = innovations
        .iter()
        .map(|(_, _, ib)| ib.abs().max(1e-3).log10())
        .fold(f64::NEG_INFINITY, f64::max)
        * 1.15;

    let draw_panel = |area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
                      title: &str,
                      y_desc: &str,
                      log_y_max: f64,
                      values: &[(f64, f64)],
                      color_pos: RGBColor,
                      color_neg: RGBColor|
     -> Result<()> {
        let mut chart = ChartBuilder::on(area)
            .caption(title, ("sans-serif", 18))
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(80)
            .build_cartesian_2d(
                (mjd_min - t_margin)..(mjd_max + t_margin),
                log_min..log_y_max,
            )?;

        chart
            .configure_mesh()
            .x_desc("MJD (TT)")
            .y_desc(y_desc)
            .x_label_formatter(&|v| format!("{:.1}", v))
            .y_label_formatter(&|v| {
                let arcsec = 10_f64.powf(*v);
                if arcsec >= 100.0 {
                    format!("{:.0}\"", arcsec)
                } else if arcsec >= 1.0 {
                    format!("{:.2}\"", arcsec)
                } else {
                    format!("{:.4}\"", arcsec)
                }
            })
            .draw()?;

        // Reference lines: 1" (log10 = 0.0) and 10" (log10 = 1.0)
        let references: &[(f64, &str, RGBColor)] = &[
            (0.0, "1\" reference", RGBColor(255, 0, 255)),
            (1.0, "10\" reference", RGBColor(160, 32, 240)),
        ];

        for (log_val, _label, color) in references.iter() {
            chart.draw_series(LineSeries::new(
                [
                    (mjd_min - t_margin, *log_val),
                    (mjd_max + t_margin, *log_val),
                ],
                color.mix(0.6).stroke_width(1),
            ))?;
        }

        // Stems and dots, color-coded by sign
        for (t, v) in values.iter() {
            let log_abs = v.abs().max(1e-3).log10();
            let color = if *v >= 0.0 { color_pos } else { color_neg };

            // Stem from log_min to the point
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(*t, log_min), (*t, log_abs)],
                color.mix(0.25).stroke_width(1),
            )))?;

            // Dot at the point
            chart.draw_series(std::iter::once(Circle::new(
                (*t, log_abs),
                3,
                color.filled(),
            )))?;
        }

        // Legend
        chart
            .draw_series(std::iter::once(Circle::new(
                (mjd_min, log_min),
                0,
                color_pos.filled(),
            )))?
            .label("positive innovation")
            .legend(move |(x, y)| Circle::new((x, y), 4, color_pos.filled()));

        chart
            .draw_series(std::iter::once(Circle::new(
                (mjd_min, log_min),
                0,
                color_neg.filled(),
            )))?
            .label("negative innovation")
            .legend(move |(x, y)| Circle::new((x, y), 4, color_neg.filled()));

        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![(mjd_min, log_min), (mjd_min, log_min)],
                RGBColor(255, 0, 255).mix(0.6).stroke_width(1),
            )))?
            .label("1\" reference")
            .legend(|(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 20, y)],
                    RGBColor(255, 0, 255).mix(0.6).stroke_width(1),
                )
            });

        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![(mjd_min, log_min), (mjd_min, log_min)],
                RGBColor(160, 32, 240).mix(0.6).stroke_width(1),
            )))?
            .label("10\" reference")
            .legend(|(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 20, y)],
                    RGBColor(160, 32, 240).mix(0.6).stroke_width(1),
                )
            });

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;

        Ok(())
    };

    let lon_values: Vec<(f64, f64)> = innovations.iter().map(|(t, il, _)| (*t, *il)).collect();
    let lat_values: Vec<(f64, f64)> = innovations.iter().map(|(t, _, ib)| (*t, *ib)).collect();

    draw_panel(
        &upper,
        "Innovation |Δλ| (log scale, sign encoded by color)",
        "log₁₀ |Δλ| (arcsec)",
        inn_lon_log_max,
        &lon_values,
        BLUE,
        RGBColor(255, 140, 0),
    )?;

    draw_panel(
        &lower,
        "Innovation |Δβ| (log scale, sign encoded by color)",
        "log₁₀ |Δβ| (arcsec)",
        inn_lat_log_max,
        &lat_values,
        RGBColor(0, 160, 0),
        RED,
    )?;

    root.present()?;
    Ok(())
}

fn analyze_cadence_vs_innovation(
    innovations: &[(f64, f64, f64)], // (mjd, inn_lon_arcsec, inn_lat_arcsec)
) {
    println!("\n=== Cadence & Innovation Analysis ===\n");

    // Compute inter-observation gaps
    println!(
        "{:>6}  {:>14}  {:>12}  {:>12}  {:>12}",
        "idx", "mjd", "gap_days", "inn_lon\"", "inn_lat\""
    );
    println!("{}", "-".repeat(62));

    let mut night_gaps: Vec<(usize, f64, f64, f64, f64)> = Vec::new(); // (idx, gap, inn_lon, inn_lat, mjd)

    for i in 1..innovations.len() {
        let (t_prev, _, _) = innovations[i - 1];
        let (t_curr, inn_lon, inn_lat) = innovations[i];
        let gap = t_curr - t_prev;

        // Only keep inter-night gaps (> 0.5 day)
        if gap > 0.5 {
            night_gaps.push((i, gap, inn_lon, inn_lat, t_curr));
            println!(
                "{:>6}  {:>14.6}  {:>12.4}  {:>12.3}  {:>12.3}",
                i, t_curr, gap, inn_lon, inn_lat
            );
        }
    }

    println!("\n=== Statistics on inter-night gaps ===\n");

    let n = night_gaps.len() as f64;
    let mean_gap = night_gaps.iter().map(|(_, g, _, _, _)| g).sum::<f64>() / n;
    let min_gap = night_gaps
        .iter()
        .map(|(_, g, _, _, _)| *g)
        .fold(f64::INFINITY, f64::min);
    let max_gap = night_gaps
        .iter()
        .map(|(_, g, _, _, _)| *g)
        .fold(f64::NEG_INFINITY, f64::max);

    println!("  Number of inter-night gaps : {}", night_gaps.len());
    println!("  Mean gap                   : {:.3} days", mean_gap);
    println!("  Min gap                    : {:.3} days", min_gap);
    println!("  Max gap                    : {:.3} days", max_gap);

    // Correlation: gap vs |innovation|
    println!("\n=== Gap vs |Innovation| correlation ===\n");
    println!(
        "{:>6}  {:>12}  {:>12}  {:>14}  {:>14}",
        "idx", "gap_days", "mjd", "|inn_lon|\"", "|inn_lat|\""
    );
    println!("{}", "-".repeat(65));

    // Sort by gap descending to see worst cases
    let mut sorted = night_gaps.clone();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (idx, gap, inn_lon, inn_lat, mjd) in &sorted {
        println!(
            "{:>6}  {:>12.4}  {:>12.4}  {:>14.3}  {:>14.3}",
            idx,
            gap,
            mjd,
            inn_lon.abs(),
            inn_lat.abs()
        );
    }

    // Linear regression: gap -> |inn_lon|
    println!("\n=== Linear regression: gap (days) -> |inn_lon| (arcsec) ===\n");

    let xs: Vec<f64> = night_gaps.iter().map(|(_, g, _, _, _)| *g).collect();
    let ys_lon: Vec<f64> = night_gaps.iter().map(|(_, _, il, _, _)| il.abs()).collect();
    let ys_lat: Vec<f64> = night_gaps.iter().map(|(_, _, _, ib, _)| ib.abs()).collect();

    let (slope_lon, intercept_lon, r2_lon) = linear_regression(&xs, &ys_lon);
    let (slope_lat, intercept_lat, r2_lat) = linear_regression(&xs, &ys_lat);

    println!(
        "  λ: |inn| = {:.3} * gap + {:.3}   (R² = {:.4})",
        slope_lon, intercept_lon, r2_lon
    );
    println!(
        "  β: |inn| = {:.3} * gap + {:.3}   (R² = {:.4})",
        slope_lat, intercept_lat, r2_lat
    );

    println!("\n=== Gap buckets ===\n");
    println!(
        "{:>20}  {:>8}  {:>14}  {:>14}",
        "gap range (days)", "count", "mean |inn_lon|\"", "mean |inn_lat|\""
    );
    println!("{}", "-".repeat(62));

    let buckets = [
        (0.5_f64, 1.0_f64),
        (1.0, 3.0),
        (3.0, 7.0),
        (7.0, 15.0),
        (15.0, f64::INFINITY),
    ];
    for (lo, hi) in &buckets {
        let subset: Vec<_> = night_gaps
            .iter()
            .filter(|(_, g, _, _, _)| *g >= *lo && *g < *hi)
            .collect();
        if subset.is_empty() {
            continue;
        }
        let mean_lon =
            subset.iter().map(|(_, _, il, _, _)| il.abs()).sum::<f64>() / subset.len() as f64;
        let mean_lat =
            subset.iter().map(|(_, _, _, ib, _)| ib.abs()).sum::<f64>() / subset.len() as f64;
        let label = if hi.is_infinite() {
            format!("[{:.1}, ∞)", lo)
        } else {
            format!("[{:.1}, {:.1})", lo, hi)
        };
        println!(
            "{:>20}  {:>8}  {:>14.3}  {:>14.3}",
            label,
            subset.len(),
            mean_lon,
            mean_lat
        );
    }
}

fn linear_regression(xs: &[f64], ys: &[f64]) -> (f64, f64, f64) {
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;

    let ss_xx = xs.iter().map(|x| (x - mean_x).powi(2)).sum::<f64>();
    let ss_xy = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>();
    let ss_yy = ys.iter().map(|y| (y - mean_y).powi(2)).sum::<f64>();

    let slope = ss_xy / ss_xx;
    let intercept = mean_y - slope * mean_x;
    let r2 = if ss_yy > 0.0 {
        (ss_xy * ss_xy) / (ss_xx * ss_yy)
    } else {
        0.0
    };

    (slope, intercept, r2)
}

pub fn load_data(parquet_path: impl AsRef<Utf8Path>) -> (DataFrame, ObsDataset) {
    let path = parquet_path.as_ref().as_str();
    let args = ScanArgsParquet {
        rechunk: true,
        ..Default::default()
    };
    let lf = LazyFrame::scan_parquet(path.into(), args).expect("scan_parquet must succeed");
    let obs_dataset = ObsDataset::from_lazy(
        lf.clone(),
        FromPolarsArgs {
            do_rechunk: Some(false),
            error_model: Some(ObsErrorModel::FCCT14),
            contiguous_choice: Some(ContiguousChoice::ContiguousNight),
        },
    )
    .expect("from_lazy must succeed for int file");

    let df = lf.collect().expect("collect must succeed");
    (df, obs_dataset)
}

/// A predicted position with its Kalman 1-sigma uncertainty (in degrees).
/// A predicted position with its Kalman 1-sigma uncertainty (in degrees).
struct PredictedPoint {
    /// MJD (TT) of the prediction epoch.
    mjd: f64,
    lon_deg: f64,
    lat_deg: f64,
    /// $\sigma_\lambda$ extracted from $P_{00}$ of the predicted covariance.
    sigma_lon_deg: f64,
    /// $\sigma_\beta$ extracted from $P_{33}$ of the predicted covariance.
    sigma_lat_deg: f64,
}

/// Maximum 1-sigma half-width used for bounding-box computation and band
/// rendering.  Beyond this threshold the Kalman filter has not yet converged
/// and the uncertainty is not informative for visualisation purposes.
const SIGMA_CLIP_DEG: f64 = 10.0;

/// Visual amplification factor applied to the Kalman sigma bands for display
/// purposes only.  The true 1-sigma values are of order ~1 arcsecond; without
/// amplification they would be sub-pixel at degree-scale axes.
const SIGMA_DISPLAY_FACTOR: f64 = 500.0;

fn plot_trajectory(
    obs_points: &[(f64, f64)],
    pred_points: &[PredictedPoint],
    output_path: &str,
) -> Result<()> {
    use plotters::prelude::*;

    let root = BitMapBackend::new(output_path, (1600, 1200)).into_drawing_area();
    root.fill(&WHITE)?;

    let (top, bottom) = root.split_vertically(700);

    // =========================================================================
    // TOP PANEL – trajectory
    // =========================================================================

    let obs_lon_min = obs_points
        .iter()
        .map(|(l, _)| *l)
        .fold(f64::INFINITY, f64::min);
    let obs_lon_max = obs_points
        .iter()
        .map(|(l, _)| *l)
        .fold(f64::NEG_INFINITY, f64::max);
    let obs_lat_min = obs_points
        .iter()
        .map(|(_, b)| *b)
        .fold(f64::INFINITY, f64::min);
    let obs_lat_max = obs_points
        .iter()
        .map(|(_, b)| *b)
        .fold(f64::NEG_INFINITY, f64::max);

    let lon_margin = (obs_lon_max - obs_lon_min).max(0.1) * 0.05;
    let lat_margin = (obs_lat_max - obs_lat_min).max(0.1) * 0.15;

    let x_range = (obs_lon_min - lon_margin)..(obs_lon_max + lon_margin);
    let y_range = (obs_lat_min - lat_margin)..(obs_lat_max + lat_margin);

    // Cross size expressed as a fraction of the axis span so it stays
    // proportional regardless of the data extent.
    let cross_half_lon = (obs_lon_max - obs_lon_min).max(0.1) * 0.012;
    let cross_half_lat = (obs_lat_max - obs_lat_min).max(0.1) * 0.035;

    let mut chart_top = ChartBuilder::on(&top)
        .caption(
            "Trajectory: observed vs predicted (ecliptic frame)",
            ("sans-serif", 32),
        )
        .margin(20)
        .x_label_area_size(65)
        .y_label_area_size(85)
        .build_cartesian_2d(x_range, y_range)?;

    chart_top
        .configure_mesh()
        .x_desc("Ecliptic longitude λ (degrees)")
        .y_desc("Ecliptic latitude β (degrees)")
        .x_label_formatter(&|v| format!("{:.1}°", v))
        .y_label_formatter(&|v| format!("{:.2}°", v))
        .x_label_style(("sans-serif", 20))
        .y_label_style(("sans-serif", 20))
        .axis_desc_style(("sans-serif", 24))
        .light_line_style(RGBColor(220, 225, 235))
        .draw()?;

    // Kalman mean – line
    chart_top
        .draw_series(LineSeries::new(
            pred_points.iter().map(|p| (p.lon_deg, p.lat_deg)),
            Into::<ShapeStyle>::into(RGBColor(30, 100, 210)).stroke_width(3),
        ))?
        .label("Kalman mean")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 25, y)],
                Into::<ShapeStyle>::into(RGBColor(30, 100, 210)).stroke_width(3),
            )
        });

    // Kalman mean – cross markers at each predicted point
    let cross_style: ShapeStyle = Into::<ShapeStyle>::into(RGBColor(30, 100, 210)).stroke_width(2);

    chart_top.draw_series(pred_points.iter().flat_map(|p| {
        let (cx, cy) = (p.lon_deg, p.lat_deg);
        // Horizontal arm
        let h = PathElement::new(
            vec![(cx - cross_half_lon, cy), (cx + cross_half_lon, cy)],
            cross_style,
        );
        // Vertical arm
        let v = PathElement::new(
            vec![(cx, cy - cross_half_lat), (cx, cy + cross_half_lat)],
            cross_style,
        );
        [h, v]
    }))?;

    // Observed
    chart_top
        .draw_series(
            obs_points
                .iter()
                .map(|(lon, lat)| Circle::new((*lon, *lat), 6, RED.filled())),
        )?
        .label("Observed")
        .legend(|(x, y)| Circle::new((x + 12, y), 6, RED.filled()));

    chart_top
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .label_font(("sans-serif", 22))
        .draw()?;

    // =========================================================================
    // BOTTOM PANEL – 1-σ and 2-σ evolution vs MJD (in arcseconds)
    // =========================================================================

    const SIGMA_PLOT_CLIP_ARCSEC: f64 = 50.0;

    let sigma_data: Vec<(f64, f64, f64)> = pred_points
        .iter()
        .map(|p| {
            let s_lon = (p.sigma_lon_deg * 3600.0).min(SIGMA_PLOT_CLIP_ARCSEC);
            let s_lat = (p.sigma_lat_deg * 3600.0).min(SIGMA_PLOT_CLIP_ARCSEC);
            (p.mjd, s_lon, s_lat)
        })
        .collect();

    let mjd_min = sigma_data
        .iter()
        .map(|(t, _, _)| *t)
        .fold(f64::INFINITY, f64::min);
    let mjd_max = sigma_data
        .iter()
        .map(|(t, _, _)| *t)
        .fold(f64::NEG_INFINITY, f64::max);
    let mjd_margin = (mjd_max - mjd_min).max(1.0) * 0.03;

    let mut chart_bot = ChartBuilder::on(&bottom)
        .caption(
            "Kalman filter uncertainty evolution (clipped at 5\")",
            ("sans-serif", 30),
        )
        .margin(20)
        .x_label_area_size(65)
        .y_label_area_size(85)
        .build_cartesian_2d(
            (mjd_min - mjd_margin)..(mjd_max + mjd_margin),
            0.0_f64..SIGMA_PLOT_CLIP_ARCSEC * 1.05,
        )?;

    chart_bot
        .configure_mesh()
        .x_desc("MJD")
        .y_desc("1-σ (arcsec)")
        .x_label_formatter(&|v| format!("{:.0}", v))
        .y_label_formatter(&|v| format!("{:.2}\"", v))
        .x_label_style(("sans-serif", 20))
        .y_label_style(("sans-serif", 20))
        .axis_desc_style(("sans-serif", 24))
        .light_line_style(RGBColor(220, 225, 235))
        .draw()?;

    // 2-σ lon band
    chart_bot
        .draw_series(AreaSeries::new(
            sigma_data.iter().map(|(t, s_lon, _)| (*t, *s_lon * 2.0)),
            0.0,
            RGBColor(30, 100, 210).mix(0.15),
        ))?
        .label("2-σ lon")
        .legend(|(x, y)| {
            Rectangle::new(
                [(x, y - 7), (x + 22, y + 7)],
                RGBColor(30, 100, 210).mix(0.15),
            )
        });

    // 1-σ lon band
    chart_bot
        .draw_series(AreaSeries::new(
            sigma_data.iter().map(|(t, s_lon, _)| (*t, *s_lon)),
            0.0,
            RGBColor(30, 100, 210).mix(0.35),
        ))?
        .label("1-σ lon")
        .legend(|(x, y)| {
            Rectangle::new(
                [(x, y - 7), (x + 22, y + 7)],
                RGBColor(30, 100, 210).mix(0.35),
            )
        });

    // 2-σ lat band
    chart_bot
        .draw_series(AreaSeries::new(
            sigma_data.iter().map(|(t, _, s_lat)| (*t, *s_lat * 2.0)),
            0.0,
            RGBColor(200, 80, 30).mix(0.15),
        ))?
        .label("2-σ lat")
        .legend(|(x, y)| {
            Rectangle::new(
                [(x, y - 7), (x + 22, y + 7)],
                RGBColor(200, 80, 30).mix(0.15),
            )
        });

    // 1-σ lat band
    chart_bot
        .draw_series(AreaSeries::new(
            sigma_data.iter().map(|(t, _, s_lat)| (*t, *s_lat)),
            0.0,
            RGBColor(200, 80, 30).mix(0.35),
        ))?
        .label("1-σ lat")
        .legend(|(x, y)| {
            Rectangle::new(
                [(x, y - 7), (x + 22, y + 7)],
                RGBColor(200, 80, 30).mix(0.35),
            )
        });

    // σ lon line
    chart_bot.draw_series(LineSeries::new(
        sigma_data.iter().map(|(t, s_lon, _)| (*t, *s_lon)),
        Into::<ShapeStyle>::into(RGBColor(30, 100, 210)).stroke_width(2),
    ))?;

    // σ lat line
    chart_bot.draw_series(LineSeries::new(
        sigma_data.iter().map(|(t, _, s_lat)| (*t, *s_lat)),
        Into::<ShapeStyle>::into(RGBColor(200, 80, 30)).stroke_width(2),
    ))?;

    chart_bot
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .label_font(("sans-serif", 20))
        .draw()?;

    root.present()?;
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let (df, obs_dataset) = load_data(&cli.alerts);

    let engine_config = load_config(&cli.config)?;

    // Sort all observations by time.
    let mut obs_vec = obs_dataset
        .materialize_trajectory(1 as u32)
        .unwrap()
        .collect_into_vec();
    obs_vec.sort_by(|a, b| {
        a.mjd_tt()
            .partial_cmp(&b.mjd_tt())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("Total observations: {}", obs_vec.len());
    println!();

    // ── Initialise the tracklet from the first pair ───────────────────────────
    let seed_tracklet = TrackletData::from_pair(
        TrackId(0),
        obs_vec[0],
        obs_vec[1],
        engine_config.pairs.acc_prior_var,
        engine_config.pairs.max_angular_speed,
        engine_config.process_noise_q,
        engine_config.singer_params,
    )
    .map(Tracklet::Seed)
    .context("Failed to build seed tracklet from first pair")?;

    println!("Seed initialised from observations 0 and 1.");
    println!("Seed state: {:?}", seed_tracklet);
    println!();

    // The running state is the inner EclipticState; we extract it from the
    // seed and drive the predict/update loop manually from observation 2
    // onward.
    let mut state = seed_tracklet.state().unwrap().clone();

    print_step_header();

    let mut obs_ecl_points: Vec<(f64, f64)> = Vec::new();
    let mut pred_ecl_points: Vec<PredictedPoint> = Vec::new();
    let mut innovations: Vec<(f64, f64, f64)> = Vec::new();

    for (idx, obs) in obs_vec.iter().enumerate().skip(2) {
        let t_obs = obs.mjd_tt();

        // ── Predict ───────────────────────────────────────────────────────────
        let process_noise_q = seed_tracklet.ecliptic_data().unwrap().state.process_noise_q;
        let predicted = state.propagate(t_obs, process_noise_q);

        let pred_lon = predicted.x[0];
        let pred_lat = predicted.x[3];

        // 1-sigma from the *predicted* covariance (before assimilation).
        // These represent the true forecast uncertainty at each step.
        let sigma_lon_pred = predicted.p[(0, 0)].sqrt();
        let sigma_lat_pred = predicted.p[(3, 3)].sqrt();

        // ── Convert observation to ecliptic ───────────────────────────────────
        let equ_cov = EquCoordCov::from_equ(*obs.equ_coord());
        let ecl_obs = EclipticCoordCov::from(equ_cov);

        obs_ecl_points.push((rad_to_deg(ecl_obs.coord.lon), rad_to_deg(ecl_obs.coord.lat)));
        pred_ecl_points.push(PredictedPoint {
            mjd: t_obs,
            lon_deg: rad_to_deg(pred_lon),
            lat_deg: rad_to_deg(pred_lat),
            sigma_lon_deg: rad_to_deg(sigma_lon_pred),
            sigma_lat_deg: rad_to_deg(sigma_lat_pred),
        });

        // Innovation (longitude wrap-aware)
        let mut d_lon = ecl_obs.coord.lon - pred_lon;
        if d_lon > PI {
            d_lon -= 2.0 * PI;
        } else if d_lon < -PI {
            d_lon += 2.0 * PI;
        }
        let d_lat = ecl_obs.coord.lat - pred_lat;

        innovations.push((t_obs, rad_to_arcsec(d_lon), rad_to_arcsec(d_lat)));

        // ── Update ────────────────────────────────────────────────────────────
        let updated = predicted
            .kalman_update(&ecl_obs)
            .context(format!("Kalman update singular at step {idx}"))?;

        let sigma_lon = updated.p[(0, 0)].sqrt();
        let sigma_lat = updated.p[(3, 3)].sqrt();

        println!(
            "{:>4}  {:>12.6}  {:>10.5}  {:>10.5}  {:>10.5}  {:>10.5}  {:>10.3}  {:>10.3}  {:>10.3}  {:>10.3}",
            idx,
            t_obs,
            rad_to_deg(pred_lon),
            rad_to_deg(pred_lat),
            rad_to_deg(ecl_obs.coord.lon),
            rad_to_deg(ecl_obs.coord.lat),
            rad_to_arcsec(d_lon),
            rad_to_arcsec(d_lat),
            rad_to_arcsec(sigma_lon),
            rad_to_arcsec(sigma_lat),
        );

        state = updated;
    }

    println!();
    println!("Final state:");
    println!("  epoch  = {:.6} MJD", state.epoch);
    println!("  λ      = {:.6}°", rad_to_deg(state.x[0]));
    println!("  λ̇      = {:.6} arcsec/day", rad_to_arcsec(state.x[1]));
    println!("  λ̈      = {:.6} arcsec/day²", rad_to_arcsec(state.x[2]));
    println!("  β      = {:.6}°", rad_to_deg(state.x[3]));
    println!("  β̇      = {:.6} arcsec/day", rad_to_arcsec(state.x[4]));
    println!("  β̈      = {:.6} arcsec/day²", rad_to_arcsec(state.x[5]));

    // After the loop
    plot_trajectory(&obs_ecl_points, &pred_ecl_points, "trajectory.png")
        .context("Failed to plot trajectory")?;

    println!("Plot saved to trajectory.png");

    plot_innovations(&innovations, "innovations.png").context("Failed to plot innovations")?;
    println!("Innovation plot saved to innovations.png\n");

    println!("Analyze cadence and innovations: \n");
    analyze_cadence_vs_innovation(&innovations);

    Ok(())
}
