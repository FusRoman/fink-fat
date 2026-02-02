//! Pair seeding diagnostic plots (Plotters backend).
//!
//! Overview
//! --------
//! This module generates **headless** (CI-friendly) PNG plots that help diagnose
//! the *pair seeding* stage in `fink-fat-engine`, using truth association from
//! [`AlertStoreWithTruth`].
//!
//! The plots are meant to answer practical questions such as:
//! - Are false links (contamination) concentrated at large Δt or large angular separation?
//! - Are most true links far from the thresholds (meaning cuts might be loosened),
//!   or are they close to the thresholds (meaning cuts are tight)?
//! - How do purity/precision change as a function of a single threshold (e.g. max_sep)?
//!
//! Inputs
//! ------
//! - `AlertStoreWithTruth`: provides per-alert truth id (`trajectory_id`) and
//!   astrometry/time for feature extraction.
//! - `Pairs`: the generated pairs to analyze.
//!
//! Output
//! ------
//! PNG files written to an output directory. All functions are deterministic.
//!
//! Notes
//! -----
//! - This module focuses on **quality diagnostics**. Runtime/CPU breakdown plots
//!   typically require instrumentation counters inside the seeding code (e.g.
//!   number of candidates scanned vs accepted). We can add that later via a
//!   `PairDiagnostics` struct if desired.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::ValueEnum;
use plotters::coord::types::{RangedCoordf64, RangedCoordi32};
use plotters::prelude::*;

use fink_fat_engine::{
    AlertId,
    seeding::pairs::{Pair, Pairs},
};

use crate::dataset::ztf_alerts::AlertStoreWithTruth;
use crate::seeding::metrics::{PairMetrics, pair_metrics};

/// Angular unit used for plot display.
///
/// Notes
/// -----
/// Internally, all angular quantities are stored and computed in **radians**.
/// This enum only controls **axis labels and value conversion for plotting**.
#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
pub enum AngularUnit {
    /// Radians (default).
    #[value(name = "rad")]
    Radian,
    /// Degrees.
    #[value(name = "deg")]
    Degree,
    /// Arcminutes.
    #[value(name = "arcmin")]
    ArcMinute,
    /// Arcseconds.
    #[value(name = "arcsec")]
    ArcSecond,
}

impl AngularUnit {
    #[inline]
    pub fn scale_from_rad(self) -> f64 {
        match self {
            AngularUnit::Radian => 1.0,
            AngularUnit::Degree => 180.0 / std::f64::consts::PI,
            AngularUnit::ArcMinute => 60.0 * 180.0 / std::f64::consts::PI,
            AngularUnit::ArcSecond => 3600.0 * 180.0 / std::f64::consts::PI,
        }
    }

    #[inline]
    pub fn label(self) -> &'static str {
        match self {
            AngularUnit::Radian => "rad",
            AngularUnit::Degree => "deg",
            AngularUnit::ArcMinute => "arcmin",
            AngularUnit::ArcSecond => "arcsec",
        }
    }
}

/// Label attached to a generated pair for plotting.
///
/// Notes
/// -----
/// - `True`: both endpoints have truth (`tid > 0`) and `tid(a) == tid(b)`.
/// - `Contaminated`: both endpoints have truth (`tid > 0`) but `tid(a) != tid(b)`.
/// - `Unknown`: at least one endpoint has `tid <= 0` (truth not defined).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PairTruthLabel {
    True,
    Contaminated,
    Unknown,
}

/// Per-pair features computed from the store and pair list.
///
/// Fields
/// ------
/// - `dt`: time difference `mjd_tt(b) - mjd_tt(a)` (days).
/// - `sep_rad`: angular separation between endpoints (radians).
/// - `label`: truth label for the pair (True/Contaminated/Unknown).
#[derive(Debug, Copy, Clone)]
pub struct PairFeat {
    pub dt: f64,
    pub sep_rad: f64,
    pub label: PairTruthLabel,
}

/// Plot configuration (bins, sizes, etc.).
#[derive(Debug, Clone)]
pub struct PairPlotConfig {
    /// Output image width (pixels).
    pub width: u32,
    /// Output image height (pixels).
    pub height: u32,

    /// Number of bins for Δt histogram.
    pub dt_bins: usize,
    /// Number of bins for separation histogram.
    pub sep_bins: usize,

    /// Optional dt range. If None, inferred from data.
    pub dt_range: Option<(f64, f64)>,
    /// Optional separation range in radians. If None, inferred from data.
    pub sep_range: Option<(f64, f64)>,

    /// Maximum number of points in scatter plots (downsample for huge outputs).
    pub scatter_max_points: usize,

    /// Angular unit used for plotting (radians internally).
    pub angular_unit: AngularUnit,
}

impl Default for PairPlotConfig {
    fn default() -> Self {
        Self {
            width: 1100,
            height: 650,
            dt_bins: 80,
            sep_bins: 80,
            dt_range: None,
            sep_range: None,
            scatter_max_points: 200_000,
            angular_unit: AngularUnit::Radian,
        }
    }
}

/// Kinematic post-hoc cut: keep pair iff sep <= omega * dt.
/// - sep in radians
/// - omega in rad/day
/// - dt in days
#[inline]
fn kinematic_keep(sep_rad: f64, dt_days: f64, omega_rad_per_day: f64) -> bool {
    // Robust: handle dt<=0 gracefully (should not happen for ordered pairs).
    if !(dt_days > 0.0) {
        return false;
    }
    sep_rad <= omega_rad_per_day * dt_days
}

/// Return truth id (`trajectory_id`) for alert `id` (may be <= 0).
#[inline]
fn tid(store: &AlertStoreWithTruth, id: AlertId) -> i32 {
    store.trajectory_id[id.idx()]
}

/// Robust angular separation between two points on the sphere (radians).
///
/// This uses a numerically stable formulation based on the spherical law
/// of cosines with clamping.
///
/// Parameters
/// ----------
/// ra1, dec1, ra2, dec2 : f64
///     Coordinates in radians.
///
/// Returns
/// -------
/// f64
///     Angular separation in radians, in [0, π].
#[inline]
fn angular_sep_rad(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    let (s1, c1) = dec1.sin_cos();
    let (s2, c2) = dec2.sin_cos();
    let d_ra = ra2 - ra1;

    let cos = (s1 * s2) + (c1 * c2 * d_ra.cos());
    cos.clamp(-1.0, 1.0).acos()
}

/// Extract per-pair features (Δt, separation, truth label).
///
/// Parameters
/// ----------
/// store : &AlertStoreWithTruth
///     Alert store with truth sidecar.
/// pairs : &Pairs
///     Generated pairs to analyze.
///
/// Returns
/// -------
/// Vec<PairFeat>
///     One feature record per pair.
///
/// Errors
/// ------
/// Returns an error if a pair references an invalid `AlertId` index.
///
/// Notes
/// -----
/// - `dt` is `mjd_tt(b) - mjd_tt(a)` in **days**.
/// - We do not reorder endpoints; we respect the pair ordering in the input.
pub fn extract_pair_features(store: &AlertStoreWithTruth, pairs: &Pairs) -> Result<Vec<PairFeat>> {
    let alerts = &store.store.alerts;

    let feats = pairs
        .iter()
        .map(|Pair { a, b }| {
            let ia = a.idx();
            let ib = b.idx();

            let aa = alerts
                .get(ia)
                .with_context(|| format!("pair references invalid a idx={ia}"))?;
            let bb = alerts
                .get(ib)
                .with_context(|| format!("pair references invalid b idx={ib}"))?;

            let dt = bb.mjd_tt - aa.mjd_tt;
            let sep_rad = angular_sep_rad(aa.ra, aa.dec, bb.ra, bb.dec);

            let ta = tid(store, *a);
            let tb = tid(store, *b);
            let label = if ta > 0 && tb > 0 {
                if ta == tb {
                    PairTruthLabel::True
                } else {
                    PairTruthLabel::Contaminated
                }
            } else {
                PairTruthLabel::Unknown
            };

            Ok(PairFeat { dt, sep_rad, label })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(feats)
}

/// Ensure `out_dir` exists and return a resolved output path.
fn ensure_out_path(out_dir: &Path, filename: &str) -> Result<PathBuf> {
    fs::create_dir_all(out_dir)
        .with_context(|| format!("failed to create output dir: {out_dir:?}"))?;
    Ok(out_dir.join(filename))
}

/// Compute min/max range from values, with a small padding.
fn infer_range(values: &[f64]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;

    for &x in values {
        if x.is_finite() {
            lo = lo.min(x);
            hi = hi.max(x);
        }
    }

    if !lo.is_finite() || !hi.is_finite() || lo == hi {
        return (0.0, 1.0);
    }

    let pad = 0.02 * (hi - lo);
    (lo - pad, hi + pad)
}

/// Build histogram counts for values in [xmin, xmax] with `bins` bins.
pub fn histogram(values: &[f64], xmin: f64, xmax: f64, bins: usize) -> Vec<usize> {
    let mut h = vec![0usize; bins];
    if values.is_empty() || !(xmin < xmax) || bins == 0 {
        return h;
    }
    let w = (xmax - xmin) / (bins as f64);
    for &x in values {
        if !x.is_finite() || x < xmin || x > xmax {
            continue;
        }
        let mut k = ((x - xmin) / w) as isize;
        if k < 0 {
            k = 0;
        }
        if k as usize >= bins {
            k = (bins as isize) - 1;
        }
        h[k as usize] += 1;
    }
    h
}

/// Plot a histogram of Δt (days), split by truth label.
///
/// Output
/// ------
/// Writes `pairs_dt_hist.png`.
///
/// Notes
/// -----
/// - Produces three series: True / Contaminated / Unknown.
/// - This plot is extremely useful to diagnose whether false links concentrate
///   at large time differences.
pub fn plot_pairs_dt_hist(
    feats: &[PairFeat],
    out_dir: &Path,
    cfg: &PairPlotConfig,
) -> Result<PathBuf> {
    let out = ensure_out_path(out_dir, "pairs_dt_hist.png")?;

    {
        let dt_true: Vec<f64> = feats
            .iter()
            .filter(|f| f.label == PairTruthLabel::True)
            .map(|f| f.dt)
            .collect();
        let dt_cont: Vec<f64> = feats
            .iter()
            .filter(|f| f.label == PairTruthLabel::Contaminated)
            .map(|f| f.dt)
            .collect();
        let dt_unk: Vec<f64> = feats
            .iter()
            .filter(|f| f.label == PairTruthLabel::Unknown)
            .map(|f| f.dt)
            .collect();

        let all_dt: Vec<f64> = feats.iter().map(|f| f.dt).collect();
        let (lo, hi) = cfg.dt_range.unwrap_or_else(|| infer_range(&all_dt));

        let h_true = histogram(&dt_true, lo, hi, cfg.dt_bins);
        let h_cont = histogram(&dt_cont, lo, hi, cfg.dt_bins);
        let h_unk = histogram(&dt_unk, lo, hi, cfg.dt_bins);

        let y_max = *h_true
            .iter()
            .chain(&h_cont)
            .chain(&h_unk)
            .max()
            .unwrap_or(&1) as i32;

        let root = BitMapBackend::new(&out, (cfg.width, cfg.height)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Pairs: Δt histogram (days) by truth label",
                ("sans-serif", 28),
            )
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(lo..hi, 0..(y_max + 1))?;

        chart
            .configure_mesh()
            .x_desc("Δt = mjd_tt(b) - mjd_tt(a) [days]")
            .y_desc("count")
            .draw()?;

        // Draw as semi-transparent filled bars by overlaying series.
        // We keep it simple: three bar series with different colors.
        draw_hist_series(
            &mut chart,
            &h_unk,
            lo,
            hi,
            &RGBColor(160, 160, 160).mix(0.35),
            "unknown",
        )?;
        draw_hist_series(
            &mut chart,
            &h_cont,
            lo,
            hi,
            &RGBColor(220, 80, 80).mix(0.35),
            "contaminated",
        )?;
        draw_hist_series(
            &mut chart,
            &h_true,
            lo,
            hi,
            &RGBColor(80, 160, 80).mix(0.35),
            "true",
        )?;

        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()?;
    }

    Ok(out)
}

/// Plot a histogram of angular separation, split by truth label.
///
/// Output
/// ------
/// Writes `pairs_sep_hist.png`.
///
/// Notes
/// -----
/// - Separations are displayed in `cfg.angular_unit` (internally stored in radians).
pub fn plot_pairs_sep_hist(
    feats: &[PairFeat],
    out_dir: &Path,
    cfg: &PairPlotConfig,
) -> Result<PathBuf> {
    let out = ensure_out_path(out_dir, "pairs_sep_hist.png")?;

    {
        let scale = cfg.angular_unit.scale_from_rad();

        let sep_true: Vec<f64> = feats
            .iter()
            .filter(|f| f.label == PairTruthLabel::True)
            .map(|f| f.sep_rad * scale)
            .collect();
        let sep_cont: Vec<f64> = feats
            .iter()
            .filter(|f| f.label == PairTruthLabel::Contaminated)
            .map(|f| f.sep_rad * scale)
            .collect();
        let sep_unk: Vec<f64> = feats
            .iter()
            .filter(|f| f.label == PairTruthLabel::Unknown)
            .map(|f| f.sep_rad * scale)
            .collect();

        let all_sep: Vec<f64> = feats.iter().map(|f| f.sep_rad * scale).collect();
        let (lo, hi) = cfg.sep_range.unwrap_or_else(|| infer_range(&all_sep));

        let h_true = histogram(&sep_true, lo, hi, cfg.sep_bins);
        let h_cont = histogram(&sep_cont, lo, hi, cfg.sep_bins);
        let h_unk = histogram(&sep_unk, lo, hi, cfg.sep_bins);

        let y_max = *h_true
            .iter()
            .chain(&h_cont)
            .chain(&h_unk)
            .max()
            .unwrap_or(&1) as i32;

        let root = BitMapBackend::new(&out, (cfg.width, cfg.height)).into_drawing_area();
        root.fill(&WHITE)?;

        let title = format!(
            "Pairs: angular separation histogram ({}) by truth label",
            cfg.angular_unit.label()
        );

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 28))
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(lo..hi, 0..(y_max + 1))?;

        chart
            .configure_mesh()
            .x_desc(format!("angular separation [{}]", cfg.angular_unit.label()))
            .y_desc("count")
            .draw()?;

        draw_hist_series(
            &mut chart,
            &h_unk,
            lo,
            hi,
            &RGBColor(160, 160, 160).mix(0.35),
            "unknown",
        )?;
        draw_hist_series(
            &mut chart,
            &h_cont,
            lo,
            hi,
            &RGBColor(220, 80, 80).mix(0.35),
            "contaminated",
        )?;
        draw_hist_series(
            &mut chart,
            &h_true,
            lo,
            hi,
            &RGBColor(80, 160, 80).mix(0.35),
            "true",
        )?;

        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()?;
    }

    Ok(out)
}

/// Plot a scatter of (Δt, separation), colored by truth label.
///
/// Output
/// ------
/// Writes `pairs_scatter_dt_sep.png`.
///
/// Notes
/// -----
/// - Separations are displayed in `cfg.angular_unit` (internally stored in radians).
pub fn plot_pairs_scatter_dt_sep(
    feats: &[PairFeat],
    out_dir: &Path,
    cfg: &PairPlotConfig,
) -> Result<PathBuf> {
    let out = ensure_out_path(out_dir, "pairs_scatter_dt_sep.png")?;

    {
        let all_dt: Vec<f64> = feats.iter().map(|f| f.dt).collect();
        let (x_lo, x_hi) = cfg.dt_range.unwrap_or_else(|| infer_range(&all_dt));

        let scale = cfg.angular_unit.scale_from_rad();
        let all_sep_disp: Vec<f64> = feats.iter().map(|f| f.sep_rad * scale).collect();
        let (y_lo, y_hi) = cfg.sep_range.unwrap_or_else(|| infer_range(&all_sep_disp));

        // Downsample if needed (deterministic): take a stride.
        let stride = (feats.len() / cfg.scatter_max_points).max(1);

        let root = BitMapBackend::new(&out, (cfg.width, cfg.height)).into_drawing_area();
        root.fill(&WHITE)?;

        let title = format!(
            "Pairs: scatter Δt vs separation ({}; colored by truth label)",
            cfg.angular_unit.label()
        );

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 28))
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(x_lo..x_hi, y_lo..y_hi)?;

        chart
            .configure_mesh()
            .x_desc("Δt [days]")
            .y_desc(format!("angular separation [{}]", cfg.angular_unit.label()))
            .draw()?;

        // Draw unknown first (grey), then contaminated (red), then true (green)
        // so true points appear on top.
        draw_scatter_by_label_with_unit(
            &mut chart,
            feats,
            stride,
            PairTruthLabel::Unknown,
            scale,
            &RGBColor(150, 150, 150).mix(0.45),
            "unknown",
        )?;
        draw_scatter_by_label_with_unit(
            &mut chart,
            feats,
            stride,
            PairTruthLabel::Contaminated,
            scale,
            &RGBColor(220, 80, 80).mix(0.60),
            "contaminated",
        )?;
        draw_scatter_by_label_with_unit(
            &mut chart,
            feats,
            stride,
            PairTruthLabel::True,
            scale,
            &RGBColor(80, 160, 80).mix(0.60),
            "true",
        )?;

        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()?;
    }

    Ok(out)
}

/// Plot a simple 1D sweep: precision_on_truth & consecutive_recall vs a separation threshold.
///
/// This is useful to tune a single parameter (e.g. `PairConfig.max_sep`).
///
/// Parameters
/// ----------
/// store : &AlertStoreWithTruth
///     Store with truth association.
/// pairs : &Pairs
///     Full generated pairs. We will filter pairs by `sep_rad <= threshold`.
/// feats : &[PairFeat]
///     Per-pair features aligned with `pairs`.
/// thresholds_rad : &[f64]
///     Threshold values in **radians** (must be increasing for a clean plot).
/// out_dir : &Path
///     Output directory.
/// cfg : &PairPlotConfig
///     Plot configuration, including the angular display unit.
///
/// Output
/// ------
/// Writes `pairs_tradeoff_sep_threshold.png`.
///
/// Notes
/// -----
/// - This is a *post-hoc* sweep: it does not re-run the seeding algorithm.
/// - Filtering is performed in **radians**. Only the **x-axis display** is converted
///   to `cfg.angular_unit`.
pub fn plot_pairs_tradeoff_vs_omega_threshold(
    store: &AlertStoreWithTruth,
    pairs: &Pairs,
    feats: &[PairFeat],
    thresholds_omega: &[f64], // rad/day
    out_dir: &Path,
    cfg: &PairPlotConfig,
) -> Result<PathBuf> {
    let out = ensure_out_path(out_dir, "pairs_tradeoff_omega_threshold.png")?;

    {
        anyhow::ensure!(
            feats.len() == pairs.len(),
            "feats and pairs must be aligned"
        );

        // Pre-zip for cheap filtering.
        let pair_feat: Vec<(Pair, PairFeat)> =
            pairs.iter().copied().zip(feats.iter().copied()).collect();

        let mut xs = Vec::with_capacity(thresholds_omega.len());
        let mut prec = Vec::with_capacity(thresholds_omega.len());
        let mut rec = Vec::with_capacity(thresholds_omega.len());

        for &omega in thresholds_omega {
            let filtered: Pairs = pair_feat
                .iter()
                .filter(|(_, f)| kinematic_keep(f.sep_rad, f.dt, omega))
                .map(|(p, _)| *p)
                .collect();

            let m: PairMetrics = pair_metrics(store, &filtered);
            xs.push(omega);
            prec.push(m.precision_on_truth);
            rec.push(m.consecutive_recall);
        }

        // Optional: choose an omega display unit? For now keep rad/day on x-axis.
        let x_lo = *xs.first().unwrap_or(&0.0);
        let x_hi = *xs.last().unwrap_or(&1.0);

        let root = BitMapBackend::new(&out, (cfg.width, cfg.height)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Pairs: quality vs angular speed threshold (rad/day)",
                ("sans-serif", 28),
            )
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(x_lo..x_hi, 0.0f64..1.0f64)?;

        chart
            .configure_mesh()
            .x_desc("angular speed threshold ω [rad/day]")
            .y_desc("ratio")
            .draw()?;

        chart
            .draw_series(LineSeries::new(
                xs.iter().copied().zip(prec.iter().copied()),
                &RGBColor(80, 160, 80),
            ))?
            .label("precision_on_truth")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RGBColor(80, 160, 80)));

        chart
            .draw_series(LineSeries::new(
                xs.iter().copied().zip(rec.iter().copied()),
                &RGBColor(80, 80, 220),
            ))?
            .label("consecutive_recall (proxy)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RGBColor(80, 80, 220)));

        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()?;
    }
    Ok(out)
}

/// Plot a 1D sweep: global purity (overall) and global completeness (proxy) vs a separation threshold.
///
/// Overview
/// --------
/// This plot complements [`plot_pairs_tradeoff_vs_sep_threshold`] by using
/// **global** metrics:
/// - **Global purity**: [`PairMetrics::purity_overall`] = `n_true / n_total`
///   over *all* generated pairs (including unknown truth endpoints).
/// - **Global completeness (proxy)**: [`PairMetrics::consecutive_recall`], i.e.
///   the fraction of *consecutive truth pairs* that are present in the generated
///   output after applying `sep <= threshold`.
///
/// The completeness definition is a scalable proxy: exact recall over all
/// same-trajectory pairs is O(k²) per trajectory and is not tractable at
/// survey scale.
///
/// Arguments
/// ---------
/// * `store` – Store with truth association.
/// * `pairs` – Full generated pairs superset (generated with a large max_sep).
/// * `feats` – Per-pair features aligned with `pairs` (must have same length).
/// * `thresholds_rad` – Threshold values in **radians** (monotonic increasing recommended).
/// * `out_dir` – Output directory.
/// * `cfg` – Plot configuration, including the angular display unit.
///
/// Return
/// ------
/// * `Ok(PathBuf)` with the written PNG path on success.
/// * `Err(anyhow::Error)` if inputs are inconsistent or plotting fails.
///
/// Output
/// ------
/// Writes `pairs_global_tradeoff_sep_threshold.png`.
///
/// Notes
/// -----
/// * This is a post-hoc sweep: it filters a fixed superset of pairs.
/// * Filtering is performed in radians; only the x-axis is converted to
///   `cfg.angular_unit`.
/// * `purity_overall` will typically **decrease** when the threshold increases,
///   as more unknown/incorrect pairs are retained.
/// * `consecutive_recall` will typically **increase** with the threshold.
pub fn plot_pairs_global_tradeoff_vs_omega_threshold(
    store: &AlertStoreWithTruth,
    pairs: &Pairs,
    feats: &[PairFeat],
    thresholds_omega: &[f64],
    out_dir: &Path,
    cfg: &PairPlotConfig,
) -> Result<PathBuf> {
    let out = ensure_out_path(out_dir, "pairs_global_tradeoff_omega_threshold.png")?;

    {
        anyhow::ensure!(
            feats.len() == pairs.len(),
            "feats and pairs must be aligned"
        );

        let pair_feat: Vec<(Pair, PairFeat)> =
            pairs.iter().copied().zip(feats.iter().copied()).collect();

        let mut xs = Vec::with_capacity(thresholds_omega.len());
        let mut purity = Vec::with_capacity(thresholds_omega.len());
        let mut completeness = Vec::with_capacity(thresholds_omega.len());

        for &omega in thresholds_omega {
            let filtered: Pairs = pair_feat
                .iter()
                .filter(|(_, f)| kinematic_keep(f.sep_rad, f.dt, omega))
                .map(|(p, _)| *p)
                .collect();

            let m: PairMetrics = pair_metrics(store, &filtered);
            xs.push(omega);
            purity.push(m.purity_overall);
            completeness.push(m.consecutive_recall);
        }

        let x_lo = *xs.first().unwrap_or(&0.0);
        let x_hi = *xs.last().unwrap_or(&1.0);

        let root = BitMapBackend::new(&out, (cfg.width, cfg.height)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Pairs: global purity & completeness vs ω threshold (rad/day)",
                ("sans-serif", 28),
            )
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(x_lo..x_hi, 0.0f64..1.0f64)?;

        chart
            .configure_mesh()
            .x_desc("angular speed threshold ω [rad/day]")
            .y_desc("ratio")
            .draw()?;

        chart
            .draw_series(LineSeries::new(
                xs.iter().copied().zip(purity.iter().copied()),
                &RGBColor(80, 160, 80),
            ))?
            .label("purity_overall (global)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RGBColor(80, 160, 80)));

        chart
            .draw_series(LineSeries::new(
                xs.iter().copied().zip(completeness.iter().copied()),
                &RGBColor(80, 80, 220),
            ))?
            .label("consecutive_recall (global completeness proxy)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RGBColor(80, 80, 220)));

        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()?;
    }
    Ok(out)
}

/// Plot a post-hoc sweep cost curve: number of kept pairs vs separation threshold.
///
/// Overview
/// --------
/// This plot complements quality tradeoff curves by showing the **computational cost**
/// implied by a separation threshold. Even if purity/recall look good, a threshold
/// that keeps too many pairs may be infeasible at survey scale.
///
/// For each threshold `thr`, we filter the pre-generated superset of pairs with
/// `sep_rad <= thr` and compute:
/// - `n_pairs_kept(thr)`
/// - `pairs_per_alert(thr) = n_pairs_kept / n_alerts`
///
/// Arguments
/// ---------
/// * `store` – Alert store with truth association (used for `n_alerts`).
/// * `pairs` – Full generated pair superset.
/// * `feats` – Per-pair features aligned with `pairs` (must match length).
/// * `thresholds_rad` – Increasing separation thresholds (radians).
/// * `out_dir` – Output directory.
/// * `cfg` – Plot configuration (size + angular display unit).
///
/// Return
/// ------
/// * `Ok(PathBuf)` – Path to the written PNG file.
/// * `Err(anyhow::Error)` – If inputs are inconsistent or plotting fails.
///
/// Notes
/// -----
/// * This is **post-hoc**: it does not re-run seeding.
/// * The y-axis uses `log10(n_pairs_kept)` to remain readable across orders of magnitude.
///   If `n_pairs_kept == 0`, we plot 0.0 by convention.
pub fn plot_pairs_cost_vs_omega_threshold(
    store: &AlertStoreWithTruth,
    pairs: &Pairs,
    feats: &[PairFeat],
    thresholds_omega: &[f64],
    out_dir: &Path,
    cfg: &PairPlotConfig,
) -> Result<PathBuf> {
    let out = ensure_out_path(out_dir, "pairs_cost_vs_omega_threshold.png")?;
    {
        anyhow::ensure!(
            feats.len() == pairs.len(),
            "feats and pairs must be aligned"
        );

        let pair_feat: Vec<(Pair, PairFeat)> =
            pairs.iter().copied().zip(feats.iter().copied()).collect();

        let n_alerts = store.store.alerts.len().max(1) as f64;

        let mut xs = Vec::with_capacity(thresholds_omega.len());
        let mut log10_n_pairs = Vec::with_capacity(thresholds_omega.len());

        for &omega in thresholds_omega {
            let n_kept = pair_feat
                .iter()
                .filter(|(_, f)| kinematic_keep(f.sep_rad, f.dt, omega))
                .count();

            let n_kept_f = n_kept as f64;

            xs.push(omega);
            log10_n_pairs.push(if n_kept == 0 { 0.0 } else { n_kept_f.log10() });

            let _pairs_per_alert = n_kept_f / n_alerts;
            // (Optionnel) tu peux aussi tracer pairs_per_alert sur un 2e axe plus tard.
        }

        let x_lo = *xs.first().unwrap_or(&0.0);
        let x_hi = *xs.last().unwrap_or(&1.0);

        let y0_min = log10_n_pairs.iter().copied().fold(f64::INFINITY, f64::min);
        let y0_max = log10_n_pairs
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        let mut y_lo = if y0_min.is_finite() { y0_min } else { 0.0 };
        let mut y_hi = if y0_max.is_finite() { y0_max } else { 1.0 };
        if y_lo == y_hi {
            y_hi = y_lo + 1.0;
        }
        let pad = 0.05 * (y_hi - y_lo);
        y_lo -= pad;
        y_hi += pad;

        let root = BitMapBackend::new(&out, (cfg.width, cfg.height)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Pairs: cost vs ω threshold (rad/day)", ("sans-serif", 28))
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(70)
            .build_cartesian_2d(x_lo..x_hi, y_lo..y_hi)?;

        chart
            .configure_mesh()
            .x_desc("angular speed threshold ω [rad/day]")
            .y_desc("log10(n_pairs_kept)")
            .draw()?;

        chart
            .draw_series(LineSeries::new(
                xs.iter().copied().zip(log10_n_pairs.iter().copied()),
                &BLACK,
            ))?
            .label("log10(n_pairs_kept)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));

        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()?;
    }
    Ok(out)
}

/// Plot a histogram of angular speed ω = sep / Δt, split by truth label.
///
/// Output
/// ------
/// Writes `pairs_omega_hist.png`.
///
/// Notes
/// -----
/// - ω is computed as `sep_rad / dt_days` (rad/day).
/// - Pairs with `dt <= 0` or non-finite values are ignored.
/// - Only the **display unit** changes with `cfg.angular_unit`:
///   the x-axis is in `{unit}/day` (e.g. `arcsec/day`), while internal
///   computation remains rad/day.
pub fn plot_pairs_omega_hist(
    feats: &[PairFeat],
    out_dir: &Path,
    cfg: &PairPlotConfig,
) -> Result<PathBuf> {
    let out = ensure_out_path(out_dir, "pairs_omega_hist.png")?;

    {
        let scale = cfg.angular_unit.scale_from_rad(); // angle unit per rad

        // Helper: compute ω in display units (unit/day), skip dt<=0.
        let omega_disp = |f: &PairFeat| -> Option<f64> {
            if !(f.dt > 0.0) {
                return None;
            }
            let w = (f.sep_rad * scale) / f.dt;
            if w.is_finite() { Some(w) } else { None }
        };

        let w_true: Vec<f64> = feats
            .iter()
            .filter(|f| f.label == PairTruthLabel::True)
            .filter_map(omega_disp)
            .collect();

        let w_cont: Vec<f64> = feats
            .iter()
            .filter(|f| f.label == PairTruthLabel::Contaminated)
            .filter_map(omega_disp)
            .collect();

        let w_unk: Vec<f64> = feats
            .iter()
            .filter(|f| f.label == PairTruthLabel::Unknown)
            .filter_map(omega_disp)
            .collect();

        let all_w: Vec<f64> = feats.iter().filter_map(omega_disp).collect();
        let (lo, hi) = infer_range(&all_w);

        // Reuse sep_bins for ω (or add omega_bins to cfg if you prefer).
        let h_true = histogram(&w_true, lo, hi, cfg.sep_bins);
        let h_cont = histogram(&w_cont, lo, hi, cfg.sep_bins);
        let h_unk = histogram(&w_unk, lo, hi, cfg.sep_bins);

        let y_max = *h_true
            .iter()
            .chain(&h_cont)
            .chain(&h_unk)
            .max()
            .unwrap_or(&1) as i32;

        let root = BitMapBackend::new(&out, (cfg.width, cfg.height)).into_drawing_area();
        root.fill(&WHITE)?;

        let unit = cfg.angular_unit.label();
        let title = format!("Pairs: angular speed histogram (ω in {unit}/day) by truth label");

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 28))
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(lo..hi, 0..(y_max + 1))?;

        chart
            .configure_mesh()
            .x_desc(format!("ω = sep / Δt [{unit}/day]"))
            .y_desc("count")
            .draw()?;

        draw_hist_series(
            &mut chart,
            &h_unk,
            lo,
            hi,
            &RGBColor(160, 160, 160).mix(0.35),
            "unknown",
        )?;
        draw_hist_series(
            &mut chart,
            &h_cont,
            lo,
            hi,
            &RGBColor(220, 80, 80).mix(0.35),
            "contaminated",
        )?;
        draw_hist_series(
            &mut chart,
            &h_true,
            lo,
            hi,
            &RGBColor(80, 160, 80).mix(0.35),
            "true",
        )?;

        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()?;
    }

    Ok(out)
}

// -------------------------------------------------------------------------------------------------
// Multi-night summary plots
// -------------------------------------------------------------------------------------------------
//
// These plots operate on the per-night summary table (DataFrame) produced by
// `seeding-multinight-eval`. They are designed for capacity planning and
// regression detection.
//
// Expected columns (as written by the multi-night binary):
// - nid (Int32)
// - n_alerts (Int64)
// - n_pairs (Int64)
// - n_triplets (Int64)
// - dt_bucket_ms, dt_pairs_ms, dt_triplets_ms (Float64)
// - pair_purity_overall, pair_precision_on_truth, pair_consecutive_recall (Float64)
// - triplet_* counterparts (Float64 or NaN)
//
// Output PNG files:
// - multinight_alerts_per_night.png
// - multinight_runtime_vs_alerts.png
// - multinight_pairs_per_alert.png
// - multinight_quality_tradeoff_pairs.png
// - multinight_triplets_per_alert.png (optional, if triplets columns exist)

use polars::prelude::{DataFrame, DataType};

/// Plot configuration for multi-night summary plots.
#[derive(Debug, Clone)]
pub struct MultiNightPlotConfig {
    /// Output image width (pixels).
    pub width: u32,
    /// Output image height (pixels).
    pub height: u32,
    /// If true, use log10 scaling on the y-axis for cost-like quantities.
    pub log_cost: bool,
}

impl Default for MultiNightPlotConfig {
    fn default() -> Self {
        Self {
            width: 1400,
            height: 900,
            log_cost: true,
        }
    }
}

/// Generate a standard set of multi-night summary plots.
///
/// Arguments
/// ---------
/// * `df` – Per-night summary table (one row per night).
/// * `out_dir` – Output directory for PNG files.
/// * `cfg` – Plot configuration.
///
/// Return
/// ------
/// * `Ok(Vec<PathBuf>)` – paths of written PNG files.
/// * `Err(anyhow::Error)` on I/O, schema mismatch, or plotting failure.
pub fn plot_multinight_summary(
    df: &DataFrame,
    out_dir: &Path,
    cfg: &MultiNightPlotConfig,
) -> Result<Vec<PathBuf>> {
    let mut outs = Vec::new();
    outs.push(plot_multinight_alerts_per_night(df, out_dir, cfg)?);
    outs.push(plot_multinight_runtime_vs_alerts(df, out_dir, cfg)?);
    outs.push(plot_multinight_pairs_per_alert(df, out_dir, cfg)?);
    outs.push(plot_multinight_quality_tradeoff_pairs(df, out_dir, cfg)?);

    // Triplets plot only if column exists (and not all zeros).
    if df.column("n_triplets").is_ok() {
        outs.push(plot_multinight_triplets_per_alert(df, out_dir, cfg)?);
    }
    Ok(outs)
}

fn col_i32(df: &DataFrame, name: &str) -> Result<Vec<i32>> {
    let s = df
        .column(name)
        .with_context(|| format!("missing column '{name}'"))?;
    let s = s
        .cast(&DataType::Int32)
        .with_context(|| format!("failed to cast '{name}' to Int32"))?;
    Ok(s.i32().unwrap().into_no_null_iter().collect())
}

fn col_i64(df: &DataFrame, name: &str) -> Result<Vec<i64>> {
    let s = df
        .column(name)
        .with_context(|| format!("missing column '{name}'"))?;
    let s = s
        .cast(&DataType::Int64)
        .with_context(|| format!("failed to cast '{name}' to Int64"))?;
    Ok(s.i64().unwrap().into_no_null_iter().collect())
}

fn col_f64(df: &DataFrame, name: &str) -> Result<Vec<f64>> {
    let s = df
        .column(name)
        .with_context(|| format!("missing column '{name}'"))?;
    let s = s
        .cast(&DataType::Float64)
        .with_context(|| format!("failed to cast '{name}' to Float64"))?;
    // allow nulls -> treat as NaN
    let ca = s.f64().unwrap();
    Ok(ca.into_iter().map(|o| o.unwrap_or(f64::NAN)).collect())
}

fn log10_or_nan(x: f64) -> f64 {
    if x.is_finite() && x > 0.0 {
        x.log10()
    } else {
        f64::NAN
    }
}

/// 1) Alerts per night (bar plot).
fn plot_multinight_alerts_per_night(
    df: &DataFrame,
    out_dir: &Path,
    cfg: &MultiNightPlotConfig,
) -> Result<PathBuf> {
    let out = ensure_out_path(out_dir, "multinight_alerts_per_night.png")?;
    {
        let nid = col_i32(df, "nid").or_else(|_| col_i32(df, crate::dataset::schema::cols::NID))?;
        let n_alerts = col_i64(df, "n_alerts")?;

        let n = nid.len().min(n_alerts.len());
        anyhow::ensure!(n > 0, "empty summary table");

        let max_y = n_alerts.iter().take(n).copied().max().unwrap_or(1) as i32;

        let root = BitMapBackend::new(&out, (cfg.width, cfg.height)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Multi-night: alerts per night", ("sans-serif", 28))
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(80)
            .build_cartesian_2d(0..(n as i32), 0..(max_y + 1))?;

        chart
            .configure_mesh()
            .x_desc("night index (sorted by nid)")
            .y_desc("n_alerts")
            .draw()?;

        chart.draw_series((0..n).map(|i| {
            Rectangle::new(
                [(i as i32, 0), (i as i32 + 1, n_alerts[i] as i32)],
                RGBColor(80, 80, 220).mix(0.45).filled(),
            )
        }))?;
    }
    Ok(out)
}

/// 2) Runtime vs alerts (scatter): dt_pairs_ms vs n_alerts (+ optional log10).
fn plot_multinight_runtime_vs_alerts(
    df: &DataFrame,
    out_dir: &Path,
    cfg: &MultiNightPlotConfig,
) -> Result<PathBuf> {
    let out = ensure_out_path(out_dir, "multinight_runtime_vs_alerts.png")?;
    {
        let n_alerts = col_i64(df, "n_alerts")?;
        let dt_bucket = col_f64(df, "dt_bucket_ms")?;
        let dt_pairs = col_f64(df, "dt_pairs_ms")?;
        let dt_trip =
            col_f64(df, "dt_triplets_ms").unwrap_or_else(|_| vec![f64::NAN; n_alerts.len()]);

        let n = n_alerts.len();

        let xs: Vec<f64> = n_alerts.iter().take(n).map(|&x| x as f64).collect();

        // log option only on Y (cost), keep X linear so "alerts" is interpretable.
        let y_bucket: Vec<f64> = dt_bucket
            .iter()
            .take(n)
            .map(|&y| if cfg.log_cost { log10_or_nan(y) } else { y })
            .collect();
        let y_pairs: Vec<f64> = dt_pairs
            .iter()
            .take(n)
            .map(|&y| if cfg.log_cost { log10_or_nan(y) } else { y })
            .collect();
        let y_trip: Vec<f64> = dt_trip
            .iter()
            .take(n)
            .map(|&y| if cfg.log_cost { log10_or_nan(y) } else { y })
            .collect();

        let x_lo = xs.iter().copied().fold(f64::INFINITY, f64::min);
        let x_hi = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let mut y_lo = f64::INFINITY;
        let mut y_hi = f64::NEG_INFINITY;
        for &y in y_bucket.iter().chain(&y_pairs).chain(&y_trip) {
            if y.is_finite() {
                y_lo = y_lo.min(y);
                y_hi = y_hi.max(y);
            }
        }
        if !y_lo.is_finite() || !y_hi.is_finite() || y_lo == y_hi {
            y_lo = 0.0;
            y_hi = 1.0;
        }
        let pad = 0.05 * (y_hi - y_lo);
        y_lo -= pad;
        y_hi += pad;

        let root = BitMapBackend::new(&out, (cfg.width, cfg.height)).into_drawing_area();
        root.fill(&WHITE)?;

        let y_label = if cfg.log_cost {
            "log10(runtime_ms)"
        } else {
            "runtime_ms"
        };

        let mut chart = ChartBuilder::on(&root)
            .caption("Multi-night: runtime vs alerts", ("sans-serif", 28))
            .margin(15)
            .x_label_area_size(50)
            .y_label_area_size(90)
            .build_cartesian_2d(x_lo..x_hi, y_lo..y_hi)?;

        chart
            .configure_mesh()
            .x_desc("n_alerts")
            .y_desc(y_label)
            .draw()?;

        // bucket-index
        chart
            .draw_series(
                xs.iter()
                    .copied()
                    .zip(y_bucket.iter().copied())
                    .filter(|(_, y)| y.is_finite())
                    .map(|(x, y)| Circle::new((x, y), 3, RGBColor(80, 80, 220).mix(0.55).filled())),
            )?
            .label("bucket_index")
            .legend(|(x, y)| Circle::new((x + 8, y), 4, RGBColor(80, 80, 220).mix(0.55).filled()));

        // pairs
        chart
            .draw_series(
                xs.iter()
                    .copied()
                    .zip(y_pairs.iter().copied())
                    .filter(|(_, y)| y.is_finite())
                    .map(|(x, y)| Circle::new((x, y), 3, RGBColor(80, 160, 80).mix(0.55).filled())),
            )?
            .label("pairs")
            .legend(|(x, y)| Circle::new((x + 8, y), 4, RGBColor(80, 160, 80).mix(0.55).filled()));

        // triplets (optional)
        chart
            .draw_series(
                xs.iter()
                    .copied()
                    .zip(y_trip.iter().copied())
                    .filter(|(_, y)| y.is_finite())
                    .map(|(x, y)| Circle::new((x, y), 3, RGBColor(220, 80, 80).mix(0.55).filled())),
            )?
            .label("triplets")
            .legend(|(x, y)| Circle::new((x + 8, y), 4, RGBColor(220, 80, 80).mix(0.55).filled()));

        chart
            .configure_series_labels()
            .border_style(&BLACK)
            .draw()?;
    }
    Ok(out)
}

/// 3) Pairs per alert (bar or line): n_pairs / n_alerts.
fn plot_multinight_pairs_per_alert(
    df: &DataFrame,
    out_dir: &Path,
    cfg: &MultiNightPlotConfig,
) -> Result<PathBuf> {
    let out = ensure_out_path(out_dir, "multinight_pairs_per_alert.png")?;
    {
        let nid = col_i32(df, "nid").or_else(|_| col_i32(df, crate::dataset::schema::cols::NID))?;
        let n_alerts = col_i64(df, "n_alerts")?;
        let n_pairs = col_i64(df, "n_pairs")?;

        let n = nid.len().min(n_alerts.len()).min(n_pairs.len());
        anyhow::ensure!(n > 0, "empty summary table");

        let vals: Vec<f64> = (0..n)
            .map(|i| (n_pairs[i] as f64) / (n_alerts[i].max(1) as f64))
            .collect();

        let y_lo = 0.0;
        let mut y_hi = vals.iter().copied().fold(0.0, f64::max);
        if y_hi <= 0.0 {
            y_hi = 1.0;
        }
        let pad = 0.05 * y_hi;
        let y_hi = y_hi + pad;

        let root = BitMapBackend::new(&out, (cfg.width, cfg.height)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Multi-night: pairs per alert", ("sans-serif", 28))
            .margin(15)
            .x_label_area_size(50)
            .y_label_area_size(90)
            .build_cartesian_2d(0..(n as i32), y_lo..y_hi)?;

        chart
            .configure_mesh()
            .x_desc("night index (sorted by nid)")
            .y_desc("n_pairs / n_alerts")
            .draw()?;

        chart.draw_series((0..n).map(|i| {
            Rectangle::new(
                [(i as i32, 0.0), (i as i32 + 1, vals[i])],
                RGBColor(80, 160, 80).mix(0.45).filled(),
            )
        }))?;
    }
    Ok(out)
}

/// 4) Quality tradeoff for pairs: purity_overall vs consecutive_recall,
/// annotated by cost proxy (pairs per alert).
fn plot_multinight_quality_tradeoff_pairs(
    df: &DataFrame,
    out_dir: &Path,
    cfg: &MultiNightPlotConfig,
) -> Result<PathBuf> {
    let out = ensure_out_path(out_dir, "multinight_quality_tradeoff_pairs.png")?;
    {
        let n_alerts = col_i64(df, "n_alerts")?;
        let n_pairs = col_i64(df, "n_pairs")?;
        let purity = col_f64(df, "pair_purity_overall")?;
        let recall = col_f64(df, "pair_consecutive_recall")?;

        let n = n_alerts
            .len()
            .min(n_pairs.len())
            .min(purity.len())
            .min(recall.len());
        anyhow::ensure!(n > 0, "empty summary table");

        // bubble radius based on pairs/alert (clamped)
        let cost: Vec<f64> = (0..n)
            .map(|i| (n_pairs[i] as f64) / (n_alerts[i].max(1) as f64))
            .collect();

        let mut cost_lo = f64::INFINITY;
        let mut cost_hi = f64::NEG_INFINITY;
        for &c in &cost {
            if c.is_finite() {
                cost_lo = cost_lo.min(c);
                cost_hi = cost_hi.max(c);
            }
        }
        if !cost_lo.is_finite() || !cost_hi.is_finite() || cost_lo == cost_hi {
            cost_lo = 0.0;
            cost_hi = 1.0;
        }

        let root = BitMapBackend::new(&out, (cfg.width, cfg.height)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Multi-night: pair quality tradeoff (bubble size ~ pairs/alert)",
                ("sans-serif", 28),
            )
            .margin(15)
            .x_label_area_size(60)
            .y_label_area_size(70)
            .build_cartesian_2d(0.0f64..1.0f64, 0.0f64..1.0f64)?;

        chart
            .configure_mesh()
            .x_desc("consecutive_recall (completeness proxy)")
            .y_desc("purity_overall")
            .draw()?;

        chart.draw_series((0..n).filter_map(|i| {
            let x = recall[i];
            let y = purity[i];
            if !(x.is_finite() && y.is_finite()) {
                return None;
            }
            let c = cost[i];
            let t = ((c - cost_lo) / (cost_hi - cost_lo)).clamp(0.0, 1.0);
            let r = (3.0 + 10.0 * t).round() as i32;
            Some(Circle::new(
                (x, y),
                r,
                RGBColor(80, 80, 220).mix(0.35).filled(),
            ))
        }))?;
    }
    Ok(out)
}

/// 5) Triplets per alert (optional).
fn plot_multinight_triplets_per_alert(
    df: &DataFrame,
    out_dir: &Path,
    cfg: &MultiNightPlotConfig,
) -> Result<PathBuf> {
    let out = ensure_out_path(out_dir, "multinight_triplets_per_alert.png")?;
    {
        let n_alerts = col_i64(df, "n_alerts")?;
        let n_trip = col_i64(df, "n_triplets")?;

        let n = n_alerts.len().min(n_trip.len());
        anyhow::ensure!(n > 0, "empty summary table");

        let vals: Vec<f64> = (0..n)
            .map(|i| (n_trip[i] as f64) / (n_alerts[i].max(1) as f64))
            .collect();

        let y_lo = 0.0;
        let mut y_hi = vals.iter().copied().fold(0.0, f64::max);
        if y_hi <= 0.0 {
            y_hi = 1.0;
        }
        let pad = 0.05 * y_hi;
        let y_hi = y_hi + pad;

        let root = BitMapBackend::new(&out, (cfg.width, cfg.height)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Multi-night: triplets per alert", ("sans-serif", 28))
            .margin(15)
            .x_label_area_size(50)
            .y_label_area_size(90)
            .build_cartesian_2d(0..(n as i32), y_lo..y_hi)?;

        chart
            .configure_mesh()
            .x_desc("night index (sorted by nid)")
            .y_desc("n_triplets / n_alerts")
            .draw()?;

        chart.draw_series((0..n).map(|i| {
            Rectangle::new(
                [(i as i32, 0.0), (i as i32 + 1, vals[i])],
                RGBColor(220, 80, 80).mix(0.45).filled(),
            )
        }))?;
    }
    Ok(out)
}

/* ------------------------------ Drawing helpers ------------------------------ */

fn draw_scatter_by_label_with_unit<DB: DrawingBackend, S: Into<ShapeStyle> + Clone>(
    chart: &mut ChartContext<'_, DB, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    feats: &[PairFeat],
    stride: usize,
    target: PairTruthLabel,
    sep_scale: f64,
    style: S,
    label: &str,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let style: ShapeStyle = style.into();

    let series = feats
        .iter()
        .step_by(stride)
        .filter(move |f| f.label == target)
        .map(|f| Circle::new((f.dt, f.sep_rad * sep_scale), 2, style.clone().filled()));

    chart
        .draw_series(series)?
        .label(label)
        .legend(move |(x, y)| Circle::new((x + 8, y), 4, style.clone().filled()));

    Ok(())
}

fn draw_hist_series<DB: DrawingBackend, S: Into<ShapeStyle> + Clone>(
    chart: &mut ChartContext<'_, DB, Cartesian2d<RangedCoordf64, RangedCoordi32>>,
    counts: &[usize],
    lo: f64,
    hi: f64,
    style: S,
    label: &str,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let style: ShapeStyle = style.into();
    let bins = counts.len();
    if bins == 0 || !(hi > lo) {
        return Ok(());
    }
    let w = (hi - lo) / (bins as f64);

    let series = counts.iter().enumerate().map(|(i, &c)| {
        let x0 = lo + (i as f64) * w;
        let x1 = x0 + w;
        Rectangle::new([(x0, 0), (x1, c as i32)], style.clone().filled())
    });

    chart
        .draw_series(series)?
        .label(label)
        .legend(move |(x, y)| {
            Rectangle::new([(x, y - 5), (x + 18, y + 5)], style.clone().filled())
        });

    Ok(())
}
