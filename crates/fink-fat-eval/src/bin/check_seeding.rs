use anyhow::{Context, Result};
use camino::{Utf8Path, Utf8PathBuf};
use clap::Parser;
use fink_fat_engine::{
    engine_config::{EngineConfig, load_engine_config_validated},
    tracklet::{Tracklet, track_storage::TrackStorage},
};
use photom::{
    io::polars::{ContiguousChoice, FromPolarsArgs},
    observation_dataset::ObsDataset,
    observer::error_model::ObsErrorModel,
};
use polars::{
    frame::DataFrame,
    lazy::frame::{LazyFrame, ScanArgsParquet},
};

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

pub fn load_config(config_path: &Utf8Path) -> Result<EngineConfig> {
    load_engine_config_validated(config_path).context("failed to load engine config")
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

use ahash::{HashMap, HashMapExt, HashSet};

// ─── Ground-truth helpers ─────────────────────────────────────────────────────

/// Map from alert `id` to its ground-truth `traj_id`.
fn build_id_to_traj(df: &DataFrame) -> HashMap<u64, u32> {
    let ids = df.column("id").unwrap().u64().unwrap();
    let traj_ids = df.column("traj_id").unwrap().u32().unwrap();
    ids.into_iter()
        .zip(traj_ids.into_iter())
        .filter_map(|(id, tid)| Some((id?, tid?)))
        .collect()
}

/// For a given night, return the set of `traj_id`s that have at least
/// `min_alerts` alerts — i.e. the asteroids that *could* produce a seed.
fn seedable_traj_ids(df: &DataFrame, night: u32, min_alerts: usize) -> HashSet<u32> {
    let night_col = df.column("night_id").unwrap().u32().unwrap();
    let traj_col = df.column("traj_id").unwrap().u32().unwrap();

    let mut counts: HashMap<u32, usize> = HashMap::new();
    for (n, t) in night_col.into_iter().zip(traj_col.into_iter()) {
        if n == Some(night) {
            if let Some(tid) = t {
                *counts.entry(tid).or_default() += 1;
            }
        }
    }
    counts
        .into_iter()
        .filter(|(_, c)| *c >= min_alerts)
        .map(|(tid, _)| tid)
        .collect()
}

// ─── Per-night seed diagnostics ───────────────────────────────────────────────

/// Threshold above which a night is considered to exhibit pathological
/// over-generation.
///
/// Chosen empirically to separate the two observed regimes:
/// - Normal     : `Pur/Cov` ≈ 1.1–1.6
/// - Explosive  : `Pur/Cov` > 3.0
const OVERGEN_THRESHOLD: f64 = 3.0;

/// Diagnostics for seed generation on a single night.
struct NightSeedDiag {
    night: u32,
    n_alerts: usize,
    n_tracklets: usize,
    /// Asteroids with ≥ `min_alerts` in this night (ground-truth seedable).
    n_seedable: usize,
    /// Seedable asteroids covered by at least one pure tracklet (TP at asteroid level).
    n_covered: usize,
    /// Tracklets whose alerts all belong to the same `traj_id` (pure tracklets).
    n_pure: usize,
    /// Tracklets containing alerts from more than one `traj_id` (mixed tracklets).
    n_mixed: usize,
    /// Seedable asteroids covered by more than one pure tracklet (duplicated coverage).
    n_duplicated: usize,
    /// Total number of pure tracklets assigned to covered asteroids.
    /// Used to derive the mean pure tracklets per covered asteroid.
    pure_tracklets_on_covered: usize,
}

impl NightSeedDiag {
    /// Asteroid-level recall: fraction of seedable asteroids covered by ≥1 pure tracklet.
    fn recall(&self) -> f64 {
        if self.n_seedable == 0 {
            return 0.0;
        }
        self.n_covered as f64 / self.n_seedable as f64
    }

    /// Tracklet-level purity: fraction of tracklets that are pure.
    fn purity(&self) -> f64 {
        if self.n_tracklets == 0 {
            return 0.0;
        }
        self.n_pure as f64 / self.n_tracklets as f64
    }

    /// Mean number of pure tracklets generated per covered asteroid.
    ///
    /// A value of 1.0 means each covered asteroid produced exactly one pure
    /// tracklet. Values above 1.0 indicate redundant seed generation.
    fn mean_pure_per_covered(&self) -> f64 {
        if self.n_covered == 0 {
            return 0.0;
        }
        self.pure_tracklets_on_covered as f64 / self.n_covered as f64
    }

    /// Duplication rate: fraction of covered asteroids that are covered by
    /// more than one pure tracklet.
    fn duplication_rate(&self) -> f64 {
        if self.n_covered == 0 {
            return 0.0;
        }
        self.n_duplicated as f64 / self.n_covered as f64
    }
}

/// Evaluate seed quality for one night.
///
/// A tracklet is **pure** if every alert it contains belongs to the same
/// ground-truth `traj_id`. A seedable asteroid is **covered** if at least one
/// pure tracklet references it.
///
/// **Over-generation metrics** — beyond the binary covered/not-covered
/// distinction, this function also records:
///
/// - `n_duplicated`: the number of seedable asteroids covered by *more than
///   one* pure tracklet. A non-zero value indicates that the seeding step
///   emits redundant candidates for the same object.
/// - `pure_tracklets_on_covered`: the total count of pure tracklets assigned
///   to covered asteroids, used to derive
///   [`NightSeedDiag::mean_pure_per_covered`].
///
/// Arguments
/// ---------
/// * `night_id`   – Numeric night identifier.
/// * `n_alerts`   – Number of raw alerts for this night.
/// * `tracklets`  – Seeds produced by `generate_seeds`.
/// * `id_to_traj` – Global alert-id → traj-id lookup table.
/// * `seedable`   – Set of `traj_id`s that have ≥ `min_alerts` in this night.
fn evaluate_night_seeds<'a>(
    night_id: u32,
    n_alerts: usize,
    tracklets: impl Iterator<Item = &'a Tracklet>,
    id_to_traj: &HashMap<u64, u32>,
    seedable: &HashSet<u32>,
) -> NightSeedDiag {
    let mut n_pure = 0usize;
    let mut n_mixed = 0usize;
    let mut n_tracklets = 0usize;
    let mut pure_count_per_traj: HashMap<u32, usize> = HashMap::new();

    for tracklet in tracklets {
        n_tracklets += 1;
        let traj_ids: HashSet<u32> = tracklet
            .obs_keys()
            .iter()
            .filter_map(|k| id_to_traj.get(k).copied())
            .collect();

        match traj_ids.len() {
            0 => {}
            1 => {
                n_pure += 1;
                let tid = *traj_ids.iter().next().unwrap();
                if seedable.contains(&tid) {
                    *pure_count_per_traj.entry(tid).or_default() += 1;
                }
            }
            _ => {
                n_mixed += 1;
            }
        }
    }

    let n_covered = pure_count_per_traj.len();
    let n_duplicated = pure_count_per_traj.values().filter(|&&c| c > 1).count();
    let pure_tracklets_on_covered = pure_count_per_traj.values().sum();

    NightSeedDiag {
        night: night_id,
        n_alerts,
        n_tracklets,
        n_seedable: seedable.len(),
        n_covered,
        n_pure,
        n_mixed,
        n_duplicated,
        pure_tracklets_on_covered,
    }
}

// ─── Aggregate statistics ─────────────────────────────────────────────────────

struct AggregateSeedDiag {
    n_nights: usize,
    total_alerts: usize,
    total_tracklets: usize,
    total_seedable: usize,
    total_covered: usize,
    total_pure: usize,
    total_mixed: usize,
    total_duplicated: usize,
    total_pure_tracklets_on_covered: usize,
    /// Number of nights where `mean_pure_per_covered` exceeds [`OVERGEN_THRESHOLD`],
    /// indicating pathological over-generation.
    n_overgen_nights: usize,
    /// Per-night recall values, for median / std computation.
    per_night_recall: Vec<f64>,
    /// Per-night purity values.
    per_night_purity: Vec<f64>,
    /// Per-night mean-pure-per-covered values.
    per_night_mean_pure_per_covered: Vec<f64>,
}

impl AggregateSeedDiag {
    fn global_recall(&self) -> f64 {
        if self.total_seedable == 0 {
            return 0.0;
        }
        self.total_covered as f64 / self.total_seedable as f64
    }

    fn global_purity(&self) -> f64 {
        if self.total_tracklets == 0 {
            return 0.0;
        }
        self.total_pure as f64 / self.total_tracklets as f64
    }

    /// Global mean number of pure tracklets per covered asteroid.
    ///
    /// Values above 1.0 signal systematic over-generation.
    fn global_mean_pure_per_covered(&self) -> f64 {
        if self.total_covered == 0 {
            return 0.0;
        }
        self.total_pure_tracklets_on_covered as f64 / self.total_covered as f64
    }

    /// Global duplication rate: fraction of covered asteroids that received
    /// more than one pure tracklet, summed across all nights.
    fn global_duplication_rate(&self) -> f64 {
        if self.total_covered == 0 {
            return 0.0;
        }
        self.total_duplicated as f64 / self.total_covered as f64
    }

    fn median(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    fn std(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let var =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        var.sqrt()
    }
}

fn aggregate(nights: &[NightSeedDiag]) -> AggregateSeedDiag {
    let per_night_recall: Vec<f64> = nights.iter().map(|d| d.recall()).collect();
    let per_night_purity: Vec<f64> = nights.iter().map(|d| d.purity()).collect();
    let per_night_mean_pure_per_covered: Vec<f64> =
        nights.iter().map(|d| d.mean_pure_per_covered()).collect();

    let n_overgen_nights = per_night_mean_pure_per_covered
        .iter()
        .filter(|&&v| v > OVERGEN_THRESHOLD)
        .count();

    AggregateSeedDiag {
        n_nights: nights.len(),
        total_alerts: nights.iter().map(|d| d.n_alerts).sum(),
        total_tracklets: nights.iter().map(|d| d.n_tracklets).sum(),
        total_seedable: nights.iter().map(|d| d.n_seedable).sum(),
        total_covered: nights.iter().map(|d| d.n_covered).sum(),
        total_pure: nights.iter().map(|d| d.n_pure).sum(),
        total_mixed: nights.iter().map(|d| d.n_mixed).sum(),
        total_duplicated: nights.iter().map(|d| d.n_duplicated).sum(),
        total_pure_tracklets_on_covered: nights.iter().map(|d| d.pure_tracklets_on_covered).sum(),
        n_overgen_nights,
        per_night_recall,
        per_night_purity,
        per_night_mean_pure_per_covered,
    }
}

// ─── Display ──────────────────────────────────────────────────────────────────

/// Render a compact ASCII histogram for a slice of `f64` values.
///
/// Bins are computed over `[min, max]` with `n_bins` equal-width buckets.
/// Each bar is scaled so that the tallest bucket fills `bar_width` characters.
///
/// Arguments
/// ---------
/// * `values`    – Input sample (need not be sorted).
/// * `n_bins`    – Number of histogram buckets.
/// * `bar_width` – Maximum bar length in characters.
fn ascii_histogram(values: &[f64], n_bins: usize, bar_width: usize) {
    if values.is_empty() || n_bins == 0 {
        return;
    }

    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if (max - min).abs() < f64::EPSILON {
        println!("  (all values identical: {:.4})", min);
        return;
    }

    let width = (max - min) / n_bins as f64;
    let mut counts = vec![0usize; n_bins];

    for &v in values {
        let idx = ((v - min) / width).floor() as usize;
        let idx = idx.min(n_bins - 1);
        counts[idx] += 1;
    }

    let max_count = *counts.iter().max().unwrap_or(&1);

    for (i, &count) in counts.iter().enumerate() {
        let lo = min + i as f64 * width;
        let hi = lo + width;
        let filled = if max_count > 0 {
            (count * bar_width) / max_count
        } else {
            0
        };
        println!(
            "  [{:6.2},{:6.2}) │{:<bar_width$}│ {}",
            lo,
            hi,
            "█".repeat(filled),
            count,
            bar_width = bar_width,
        );
    }
}

fn print_per_night(diags: &[NightSeedDiag]) {
    println!();
    println!(
        "{:<8} {:>8} {:>10} {:>10} {:>10} {:>8} {:>8} {:>8} {:>10} {:>6}",
        "Night",
        "Alerts",
        "Tracklets",
        "Seedable",
        "Covered",
        "Recall",
        "Pure",
        "Purity",
        "Pur/Cov",
        "Dup%"
    );
    println!("{}", "─".repeat(100));
    for d in diags {
        println!(
            "{:<8} {:>8} {:>10} {:>10} {:>10} {:>7.3} {:>8} {:>7.3} {:>10.3} {:>5.1}",
            d.night,
            d.n_alerts,
            d.n_tracklets,
            d.n_seedable,
            d.n_covered,
            d.recall(),
            d.n_pure,
            d.purity(),
            d.mean_pure_per_covered(),
            d.duplication_rate() * 100.0,
        );
    }
}

fn print_aggregate(agg: &AggregateSeedDiag) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              Seed generation — aggregate statistics          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!();
    println!("── Dataset ─────────────────────────────────────────────────────");
    println!("  Nights processed : {}", agg.n_nights);
    println!("  Total alerts     : {}", agg.total_alerts);
    println!("  Total tracklets  : {}", agg.total_tracklets);

    println!();
    println!("── Global metrics ──────────────────────────────────────────────");
    println!(
        "  Recall  (covered / seedable) : {:.4}  ({}/{})",
        agg.global_recall(),
        agg.total_covered,
        agg.total_seedable,
    );
    println!(
        "  Purity  (pure / total)       : {:.4}  ({}/{})",
        agg.global_purity(),
        agg.total_pure,
        agg.total_tracklets,
    );
    println!("  Mixed tracklets              : {}", agg.total_mixed);

    println!();
    println!("── Over-generation ─────────────────────────────────────────────");
    println!(
        "  Mean pure tracklets / covered asteroid : {:.4}  ({}/{})",
        agg.global_mean_pure_per_covered(),
        agg.total_pure_tracklets_on_covered,
        agg.total_covered,
    );
    println!(
        "  Duplication rate (>1 pure / asteroid)  : {:.2}%  ({}/{})",
        agg.global_duplication_rate() * 100.0,
        agg.total_duplicated,
        agg.total_covered,
    );
    println!(
        "  Explosive nights (Pur/Cov > {:.1})        : {} / {}  ({:.1}%)",
        OVERGEN_THRESHOLD,
        agg.n_overgen_nights,
        agg.n_nights,
        100.0 * agg.n_overgen_nights as f64 / agg.n_nights as f64,
    );
    println!();
    println!("  Interpretation guide:");
    println!("    pure/cov = 1.00 → exactly one pure tracklet per asteroid  ✓");
    println!("    pure/cov > 1.00 → redundant tracklets (over-generation)   ↑");
    println!("    dup%     = 0%   → no asteroid covered more than once      ✓");

    println!();
    println!("── Per-night recall distribution ───────────────────────────────");
    println!(
        "  Mean   : {:.4}   Median : {:.4}   Std : {:.4}",
        agg.per_night_recall.iter().sum::<f64>() / agg.n_nights as f64,
        AggregateSeedDiag::median(&agg.per_night_recall),
        AggregateSeedDiag::std(&agg.per_night_recall),
    );
    println!(
        "  Min    : {:.4}   Max    : {:.4}",
        agg.per_night_recall
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min),
        agg.per_night_recall
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max),
    );
    ascii_histogram(&agg.per_night_recall, 10, 40);

    println!();
    println!("── Per-night purity distribution ───────────────────────────────");
    println!(
        "  Mean   : {:.4}   Median : {:.4}   Std : {:.4}",
        agg.per_night_purity.iter().sum::<f64>() / agg.n_nights as f64,
        AggregateSeedDiag::median(&agg.per_night_purity),
        AggregateSeedDiag::std(&agg.per_night_purity),
    );
    println!(
        "  Min    : {:.4}   Max    : {:.4}",
        agg.per_night_purity
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min),
        agg.per_night_purity
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max),
    );
    ascii_histogram(&agg.per_night_purity, 10, 40);

    println!();
    println!("── Per-night over-generation (Pur/Cov) distribution ────────────");
    println!(
        "  Mean   : {:.4}   Median : {:.4}   Std : {:.4}",
        agg.per_night_mean_pure_per_covered.iter().sum::<f64>() / agg.n_nights as f64,
        AggregateSeedDiag::median(&agg.per_night_mean_pure_per_covered),
        AggregateSeedDiag::std(&agg.per_night_mean_pure_per_covered),
    );
    println!(
        "  Min    : {:.4}   Max    : {:.4}",
        agg.per_night_mean_pure_per_covered
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min),
        agg.per_night_mean_pure_per_covered
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max),
    );
    println!("  Full range (all nights):");
    ascii_histogram(&agg.per_night_mean_pure_per_covered, 14, 40);

    // Zoomed view restricted to normal nights for readability.
    let normal_nights: Vec<f64> = agg
        .per_night_mean_pure_per_covered
        .iter()
        .cloned()
        .filter(|&v| v <= OVERGEN_THRESHOLD)
        .collect();
    if !normal_nights.is_empty() {
        println!(
            "  Normal nights only (Pur/Cov ≤ {:.1})  [{} / {} nights]:",
            OVERGEN_THRESHOLD,
            normal_nights.len(),
            agg.n_nights,
        );
        ascii_histogram(&normal_nights, 10, 40);
    }
}

use plotters::prelude::*;

// ─── Plotting ─────────────────────────────────────────────────────────────────

/// Compute histogram bins over `[min, max]`.
///
/// Returns a vector of `(bin_lo, bin_hi, count)` tuples.
///
/// Arguments
/// ---------
/// * `values` – Input sample.
/// * `n_bins` – Number of equal-width buckets.
fn compute_bins(values: &[f64], n_bins: usize) -> Vec<(f64, f64, usize)> {
    if values.is_empty() || n_bins == 0 {
        return vec![];
    }
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() < f64::EPSILON {
        return vec![(min, max, values.len())];
    }
    let width = (max - min) / n_bins as f64;
    let mut counts = vec![0usize; n_bins];
    for &v in values {
        let idx = ((v - min) / width).floor() as usize;
        counts[idx.min(n_bins - 1)] += 1;
    }
    counts
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            let lo = min + i as f64 * width;
            (lo, lo + width, c)
        })
        .collect()
}

/// Draw a bar histogram on a `ChartContext`.
///
/// Arguments
/// ---------
/// * `chart`  – Target chart context.
/// * `bins`   – Output of [`compute_bins`].
/// * `color`  – Fill color for the bars.
fn draw_histogram<DB: DrawingBackend>(
    chart: &mut ChartContext<
        DB,
        Cartesian2d<plotters::coord::types::RangedCoordf64, plotters::coord::types::RangedCoordu32>,
    >,
    bins: &[(f64, f64, usize)],
    color: RGBColor,
) -> Result<(), Box<dyn std::error::Error>>
where
    DB::ErrorType: 'static,
{
    for &(lo, hi, count) in bins {
        chart.draw_series(std::iter::once(Rectangle::new(
            [(lo, 0u32), (hi, count as u32)],
            color.filled(),
        )))?;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(lo, 0u32), (hi, count as u32)],
            BLACK.stroke_width(1),
        )))?;
    }
    Ok(())
}

/// Colors used across all plots for visual consistency.
const COL_RECALL: RGBColor = RGBColor(70, 130, 180); // steel blue
const COL_PURITY: RGBColor = RGBColor(60, 179, 113); // medium sea green
const COL_OVERGEN: RGBColor = RGBColor(210, 105, 30); // chocolate
const COL_SCATTER: RGBColor = RGBColor(148, 103, 189); // medium purple
const COL_EXPLOSIVE: RGBColor = RGBColor(214, 39, 40); // red

/// Plot 1 — Per-night recall histogram (full range + zoomed on [0.8, 1.0]).
///
/// Saved as `seed_diag_recall.svg`.
fn plot_recall_histogram(diags: &[NightSeedDiag]) -> Result<(), Box<dyn std::error::Error>> {
    let recalls: Vec<f64> = diags.iter().map(|d| d.recall()).collect();

    let root = SVGBackend::new("seed_diag_recall.svg", (900, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let (left, right) = root.split_horizontally(450);

    // — Full range —
    {
        let bins = compute_bins(&recalls, 20);
        let max_count = bins.iter().map(|b| b.2).max().unwrap_or(1) as u32;
        let mut chart = ChartBuilder::on(&left)
            .caption("Per-night recall (full range)", ("sans-serif", 16))
            .margin(20)
            .x_label_area_size(35)
            .y_label_area_size(45)
            .build_cartesian_2d(0f64..1.05f64, 0u32..max_count + 2)?;
        chart
            .configure_mesh()
            .x_desc("Recall")
            .y_desc("Nights")
            .draw()?;
        draw_histogram(&mut chart, &bins, COL_RECALL)?;
    }

    // — Zoomed on [0.8, 1.0] —
    {
        let zoomed: Vec<f64> = recalls.iter().cloned().filter(|&v| v >= 0.8).collect();
        let bins = compute_bins(&zoomed, 20);
        let max_count = bins.iter().map(|b| b.2).max().unwrap_or(1) as u32;
        let mut chart = ChartBuilder::on(&right)
            .caption("Per-night recall (zoom [0.8, 1.0])", ("sans-serif", 16))
            .margin(20)
            .x_label_area_size(35)
            .y_label_area_size(45)
            .build_cartesian_2d(0.8f64..1.005f64, 0u32..max_count + 2)?;
        chart
            .configure_mesh()
            .x_desc("Recall")
            .y_desc("Nights")
            .draw()?;
        draw_histogram(&mut chart, &bins, COL_RECALL)?;
    }

    root.present()?;
    println!("  Saved: seed_diag_recall.svg");
    Ok(())
}

/// Plot 2 — Per-night purity histogram (full range + zoomed on [0.8, 1.0]).
///
/// Saved as `seed_diag_purity.svg`.
fn plot_purity_histogram(diags: &[NightSeedDiag]) -> Result<(), Box<dyn std::error::Error>> {
    let purities: Vec<f64> = diags.iter().map(|d| d.purity()).collect();

    let root = SVGBackend::new("seed_diag_purity.svg", (900, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let (left, right) = root.split_horizontally(450);

    // — Full range —
    {
        let bins = compute_bins(&purities, 20);
        let max_count = bins.iter().map(|b| b.2).max().unwrap_or(1) as u32;
        let mut chart = ChartBuilder::on(&left)
            .caption("Per-night purity (full range)", ("sans-serif", 16))
            .margin(20)
            .x_label_area_size(35)
            .y_label_area_size(45)
            .build_cartesian_2d(0f64..1.05f64, 0u32..max_count + 2)?;
        chart
            .configure_mesh()
            .x_desc("Purity")
            .y_desc("Nights")
            .draw()?;
        draw_histogram(&mut chart, &bins, COL_PURITY)?;
    }

    // — Zoomed on [0.8, 1.0] —
    {
        let zoomed: Vec<f64> = purities.iter().cloned().filter(|&v| v >= 0.8).collect();
        let bins = compute_bins(&zoomed, 20);
        let max_count = bins.iter().map(|b| b.2).max().unwrap_or(1) as u32;
        let mut chart = ChartBuilder::on(&right)
            .caption("Per-night purity (zoom [0.8, 1.0])", ("sans-serif", 16))
            .margin(20)
            .x_label_area_size(35)
            .y_label_area_size(45)
            .build_cartesian_2d(0.8f64..1.005f64, 0u32..max_count + 2)?;
        chart
            .configure_mesh()
            .x_desc("Purity")
            .y_desc("Nights")
            .draw()?;
        draw_histogram(&mut chart, &bins, COL_PURITY)?;
    }

    root.present()?;
    println!("  Saved: seed_diag_purity.svg");
    Ok(())
}

/// Plot 3 — Per-night Pur/Cov histogram (full range + normal nights only).
///
/// Normal nights are defined as `Pur/Cov ≤ OVERGEN_THRESHOLD`.
/// Saved as `seed_diag_overgen.svg`.
fn plot_overgen_histogram(diags: &[NightSeedDiag]) -> Result<(), Box<dyn std::error::Error>> {
    let overgen: Vec<f64> = diags.iter().map(|d| d.mean_pure_per_covered()).collect();

    let root = SVGBackend::new("seed_diag_overgen.svg", (900, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let (left, right) = root.split_horizontally(450);

    // — Full range —
    {
        let bins = compute_bins(&overgen, 20);
        let max_count = bins.iter().map(|b| b.2).max().unwrap_or(1) as u32;
        let x_max = overgen.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 1.05;
        let mut chart = ChartBuilder::on(&left)
            .caption("Pure/Covered ratio (full range)", ("sans-serif", 16))
            .margin(20)
            .x_label_area_size(35)
            .y_label_area_size(45)
            .build_cartesian_2d(0f64..x_max, 0u32..max_count + 2)?;
        chart
            .configure_mesh()
            .x_desc("Pur/Cov")
            .y_desc("Nights")
            .draw()?;
        draw_histogram(&mut chart, &bins, COL_OVERGEN)?;
        // Threshold line
        chart.draw_series(std::iter::once(PathElement::new(
            vec![
                (OVERGEN_THRESHOLD, 0u32),
                (OVERGEN_THRESHOLD, max_count + 2),
            ],
            COL_EXPLOSIVE.stroke_width(2),
        )))?;
    }

    // — Normal nights —
    {
        let normal: Vec<f64> = overgen
            .iter()
            .cloned()
            .filter(|&v| v <= OVERGEN_THRESHOLD)
            .collect();
        let bins = compute_bins(&normal, 20);
        let max_count = bins.iter().map(|b| b.2).max().unwrap_or(1) as u32;
        let mut chart = ChartBuilder::on(&right)
            .caption(
                format!("Pure/Covered (≤ {:.1})", OVERGEN_THRESHOLD),
                ("sans-serif", 16),
            )
            .margin(20)
            .x_label_area_size(35)
            .y_label_area_size(45)
            .build_cartesian_2d(0f64..OVERGEN_THRESHOLD * 1.02, 0u32..max_count + 2)?;
        chart
            .configure_mesh()
            .x_desc("Pur/Cov")
            .y_desc("Nights")
            .draw()?;
        draw_histogram(&mut chart, &bins, COL_OVERGEN)?;
    }

    root.present()?;
    println!("  Saved: seed_diag_overgen.svg");
    Ok(())
}

/// Plot 4 — Recall vs purity scatter, colored by Pur/Cov.
///
/// Each point is one night. Color encodes the Pur/Cov ratio:
/// - blue   → low over-generation (ratio ≈ 1),
/// - red    → explosive over-generation (ratio > [`OVERGEN_THRESHOLD`]).
///
/// Saved as `seed_diag_scatter.svg`.
fn plot_recall_purity_scatter(diags: &[NightSeedDiag]) -> Result<(), Box<dyn std::error::Error>> {
    let root = SVGBackend::new("seed_diag_scatter.svg", (700, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_overgen = diags
        .iter()
        .map(|d| d.mean_pure_per_covered())
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1.0);

    let mut chart = ChartBuilder::on(&root)
        .caption("Recall vs Purity per night", ("sans-serif", 18))
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0f64..1.05f64, 0f64..1.05f64)?;

    chart
        .configure_mesh()
        .x_desc("Recall")
        .y_desc("Purity")
        .draw()?;

    // Color: linearly interpolate COL_RECALL → COL_EXPLOSIVE by normalized Pur/Cov.
    let points: Vec<(f64, f64, RGBColor)> = diags
        .iter()
        .map(|d| {
            let t = (d.mean_pure_per_covered() / max_overgen).clamp(0.0, 1.0);
            let r = (COL_RECALL.0 as f64 * (1.0 - t) + COL_EXPLOSIVE.0 as f64 * t) as u8;
            let g = (COL_RECALL.1 as f64 * (1.0 - t) + COL_EXPLOSIVE.1 as f64 * t) as u8;
            let b = (COL_RECALL.2 as f64 * (1.0 - t) + COL_EXPLOSIVE.2 as f64 * t) as u8;
            (d.recall(), d.purity(), RGBColor(r, g, b))
        })
        .collect();

    chart.draw_series(
        points
            .iter()
            .map(|&(x, y, color)| Circle::new((x, y), 5, color.filled())),
    )?;

    // Reference lines at recall = 1 and purity = 1.
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(1.0f64, 0.0f64), (1.0f64, 1.05f64)],
        BLACK.stroke_width(1),
    )))?;
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(0.0f64, 1.0f64), (1.05f64, 1.0f64)],
        BLACK.stroke_width(1),
    )))?;

    root.present()?;
    println!("  Saved: seed_diag_scatter.svg");
    Ok(())
}

/// Plot 5 — Timeline of recall, purity and Pur/Cov by night index.
///
/// The three series share the same x-axis (night index, sorted) but use
/// two y-axes:
/// - left  → recall and purity in `[0, 1]`,
/// - right → Pur/Cov ratio (capped at [`OVERGEN_CAP`] for readability).
///
/// A Pur/Cov ratio close to 1 indicates a balanced filter: purity and
/// recall are comparable. Values significantly above 1 indicate that the
/// filter is overly conservative — it builds mostly pure trajectories but
/// misses a large fraction of true asteroids.
///
/// Nights are plotted in sorted order (matching `night_ids`).
/// Saved as `seed_diag_timeline.svg`.
fn plot_timeline(diags: &[NightSeedDiag]) -> Result<(), Box<dyn std::error::Error>> {
    const OVERGEN_CAP: f64 = 10.0;

    let root = SVGBackend::new("seed_diag_timeline.svg", (1600, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let n = diags.len();
    if n == 0 {
        return Ok(());
    }

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Recall / Purity / Pur·Cov timeline (sorted by night)",
            ("sans-serif", 32),
        )
        .margin(30)
        .x_label_area_size(70)
        .y_label_area_size(80)
        .right_y_label_area_size(80)
        .build_cartesian_2d(0usize..n, 0f64..1.05f64)?
        .set_secondary_coord(0usize..n, 0f64..OVERGEN_CAP * 1.05);

    chart
        .configure_mesh()
        .x_desc("Night index (sorted)")
        .y_desc("Recall / Purity")
        .x_label_style(("sans-serif", 22))
        .y_label_style(("sans-serif", 22))
        .axis_desc_style(("sans-serif", 28))
        .light_line_style(RGBColor(220, 225, 235))
        .draw()?;

    chart
        .configure_secondary_axes()
        .y_desc(format!("Pur/Cov  (capped at {OVERGEN_CAP:.0})"))
        .label_style(("sans-serif", 22))
        .axis_desc_style(("sans-serif", 28))
        .draw()?;

    // Recall
    chart
        .draw_series(LineSeries::new(
            diags.iter().enumerate().map(|(i, d)| (i, d.recall())),
            COL_RECALL.stroke_width(3),
        ))?
        .label("Recall")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 30, y)], COL_RECALL.stroke_width(3)));

    // Purity
    chart
        .draw_series(LineSeries::new(
            diags.iter().enumerate().map(|(i, d)| (i, d.purity())),
            COL_PURITY.stroke_width(3),
        ))?
        .label("Purity")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 30, y)], COL_PURITY.stroke_width(3)));

    // Pur/Cov on secondary axis (capped)
    chart
        .draw_secondary_series(LineSeries::new(
            diags
                .iter()
                .enumerate()
                .map(|(i, d)| (i, d.mean_pure_per_covered().min(OVERGEN_CAP))),
            COL_OVERGEN.stroke_width(3),
        ))?
        .label(format!("Pur/Cov  (cap {OVERGEN_CAP:.0})"))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 30, y)], COL_OVERGEN.stroke_width(3)));

    // Threshold reference line on secondary axis.
    chart
        .draw_secondary_series(std::iter::once(PathElement::new(
            vec![(0usize, OVERGEN_THRESHOLD), (n, OVERGEN_THRESHOLD)],
            Into::<ShapeStyle>::into(COL_EXPLOSIVE).stroke_width(2),
        )))?
        .label(format!("Threshold  ({OVERGEN_THRESHOLD:.1})"))
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 30, y)],
                Into::<ShapeStyle>::into(COL_EXPLOSIVE).stroke_width(2),
            )
        });

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerLeft)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .label_font(("sans-serif", 26))
        .draw()?;

    root.present()?;
    println!("  Saved: seed_diag_timeline.svg");
    Ok(())
}

/// Generate and save all diagnostic plots to the current directory.
///
/// Plots produced
/// --------------
/// 1. `seed_diag_recall.svg`   — Per-night recall histogram (full + zoom).
/// 2. `seed_diag_purity.svg`   — Per-night purity histogram (full + zoom).
/// 3. `seed_diag_overgen.svg`  — Pur/Cov histogram (full + normal nights).
/// 4. `seed_diag_scatter.svg`  — Recall vs purity scatter colored by Pur/Cov.
/// 5. `seed_diag_timeline.svg` — Recall, purity and Pur/Cov over sorted nights.
///
/// Arguments
/// ---------
/// * `diags` – Per-night diagnostics, sorted by night id.
fn save_plots(diags: &[NightSeedDiag]) -> Result<()> {
    println!();
    println!("── Saving plots ─────────────────────────────────────────────────");
    plot_recall_histogram(diags).map_err(|e| anyhow::anyhow!("recall histogram: {e}"))?;
    plot_purity_histogram(diags).map_err(|e| anyhow::anyhow!("purity histogram: {e}"))?;
    plot_overgen_histogram(diags).map_err(|e| anyhow::anyhow!("overgen histogram: {e}"))?;
    plot_recall_purity_scatter(diags).map_err(|e| anyhow::anyhow!("recall/purity scatter: {e}"))?;
    plot_timeline(diags).map_err(|e| anyhow::anyhow!("timeline: {e}"))?;
    Ok(())
}

pub fn main() -> Result<()> {
    let cli = Cli::parse();

    let (df, obs_dataset) = load_data(&cli.alerts);

    let mut night_ids: Vec<_> = obs_dataset.iter_night_id().unwrap().collect();
    night_ids.sort();

    let engine_config = load_config(&cli.config)?;
    let id_to_traj = build_id_to_traj(&df);

    // Minimum alerts per asteroid to be considered seedable (pair = 2, triplet = 3).
    const MIN_ALERTS_FOR_SEED: usize = 2;

    let mut per_night_diags: Vec<NightSeedDiag> = Vec::with_capacity(night_ids.len());

    for night_id in &night_ids {
        println!("Building seeds for {}", night_id);

        let night_u32 = night_id.0;

        let observations = match obs_dataset.materialize_night(night_id) {
            Some(photom::observation_dataset::iter::MemLayoutObservations::Contiguous(s)) => s,
            _ => {
                eprintln!("Skipping night {night_u32}: not contiguous or missing");
                continue;
            }
        };

        let (track_storage, _) = TrackStorage::new()
            .generate_seeds(&engine_config, observations)
            .unwrap();

        let seedable = seedable_traj_ids(&df, night_u32, MIN_ALERTS_FOR_SEED);

        let diag = evaluate_night_seeds(
            night_u32,
            observations.len(),
            track_storage.iter_tracklets(),
            &id_to_traj,
            &seedable,
        );

        per_night_diags.push(diag);
    }

    print_per_night(&per_night_diags);
    let agg = aggregate(&per_night_diags);
    print_aggregate(&agg);

    print_aggregate(&agg);
    save_plots(&per_night_diags)?;

    Ok(())
}
