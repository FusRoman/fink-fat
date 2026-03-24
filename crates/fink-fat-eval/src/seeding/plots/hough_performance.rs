//! Hough-transform seeding performance instrumentation.
//!
//! This module re-runs nightly Hough seeding with the active engine
//! configuration in order to measure runtime and expose the internal counters
//! produced by [`crate::seeding::hough::build_hough_seeds_for_night`].
//!
//! The collected diagnostics are written to disk as a CSV table and as a small
//! set of nightly plots that summarize:
//!
//! - runtime,
//! - vote throughput,
//! - accumulator occupancy,
//! - peak counts,
//! - seed quality against the ground-truth map.
//!
//! The module is only activated when `ctx.engine_config.seeding.method` is
//! [`SeedingMethod::Hough`](fink_fat_engine::engine_config::seeding_config::SeedingMethod::Hough).

use std::{path::Path, time::Instant};

use ahash::AHashSet;
use anyhow::Result;
use camino::Utf8Path;
use fink_fat_engine::{
    engine_config::seeding_config::SeedingMethod,
    pipeline::PipelineContext,
    seeding::hough::{self, HoughSeedStats},
};
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;

use crate::truth_sso::{TrajId, TruthClass, TruthSSO};

const C_RUNTIME: RGBColor = RGBColor(70, 130, 180);
const C_VOTES: RGBColor = RGBColor(0, 128, 128);
const C_BINS: RGBColor = RGBColor(120, 120, 120);
const C_PEAKS: RGBColor = RGBColor(255, 140, 0);
const C_PURITY: RGBColor = RGBColor(65, 105, 225);
const C_RECALL: RGBColor = RGBColor(220, 20, 60);

/// Per-night row of Hough performance measurements.
///
/// This is the internal record type used by the Hough diagnostics pipeline.
/// Each row aggregates the measurements collected for a single night after
/// replaying Hough seeding:
///
/// - runtime of the replay,
/// - raw Hough counters returned by the seeder,
/// - seed classification counts against the truth map,
/// - recovery counts for the ground-truth trajectories.
///
/// The same record feeds both CSV export and plotting so the numerical values
/// stay consistent across all generated artefacts.
#[derive(Debug, Clone)]
struct HoughNightPerfRow {
    /// Night label used in the CSV and on the x-axis of plots.
    night_label: String,
    /// Number of alerts processed for this night.
    n_alerts: usize,
    /// Wall-clock time spent replaying Hough seeding for this night, in milliseconds.
    elapsed_ms: f64,
    /// Number of velocity hypotheses evaluated by the Hough grid.
    n_velocity_hypotheses: u64,
    /// Approximate number of votes cast into the sparse accumulator.
    n_votes_total: u64,
    /// Vote throughput derived from `n_votes_total / elapsed_s`.
    votes_per_sec: f64,
    /// Number of non-empty sparse accumulator cells.
    n_accumulator_bins: u64,
    /// Number of accumulator peaks kept after score ranking.
    n_peaks: u64,
    /// Number of peaks that survived the photometric compatibility filter.
    n_peaks_after_photometric_filter: u64,
    /// Number of pair seeds emitted from the retained peaks.
    n_pair_seeds: u64,
    /// Number of triplet seeds emitted from the retained peaks.
    n_triplet_seeds: u64,
    /// Total number of emitted seeds.
    n_seeds_total: usize,
    /// Number of seeds classified as true positives.
    n_tp: usize,
    /// Number of seeds classified as false positives.
    n_fp: usize,
    /// Number of seeds classified as unknown.
    n_unknown: usize,
    /// Number of trajectories that were recoverable on this night.
    n_recoverable_trajs: usize,
    /// Number of recoverable trajectories actually recovered by at least one TP seed.
    n_recovered_trajs: usize,
    /// Seed purity, defined as `n_tp / (n_tp + n_fp)`.
    purity: f64,
    /// Trajectory recall, defined as `n_recovered_trajs / n_recoverable_trajs`.
    recall: f64,
}

impl HoughNightPerfRow {
    /// Build one Hough performance row from raw measurements and truth counts.
    ///
    /// This constructor performs the small derived computations used in the
    /// diagnostics layer:
    ///
    /// - purity from the TP/FP split,
    /// - recall from the recovered vs recoverable trajectories,
    /// - votes-per-second from the accumulator vote count and measured runtime.
    ///
    /// Arguments
    /// ---------
    /// * `night_label` – Human-readable label for the processed night.
    /// * `n_alerts` – Number of alerts used as Hough input for the night.
    /// * `elapsed_ms` – Measured wall-clock duration of the replay, in milliseconds.
    /// * `stats` – Internal counters returned by the Hough seeder.
    /// * `n_seeds_total` – Total number of seeds emitted for the night.
    /// * `n_tp` – Number of seeds classified as true positives.
    /// * `n_fp` – Number of seeds classified as false positives.
    /// * `n_unknown` – Number of seeds classified as unknown.
    /// * `n_recoverable_trajs` – Number of ground-truth trajectories recoverable on this night.
    /// * `n_recovered_trajs` – Number of recoverable trajectories recovered by at least one TP seed.
    ///
    /// Return
    /// ------
    /// A fully populated [`HoughNightPerfRow`] ready for CSV export and plotting.
    fn from_measurement(
        night_label: String,
        n_alerts: usize,
        elapsed_ms: f64,
        stats: HoughSeedStats,
        n_seeds_total: usize,
        n_tp: usize,
        n_fp: usize,
        n_unknown: usize,
        n_recoverable_trajs: usize,
        n_recovered_trajs: usize,
    ) -> Self {
        let classifiable = n_tp + n_fp;
        let purity = if classifiable == 0 {
            f64::NAN
        } else {
            n_tp as f64 / classifiable as f64
        };
        let recall = if n_recoverable_trajs == 0 {
            f64::NAN
        } else {
            n_recovered_trajs as f64 / n_recoverable_trajs as f64
        };

        let elapsed_s = (elapsed_ms / 1000.0).max(1e-9);
        let n_votes_total = stats.n_velocity_hypotheses * n_alerts as u64;

        Self {
            night_label,
            n_alerts,
            elapsed_ms,
            n_velocity_hypotheses: stats.n_velocity_hypotheses,
            n_votes_total,
            votes_per_sec: n_votes_total as f64 / elapsed_s,
            n_accumulator_bins: stats.n_accumulator_bins,
            n_peaks: stats.n_peaks,
            n_peaks_after_photometric_filter: stats.n_peaks_after_photometric_filter,
            n_pair_seeds: stats.n_pair_seeds,
            n_triplet_seeds: stats.n_triplet_seeds,
            n_seeds_total,
            n_tp,
            n_fp,
            n_unknown,
            n_recoverable_trajs,
            n_recovered_trajs,
            purity,
            recall,
        }
    }
}

/// Build Hough performance diagnostics for each night and write CSV + PNG files.
///
/// This entry point replays Hough seeding with the current configuration so it
/// can collect runtime measurements and the internal counters returned by
/// [`HoughSeedStats`]. It does not mutate the pipeline stores.
///
/// Arguments
/// ---------
/// * `ctx` – Pipeline context containing the alert store and Hough seeding configuration.
/// * `truth` – Ground-truth identity map used to classify the emitted seeds.
/// * `out_dir` – Output directory receiving the CSV file and PNG plots.
///
/// Return
/// ------
/// * `Ok(())` – The diagnostics were written successfully.
/// * `Err(...)` – File I/O, plotting, or classification failed.
///
/// If seeding is not configured in Hough mode, this function exits without
/// writing files.
pub fn hough_performance_plots(
    ctx: &PipelineContext<'_>,
    truth: &TruthSSO,
    out_dir: &Utf8Path,
) -> Result<()> {
    if ctx.engine_config.seeding.method != SeedingMethod::Hough {
        return Ok(());
    }

    std::fs::create_dir_all(out_dir)?;

    let rows = collect_hough_perf_rows(ctx, truth);
    if rows.is_empty() {
        return Ok(());
    }

    write_hough_stats_csv(
        &rows,
        &out_dir.as_std_path().join("hough_performance_stats.csv"),
    )?;
    plot_hough_runtime(&rows, &out_dir.as_std_path().join("hough_runtime_ms.png"))?;
    plot_hough_votes(
        &rows,
        &out_dir.as_std_path().join("hough_votes_per_sec.png"),
    )?;
    plot_hough_accumulator(&rows, &out_dir.as_std_path().join("hough_accumulator.png"))?;
    plot_hough_quality(&rows, &out_dir.as_std_path().join("hough_quality.png"))?;

    Ok(())
}

/// Collect the per-night Hough performance rows used by the CSV export and plots.
///
/// This function is the instrumentation core of the module. It iterates over the
/// nights present in the alert store, reruns Hough seeding for each night, and
/// combines the raw seeder counters with truth-based seed classification.
///
/// The result is a vector ordered by night ID, which makes it suitable for both
/// CSV export and line/bar plots.
///
/// Arguments
/// ---------
/// * `ctx` – Pipeline context supplying the alert store and Hough configuration.
/// * `truth` – Ground-truth map used to classify each produced seed.
///
/// Return
/// ------
/// A vector of [`HoughNightPerfRow`] values, one row per processed night.
fn collect_hough_perf_rows(ctx: &PipelineContext<'_>, truth: &TruthSSO) -> Vec<HoughNightPerfRow> {
    let alert_store = &ctx.runtime_state.alert_store;
    let cfg = &ctx.engine_config.seeding.hough;
    let triplet_only = ctx.engine_config.seeding.triplet_only;

    let mut nights: Vec<_> = alert_store.nights().copied().collect();
    nights.sort();

    let mut out = Vec::with_capacity(nights.len());

    for night_id in nights {
        let Some(alerts) = alert_store.get(&night_id) else {
            continue;
        };

        let t0 = Instant::now();
        let (seeds, stats) =
            hough::build_hough_seeds_for_night(alerts, night_id, cfg, triplet_only);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let mut n_tp = 0usize;
        let mut n_fp = 0usize;
        let mut n_unknown = 0usize;
        let mut recovered: AHashSet<TrajId> = AHashSet::new();

        for seed in &seeds {
            let resolved = match seed.resolve_members(alert_store) {
                Ok(resolved) => resolved,
                Err(_) => {
                    n_unknown += 1;
                    continue;
                }
            };
            match truth.classify(&resolved) {
                TruthClass::TruePositive => {
                    n_tp += 1;
                    if let Some(traj_id) = resolved.first().and_then(|a| truth.get_truth_traj_id(a))
                    {
                        recovered.insert(traj_id);
                    }
                }
                TruthClass::FalsePositive => n_fp += 1,
                TruthClass::Unknown => n_unknown += 1,
            }
        }

        let recoverable: AHashSet<TrajId> = truth.recoverable_seeds(night_id, 2).collect();

        out.push(HoughNightPerfRow::from_measurement(
            night_id.to_string(),
            alerts.len(),
            elapsed_ms,
            stats,
            seeds.len(),
            n_tp,
            n_fp,
            n_unknown,
            recoverable.len(),
            recovered.intersection(&recoverable).count(),
        ));
    }

    out
}

/// Write the collected Hough performance rows as a CSV file.
///
/// The CSV is the machine-readable counterpart of the figures generated by this
/// module. It is intended for offline analysis, comparison across runs, and
/// regression tracking.
///
/// Arguments
/// ---------
/// * `rows` – Nightly performance rows to serialize.
/// * `path` – Target CSV file path.
///
/// Return
/// ------
/// `Ok(())` when the file is written successfully.
fn write_hough_stats_csv(rows: &[HoughNightPerfRow], path: &Path) -> Result<()> {
    let mut csv = String::new();
    csv.push_str(
        "night,n_alerts,elapsed_ms,n_velocity_hypotheses,n_votes_total,votes_per_sec,n_accumulator_bins,n_peaks,n_peaks_after_photometric_filter,n_pair_seeds,n_triplet_seeds,n_seeds_total,n_tp,n_fp,n_unknown,n_recoverable_trajs,n_recovered_trajs,purity,recall\n",
    );

    for r in rows {
        csv.push_str(&format!(
            "{},{},{:.6},{},{},{:.6},{},{},{},{},{},{},{},{},{},{},{},{:.6},{:.6}\n",
            r.night_label,
            r.n_alerts,
            r.elapsed_ms,
            r.n_velocity_hypotheses,
            r.n_votes_total,
            r.votes_per_sec,
            r.n_accumulator_bins,
            r.n_peaks,
            r.n_peaks_after_photometric_filter,
            r.n_pair_seeds,
            r.n_triplet_seeds,
            r.n_seeds_total,
            r.n_tp,
            r.n_fp,
            r.n_unknown,
            r.n_recoverable_trajs,
            r.n_recovered_trajs,
            r.purity,
            r.recall,
        ));
    }

    std::fs::write(path, csv)?;
    tracing::debug!("wrote {}", path.display());
    Ok(())
}

/// Plot per-night Hough runtime in milliseconds.
///
/// The chart shows the end-to-end time required to replay Hough seeding for
/// each night, including accumulator construction and truth-based classification.
///
/// Arguments
/// ---------
/// * `rows` – Per-night measurements to plot.
/// * `path` – Destination PNG path.
///
/// Return
/// ------
/// `Ok(())` when the figure is written successfully.
fn plot_hough_runtime(rows: &[HoughNightPerfRow], path: &Path) -> Result<()> {
    let n = rows.len();
    let y_max = rows
        .iter()
        .map(|r| r.elapsed_ms)
        .fold(0.0_f64, f64::max)
        .max(1.0)
        * 1.15;

    let root =
        BitMapBackend::new(path.to_str().unwrap_or_default(), (1000, 480)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Hough runtime per night (ms)", ("sans-serif", 20))
        .margin(30u32)
        .x_label_area_size(60u32)
        .y_label_area_size(80u32)
        .build_cartesian_2d(-0.5f64..(n as f64 - 0.5), 0f64..y_max)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    configure_night_mesh(&mut chart, rows, "runtime [ms]")?;

    chart
        .draw_series(rows.iter().enumerate().map(|(i, r)| {
            let x = i as f64;
            Rectangle::new(
                [(x - 0.35, 0.0), (x + 0.35, r.elapsed_ms)],
                C_RUNTIME.filled(),
            )
        }))
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    root.present().map_err(|e| anyhow::anyhow!("{e:?}"))?;
    tracing::debug!("wrote {}", path.display());
    Ok(())
}

/// Plot per-night Hough vote throughput in votes per second.
///
/// This plot normalizes the accumulator work by the measured replay duration
/// to expose nights where the velocity search is unusually expensive.
///
/// Arguments
/// ---------
/// * `rows` – Per-night measurements to plot.
/// * `path` – Destination PNG path.
///
/// Return
/// ------
/// `Ok(())` when the figure is written successfully.
fn plot_hough_votes(rows: &[HoughNightPerfRow], path: &Path) -> Result<()> {
    let n = rows.len();
    let y_max = rows
        .iter()
        .map(|r| r.votes_per_sec)
        .fold(0.0_f64, f64::max)
        .max(1.0)
        * 1.15;

    let root =
        BitMapBackend::new(path.to_str().unwrap_or_default(), (1000, 480)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Hough vote throughput per night", ("sans-serif", 20))
        .margin(30u32)
        .x_label_area_size(60u32)
        .y_label_area_size(80u32)
        .build_cartesian_2d(-0.5f64..(n as f64 - 0.5), 0f64..y_max)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    configure_night_mesh(&mut chart, rows, "votes / s")?;

    chart
        .draw_series(rows.iter().enumerate().map(|(i, r)| {
            let x = i as f64;
            Rectangle::new(
                [(x - 0.35, 0.0), (x + 0.35, r.votes_per_sec)],
                C_VOTES.filled(),
            )
        }))
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    root.present().map_err(|e| anyhow::anyhow!("{e:?}"))?;
    tracing::debug!("wrote {}", path.display());
    Ok(())
}

/// Plot the accumulator occupancy and retained peak counts per night.
///
/// This chart contrasts the sparse accumulator footprint with the number of
/// peaks that survive ranking and the photometric filter. It is useful for
/// spotting nights where a dense accumulator still yields few viable peaks.
///
/// Arguments
/// ---------
/// * `rows` – Per-night measurements to plot.
/// * `path` – Destination PNG path.
///
/// Return
/// ------
/// `Ok(())` when the figure is written successfully.
fn plot_hough_accumulator(rows: &[HoughNightPerfRow], path: &Path) -> Result<()> {
    let n = rows.len();
    let y_max = rows
        .iter()
        .map(|r| r.n_accumulator_bins.max(r.n_peaks_after_photometric_filter) as f64)
        .fold(0.0_f64, f64::max)
        .max(1.0)
        * 1.15;

    let root =
        BitMapBackend::new(path.to_str().unwrap_or_default(), (1000, 520)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Hough accumulator load per night", ("sans-serif", 20))
        .margin(30u32)
        .x_label_area_size(60u32)
        .y_label_area_size(90u32)
        .build_cartesian_2d(-0.5f64..(n as f64 - 0.5), 0f64..y_max)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    configure_night_mesh(&mut chart, rows, "count")?;

    chart
        .draw_series(rows.iter().enumerate().map(|(i, r)| {
            let x = i as f64;
            Rectangle::new(
                [(x - 0.40, 0.0), (x - 0.02, r.n_accumulator_bins as f64)],
                C_BINS.mix(0.7).filled(),
            )
        }))
        .map_err(|e| anyhow::anyhow!("{e:?}"))?
        .label("accumulator bins")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 20, y + 5)], C_BINS.mix(0.7).filled()));

    chart
        .draw_series(rows.iter().enumerate().map(|(i, r)| {
            let x = i as f64;
            Rectangle::new(
                [
                    (x + 0.02, 0.0),
                    (x + 0.40, r.n_peaks_after_photometric_filter as f64),
                ],
                C_PEAKS.mix(0.8).filled(),
            )
        }))
        .map_err(|e| anyhow::anyhow!("{e:?}"))?
        .label("peaks after photometric filter")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 20, y + 5)], C_PEAKS.mix(0.8).filled()));

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.85))
        .draw()
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    root.present().map_err(|e| anyhow::anyhow!("{e:?}"))?;
    tracing::debug!("wrote {}", path.display());
    Ok(())
}

/// Plot per-night purity and recall against the truth map.
///
/// Purity captures how selective the Hough seeds are, while recall captures
/// how many recoverable trajectories are covered at least once. The two curves
/// are the main quality summary for the Hough replay.
///
/// Arguments
/// ---------
/// * `rows` – Per-night measurements to plot.
/// * `path` – Destination PNG path.
///
/// Return
/// ------
/// `Ok(())` when the figure is written successfully.
fn plot_hough_quality(rows: &[HoughNightPerfRow], path: &Path) -> Result<()> {
    let n = rows.len();

    let root =
        BitMapBackend::new(path.to_str().unwrap_or_default(), (1000, 480)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Hough seeding quality per night", ("sans-serif", 20))
        .margin(30u32)
        .x_label_area_size(60u32)
        .y_label_area_size(80u32)
        .build_cartesian_2d(-0.5f64..(n as f64 - 0.5), 0f64..1.05f64)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    configure_night_mesh(&mut chart, rows, "ratio")?;

    let purity_pts: Vec<(f64, f64)> = rows
        .iter()
        .enumerate()
        .filter_map(|(i, r)| r.purity.is_finite().then_some((i as f64, r.purity)))
        .collect();

    let recall_pts: Vec<(f64, f64)> = rows
        .iter()
        .enumerate()
        .filter_map(|(i, r)| r.recall.is_finite().then_some((i as f64, r.recall)))
        .collect();

    chart
        .draw_series(LineSeries::new(
            purity_pts.clone(),
            ShapeStyle::from(&C_PURITY).stroke_width(2),
        ))
        .map_err(|e| anyhow::anyhow!("{e:?}"))?
        .label("purity")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 20, y)],
                ShapeStyle::from(&C_PURITY).stroke_width(2),
            )
        });

    chart
        .draw_series(
            purity_pts
                .iter()
                .map(|&(x, y)| Circle::new((x, y), 3, C_PURITY.filled())),
        )
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    chart
        .draw_series(LineSeries::new(
            recall_pts.clone(),
            ShapeStyle::from(&C_RECALL).stroke_width(2),
        ))
        .map_err(|e| anyhow::anyhow!("{e:?}"))?
        .label("recall")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 20, y)],
                ShapeStyle::from(&C_RECALL).stroke_width(2),
            )
        });

    chart
        .draw_series(
            recall_pts
                .iter()
                .map(|&(x, y)| Circle::new((x, y), 3, C_RECALL.filled())),
        )
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.85))
        .draw()
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    root.present().map_err(|e| anyhow::anyhow!("{e:?}"))?;
    tracing::debug!("wrote {}", path.display());
    Ok(())
}

/// Plotters chart type used by the nightly Hough diagnostic plots.
///
/// This alias keeps the mesh configuration helpers compact and avoids repeating
/// the full Plotters coordinate type in every helper signature.
type NightChart<'a, 'b> =
    ChartContext<'a, BitMapBackend<'b>, Cartesian2d<RangedCoordf64, RangedCoordf64>>;

/// Configure the shared x-axis labelling and y-axis descriptor for nightly plots.
///
/// The helper maps the x-axis indices back to the string night labels stored in
/// [`HoughNightPerfRow`]. It is shared by all figures so the x-axis formatting
/// stays identical across runtime, throughput, accumulator, and quality plots.
///
/// Arguments
/// ---------
/// * `chart` – Plotters chart being configured.
/// * `rows` – Nightly data rows used to derive the x-axis labels.
/// * `y_desc` – Label displayed on the y-axis.
///
/// Return
/// ------
/// `Ok(())` after the mesh has been configured and drawn.
fn configure_night_mesh(
    chart: &mut NightChart<'_, '_>,
    rows: &[HoughNightPerfRow],
    y_desc: &str,
) -> Result<()> {
    chart
        .configure_mesh()
        .x_labels(rows.len())
        .x_label_formatter(&|x: &f64| {
            let ix = x.round() as isize;
            if ix < 0 || (ix as usize) >= rows.len() {
                String::new()
            } else {
                rows[ix as usize].night_label.clone()
            }
        })
        .x_desc("night")
        .y_desc(y_desc)
        .light_line_style(WHITE.mix(0.15))
        .draw()
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    Ok(())
}
