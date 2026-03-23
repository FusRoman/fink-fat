//! Per-night seeding result plots: seed counts (TP/FP/unknown), purity,
//! recall, and trajectory recovery.
//!
//! Entry-point: [`plot_seed_results`].

use std::path::Path;

use anyhow::Result;
use camino::Utf8Path;
use plotters::prelude::*;

use crate::seeding::SeedStats;

// ─────────────────────────────────────────────────────────────────────────────
// Colour palette for seeding result plots
// ─────────────────────────────────────────────────────────────────────────────

const C_TP: RGBColor = RGBColor(34, 139, 34);
const C_FP: RGBColor = RGBColor(220, 20, 60);
const C_UNK: RGBColor = RGBColor(128, 128, 128);
const C_PURITY: RGBColor = RGBColor(70, 130, 180);
const C_RECALL: RGBColor = RGBColor(255, 140, 0);
const C_RECOV: RGBColor = RGBColor(34, 139, 34);
const C_TOTRL: RGBColor = RGBColor(51, 51, 255);

// ─────────────────────────────────────────────────────────────────────────────
// Data row
// ─────────────────────────────────────────────────────────────────────────────

/// One row of per-night seeding results used for charting.
pub struct NightResultRow {
    pub label: String,
    pub n_seeds: usize,
    pub n_tp: usize,
    pub n_fp: usize,
    pub n_unk: usize,
    pub purity: f64,
    pub recall: f64,
    pub n_recovered: usize,
    pub n_recoverable: usize,
}

impl NightResultRow {
    pub fn from_stats(label: impl Into<String>, stats: &SeedStats) -> Self {
        Self {
            label: label.into(),
            n_seeds: stats.n_seeds,
            n_tp: stats.n_true_positive,
            n_fp: stats.n_false_positive,
            n_unk: stats.n_unknown,
            purity: stats.purity(),
            recall: stats.recall(),
            n_recovered: stats.n_recovered_trajs,
            n_recoverable: stats.n_recoverable_trajs,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry-point
// ─────────────────────────────────────────────────────────────────────────────

/// Write all per-night seeding result charts to `out_dir`.
///
/// Files produced:
/// - `seed_counts.png`   – stacked bar chart: TP / FP / unknown per night
/// - `seed_quality.png`  – purity and recall line chart per night
/// - `seed_recovery.png` – n_recovered vs n_recoverable per night
pub fn plot_seed_results(rows: &[NightResultRow], out_dir: &Utf8Path) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    plot_seed_counts(rows, &out_dir.as_std_path().join("seed_counts.png"))?;
    plot_seed_quality(rows, &out_dir.as_std_path().join("seed_quality.png"))?;
    plot_seed_recovery(rows, &out_dir.as_std_path().join("seed_recovery.png"))?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Individual charts
// ─────────────────────────────────────────────────────────────────────────────

/// Grouped bar chart: TP (green) / FP (red) / unknown (grey) side by side per night.
pub fn plot_seed_counts(rows: &[NightResultRow], path: &Path) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let n = rows.len();
    let y_max = rows.iter().map(|r| r.n_seeds).max().unwrap_or(1) as f64 * 1.15;

    let path_str = path.to_str().unwrap_or_default();
    let root = BitMapBackend::new(path_str, (900, 500)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Seeding results per night", ("sans-serif", 20))
        .margin(30u32)
        .x_label_area_size(50u32)
        .y_label_area_size(60u32)
        .build_cartesian_2d(-0.5f64..(n as f64 - 0.5), 0f64..y_max)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    configure_night_mesh(&mut chart, rows)?;

    draw_grouped_tp(&mut chart, rows)?;
    draw_grouped_fp(&mut chart, rows)?;
    draw_grouped_unk(&mut chart, rows)?;

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.8))
        .draw()
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    root.present().map_err(|e| anyhow::anyhow!("{e:?}"))?;
    tracing::debug!("wrote {}", path.display());
    Ok(())
}

/// Dual line chart: purity (blue) and recall (orange) per night.
pub fn plot_seed_quality(rows: &[NightResultRow], path: &Path) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let n = rows.len();
    let path_str = path.to_str().unwrap_or_default();
    let root = BitMapBackend::new(path_str, (900, 450)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Seeding quality per night", ("sans-serif", 20))
        .margin(30u32)
        .x_label_area_size(50u32)
        .y_label_area_size(60u32)
        .build_cartesian_2d(-0.5f64..(n as f64 - 0.5), 0f64..1.05f64)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    configure_night_mesh(&mut chart, rows)?;

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
                .map(|&(x, y)| Circle::new((x, y), 4, C_PURITY.filled())),
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
                .map(|&(x, y)| Circle::new((x, y), 4, C_RECALL.filled())),
        )
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.8))
        .draw()
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    root.present().map_err(|e| anyhow::anyhow!("{e:?}"))?;
    tracing::debug!("wrote {}", path.display());
    Ok(())
}

/// Grouped bar chart: n_recoverable (grey) and n_recovered (green) per night.
pub fn plot_seed_recovery(rows: &[NightResultRow], path: &Path) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let n = rows.len();
    let y_max = rows.iter().map(|r| r.n_recoverable).max().unwrap_or(1) as f64 * 1.15;

    let path_str = path.to_str().unwrap_or_default();
    let root = BitMapBackend::new(path_str, (900, 450)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Trajectory recovery per night", ("sans-serif", 20))
        .margin(30u32)
        .x_label_area_size(50u32)
        .y_label_area_size(60u32)
        .build_cartesian_2d(-0.5f64..(n as f64 - 0.5), 0f64..y_max)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    configure_night_mesh(&mut chart, rows)?;

    // Left bar = n_recoverable (light grey), right bar = n_recovered (green).
    // Slot width = 0.8; 2 bars × 0.38 with 0.04 gap in the middle.
    chart
        .draw_series(rows.iter().enumerate().map(|(i, r)| {
            let x = i as f64;
            Rectangle::new(
                [(x - 0.40, 0.0), (x - 0.02, r.n_recoverable as f64)],
                C_TOTRL.mix(0.6).filled(),
            )
        }))
        .map_err(|e| anyhow::anyhow!("{e:?}"))?
        .label("recoverable")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 20, y + 5)], C_TOTRL.mix(0.6).filled()));

    chart
        .draw_series(rows.iter().enumerate().map(|(i, r)| {
            let x = i as f64;
            Rectangle::new(
                [(x + 0.02, 0.0), (x + 0.40, r.n_recovered as f64)],
                C_RECOV.mix(0.8).filled(),
            )
        }))
        .map_err(|e| anyhow::anyhow!("{e:?}"))?
        .label("recovered")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 20, y + 5)], C_RECOV.mix(0.8).filled()));

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.8))
        .draw()
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    root.present().map_err(|e| anyhow::anyhow!("{e:?}"))?;
    tracing::debug!("wrote {}", path.display());
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Stacked bar helpers
// ─────────────────────────────────────────────────────────────────────────────

type NightChart<'a, 'b> = ChartContext<
    'a,
    BitMapBackend<'b>,
    Cartesian2d<plotters::coord::types::RangedCoordf64, plotters::coord::types::RangedCoordf64>,
>;

fn configure_night_mesh(chart: &mut NightChart<'_, '_>, rows: &[NightResultRow]) -> Result<()> {
    let n = rows.len();
    let label_step = (n / 30).max(1); // show at most ~30 labels
    chart
        .configure_mesh()
        .x_desc("night index")
        .y_desc("count")
        .x_label_formatter(&|x| {
            let i = x.round() as usize;
            if i < rows.len() && i.is_multiple_of(label_step) {
                rows[i].label.clone()
            } else {
                String::new()
            }
        })
        .draw()
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    Ok(())
}

// Grouped bar layout for 3 series per night slot (width 0.8):
//   TP   : [i-0.40, i-0.16]  (bar_w=0.24, gap=0.04)
//   FP   : [i-0.12, i+0.12]
//   unk  : [i+0.16, i+0.40]

fn draw_grouped_tp(chart: &mut NightChart<'_, '_>, rows: &[NightResultRow]) -> Result<()> {
    chart
        .draw_series(rows.iter().enumerate().map(|(i, r)| {
            let x = i as f64;
            Rectangle::new(
                [(x - 0.40, 0.0), (x - 0.16, r.n_tp as f64)],
                C_TP.mix(0.8).filled(),
            )
        }))
        .map_err(|e| anyhow::anyhow!("{e:?}"))?
        .label("TP")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 20, y + 5)], C_TP.mix(0.8).filled()));
    Ok(())
}

fn draw_grouped_fp(chart: &mut NightChart<'_, '_>, rows: &[NightResultRow]) -> Result<()> {
    chart
        .draw_series(rows.iter().enumerate().map(|(i, r)| {
            let x = i as f64;
            Rectangle::new(
                [(x - 0.12, 0.0), (x + 0.12, r.n_fp as f64)],
                C_FP.mix(0.8).filled(),
            )
        }))
        .map_err(|e| anyhow::anyhow!("{e:?}"))?
        .label("FP")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 20, y + 5)], C_FP.mix(0.8).filled()));
    Ok(())
}

fn draw_grouped_unk(chart: &mut NightChart<'_, '_>, rows: &[NightResultRow]) -> Result<()> {
    chart
        .draw_series(rows.iter().enumerate().map(|(i, r)| {
            let x = i as f64;
            Rectangle::new(
                [(x + 0.16, 0.0), (x + 0.40, r.n_unk as f64)],
                C_UNK.mix(0.8).filled(),
            )
        }))
        .map_err(|e| anyhow::anyhow!("{e:?}"))?
        .label("unknown")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 20, y + 5)], C_UNK.mix(0.8).filled()));
    Ok(())
}
