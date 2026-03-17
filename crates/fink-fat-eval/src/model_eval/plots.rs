//! Evaluation plots for the ONNX model: ROC curve, PR curve, score distribution.
//!
//! All plots are written as PNG files into a user-specified directory.

use anyhow::{Context, Result};
use camino::Utf8Path;
use plotters::prelude::*;

use super::ModelMetrics;

const PLOT_W: u32 = 800;
const PLOT_H: u32 = 600;
const MARGIN: u32 = 40;

const C_LINE: RGBColor = RGBColor(31, 119, 180);
const C_DIAG: RGBColor = RGBColor(160, 160, 160);
const C_TP: RGBColor = RGBColor(34, 139, 34);
const C_FP: RGBColor = RGBColor(220, 20, 60);

/// Write all evaluation plots to `out_dir`.
///
/// Plots produced:
/// - `roc_curve.png`          – ROC curve with AUC annotation.
/// - `pr_curve.png`           – Precision-Recall curve with AP annotation.
/// - `score_distribution.png` – Score histograms for TP and FP edges.
pub fn write_eval_plots(metrics: &ModelMetrics, out_dir: &Utf8Path) -> Result<()> {
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("cannot create plot dir: {out_dir}"))?;

    plot_roc_curve(metrics, &out_dir.join("roc_curve.png"))?;
    plot_pr_curve(metrics, &out_dir.join("pr_curve.png"))?;
    plot_score_distribution(metrics, &out_dir.join("score_distribution.png"))?;

    tracing::info!("model eval plots written to {out_dir}");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// ROC curve
// ─────────────────────────────────────────────────────────────────────────────

fn plot_roc_curve(metrics: &ModelMetrics, path: &camino::Utf8Path) -> Result<()> {
    let path_str = path.as_str();
    let root = BitMapBackend::new(path_str, (PLOT_W, PLOT_H)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| anyhow::anyhow!("fill: {e:?}"))?;

    let title = format!("ROC curve  (AUC = {:.4})", metrics.roc_auc);
    let mut chart = ChartBuilder::on(&root)
        .caption(&title, ("sans-serif", 22))
        .margin(MARGIN)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0f64..1f64, 0f64..1f64)
        .map_err(|e| anyhow::anyhow!("chart: {e:?}"))?;

    chart
        .configure_mesh()
        .x_desc("False Positive Rate")
        .y_desc("True Positive Rate")
        .draw()
        .map_err(|e| anyhow::anyhow!("mesh: {e:?}"))?;

    // Random-classifier diagonal.
    chart
        .draw_series(LineSeries::new(
            [(0.0, 0.0), (1.0, 1.0)],
            C_DIAG.stroke_width(1),
        ))
        .map_err(|e| anyhow::anyhow!("diag: {e:?}"))?;

    // ROC curve.
    chart
        .draw_series(LineSeries::new(
            metrics.roc_curve.iter().map(|&(fpr, tpr)| (fpr, tpr)),
            C_LINE.stroke_width(2),
        ))
        .map_err(|e| anyhow::anyhow!("roc line: {e:?}"))?
        .label(format!("ONNX model (AUC = {:.4})", metrics.roc_auc))
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], C_LINE.stroke_width(2)));

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.8))
        .draw()
        .map_err(|e| anyhow::anyhow!("legend: {e:?}"))?;

    root.present()
        .map_err(|e| anyhow::anyhow!("write PNG: {e:?}"))?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// PR curve
// ─────────────────────────────────────────────────────────────────────────────

fn plot_pr_curve(metrics: &ModelMetrics, path: &camino::Utf8Path) -> Result<()> {
    let baseline = metrics.n_positive as f64 / metrics.n_samples.max(1) as f64;

    let path_str = path.as_str();
    let root = BitMapBackend::new(path_str, (PLOT_W, PLOT_H)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| anyhow::anyhow!("fill: {e:?}"))?;

    let title = format!("Precision-Recall curve  (AP = {:.4})", metrics.pr_auc);
    let mut chart = ChartBuilder::on(&root)
        .caption(&title, ("sans-serif", 22))
        .margin(MARGIN)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0f64..1f64, 0f64..1f64)
        .map_err(|e| anyhow::anyhow!("chart: {e:?}"))?;

    chart
        .configure_mesh()
        .x_desc("Recall")
        .y_desc("Precision")
        .draw()
        .map_err(|e| anyhow::anyhow!("mesh: {e:?}"))?;

    // Baseline (random classifier precision).
    chart
        .draw_series(LineSeries::new(
            [(0.0, baseline), (1.0, baseline)],
            C_DIAG.stroke_width(1),
        ))
        .map_err(|e| anyhow::anyhow!("baseline: {e:?}"))?
        .label(format!("baseline ({baseline:.4})"))
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], C_DIAG.stroke_width(1)));

    // PR curve.
    chart
        .draw_series(LineSeries::new(
            metrics
                .pr_curve
                .iter()
                .map(|&(recall, precision)| (recall, precision)),
            C_LINE.stroke_width(2),
        ))
        .map_err(|e| anyhow::anyhow!("pr line: {e:?}"))?
        .label(format!("ONNX model (AP = {:.4})", metrics.pr_auc))
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], C_LINE.stroke_width(2)));

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.8))
        .draw()
        .map_err(|e| anyhow::anyhow!("legend: {e:?}"))?;

    root.present()
        .map_err(|e| anyhow::anyhow!("write PNG: {e:?}"))?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Score distribution  TP vs FP
// ─────────────────────────────────────────────────────────────────────────────

fn plot_score_distribution(metrics: &ModelMetrics, path: &camino::Utf8Path) -> Result<()> {
    const N_BINS: usize = 60;

    let mut scores_tp: Vec<f32> = Vec::new();
    let mut scores_fp: Vec<f32> = Vec::new();
    for (&score, &label) in metrics.scores.iter().zip(metrics.labels.iter()) {
        if label {
            scores_tp.push(score);
        } else {
            scores_fp.push(score);
        }
    }

    let path_str = path.as_str();

    let edges: Vec<f32> = (0..=N_BINS).map(|i| i as f32 / N_BINS as f32).collect();

    let counts_tp = bin_f32(&edges, &scores_tp);
    let counts_fp = bin_f32(&edges, &scores_fp);
    let max_h = counts_tp
        .iter()
        .chain(counts_fp.iter())
        .copied()
        .max()
        .unwrap_or(1);
    let y_max = (max_h as f64 + 1.0).ln() * 1.1;

    let root = BitMapBackend::new(path_str, (PLOT_W, PLOT_H)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| anyhow::anyhow!("fill: {e:?}"))?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Score distribution  (TP vs FP)", ("sans-serif", 22))
        .margin(MARGIN)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0f32..1f32, 0f64..y_max)
        .map_err(|e| anyhow::anyhow!("chart: {e:?}"))?;

    chart
        .configure_mesh()
        .x_desc("p(true_edge)")
        .y_desc("count (ln(n+1))")
        .draw()
        .map_err(|e| anyhow::anyhow!("mesh: {e:?}"))?;

    // FP bars (underneath).
    chart
        .draw_series((0..N_BINS).filter_map(|i| {
            if counts_fp[i] > 0 {
                let h = (counts_fp[i] as f64 + 1.0).ln();
                Some(Rectangle::new(
                    [(edges[i], 0f64), (edges[i + 1], h)],
                    C_FP.mix(0.55).filled(),
                ))
            } else {
                None
            }
        }))
        .map_err(|e| anyhow::anyhow!("fp bars: {e:?}"))?
        .label("FP")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 15, y + 5)], C_FP.mix(0.55).filled()));

    // TP bars (on top).
    chart
        .draw_series((0..N_BINS).filter_map(|i| {
            if counts_tp[i] > 0 {
                let h = (counts_tp[i] as f64 + 1.0).ln();
                Some(Rectangle::new(
                    [(edges[i], 0f64), (edges[i + 1], h)],
                    C_TP.mix(0.65).filled(),
                ))
            } else {
                None
            }
        }))
        .map_err(|e| anyhow::anyhow!("tp bars: {e:?}"))?
        .label("TP")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 15, y + 5)], C_TP.mix(0.65).filled()));

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.8))
        .draw()
        .map_err(|e| anyhow::anyhow!("legend: {e:?}"))?;

    root.present()
        .map_err(|e| anyhow::anyhow!("write PNG: {e:?}"))?;
    Ok(())
}

fn bin_f32(edges: &[f32], values: &[f32]) -> Vec<usize> {
    let n = edges.len().saturating_sub(1);
    let mut counts = vec![0usize; n];
    for &v in values {
        if v.is_nan() {
            continue;
        }
        // Binary search for the right bin.
        let idx = edges.partition_point(|&e| e <= v).saturating_sub(1);
        if idx < n {
            counts[idx] += 1;
        }
    }
    counts
}
