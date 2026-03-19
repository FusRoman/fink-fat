//! Evaluation plots for the ONNX model: ROC curve, PR curve, score distribution,
//! confusion matrix, and per-seed ranking curves.
//!
//! All plots are written as PNG files into a user-specified directory.

use anyhow::{Context, Result};
use camino::Utf8Path;
use plotters::prelude::*;

use super::{ModelMetrics, RankingMetrics};

const PLOT_W: u32 = 800;
const PLOT_H: u32 = 600;
const MARGIN: u32 = 40;

const C_LINE: RGBColor = RGBColor(31, 119, 180); // matplotlib blue
const C_DIAG: RGBColor = RGBColor(160, 160, 160);
const C_TP: RGBColor = RGBColor(34, 139, 34);
const C_FP: RGBColor = RGBColor(220, 20, 60);
const C_PREC: RGBColor = RGBColor(255, 127, 14); // matplotlib orange
const C_HITR: RGBColor = RGBColor(44, 160, 44); // matplotlib green

/// Write all evaluation plots to `out_dir`.
///
/// Plots produced:
/// - `roc_curve.png`          – ROC curve with AUC annotation.
/// - `pr_curve.png`           – Precision-Recall curve with AP annotation.
/// - `score_distribution.png` – Score histograms for TP and FP edges.
/// - `confusion_matrix.png`   – 2×2 confusion matrix at threshold 0.5.
pub fn write_eval_plots(metrics: &ModelMetrics, out_dir: &Utf8Path) -> Result<()> {
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("cannot create plot dir: {out_dir}"))?;

    plot_roc_curve(metrics, &out_dir.join("roc_curve.png"))?;
    plot_pr_curve(metrics, &out_dir.join("pr_curve.png"))?;
    plot_score_distribution(metrics, &out_dir.join("score_distribution.png"))?;
    plot_confusion_matrix(metrics, &out_dir.join("confusion_matrix.png"))?;

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
    let bin_width = 1.0 / N_BINS as f64;

    let counts_tp = bin_f32(&edges, &scores_tp);
    let counts_fp = bin_f32(&edges, &scores_fp);

    // Normalise each class independently to density (area under histogram = 1).
    let total_tp = scores_tp.len().max(1) as f64;
    let total_fp = scores_fp.len().max(1) as f64;
    let density_tp: Vec<f64> = counts_tp
        .iter()
        .map(|&c| c as f64 / (total_tp * bin_width))
        .collect();
    let density_fp: Vec<f64> = counts_fp
        .iter()
        .map(|&c| c as f64 / (total_fp * bin_width))
        .collect();

    let y_max = density_tp
        .iter()
        .chain(density_fp.iter())
        .copied()
        .fold(0f64, f64::max)
        * 1.05;

    let root = BitMapBackend::new(path_str, (PLOT_W, PLOT_H)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| anyhow::anyhow!("fill: {e:?}"))?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Score distribution \u{2014} TP vs FP", ("sans-serif", 22))
        .margin(MARGIN)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0f32..1f32, 0f64..y_max)
        .map_err(|e| anyhow::anyhow!("chart: {e:?}"))?;

    chart
        .configure_mesh()
        .x_desc("Predicted probability P(TP)")
        .y_desc("Density")
        .draw()
        .map_err(|e| anyhow::anyhow!("mesh: {e:?}"))?;

    // FP bars.
    chart
        .draw_series((0..N_BINS).filter_map(|i| {
            if density_fp[i] > 0.0 {
                Some(Rectangle::new(
                    [(edges[i], 0f64), (edges[i + 1], density_fp[i])],
                    C_FP.mix(0.55).filled(),
                ))
            } else {
                None
            }
        }))
        .map_err(|e| anyhow::anyhow!("fp bars: {e:?}"))?
        .label("FP (0)")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 15, y + 5)], C_FP.mix(0.55).filled()));

    // TP bars.
    chart
        .draw_series((0..N_BINS).filter_map(|i| {
            if density_tp[i] > 0.0 {
                Some(Rectangle::new(
                    [(edges[i], 0f64), (edges[i + 1], density_tp[i])],
                    C_TP.mix(0.65).filled(),
                ))
            } else {
                None
            }
        }))
        .map_err(|e| anyhow::anyhow!("tp bars: {e:?}"))?
        .label("TP (1)")
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

// ─────────────────────────────────────────────────────────────────────────────
// Confusion matrix
// ─────────────────────────────────────────────────────────────────────────────

fn plot_confusion_matrix(metrics: &ModelMetrics, path: &camino::Utf8Path) -> Result<()> {
    const THRESHOLD: f32 = 0.5;

    // ── counts ───────────────────────────────────────────────────────────────
    let (mut tp, mut fp, mut tn, mut fn_) = (0usize, 0usize, 0usize, 0usize);
    for (&score, &label) in metrics.scores.iter().zip(metrics.labels.iter()) {
        match (label, score >= THRESHOLD) {
            (true, true) => tp += 1,
            (false, true) => fp += 1,
            (false, false) => tn += 1,
            (true, false) => fn_ += 1,
        }
    }

    let total = (tp + fp + tn + fn_).max(1) as f64;
    let precision = tp as f64 / (tp + fp).max(1) as f64;
    let recall = tp as f64 / (tp + fn_).max(1) as f64;
    let f1 = 2.0 * precision * recall / (precision + recall + 1e-12);
    let max_val = [tp, fp, tn, fn_].iter().copied().max().unwrap_or(1) as f64;

    // White → steel-blue heatmap based on relative count.
    let cell_color = |v: usize| -> RGBAColor {
        let t = v as f32 / max_val as f32;
        RGBAColor(
            (255.0 * (1.0 - t * (1.0 - 0.122))) as u8,
            (255.0 * (1.0 - t * (1.0 - 0.467))) as u8,
            (255.0 * (1.0 - t * (1.0 - 0.706))) as u8,
            1.0,
        )
    };
    // White text on dark cells, black text on light cells.
    let text_color = |v: usize| -> RGBColor {
        if v as f64 / max_val > 0.55 {
            WHITE
        } else {
            BLACK
        }
    };

    let path_str = path.as_str();
    // Square canvas.
    let root = BitMapBackend::new(path_str, (PLOT_W, PLOT_W)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| anyhow::anyhow!("fill: {e:?}"))?;

    let title = format!(
        "Confusion Matrix (thr=0.5)   Precision={precision:.3}  Recall={recall:.3}  F1={f1:.3}"
    );

    // Data space: x ∈ [-0.6, 2.1], y ∈ [-0.28, 2.15]
    // Matrix cells occupy [0, 2] × [0, 2] (y grows upward):
    //   TN: x=[0,1] y=[1,2]   FP: x=[1,2] y=[1,2]   (Actual 0, top)
    //   FN: x=[0,1] y=[0,1]   TP: x=[1,2] y=[0,1]   (Actual 1, bottom)
    let mut chart = ChartBuilder::on(&root)
        .caption(&title, ("sans-serif", 16))
        .margin(MARGIN)
        .x_label_area_size(0)
        .y_label_area_size(0)
        .build_cartesian_2d(-0.6f64..2.1f64, -0.28f64..2.15f64)
        .map_err(|e| anyhow::anyhow!("chart: {e:?}"))?;

    chart
        .configure_mesh()
        .disable_axes()
        .disable_mesh()
        .draw()
        .map_err(|e| anyhow::anyhow!("mesh: {e:?}"))?;

    // ── cells ────────────────────────────────────────────────────────────────
    let cells: &[(f64, f64, f64, f64, usize, &str)] = &[
        (0.0, 1.0, 1.0, 2.0, tn, "TN"),
        (1.0, 2.0, 1.0, 2.0, fp, "FP"),
        (0.0, 1.0, 0.0, 1.0, fn_, "FN"),
        (1.0, 2.0, 0.0, 1.0, tp, "TP"),
    ];

    for &(x0, x1, y0, y1, count, abbr) in cells {
        // Filled background.
        chart
            .draw_series(std::iter::once(Rectangle::new(
                [(x0, y0), (x1, y1)],
                cell_color(count).filled(),
            )))
            .map_err(|e| anyhow::anyhow!("rect {abbr}: {e:?}"))?;
        // Border.
        chart
            .draw_series(std::iter::once(Rectangle::new(
                [(x0, y0), (x1, y1)],
                BLACK.stroke_width(1),
            )))
            .map_err(|e| anyhow::anyhow!("border {abbr}: {e:?}"))?;

        let tc = text_color(count);
        let cx = (x0 + x1) / 2.0;
        let cy = (y0 + y1) / 2.0;
        let pct = count as f64 / total * 100.0;

        // Abbreviation near top of cell.
        chart
            .draw_series(std::iter::once(Text::new(
                abbr.to_string(),
                (cx - 0.06, y1 - 0.20),
                ("sans-serif", 20).into_font().color(&tc),
            )))
            .map_err(|e| anyhow::anyhow!("abbr {abbr}: {e:?}"))?;

        // Count in the centre.
        chart
            .draw_series(std::iter::once(Text::new(
                format!("{count}"),
                (cx - 0.14, cy),
                ("sans-serif", 28).into_font().color(&tc),
            )))
            .map_err(|e| anyhow::anyhow!("count {abbr}: {e:?}"))?;

        // Percentage near bottom of cell.
        chart
            .draw_series(std::iter::once(Text::new(
                format!("{pct:.1}%"),
                (cx - 0.12, y0 + 0.10),
                ("sans-serif", 16).into_font().color(&tc),
            )))
            .map_err(|e| anyhow::anyhow!("pct {abbr}: {e:?}"))?;
    }

    // ── column headers (Predicted) ────────────────────────────────────────────
    chart
        .draw_series(std::iter::once(Text::new(
            "Predicted".to_string(),
            (0.65, -0.20),
            ("sans-serif", 16).into_font().color(&BLACK),
        )))
        .map_err(|e| anyhow::anyhow!("x header: {e:?}"))?;
    for (cx, label) in [(0.5f64, "0  (neg)"), (1.5, "1  (pos)")] {
        chart
            .draw_series(std::iter::once(Text::new(
                label.to_string(),
                (cx - 0.14, -0.10),
                ("sans-serif", 14).into_font().color(&BLACK),
            )))
            .map_err(|e| anyhow::anyhow!("col label: {e:?}"))?;
    }

    // ── row headers (Actual) ──────────────────────────────────────────────────
    // y increases upward: Actual 0 (neg) → top row y∈[1,2], Actual 1 (pos) → bottom y∈[0,1].
    chart
        .draw_series(std::iter::once(Text::new(
            "Actual".to_string(),
            (-0.55, 1.05),
            ("sans-serif", 16).into_font().color(&BLACK),
        )))
        .map_err(|e| anyhow::anyhow!("y header: {e:?}"))?;
    for (cy, label) in [(1.5f64, "0  (neg)"), (0.5, "1  (pos)")] {
        chart
            .draw_series(std::iter::once(Text::new(
                label.to_string(),
                (-0.55, cy - 0.04),
                ("sans-serif", 14).into_font().color(&BLACK),
            )))
            .map_err(|e| anyhow::anyhow!("row label: {e:?}"))?;
    }

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

// ─────────────────────────────────────────────────────────────────────────────
// Ranking evaluation plots (top-k per left seed)
// ─────────────────────────────────────────────────────────────────────────────

/// Write ranking evaluation plots to `out_dir`.
///
/// Plots produced:
/// - `ranking_curves.png`   – Recall@k, Precision@k and Hit-Rate@k vs k.
/// - `first_tp_rank.png`    – Histogram of the rank of the first TP per left seed.
pub fn write_ranking_plots(ranking: &RankingMetrics, out_dir: &Utf8Path) -> Result<()> {
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("cannot create plot dir: {out_dir}"))?;

    plot_ranking_curves(ranking, &out_dir.join("ranking_curves.png"))?;
    plot_first_tp_rank(ranking, &out_dir.join("first_tp_rank.png"))?;

    tracing::info!("ranking plots written to {out_dir}");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Recall@k / Precision@k / Hit-Rate@k
// ─────────────────────────────────────────────────────────────────────────────

fn plot_ranking_curves(ranking: &RankingMetrics, path: &camino::Utf8Path) -> Result<()> {
    let k_max = ranking.k_max;
    let path_str = path.as_str();

    let root = BitMapBackend::new(path_str, (PLOT_W, PLOT_H)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| anyhow::anyhow!("fill: {e:?}"))?;

    let title = format!(
        "Top-k ranking (per left seed) — {} seeds, {} with TP",
        ranking.n_seeds, ranking.n_seeds_with_tp
    );
    let mut chart = ChartBuilder::on(&root)
        .caption(&title, ("sans-serif", 20))
        .margin(MARGIN)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(1usize..k_max, 0f64..1.05f64)
        .map_err(|e| anyhow::anyhow!("chart: {e:?}"))?;

    chart
        .configure_mesh()
        .x_desc("k  (top-k edges per left seed)")
        .y_desc("Score")
        .x_labels(10)
        .draw()
        .map_err(|e| anyhow::anyhow!("mesh: {e:?}"))?;

    // Recall@k
    chart
        .draw_series(LineSeries::new(
            (1..=k_max).map(|k| (k, ranking.recall_at_k[k - 1])),
            C_LINE.stroke_width(2),
        ))
        .map_err(|e| anyhow::anyhow!("recall: {e:?}"))?
        .label(format!("Recall@k  ({:.3} @ k=10)", ranking.recall_at_k[9]))
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], C_LINE.stroke_width(2)));

    // Precision@k
    chart
        .draw_series(LineSeries::new(
            (1..=k_max).map(|k| (k, ranking.precision_at_k[k - 1])),
            C_PREC.stroke_width(2),
        ))
        .map_err(|e| anyhow::anyhow!("precision: {e:?}"))?
        .label(format!(
            "Precision@k  ({:.3} @ k=10)",
            ranking.precision_at_k[9]
        ))
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], C_PREC.stroke_width(2)));

    // Hit-Rate@k
    chart
        .draw_series(LineSeries::new(
            (1..=k_max).map(|k| (k, ranking.hit_rate_at_k[k - 1])),
            C_HITR.stroke_width(2),
        ))
        .map_err(|e| anyhow::anyhow!("hitrate: {e:?}"))?
        .label(format!(
            "Hit-Rate@k  ({:.3} @ k=5)",
            ranking.hit_rate_at_k[4]
        ))
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], C_HITR.stroke_width(2)));

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
// Distribution of the rank of the first TP
// ─────────────────────────────────────────────────────────────────────────────

fn plot_first_tp_rank(ranking: &RankingMetrics, path: &camino::Utf8Path) -> Result<()> {
    let k_max = ranking.k_max;

    // Count occurrences of each rank (1..=k_max).  Rank = None means “not found”.
    let mut counts = vec![0usize; k_max]; // index i → rank i+1
    let mut not_found = 0usize;
    for &r in ranking
        .first_tp_rank
        .iter()
        .filter(|r| r.is_some() || ranking.n_seeds_with_tp > 0)
    {
        match r {
            Some(rank) if rank >= 1 && rank <= k_max => counts[rank - 1] += 1,
            Some(_) => not_found += 1,
            None => {}
        }
    }
    // Seeds with TP that were not found in top-k_max (rank > k_max or truly absent)
    // are already captured by not_found.

    let max_count = counts.iter().copied().max().unwrap_or(1).max(1);
    let y_max = max_count as f64 * 1.15;

    let path_str = path.as_str();
    let root = BitMapBackend::new(path_str, (PLOT_W, PLOT_H)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| anyhow::anyhow!("fill: {e:?}"))?;

    let title = format!(
        "Rank of first TP per left seed  ({} seeds with TP, {} not found in top-{})",
        ranking.n_seeds_with_tp, not_found, k_max
    );
    let mut chart = ChartBuilder::on(&root)
        .caption(&title, ("sans-serif", 18))
        .margin(MARGIN)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d((1usize..(k_max + 1)).into_segmented(), 0f64..y_max)
        .map_err(|e| anyhow::anyhow!("chart: {e:?}"))?;

    chart
        .configure_mesh()
        .x_desc("Rank of first TP (1 = best)")
        .y_desc("Number of left seeds")
        .x_labels(15)
        .draw()
        .map_err(|e| anyhow::anyhow!("mesh: {e:?}"))?;

    chart
        .draw_series(
            Histogram::vertical(&chart)
                .style(C_TP.mix(0.7).filled())
                .margin(1)
                .data((1..=k_max).map(|k| (k, counts[k - 1] as f64))),
        )
        .map_err(|e| anyhow::anyhow!("bars: {e:?}"))?
        .label("seeds-with-TP")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 15, y + 5)], C_TP.mix(0.7).filled()));

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
