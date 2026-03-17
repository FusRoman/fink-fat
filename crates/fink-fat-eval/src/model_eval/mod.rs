//! ONNX model evaluation against a labelled edge-feature Parquet file.
//!
//! This module loads:
//! - a Parquet file of edge features (exported by `edge-eval --export-features`),
//! - an ONNX model whose path is read from the `xgb_params.yml` config file,
//!
//! runs inference, computes ROC-AUC / PR-AUC, logs the metrics and writes
//! diagnostic plots to `plot_dir` via [`plots`].

pub mod plots;

use anyhow::{Context, Result, bail};
use camino::{Utf8Path, Utf8PathBuf};
use fink_fat_engine::graph::edge::edge_features::EDGE_FEATURE_KEYS;
use fink_fat_engine::graph::edge::edge_prediction::EdgeRankingModelPool;
use ndarray::Array2;
use polars::prelude::*;
use serde::Deserialize;

// ─────────────────────────────────────────────────────────────────────────────
// Config parsing
// ─────────────────────────────────────────────────────────────────────────────

/// Minimal subset of `xgb_params.yml` needed to locate the ONNX model.
#[derive(Debug, Deserialize)]
struct XgbTrainingCfg {
    onnx_output: String,
}

#[derive(Debug, Deserialize)]
struct XgbParams {
    training: XgbTrainingCfg,
}

/// Read `xgb_params.yml` and return the absolute path to the ONNX file.
///
/// The `onnx_output` value in the YAML is interpreted relative to the
/// directory that contains the YAML file itself.
pub fn resolve_onnx_path(xgb_params_path: &Utf8Path) -> Result<Utf8PathBuf> {
    let content = std::fs::read_to_string(xgb_params_path)
        .with_context(|| format!("cannot read {xgb_params_path}"))?;
    let cfg: XgbParams = serde_yaml::from_str(&content)
        .with_context(|| format!("cannot parse {xgb_params_path} as YAML"))?;

    let base = xgb_params_path
        .parent()
        .unwrap_or_else(|| Utf8Path::new("."));
    let onnx_path = base.join(&cfg.training.onnx_output);
    Ok(onnx_path)
}

// ─────────────────────────────────────────────────────────────────────────────
// Parquet loading
// ─────────────────────────────────────────────────────────────────────────────

/// Loaded data from the edge-feature Parquet file.
pub struct FeatureData {
    /// Dense feature matrix, shape `[N, 17]`, in canonical EDGE_FEATURE_KEYS order.
    pub features: Array2<f32>,
    /// Ground-truth labels: `true` = true positive edge.
    pub labels: Vec<bool>,
}

/// Load the feature matrix and truth labels from a Parquet file.
///
/// Expected columns:
/// - 17 feature columns named after `EDGE_FEATURE_KEYS` paths,
/// - `is_true_edge` (i8 or boolean): 1 = true positive, 0 otherwise.
pub fn load_features(path: &Utf8Path) -> Result<FeatureData> {
    let feature_names: Vec<&'static str> = EDGE_FEATURE_KEYS.iter().map(|k| k.path()).collect();

    // Select only the columns we need.
    let mut select_cols: Vec<Expr> = feature_names.iter().map(|n| col(*n)).collect();
    select_cols.push(col("is_true_edge"));

    let df = LazyFrame::scan_parquet(path.as_str(), ScanArgsParquet::default())
        .with_context(|| format!("failed to open parquet: {path}"))?
        .select(select_cols)
        .collect()
        .with_context(|| format!("failed to collect columns from {path}"))?;

    let n = df.height();
    if n == 0 {
        bail!("parquet file is empty: {path}");
    }

    let n_feat = feature_names.len(); // 17
    let mut flat: Vec<f32> = Vec::with_capacity(n * n_feat);

    for name in &feature_names {
        let col_data = df
            .column(name)
            .with_context(|| format!("missing column '{name}' in {path}"))?
            .f64()
            .with_context(|| format!("column '{name}' is not f64"))?;

        for val in col_data.iter() {
            flat.push(val.unwrap_or(f64::NAN) as f32);
        }
    }

    // `flat` is column-major (all values of col0, then col1 …).
    // We want row-major `[N, 17]`: transpose by reading i*n+j → j*n+i.
    let mut features = Array2::<f32>::zeros((n, n_feat));
    for feat_idx in 0..n_feat {
        for row_idx in 0..n {
            features[(row_idx, feat_idx)] = flat[feat_idx * n + row_idx];
        }
    }

    // Load truth labels.
    let label_col = df
        .column("is_true_edge")
        .context("missing column 'is_true_edge'")?;

    let labels: Vec<bool> = match label_col.dtype() {
        DataType::Int8 => label_col
            .i8()
            .context("'is_true_edge' is not i8")?
            .iter()
            .map(|v| v.unwrap_or(0) != 0)
            .collect(),
        DataType::Boolean => label_col
            .bool()
            .context("'is_true_edge' is not bool")?
            .iter()
            .map(|v| v.unwrap_or(false))
            .collect(),
        other => bail!("'is_true_edge' has unexpected dtype {other}"),
    };

    tracing::info!(
        n_samples = n,
        n_positive = labels.iter().filter(|&&b| b).count(),
        n_negative = labels.iter().filter(|&&b| !b).count(),
        "features loaded",
    );

    Ok(FeatureData { features, labels })
}

// ─────────────────────────────────────────────────────────────────────────────
// Metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Classification metrics computed on the test set.
#[derive(Debug, Clone)]
pub struct ModelMetrics {
    pub roc_auc: f64,
    pub pr_auc: f64,
    pub n_samples: usize,
    pub n_positive: usize,
    /// ROC curve as `(fpr, tpr)` pairs, sorted by ascending FPR.
    pub roc_curve: Vec<(f64, f64)>,
    /// PR curve as `(recall, precision)` pairs, sorted by ascending recall.
    pub pr_curve: Vec<(f64, f64)>,
    /// Raw scores per sample, together with labels (for distribution plot).
    pub scores: Vec<f32>,
    pub labels: Vec<bool>,
}

/// Compute ROC-AUC, PR-AUC and the corresponding curves.
pub fn compute_metrics(scores: Vec<f32>, labels: Vec<bool>) -> ModelMetrics {
    let n = scores.len();
    let n_positive = labels.iter().filter(|&&b| b).count();
    let n_negative = n - n_positive;

    // Sort by descending score.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));

    // ── ROC curve ───────────────────────────────────────────────────────────
    let mut roc_curve: Vec<(f64, f64)> = Vec::with_capacity(n + 2);
    roc_curve.push((0.0, 0.0));

    let mut tp = 0usize;
    let mut fp = 0usize;
    let fp_denom = n_negative.max(1) as f64;
    let tp_denom = n_positive.max(1) as f64;

    for &idx in &order {
        if labels[idx] {
            tp += 1;
        } else {
            fp += 1;
        }
        roc_curve.push((fp as f64 / fp_denom, tp as f64 / tp_denom));
    }
    roc_curve.push((1.0, 1.0));

    let roc_auc = trapezoidal_auc(&roc_curve);

    // ── PR curve ────────────────────────────────────────────────────────────
    // Walk from high score to low, accumulating TP/FP.
    let mut pr_points: Vec<(f64, f64)> = Vec::with_capacity(n + 1);
    let mut tp = 0usize;
    let mut fp = 0usize;

    for &idx in &order {
        if labels[idx] {
            tp += 1;
        } else {
            fp += 1;
        }
        let precision = tp as f64 / (tp + fp) as f64;
        let recall = tp as f64 / tp_denom;
        pr_points.push((recall, precision));
    }

    // Sort by recall ascending for the plot.
    pr_points.sort_by(|a, b| a.0.total_cmp(&b.0));

    let pr_auc = average_precision(&pr_points);

    ModelMetrics {
        roc_auc,
        pr_auc,
        n_samples: n,
        n_positive,
        roc_curve,
        pr_curve: pr_points,
        scores,
        labels,
    }
}

fn trapezoidal_auc(curve: &[(f64, f64)]) -> f64 {
    curve
        .windows(2)
        .map(|w| {
            let dx = w[1].0 - w[0].0;
            let avg_y = (w[0].1 + w[1].1) / 2.0;
            dx * avg_y
        })
        .sum::<f64>()
        .abs()
}

/// Average precision = area under the PR curve, computed with the
/// step-function (sklearn-compatible) estimator.
fn average_precision(pr: &[(f64, f64)]) -> f64 {
    if pr.is_empty() {
        return 0.0;
    }
    // AP = sum of precision[i] * (recall[i] - recall[i-1])
    let mut ap = 0.0;
    let mut prev_recall = 0.0;
    for &(recall, precision) in pr {
        ap += precision * (recall - prev_recall).max(0.0);
        prev_recall = recall;
    }
    ap
}

// ─────────────────────────────────────────────────────────────────────────────
// Main evaluation entry-point
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate the ONNX edge-ranking model against a labelled Parquet file.
///
/// Steps:
/// 1. Load features and labels from `features_parquet`.
/// 2. Resolve the ONNX model path from `xgb_params_path`.
/// 3. Run inference with [`EdgeRankingModelPool`].
/// 4. Compute ROC-AUC and PR-AUC.
/// 5. Optionally write evaluation plots to `plot_dir`.
pub fn model_evaluation(
    features_parquet: &Utf8Path,
    xgb_params_path: &Utf8Path,
    plot_dir: Option<&Utf8Path>,
) -> Result<()> {
    // ── Locate ONNX model ──────────────────────────────────────────────────
    let onnx_path = resolve_onnx_path(xgb_params_path)?;
    tracing::info!(path = %onnx_path, "resolved ONNX model path");

    if !onnx_path.exists() {
        bail!(
            "ONNX model not found: {onnx_path}\n\
             Run `pdm run python src/train.py` first to generate the model."
        );
    }

    // ── Load features ──────────────────────────────────────────────────────
    tracing::info!(path = %features_parquet, "loading edge features");
    let data = load_features(features_parquet)?;
    let n = data.features.nrows();
    tracing::info!(n_samples = n, "features loaded");

    // ── Run inference ──────────────────────────────────────────────────────
    let pool = EdgeRankingModelPool::new(&onnx_path);
    let scores =
        pool.with_mut(|model| model.predict_positive_proba_from_array(data.features.clone()))?;

    // ── Compute metrics ────────────────────────────────────────────────────
    let metrics = compute_metrics(scores, data.labels);

    tracing::info!("Model evaluation results:");
    tracing::info!("{:-<48}", "");
    tracing::info!("  Samples   : {}", metrics.n_samples);
    tracing::info!(
        "  Positives : {} ({:.1}%)",
        metrics.n_positive,
        100.0 * metrics.n_positive as f64 / metrics.n_samples.max(1) as f64,
    );
    tracing::info!("  ROC-AUC   : {:.4}", metrics.roc_auc);
    tracing::info!("  PR-AUC    : {:.4}", metrics.pr_auc);

    println!();
    println!("=== ONNX model evaluation ===");
    println!("  Samples   : {}", metrics.n_samples);
    println!(
        "  Positives : {} ({:.1}%)",
        metrics.n_positive,
        100.0 * metrics.n_positive as f64 / metrics.n_samples.max(1) as f64,
    );
    println!("  ROC-AUC   : {:.4}", metrics.roc_auc);
    println!("  PR-AUC    : {:.4}", metrics.pr_auc);

    // ── Plots ──────────────────────────────────────────────────────────────
    if let Some(dir) = plot_dir {
        tracing::info!(dir = %dir, "writing evaluation plots");
        plots::write_eval_plots(&metrics, dir)?;
        println!("  Plots written to: {dir}");
    }

    Ok(())
}
