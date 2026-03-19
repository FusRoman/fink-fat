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
    /// Left-seed identifier (`from_seed_id`), used to group edges by source seed
    /// for per-seed ranking evaluation.
    pub seed_ids: Vec<u64>,
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
    select_cols.push(col("from_seed_id"));

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

    // Load left-seed identifiers.
    let seed_col = df
        .column("from_seed_id")
        .context("missing column 'from_seed_id'")?;
    let seed_ids: Vec<u64> = seed_col
        .u64()
        .context("'from_seed_id' is not u64")?
        .iter()
        .map(|v| v.unwrap_or(0))
        .collect();

    tracing::info!(
        n_samples = n,
        n_positive = labels.iter().filter(|&&b| b).count(),
        n_negative = labels.iter().filter(|&&b| !b).count(),
        "features loaded",
    );

    Ok(FeatureData {
        features,
        labels,
        seed_ids,
    })
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
    /// Left-seed identifier per sample (`from_seed_id`), used for ranking evaluation.
    pub seed_ids: Vec<u64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Ranking metrics (top-k per left seed)
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum k evaluated in the ranking curves.
pub const RANKING_K_MAX: usize = 50;

/// Per-seed top-k ranking evaluation metrics.
///
/// All `*_at_k` vectors have length [`RANKING_K_MAX`]: index `i` corresponds to
/// `k = i + 1`. Averages are computed over seeds that contain at least one
/// true-positive edge (`n_seeds_with_tp`).
#[derive(Debug, Clone)]
pub struct RankingMetrics {
    /// Mean Recall@k across seeds-with-TP: fraction of TPs recovered in the top-k.
    pub recall_at_k: Vec<f64>,
    /// Mean Precision@k across seeds-with-TP: fraction of top-k slots that are TP.
    pub precision_at_k: Vec<f64>,
    /// Hit-Rate@k: fraction of seeds-with-TP that have ≥1 TP in their top-k.
    pub hit_rate_at_k: Vec<f64>,
    /// `k_max` used for the curves (= [`RANKING_K_MAX`]).
    pub k_max: usize,
    /// Rank of the first TP per left seed (1-indexed).
    /// `None` means no TP exists in the seed group (or none found within top-k_max).
    pub first_tp_rank: Vec<Option<usize>>,
    /// Total number of distinct left seeds.
    pub n_seeds: usize,
    /// Number of left seeds that contain at least one TP edge.
    pub n_seeds_with_tp: usize,
}

/// Compute per-seed top-k ranking metrics from `ModelMetrics`.
///
/// For each distinct `from_seed_id`, edges are sorted by descending model score.
/// Recall, Precision and Hit-Rate are accumulated for every k in 1..=`RANKING_K_MAX`
/// and then averaged over seeds that contain at least one TP.
pub fn compute_ranking_metrics(metrics: &ModelMetrics) -> RankingMetrics {
    use std::collections::HashMap;

    // ── Group sample indices by left seed ───────────────────────────────────
    let mut groups: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &sid) in metrics.seed_ids.iter().enumerate() {
        groups.entry(sid).or_default().push(i);
    }
    let n_seeds = groups.len();

    let mut recall_sums = vec![0.0f64; RANKING_K_MAX];
    let mut precision_sums = vec![0.0f64; RANKING_K_MAX];
    let mut hit_counts = vec![0usize; RANKING_K_MAX];
    let mut first_tp_ranks: Vec<Option<usize>> = Vec::with_capacity(n_seeds);
    let mut n_seeds_with_tp = 0usize;

    for (_sid, mut indices) in groups {
        // Sort by descending model score.
        indices.sort_unstable_by(|&a, &b| metrics.scores[b].total_cmp(&metrics.scores[a]));

        let n_tp = indices.iter().filter(|&&i| metrics.labels[i]).count();
        let k_eff = RANKING_K_MAX.min(indices.len());

        // Rank of first TP within top-K_MAX (1-indexed), None if absent.
        first_tp_ranks.push(
            indices[..k_eff]
                .iter()
                .position(|&i| metrics.labels[i])
                .map(|p| p + 1),
        );

        if n_tp == 0 {
            continue;
        }
        n_seeds_with_tp += 1;

        // Accumulate per-k metrics up to k_eff.
        let mut tp_in_topk = 0usize;
        for k in 1..=k_eff {
            if metrics.labels[indices[k - 1]] {
                tp_in_topk += 1;
            }
            recall_sums[k - 1] += tp_in_topk as f64 / n_tp as f64;
            precision_sums[k - 1] += tp_in_topk as f64 / k as f64;
            if tp_in_topk > 0 {
                hit_counts[k - 1] += 1;
            }
        }
        // Extend beyond k_eff: recall is saturated, precision continues to decay.
        let tp_final = tp_in_topk;
        for k in (k_eff + 1)..=RANKING_K_MAX {
            recall_sums[k - 1] += tp_final as f64 / n_tp as f64;
            precision_sums[k - 1] += tp_final as f64 / k as f64;
            if tp_final > 0 {
                hit_counts[k - 1] += 1;
            }
        }
    }

    let denom = n_seeds_with_tp.max(1) as f64;
    RankingMetrics {
        recall_at_k: recall_sums.iter().map(|&s| s / denom).collect(),
        precision_at_k: precision_sums.iter().map(|&s| s / denom).collect(),
        hit_rate_at_k: hit_counts.iter().map(|&c| c as f64 / denom).collect(),
        k_max: RANKING_K_MAX,
        first_tp_rank: first_tp_ranks,
        n_seeds,
        n_seeds_with_tp,
    }
}

/// Compute ROC-AUC, PR-AUC and the corresponding curves.
pub fn compute_metrics(scores: Vec<f32>, labels: Vec<bool>, seed_ids: Vec<u64>) -> ModelMetrics {
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
        seed_ids,
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
///
/// Arguments
/// ---------
/// * `features_parquet` – Path to the edge-features Parquet file produced by
///   `edge-eval --export-features`.
/// * `xgb_params_path` – Path to the XGBoost training config (`xgb_params.yml`).
///   The `training.onnx_output` field is read to locate the ONNX model; the
///   path is resolved relative to the config file's directory.
/// * `plot_dir` – Optional output directory for evaluation plots (`roc_curve.png`,
///   `pr_curve.png`, `score_distribution.png`). No plots are written when `None`.
/// * `onnx_intra_threads` – Optional intra-op thread count for the ORT session.
///   `None` lets ORT choose automatically (typically = logical CPU count).
///   `Some(n)` pins the session to `n` intra-op threads; set to `1` for
///   single-threaded inference on shared machines.
///
/// Return
/// ------
/// * `Ok(())` – Evaluation complete; results emitted via `tracing`.
/// * `Err(_)` – If features cannot be loaded, the ONNX model is missing,
///   inference fails, or plot writing fails.
pub fn model_evaluation(
    features_parquet: &Utf8Path,
    xgb_params_path: &Utf8Path,
    plot_dir: Option<&Utf8Path>,
    onnx_intra_threads: Option<usize>,
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
    let scores = pool.with_mut(onnx_intra_threads, |model| {
        model.predict_positive_proba_from_array(data.features.clone())
    })?;

    // ── Compute metrics ────────────────────────────────────────────────────
    let metrics = compute_metrics(scores, data.labels, data.seed_ids);
    let ranking = compute_ranking_metrics(&metrics);

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
    tracing::info!(
        "  Seeds     : {} total, {} with TP",
        ranking.n_seeds,
        ranking.n_seeds_with_tp
    );
    tracing::info!("  Hit-Rate@1 : {:.4}", ranking.hit_rate_at_k[0]);
    tracing::info!("  Hit-Rate@5 : {:.4}", ranking.hit_rate_at_k[4]);
    tracing::info!("  Recall@5   : {:.4}", ranking.recall_at_k[4]);
    tracing::info!("  Recall@10  : {:.4}", ranking.recall_at_k[9]);
    tracing::info!("  Recall@32  : {:.4}", ranking.recall_at_k[31]);

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
    println!(
        "  Seeds     : {} total, {} with TP",
        ranking.n_seeds, ranking.n_seeds_with_tp
    );
    println!("  Hit-Rate@1 : {:.4}", ranking.hit_rate_at_k[0]);
    println!("  Hit-Rate@5 : {:.4}", ranking.hit_rate_at_k[4]);
    println!("  Recall@5   : {:.4}", ranking.recall_at_k[4]);
    println!("  Recall@10  : {:.4}", ranking.recall_at_k[9]);
    println!("  Recall@32  : {:.4}", ranking.recall_at_k[31]);

    // ── Plots ──────────────────────────────────────────────────────────────
    if let Some(dir) = plot_dir {
        tracing::info!(dir = %dir, "writing evaluation plots");
        plots::write_eval_plots(&metrics, dir)?;
        plots::write_ranking_plots(&ranking, dir)?;
        println!("  Plots written to: {dir}");
    }

    Ok(())
}
