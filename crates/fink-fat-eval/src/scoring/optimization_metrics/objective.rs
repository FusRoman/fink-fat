// src/scoring/optimization_metrics/objective.rs

use crate::scoring::optimization_metrics::EdgeSeparationMetrics;

/// Configuration for scalar objectives used by hyper-parameter optimizers.
///
/// Overview
/// --------
/// `ObjectiveConfig` contains all tunable weights and safety guards used by the
/// scalar objective functions defined in this module:
/// - [`objective_value`] (balanced, "general-purpose"),
/// - [`objective_graph_first`] (Top-K / ranking oriented),
/// - [`objective_sparse_gate`] (global sparse gating oriented).
///
/// Design intent
/// -------------
/// Edge scoring in inter-night linking typically faces two distinct operational regimes:
///
/// 1) **Graph-first**: keep only a small number of candidates per source (Top-K),
///    then solve globally (assignment / flow).
///    In that regime, per-source ranking metrics such as MRR / Hit@K are most relevant.
///
/// 2) **Sparse-gate**: apply a near-global cost threshold to maintain a very sparse graph.
///    In that regime, low-contamination operating points such as `TPR @ FPR=1e-3`
///    are decisive.
///
/// Additionally, the candidate generator may not expose all true edges:
/// - `coverage = n_good / n_true_possible`
///
/// We must therefore penalize configurations that "cheat" by achieving good separation
/// on a tiny reachable subset (very low coverage).
#[derive(Debug, Clone, Copy)]
pub struct ObjectiveConfig {
    // -----------------------
    // General-purpose weights
    // -----------------------
    /// Weight on `tpr_at_fpr_1e3` (higher is better).
    pub w_tpr_at_fpr_1e3: f64,
    /// Weight on `auc_pr` (higher is better).
    pub w_auc_pr: f64,
    /// Weight on `ks` (higher is better).
    pub w_ks: f64,
    /// Weight on candidate `coverage` (higher is better).
    pub w_coverage: f64,
    /// Weight on penalty term `-ln(eps + fpr_at_tpr_99)` (lower FPR is better).
    pub w_log_fpr_at_tpr_99: f64,
    /// Weight on penalty term `-ln(eps + fpr_at_tpr_95)` (lower FPR is better).
    pub w_log_fpr_at_tpr_95: f64,

    // -----------------------
    // Shared numeric guards
    // -----------------------
    /// Numerical stability epsilon for logs.
    pub eps: f64,
    /// Hard floor on coverage; below this, the objective returns a large penalty.
    pub min_coverage: f64,
    /// Large penalty returned on invalid/unusable metrics.
    pub invalid_penalty: f64,

    // -----------------------
    // Graph-first objective
    // -----------------------
    /// Weight on `MRR` (higher is better).
    pub w_mrr: f64,
    /// Weight on source coverage penalty `(1 - source_coverage)`.
    pub w_source_coverage_penalty: f64,

    // -----------------------
    // Sparse-gate objective
    // -----------------------
    /// Weight on `tpr_at_fpr_1e3` for sparse gating (higher is better).
    pub w_tpr_1e3: f64,
    /// Weight on penalty `-ln(eps + fpr_at_tpr_90)` (lower is better).
    pub w_log_fpr90: f64,
}

impl Default for ObjectiveConfig {
    fn default() -> Self {
        Self {
            // general-purpose
            w_tpr_at_fpr_1e3: 2.0,
            w_auc_pr: 1.0,
            w_ks: 0.5,
            w_coverage: 1.5,
            w_log_fpr_at_tpr_99: 0.7,
            w_log_fpr_at_tpr_95: 0.3,
            // guards
            eps: 1e-12,
            min_coverage: 0.05,
            invalid_penalty: 1e9,
            // graph-first
            w_mrr: 5.0,
            w_source_coverage_penalty: 2.0,
            // sparse-gate
            w_tpr_1e3: 4.0,
            w_log_fpr90: 0.5,
        }
    }
}

/// Compute a scalar "general-purpose" loss value from [`EdgeSeparationMetrics`].
///
/// Overview
/// --------
/// This objective mixes:
/// - **low-contamination recall** (`TPR @ FPR=1e-3`),
/// - **ranking quality under imbalance** (PR-AUC),
/// - **global separation** (KS),
/// - **candidate reachability** (coverage),
/// and penalizes high contamination required to reach very high recall
/// (`FPR @ TPR=0.99`, `FPR @ TPR=0.95`).
///
/// The output is a **loss to minimize**.
///
/// Arguments
/// ---------
/// * `m` – Metrics computed on the evaluated population (including `coverage`).
/// * `cfg` – Objective weights and numerical guards.
///
/// Return
/// ------
/// * Finite `f64` loss to **minimize**.
/// * Returns `cfg.invalid_penalty` if metrics are invalid or coverage is too low.
///
/// Notes
/// -----
/// - Penalties are based on `-ln(eps + fpr)`, which grows when `fpr` becomes large,
///   and remains finite at `fpr = 0`.
/// - This objective can still be too "mixed" depending on your operational regime.
///   If you know you are strictly in Top-K mode, prefer [`objective_graph_first`].
///   If you enforce a global threshold for sparsity, prefer [`objective_sparse_gate`].
///
/// See also
/// --------
/// * [`objective_graph_first`] – Ranking/Top-K focused.
/// * [`objective_sparse_gate`] – Global sparse gating focused.
pub fn objective_value(m: &EdgeSeparationMetrics, cfg: ObjectiveConfig) -> f64 {
    let s = &m.edge_metrics;

    let fields = [
        s.auc_pr,
        s.ks,
        s.tpr_at_fpr_1e3,
        s.fpr_at_tpr_99,
        s.fpr_at_tpr_95,
        m.coverage,
    ];

    if fields.iter().any(|&x| !x.is_finite()) {
        return cfg.invalid_penalty;
    }

    if m.coverage < cfg.min_coverage {
        return cfg.invalid_penalty;
    }

    // Reward terms (higher is better).
    let reward = cfg.w_tpr_at_fpr_1e3 * s.tpr_at_fpr_1e3
        + cfg.w_auc_pr * s.auc_pr
        + cfg.w_ks * s.ks
        + cfg.w_coverage * m.coverage;

    // Penalty terms: -ln(eps + fpr) is positive and increases as fpr increases toward 1.
    let p99 = -((cfg.eps + s.fpr_at_tpr_99).ln());
    let p95 = -((cfg.eps + s.fpr_at_tpr_95).ln());
    let penalty = cfg.w_log_fpr_at_tpr_99 * p99 + cfg.w_log_fpr_at_tpr_95 * p95;

    let loss = penalty - reward;
    if loss.is_finite() {
        loss
    } else {
        cfg.invalid_penalty
    }
}

/// Objective focused on Top-K / per-source ranking quality.
///
/// Overview
/// --------
/// This objective is intended for pipelines that:
/// 1) rank edges per source seed,
/// 2) keep Top-K candidates,
/// 3) solve globally (assignment / min-cost flow).
///
/// It mainly rewards:
/// - high MRR (true edge is near the top),
/// - good source coverage (many sources have at least one reachable true edge),
/// - and optionally PR-AUC for global stability.
///
/// Arguments
/// ---------
/// * `m` – Metrics computed on the evaluated population.
/// * `cfg` – Objective weights and numerical guards.
///
/// Return
/// ------
/// * Finite `f64` loss to **minimize**.
/// * `cfg.invalid_penalty` if graph ranking metrics are absent or invalid.
///
/// Notes
/// -----
/// - If your MRR is already saturated (≈ 1.0), this objective may provide little
///   gradient for an optimizer. In that case, consider:
///   - increasing difficulty (more nights, denser negatives),
///   - or optimizing the sparse-gate objective if you do global thresholding.
///
/// See also
/// --------
/// * [`objective_sparse_gate`] – When you enforce global sparsity via thresholds.
pub fn objective_graph_first(m: &EdgeSeparationMetrics, cfg: ObjectiveConfig) -> f64 {
    let Some(grm) = m.graph_ranking_metrics else {
        return cfg.invalid_penalty;
    };

    let auc_pr = m.edge_metrics.auc_pr;
    if !grm.mrr.is_finite() || !grm.coverage.is_finite() || !auc_pr.is_finite() {
        return cfg.invalid_penalty;
    }

    let coverage_penalty = (1.0 - grm.coverage).max(0.0);

    let loss = -cfg.w_mrr * grm.mrr - cfg.w_auc_pr * auc_pr
        + cfg.w_source_coverage_penalty * coverage_penalty;

    if loss.is_finite() {
        loss
    } else {
        cfg.invalid_penalty
    }
}

#[inline]
fn safe_neg_log1p_minus(x: f64, eps: f64) -> f64 {
    // returns -ln(1 - x), x clamped in [0, 1)
    let x = x.max(0.0).min(1.0 - eps);
    -(1.0 - x).ln()
}

/// Sparse-gate objective: positive loss, penalties always >= 0.
///
/// This objective is designed for regimes where a near-global threshold is used
/// to keep the graph sparse.
pub fn objective_sparse_gate(m: &EdgeSeparationMetrics, cfg: ObjectiveConfig) -> f64 {
    let s = &m.edge_metrics;

    let fields = [s.tpr_at_fpr_1e3, s.auc_pr, m.coverage, s.fpr_at_tpr_90];
    if fields.iter().any(|&x| !x.is_finite()) {
        return cfg.invalid_penalty;
    }
    if m.coverage < cfg.min_coverage {
        return cfg.invalid_penalty;
    }

    // Clamp reward components to [0,1] to keep scales stable.
    let tpr = s.tpr_at_fpr_1e3.max(0.0).min(1.0);
    let auc_pr = s.auc_pr.max(0.0).min(1.0);
    let cov = m.coverage.max(0.0).min(1.0);

    // Weighted average reward in [0,1] (if weights positive).
    let w_sum = cfg.w_tpr_1e3 + cfg.w_auc_pr + cfg.w_coverage;
    if w_sum <= 0.0 {
        return cfg.invalid_penalty;
    }
    let reward01 = (cfg.w_tpr_1e3 * tpr + cfg.w_auc_pr * auc_pr + cfg.w_coverage * cov) / w_sum;

    // Penalty: grows slowly for moderate FPR, explodes near 1. Always >= 0.
    let pen = cfg.w_log_fpr90 * safe_neg_log1p_minus(s.fpr_at_tpr_90, cfg.eps);

    // Positive loss: want small.
    let loss = (1.0 - reward01) + pen;

    if loss.is_finite() {
        loss
    } else {
        cfg.invalid_penalty
    }
}

#[inline]
fn hinge2(x: f64) -> f64 {
    // Smooth-ish squared hinge: max(0, x)^2
    let t = x.max(0.0);
    t * t
}

/// Configuration for the linking-aware objective.
///
/// This objective is designed for the exact situation you described:
/// you recover true edges very well, but you also generate many false edges.
///
/// The objective therefore:
/// - enforces *minimum* constraints on recall and coverage (softly via hinge penalties),
/// - strongly penalizes high FPR at high recall (via `-ln(1 - fpr)`),
/// - rewards ranking correctness (Hit@1/MRR, AUC_PR) so that good edges get lower costs.
///
/// All outputs are **positive losses to minimize** (Optuna-friendly and interpretable).
#[derive(Debug, Clone, Copy)]
pub struct LinkingObjectiveConfig {
    // Hard-ish targets (soft constraints)
    pub target_coverage: f64,
    pub target_source_coverage: f64,
    pub target_tpr_at_fpr_1e3: f64,
    pub target_hit_at_1: f64,

    // Weights for soft constraints (hinge penalties)
    pub w_cov_floor: f64,
    pub w_source_cov_floor: f64,
    pub w_tpr_floor: f64,
    pub w_hit1_floor: f64,

    // Weights for ranking quality (continuous rewards)
    pub w_auc_pr: f64,
    pub w_mrr: f64,
    pub w_hit1: f64,
    pub w_ks: f64,

    // Weights for contamination penalties
    pub w_fpr90: f64,
    pub w_fpr95: f64,
    pub w_fpr99: f64,

    // Numeric guards
    pub eps: f64,
    pub invalid_penalty: f64,
}

impl Default for LinkingObjectiveConfig {
    fn default() -> Self {
        Self {
            // Targets: tweak to your operational requirements.
            target_coverage: 0.80,
            target_source_coverage: 0.70,
            target_tpr_at_fpr_1e3: 0.30,
            target_hit_at_1: 0.95,

            // Floors: make them strong enough to matter
            w_cov_floor: 20.0,
            w_source_cov_floor: 10.0,
            w_tpr_floor: 20.0,
            w_hit1_floor: 10.0,

            // Ranking quality: stabilize "good edges must be cheapest"
            w_auc_pr: 2.0,
            w_mrr: 2.0,
            w_hit1: 3.0,
            w_ks: 1.0,

            // Contamination: punish big FPR at high recall HARD
            w_fpr90: 5.0,
            w_fpr95: 5.0,
            w_fpr99: 8.0,

            eps: 1e-12,
            invalid_penalty: 1e9,
        }
    }
}

/// Linking-aware objective: keep true positives, minimize false positives, maximize coverage,
/// and enforce that good edges receive lower costs than bad ones.
///
/// Return
/// ------
/// * A **positive** finite loss to minimize.
/// * `cfg.invalid_penalty` if inputs are missing/invalid.
///
/// Notes
/// -----
/// - Uses soft constraints (squared hinge) to enforce minimum coverage/recall.
/// - Uses `-ln(1 - fpr)` penalties so that contamination near 1 explodes.
/// - Rewards ranking correctness via AUC_PR + Hit@1 + MRR + KS.
pub fn objective_linking_quality(m: &EdgeSeparationMetrics, cfg: LinkingObjectiveConfig) -> f64 {
    let s = &m.edge_metrics;

    let Some(grm) = m.graph_ranking_metrics else {
        return cfg.invalid_penalty;
    };

    // Basic validity checks.
    let fields = [
        m.coverage,
        grm.coverage,
        grm.mrr,
        grm.hit_at_1,
        s.auc_pr,
        s.ks,
        s.tpr_at_fpr_1e3,
        s.fpr_at_tpr_90,
        s.fpr_at_tpr_95,
        s.fpr_at_tpr_99,
    ];
    if fields.iter().any(|&x| !x.is_finite()) {
        return cfg.invalid_penalty;
    }

    // Clamp "good metrics" to [0,1] for stable scaling
    let cov = m.coverage.max(0.0).min(1.0);
    let src_cov = grm.coverage.max(0.0).min(1.0);
    let tpr1e3 = s.tpr_at_fpr_1e3.max(0.0).min(1.0);
    let hit1 = grm.hit_at_1.max(0.0).min(1.0);
    let mrr = grm.mrr.max(0.0).min(1.0);
    let auc_pr = s.auc_pr.max(0.0).min(1.0);
    let ks = s.ks.max(0.0).min(1.0);

    // ---------------------------------------------------------------------
    // 1) Soft constraints: enforce minimum reachability & low-FPR recall.
    // ---------------------------------------------------------------------
    let floor_penalty = cfg.w_cov_floor * hinge2(cfg.target_coverage - cov)
        + cfg.w_source_cov_floor * hinge2(cfg.target_source_coverage - src_cov)
        + cfg.w_tpr_floor * hinge2(cfg.target_tpr_at_fpr_1e3 - tpr1e3)
        + cfg.w_hit1_floor * hinge2(cfg.target_hit_at_1 - hit1);

    // ---------------------------------------------------------------------
    // 2) Contamination penalties: punish large FPR at high recall.
    //    -ln(1 - fpr) is >= 0 and explodes when fpr -> 1.
    // ---------------------------------------------------------------------
    let cont_penalty = cfg.w_fpr90 * safe_neg_log1p_minus(s.fpr_at_tpr_90, cfg.eps)
        + cfg.w_fpr95 * safe_neg_log1p_minus(s.fpr_at_tpr_95, cfg.eps)
        + cfg.w_fpr99 * safe_neg_log1p_minus(s.fpr_at_tpr_99, cfg.eps);

    // ---------------------------------------------------------------------
    // 3) Ranking quality: ensure good edges get lower costs than bad edges.
    //    (AUC_PR / Hit@1 / MRR / KS capture that ordering.)
    // ---------------------------------------------------------------------
    let ranking_loss = cfg.w_auc_pr * (1.0 - auc_pr)
        + cfg.w_hit1 * (1.0 - hit1)
        + cfg.w_mrr * (1.0 - mrr)
        + cfg.w_ks * (1.0 - ks);

    // Total positive loss
    let loss = floor_penalty + cont_penalty + ranking_loss;

    if loss.is_finite() {
        loss
    } else {
        cfg.invalid_penalty
    }
}

#[cfg(test)]
mod objective_tests {
    use super::*;
    use crate::scoring::optimization_metrics::{
        graph_metrics::GraphRankingMetrics,
        metrics::{BestF1, BestYouden, MetricsSummary, OperatingPoint},
    };

    fn mk_metrics(
        coverage: f64,
        auc_pr: f64,
        ks: f64,
        tpr_1e3: f64,
        fpr90: f64,
        fpr95: f64,
        fpr99: f64,
        mrr: f64,
        source_cov: f64,
    ) -> EdgeSeparationMetrics {
        EdgeSeparationMetrics {
            n_good: 10,
            n_bad: 90,
            n_true_possible: 100,
            coverage,
            edge_metrics: MetricsSummary {
                auc: 0.9,
                auc_pr,
                ks,
                best_youden: BestYouden::default(),
                best_f1: BestF1 {
                    best_f1: 0.1,
                    op: OperatingPoint::default(),
                },
                mcc_at_best_f1: 0.0,
                fpr_at_tpr_90: fpr90,
                fpr_at_tpr_95: fpr95,
                fpr_at_tpr_99: fpr99,
                tpr_at_fpr_1e3: tpr_1e3,
            },
            graph_ranking_metrics: Some(GraphRankingMetrics {
                n_sources_total: 100,
                n_sources_with_good: (source_cov * 100.0).round() as usize,
                coverage: source_cov,
                mrr,
                mean_rank: 1.0,
                hit_at_1: 0.0,
                hit_at_5: 0.0,
                hit_at_10: 0.0,
            }),
        }
    }

    #[test]
    fn objective_value_returns_penalty_on_nan() {
        let mut m = mk_metrics(0.2, f64::NAN, 0.5, 0.6, 1e-2, 1e-3, 1e-4, 1.0, 1.0);
        let cfg = ObjectiveConfig::default();
        let v = objective_value(&mut m, cfg);
        assert_eq!(v, cfg.invalid_penalty);
    }

    #[test]
    fn objective_value_penalizes_low_coverage() {
        let m = mk_metrics(0.001, 0.4, 0.5, 0.6, 1e-2, 1e-3, 1e-4, 1.0, 1.0);
        let cfg = ObjectiveConfig::default();
        let v = objective_value(&m, cfg);
        assert_eq!(v, cfg.invalid_penalty);
    }

    #[test]
    fn objective_value_rewards_higher_tpr_at_low_fpr() {
        let cfg = ObjectiveConfig::default();

        let m_low = mk_metrics(0.2, 0.4, 0.5, 0.2, 1e-2, 1e-3, 1e-4, 1.0, 1.0);
        let m_high = mk_metrics(0.2, 0.4, 0.5, 0.8, 1e-2, 1e-3, 1e-4, 1.0, 1.0);

        let v_low = objective_value(&m_low, cfg);
        let v_high = objective_value(&m_high, cfg);

        // Higher TPR@1e-3 should reduce the loss.
        assert!(v_high < v_low);
    }

    #[test]
    fn objective_value_penalizes_higher_fpr_at_tpr99() {
        let cfg = ObjectiveConfig::default();

        let m_good = mk_metrics(0.2, 0.4, 0.5, 0.6, 1e-2, 1e-3, 1e-6, 1.0, 1.0);
        let m_bad = mk_metrics(0.2, 0.4, 0.5, 0.6, 1e-2, 1e-3, 1e-2, 1.0, 1.0);

        let v_good = objective_value(&m_good, cfg);
        let v_bad = objective_value(&m_bad, cfg);

        // Larger FPR@TPR99 should increase the loss.
        assert!(v_bad > v_good);
    }

    #[test]
    fn graph_first_rewards_higher_mrr() {
        let cfg = ObjectiveConfig::default();

        let m_low = mk_metrics(0.2, 0.4, 0.5, 0.6, 1e-2, 1e-3, 1e-4, 0.5, 0.9);
        let m_high = mk_metrics(0.2, 0.4, 0.5, 0.6, 1e-2, 1e-3, 1e-4, 0.99, 0.9);

        let v_low = objective_graph_first(&m_low, cfg);
        let v_high = objective_graph_first(&m_high, cfg);

        // Higher MRR should reduce the loss.
        assert!(v_high < v_low);
    }

    #[test]
    fn sparse_gate_penalizes_high_fpr90() {
        let cfg = ObjectiveConfig::default();

        let m_low = mk_metrics(0.2, 0.4, 0.5, 0.6, 1e-6, 1e-3, 1e-4, 1.0, 1.0);
        let m_high = mk_metrics(0.2, 0.4, 0.5, 0.6, 0.5, 1e-3, 1e-4, 1.0, 1.0);

        let v_low = objective_sparse_gate(&m_low, cfg);
        let v_high = objective_sparse_gate(&m_high, cfg);

        // Larger FPR@TPR90 should increase the loss.
        assert!(v_high > v_low);
    }

    #[test]
    fn sparse_gate_respects_min_coverage_floor() {
        let cfg = ObjectiveConfig::default();

        let m = mk_metrics(0.001, 0.4, 0.5, 0.6, 1e-2, 1e-3, 1e-4, 1.0, 1.0);

        let v = objective_sparse_gate(&m, cfg);
        assert_eq!(v, cfg.invalid_penalty);
    }
}
