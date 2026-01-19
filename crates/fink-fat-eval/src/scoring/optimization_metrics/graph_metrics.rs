//! Graph-aware ranking metrics for scored candidate edges.
//!
//! Overview
//! --------
//! Linking pipelines (Top-K pruning → assignment / min-cost flow) care less about
//! global classification performance and more about **whether the correct match
//! is ranked near the top for each source seed**.
//!
//! This module computes **per-source ranking metrics** from a list of labeled,
//! scored edges (`from -> to`), where:
//! - lower `cost` is better,
//! - `same=true` means the edge is a *true* association (same asteroid truth id).
//!
//! Core idea
//! ---------
//! For each distinct `from` seed:
//! 1) collect all its candidate edges,
//! 2) sort candidates by increasing `cost`,
//! 3) find the **rank (1-based)** of the first true edge,
//! 4) aggregate ranks across sources.
//!
//! We only compute ranking metrics on sources that have **at least one true edge**
//! among their candidates (otherwise MRR/Hit@K are undefined: there is no “correct answer”
//! to retrieve).
//!
//! Metric definitions
//! ------------------
//! Let `rank(from)` be the 1-based rank of the first true edge for a given source.
//!
//! - **MRR (Mean Reciprocal Rank)**: `mean(1 / rank(from))`
//!   - Range: (0, 1]
//!   - Interpretation: higher is better. A value near 1 means the true edge is
//!     usually ranked first.
//!
//! - **Mean rank**: `mean(rank(from))`
//!   - Range: [1, +∞)
//!   - Interpretation: lower is better. A mean rank of 2 means the first true edge
//!     appears on average at position #2.
//!
//! - **Hit@K**: fraction of sources with `rank(from) <= K`
//!   - Range: [0, 1]
//!   - Interpretation: higher is better. Hit@10≈0.95 means “for 95% of sources,
//!     the first correct match is within the Top-10 candidates by cost”.
//!
//! - **Coverage**: `n_sources_with_good / n_sources_total`
//!   - Interpretation: “How often does a source have at least one true candidate
//!     edge in the provided dataset?” This is a *dataset / candidate-generation*
//!     property as much as a scoring property. Low coverage often indicates that
//!     the candidate generator is too strict or that the truth labeling is sparse.
//!
//! Notes
//! -----
//! - Non-finite costs (`NaN`, `±inf`) are ignored.
//! - Sorting uses `f64::total_cmp` to avoid panics and handle edge cases consistently.
//! - Tie handling: ties on `cost` follow the sort order; in practice, exact ties are rare
//!   unless costs are quantized.

use std::collections::HashMap;

use fink_fat_engine::seeding::seed_id::SeedId;

use crate::night_seeds::LabeledEdge;

/// Graph-aware ranking metrics computed per source seed (`from`).
///
/// The intent is to evaluate how well the edge cost ranking matches the needs
/// of graph linking pipelines that apply Top-K pruning and assignment/flow solvers.
///
/// Metrics are computed **per source seed**, then aggregated across sources that
/// have at least one true edge among their candidates.
///
/// Fields
/// ------
/// - `n_sources_total`: number of distinct sources seen in the input (after filtering
///   out non-finite-cost edges).
/// - `n_sources_with_good`: number of sources that have at least one true edge.
/// - `coverage`: fraction of sources with at least one true edge.
/// - `mrr`: mean reciprocal rank of the first true edge.
/// - `mean_rank`: mean rank (1-based) of the first true edge.
/// - `hit_at_1`, `hit_at_5`, `hit_at_10`: Hit@K rates.
#[derive(Debug, Clone, Copy)]
pub struct GraphRankingMetrics {
    /// Total number of distinct `from` sources in the input (after filtering invalid costs).
    pub n_sources_total: usize,
    /// Number of sources that have at least one true edge (`same=true`).
    pub n_sources_with_good: usize,
    /// Fraction `n_sources_with_good / n_sources_total`.
    pub coverage: f64,
    /// Mean reciprocal rank over sources with at least one true edge.
    pub mrr: f64,
    /// Mean rank (1-based) of the first true edge over sources with at least one true edge.
    pub mean_rank: f64,
    /// Hit@1 over sources with at least one true edge.
    pub hit_at_1: f64,
    /// Hit@5 over sources with at least one true edge.
    pub hit_at_5: f64,
    /// Hit@10 over sources with at least one true edge.
    pub hit_at_10: f64,
}

impl GraphRankingMetrics {
    /// Compute graph-aware ranking metrics from labeled scored edges.
    ///
    /// This function groups candidate edges by their `from` seed, ranks each group
    /// by increasing `cost`, then measures where the first true edge appears.
    ///
    /// Computation details
    /// -------------------
    /// For each `from` seed:
    /// - candidates are sorted by increasing `cost`,
    /// - the **rank** of the first true edge is the first index (1-based) where `same=true`,
    /// - ranks are aggregated across sources that have at least one true edge:
    ///   - `MRR = mean(1 / rank)`
    ///   - `mean_rank = mean(rank)`
    ///   - `Hit@K = fraction(rank <= K)`
    ///
    /// Arguments
    /// ---------
    /// * `labeled_edges` - Labeled scored edges (`same=true` means correct match).
    ///
    /// Return
    /// ------
    /// * `Some(GraphRankingMetrics)` if at least one source has a true edge.
    /// * `None` if no source has any true edge (cannot compute MRR/Hit@K).
    pub fn from_labeled_edges(labeled_edges: &[LabeledEdge]) -> Option<Self> {
        // Group candidate edges by their source seed:
        //   source_seed -> list of (edge_cost, is_true_edge)
        let mut candidates_by_source: HashMap<SeedId, Vec<(f64, bool)>> = HashMap::new();

        for labeled_edge in labeled_edges {
            let edge_cost = labeled_edge.edge.cost;

            // Ignore non-finite costs to keep ranking well-defined and avoid NaN issues.
            if !edge_cost.is_finite() {
                continue;
            }

            let source_seed = labeled_edge.edge.from;
            let is_true_edge = labeled_edge.same;

            candidates_by_source
                .entry(source_seed)
                .or_default()
                .push((edge_cost, is_true_edge));
        }

        // Number of unique sources that have at least one finite-cost candidate edge.
        let n_sources_total = candidates_by_source.len();
        if n_sources_total == 0 {
            return None;
        }

        // Accumulators across sources that have at least one true edge.
        let mut n_sources_with_good = 0usize;

        let mut reciprocal_rank_sum = 0.0f64;
        let mut rank_sum = 0.0f64;

        let mut hit_at_1_count = 0usize;
        let mut hit_at_5_count = 0usize;
        let mut hit_at_10_count = 0usize;

        for (_source_seed, mut candidate_list) in candidates_by_source {
            // Ranking metrics are only meaningful if at least one true edge exists.
            let source_has_true_edge = candidate_list.iter().any(|(_cost, is_true)| *is_true);
            if !source_has_true_edge {
                continue;
            }
            n_sources_with_good += 1;

            // Sort by increasing cost (best candidates first).
            // Use total ordering to avoid panics and handle corner cases robustly.
            candidate_list.sort_by(|(cost_a, _), (cost_b, _)| cost_a.total_cmp(cost_b));

            // Find the 1-based rank of the first true edge.
            let mut first_true_rank_1based: Option<usize> = None;
            for (rank_zero_based, (_cost, is_true)) in candidate_list.iter().enumerate() {
                if *is_true {
                    first_true_rank_1based = Some(rank_zero_based + 1);
                    break;
                }
            }

            // `source_has_true_edge` guarantees we find a true edge.
            let rank_1based = first_true_rank_1based.unwrap();

            reciprocal_rank_sum += 1.0 / (rank_1based as f64);
            rank_sum += rank_1based as f64;

            if rank_1based <= 1 {
                hit_at_1_count += 1;
            }
            if rank_1based <= 5 {
                hit_at_5_count += 1;
            }
            if rank_1based <= 10 {
                hit_at_10_count += 1;
            }
        }

        if n_sources_with_good == 0 {
            // No source had a true edge, so ranking metrics are undefined.
            return None;
        }

        let n_eval_sources_f64 = n_sources_with_good as f64;

        // Coverage is defined relative to all sources present in the input (finite-cost),
        // not only the evaluated ones.
        let coverage = (n_sources_with_good as f64) / (n_sources_total as f64);

        Some(GraphRankingMetrics {
            n_sources_total,
            n_sources_with_good,
            coverage,
            mrr: reciprocal_rank_sum / n_eval_sources_f64,
            mean_rank: rank_sum / n_eval_sources_f64,
            hit_at_1: (hit_at_1_count as f64) / n_eval_sources_f64,
            hit_at_5: (hit_at_5_count as f64) / n_eval_sources_f64,
            hit_at_10: (hit_at_10_count as f64) / n_eval_sources_f64,
        })
    }
}
