use std::collections::HashMap;

use fink_fat_engine::seeding::seed_id::SeedId;

use crate::scoring::frozen_pairs::LabeledEdge;

/// Graph-aware ranking metrics computed per source seed (`from`).
///
/// The intent is to evaluate how well the edge cost ranking matches the needs
/// of graph linking pipelines that apply Top-K pruning and assignment/flow solvers.
///
/// Definitions
/// -----------
/// For each unique `from`, we consider the set of candidate edges `(from -> to)`
/// and sort them by `cost` ascending (lower is better).
///
/// We only evaluate sources that have at least one true edge (`same=true`)
/// among their candidates. This mirrors typical linking evaluation:
/// Hit@K / MRR are only meaningful when a "correct answer" exists.
///
/// Metrics
/// -------
/// * Hit@K: fraction of sources whose best true edge is within the top K ranks.
/// * MRR: mean reciprocal rank of the first true edge (1/rank, rank is 1-based).
/// * mean_rank: mean rank of the first true edge (lower is better).
/// * coverage: fraction of sources that have at least one true edge.
///
/// Notes
/// -----
/// - Tie handling: ties on `cost` are broken by stable sort order. In practice,
///   floating-point ties are rare unless costs are quantized.
#[derive(Debug, Clone, Copy)]
pub struct GraphRankingMetrics {
    /// Total number of distinct `from` sources in the input.
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
    /// Arguments
    /// ---------
    /// * `edges` - Labeled edges (`same=true` means correct match).
    ///
    /// Return
    /// ------
    /// * `Some(GraphRankingMetrics)` if at least one source has a true edge.
    /// * `None` if no source has any true edge (cannot compute MRR/Hit@K).
    pub fn graph_ranking_metrics(edges: &[LabeledEdge]) -> Option<GraphRankingMetrics> {
        // Group per `from`: store (cost, is_good)
        let mut groups: HashMap<SeedId, Vec<(f64, bool)>> = HashMap::new();

        for e in edges {
            let cost = e.edge.cost;
            if !cost.is_finite() {
                continue;
            }
            groups.entry(e.edge.from).or_default().push((cost, e.same));
        }

        let n_sources_total = groups.len();
        if n_sources_total == 0 {
            return None;
        }

        let mut n_sources_with_good = 0usize;

        let mut sum_rr = 0.0f64;
        let mut sum_rank = 0.0f64;

        let mut hit1 = 0usize;
        let mut hit5 = 0usize;
        let mut hit10 = 0usize;

        for (_from, mut items) in groups {
            // Skip sources with no true edge: Hit@K / MRR are only meaningful if a true match exists.
            let has_good = items.iter().any(|(_, is_good)| *is_good);
            if !has_good {
                continue;
            }
            n_sources_with_good += 1;

            // Rank candidates by cost (ascending: best first).
            items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Find 1-based rank of the first good edge.
            let mut first_good_rank = None;
            for (idx, (_cost, is_good)) in items.iter().enumerate() {
                if *is_good {
                    first_good_rank = Some(idx + 1);
                    break;
                }
            }

            // has_good implies we must find one.
            let r = first_good_rank.unwrap();

            sum_rr += 1.0 / (r as f64);
            sum_rank += r as f64;

            if r <= 1 {
                hit1 += 1;
            }
            if r <= 5 {
                hit5 += 1;
            }
            if r <= 10 {
                hit10 += 1;
            }
        }

        if n_sources_with_good == 0 {
            return None;
        }

        let denom = n_sources_with_good as f64;
        let coverage = (n_sources_with_good as f64) / (n_sources_total as f64);

        Some(GraphRankingMetrics {
            n_sources_total,
            n_sources_with_good,
            coverage,
            mrr: sum_rr / denom,
            mean_rank: sum_rank / denom,
            hit_at_1: (hit1 as f64) / denom,
            hit_at_5: (hit5 as f64) / denom,
            hit_at_10: (hit10 as f64) / denom,
        })
    }
}
