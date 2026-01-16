pub mod graph_metrics;
pub mod metrics;
pub mod edge_item;

use crate::scoring::{
    frozen_pairs::LabeledEdge,
    optimization_metrics::{
        edge_item::EdgeItem, graph_metrics::GraphRankingMetrics, metrics::Metrics
    },
};

pub use metrics::{BestYouden, MetricsSummary};

/// A small container to carry the filtered population and class counts.
///
/// This avoids repeating the same parsing / filtering logic across functions.
#[derive(Debug)]
struct EdgePopulation {
    items: Vec<EdgeItem>,
    n_good: usize,
    n_bad: usize,
}

impl EdgePopulation {
    /// Build a filtered population from labeled edges.
    ///
    /// Rules
    /// -----
    /// - non-finite costs are ignored,
    /// - `score = -cost`,
    /// - `same=true` => "good" class.
    fn from_labeled_edges(edges: &[LabeledEdge]) -> Self {
        let mut items = Vec::with_capacity(edges.len());
        let mut n_good = 0usize;
        let mut n_bad = 0usize;

        for e in edges {
            let cost = e.edge.cost;
            if !cost.is_finite() {
                continue;
            }

            let is_good = e.same;
            if is_good {
                n_good += 1;
            } else {
                n_bad += 1;
            }

            items.push(EdgeItem {
                score: -cost,
                cost,
                is_good,
            });
        }

        Self {
            items,
            n_good,
            n_bad,
        }
    }

    /// Return `None` if one class is missing or if the population is empty.
    fn ensure_both_classes(self) -> Option<Self> {
        if self.items.is_empty() || self.n_good == 0 || self.n_bad == 0 {
            None
        } else {
            Some(self)
        }
    }
}

/// Separation quality metrics for labeled edges.
///
/// Notes
/// -----
/// We assume `ScoredEdge.cost` is a **cost** (lower is better).
/// For ranking-based metrics we use `score = -cost` so that
/// "better" edges have higher scores.
///
/// The main returned values are:
/// - `auc`: ROC AUC on `score = -cost`
/// - `best_j`: best Youden's J (TPR - FPR) over all possible thresholds on `cost`
/// - `ks`: Kolmogorov–Smirnov statistic between the cost distributions
#[derive(Debug, Clone, Copy)]
pub struct EdgeSeparationMetrics {
    /// Number of "same asteroid" edges.
    pub n_good: usize,
    /// Number of "different asteroid" edges.
    pub n_bad: usize,

    pub edge_metrics: MetricsSummary,

    pub graph_ranking_metrics: Option<GraphRankingMetrics>,
}

impl EdgeSeparationMetrics {
    /// Compute separation metrics between good (same asteroid) and bad edges.
    ///
    /// Parameters
    /// ----------
    /// edges : &[LabeledEdge]
    ///     Labeled edges where `same=true` means the endpoints share the same truth id.
    ///
    /// Returns
    /// -------
    /// Option<EdgeSeparationMetrics>
    ///     `None` if one class is missing or if there are not enough valid edges.
    ///
    /// Notes
    /// -----
    /// - AUC uses the Mann–Whitney rank formulation on `score = -cost`.
    /// - Youden's J and KS are computed on `cost` directly, where lower is better.
    /// - Threshold convention: predict "good" if `cost <= threshold`.
    pub fn edge_metrics(edges: &[LabeledEdge]) -> Option<Self> {
        // 1) Parse once, filter non-finite, count classes.
        let pop = EdgePopulation::from_labeled_edges(edges).ensure_both_classes()?;

        let edge_item: &[EdgeItem] = &pop.items;

        // 2) Compute metrics.
        let metric_summary = edge_item.summary(pop.n_good, pop.n_bad);

        let graph_ranking_metrics = GraphRankingMetrics::graph_ranking_metrics(edges);

        Some(Self {
            n_good: pop.n_good,
            n_bad: pop.n_bad,
            edge_metrics: metric_summary,
            graph_ranking_metrics,
        })
    }
}
