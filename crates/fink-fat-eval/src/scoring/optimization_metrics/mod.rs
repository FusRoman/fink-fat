pub mod edge_item;
pub mod graph_metrics;
pub mod metrics;
pub mod objective;

use std::fmt;

use crate::{
    night_seeds::LabeledEdge,
    scoring::optimization_metrics::{
        edge_item::EdgeItem,
        graph_metrics::GraphRankingMetrics,
        metrics::{BestF1, Metrics},
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
    /// Number of "same asteroid" edges produced by the candidate generator.
    ///
    /// Note that this count may be strictly lower than the total number of true edges
    /// theoretically possible between the two nights, which is tracked separately in
    /// [`n_true_possible`].
    pub n_good: usize,
    /// Number of "different asteroid" edges.
    pub n_bad: usize,

    /// Total number of true edges theoretically possible between the two nights.
    ///
    /// This value is derived from `NightSeeds::nb_true_possible_edges` and represents
    /// the size of the positive class in an idealized setting where the candidate
    /// generator is exhaustive. It is used to compute the [`coverage`] field.
    pub n_true_possible: usize,
    /// Fraction of theoretically possible true edges that were actually present in
    /// the candidate set: `coverage = n_good / n_true_possible`.
    ///
    /// A coverage of 1.0 indicates that all possible true edges were generated and
    /// available for scoring, whereas lower values highlight that some fraction
    /// of the positive class was never reachable and thus will be counted as
    /// false negatives in any threshold-based evaluation.
    pub coverage: f64,

    pub edge_metrics: MetricsSummary,

    pub graph_ranking_metrics: Option<GraphRankingMetrics>,
}

impl fmt::Display for EdgeSeparationMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Edge separation metrics")?;
        writeln!(f, "-----------------------")?;

        let n_total = self.n_good.saturating_add(self.n_bad);
        let good_frac = if n_total > 0 {
            (self.n_good as f64) / (n_total as f64)
        } else {
            f64::NAN
        };

        writeln!(f, "Population")?;
        writeln!(f, "  Good edges (same)     : {}", self.n_good)?;
        writeln!(f, "  Bad edges (diff)      : {}", self.n_bad)?;
        if n_total > 0 {
            writeln!(f, "  Good fraction         : {:.4}", good_frac)?;
        } else {
            writeln!(f, "  Good fraction         : NaN (empty)")?;
        }

        // Display the theoretical number of true edges and the coverage of produced edges
        if self.n_true_possible > 0 {
            writeln!(f, "  True edges possible    : {}", self.n_true_possible)?;
            writeln!(f, "  True-edge coverage     : {:.4}", self.coverage)?;
        }

        writeln!(f)?;
        // Reuse MetricsSummary display.
        writeln!(f, "{}", self.edge_metrics)?;

        writeln!(f)?;
        match self.graph_ranking_metrics {
            Some(m) => {
                // Reuse GraphRankingMetrics display.
                writeln!(f, "{m}")?;
            }
            None => {
                writeln!(f, "Graph ranking metrics")?;
                writeln!(f, "---------------------")?;
                writeln!(f, "  (not computed)")?;
            }
        }

        Ok(())
    }
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
    /// Compute separation metrics between good (same asteroid) and bad edges,
    /// while taking into account the total number of true edges that are
    /// theoretically possible between the two nights.
    ///
    /// The additional `n_true_possible` parameter should be obtained via
    /// `NightSeeds::nb_true_possible_edges` on the corresponding pair of nights.
    /// It allows the computation of the [`coverage`] field, defined as
    /// `coverage = n_good / n_true_possible`.
    ///
    /// If `n_true_possible` is zero, coverage will be reported as `0.0` and
    /// the returned struct will still contain valid AUC and other metrics on
    /// the observed edges.
    pub fn edge_metrics(edges: &[LabeledEdge], n_true_possible: usize) -> Option<Self> {
        // 1) Parse once, filter non-finite, count classes.
        let pop = EdgePopulation::from_labeled_edges(edges).ensure_both_classes()?;

        let edge_item: &[EdgeItem] = &pop.items;

        // 2) Compute metrics.
        let metric_summary = edge_item.summary(pop.n_good, pop.n_bad);

        let graph_ranking_metrics = GraphRankingMetrics::from_labeled_edges(edges);

        // Compute coverage: fraction of theoretically possible true edges that were generated.
        let coverage = if n_true_possible > 0 {
            (pop.n_good as f64) / (n_true_possible as f64)
        } else {
            0.0
        };

        Some(Self {
            n_good: pop.n_good,
            n_bad: pop.n_bad,
            n_true_possible,
            coverage,
            edge_metrics: metric_summary,
            graph_ranking_metrics,
        })
    }

    /// Aggregate a list of per-slice/per-night `EdgeSeparationMetrics` into a single global summary.
    ///
    /// Notes
    /// -----
    /// - `n_good`/`n_bad` are summed.
    /// - `MetricsSummary` numeric fields are averaged with weight `n_edges = n_good + n_bad`.
    /// - `best_youden` and `best_f1` are taken from the entry with the largest `n_edges`
    ///   (representative, stable choice without needing to know their internal fields).
    /// - `GraphRankingMetrics` are aggregated by summing counts and weighted-averaging rates
    ///   over `n_sources_with_good`.
    pub fn aggregate(items: &[Option<EdgeSeparationMetrics>]) -> Option<EdgeSeparationMetrics> {
        let mut any = false;

        let mut total_good: usize = 0;
        let mut total_bad: usize = 0;
        // Sum of true edges theoretically possible across slices. Used to compute coverage.
        let mut total_true_possible: usize = 0;

        // Weighted sums for MetricsSummary (weight = n_edges)
        let mut w_edges: f64 = 0.0;

        let mut auc_sum = 0.0;
        let mut auc_pr_sum = 0.0;
        let mut ks_sum = 0.0;
        let mut mcc_sum = 0.0;
        let mut fpr90_sum = 0.0;
        let mut fpr95_sum = 0.0;
        let mut fpr99_sum = 0.0;
        let mut tpr_fpr1e3_sum = 0.0;

        // Representative choices for these (we keep the_attach with the largest n_edges)
        let mut best_rep_edges: usize = 0;
        let mut rep_best_youden: Option<BestYouden> = None;
        let mut rep_best_f1: Option<BestF1> = None;

        // GraphRankingMetrics aggregation
        let mut grm_any = false;
        let mut grm_sources_total: usize = 0;
        let mut grm_sources_with_good: usize = 0;

        // Weighted sums over sources-with-good (these metrics are defined over that subset)
        let mut w_sources_good: f64 = 0.0;
        let mut grm_mrr_sum = 0.0;
        let mut grm_mean_rank_sum = 0.0;
        let mut grm_hit1_sum = 0.0;
        let mut grm_hit5_sum = 0.0;
        let mut grm_hit10_sum = 0.0;

        for opt in items.iter() {
            let m = match opt {
                Some(v) => v,
                None => continue,
            };
            any = true;

            let n_edges_usize = m.n_good.saturating_add(m.n_bad);
            let n_edges = n_edges_usize as f64;

            total_good = total_good.saturating_add(m.n_good);
            total_bad = total_bad.saturating_add(m.n_bad);
            total_true_possible = total_true_possible.saturating_add(m.n_true_possible);

            // Representative for threshold-based structs
            if n_edges_usize > best_rep_edges {
                best_rep_edges = n_edges_usize;
                rep_best_youden = Some(m.edge_metrics.best_youden);
                rep_best_f1 = Some(m.edge_metrics.best_f1);
            }

            // Weighted averages for f64 fields (ignore non-finite values)
            if n_edges > 0.0 {
                let s = &m.edge_metrics;

                // Only accumulate weight for fields we actually add (avoids poisoning with NaNs)
                // Here we keep it simple: if auc is finite, we accept this entry for all fields.
                // You can make this per-field if you prefer.
                if s.auc.is_finite()
                    && s.auc_pr.is_finite()
                    && s.ks.is_finite()
                    && s.mcc_at_best_f1.is_finite()
                    && s.fpr_at_tpr_90.is_finite()
                    && s.fpr_at_tpr_95.is_finite()
                    && s.fpr_at_tpr_99.is_finite()
                    && s.tpr_at_fpr_1e3.is_finite()
                {
                    w_edges += n_edges;
                    auc_sum += n_edges * s.auc;
                    auc_pr_sum += n_edges * s.auc_pr;
                    ks_sum += n_edges * s.ks;
                    mcc_sum += n_edges * s.mcc_at_best_f1;
                    fpr90_sum += n_edges * s.fpr_at_tpr_90;
                    fpr95_sum += n_edges * s.fpr_at_tpr_95;
                    fpr99_sum += n_edges * s.fpr_at_tpr_99;
                    tpr_fpr1e3_sum += n_edges * s.tpr_at_fpr_1e3;
                }
            }

            // Graph ranking metrics
            if let Some(g) = m.graph_ranking_metrics {
                grm_any = true;

                grm_sources_total = grm_sources_total.saturating_add(g.n_sources_total);
                grm_sources_with_good = grm_sources_with_good.saturating_add(g.n_sources_with_good);

                let w = g.n_sources_with_good as f64;
                if w > 0.0
                    && g.mrr.is_finite()
                    && g.mean_rank.is_finite()
                    && g.hit_at_1.is_finite()
                    && g.hit_at_5.is_finite()
                    && g.hit_at_10.is_finite()
                {
                    w_sources_good += w;
                    grm_mrr_sum += w * g.mrr;
                    grm_mean_rank_sum += w * g.mean_rank;
                    grm_hit1_sum += w * g.hit_at_1;
                    grm_hit5_sum += w * g.hit_at_5;
                    grm_hit10_sum += w * g.hit_at_10;
                }
            }
        }

        if !any {
            return None;
        }

        // Build aggregated MetricsSummary
        // If we couldn't accumulate any finite metric entry, return None (or set NaNs if you prefer).
        let edge_metrics = if w_edges > 0.0 {
            MetricsSummary {
                auc: auc_sum / w_edges,
                auc_pr: auc_pr_sum / w_edges,
                ks: ks_sum / w_edges,
                best_youden: rep_best_youden.unwrap_or_else(|| {
                    // Fallback: you may prefer to return None instead if you require it.
                    // This assumes BestYouden is Copy and has a Default.
                    BestYouden::default()
                }),
                best_f1: rep_best_f1.unwrap_or_else(|| BestF1::default()),
                mcc_at_best_f1: mcc_sum / w_edges,
                fpr_at_tpr_90: fpr90_sum / w_edges,
                fpr_at_tpr_95: fpr95_sum / w_edges,
                fpr_at_tpr_99: fpr99_sum / w_edges,
                tpr_at_fpr_1e3: tpr_fpr1e3_sum / w_edges,
            }
        } else {
            // No finite summaries found
            return None;
        };

        let graph_ranking_metrics = if grm_any && grm_sources_total > 0 {
            let coverage = (grm_sources_with_good as f64) / (grm_sources_total as f64);

            // If no sources-with-good contributed finite rates, set them to NaN (or 0.0).
            let (mrr, mean_rank, hit1, hit5, hit10) = if w_sources_good > 0.0 {
                (
                    grm_mrr_sum / w_sources_good,
                    grm_mean_rank_sum / w_sources_good,
                    grm_hit1_sum / w_sources_good,
                    grm_hit5_sum / w_sources_good,
                    grm_hit10_sum / w_sources_good,
                )
            } else {
                (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN)
            };

            Some(GraphRankingMetrics {
                n_sources_total: grm_sources_total,
                n_sources_with_good: grm_sources_with_good,
                coverage,
                mrr,
                mean_rank,
                hit_at_1: hit1,
                hit_at_5: hit5,
                hit_at_10: hit10,
            })
        } else {
            None
        };

        // Compute aggregated coverage over all slices. We define coverage as the fraction
        // of theoretically possible true edges that were actually generated across all
        // evaluated slices: sum(n_good) / sum(n_true_possible). If no slice had any
        // possible true edges, coverage is 0.0.
        let coverage = if total_true_possible > 0 {
            (total_good as f64) / (total_true_possible as f64)
        } else {
            0.0
        };

        Some(EdgeSeparationMetrics {
            n_good: total_good,
            n_bad: total_bad,
            n_true_possible: total_true_possible,
            coverage,
            edge_metrics,
            graph_ranking_metrics,
        })
    }
}
