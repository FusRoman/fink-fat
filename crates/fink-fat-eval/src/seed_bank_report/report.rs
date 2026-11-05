//! Dataset-wide aggregation and printing of [`NightSeedingStats`].

use crate::{
    seed_bank_report::night_stats::{NightSeedingStats, percentage},
    trajectory_processing::{fmt_stats, metric_stats},
};

/// Per-night seeding statistics for a full dataset run, in night order.
#[derive(Debug, Clone, Default)]
pub struct SeedingReport {
    pub per_night: Vec<NightSeedingStats>,
}

impl SeedingReport {
    pub fn push(&mut self, stats: NightSeedingStats) {
        self.per_night.push(stats);
    }

    /// Recall over the whole dataset: ground-truth multi-detection objects
    /// covered by ≥1 pure seed, summed across every night.
    pub fn overall_recall_pct(&self) -> f64 {
        percentage(
            self.sum_by(|n| n.n_objects_recalled),
            self.sum_by(|n| n.n_multi_detection_objects),
        )
    }

    /// Looser recall: objects touched by *any* seed, pure or mixed.
    pub fn overall_loose_recall_pct(&self) -> f64 {
        percentage(
            self.sum_by(|n| n.n_objects_touched),
            self.sum_by(|n| n.n_multi_detection_objects),
        )
    }

    /// Purity over the whole dataset: fraction of produced seeds that are
    /// uncontaminated (single true object).
    pub fn overall_purity_pct(&self) -> f64 {
        percentage(
            self.sum_by(|n| n.n_pure_branches),
            self.sum_by(|n| n.n_branches),
        )
    }

    pub fn total_branches(&self) -> usize {
        self.sum_by(|n| n.n_branches)
    }

    pub fn total_observations(&self) -> usize {
        self.sum_by(|n| n.n_observations)
    }

    /// Hypothesis count of every bank produced across every night, as
    /// `f64` for direct use with [`metric_stats`].
    pub fn all_hypothesis_counts(&self) -> Vec<f64> {
        self.per_night
            .iter()
            .flat_map(|n| n.hypotheses_per_branch.iter().map(|&h| h as f64))
            .collect()
    }

    fn sum_by(&self, f: impl Fn(&NightSeedingStats) -> usize) -> usize {
        self.per_night.iter().map(f).sum()
    }

    /// Print a dataset-wide summary to stdout: headline counters, recall/
    /// purity, and the hypotheses-per-bank distribution.
    pub fn print_summary(&self) {
        let sep = "=".repeat(78);
        println!("\n{sep}");
        println!(
            "Intra-night seeding report — {} nights, {} observations",
            self.per_night.len(),
            self.total_observations()
        );
        println!("{sep}");
        println!(
            "  Seed branches (Kalman banks) produced : {}",
            self.total_branches()
        );
        println!(
            "  Recall  (pure seed found)              : {:.2}%",
            self.overall_recall_pct()
        );
        println!(
            "  Recall  (any seed touched the object)  : {:.2}%",
            self.overall_loose_recall_pct()
        );
        println!(
            "  Purity  (seeds matching a single object): {:.2}%",
            self.overall_purity_pct()
        );

        let hypothesis_counts = self.all_hypothesis_counts();
        println!(
            "  Hypotheses per bank                    : {}",
            fmt_stats(&metric_stats(&hypothesis_counts, |&h| h))
        );
        println!("{sep}\n");
    }
}
