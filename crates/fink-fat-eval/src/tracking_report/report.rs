//! Dataset-wide aggregation, JSON (de)serialization, and printing of
//! [`NightTrackingStats`].

use anyhow::{Context, Result};
use camino::Utf8Path;
use serde::{Deserialize, Serialize};

use crate::{
    tracking_report::night_stats::NightTrackingStats,
    trajectory_processing::{MetricStats, fmt_stats, metric_stats},
};

/// Percentage helper matching [`crate::seed_bank_report::night_stats::percentage`]'s
/// NaN-for-empty convention.
fn percentage(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        f64::NAN
    } else {
        100.0 * numerator as f64 / denominator as f64
    }
}

/// Per-night tracking statistics for a full run. Fully `Serialize`/
/// `Deserialize` so a run can be persisted to disk and its plots
/// regenerated without rerunning the engine (see `bin/tracking_analysis.rs`'s
/// `--from-json` mode) — the dataset-wide [`AggregatedTrackingStats`] is
/// intentionally *not* stored alongside it, just recomputed on demand via
/// [`TrackingReport::finalize`], so there is only ever one source of truth.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrackingReport {
    pub per_night: Vec<NightTrackingStats>,
}

/// Dataset-wide summary computed once, after every night has been pushed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedTrackingStats {
    pub n_nights: usize,
    pub total_observations: usize,
    pub total_branches_produced: usize,
    #[serde(with = "crate::trajectory_processing::finite_f64")]
    pub overall_recall_pct: f64,
    #[serde(with = "crate::trajectory_processing::finite_f64")]
    pub overall_loose_recall_pct: f64,
    #[serde(with = "crate::trajectory_processing::finite_f64")]
    pub overall_purity_pct: f64,
    #[serde(with = "crate::trajectory_processing::finite_f64")]
    pub final_completeness_pct: f64,
    pub branches_per_lineage: MetricStats,
    pub hypotheses_per_branch: MetricStats,
    pub cumulative_llr: MetricStats,
    pub effective_sample_size: MetricStats,
    pub hypothesis_error_box_radius_arcsec: MetricStats,
    pub bank_error_box_radius_arcsec: MetricStats,
    pub n_observations_in_box_next_night: MetricStats,
    pub lineage_survival_nights: MetricStats,
    pub elapsed_ms: MetricStats,
    pub total_elapsed_ms: f64,

    /// Raw samples backing the aggregated histograms in
    /// [`plots`](super::plots). The error-box/branching-factor samples are
    /// one value **per night** (that night's mean) rather than one value
    /// per individual branch/hypothesis — a deliberate size/detail
    /// trade-off, since a full per-branch history across hundreds of nights
    /// with thousands of branches each would make the JSON report
    /// unreasonably large. `lineage_survival_samples` is the one exception:
    /// it's already bounded by the number of lineages that died over the
    /// run, so it's kept at full per-lineage resolution.
    pub hypothesis_error_box_radius_samples: Vec<f64>,
    pub bank_error_box_radius_samples: Vec<f64>,
    pub n_observations_in_box_samples: Vec<f64>,
    pub branches_per_lineage_samples: Vec<f64>,
    pub lineage_survival_samples: Vec<f64>,
}

impl TrackingReport {
    pub fn push(&mut self, stats: NightTrackingStats) {
        self.per_night.push(stats);
    }

    /// Compute the dataset-wide aggregate from every pushed night.
    pub fn finalize(&self) -> AggregatedTrackingStats {
        let sum_by =
            |f: fn(&NightTrackingStats) -> usize| -> usize { self.per_night.iter().map(f).sum() };

        let final_completeness_pct = self
            .per_night
            .last()
            .map(|n| n.completeness_pct_so_far)
            .unwrap_or(f64::NAN);

        let all_hypothesis_radii: Vec<f64> = self
            .per_night
            .iter()
            .map(|n| n.hypothesis_error_box_radius_arcsec.mean)
            .filter(|v| v.is_finite())
            .collect();
        let all_bank_radii: Vec<f64> = self
            .per_night
            .iter()
            .filter_map(|n| n.bank_error_box_radius_arcsec.as_ref())
            .map(|s| s.mean)
            .filter(|v| v.is_finite())
            .collect();
        let all_in_box: Vec<f64> = self
            .per_night
            .iter()
            .filter_map(|n| n.n_observations_in_box_next_night.as_ref())
            .map(|s| s.mean)
            .filter(|v| v.is_finite())
            .collect();
        let all_survival: Vec<f64> = self
            .per_night
            .iter()
            .flat_map(|n| n.lineage_survival_nights_of_died.iter().copied())
            .collect();
        let all_branches_per_lineage: Vec<f64> = self
            .per_night
            .iter()
            .map(|n| n.branches_per_lineage.mean)
            .collect();

        AggregatedTrackingStats {
            n_nights: self.per_night.len(),
            total_observations: sum_by(|n| n.n_observations),
            total_branches_produced: sum_by(|n| n.n_branches),
            overall_recall_pct: percentage(
                sum_by(|n| n.n_objects_recalled_tonight),
                sum_by(|n| n.n_multi_detection_objects_tonight),
            ),
            overall_loose_recall_pct: percentage(
                sum_by(|n| n.n_objects_touched_tonight),
                sum_by(|n| n.n_multi_detection_objects_tonight),
            ),
            overall_purity_pct: percentage(sum_by(|n| n.n_branches_pure), sum_by(|n| n.n_branches)),
            final_completeness_pct,
            branches_per_lineage: metric_stats(&self.per_night, |n| n.branches_per_lineage.mean),
            hypotheses_per_branch: metric_stats(&self.per_night, |n| n.hypotheses_per_branch.mean),
            cumulative_llr: metric_stats(&self.per_night, |n| n.cumulative_llr.mean),
            effective_sample_size: metric_stats(&self.per_night, |n| n.effective_sample_size.mean),
            hypothesis_error_box_radius_arcsec: metric_stats(&all_hypothesis_radii, |&r| r),
            bank_error_box_radius_arcsec: metric_stats(&all_bank_radii, |&r| r),
            n_observations_in_box_next_night: metric_stats(&all_in_box, |&r| r),
            lineage_survival_nights: metric_stats(&all_survival, |&r| r),
            elapsed_ms: metric_stats(&self.per_night, |n| n.elapsed_ms),
            total_elapsed_ms: sum_by_f64(&self.per_night, |n| n.elapsed_ms),
            hypothesis_error_box_radius_samples: all_hypothesis_radii,
            bank_error_box_radius_samples: all_bank_radii,
            n_observations_in_box_samples: all_in_box,
            branches_per_lineage_samples: all_branches_per_lineage,
            lineage_survival_samples: all_survival,
        }
    }

    pub fn write_json(&self, path: &Utf8Path) -> Result<()> {
        let file =
            std::fs::File::create(path).with_context(|| format!("failed to create {path}"))?;
        serde_json::to_writer_pretty(file, self)
            .with_context(|| format!("failed to write JSON to {path}"))
    }

    pub fn read_json(path: &Utf8Path) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| format!("failed to open {path}"))?;
        serde_json::from_reader(file).with_context(|| format!("failed to parse JSON from {path}"))
    }
}

fn sum_by_f64(items: &[NightTrackingStats], f: impl Fn(&NightTrackingStats) -> f64) -> f64 {
    items.iter().map(f).sum()
}

impl AggregatedTrackingStats {
    pub fn print_summary(&self) {
        let sep = "=".repeat(78);
        println!("\n{sep}");
        println!(
            "Night-after-night tracking report — {} nights, {} observations",
            self.n_nights, self.total_observations
        );
        println!("{sep}");
        println!(
            "  Branches (candidate trajectories) produced : {}",
            self.total_branches_produced
        );
        println!(
            "  Recall  (pure branch found, per night)      : {:.2}%",
            self.overall_recall_pct
        );
        println!(
            "  Recall  (any branch touched, per night)     : {:.2}%",
            self.overall_loose_recall_pct
        );
        println!(
            "  Purity  (branches matching a single object) : {:.2}%",
            self.overall_purity_pct
        );
        println!(
            "  Completeness (final, cumulative)            : {:.2}%",
            self.final_completeness_pct
        );
        println!(
            "  Branches per lineage                        : {}",
            fmt_stats(&self.branches_per_lineage)
        );
        println!(
            "  Hypotheses per branch                       : {}",
            fmt_stats(&self.hypotheses_per_branch)
        );
        println!(
            "  Cumulative LLR                              : {}",
            fmt_stats(&self.cumulative_llr)
        );
        println!(
            "  Effective sample size                       : {}",
            fmt_stats(&self.effective_sample_size)
        );
        println!(
            "  Error box radius, per hypothesis (arcsec)   : {}",
            fmt_stats(&self.hypothesis_error_box_radius_arcsec)
        );
        println!(
            "  Error box radius, per bank (arcsec)         : {}",
            fmt_stats(&self.bank_error_box_radius_arcsec)
        );
        println!(
            "  Observations inside next-night box          : {}",
            fmt_stats(&self.n_observations_in_box_next_night)
        );
        println!(
            "  Lineage survival (nights, died lineages)    : {}",
            fmt_stats(&self.lineage_survival_nights)
        );
        println!(
            "  Per-night wall time (ms)                    : {}  (total {:.0} ms)",
            fmt_stats(&self.elapsed_ms),
            self.total_elapsed_ms
        );
        println!("{sep}\n");
    }
}
