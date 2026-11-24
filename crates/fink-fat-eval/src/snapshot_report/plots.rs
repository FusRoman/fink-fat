//! `plotters` charts for a single `BranchCollection` snapshot. Reuses the
//! generic histogram/bar-chart primitives from
//! [`crate::tracking_report::plots`] rather than duplicating the `plotters`
//! boilerplate.

use anyhow::Result;
use camino::Utf8Path;

use crate::snapshot_report::efficacy::ReconstructionOutcome;
use crate::snapshot_report::{efficacy::ReconstructionEfficacy, stats::SnapshotStats};
use crate::tracking_report::plots::{plot_bar_chart, plot_histogram, plot_signed_histogram};

/// Histogram of hypotheses-per-branch across the snapshot.
pub fn plot_hypotheses_per_branch_histogram(
    stats: &SnapshotStats,
    output_path: &Utf8Path,
) -> Result<()> {
    plot_histogram(
        &stats.hypotheses_per_branch_samples,
        "Hypotheses per branch (snapshot)",
        "Hypotheses in branch",
        output_path,
    )
}

/// Histogram of observations-per-branch (track length) across the snapshot.
pub fn plot_track_length_histogram(stats: &SnapshotStats, output_path: &Utf8Path) -> Result<()> {
    plot_histogram(
        &stats.track_length_per_branch_samples,
        "Observations per branch (snapshot)",
        "Observations in branch's track_ids",
        output_path,
    )
}

/// Histogram of each lineage's best `cumulative_llr` across the snapshot —
/// the same per-lineage aggregate
/// `purge_stale_lineages`/`stale_llr_floor` gate on, so this is the
/// distribution to look at when calibrating that config value against a
/// real run.
pub fn plot_cumulative_llr_per_lineage_histogram(
    stats: &SnapshotStats,
    output_path: &Utf8Path,
) -> Result<()> {
    plot_signed_histogram(
        &stats.cumulative_llr_per_lineage_samples,
        "Best cumulative_llr per lineage (snapshot)",
        "cumulative_llr",
        output_path,
    )
}

/// Bar chart: observation count per night, cross-sectional (this
/// snapshot's footprint only — distinct from `tracking_report::plots`'
/// per-night *line* charts, which need the full run history this mode
/// doesn't have).
pub fn plot_observations_per_night(stats: &SnapshotStats, output_path: &Utf8Path) -> Result<()> {
    let labels: Vec<String> = stats
        .observations_per_night
        .iter()
        .map(|&(night, _)| night.to_string())
        .collect();
    let counts: Vec<(&str, usize)> = labels
        .iter()
        .zip(stats.observations_per_night.iter())
        .map(|(label, &(_, count))| (label.as_str(), count))
        .collect();
    plot_bar_chart(
        &counts,
        "Observations per night (snapshot)",
        "Observations",
        output_path,
    )
}

/// Bar chart of reconstruction-outcome counts.
pub fn plot_reconstruction_outcome_breakdown(
    efficacy: &ReconstructionEfficacy,
    output_path: &Utf8Path,
) -> Result<()> {
    let labels: Vec<String> = ReconstructionOutcome::all()
        .iter()
        .map(|o| o.label())
        .collect();
    let counts: Vec<(&str, usize)> = labels
        .iter()
        .enumerate()
        .map(|(i, l)| (l.as_str(), efficacy.counts[i]))
        .collect();
    plot_bar_chart(
        &counts,
        "Reconstruction outcome breakdown (snapshot)",
        "Trajectories",
        output_path,
    )
}

/// Coverage-ratio histogram: `covered / total` for every `Perfect`/`Partial`
/// trajectory, 0 for every other outcome (mirrors
/// `tracking_report::plots::plot_best_pure_coverage_histogram`'s shape).
pub fn plot_reconstruction_coverage_histogram(
    efficacy: &ReconstructionEfficacy,
    output_path: &Utf8Path,
) -> Result<()> {
    plot_histogram(
        &efficacy.coverage_samples,
        "Reconstruction coverage ratio per trajectory (snapshot)",
        "Coverage ratio (covered / total observations)",
        output_path,
    )
}
