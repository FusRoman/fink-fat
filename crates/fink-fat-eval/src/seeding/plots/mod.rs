//! Seeding evaluation plot suite.
//!
//! This module groups all plotting code for seeding evaluation into four
//! focused sub-modules:
//!
//! | Sub-module              | Responsibility                                   |
//! |-------------------------|--------------------------------------------------|
//! [`chart_utils`]           | Pure data helpers: histogram bins, percentiles   |
//! [`draw_helpers`]          | Generic plotters primitives (MetricPlot)         |
//! [`truth_distributions`]   | True-pair / triplet parameter distributions      |
//! [`seed_results`]          | Per-night TP/FP/recall/purity result charts      |
//!
//! # Entry-point
//!
//! ```ignore
//! seeding::plots::seeding_plots(ctx, truth, out_dir)?;
//! ```

pub mod chart_utils;
pub mod draw_helpers;
pub mod seed_results;
pub mod truth_distributions;

use anyhow::Result;
use camino::Utf8Path;
use fink_fat_engine::pipeline::PipelineContext;

use crate::truth_sso::TruthSSO;

use seed_results::{NightResultRow, plot_seed_results};
use truth_distributions::{
    collect_truth_data, plot_pair_distributions, plot_triplet_distributions,
};

/// Generate the full seeding plot suite and write all PNGs to `out_dir`.
///
/// Sub-directories are created automatically if absent.
///
/// # Plots produced
///
/// **Truth parameter distributions** (calibration guides for `eval_config.yml`):
/// - `pairs_dt.png`, `pairs_angular_speed.png`, `pairs_mag_diff.png`
/// - `triplets_max_dt.png`, `triplets_pair_sep.png`, `triplets_residual.png`,
///   `triplets_mag_diff.png`
///
/// **Per-night seeding results**:
/// - `seed_counts.png`   – TP / FP / unknown stacked bar chart
/// - `seed_quality.png`  – purity and recall over nights
/// - `seed_recovery.png` – number of recovered vs recoverable trajectories
pub fn seeding_plots(
    plot_pair_triplet_distributions: bool,
    ctx: &PipelineContext<'_>,
    truth: &TruthSSO,
    out_dir: &Utf8Path,
) -> Result<()> {
    let alert_store = &ctx.runtime_state.alert_store;
    let pair_cfg = &ctx.engine_config.pairs;
    let triplet_cfg = &ctx.engine_config.triplets;

    // ── Truth parameter distributions ─────────────────────────────────────────
    if plot_pair_triplet_distributions {
        tracing::info!("computing truth pair/triplet parameter distributions…");
        let (pair_data, triplet_data) = collect_truth_data(alert_store, truth);

        plot_pair_distributions(pair_data, pair_cfg, out_dir)?;
        plot_triplet_distributions(triplet_data, triplet_cfg, out_dir)?;
    }

    // ── Per-night seeding results ──────────────────────────────────────────────
    tracing::info!("computing per-night seeding stats for plots…");
    let (_global, per_night) = super::compute_seeding_stats(ctx, truth)?;

    let rows: Vec<NightResultRow> = per_night
        .iter()
        .map(|(label, stats)| NightResultRow::from_stats(label, stats))
        .collect();

    plot_seed_results(&rows, out_dir)?;

    tracing::info!("seeding plots written to {out_dir}");
    Ok(())
}
