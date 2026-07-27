//! Entry point for the Kalman-filter-bank evaluation tool.
//!
//! This binary scans every trajectory of an observation dataset, runs the
//! topocentric Kalman filter bank on each one, and prints:
//!   - dataset-wide aggregate statistics,
//!   - the best/worst trajectories ranked by predictive-region coverage,
//!   - a detailed per-step report for each of those extreme trajectories.
//!
//! The actual work is split into two modules so this file stays a thin
//! orchestration layer:
//!   - [`trajectory_processing`]: materializing trajectories, running the
//!     Kalman filter bank, summarizing results, and ranking trajectories.
//!   - [`reporting`]: all `stdout` formatting and printing.

use anyhow::Result;
use clap::Parser;

use fink_fat_engine::engine_config::EngineConfig;
use fink_fat_eval::{
    cli::{Cli, load_data},
    ground_truth_state::TruthLookup,
    parquet_export::{export_steps_parquet, export_summary_parquet},
    reporting::{
        print_detailed_reports, print_extremes_table, print_global_aggregate_stats,
        print_inflation_diagnostics, print_nees_rmse_dataset_summary,
        print_nis_by_step_since_bootstrap, print_nis_calibration_summary, print_run_counters,
        print_stop_reason_histogram,
    },
    trajectory_processing::{process_all_trajectories, select_extremes},
};
use photom::TrajId;

/// Number of best/worst trajectories to report in detail.
const N_EXTREMES: usize = 0;

/// Configure the global tracing subscriber.
///
/// Verbosity is controlled by the `FINKFAT_LOG` environment variable
/// (defaults to `warn`).
fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("FINKFAT_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_target(true)
        .without_time()
        .with_ansi(false)
        .init();
}

fn main() -> Result<()> {
    init_tracing();

    let cli = Cli::parse();
    let (_, obs_dataset) = load_data(&cli.alerts, cli.override_obs_error_arcsec);

    let engine_config = EngineConfig::load_engine_config_validated(cli.config)?;
    let kalman_ctx = engine_config.build_context();

    let truth_lookup = cli
        .ground_truth
        .as_ref()
        .map(TruthLookup::load)
        .transpose()?;

    println!("Scanning the dataset and running the Kalman filter bank on every trajectory…");

    println!(
        "\n-- Test with q0 = {} --",
        engine_config.kalman_shared_context.config.q0
    );
    println!(
        "-- Test with search_region_chi2 = {} --\n",
        engine_config.kfbank_config.search_region_chi2
    );

    let (summaries, counters, nis_step_buckets) = process_all_trajectories(
        &obs_dataset,
        &kalman_ctx,
        &engine_config.kfbank_config,
        &engine_config.seeding_grid_config,
        &engine_config.advance_params,
        truth_lookup.as_ref(),
    );

    print_run_counters(&counters, summaries.len());
    print_stop_reason_histogram(&counters);
    print_nis_by_step_since_bootstrap(&nis_step_buckets);

    if summaries.is_empty() {
        println!("\nNo trajectory produced a usable Kalman-filter result.");
        return Ok(());
    }

    print_global_aggregate_stats(&summaries);
    print_nis_calibration_summary(&summaries);
    print_nees_rmse_dataset_summary(&summaries);
    print_inflation_diagnostics(&summaries);

    if let Some(out_path) = &cli.summary_parquet_out {
        export_summary_parquet(&summaries, out_path)?;
    }

    let (best, worst) = select_extremes(&summaries, N_EXTREMES);
    print_extremes_table("🏆 Best trajectories (highest 3σ coverage)", &best);
    print_extremes_table("⚠️  Worst trajectories (lowest 3σ coverage)", &worst);

    let best_ids: Vec<TrajId> = best.iter().map(|s| s.traj_id.clone()).collect();
    let worst_ids: Vec<TrajId> = worst.iter().map(|s| s.traj_id.clone()).collect();

    let mut steps = print_detailed_reports(
        "BEST",
        &best_ids,
        &obs_dataset,
        &kalman_ctx,
        &engine_config.kfbank_config,
        &engine_config.seeding_grid_config,
        &engine_config.advance_params,
        truth_lookup.as_ref(),
        cli.output_result.as_deref(),
    );
    steps.extend(print_detailed_reports(
        "WORST",
        &worst_ids,
        &obs_dataset,
        &kalman_ctx,
        &engine_config.kfbank_config,
        &engine_config.seeding_grid_config,
        &engine_config.advance_params,
        truth_lookup.as_ref(),
        cli.output_result.as_deref(),
    ));

    if let Some(out_path) = &cli.steps_parquet_out {
        export_steps_parquet(&steps, out_path)?;
    }

    Ok(())
}
