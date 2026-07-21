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
    reporting::{
        print_detailed_reports, print_extremes_table, print_global_aggregate_stats,
        print_nis_by_step_since_bootstrap, print_nis_calibration_summary, print_run_counters,
        print_stop_reason_histogram,
    },
    trajectory_processing::{process_all_trajectories, select_extremes},
};
use photom::TrajId;

/// Number of best/worst trajectories to report in detail.
const N_EXTREMES: usize = 5;

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
    let (_, obs_dataset) = load_data(&cli.alerts);

    let engine_config = EngineConfig::load_engine_config_validated(cli.config)?;
    let kalman_ctx = engine_config.build_context();

    println!("Scanning the dataset and running the Kalman filter bank on every trajectory…");
    let (summaries, counters, nis_step_buckets) = process_all_trajectories(
        &obs_dataset,
        &kalman_ctx,
        &engine_config.kfbank_config,
        &engine_config.seeding_grid_config,
        &engine_config.advance_params,
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

    let (best, worst) = select_extremes(&summaries, N_EXTREMES);
    print_extremes_table("🏆 Best trajectories (highest 3σ coverage)", &best);
    print_extremes_table("⚠️  Worst trajectories (lowest 3σ coverage)", &worst);

    let best_ids: Vec<TrajId> = best.iter().map(|s| s.traj_id.clone()).collect();
    let worst_ids: Vec<TrajId> = worst.iter().map(|s| s.traj_id.clone()).collect();

    print_detailed_reports(
        "BEST",
        &best_ids,
        &obs_dataset,
        &kalman_ctx,
        &engine_config.kfbank_config,
        &engine_config.seeding_grid_config,
        &engine_config.advance_params,
    );
    print_detailed_reports(
        "WORST",
        &worst_ids,
        &obs_dataset,
        &kalman_ctx,
        &engine_config.kfbank_config,
        &engine_config.seeding_grid_config,
        &engine_config.advance_params,
    );

    Ok(())
}
