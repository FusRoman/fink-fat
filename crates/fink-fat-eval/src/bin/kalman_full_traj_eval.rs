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

use fink_fat_engine::topocentric_kf::{KalmanContext, config::KalmanConfig};
use fink_fat_eval::{
    cli::{Cli, load_data},
    reporting::{
        print_detailed_reports, print_extremes_table, print_global_aggregate_stats,
        print_run_counters,
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

/// Build the [`KalmanContext`] shared by every trajectory processed in this
/// run.
///
/// Uses a slightly tightened process-noise (`q0`) compared to the default
/// configuration, and the DE440 JPL Horizons ephemeris for topocentric
/// corrections.
fn build_kalman_context() -> KalmanContext {
    KalmanContext::new(
        KalmanConfig {
            q0: 1e-13,
            ..KalmanConfig::default()
        },
        "horizon:DE440",
        None,
    )
}

fn main() -> Result<()> {
    init_tracing();

    let cli = Cli::parse();
    let (_, obs_dataset) = load_data(&cli.alerts);
    let kalman_ctx = build_kalman_context();

    println!("Scanning the dataset and running the Kalman filter bank on every trajectory…");
    let (summaries, counters) = process_all_trajectories(&obs_dataset, &kalman_ctx);

    print_run_counters(&counters, summaries.len());

    if summaries.is_empty() {
        println!("\nNo trajectory produced a usable Kalman-filter result.");
        return Ok(());
    }

    print_global_aggregate_stats(&summaries);

    let (best, worst) = select_extremes(&summaries, N_EXTREMES);
    print_extremes_table("🏆 Best trajectories (highest 3σ coverage)", &best);
    print_extremes_table("⚠️  Worst trajectories (lowest 3σ coverage)", &worst);

    let best_ids: Vec<TrajId> = best.iter().map(|s| s.traj_id.clone()).collect();
    let worst_ids: Vec<TrajId> = worst.iter().map(|s| s.traj_id.clone()).collect();

    print_detailed_reports("BEST", &best_ids, &obs_dataset, &kalman_ctx);
    print_detailed_reports("WORST", &worst_ids, &obs_dataset, &kalman_ctx);

    Ok(())
}
