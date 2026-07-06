use anyhow::Result;
use clap::Parser;

use fink_fat_engine::topocentric_kf::{KalmanContext, config::KalmanConfig};
use fink_fat_eval::{
    cli::{Cli, load_data},
    reporting::print_detailed_reports,
};
use outfit::kepler::{SolverKind, SolverType};
use photom::TrajId;

/// Build the [`KalmanContext`] shared by every trajectory processed in this
/// run.
///
/// Uses a slightly tightened process-noise (`q0`) compared to the default
/// configuration, and the DE440 JPL Horizons ephemeris for topocentric
/// corrections.
fn build_kalman_context() -> KalmanContext {
    let kalman_config = KalmanConfig {
        q0: 1e-12,
        solver_type: SolverType {
            kind: SolverKind::Auto,
            ..Default::default()
        },
        ..Default::default()
    };

    KalmanContext::new(kalman_config, "horizon:DE440", None)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("FINKFAT_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_target(true)
        .without_time()
        .with_ansi(false)
        .init();

    let cli = Cli::parse();

    let (_, obs_dataset) = load_data(&cli.alerts);

    let vec_traj_id = vec![
        TrajId::Int(35038), // 8467, benoitcarry asteroid
                            // TrajId::Int(22),
                            // TrajId::Int(92450),
                            // TrajId::Int(54013),
    ];

    let kalman_ctx = build_kalman_context();
    print_detailed_reports("Single", &vec_traj_id.as_slice(), &obs_dataset, &kalman_ctx);

    Ok(())
}
