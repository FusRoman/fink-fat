use anyhow::Result;
use clap::Parser;

use fink_fat_engine::engine_config::EngineConfig;
use fink_fat_eval::{
    cli::{Cli, load_data},
    ground_truth_state::TruthLookup,
    parquet_export::export_steps_parquet,
    reporting::print_detailed_reports,
};
use photom::TrajId;

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

    let obs_dataset = load_data(&cli.alerts, cli.override_obs_error_arcsec);

    let engine_config = EngineConfig::load_engine_config_validated(cli.config)?;
    let kalman_ctx = engine_config.build_context();

    let truth_lookup = cli
        .ground_truth
        .as_ref()
        .map(TruthLookup::load)
        .transpose()?;

    let vec_traj_id = vec![
        TrajId::Int(35038), // 8467, benoitcarry asteroid
        TrajId::Int(22),
        TrajId::Int(92450),
        TrajId::Int(54013),
        // TrajId::Int(255848),
        // TrajId::Int(146240), // this one is impossible to reconstruct with the kalman.
    ];

    let steps = print_detailed_reports(
        "Single",
        vec_traj_id.as_slice(),
        &obs_dataset,
        &kalman_ctx,
        &engine_config.kfbank_config,
        &engine_config.seeding_grid_config,
        &engine_config.advance_params,
        truth_lookup.as_ref(),
        cli.output_result.as_deref(),
    );

    if let Some(out_path) = &cli.steps_parquet_out {
        export_steps_parquet(&steps, out_path)?;
    }

    Ok(())
}
