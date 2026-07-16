//! # Fink-FAT
//!
//! Fink-FAT is the command-line entrypoint of the asteroid-linking pipeline.
//! It ingests photometric alerts, groups detections into intra-night seeds,
//! links those seeds across nights, and forwards the resulting trajectories to
//! orbit fitting and evaluation stages.
//!
//! The workspace is split into three crates with clear responsibilities:
//!
//! - `fink-fat` — binary crate. Parses the CLI, loads the engine configuration,
//!   initialises logging and progress reporting, and launches the runtime
//!   pipeline.
//! - `fink-fat-engine` — core library. Implements alert models, seeding, edge
//!   construction, solver routing, persistence, and configuration loading.
//! - `fink-fat-eval` — evaluation binaries. Runs the engine on labelled data to
//!   measure seeding, edge, solver, and model quality.
//!
//! ## Typical usage
//!
//! Run the binary with a night's alerts file and a validated configuration
//! file:
//!
//! ```bash
//! fink-fat --alerts /path/to/alerts.parquet --config /path/to/config.yml
//! ```
//!
//! ## Runtime flow
//!
//! One invocation processes one night:
//!
//! 1. Parse CLI arguments in [`crate::init_cli`].
//! 2. Load the night's alerts (Parquet) into an `ObsDataset`.
//! 3. Load the validated engine configuration and build the `KalmanContext`.
//! 4. Load the `BranchCollection` snapshot from `storage_path` if one exists,
//!    otherwise start from an empty collection.
//! 5. Advance the collection by one night.
//! 6. Overwrite the on-disk snapshot with the result, ready for the next
//!    night's invocation.
//!
//! For algorithmic details and configuration schemas, refer to
//! `fink-fat-engine`.
//!

pub mod init_cli;

use fink_fat_engine::{
    engine_config::{EngineConfig, log_level::LogLevel},
    logging::registry::{all_targets, build_env_filter_directive},
    topocentric_kf::branching::BranchCollection,
};
use photom::{
    io::polars::{ContiguousChoice, FromPolarsArgs},
    observation_dataset::ObsDataset,
    observer::error_model::ObsErrorModel,
};
use polars::lazy::frame::{LazyFrame, ScanArgsParquet};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::init_cli::cli_builder;

const STEP_FILENAME: &str = "branch_collection.step";

/// Parse one `--log-target TARGET=LEVEL` argument into `(target, level)`.
fn parse_log_target_override(arg: &str) -> Result<(String, LogLevel), String> {
    let (target, level) = arg
        .split_once('=')
        .ok_or_else(|| format!("invalid --log-target {arg:?}, expected TARGET=LEVEL"))?;
    let level: LogLevel = level
        .parse()
        .map_err(|e| format!("invalid --log-target {arg:?}: {e}"))?;
    Ok((target.to_string(), level))
}

/// Print every tracing target's name, description and levels.
fn print_log_targets() {
    for target in all_targets() {
        println!(
            "{:<24} {:?}  {}",
            target.name, target.levels, target.description
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = cli_builder();

    if cli.list_log_targets {
        print_log_targets();
        return Ok(());
    }

    let alerts = cli.alerts.expect("required unless --list-log-targets");
    let config = cli.config.expect("required unless --list-log-targets");

    let engine_config = EngineConfig::load_engine_config_validated(&config)?;

    std::fs::create_dir_all(engine_config.storage_path())?;

    // Kept alive for the whole run: dropping it flushes the file writer's
    // background worker. `None` when `--logs` wasn't passed.
    let mut _file_log_guard: Option<tracing_appender::non_blocking::WorkerGuard> = None;

    if cli.logs {
        let mut log_targets = engine_config.log_targets.clone();
        for arg in &cli.log_targets {
            let (target, level) = parse_log_target_override(arg)?;
            log_targets.insert(target, level);
        }
        let directive = build_env_filter_directive(engine_config.log_level, &log_targets);
        let filter = tracing_subscriber::EnvFilter::try_new(&directive)?;

        // Same directory as the BranchCollection snapshot (`storage_path`),
        // so logs and pipeline state travel together. Daily rotation, oldest
        // files beyond `log_retention_days` deleted automatically — since
        // rotation is daily, N files kept == N days retained.
        let retention_days = cli
            .log_retention_days
            .unwrap_or(engine_config.log_retention_days);
        let file_appender = tracing_appender::rolling::RollingFileAppender::builder()
            .rotation(tracing_appender::rolling::Rotation::DAILY)
            .filename_prefix("fink_fat")
            .filename_suffix("log")
            .max_log_files(retention_days)
            .build(engine_config.storage_path())?;
        let (non_blocking_file, guard) = tracing_appender::non_blocking(file_appender);
        _file_log_guard = Some(guard);

        // Two separate layers rather than one writer combining both
        // destinations: the file must never carry ANSI color escapes (they
        // show up as garbage in a plain-text log viewer), while the
        // terminal should keep them.
        let stdout_layer = tracing_subscriber::fmt::layer().with_target(true);
        let file_layer = tracing_subscriber::fmt::layer()
            .with_target(true)
            .with_ansi(false)
            .with_writer(non_blocking_file);

        tracing_subscriber::registry()
            .with(filter)
            .with(stdout_layer)
            .with(file_layer)
            .init();
    }

    let lf = LazyFrame::scan_parquet(alerts.as_str().into(), ScanArgsParquet::default())?;
    let obs_dataset = ObsDataset::from_lazy(
        lf,
        FromPolarsArgs {
            do_rechunk: Some(false),
            error_model: Some(ObsErrorModel::FCCT14),
            contiguous_choice: Some(ContiguousChoice::ContiguousNight),
        },
    )?;
    let kalman_context = engine_config.build_context();

    let snapshot_path = engine_config.snapshot_path();
    let step_path = engine_config.storage_path_buf().join(STEP_FILENAME);

    let (collection, current_step) = if snapshot_path.exists() {
        let collection = BranchCollection::load_snapshot_from_disk(
            &snapshot_path,
            &kalman_context,
            &engine_config,
        )?;
        let step: usize = std::fs::read_to_string(&step_path)?.trim().parse()?;
        (collection, step)
    } else {
        (BranchCollection::empty(), 0)
    };

    let night_obs: Vec<&_> = obs_dataset.iter_observations().collect();

    let new_collection = collection.advance_one_night(
        &night_obs,
        &obs_dataset,
        &engine_config,
        &kalman_context,
        current_step,
    )?;

    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&new_collection.to_snapshot())?;
    std::fs::write(&snapshot_path, &bytes)?;
    std::fs::write(&step_path, (current_step + 1).to_string())?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;
    use crate::init_cli::FinkFatCliArgs;

    #[test]
    fn parse_log_target_override_accepts_target_equals_level() {
        let (target, level) = parse_log_target_override("propagation=trace").unwrap();
        assert_eq!(target, "propagation");
        assert_eq!(level, LogLevel::Trace);
    }

    #[test]
    fn parse_log_target_override_rejects_missing_equals() {
        assert!(parse_log_target_override("propagation").is_err());
    }

    #[test]
    fn parse_log_target_override_rejects_unknown_level() {
        assert!(parse_log_target_override("propagation=verbose").is_err());
    }

    #[test]
    fn cli_accepts_repeated_log_target_flags() {
        let cli = FinkFatCliArgs::try_parse_from([
            "fink-fat",
            "--alerts",
            "alerts.parquet",
            "--config",
            "config.yml",
            "--log-target",
            "propagation=trace",
            "--log-target",
            "update=debug",
        ])
        .unwrap();

        assert_eq!(
            cli.log_targets,
            vec!["propagation=trace".to_string(), "update=debug".to_string()]
        );
    }

    #[test]
    fn cli_list_log_targets_does_not_require_alerts_or_config() {
        let cli = FinkFatCliArgs::try_parse_from(["fink-fat", "--list-log-targets"]).unwrap();
        assert!(cli.list_log_targets);
        assert!(cli.alerts.is_none());
        assert!(cli.config.is_none());
    }

    #[test]
    fn cli_requires_alerts_and_config_without_list_log_targets() {
        let err = FinkFatCliArgs::try_parse_from(["fink-fat"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
    }
}
