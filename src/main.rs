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
//! Run the binary with an alerts file (one night, or several nights tagged
//! with distinct `night_id`s) and a validated configuration file:
//!
//! ```bash
//! fink-fat --alerts /path/to/alerts.parquet --config /path/to/config.yml
//! ```
//!
//! ## Runtime flow
//!
//! One invocation processes every night present in `--alerts`, in
//! chronological order:
//!
//! 1. Parse CLI arguments in [`crate::init_cli`].
//! 2. Load the alerts (Parquet) into an `ObsDataset`.
//! 3. Load the validated engine configuration and build the `KalmanContext`.
//! 4. Load the `BranchCollection` snapshot from `storage_path` if one exists,
//!    otherwise start from an empty collection.
//! 5. Split the dataset into per-night observation batches — a single batch
//!    (the whole dataset) if it carries no `night_id` index or only one
//!    night, otherwise one batch per `night_id`, sorted chronologically —
//!    and advance the collection through each batch in turn.
//! 6. Overwrite the on-disk snapshot with the result: always after the last
//!    night, and additionally every `--snapshot-every N` nights if that
//!    flag is set (useful to bound work lost to a crash mid-batch).
//!
//! For algorithmic details and configuration schemas, refer to
//! `fink-fat-engine`.
//!

pub mod converter;
pub mod error;
pub mod init_cli;
pub mod logging;
pub mod track;

use crate::{
    converter::convert,
    init_cli::{
        FinkFatCommands::{Convert, Track},
        cli_builder,
    },
    track::tracking,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = cli_builder();

    match cli.command {
        Track(fink_fat_cli_args) => tracking(fink_fat_cli_args),
        Convert {
            config,
            format,
            database_url,
            path_observation,
        } => convert(config, format, database_url, path_observation),
    }
}

#[cfg(test)]
mod main_fink_fat_tests {
    use clap::Parser;
    use fink_fat_engine::engine_config::log_level::LogLevel;

    use crate::{init_cli::FinkFatCliArgs, logging::parse_log_target_override};

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

    #[test]
    fn cli_snapshot_every_defaults_to_none() {
        let cli = FinkFatCliArgs::try_parse_from([
            "fink-fat",
            "--alerts",
            "alerts.parquet",
            "--config",
            "config.yml",
        ])
        .unwrap();
        assert_eq!(cli.snapshot_every, None);
    }

    #[test]
    fn cli_snapshot_every_parses_value() {
        let cli = FinkFatCliArgs::try_parse_from([
            "fink-fat",
            "--alerts",
            "alerts.parquet",
            "--config",
            "config.yml",
            "--snapshot-every",
            "10",
        ])
        .unwrap();
        assert_eq!(cli.snapshot_every, Some(10));
    }
}
