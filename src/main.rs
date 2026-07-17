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

pub mod init_cli;
pub mod logging;

use fink_fat_engine::{engine_config::EngineConfig, topocentric_kf::branching::BranchCollection};
use photom::{
    io::polars::{ContiguousChoice, FromPolarsArgs},
    observation_dataset::{ObsDataset, observation::Observation},
    observer::error_model::ObsErrorModel,
};
use polars::lazy::frame::{LazyFrame, ScanArgsParquet};

use crate::{
    init_cli::cli_builder,
    logging::{initialize_logs, print_log_targets},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = cli_builder();

    if cli.list_log_targets {
        print_log_targets();
        return Ok(());
    }

    let alerts = cli
        .alerts
        .clone()
        .expect("required unless --list-log-targets");
    let config = cli
        .config
        .clone()
        .expect("required unless --list-log-targets");

    let engine_config = EngineConfig::load_engine_config_validated(config)?;

    std::fs::create_dir_all(engine_config.storage_path())?;

    // Kept alive for the whole run: dropping it flushes the file writer's
    // background worker. `None` when `--logs` wasn't passed.
    let mut _file_log_guard: Option<tracing_appender::non_blocking::WorkerGuard> = None;

    if cli.logs {
        initialize_logs(&cli, &engine_config, &mut _file_log_guard)?;
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

    let mut collection = if snapshot_path.exists() {
        BranchCollection::load_snapshot_from_disk(&snapshot_path, &kalman_context, &engine_config)?
    } else {
        BranchCollection::empty()
    };

    // A dataset tagged with more than one `night_id` is processed one night
    // at a time, in chronological order; anything else (no night index, or
    // a single night) is treated as one logical night, exactly as before.
    let night_batches: Vec<Vec<&Observation>> = match obs_dataset.nb_night() {
        Some(n) if n > 1 => {
            let mut night_ids: Vec<_> = obs_dataset
                .iter_night_id()
                .expect("nb_night() > 1 implies a night index exists")
                .copied()
                .collect();
            night_ids.sort_unstable();
            night_ids
                .iter()
                .map(|night_id| {
                    obs_dataset
                        .iter_night_observations(night_id)
                        .expect("night_id came from iter_night_id()")
                        .collect()
                })
                .collect()
        }
        _ => vec![obs_dataset.iter_observations().collect()],
    };

    let n_batches = night_batches.len();
    for (i, night_obs) in night_batches.iter().enumerate() {
        let current_step = collection.current_step;
        collection = collection.advance_one_night(
            night_obs,
            &obs_dataset,
            &engine_config,
            &kalman_context,
            current_step,
        )?;

        if should_write_snapshot(i + 1, n_batches, cli.snapshot_every) {
            let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&collection.to_snapshot())?;
            std::fs::write(&snapshot_path, &bytes)?;
        }
    }

    Ok(())
}

/// Whether the snapshot should be written to disk after processing the
/// `nights_done`-th night (1-based) out of `total` in this run.
///
/// Always `true` on the last night, regardless of `snapshot_every`, so a
/// run never finishes without persisting its final state. Otherwise `true`
/// every `snapshot_every` nights, if set.
fn should_write_snapshot(nights_done: usize, total: usize, snapshot_every: Option<usize>) -> bool {
    nights_done == total || snapshot_every.is_some_and(|n| n > 0 && nights_done.is_multiple_of(n))
}

#[cfg(test)]
mod main_fink_fat_tests {
    use clap::Parser;
    use fink_fat_engine::engine_config::log_level::LogLevel;

    use crate::{
        init_cli::FinkFatCliArgs, logging::parse_log_target_override, should_write_snapshot,
    };

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

    #[test]
    fn should_write_snapshot_always_true_on_last_night() {
        assert!(should_write_snapshot(1, 1, None));
        assert!(should_write_snapshot(5, 5, None));
        assert!(should_write_snapshot(5, 5, Some(1000)));
    }

    #[test]
    fn should_write_snapshot_false_between_intervals_without_flag() {
        assert!(!should_write_snapshot(1, 5, None));
        assert!(!should_write_snapshot(4, 5, None));
    }

    #[test]
    fn should_write_snapshot_true_every_n_nights() {
        assert!(should_write_snapshot(3, 10, Some(3)));
        assert!(!should_write_snapshot(4, 10, Some(3)));
        assert!(should_write_snapshot(6, 10, Some(3)));
    }
}
