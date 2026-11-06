//! Thin adapter between `fink-fat`'s subcommands and the engine's shared
//! subscriber setup ([`fink_fat_engine::logging::subscriber`]).

use fink_fat_engine::{
    engine_config::EngineConfig,
    logging::subscriber::{LogInitError, SubscriberOptions, init_subscriber},
};
use tracing_appender::non_blocking::WorkerGuard;

pub use fink_fat_engine::logging::subscriber::{parse_log_target_override, print_log_targets};

/// Install the global subscriber of a subcommand: terminal, plus
/// `<file_prefix>.<date>.log` under the configuration's `storage_path`,
/// rotated daily and pruned beyond the retention.
///
/// # Arguments
/// * `engine_config` – Supplies the levels, `storage_path` and the default
///   retention.
/// * `file_prefix` – Log file name prefix (`"fink_fat"` for `track`,
///   `"fink_fat_stream"` for `stream`).
/// * `log_targets` – Raw `--log-target TARGET=LEVEL` overrides.
/// * `log_retention_days` – `--log-retention-days`, overriding the
///   configuration's retention.
///
/// # Returns
/// The guard of the background file writer: keep it alive for the whole
/// run, dropping it flushes the log file.
///
/// # Errors
/// See [`LogInitError`].
pub fn initialize_logs(
    engine_config: &EngineConfig,
    file_prefix: &str,
    log_targets: &[String],
    log_retention_days: Option<usize>,
) -> Result<WorkerGuard, LogInitError> {
    init_subscriber(
        engine_config,
        &SubscriberOptions {
            file_prefix,
            target_overrides: log_targets,
            retention_days: log_retention_days,
            stdout: true,
        },
    )
}
