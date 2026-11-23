//! Installation of the global `tracing` subscriber shared by the engine's
//! binaries (`fink-fat track`, `fink-fat-stream`).
//!
//! The filter is built from the [`EngineConfig`]'s `log_level` /
//! `log_targets`, plus per-target overrides from the command line
//! (`--log-target TARGET=LEVEL`, see [`parse_log_target_override`]). Events
//! always go to a daily-rotated file under the engine's `storage_path`, and
//! optionally to the terminal.

use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::{
    engine_config::{EngineConfig, log_level::LogLevel},
    logging::registry::{all_targets, build_env_filter_directive},
};

/// Failure to install the global subscriber (see [`init_subscriber`]).
#[derive(Debug, thiserror::Error)]
pub enum LogInitError {
    /// A `--log-target` override is malformed.
    #[error("{0}")]
    InvalidTargetOverride(String),
    /// The combined filter directive is not a valid `EnvFilter`.
    #[error(transparent)]
    Filter(#[from] tracing_subscriber::filter::ParseError),
    /// The rolling log file could not be created under `storage_path`.
    #[error(transparent)]
    Appender(#[from] tracing_appender::rolling::InitError),
    /// A global subscriber is already installed in this process.
    #[error(transparent)]
    AlreadyInitialized(#[from] tracing_subscriber::util::TryInitError),
}

/// How [`init_subscriber`] builds the subscriber, beyond what the
/// [`EngineConfig`] says.
#[derive(Debug, Clone)]
pub struct SubscriberOptions<'a> {
    /// Log file name prefix, e.g. `"fink_fat"` → `fink_fat.<date>.log`.
    pub file_prefix: &'a str,
    /// Raw `TARGET=LEVEL` overrides, merged over the configuration's
    /// `log_targets` (an override wins on the same target).
    pub target_overrides: &'a [String],
    /// Days of log files kept; `None` uses the configuration's
    /// `log_retention_days`.
    pub retention_days: Option<usize>,
    /// Whether events are also printed to the terminal (with ANSI colors).
    pub stdout: bool,
}

/// Parse one `--log-target TARGET=LEVEL` argument into `(target, level)`.
///
/// # Errors
/// A human-readable message when the `=` is missing or the level unknown.
pub fn parse_log_target_override(arg: &str) -> Result<(String, LogLevel), String> {
    let (target, level) = arg
        .split_once('=')
        .ok_or_else(|| format!("invalid --log-target {arg:?}, expected TARGET=LEVEL"))?;
    let level: LogLevel = level
        .parse()
        .map_err(|e| format!("invalid --log-target {arg:?}: {e}"))?;
    Ok((target.to_string(), level))
}

/// Print every engine tracing target's name, levels and description, one
/// per line (the `--list-log-targets` output).
pub fn print_log_targets() {
    for target in all_targets() {
        println!(
            "{:<24} {:?}  {}",
            target.name, target.levels, target.description
        );
    }
}

/// `EnvFilter` directive of `engine_config`'s levels with `overrides` merged
/// over its `log_targets`.
///
/// # Errors
/// [`LogInitError::InvalidTargetOverride`] for a malformed override.
pub fn filter_directive(
    engine_config: &EngineConfig,
    overrides: &[String],
) -> Result<String, LogInitError> {
    let mut log_targets = engine_config.log_targets.clone();
    for arg in overrides {
        let (target, level) =
            parse_log_target_override(arg).map_err(LogInitError::InvalidTargetOverride)?;
        log_targets.insert(target, level);
    }
    Ok(build_env_filter_directive(
        engine_config.log_level,
        &log_targets,
    ))
}

/// Install the process-wide `tracing` subscriber.
///
/// Logs go to `storage_path` (next to the branch-collection snapshot, so
/// logs and pipeline state travel together), rotated daily; files older
/// than the retention are deleted automatically — with daily rotation, N
/// files kept is N days retained.
///
/// # Returns
/// The guard of the background file writer: keep it alive for the whole
/// run, dropping it flushes and stops the writer.
///
/// # Errors
/// See [`LogInitError`].
pub fn init_subscriber(
    engine_config: &EngineConfig,
    options: &SubscriberOptions<'_>,
) -> Result<tracing_appender::non_blocking::WorkerGuard, LogInitError> {
    let directive = filter_directive(engine_config, options.target_overrides)?;
    let filter = tracing_subscriber::EnvFilter::try_new(&directive)?;

    let retention_days = options
        .retention_days
        .unwrap_or(engine_config.log_retention_days);
    let file_appender = tracing_appender::rolling::RollingFileAppender::builder()
        .rotation(tracing_appender::rolling::Rotation::DAILY)
        .filename_prefix(options.file_prefix)
        .filename_suffix("log")
        .max_log_files(retention_days)
        .build(engine_config.storage_path())?;
    let (non_blocking_file, guard) = tracing_appender::non_blocking(file_appender);

    // Separate layers rather than one writer combining both destinations:
    // the file must never carry ANSI color escapes (they show up as garbage
    // in a plain-text log viewer), while the terminal should keep them.
    let file_layer = tracing_subscriber::fmt::layer()
        .with_target(true)
        .with_ansi(false)
        .with_writer(non_blocking_file);
    let stdout_layer = options
        .stdout
        .then(|| tracing_subscriber::fmt::layer().with_target(true));

    tracing_subscriber::registry()
        .with(filter)
        .with(stdout_layer)
        .with(file_layer)
        .try_init()?;
    Ok(guard)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_log_target_override_accepts_target_equals_level() {
        assert_eq!(
            parse_log_target_override("propagation=trace").unwrap(),
            ("propagation".to_string(), LogLevel::Trace)
        );
    }

    #[test]
    fn parse_log_target_override_rejects_malformed_input() {
        assert!(parse_log_target_override("propagation").is_err());
        assert!(parse_log_target_override("propagation=verbose").is_err());
    }

    #[test]
    fn overrides_win_over_the_configuration() {
        let mut config = EngineConfig::default();
        config.log_level = LogLevel::Warn;
        config
            .log_targets
            .insert("update".to_string(), LogLevel::Info);
        let directive = filter_directive(&config, &["update=trace".to_string()]).unwrap();
        assert_eq!(directive, "warn,update=trace");
    }

    #[test]
    fn a_malformed_override_is_reported() {
        let err = filter_directive(&EngineConfig::default(), &["nope".to_string()]).unwrap_err();
        assert!(matches!(err, LogInitError::InvalidTargetOverride(_)));
    }
}
