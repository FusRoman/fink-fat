use fink_fat_engine::{
    engine_config::{EngineConfig, log_level::LogLevel},
    logging::registry::{all_targets, build_env_filter_directive},
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::init_cli::FinkFatCliArgs;

/// Parse one `--log-target TARGET=LEVEL` argument into `(target, level)`.
pub fn parse_log_target_override(arg: &str) -> Result<(String, LogLevel), String> {
    let (target, level) = arg
        .split_once('=')
        .ok_or_else(|| format!("invalid --log-target {arg:?}, expected TARGET=LEVEL"))?;
    let level: LogLevel = level
        .parse()
        .map_err(|e| format!("invalid --log-target {arg:?}: {e}"))?;
    Ok((target.to_string(), level))
}

/// Print every tracing target's name, description and levels.
pub fn print_log_targets() {
    for target in all_targets() {
        println!(
            "{:<24} {:?}  {}",
            target.name, target.levels, target.description
        );
    }
}

pub fn initialize_logs(
    cli: &FinkFatCliArgs,
    engine_config: &EngineConfig,
    _file_log_guard: &mut Option<tracing_appender::non_blocking::WorkerGuard>,
) -> Result<(), Box<dyn std::error::Error>> {
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

    *_file_log_guard = Some(guard);

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
    Ok(())
}
