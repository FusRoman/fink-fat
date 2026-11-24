//! Structured logging for `fink-fat submit`, independent of
//! [`crate::logging::initialize_logs`] (which is `EngineConfig`/target-
//! registry-coupled, built for `track` — `submit` never loads an
//! `EngineConfig`, see [`mod@crate::submit`]'s module docs, so it can't reuse
//! that per-target directive machinery). `submit`'s logging surface is one
//! command with a handful of steps per lineage, so a single global level is
//! enough — no per-target overrides.

use camino::Utf8Path;
use fink_fat_engine::engine_config::log_level::LogLevel;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Initializes tracing for one `fink-fat submit` run.
///
/// # Arguments
/// * `logs` — whether logging is enabled at all (`--logs`). When `false`,
///   this is a no-op — `submit`'s per-lineage summary and final count still
///   print via `println!`, independent of tracing.
/// * `log_file` — also write logs here, in addition to stderr (appended to,
///   not rotated: a single CLI invocation's own log, unlike `track`'s
///   daily-rotated series).
/// * `level` — the global minimum level recorded.
///
/// # Return
/// A `WorkerGuard` that must be kept alive for the rest of the process for
/// the file layer's buffered writer to flush on drop, or `None` if `logs` is
/// `false` or no `log_file` was given.
///
/// # Errors
/// Returns the file appender's [`std::io::Error`] if `log_file`'s parent
/// directory doesn't exist or isn't writable.
pub fn init_submit_logging(
    logs: bool,
    log_file: Option<&Utf8Path>,
    level: LogLevel,
) -> Result<Option<tracing_appender::non_blocking::WorkerGuard>, std::io::Error> {
    if !logs {
        return Ok(None);
    }

    let level_filter = tracing_subscriber::filter::LevelFilter::from_level(level.into());
    // `fmt::layer()` defaults to ANSI *on* regardless of the writer — it does
    // not auto-detect a non-terminal destination itself. Without this check,
    // redirecting/piping stderr to a file (`fink-fat submit --logs >
    // out.log 2>&1`, a natural thing to do instead of `--log-file`) writes
    // the raw `\x1b[...m` escape codes into that file verbatim instead of
    // color, since nothing along that path interprets them.
    let stderr_is_tty = std::io::IsTerminal::is_terminal(&std::io::stderr());
    let stderr_layer = tracing_subscriber::fmt::layer()
        .with_target(true)
        .with_ansi(stderr_is_tty)
        .with_writer(std::io::stderr);

    let Some(log_file) = log_file else {
        tracing_subscriber::registry()
            .with(level_filter)
            .with(stderr_layer)
            .init();
        return Ok(None);
    };

    // A plain (non-rotating) appender: one `fink-fat submit` invocation, one
    // log file, appended to across runs against the same `--log-file` path.
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_file)?;
    let (non_blocking_file, guard) = tracing_appender::non_blocking(file);

    // Two layers, not one writer covering both destinations: the terminal
    // gets the human-readable colored format, the file gets structured JSON
    // (one object per line). This isn't just a style choice: `.with_ansi(false)`
    // alone does *not* fully suppress ANSI here — a long-standing
    // `tracing-subscriber` gap where `with_ansi(false)` fails to reach the
    // *span field* formatter (the `{lineage_designation=... endpoint=...}`
    // context printed after a span's name), only the event's own fields, so
    // every `#[instrument]`ed line still carried raw `\x1b[...m` escapes into
    // the file. `.json()` sidesteps the bug entirely (its formatter never
    // emits ANSI at all) and is arguably more useful for a log file meant to
    // be `grep`/`jq`-ed rather than eyeballed in a terminal.
    let file_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_target(true)
        .with_writer(non_blocking_file);

    tracing_subscriber::registry()
        .with(level_filter)
        .with(stderr_layer)
        .with(file_layer)
        .init();

    Ok(Some(guard))
}
