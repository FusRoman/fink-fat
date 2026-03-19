//! Tracing subscriber initialisation for the fink-fat-eval CLI.
//!
//! Provides [`init_logging`] which wires up two output channels:
//!
//! 1. **Terminal** – plain stderr with ANSI colour codes.
//! 2. **Log file** – a per-run file at the path supplied by the caller,
//!    written through a non-blocking background thread.
//!
//! Logging is always active when the binary runs; no CLI flag is required.
//! The log level is derived from [`fink_fat_engine::engine_config::log_level::LogLevel`]
//! in the loaded engine configuration.

use std::io::{self, Write};
use std::sync::Arc;

use camino::Utf8Path;
use chrono::Local;
use indicatif::MultiProgress;
use tracing::{Event, Level, Subscriber};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{
    filter::LevelFilter,
    fmt::{
        self, FmtContext,
        format::{FormatEvent, FormatFields, Writer},
    },
    prelude::*,
    registry::LookupSpan,
};

// ── Custom event formatter ─────────────────────────────────────────────────────

/// Visitor that splits event fields into the `message` pseudo-field and
/// remaining structured key-value pairs.
struct EventFields {
    message: String,
    extras: String,
}

impl tracing::field::Visit for EventFields {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{value:?}");
        } else {
            if !self.extras.is_empty() {
                self.extras.push(' ');
            }
            let _ = std::fmt::write(
                &mut self.extras,
                format_args!("{}={:?}", field.name(), value),
            );
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            self.message = value.to_string();
        } else {
            if !self.extras.is_empty() {
                self.extras.push(' ');
            }
            let _ = std::fmt::write(
                &mut self.extras,
                format_args!("{}={:?}", field.name(), value),
            );
        }
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        if !self.extras.is_empty() {
            self.extras.push(' ');
        }
        let _ = std::fmt::write(&mut self.extras, format_args!("{}={}", field.name(), value));
    }

    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        if !self.extras.is_empty() {
            self.extras.push(' ');
        }
        let _ = std::fmt::write(&mut self.extras, format_args!("{}={}", field.name(), value));
    }

    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        if !self.extras.is_empty() {
            self.extras.push(' ');
        }
        let _ = std::fmt::write(&mut self.extras, format_args!("{}={}", field.name(), value));
    }

    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        if !self.extras.is_empty() {
            self.extras.push(' ');
        }
        let _ = std::fmt::write(&mut self.extras, format_args!("{}={}", field.name(), value));
    }
}

/// Shorten a fully-qualified module path to its last two `::` components.
fn shorten_target(target: &str) -> &str {
    let bytes = target.as_bytes();
    let mut sep_seen = 0usize;
    let mut pos = target.len();
    while pos >= 2 {
        if bytes[pos - 2] == b':' && bytes[pos - 1] == b':' {
            sep_seen += 1;
            if sep_seen == 2 {
                return &target[pos..];
            }
            pos -= 2;
        } else {
            pos -= 1;
        }
    }
    target
}

/// ANSI colour and bold codes for log levels.
fn level_style(level: Level, ansi: bool) -> (&'static str, &'static str) {
    if !ansi {
        return ("", "");
    }
    match level {
        Level::ERROR => ("\x1b[1;31m", "\x1b[0m"),
        Level::WARN => ("\x1b[1;33m", "\x1b[0m"),
        Level::INFO => ("\x1b[1;32m", "\x1b[0m"),
        Level::DEBUG => ("\x1b[1;34m", "\x1b[0m"),
        Level::TRACE => ("\x1b[2;37m", "\x1b[0m"),
    }
}

/// Custom [`FormatEvent`] for fink-fat-eval logs.
///
/// Produces compact, readable log lines:
/// ```text
/// 16:21:35.073 INFO  pipeline::stages: stage starting stage="Build seeds"
/// ```
struct FinkFatEvalFormat {
    ansi: bool,
}

impl<S, N> FormatEvent<S, N> for FinkFatEvalFormat
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        _ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> std::fmt::Result {
        let mut fields = EventFields {
            message: String::new(),
            extras: String::new(),
        };
        event.record(&mut fields);

        let meta = event.metadata();
        let level = *meta.level();
        let target = shorten_target(meta.target());

        let is_boundary = matches!(
            fields.message.as_str(),
            "stage starting" | "pipeline starting" | "pipeline complete"
        );
        if is_boundary {
            writer.write_char('\n')?;
        }

        write!(writer, "{} ", Local::now().format("%H:%M:%S.%3f"))?;

        let (col, reset) = level_style(level, self.ansi);
        let level_str = match level {
            Level::ERROR => "ERROR",
            Level::WARN => "WARN ",
            Level::INFO => "INFO ",
            Level::DEBUG => "DEBUG",
            Level::TRACE => "TRACE",
        };
        write!(writer, "{col}{level_str}{reset} ")?;
        write!(writer, "{target}: ")?;
        write!(writer, "{}", fields.message)?;
        if !fields.extras.is_empty() {
            write!(writer, " {}", fields.extras)?;
        }
        writeln!(writer)
    }
}

// ── Terminal writer (MultiProgress-aware) ────────────────────────────────────

/// Buffers a single log line and flushes it through [`MultiProgress::println`]
/// when available, falling back to raw stderr.
struct TerminalWriterGuard {
    mp: Option<Arc<MultiProgress>>,
    buf: Vec<u8>,
}

impl io::Write for TerminalWriterGuard {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buf.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        match &self.mp {
            Some(mp) => {
                let s = String::from_utf8_lossy(&self.buf);
                let line = s.trim_end_matches('\n');
                if !line.is_empty() {
                    let _ = mp.println(line);
                }
            }
            None => {
                io::stderr().write_all(&self.buf)?;
            }
        }
        self.buf.clear();
        Ok(())
    }
}

impl Drop for TerminalWriterGuard {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

struct TerminalMakeWriter {
    mp: Option<Arc<MultiProgress>>,
}

impl<'a> fmt::MakeWriter<'a> for TerminalMakeWriter {
    type Writer = TerminalWriterGuard;

    fn make_writer(&'a self) -> Self::Writer {
        TerminalWriterGuard {
            mp: self.mp.clone(),
            buf: Vec::new(),
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Opaque guard returned by [`init_logging`].
///
/// **Keep this value alive** for the duration of the program's execution.
/// Dropping it flushes and closes the background file-writer thread.
pub struct LoggingGuard {
    _guard: WorkerGuard,
}

/// Initialise the global tracing subscriber with file + stderr output.
///
/// Arguments
/// ---------
/// * `level` – Minimum event level to record.
/// * `log_path` – Absolute path for the per-run log file (created if absent).
///
/// Return
/// ------
/// * `Ok(LoggingGuard)` – Guard that must be kept alive until process exit.
/// * `Err(...)` – If the log directory cannot be created, the file cannot be
///   opened, or a global subscriber has already been installed.
///
/// Notes
/// -----
/// - The file layer writes plain text without ANSI codes.
/// - The terminal layer writes with ANSI colours when a TTY is detected.
/// - The file appender runs on a dedicated background thread.
pub fn init_logging(
    level: Level,
    log_path: &Utf8Path,
    multi_progress: Option<Arc<MultiProgress>>,
) -> Result<LoggingGuard, Box<dyn std::error::Error>> {
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;

    let (non_blocking_file, guard) = tracing_appender::non_blocking(file);

    let level_filter = LevelFilter::from_level(level);

    let stderr_layer = fmt::Layer::new()
        .event_format(FinkFatEvalFormat { ansi: true })
        .with_writer(TerminalMakeWriter { mp: multi_progress })
        .with_ansi(true);

    let file_layer = fmt::Layer::new()
        .event_format(FinkFatEvalFormat { ansi: false })
        .with_writer(non_blocking_file)
        .with_ansi(false);

    let subscriber = tracing_subscriber::registry()
        .with(level_filter)
        .with(stderr_layer)
        .with(file_layer);

    tracing::subscriber::set_global_default(subscriber)?;

    Ok(LoggingGuard { _guard: guard })
}
