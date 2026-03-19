//! Tracing subscriber initialisation for the `fink-fat` CLI.
//!
//! This module configures the process-wide `tracing` subscriber used by the
//! binary. It writes events to two destinations:
//!
//! - the terminal, either through [`indicatif::MultiProgress::println`] when
//!   progress bars are active or directly to stderr otherwise;
//! - a per-run log file under `<storage_root>/logs/run-<session_id>.log`.
//!
//! The terminal and file layers share the same log-level filter derived from
//! [`fink_fat_engine::engine_config::log_level::LogLevel`].
//!
//! ## Indicatif integration
//!
//! [`indicatif::MultiProgress`] temporarily takes control of the terminal while
//! progress bars are rendered. Direct writes to stdout or stderr can therefore
//! corrupt the display. This module avoids that problem by routing terminal log
//! lines through [`MultiProgress::println`] whenever a progress handle is
//! available.
//!
//! The [`TerminalMakeWriter`] wrapper provides a single monomorphic writer type
//! for both runtime modes. That keeps the subscriber composition simple and
//! avoids complex conditional layer types.

use std::{
    io::{self, Write},
    sync::Arc,
};

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

// ── Runtime-dispatching terminal writer ───────────────────────────────────────

/// Per-event byte buffer that flushes to the appropriate backend on drop.
///
/// The formatter writes one event at a time into this buffer and flushes the
/// bytes either to stderr or to [`MultiProgress::println`] when the writer is
/// dropped.
struct TerminalWriterGuard {
    /// When `Some`, lines go through `MultiProgress::println`; when `None`,
    /// the bytes are forwarded directly to stderr.
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
                // `MultiProgress::println` suspends bar rendering, prints the
                // line above the bars, and resumes – correct interleaving.
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
        // Ensure the line is printed even if the fmt layer never calls flush()
        // (which it does not on every event).
        let _ = self.flush();
    }
}

/// A [`fmt::MakeWriter`] that produces one [`TerminalWriterGuard`] per event.
///
/// This is the runtime switch between the plain-stderr path and the
/// `MultiProgress`-aware path.
///
/// Constructed once; cheaply cloned via `Arc` internally per writer call.
struct TerminalMakeWriter {
    /// `Some(mp)` → indicatif path, `None` → stderr path.
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

// ── Custom event formatter ─────────────────────────────────────────────────────

/// Visitor that splits event fields into the `message` pseudo-field and the
/// remaining structured key-value pairs.
///
/// `tracing` treats `message` specially. This visitor preserves the message
/// text separately from structured fields so the formatter can render compact
/// log lines without losing key-value context.
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
///
/// This keeps log targets readable while preserving enough context to locate
/// the emitting module.
///
/// ```text
/// fink_fat_engine::pipeline::stages::alert_inputs::alert_loader
///   → alert_inputs::alert_loader
/// fink_fat_engine::pipeline::stages
///   → pipeline::stages
/// fink_fat
///   → fink_fat   (unchanged, fewer than 2 separators)
/// ```
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
///
/// Returns empty strings when ANSI is disabled so the same formatter can render
/// coloured and plain output without branching at the call site.
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

/// Custom [`FormatEvent`] for `fink-fat` logs.
///
/// The formatter produces compact single-line entries with:
///
/// - a local timestamp,
/// - a fixed-width log level,
/// - a shortened target path, and
/// - any structured fields appended as `key=value` pairs.
///
/// ```text
/// 16:21:35.073 INFO  pipeline::stages: stage starting stage="Build seeds"
/// ```
/// - Timestamp: local `HH:MM:SS.ms` (no date, no timezone).
/// - Level: fixed 5-char width, optionally ANSI-coloured.
/// - Target: shortened to the last two `::` components.
/// - Blank line emitted before pipeline/stage boundary events.
struct FinkFatFormat {
    ansi: bool,
}

impl<S, N> FormatEvent<S, N> for FinkFatFormat
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
        // ── Collect fields ────────────────────────────────────────────────────
        let mut fields = EventFields {
            message: String::new(),
            extras: String::new(),
        };
        event.record(&mut fields);

        let meta = event.metadata();
        let level = *meta.level();
        let target = shorten_target(meta.target());

        // ── Blank line before pipeline / stage boundary events ────────────────
        let is_boundary = matches!(
            fields.message.as_str(),
            "stage starting" | "pipeline starting" | "pipeline complete"
        );
        if is_boundary {
            writer.write_char('\n')?;
        }

        // ── Timestamp: HH:MM:SS.ms (local time) ──────────────────────────────
        write!(writer, "{} ", Local::now().format("%H:%M:%S.%3f"))?;

        // ── Level (5 chars, optionally coloured) ─────────────────────────────
        let (col, reset) = level_style(level, self.ansi);
        let level_str = match level {
            Level::ERROR => "ERROR",
            Level::WARN => "WARN ",
            Level::INFO => "INFO ",
            Level::DEBUG => "DEBUG",
            Level::TRACE => "TRACE",
        };
        write!(writer, "{col}{level_str}{reset} ")?;

        // ── Shortened target ──────────────────────────────────────────────────
        write!(writer, "{target}: ")?;

        // ── Message + structured fields ───────────────────────────────────────
        write!(writer, "{}", fields.message)?;
        if !fields.extras.is_empty() {
            write!(writer, " {}", fields.extras)?;
        }
        writeln!(writer)
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

/// Initialise the global tracing subscriber with file and terminal output.
///
/// The subscriber installs two `fmt` layers that share the same level filter:
/// one for the terminal and one for the persistent run log file.
///
/// Arguments
/// ---------
/// * `level` — minimum event level to record, usually derived from
///   `EngineConfig::log_level`.
/// * `log_path` — absolute path to the per-run log file. The parent directory
///   is created if needed.
/// * `multi_progress` — when `Some`, terminal log lines are routed through
///   [`MultiProgress::println`] so they appear above active progress bars.
///   When `None`, logs are written directly to stderr.
///
/// Return
/// ------
/// * `Ok(LoggingGuard)` — logging was initialised successfully and the guard
///   must be retained until shutdown so the background writer can flush.
/// * `Err(Box<dyn std::error::Error>)` — log directory creation, file opening,
///   or global subscriber installation failed.
///
/// Notes
/// -----
/// - The file layer writes plain text without ANSI codes.
/// - The terminal layer writes with ANSI colours when a TTY is detected.
/// - The file appender runs on a dedicated background thread; its guard is
///   returned so the caller can control the flush at shutdown.
pub fn init_logging(
    level: Level,
    log_path: &Utf8Path,
    multi_progress: Option<Arc<MultiProgress>>,
) -> Result<LoggingGuard, Box<dyn std::error::Error>> {
    // ── Create log directory and open the file ────────────────────────────────
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;

    // Non-blocking writer: log events are queued in memory and written to the
    // file by a background thread.  The returned `WorkerGuard` flushes the
    // queue when dropped.
    let (non_blocking_file, guard) = tracing_appender::non_blocking(file);

    // ── Compose layers ────────────────────────────────────────────────────────
    // Using a single monomorphic `TerminalMakeWriter` avoids the type-level
    // complexity of trying to box or conditionally select different `fmt::Layer`
    // types together (which trips over tracing-subscriber's `Layered<…>` bounds).
    let level_filter = LevelFilter::from_level(level);

    let console_layer = fmt::Layer::new()
        .event_format(FinkFatFormat { ansi: true })
        .with_writer(TerminalMakeWriter { mp: multi_progress })
        .with_ansi(true);

    let file_layer = fmt::Layer::new()
        .event_format(FinkFatFormat { ansi: false })
        .with_writer(non_blocking_file)
        .with_ansi(false);

    let subscriber = tracing_subscriber::registry()
        .with(level_filter)
        .with(console_layer)
        .with(file_layer);

    tracing::subscriber::set_global_default(subscriber)?;

    Ok(LoggingGuard { _guard: guard })
}
