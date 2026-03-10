//! Log-level configuration for the engine.
//!
//! A thin serde-friendly enum that maps to [`tracing::Level`] in the CLI layer.
//! The engine only stores this value; the subscriber is installed by the caller.
//!
//! YAML values (case-insensitive): `"trace"`, `"debug"`, `"info"`, `"warn"`, `"error"`.

use serde::{Deserialize, Serialize};

/// Minimum tracing/log level that should be recorded.
///
/// This value is read from the `log_level` key in the YAML configuration file.
/// The CLI inspects it when initialising the tracing subscriber.
///
/// Defaults to [`LogLevel::Info`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Trace,
    Debug,
    #[default]
    Info,
    Warn,
    Error,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            LogLevel::Trace => "trace",
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Warn => "warn",
            LogLevel::Error => "error",
        };
        f.write_str(s)
    }
}
