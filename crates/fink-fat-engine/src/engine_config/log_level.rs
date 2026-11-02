//! Log-level configuration for the engine.
//!
//! A thin serde-friendly enum that maps to [`tracing::Level`] in the CLI layer.
//! The engine only stores this value; the subscriber is installed by the caller.
//!
//! YAML values must be written in **lowercase**: `"trace"`, `"debug"`,
//! `"info"`, `"warn"`, `"error"` (enforced by `#[serde(rename_all =
//! "lowercase")]`; mixed/upper case, e.g. `"Info"` or `"INFO"`, is rejected
//! as an unknown variant, not silently matched case-insensitively).

use serde::{Deserialize, Serialize};

use crate::engine_config::{Validate, error::FieldError};

/// Minimum tracing/log level that should be recorded.
///
/// This value is read from the `log_level` key in the YAML configuration file.
/// The CLI inspects it when initialising the tracing subscriber.
///
/// Defaults to [`LogLevel::Info`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    /// Most verbose level; every traced event, including hot-path internals.
    Trace,
    /// Verbose diagnostic events useful during development.
    Debug,
    /// Default level: coarse progress/status events.
    #[default]
    Info,
    /// Recoverable but noteworthy issues.
    Warn,
    /// Unrecoverable or user-facing failures.
    Error,
}

impl Validate for LogLevel {
    /// Always valid: serde already rejects any YAML value that isn't one of
    /// the five known variants, so there is no invalid post-deserialization
    /// state to check here. Implemented for uniformity with every other
    /// [`crate::engine_config::EngineConfig`] field.
    fn validate(&self) -> Result<(), Vec<FieldError>> {
        Ok(())
    }
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

/// Error returned by `LogLevel`'s [`FromStr`](std::str::FromStr) impl for an unrecognized level string.
#[derive(Debug, thiserror::Error)]
#[error("unknown log level {0:?} (expected one of: trace, debug, info, warn, error)")]
pub struct ParseLogLevelError(String);

impl std::str::FromStr for LogLevel {
    type Err = ParseLogLevelError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "trace" => Ok(LogLevel::Trace),
            "debug" => Ok(LogLevel::Debug),
            "info" => Ok(LogLevel::Info),
            "warn" => Ok(LogLevel::Warn),
            "error" => Ok(LogLevel::Error),
            other => Err(ParseLogLevelError(other.to_string())),
        }
    }
}

impl From<LogLevel> for tracing::Level {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Trace => tracing::Level::TRACE,
            LogLevel::Debug => tracing::Level::DEBUG,
            LogLevel::Info => tracing::Level::INFO,
            LogLevel::Warn => tracing::Level::WARN,
            LogLevel::Error => tracing::Level::ERROR,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_str_parses_every_variant() {
        assert_eq!("trace".parse::<LogLevel>().unwrap(), LogLevel::Trace);
        assert_eq!("debug".parse::<LogLevel>().unwrap(), LogLevel::Debug);
        assert_eq!("info".parse::<LogLevel>().unwrap(), LogLevel::Info);
        assert_eq!("warn".parse::<LogLevel>().unwrap(), LogLevel::Warn);
        assert_eq!("error".parse::<LogLevel>().unwrap(), LogLevel::Error);
    }

    #[test]
    fn from_str_rejects_unknown_string() {
        assert!("verbose".parse::<LogLevel>().is_err());
        assert!("INFO".parse::<LogLevel>().is_err());
    }
}
