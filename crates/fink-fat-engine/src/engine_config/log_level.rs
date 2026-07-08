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
