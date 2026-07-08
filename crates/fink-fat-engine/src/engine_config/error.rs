//! # Engine configuration error types (`ConfigError`)
//!
//! This module defines the error type used by the **engine configuration layer**.
//!
//! The configuration system has three stages:
//!
//! 1) **Loading / merging** configuration sources (YAML file, environment
//!    overrides, Rust defaults) using the `config` crate — see
//!    [`crate::engine_config::load_engine_config_validated`].
//! 2) **Deserialization** into strongly-typed Rust structs (e.g. `PairConfig`,
//!    `TripletConfig`, `KFBankConfig`).
//! 3) **Validation** of numeric ranges and cross-field invariants — see
//!    [`crate::engine_config::EngineConfig::validate`].
//!
//! The error in this module is designed to preserve enough context so the
//! caller can decide whether the failure is:
//! - an I/O / parsing problem (cannot load the config),
//! - a schema mismatch (unsupported version),
//! - a user mistake (invalid value, wrong unit string, unknown field),
//! - or a higher-level invariant violation detected by explicit validation.
//!
//! -----------------------------------------------------------------------------
//! Error taxonomy
//! -----------------------------------------------------------------------------
//!
//! [`ConfigError`] is the high-level error enum returned by the configuration
//! loader / validator. It groups failures by their origin:
//!
//! - Loader failures from the `config` crate (`ConfigRs`): missing file, invalid YAML,
//!   environment override parsing errors, type mismatches during
//!   deserialization (this is also where `deny_unknown_fields` violations and
//!   `engine_config::units` unit-parsing failures surface, since both happen
//!   during deserialization).
//! - Semantic validation failures (`Validation`): every [`FieldError`]
//!   collected across the whole configuration tree by
//!   [`crate::engine_config::Validate`] implementations (schema version,
//!   numeric ranges, cross-field consistency, ...), wrapped in
//!   [`ValidationErrors`] for pretty multi-error display.
//!
//! This structure enables the top-level CLI / application to provide clear
//! user-facing messages such as:
//! - “YAML parsing error”
//! - “version: unsupported schema version 2 → hint: set version: 1”
//! - “pairs.max_dt: must be finite and non-negative, got -0.01”
//!
//! -----------------------------------------------------------------------------
//! Propagation and `#[from]` conversions
//! -----------------------------------------------------------------------------
//!
//! `ConfigError` uses `thiserror` and relies on `#[from]` to support
//! ergonomic propagation with `?`.
//!
//! Typical usage:
//!
//! ```rust, ignore
//! fn load_and_validate() -> Result<EngineConfig, ConfigError> {
//!     let cfg: EngineConfig = loader.load()?;      // may produce ConfigRsError
//!     cfg.validate()                                // Vec<FieldError>, accumulated
//!         .map_err(|errs| ConfigError::Validation(ValidationErrors(errs)))?;
//!     Ok(cfg)
//! }
//! ```
//!
//! -----------------------------------------------------------------------------
//! Notes for user-facing diagnostics
//! -----------------------------------------------------------------------------
//!
//! - Deserialization failures caused by `deny_unknown_fields` on config structs
//!   are surfaced as `ConfigError::ConfigRs` during the load/deserialize step.
//! - Unit parsing failures (e.g. `"35 foounit/day"`) from `engine_config::units`
//!   also surface as deserialization errors and are therefore wrapped by
//!   `ConfigError::ConfigRs`.
//! - Post-deserialization semantic checks (finite / non-negative / cross-field
//!   constraints, on every nested config struct) are performed by each type's
//!   [`crate::engine_config::Validate`] implementation and accumulated — see
//!   [`FieldError`] and [`ValidationErrors`].

use thiserror::Error;

use config::ConfigError as ConfigRsError;

/// A single, actionable configuration validation failure.
///
/// Produced by [`crate::engine_config::Validate::validate`] implementations.
/// Unlike a plain error message, a [`FieldError`] always identifies *which*
/// field is wrong (`field`, a dotted path such as `"kfbank_config.gate_chi2"`
/// built up by [`prefix_errors`] as errors bubble up through nested structs)
/// and, where possible, *how to fix it* (`hint`).
#[derive(Debug, Clone, Error)]
#[error("{field}: {message}")]
pub struct FieldError {
    /// Dotted path to the offending field, e.g. `"kfbank_config.gate_chi2"`.
    pub field: String,
    /// What is wrong with the current value (should include the observed value).
    pub message: String,
    /// Optional actionable suggestion on how to fix it.
    pub hint: Option<String>,
}

impl FieldError {
    /// Create a new field error without a hint.
    pub fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
            hint: None,
        }
    }

    /// Attach an actionable suggestion on how to fix this error.
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }
}

/// Prepend `prefix` to the `field` of every error, joined by `.`.
///
/// Used by every struct that owns nested [`crate::engine_config::Validate`]
/// fields to turn a child's locally-scoped field name (e.g. `"gate_chi2"`)
/// into a fully-qualified path (e.g. `"kfbank_config.gate_chi2"`) as errors
/// are collected up the tree.
pub fn prefix_errors(errors: Vec<FieldError>, prefix: &str) -> Vec<FieldError> {
    errors
        .into_iter()
        .map(|e| FieldError {
            field: format!("{prefix}.{}", e.field),
            ..e
        })
        .collect()
}

/// Every [`FieldError`] collected while validating an [`EngineConfig`](crate::engine_config::EngineConfig).
///
/// Unlike a fail-fast validator, [`crate::engine_config::Validate`]
/// implementations accumulate *all* problems found in a configuration tree
/// instead of stopping at the first one, so a user can fix every mistake in
/// one pass instead of playing whack-a-mole with repeated reloads.
#[derive(Debug)]
pub struct ValidationErrors(pub Vec<FieldError>);

impl std::fmt::Display for ValidationErrors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "configuration is invalid ({} error{}):",
            self.0.len(),
            if self.0.len() == 1 { "" } else { "s" }
        )?;
        for (i, e) in self.0.iter().enumerate() {
            writeln!(f, "  {}. {}: {}", i + 1, e.field, e.message)?;
            if let Some(hint) = &e.hint {
                writeln!(f, "     \u{2192} {hint}")?;
            }
        }
        Ok(())
    }
}

impl std::error::Error for ValidationErrors {}

/// Top-level configuration error returned by config loading and validation.
///
/// This error type is intended to be the primary return type of an engine
/// configuration loader. It covers both:
/// - failures while loading/merging/parsing configuration sources, and
/// - failures after deserialization when validating semantic invariants.
///
/// Variants
/// --------
/// - [`ConfigError::ConfigRs`]:
///   error originating from the `config` crate (I/O, parsing, merging, etc.).
/// - [`ConfigError::Validation`]:
///   one or more semantic invariants failed, collected by
///   [`crate::engine_config::Validate`] implementations across the whole
///   configuration tree (see [`ValidationErrors`]).
#[derive(Debug, Error)]
pub enum ConfigError {
    /// Error produced by the `config` crate while loading configuration sources.
    ///
    /// This includes (non-exhaustive):
    /// - missing or unreadable configuration file,
    /// - YAML/TOML/JSON syntax errors,
    /// - environment override parse errors,
    /// - type mismatch during deserialization.
    #[error("config loader error: {0}")]
    ConfigRs(#[from] ConfigRsError),

    /// One or more semantic invariants failed post-deserialization.
    ///
    /// Produced by [`crate::engine_config::EngineConfig::validate`], which
    /// accumulates every [`FieldError`] found across the whole configuration
    /// tree (schema version, numeric ranges, cross-field consistency, ...)
    /// instead of stopping at the first failure.
    #[error("{0}")]
    Validation(#[from] ValidationErrors),
}
