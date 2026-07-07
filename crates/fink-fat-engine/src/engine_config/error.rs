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
//! - Versioning errors (`UnsupportedVersion`): the config file declares a schema
//!   version the engine does not understand.
//! - Static “invalid config” markers (`Invalid`): used for invariants checked
//!   directly in [`EngineConfig::validate`] (e.g. `storage_path`,
//!   `healpix_depth`) that don't warrant a dedicated error type.
//! - Parameter-level validation failures for seeding (`Seed`): pairs/triplets
//!   parameter validation, propagated from [`SeedError`].
//!
//! This structure enables the top-level CLI / application to provide clear
//! user-facing messages such as:
//! - “YAML parsing error”
//! - “unsupported config version”
//! - “pairs.max_dt must be non-negative”
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
//!     cfg.pairs.validate()?;                        // may produce SeedError
//!     cfg.triplets.validate()?;                     // may produce SeedError
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
//!   constraints) use the dedicated `validate()` routines on `PairConfig` and
//!   `TripletConfig` and produce `ConfigError::Seed` for precise messages.

use thiserror::Error;

use config::ConfigError as ConfigRsError;

use crate::error::SeedError;

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
/// - [`ConfigError::UnsupportedVersion`]:
///   the configuration declares a schema version that this binary does not support.
/// - [`ConfigError::Invalid`]:
///   a coarse invalid-config marker for invariants that do not have a dedicated error.
/// - [`ConfigError::Seed`]:
///   seeding parameter validation failure (pairs/triplets).
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

    /// The configuration file declares a schema version not supported by this binary.
    ///
    /// This variant is typically emitted after reading a version field (e.g.
    /// `config_version`) but before attempting to interpret the rest of the file.
    #[error("unsupported config version: {0}")]
    UnsupportedVersion(u32),

    /// Coarse invalid-config marker.
    ///
    /// This is useful for simple invariants where creating a dedicated error
    /// type would not add much value.
    ///
    /// Notes
    /// -----
    /// The message is `'static` so it can be used as a stable identifier in tests
    /// or for downstream mapping to user-facing help.
    #[error("invalid config: {msg}")]
    Invalid { msg: String },

    /// Pairs/triplets configuration error.
    ///
    /// Produced by `PairConfig::validate()` / `TripletConfig::validate()` and other
    /// seeding-related validation routines.
    #[error("pairs/triplets config error: {0}")]
    Seed(#[from] SeedError),
}
