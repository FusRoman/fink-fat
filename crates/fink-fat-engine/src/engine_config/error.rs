//! # Engine configuration error types (`ConfigError`, `EdgeConfigError`)
//!
//! This module defines the error types used by the **engine configuration layer**.
//!
//! The configuration system typically has three stages:
//!
//! 1) **Loading / merging** configuration sources (files, environment overrides, defaults)
//!    using the `config` crate.
//! 2) **Deserialization** into strongly-typed Rust structs (e.g. `PairConfig`,
//!    `TripletConfig`, `PredictorParams`, `EdgeConfig`).
//! 3) **Validation** of numeric ranges and cross-field invariants.
//!
//! The errors in this module are designed to preserve enough context so the
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
//! ## [`ConfigError`]
//!
//! High-level error enum returned by the configuration loader / validator.
//! It groups failures by their origin:
//!
//! - Loader failures from the `config` crate (`ConfigRs`): missing file, invalid YAML,
//!   environment override parsing errors, etc.
//! - Versioning errors (`UnsupportedVersion`): the config file declares a schema
//!   version the engine does not understand.
//! - Static “invalid config” markers (`Invalid`): used when a particular invariant
//!   is violated but does not fit a more specialized error type.
//! - Parameter-level validation failures for:
//!   - seeding (`Seed`): pairs/triplets parameter validation,
//!   - propagation predictor (`Predictor`): predictor parameters / noise schedule,
//!   - edge construction (`Edges`): edge configuration constraints.
//!
//! This structure enables the top-level CLI / application to provide clear
//! user-facing messages such as:
//! - “YAML parsing error”
//! - “unsupported config version”
//! - “pairs.max_dt must be non-negative”
//! - “edges.top_k_per_left must be > 0”
//!
//! ## [`EdgeConfigError`]
//!
//! Specialized error enum for edge configuration validation.
//! This is separated so the edge subsystem can evolve its own invariants without
//! making `ConfigError` too large or too coupled to edge internals.
//!
//! -----------------------------------------------------------------------------
//! Propagation and `#[from]` conversions
//! -----------------------------------------------------------------------------
//!
//! Both error enums use `thiserror` and rely heavily on `#[from]` to support
//! ergonomic propagation with `?`.
//!
//! Typical usage:
//!
//! ```rust, ignore
//! fn load_and_validate() -> Result<EngineConfig, ConfigError> {
//!     let cfg: EngineConfig = loader.load()?;      // may produce ConfigRsError
//!     cfg.pairs.validate()?;                        // may produce SeedError
//!     cfg.triplets.validate()?;                     // may produce SeedError
//!     cfg.edges.validate()?;                        // may produce EdgeConfigError
//!     cfg.edges.predictor_config.validate()?;       // may produce PredictorParamError
//!     Ok(cfg)
//! }
//! ```
//!
//! -----------------------------------------------------------------------------
//! Notes for user-facing diagnostics
//! -----------------------------------------------------------------------------
//!
//! - Deserialization failures caused by `deny_unknown_fields` on config structs
//!   are usually surfaced as `ConfigRsError` during the load/deserialize step.
//! - Unit parsing failures (e.g. `"35 foounit/day"`) from `engine_config::units`
//!   also surface as deserialization errors and are therefore typically wrapped
//!   by `ConfigRsError`.
//! - Post-deserialization semantic checks (finite / non-negative / cross-field
//!   constraints) should use the dedicated `validate()` routines and produce
//!   `SeedError`, `PredictorParamError`, or `EdgeConfigError` for precise messages.

use thiserror::Error;

use config::ConfigError as ConfigRsError;

use crate::{
    error::{PredictorParamError, SeedError},
    graph::edge::error::EdgeModelError,
};

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
/// - [`ConfigError::Predictor`]:
///   predictor parameter validation failure (kσ, noise coefficients, etc.).
/// - [`ConfigError::Edges`]:
///   edge construction configuration validation failure.
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

    /// Propagation predictor configuration error.
    ///
    /// Produced by `PredictorParams::validate()` and related builder checks.
    #[error("predictor config error: {0}")]
    Predictor(#[from] PredictorParamError),

    /// Inter-night edge configuration error.
    ///
    /// Produced by `EdgeConfig::validate()` and other edge-related invariants.
    #[error("edges config error: {0}")]
    Edges(#[from] EdgeConfigError),
}

/// Edge configuration validation errors.
///
/// This error enum groups semantic constraints specific to the inter-night edge
/// construction subsystem.
#[derive(Debug, Error)]
pub enum EdgeConfigError {
    /// `edges.top_k_per_left` must be strictly positive.
    ///
    /// In ML Top-K mode, a value of `0` would result in emitting zero edges,
    /// silently disabling linking. Enforcing `> 0` keeps the configuration
    /// unambiguous.
    #[error("edges.top_k_per_left must be > 0")]
    TopKPerLeftZero,

    /// `edges.max_total_edges` must be strictly positive.
    ///
    /// This constraint is relevant if the edge subsystem supports a global
    /// cap on the total number of emitted edges (not shown in the `EdgeConfig`
    /// snippet). The validator emits this error when that cap is configured
    /// but set to `0`.
    #[error("edges.max_total_edges must be > 0")]
    MaxTotalEdgesZero,

    /// `edges.predictor_config` is invalid.
    ///
    /// This error wraps any validation failure from the predictor configuration,
    /// such as invalid `k_sigma` or noise parameters.
    #[error("edges.predictor_config error: {0}")]
    PredictorConfig(#[from] PredictorParamError),

    /// Edge Model Error
    #[error(transparent)]
    EdgeModel(#[from] EdgeModelError),
}
