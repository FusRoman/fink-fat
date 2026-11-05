//! # Engine configuration (`engine_config`)
//!
//! This module is the engine's serde-based YAML configuration schema, rooted
//! at [`EngineConfig`]. Loading and validating a configuration file is a
//! three-stage pipeline:
//!
//! 1. **Load / merge** — [`EngineConfig::load_engine_config_validated`] builds a
//!    `config::Config` from the YAML file at the given path, overlaid by
//!    `FINK_FAT__`-prefixed environment variables (nested separator `__`),
//!    and deserializes the result into [`EngineConfig`]. There is
//!    deliberately no separate "Rust defaults" source merged in ahead of the
//!    YAML file — see that function's doc for why (short version: `config`
//!    merges nested tables key-by-key, which breaks overriding
//!    externally-tagged enum fields like `kfbank_config.cap_schedule` to a
//!    different variant than the default).
//! 2. **Deserialize** — every nested section uses `#[serde(default,
//!    deny_unknown_fields)]` so missing keys fall back to Rust defaults
//!    (purely at the serde level, independent of `config`'s own source
//!    merging) and unknown keys are rejected as YAML typos rather than
//!    silently ignored.
//! 3. **Validate** — [`EngineConfig::validate`] (via the [`Validate`] trait,
//!    implemented by every nested config struct/enum) checks numeric ranges
//!    and cross-field invariants not expressible through types alone,
//!    **accumulating every failure** instead of stopping at the first one;
//!    see [`error::ConfigError`] for the resulting error taxonomy.
//!
//! Submodules
//! ----------
//! - [`pair_config`] / [`triplet_config`]: intra-night pair/triplet seeding.
//! - [`kalman_context`] / [`single_kalman_config`]: shared ephemeris state
//!   and per-hypothesis Kalman filter tuning.
//! - [`kf_bank_config`] / [`hypothesis_cap`]: hypothesis-bank pruning/merging
//!   and the live-hypothesis-count decay schedule.
//! - [`grid_population`]: the `(ρ, ρ̇)` admissible-region seeding grid and
//!   its dynamical population priors.
//! - [`night_advance_params`]: tuning for advancing all banks by one night.
//! - [`log_level`]: the engine's tracing verbosity setting.
//! - [`error`]: the error types returned by the load/validate pipeline.
//! - [`units`]: human-friendly YAML unit parsing shared by the fields above
//!   (e.g. `"35 arcmin/day"`, `"86.4 min"`, `"0.02 au"`) — see that module's
//!   doc for the full list of supported quantities and unit tokens.

pub mod error;
pub mod grid_population;
pub mod hypothesis_cap;
pub mod kalman_context;
pub mod kf_bank_config;
pub mod log_level;
pub mod main_config;
pub mod night_advance_params;
pub mod pair_config;
pub mod single_kalman_config;
pub mod triplet_config;
pub mod units;
pub(crate) mod validate_helpers;

pub use main_config::EngineConfig;

use crate::engine_config::error::FieldError;

pub const CONFIGURATION_VERSION: u32 = 1;

/// Implemented by every nested config struct/enum reachable from
/// [`EngineConfig`] so that the whole configuration tree can be validated
/// uniformly.
///
/// Unlike a fail-fast validator, implementations are expected to
/// **accumulate every problem found** in `self` (and, when the type owns
/// nested [`Validate`] fields, in those too — with their field names
/// prefixed via [`crate::engine_config::error::prefix_errors`]) instead of
/// returning on the first failure. This lets a user fix every mistake in
/// their configuration file in one pass.
pub trait Validate {
    fn validate(&self) -> Result<(), Vec<FieldError>>;
}
