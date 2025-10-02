//! # Parameter Errors
//!
//! This module defines [`ParamError`], the error type used to report invalid
//! or inconsistent configuration parameters in **Fink-FAT**.
//!
//! ## Overview
//!
//! During initialization or runtime checks, configuration values such as
//! time intervals, angular separations, photometric thresholds, or residual
//! cutoffs are validated. If a value is non-finite, negative, or mutually
//! inconsistent with other parameters, a [`ParamError`] is returned.
//!
//! ## Design
//!
//! - Errors are grouped by *parameter class* (time, angle, residual,
//!   photometry, consistency).
//! - Each variant carries a static string giving the context, usually the
//!   name of the offending parameter (e.g. `"pair.max_dt"`).
//! - The type derives [`thiserror::Error`] for friendly display strings,
//!   and implements `Clone`, `PartialEq`, `Eq` for testability.
//!
//! ## Example
//!
//! ```rust
//! use fink_fat::errors::ParamError;
//!
//! fn validate_sep(sep: f64) -> Result<(), ParamError> {
//!     if !sep.is_finite() || sep < 0.0 {
//!         return Err(ParamError::NonFiniteOrNegativeAngle("pair.max_sep"));
//!     }
//!     Ok(())
//! }
//!
//! // Invalid separation → error
//! assert!(validate_sep(f64::NAN).is_err());
//!
//! // Valid separation → ok
//! assert!(validate_sep(0.05).is_ok());
//! ```

use thiserror::Error;

/// Error type for invalid or inconsistent parameters in Fink-FAT.
///
/// This enum captures validation failures when constructing or using
/// [`FinkFatParams`](crate::params::FinkFatParams) and related configuration
/// structures. Each variant points to a specific class of error such as
/// non-finite values, negative thresholds, or mutually inconsistent settings.
///
/// Typical usage
/// -------------
/// Validation routines return a `Result<T, ParamError>`. Errors can be matched
/// to handle specific cases:
///
/// ```rust
/// use fink_fat::errors::ParamError;
///
/// fn validate_time(dt: f64) -> Result<(), ParamError> {
///     if !dt.is_finite() || dt < 0.0 {
///         return Err(ParamError::NonFiniteOrNegativeTime("pair.max_dt"));
///     }
///     Ok(())
/// }
///
/// assert!(validate_time(-1.0).is_err());
/// ```
///
/// Notes
/// -----
/// - All variants carry a `&'static str` giving context (usually the parameter
///   name such as `"pair.max_dt"` or `"triplet.max_sep"`).
/// - Errors are `Clone`, `PartialEq`, and `Eq` to simplify testing.
/// - The [`thiserror::Error`] derive provides user-friendly display strings.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ParamError {
    /// A time interval must be finite and non-negative.
    ///
    /// Triggered when parameters like `pair.max_dt` or `triplet.max_dt_between`
    /// are NaN, infinite, or strictly negative.
    #[error("invalid time parameter: {0}")]
    NonFiniteOrNegativeTime(&'static str),

    /// An angular separation must be finite and non-negative.
    ///
    /// Raised for invalid angle-like parameters such as `pair.max_sep` or
    /// `triplet.max_pair_sep`.
    #[error("invalid angle parameter: {0}")]
    NonFiniteOrNegativeAngle(&'static str),

    /// A residual must be finite and non-negative.
    ///
    /// Used when trajectory-fitting residual thresholds are NaN, infinite, or
    /// below zero.
    #[error("invalid residual parameter: {0}")]
    NonFiniteOrNegativeResidual(&'static str),

    /// A flux or magnitude threshold must be finite and non-negative.
    ///
    /// Raised for photometric cutoffs such as `pair.max_flux_difference`.
    #[error("invalid photometry parameter: {0}")]
    NonFiniteOrNegativePhotometry(&'static str),

    /// Parameters are mutually inconsistent.
    ///
    /// Returned when individual parameters are valid in isolation, but conflict
    /// with each other when combined. For example:
    /// - `triplet.max_dt_between` < `pair.max_dt`,
    /// - required HEALPix depth outside allowed range,
    /// - conflicting filter rules.
    #[error("inconsistent parameter set: {0}")]
    Inconsistent(&'static str),
}
