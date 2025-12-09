use std::io;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SeedError {
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

/// Unified error type for the fink-fat engine.
///
/// Notes
/// -----
/// This error groups together:
/// - filesystem I/O errors,
/// - JSON serialization errors (night summaries),
/// - bincode encoding/decoding errors (seeds.bin),
/// - custom domain-level errors.
///
/// More fine-grained variants can be added as the project evolves.
#[derive(Error, Debug)]
pub enum FinkFatError {
    /// Wrapper for standard I/O errors.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// JSON (serde) serialization or deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// bincode 2.x encode error.
    #[error("Bincode encoding error: {0}")]
    BincodeEncode(String),

    /// bincode 2.x decode error.
    #[error("Bincode decoding error: {0}")]
    BincodeDecode(String),

    /// General domain-level errors with human-readable context.
    #[error("fink-fat error: {0}")]
    Message(String),
}

impl From<bincode::error::EncodeError> for FinkFatError {
    fn from(err: bincode::error::EncodeError) -> Self {
        Self::BincodeEncode(err.to_string())
    }
}

impl From<bincode::error::DecodeError> for FinkFatError {
    fn from(err: bincode::error::DecodeError) -> Self {
        Self::BincodeDecode(err.to_string())
    }
}

impl From<String> for FinkFatError {
    fn from(msg: String) -> Self {
        Self::Message(msg)
    }
}

impl From<&str> for FinkFatError {
    fn from(msg: &str) -> Self {
        Self::Message(msg.to_string())
    }
}

/* -------------------------------------------------------------------------- */
/*  Predictor Errors                                                          */
/* -------------------------------------------------------------------------- */

/// Parameter validation errors for the predictor builder.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum PredictorParamError {
    /// k_sigma must be finite and strictly positive.
    #[error("invalid k_sigma = {0:?} (expect finite and > 0)")]
    InvalidKSigma(f64),
    /// Noise coefficients must be finite and non-negative.
    #[error("invalid noise coefficient `{name}` = {value:?} (expect finite and >= 0)")]
    InvalidNoiseCoeff { name: &'static str, value: f64 },
}
