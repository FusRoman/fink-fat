/* ----------------------------- Errors ------------------------------ */

use thiserror::Error;

/// Parameter validation error for Fink-FAT configuration.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ParamError {
    /// A time interval must be finite and non-negative.
    #[error("invalid time parameter: {0}")]
    NonFiniteOrNegativeTime(&'static str),

    /// An angular separation must be finite and non-negative.
    #[error("invalid angle parameter: {0}")]
    NonFiniteOrNegativeAngle(&'static str),

    /// A residual must be finite and non-negative.
    #[error("invalid residual parameter: {0}")]
    NonFiniteOrNegativeResidual(&'static str),

    /// A flux/magnitude threshold must be finite and non-negative.
    #[error("invalid photometry parameter: {0}")]
    NonFiniteOrNegativePhotometry(&'static str),

    /// Parameters are mutually inconsistent.
    #[error("inconsistent parameter set: {0}")]
    Inconsistent(&'static str),
}
