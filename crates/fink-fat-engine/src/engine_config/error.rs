use thiserror::Error;

use config::ConfigError as ConfigRsError;

use crate::error::{PredictorParamError, SeedError};

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("config loader error: {0}")]
    ConfigRs(#[from] ConfigRsError),

    #[error("unsupported config version: {0}")]
    UnsupportedVersion(u32),

    #[error("invalid config: {msg}")]
    Invalid { msg: &'static str },

    #[error("pairs/triplets config error: {0}")]
    Seed(#[from] SeedError),

    #[error("predictor config error: {0}")]
    Predictor(#[from] PredictorParamError),

    #[error("scoring config error: {0}")]
    Scoring(#[from] ScoringConfigError),

    #[error("edges config error: {0}")]
    Edges(#[from] EdgeConfigError),
}

#[derive(Debug, Error)]
pub enum EdgeConfigError {
    #[error("edges.top_k_per_left must be > 0")]
    TopKPerLeftZero,

    #[error("edges.max_total_edges must be > 0")]
    MaxTotalEdgesZero,
}

#[derive(Debug, thiserror::Error)]
pub enum ScoringConfigError {
    #[error("scoring.numeric.min_variance must be finite and >= 0 (got {value})")]
    InvalidMinVariance { value: f64 },

    #[error("scoring.position.max_d2 must be finite and > 0 (got {value})")]
    InvalidMaxD2Pos { value: f64 },

    #[error("scoring.velocity.max_theta must be finite and >= 0 (got {value})")]
    InvalidMaxThetaVel { value: f64 },

    #[error("scoring.velocity.max_speed_diff must be finite and >= 0 (got {value})")]
    InvalidMaxSpeedDiff { value: f64 },

    #[error("scoring: non-finite weight (got {field}={value})")]
    NonFiniteWeight { field: &'static str, value: f64 },

    #[error("scoring: invalid velocity scale (got {field}={value})")]
    InvalidVelocityScale { field: &'static str, value: f64 },

    #[error("scoring: invalid photometry scale (got {field}={value})")]
    InvalidPhotometryScale { field: &'static str, value: f64 },

    #[error("scoring: invalid gap scale (got {field}={value})")]
    InvalidGapScale { field: &'static str, value: f64 },
}
