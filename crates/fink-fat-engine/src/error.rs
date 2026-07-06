use std::io;

use hifitime::HifitimeError;
use outfit::OutfitError;
use photom::observation_dataset::{ObsDatasetError, ObsId};
use thiserror::Error;

use crate::{
    engine_config::error::ConfigError,
    pipeline::PipelineStage,
    seeding::error::SeedingError,
    tracklet::track_storage::TrackId, // graph::edge::error::EdgeBuilderError,
                                      // persistence::error::{PersistenceError, PersistenceIoError},
                                      // pipeline::stages::PipelineStage,
                                      // solver::{components::error::ComponentError, error::SolverError},
};

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

    #[error("Invalid Night window: {0}")]
    InvalidNightWindow(String),
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
    /// max_cone_radius must be finite and strictly positive when set.
    #[error("invalid max_cone_radius = {0:?} (expect finite and > 0)")]
    InvalidMaxConeRadius(f64),
    /// max_norm_offset must be finite and strictly positive when set.
    #[error("invalid max_norm_offset = {0:?} (expect finite and > 0)")]
    InvalidMaxNormOffset(f64),
}

#[derive(Debug, Error)]
pub enum EngineError {
    /// The pipeline plan is invalid (stages, window, invariants, etc).
    #[error("invalid pipeline plan: {0}")]
    InvalidPlan(&'static str),

    /// A pipeline stage failed with a message that is specific but not (yet) typed.
    ///
    /// This is useful as a temporary catch-all while the project evolves.
    #[error("stage {stage:?} failed: {message}")]
    StageFailed {
        stage: PipelineStage,
        message: String,
    },

    /// Pipeline was cancelled by the caller (hooks).
    #[error("pipeline cancelled")]
    Cancelled,

    // -------------------------------------------------------------------------
    // Wrappers for lower-level subsystems (add as you wire real calls)
    // -------------------------------------------------------------------------
    /// Seeding subsystem error.
    #[error(transparent)]
    Seed(#[from] SeedError),

    /// Edge building / ML inference error.
    // #[error(transparent)]
    // Edge(#[from] EdgeBuilderError),

    // /// Persistence I/O (manifest, stores, journals, etc).
    // #[error(transparent)]
    // Persistence(#[from] PersistenceError),

    /// Generic engine error (if you already use `FinkFatError` as a top-level error).
    #[error(transparent)]
    FinkFat(#[from] FinkFatError),

    /// Component error
    // #[error(transparent)]
    // Component(#[from] ComponentError),

    // /// Solver error
    // #[error(transparent)]
    // Solver(#[from] SolverError),

    /// Outfit error
    #[error(transparent)]
    Outfit(#[from] OutfitError),

    /// Orbit fitting error
    #[error("orbit fitting error: {0}")]
    OrbitFitting(String),

    /// Config validation error
    #[error(transparent)]
    Config(#[from] ConfigError),

    /// ObsDataset error comming from the `photom` crate.
    #[error(transparent)]
    ObsDataset(#[from] ObsDatasetError),

    /// Time scale error from the `hifitime` crate (e.g. UT1 provider issues).
    #[error(transparent)]
    Ut1(#[from] HifitimeError),

    #[error("obs dataset id not found : {0}")]
    ObsDatasetIdNotFound(ObsId),

    #[error("Attempt to get TrackletData on Orbit variant tracklet : {0:?}")]
    NotTrackletVariant(TrackId),

    #[error("{0:?}")]
    TrackIdNotFound(TrackId),

    #[error(transparent)]
    Sedding(#[from] SeedingError),

    #[error(transparent)]
    TopoError(#[from] TopocentricRangeError),
}

#[derive(Debug, Error)]
pub enum ObservationJacobianError {
    #[error("topocentric distance ρ = {rho:.2e} is below numerical threshold")]
    TopocentricDistanceTooSmall { rho: f64 },

    #[error("polar singularity: Dec = {dec:.6} rad, cos(Dec) is near zero")]
    PolarSingularity { dec: f64 },
}

/// Errors that can occur during a [`KFState::update`] step.
#[derive(Debug, thiserror::Error)]
pub enum KFUpdateError {
    /// The topocentric distance $\rho$ is below the numerical threshold.
    #[error("topocentric distance too small: rho = {rho:.3e} AU")]
    TopocentricDistanceTooSmall { rho: f64 },

    /// The observation Jacobian could not be evaluated.
    #[error("observation Jacobian error: {0}")]
    Jacobian(#[from] ObservationJacobianError),

    /// The innovation covariance matrix $S$ is singular and cannot be inverted.
    #[error("innovation covariance matrix S is singular")]
    SingularInnovationCovariance,
}

#[derive(Debug, Error)]
pub enum TopocentricRangeError {
    #[error(
        "Negative discriminant (Δ = {discriminant:.6e}): the heliocentric distance prior \
         r = {r:.6} AU is geometrically inconsistent with the observer position \
         |r_obs| = {r_obs_norm:.6} AU at solar elongation φ = {phi_deg:.3}°. \
         No real topocentric range exists."
    )]
    NegativeDiscriminant {
        discriminant: f64,
        r: f64,
        r_obs_norm: f64,
        phi_deg: f64,
    },

    #[error(
        "No positive root found (rho1 = {rho1:.6e} AU, rho2 = {rho2:.6e} AU): \
         both solutions to the Al-Kashi quadratic are negative or zero. \
         The observer may be beyond the asteroid for the given prior r = {r:.6} AU."
    )]
    NoPositiveRoot { rho1: f64, rho2: f64, r: f64 },
}

pub trait OptionExt<T> {
    fn stage_err(self, stage: PipelineStage, message: impl Into<String>) -> Result<T, EngineError>;
}

impl<T> OptionExt<T> for Option<T> {
    fn stage_err(self, stage: PipelineStage, message: impl Into<String>) -> Result<T, EngineError> {
        self.ok_or_else(|| EngineError::StageFailed {
            stage,
            message: message.into(),
        })
    }
}
