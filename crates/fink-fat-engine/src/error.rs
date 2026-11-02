use std::io;

use hifitime::HifitimeError;
use outfit::OutfitError;
use photom::observation_dataset::{ObsDatasetError, ObsId};
use thiserror::Error;

use crate::{engine_config::error::ConfigError, seeding::error::SeedingError};

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

#[derive(Debug, Error)]
pub enum EngineError {
    /// The pipeline plan is invalid (stages, window, invariants, etc).
    #[error("invalid pipeline plan: {0}")]
    InvalidPlan(&'static str),

    /// Pipeline was cancelled by the caller (hooks).
    #[error("pipeline cancelled")]
    Cancelled,

    // -------------------------------------------------------------------------
    // Wrappers for lower-level subsystems (add as you wire real calls)
    // -------------------------------------------------------------------------
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

/// Errors that can occur during a [`crate::topocentric_kf::single_kalman::KFState::update`] step.
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
