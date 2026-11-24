//! Crate-level error types of `fink-fat-engine`.
//!
//! [`EngineError`] aggregates the errors of every subsystem (configuration,
//! seeding, `outfit`, `photom`, `hifitime`) through `#[from]` conversions, so
//! that engine entry points can propagate them with `?`. [`FinkFatError`]
//! holds generic I/O, JSON and message errors.

use std::io;

use hifitime::HifitimeError;
use outfit::OutfitError;
use photom::{
    NightId,
    observation_dataset::{ObsDatasetError, ObsId},
};
use thiserror::Error;

use crate::{engine_config::error::ConfigError, seeding::error::SeedingError};

/// Unified error type for the fink-fat engine.
///
/// Notes
/// -----
/// This error groups together:
/// - filesystem I/O errors,
/// - JSON serialization errors,
/// - binary encoding/decoding errors (`BincodeEncode` / `BincodeDecode`,
///   legacy variants kept for API compatibility; the snapshot and archive
///   are serialized with `rkyv`),
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
    // Wrappers for lower-level subsystems
    // -------------------------------------------------------------------------
    /// Generic engine error (I/O, JSON, free-form messages).
    #[error(transparent)]
    FinkFat(#[from] FinkFatError),

    /// Error from the `outfit` crate (ephemeris, observer cache, Kepler solver).
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

    /// An observation id referenced by a pair, branch or track is missing
    /// from the `ObsDataset`.
    #[error("obs dataset id not found : {0}")]
    ObsDatasetIdNotFound(ObsId),

    /// Intra-night seeding error.
    #[error(transparent)]
    Seeding(#[from] SeedingError),

    /// Topocentric range estimation failed (no admissible root).
    #[error(transparent)]
    TopoError(#[from] TopocentricRangeError),

    /// A night was opened on a collection that already closed it, or a later
    /// night (see
    /// [`BranchCollection::last_closed_night_id`](crate::topocentric_kf::branching::BranchCollection::last_closed_night_id)).
    /// Advancing it again would re-seed its observations and double-count
    /// the night.
    #[error("night {night_id} is not after the last closed night {last_closed}")]
    NightAlreadyClosed {
        /// Night the caller tried to open.
        night_id: NightId,
        /// Last night the collection closed.
        last_closed: NightId,
    },

    /// The branch-collection snapshot on disk is unreadable.
    #[error(transparent)]
    Snapshot(#[from] SnapshotError),
}

/// Failure to read or write a branch-collection snapshot (see
/// [`BranchCollection::save_snapshot_to_disk`](crate::topocentric_kf::branching::BranchCollection::save_snapshot_to_disk)).
#[derive(Debug, Error)]
pub enum SnapshotError {
    /// The file starts with the snapshot magic but is shorter than the header.
    #[error("snapshot header is truncated")]
    TruncatedHeader,

    /// The header carries a format version this build cannot read.
    #[error("unsupported snapshot format version {0}")]
    UnsupportedVersion(u32),

    /// The rkyv payload failed to serialize.
    #[error("snapshot encoding failed: {0}")]
    Encode(String),

    /// The rkyv payload failed validation or deserialization.
    #[error("snapshot decoding failed: {0}")]
    Decode(String),
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
