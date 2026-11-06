//! Error type of the streaming driver.

use camino::Utf8PathBuf;
use fink_fat_engine::{
    engine_config::error::ConfigError, error::EngineError, logging::subscriber::LogInitError,
};
use photom::{
    io::lsst_alert::LsstAlertError,
    observation_dataset::{ObsDatasetError, ObsId},
};
use polars::error::PolarsError;
use thiserror::Error;

/// Everything that can stop the streaming driver.
///
/// Per-alert problems (quality cuts, unconvertible alerts, duplicates) are
/// *not* errors: they are counted and logged (see
/// [`Rejection`](crate::ingest::Rejection)). An error here means the driver
/// cannot go on safely.
#[derive(Debug, Error)]
pub enum StreamError {
    /// The tracking engine failed (night refused, discovery, snapshot, log).
    #[error(transparent)]
    Engine(#[from] EngineError),

    /// The engine configuration is invalid.
    #[error(transparent)]
    EngineConfig(#[from] ConfigError),

    /// The night's observation dataset rejected an insertion.
    #[error(transparent)]
    ObsDataset(#[from] ObsDatasetError),

    /// An observation id reached tonight's dataset twice. The ingestion step
    /// deduplicates alerts, so this signals a bug, not bad input.
    #[error("observation {0} is already in tonight's dataset")]
    DuplicateObservation(ObsId),

    /// An alert's `visit` does not encode a valid observing night.
    #[error(transparent)]
    Alert(#[from] LsstAlertError),

    /// The replay dump could not be read.
    #[error(transparent)]
    Polars(#[from] PolarsError),

    /// Filesystem error (journal, archive log, storage directory).
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// The stream configuration file could not be read or parsed.
    #[error("stream configuration {path}: {message}")]
    Config {
        /// Path of the configuration file.
        path: Utf8PathBuf,
        /// What went wrong.
        message: String,
    },

    /// The logging subscriber could not be installed.
    #[error(transparent)]
    Logging(#[from] LogInitError),
}

/// `Result` with [`StreamError`].
pub type Result<T> = std::result::Result<T, StreamError>;
