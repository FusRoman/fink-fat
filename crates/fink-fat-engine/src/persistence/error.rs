use photom::observation_dataset::ObsId;
use thiserror::Error;

use crate::seeding::SeedKey;

/// Errors raised when validating a [`DiskEnvelope`](crate::persistence::envelope::DiskEnvelope).
///
/// Notes
/// -----
/// These errors are strictly about the envelope header (magic + version).
/// Actual I/O and encoding/decoding errors are reported as [`PersistenceIoError`].
#[derive(Debug, thiserror::Error)]
pub enum EnvelopeError {
    /// The file signature does not match [`DISK_MAGIC`](crate::persistence::envelope::DISK_MAGIC).
    ///
    /// This usually means:
    /// - the wrong file path was provided, or
    /// - the file was produced by another tool/version, or
    /// - the file is corrupted.
    #[error("invalid persistence magic bytes")]
    InvalidMagic,

    /// The file schema version is not supported by the current code.
    ///
    /// This usually means you need:
    /// - a migration step, or
    /// - to regenerate the persistence files with the current version.
    #[error("unsupported schema version: found {found}, supported {supported}")]
    UnsupportedSchemaVersion { found: u32, supported: u32 },
}

/// Errors that can occur while reading/writing persistence files.
#[derive(Debug, Error)]
pub enum PersistenceIoError {
    /// Any I/O error (open/read/write/fsync/rename).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Postcard encode/decode error.
    #[error("postcard error: {0}")]
    Postcard(String),

    /// Arrow / Parquet error (schema construction, record batch build, writer).
    #[error("arrow/parquet error: {0}")]
    Arrow(String),

    /// JSON encode/decode error.
    #[error("json error: {0}")]
    Json(String),

    /// Compression or decompression error.
    #[error("compression error: {0}")]
    Compression(String),

    /// Envelope validation error (magic/version).
    #[error(transparent)]
    Envelope(#[from] EnvelopeError),

    /// Any other error (e.g., missing data for payload construction).
    #[error("other error: {0}")]
    Other(String),

    /// A persistence error annotated with the file path that triggered it.
    ///
    /// Wrap any [`PersistenceIoError`] with `.with_path(path)` at a load/save
    /// site to ensure the failing path is always visible in error messages.
    #[error("error for path `{path}`: {source}")]
    WithPath {
        /// The file path that was being accessed when the error occurred.
        path: String,
        /// The underlying error.
        #[source]
        source: Box<PersistenceIoError>,
    },
}

impl PersistenceIoError {
    /// Annotate this error with a file path for better diagnostics.
    ///
    /// Use this at every load/save call site so that the failing path is
    /// always surfaced in the error message, e.g.:
    ///
    /// ```no_run
    /// # use camino::Utf8Path;
    /// # use fink_fat_engine::persistence::error::PersistenceIoError;
    /// # fn example(path: &Utf8Path) -> Result<(), PersistenceIoError> {
    /// some_load(path).map_err(|e| e.with_path(path))?;
    /// # Ok(()) }
    /// # fn some_load(_: &Utf8Path) -> Result<(), PersistenceIoError> { Ok(()) }
    /// ```
    #[inline]
    pub fn with_path(self, path: impl std::fmt::Display) -> Self {
        Self::WithPath {
            path: path.to_string(),
            source: Box::new(self),
        }
    }

    /// Returns `true` if this error (or an error nested inside a [`Self::WithPath`]
    /// wrapper) is an [`std::io::Error`] with [`std::io::ErrorKind::NotFound`].
    ///
    /// Use this helper instead of matching on `Io(e)` directly so that the
    /// check still works after a `.with_path()` annotation.
    pub fn is_not_found(&self) -> bool {
        match self {
            Self::Io(e) => e.kind() == std::io::ErrorKind::NotFound,
            Self::WithPath { source, .. } => source.is_not_found(),
            _ => false,
        }
    }
}

#[derive(Debug, Error)]
pub enum BorrowError {
    #[error("missing seed for key {0:?}")]
    MissingSeed(SeedKey),

    #[error("missing observation for id {0}")]
    MissingObservation(ObsId),
}

#[derive(Debug, Error)]
pub enum PersistenceError {
    #[error(transparent)]
    Io(#[from] PersistenceIoError),

    #[error(transparent)]
    Borrow(#[from] BorrowError),
}
