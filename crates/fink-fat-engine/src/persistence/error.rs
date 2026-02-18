use thiserror::Error;

use crate::{AlertKey, seeding::SeedKey};

/// Errors raised when validating a [`DiskEnvelope`].
///
/// Notes
/// -----
/// These errors are strictly about the envelope header (magic + version).
/// Actual I/O and encoding/decoding errors are reported as [`PersistenceIoError`].
#[derive(Debug, thiserror::Error)]
pub enum EnvelopeError {
    /// The file signature does not match [`DISK_MAGIC`].
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

    /// Bitcode encode/decode error.
    #[error("bitcode error: {0}")]
    Bitcode(String),

    /// Envelope validation error (magic/version).
    #[error(transparent)]
    Envelope(#[from] EnvelopeError),

    /// Any other error (e.g., missing data for payload construction).
    #[error("other error: {0}")]
    Other(String),
}

#[derive(Debug, Error)]
pub enum BorrowError {
    #[error("missing seed for key {0:?}")]
    MissingSeed(SeedKey),

    #[error("missing alert for key {0:?}")]
    MissingAlert(AlertKey),
}

#[derive(Debug, Error)]
pub enum PersistenceError {
    #[error(transparent)]
    Io(#[from] PersistenceIoError),

    #[error(transparent)]
    Borrow(#[from] BorrowError),
}
