//! Versioned persistence envelope and binary I/O for Fink-FAT on-disk payloads.
//!
//! Overview
//! --------
//! This module provides a small, stable wrapper [`DiskEnvelope<T>`] that should
//! be used around **all** serialized payloads written by the Fink-FAT pipeline.
//! It also includes minimal, robust binary I/O helpers using the `bitcode` crate.
//!
//! The central idea is to wrap each payload `T` as:
//!
//! - a **magic signature** (`DISK_MAGIC`) to quickly detect wrong files,
//! - a **schema version** (`schema_version`) for forward/backward compatibility,
//! - a **creation timestamp** (`created_unix_s`) for provenance/debugging,
//! - the actual `payload`.
//!
//! Why use an envelope?
//! --------------------
//! - **File type detection**: `magic` prevents confusing unrelated files with
//!   persistence blobs.
//! - **Schema evolution**: `schema_version` allows you to bump the format when
//!   the structure of a payload changes, and reject/upgrade older files cleanly.
//! - **Traceability**: `created_unix_s` helps debugging and reproducibility.
//!
//! Encoding format
//! ---------------
//! This module uses `bitcode` (with `serde`) to serialize/deserialize data to a
//! compact binary representation.
//!
//! Atomic writes
//! ------------
//! To avoid partially-written files, writes follow the common pattern:
//!
//! 1. Write bytes to a temporary file next to the target (`<path>.tmp`).
//! 2. Flush and `sync_all()` the temporary file.
//! 3. Rename the temporary file to the final path.
//!
//! On most filesystems, the rename step is atomic when both files are on the
//! same filesystem. This guarantees that readers see either the old file or the
//! new file, but never a truncated intermediate state.
//!
//! Notes
//! -----
//! - The `sync_all()` call makes durability stronger but can be slower; it is
//!   typically acceptable for once-per-day persistence.
//! - This module keeps I/O primitives local for simplicity. You may later split
//!   them into `envelope.rs` + `io.rs` if preferred.
//!
//! See also
//! --------
//! - Your `persistence::*Owned` structs (alerts, seeds, graph) that are intended
//!   to be wrapped into [`DiskEnvelope`] before writing.

use camino::Utf8Path;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    fs::{self, File},
    io::{Read, Write},
};

use thiserror::Error;

/// Magic bytes used to identify Fink-FAT persistence files.
///
/// This constant should remain stable across releases. It provides a quick
/// sanity check when reading from disk.
pub const DISK_MAGIC: [u8; 8] = *b"FINKFAT\0";

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

/// Versioned wrapper for any on-disk payload.
///
/// This is the recommended container for every persisted payload type in the
/// pipeline (alerts, seeds, graph blocks, etc.).
///
/// Fields
/// ------
/// - `magic` – File signature used to detect file type.
/// - `schema_version` – Payload schema version (bump when `T` changes on disk).
/// - `created_unix_s` – Unix timestamp (seconds) for traceability.
/// - `payload` – Actual stored data.
///
/// Notes
/// -----
/// - The envelope itself should remain stable.
/// - Only bump `schema_version` when the *payload schema* changes.
/// - When reading, always validate `magic` and `schema_version` before trusting
///   the payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskEnvelope<T> {
    /// File signature (`DISK_MAGIC`).
    pub magic: [u8; 8],
    /// Payload schema version.
    pub schema_version: u32,
    /// Unix timestamp (seconds).
    pub created_unix_s: i64,
    /// Payload.
    pub payload: T,
}

impl<T: DeserializeOwned + Serialize> DiskEnvelope<T> {
    /// Create a new envelope around `payload`.
    ///
    /// Parameters
    /// ----------
    /// payload : T
    ///     The payload to wrap and serialize.
    /// schema_version : u32
    ///     Schema version for the payload type. Bump this when the on-disk
    ///     representation changes.
    /// created_unix_s : i64
    ///     Unix timestamp (seconds) to store in the envelope.
    ///
    /// Returns
    /// -------
    /// DiskEnvelope<T>
    ///     A new envelope ready to be serialized to disk.
    #[inline]
    pub fn new(payload: T, schema_version: u32, created_unix_s: i64) -> Self {
        Self {
            magic: DISK_MAGIC,
            schema_version,
            created_unix_s,
            payload,
        }
    }

    /// Validate the envelope header (magic + schema version).
    ///
    /// Parameters
    /// ----------
    /// supported_schema_version : u32
    ///     The expected schema version for this payload type.
    ///
    /// Returns
    /// -------
    /// Result<(), EnvelopeError>
    ///     `Ok(())` if both magic and schema version match, otherwise an error.
    ///
    /// Errors
    /// ------
    /// EnvelopeError::InvalidMagic
    ///     If `magic` does not match [`DISK_MAGIC`].
    /// EnvelopeError::UnsupportedSchemaVersion
    ///     If `schema_version` differs from `supported_schema_version`.
    #[inline]
    pub fn validate(&self, supported_schema_version: u32) -> Result<(), EnvelopeError> {
        if self.magic != DISK_MAGIC {
            return Err(EnvelopeError::InvalidMagic);
        }
        if self.schema_version != supported_schema_version {
            return Err(EnvelopeError::UnsupportedSchemaVersion {
                found: self.schema_version,
                supported: supported_schema_version,
            });
        }
        Ok(())
    }

    /// Consume the envelope and return the payload after validation.
    ///
    /// Parameters
    /// ----------
    /// supported_schema_version : u32
    ///     The expected schema version for this payload type.
    ///
    /// Returns
    /// -------
    /// Result<T, EnvelopeError>
    ///     The payload `T` if the envelope is valid.
    #[inline]
    pub fn into_payload(self, supported_schema_version: u32) -> Result<T, EnvelopeError> {
        self.validate(supported_schema_version)?;
        Ok(self.payload)
    }

    /// Write the enveloped payload to disk as a binary blob (bitcode).
    ///
    /// This method performs an **atomic write**:
    /// - write `<path>.tmp`,
    /// - flush + fsync the file,
    /// - rename to `path`.
    ///
    /// Parameters
    /// ----------
    /// path : &Utf8Path
    ///     Destination file path. Parent directories are created if needed.
    ///
    /// Returns
    /// -------
    /// Result<(), PersistenceIoError>
    ///     `Ok(())` on success.
    ///
    /// Notes
    /// -----
    /// - This serializes an envelope-by-reference to avoid cloning the payload.
    /// - The `rename` step is expected to be atomic when source/destination are
    ///   on the same filesystem.
    pub fn save_enveloped(&self, path: &Utf8Path) -> Result<(), PersistenceIoError> {
        #[derive(serde::Serialize)]
        struct DiskEnvelopeRef<'a, T> {
            magic: [u8; 8],
            schema_version: u32,
            created_unix_s: i64,
            payload: &'a T,
        }

        let env = DiskEnvelopeRef {
            magic: super::envelope::DISK_MAGIC,
            schema_version: self.schema_version,
            created_unix_s: self.created_unix_s,
            payload: &self.payload,
        };

        let bytes = encode_bytes(&env)?;
        atomic_write_utf8(path, &bytes)?;
        Ok(())
    }

    /// Load a payload from disk by decoding a [`DiskEnvelope<T>`] and validating it.
    ///
    /// Parameters
    /// ----------
    /// path : &Utf8Path
    ///     Input file path.
    /// schema_version : u32
    ///     The expected schema version for this payload type.
    ///
    /// Returns
    /// -------
    /// Result<T, PersistenceIoError>
    ///     The decoded payload if the envelope is valid.
    ///
    /// Errors
    /// ------
    /// PersistenceIoError::Io
    ///     If the file cannot be opened/read.
    /// PersistenceIoError::Bitcode
    ///     If decoding fails (corruption or incompatible format).
    /// PersistenceIoError::Envelope
    ///     If the envelope header is invalid (wrong magic/version).
    pub fn load_enveloped(path: &Utf8Path, schema_version: u32) -> Result<T, PersistenceIoError> {
        let bytes = read_all(path)?;
        let env: DiskEnvelope<T> = decode_bytes(&bytes)?;
        env.into_payload(schema_version)
            .map_err(PersistenceIoError::from)
    }
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
}

/// Encode a value into bytes using `bitcode`.
///
/// Parameters
/// ----------
/// value : &T
///     Any serializable value.
///
/// Returns
/// -------
/// Result<Vec<u8>, PersistenceIoError>
///     Encoded bytes.
///
/// Errors
/// ------
/// PersistenceIoError::Bitcode
///     If encoding fails.
#[inline]
fn encode_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, PersistenceIoError> {
    bitcode::serialize(value).map_err(|e| PersistenceIoError::Bitcode(e.to_string()))
}

/// Decode a value from bytes using `bitcode`.
///
/// Parameters
/// ----------
/// bytes : &[u8]
///     Input buffer.
///
/// Returns
/// -------
/// Result<T, PersistenceIoError>
///     Decoded value.
///
/// Errors
/// ------
/// PersistenceIoError::Bitcode
///     If decoding fails.
#[inline]
fn decode_bytes<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, PersistenceIoError> {
    bitcode::deserialize(bytes).map_err(|e| PersistenceIoError::Bitcode(e.to_string()))
}

/// Write bytes to disk atomically using a temporary file + rename.
///
/// Parameters
/// ----------
/// path : &Utf8Path
///     Final destination path.
/// bytes : &[u8]
///     Content to write.
///
/// Returns
/// -------
/// Result<(), PersistenceIoError>
///     `Ok(())` on success.
///
/// Notes
/// -----
/// The algorithm is:
/// 1. Create parent directories if needed.
/// 2. Write `<path>.tmp`.
/// 3. Flush and `sync_all()` the temporary file.
/// 4. Rename the temporary file to `path`.
///
/// This prevents readers from observing a partially written file.
fn atomic_write_utf8(path: &Utf8Path, bytes: &[u8]) -> Result<(), PersistenceIoError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent.as_std_path())?;
    }

    let tmp_path = Utf8Path::new(&format!("{}.tmp", path)).to_path_buf();

    let mut f = File::create(tmp_path.as_std_path())?;
    f.write_all(bytes)?;
    f.flush()?;
    f.sync_all()?; // stronger durability guarantee (optional but recommended)

    fs::rename(tmp_path.as_std_path(), path.as_std_path())?;
    Ok(())
}

/// Read the full file into memory.
///
/// Parameters
/// ----------
/// path : &Utf8Path
///     Input file path.
///
/// Returns
/// -------
/// Result<Vec<u8>, PersistenceIoError>
///     File content as bytes.
///
/// Errors
/// ------
/// PersistenceIoError::Io
///     If the file cannot be opened/read.
fn read_all(path: &Utf8Path) -> Result<Vec<u8>, PersistenceIoError> {
    let mut f = File::open(path.as_std_path())?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    Ok(buf)
}
