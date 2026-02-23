//! Versioned persistence envelope and binary I/O for Fink-FAT on-disk payloads.
//!
//! Overview
//! --------
//! This module defines a small, stable container [`DiskEnvelope<T>`] and a set of
//! minimal binary I/O helpers to persist Fink-FAT artifacts on disk.
//!
//! The persistence layer writes many heterogeneous payloads (alerts, seeds, graph
//! snapshots, edge journals, trajectories, …). To make those files:
//! - easy to *identify*,
//! - safe to *validate*,
//! - and robust to *schema evolution*,
//! every payload is wrapped in a versioned envelope that contains:
//!
//! - a **magic signature** ([`DISK_MAGIC`]) to detect wrong file types early,
//! - a **schema version** (`schema_version`) to gate compatibility,
//! - a **creation timestamp** (`created_unix_s`) for provenance/debugging,
//! - the actual **payload** (`payload: T`).
//!
//! Using an envelope is intentionally boring but extremely effective:
//! - It avoids confusing unrelated blobs as valid persistence files.
//! - It centralizes schema/version checks, instead of sprinkling ad-hoc guards in
//!   every store.
//! - It provides a consistent place to store lightweight provenance metadata.
//!
//! Design goals
//! ------------
//! - **Stability**: the envelope structure should remain stable across releases.
//! - **Explicit compatibility**: payload formats are gated by `schema_version`.
//! - **Robustness**: atomic writes prevent partially-written files.
//! - **Simplicity**: helpers stay local and do not require a complex I/O layer.
//!
//! Encoding format
//! ---------------
//! Serialization uses the `bitcode` crate (with `serde`) to produce a compact
//! binary representation.
//!
//! The exact byte-level format is an implementation detail of `bitcode`.
//! Consequently, long-term persistence guarantees are achieved via:
//! - a stable envelope layout (this module),
//! - explicit schema versions (per payload type),
//! - and consistent migration/rejection policy at load time.
//!
//! Atomic writes
//! ------------
//! Writes follow the classic "temp file + rename" strategy:
//!
//! 1. Serialize to bytes in memory.
//! 2. Write to `<path>.tmp` next to the destination.
//! 3. `flush()` then `sync_all()` the temporary file.
//! 4. `rename()` the temporary file to the final path.
//!
//! On typical filesystems, `rename()` is atomic when source and destination are
//! on the same filesystem. Readers will observe either:
//! - the old complete file, or
//! - the new complete file,
//! but never a truncated intermediate state.
//!
//! Notes
//! -----
//! - `sync_all()` improves durability (data reaches stable storage) but can be
//!   slower. This trade-off is usually acceptable for persistence performed at
//!   low cadence (e.g., once per night / per batch).
//! - Helpers here load entire files into memory. This is appropriate for the
//!   expected payload sizes (manifests, nightly blobs, graph blocks). If very
//!   large payloads are introduced, consider adding streaming I/O.
//!
//! Schema versioning strategy
//! --------------------------
//! - Each persisted payload type owns its *schema version constant* (e.g.
//!   `ALERT_STORE_SCHEMA_VERSION`, `SEED_STORE_SCHEMA_VERSION`, …).
//! - When the *on-disk representation* of that payload changes, bump the constant.
//! - When reading, pass the expected version to [`DiskEnvelope::load_enveloped`]
//!   and reject unsupported versions cleanly.
//!
//! See also
//! --------
//! - `crate::persistence::*Owned` payload structs intended to be stored on disk.
//! - `crate::persistence::error::{EnvelopeError, PersistenceIoError}` for error
//!   types used by this module.

use camino::Utf8Path;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::parquet::file::properties::WriterProperties;
use datafusion::{
    arrow::array::RecordBatch, parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    fs::{self, File},
    io::{Read, Write},
};

use crate::persistence::error::{EnvelopeError, PersistenceIoError};

/// Magic bytes used to identify Fink-FAT persistence files.
///
/// This signature is written at the beginning of every serialized
/// [`DiskEnvelope<T>`]. It provides a fast sanity check when reading from disk.
///
/// Stability
/// ---------
/// This constant should remain stable across releases. Changing it would
/// invalidate all previously written blobs, and should only be done as part of a
/// deliberate, breaking change to the persistence format.
pub const DISK_MAGIC: [u8; 8] = *b"FINKFAT\0";

/// Versioned wrapper for any on-disk payload.
///
/// This is the recommended container for every persisted payload type in the
/// pipeline (alerts, seeds, graph blocks, edge journals, …).
///
/// Fields
/// ------
/// - `magic`
///   File signature used to detect wrong file types. Must match [`DISK_MAGIC`].
/// - `schema_version`
///   Schema version for the payload `T`. Bump when the on-disk representation of
///   `T` changes.
/// - `created_unix_s`
///   Unix timestamp (seconds) stored for provenance/debugging.
/// - `payload`
///   The actual persisted data.
///
/// Validation contract
/// -------------------
/// Before the payload is trusted, loaders must validate:
/// - `magic == DISK_MAGIC`
/// - `schema_version == expected_schema_version`
///
/// This module provides [`DiskEnvelope::validate`] and [`DiskEnvelope::into_payload`]
/// to enforce this contract.
///
/// Compatibility
/// -------------
/// The envelope layout should remain stable. Only `schema_version` should change
/// as payload formats evolve.
///
/// Security & robustness notes
/// ---------------------------
/// - Envelope validation does not protect against malicious inputs. It is meant
///   to catch accidental misuse (wrong file) and routine incompatibilities
///   (wrong schema version).
/// - Payload decoding uses `bitcode` + `serde`. Decoding errors are surfaced via
///   [`PersistenceIoError::Bitcode`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskEnvelope<T> {
    /// File signature (must equal [`DISK_MAGIC`]).
    pub magic: [u8; 8],

    /// Payload schema version (bump when `T` changes on disk).
    pub schema_version: u32,

    /// Unix timestamp (seconds) stored for traceability.
    pub created_unix_s: i64,

    /// The persisted payload.
    pub payload: T,
}

impl<T: DeserializeOwned + Serialize> DiskEnvelope<T> {
    /// Create a new envelope around a payload.
    ///
    /// This constructor sets `magic` to [`DISK_MAGIC`] and stores the provided
    /// schema version and timestamp.
    ///
    /// Arguments
    /// ---------
    /// * `payload` - The value to persist.
    /// * `schema_version` - Schema version associated with the payload type `T`.
    ///   Bump this value whenever the on-disk representation of `T` changes.
    /// * `created_unix_s` - Unix timestamp (seconds) recorded for provenance.
    ///
    /// Return
    /// ------
    /// A fully-populated [`DiskEnvelope<T>`] ready to be serialized.
    ///
    /// Notes
    /// -----
    /// This does not perform any I/O. Use [`DiskEnvelope::save_enveloped`] to
    /// write the envelope to disk.
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
    /// This check is intentionally strict:
    /// - `magic` must match [`DISK_MAGIC`],
    /// - `schema_version` must equal `supported_schema_version`.
    ///
    /// Arguments
    /// ---------
    /// * `supported_schema_version` - The expected schema version for the payload
    ///   type being loaded.
    ///
    /// Return
    /// ------
    /// - `Ok(())` if the header is valid.
    /// - `Err(EnvelopeError)` otherwise.
    ///
    /// Errors
    /// ------
    /// - [`EnvelopeError::InvalidMagic`]
    ///   If the file signature does not match [`DISK_MAGIC`].
    /// - [`EnvelopeError::UnsupportedSchemaVersion`]
    ///   If the schema version differs from `supported_schema_version`.
    ///
    /// Notes
    /// -----
    /// This method does not validate the semantic correctness of `payload`.
    /// Payload-level validation belongs to the payload type or higher-level
    /// stores.
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
    /// This is a convenience method equivalent to:
    /// 1. [`DiskEnvelope::validate`]
    /// 2. return `payload` if valid
    ///
    /// Arguments
    /// ---------
    /// * `supported_schema_version` - The expected schema version for the payload
    ///   type being loaded.
    ///
    /// Return
    /// ------
    /// - `Ok(T)` if the envelope header is valid.
    /// - `Err(EnvelopeError)` if the header is invalid.
    ///
    /// Notes
    /// -----
    /// The envelope is consumed even on success, which avoids cloning `payload`.
    #[inline]
    pub fn into_payload(self, supported_schema_version: u32) -> Result<T, EnvelopeError> {
        self.validate(supported_schema_version)?;
        Ok(self.payload)
    }

    /// Write the enveloped payload to disk as a binary blob (bitcode).
    ///
    /// Behavior
    /// --------
    /// Performs an **atomic write**:
    /// - serialize to bytes,
    /// - write `<path>.tmp`,
    /// - `flush()` + `sync_all()` the temporary file,
    /// - rename the temporary file to `path`.
    ///
    /// Arguments
    /// ---------
    /// * `path` - Destination file path. Parent directories are created if needed.
    ///
    /// Return
    /// ------
    /// `Ok(())` on success, otherwise a [`PersistenceIoError`].
    ///
    /// Errors
    /// ------
    /// - [`PersistenceIoError::Io`] (via `?`)
    ///   If directory creation, file creation, write, fsync, or rename fails.
    /// - [`PersistenceIoError::Bitcode`]
    ///   If serialization fails.
    ///
    /// Performance notes
    /// -----------------
    /// - The payload is serialized by reference (no clone required).
    /// - The resulting buffer is currently built in memory before writing.
    ///
    /// Atomicity notes
    /// --------------
    /// `rename()` is expected to be atomic when the temporary file is created on
    /// the same filesystem as the destination. This function creates `<path>.tmp`
    /// next to `path`, which satisfies that condition.
    pub fn save_enveloped(&self, path: &Utf8Path) -> Result<(), PersistenceIoError> {
        /// Lightweight borrowed view used to avoid cloning `payload`.
        ///
        /// This struct is serialized instead of `DiskEnvelope<T>` to ensure the
        /// serialization uses `&T` rather than requiring `T: Clone`.
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
    /// Behavior
    /// --------
    /// 1. Read the entire file into memory.
    /// 2. Decode a [`DiskEnvelope<T>`] using `bitcode`.
    /// 3. Validate `magic` and `schema_version`.
    /// 4. Return the decoded payload `T`.
    ///
    /// Arguments
    /// ---------
    /// * `path` - Input file path.
    /// * `schema_version` - Expected schema version for payload type `T`.
    ///
    /// Return
    /// ------
    /// - `Ok(T)` if the file is readable, decodes successfully, and passes header
    ///   validation.
    /// - `Err(PersistenceIoError)` otherwise.
    ///
    /// Errors
    /// ------
    /// - [`PersistenceIoError::Io`] (via `?`)
    ///   If the file cannot be opened/read.
    /// - [`PersistenceIoError::Bitcode`]
    ///   If decoding fails (corruption or incompatible encoding).
    /// - [`PersistenceIoError::Envelope`]
    ///   If `magic` or `schema_version` are invalid.
    ///
    /// Notes
    /// -----
    /// This method is a "typed" loader: callers select `T` at compile-time. If
    /// the file contains a different payload type, decoding will fail (often as
    /// `Bitcode`), or the envelope validation may fail early if schema versions
    /// differ.
    pub fn load_enveloped(path: &Utf8Path, schema_version: u32) -> Result<T, PersistenceIoError> {
        let bytes = read_all(path)?;
        let env: DiskEnvelope<T> = decode_bytes(&bytes)?;
        env.into_payload(schema_version)
            .map_err(PersistenceIoError::from)
    }
}

/// Encode a value into bytes using `bitcode`.
///
/// This helper centralizes the `bitcode` error mapping into [`PersistenceIoError`].
///
/// Arguments
/// ---------
/// * `value` - Any serializable value.
///
/// Return
/// ------
/// Encoded bytes.
///
/// Errors
/// ------
/// - [`PersistenceIoError::Bitcode`]
///   If encoding fails.
///
/// Notes
/// -----
/// This function does not perform any I/O.
#[inline]
fn encode_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, PersistenceIoError> {
    bitcode::serialize(value).map_err(|e| PersistenceIoError::Bitcode(e.to_string()))
}

/// Decode a value from bytes using `bitcode`.
///
/// Arguments
/// ---------
/// * `bytes` - Input buffer containing the serialized representation.
///
/// Return
/// ------
/// Decoded value.
///
/// Errors
/// ------
/// - [`PersistenceIoError::Bitcode`]
///   If decoding fails.
///
/// Notes
/// -----
/// This function does not perform any I/O.
#[inline]
fn decode_bytes<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, PersistenceIoError> {
    bitcode::deserialize(bytes).map_err(|e| PersistenceIoError::Bitcode(e.to_string()))
}

/// Write bytes to disk atomically using a temporary file + rename.
///
/// Arguments
/// ---------
/// * `path` - Final destination path.
/// * `bytes` - Content to write.
///
/// Return
/// ------
/// `Ok(())` on success, otherwise a [`PersistenceIoError`].
///
/// Algorithm
/// ---------
/// 1. Create parent directories if needed.
/// 2. Write `<path>.tmp`.
/// 3. Flush and `sync_all()` the temporary file.
/// 4. Rename the temporary file to `path`.
///
/// Errors
/// ------
/// Returns [`PersistenceIoError::Io`] (via `?`) if any filesystem operation fails.
///
/// Notes
/// -----
/// - This routine is meant to prevent readers from observing partially-written
///   files.
/// - `sync_all()` requests that the file contents are durably persisted. This
///   improves crash-safety but can cost latency.
/// - This function does not fsync the parent directory. If directory fsync is
///   required for stronger guarantees across sudden power loss, add it at a
///   higher level where it can be amortized.
fn atomic_write_utf8(path: &Utf8Path, bytes: &[u8]) -> Result<(), PersistenceIoError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent.as_std_path())?;
    }

    // Write a temporary file next to the destination to keep rename atomic.
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
/// Arguments
/// ---------
/// * `path` - Input file path.
///
/// Return
/// ------
/// File content as bytes.
///
/// Errors
/// ------
/// Returns [`PersistenceIoError::Io`] (via `?`) if the file cannot be opened or read.
///
/// Notes
/// -----
/// This loads the entire file into memory. For very large payloads, consider a
/// streaming design (but keep the envelope validation semantics).
fn read_all(path: &Utf8Path) -> Result<Vec<u8>, PersistenceIoError> {
    let mut f = File::open(path.as_std_path())?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    Ok(buf)
}

// =============================================================================
// Parquet I/O helpers
// =============================================================================

/// Write a single Arrow `RecordBatch` to a Parquet file atomically.
///
/// Behavior
/// --------
/// Performs an **atomic write** using the same "temp file + rename" strategy
/// as [`DiskEnvelope::save_enveloped`]:
///
/// 1. Create parent directories if needed.
/// 2. Write `<path>.tmp` with the Parquet content.
/// 3. Flush, sync, and rename to `path`.
///
/// Arguments
/// ---------
/// * `path`  - Destination Parquet file path.
/// * `batch` - Arrow `RecordBatch` to write.
///
/// Return
/// ------
/// `Ok(())` on success, otherwise a [`PersistenceIoError`].
pub fn save_parquet(path: &Utf8Path, batch: &RecordBatch) -> Result<(), PersistenceIoError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent.as_std_path())?;
    }

    let tmp_path = Utf8Path::new(&format!("{}.tmp", path)).to_path_buf();
    let file = File::create(tmp_path.as_std_path())?;

    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))
        .map_err(|e| PersistenceIoError::Arrow(e.to_string()))?;

    writer
        .write(batch)
        .map_err(|e| PersistenceIoError::Arrow(e.to_string()))?;

    writer
        .close()
        .map_err(|e| PersistenceIoError::Arrow(e.to_string()))?;

    fs::rename(tmp_path.as_std_path(), path.as_std_path())?;
    Ok(())
}

/// Read all record batches from a Parquet file.
///
/// Arguments
/// ---------
/// * `path` - Input Parquet file path.
///
/// Return
/// ------
/// `Ok(Vec<RecordBatch>)` containing every row group.
pub fn load_parquet(path: &Utf8Path) -> Result<(SchemaRef, Vec<RecordBatch>), PersistenceIoError> {
    let file = File::open(path.as_std_path())?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| PersistenceIoError::Arrow(e.to_string()))?;
    let schema = builder.schema().clone();
    let reader = builder
        .build()
        .map_err(|e| PersistenceIoError::Arrow(e.to_string()))?;

    let batches: Vec<RecordBatch> = reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| PersistenceIoError::Arrow(e.to_string()))?;

    Ok((schema, batches))
}

#[cfg(test)]
mod envelope_tests {
    use super::*;
    use camino::{Utf8Path, Utf8PathBuf};
    use serde::{Deserialize, Serialize};
    use tempfile::tempdir;

    /// Simple payload type used for roundtrip tests.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct DummyPayload {
        a: u32,
        b: String,
        v: Vec<i64>,
    }

    fn to_utf8_pathbuf(p: std::path::PathBuf) -> Utf8PathBuf {
        Utf8PathBuf::from_path_buf(p).expect("temp paths should be valid UTF-8")
    }

    fn tmp_path_for(dir: &tempfile::TempDir, name: &str) -> Utf8PathBuf {
        to_utf8_pathbuf(dir.path().join(name))
    }

    fn read_bytes(path: &Utf8Path) -> Vec<u8> {
        let mut f = std::fs::File::open(path.as_std_path()).unwrap();
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut f, &mut buf).unwrap();
        buf
    }

    #[test]
    fn validate_ok_when_magic_and_version_match() {
        let payload = DummyPayload {
            a: 1,
            b: "ok".to_string(),
            v: vec![1, 2, 3],
        };

        let env = DiskEnvelope::new(payload, 42, 123);
        env.validate(42).unwrap();
    }

    #[test]
    fn validate_err_on_invalid_magic() {
        let payload = DummyPayload {
            a: 1,
            b: "x".to_string(),
            v: vec![],
        };

        let mut env = DiskEnvelope::new(payload, 1, 0);
        env.magic = *b"NOTFINK\0";

        let err = env.validate(1).unwrap_err();
        assert!(matches!(err, EnvelopeError::InvalidMagic));
    }

    #[test]
    fn validate_err_on_unsupported_schema_version() {
        let payload = DummyPayload {
            a: 1,
            b: "x".to_string(),
            v: vec![],
        };

        let env = DiskEnvelope::new(payload, 7, 0);
        let err = env.validate(8).unwrap_err();

        assert!(matches!(
            err,
            EnvelopeError::UnsupportedSchemaVersion {
                found: 7,
                supported: 8
            }
        ));
    }

    #[test]
    fn into_payload_ok_returns_payload() {
        let payload = DummyPayload {
            a: 7,
            b: "hello".to_string(),
            v: vec![10, -5],
        };

        let env = DiskEnvelope::new(payload.clone(), 3, 999);
        let got = env.into_payload(3).unwrap();
        assert_eq!(got, payload);
    }

    #[test]
    fn into_payload_err_propagates_validation() {
        let payload = DummyPayload {
            a: 7,
            b: "hello".to_string(),
            v: vec![10, -5],
        };

        let env = DiskEnvelope::new(payload, 3, 999);
        let err = env.into_payload(4).unwrap_err();
        assert!(matches!(
            err,
            EnvelopeError::UnsupportedSchemaVersion {
                found: 3,
                supported: 4
            }
        ));
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempdir().unwrap();
        let path = tmp_path_for(&dir, "payload.bin");

        let payload = DummyPayload {
            a: 42,
            b: "roundtrip".to_string(),
            v: vec![1, 2, 3, 4],
        };

        let env = DiskEnvelope::new(payload.clone(), 10, 1_700_000_000);
        env.save_enveloped(&path).unwrap();

        let got = DiskEnvelope::<DummyPayload>::load_enveloped(&path, 10).unwrap();
        assert_eq!(got, payload);
    }

    #[test]
    fn save_creates_parent_directories() {
        let dir = tempdir().unwrap();
        let nested = to_utf8_pathbuf(dir.path().join("a/b/c/payload.bin"));

        let payload = DummyPayload {
            a: 1,
            b: "nested".to_string(),
            v: vec![],
        };

        let env = DiskEnvelope::new(payload.clone(), 1, 0);
        env.save_enveloped(&nested).unwrap();

        assert!(nested.exists());
        let got = DiskEnvelope::<DummyPayload>::load_enveloped(&nested, 1).unwrap();
        assert_eq!(got, payload);
    }

    #[test]
    fn save_is_atomic_style_and_leaves_no_tmp_file() {
        let dir = tempdir().unwrap();
        let path = tmp_path_for(&dir, "atomic.bin");
        let tmp_path = Utf8Path::new(&format!("{}.tmp", path)).to_path_buf();

        let payload = DummyPayload {
            a: 5,
            b: "atomic".to_string(),
            v: vec![9],
        };

        let env = DiskEnvelope::new(payload, 1, 0);
        env.save_enveloped(&path).unwrap();

        assert!(path.exists());
        assert!(!tmp_path.exists(), "temporary file must be renamed away");
    }

    #[test]
    fn save_overwrites_existing_file_cleanly() {
        let dir = tempdir().unwrap();
        let path = tmp_path_for(&dir, "overwrite.bin");

        let env1 = DiskEnvelope::new(
            DummyPayload {
                a: 1,
                b: "first".to_string(),
                v: vec![1],
            },
            1,
            0,
        );
        env1.save_enveloped(&path).unwrap();
        let got1 = DiskEnvelope::<DummyPayload>::load_enveloped(&path, 1).unwrap();
        assert_eq!(
            got1,
            DummyPayload {
                a: 1,
                b: "first".to_string(),
                v: vec![1]
            }
        );

        let env2 = DiskEnvelope::new(
            DummyPayload {
                a: 2,
                b: "second".to_string(),
                v: vec![2, 2],
            },
            1,
            0,
        );
        env2.save_enveloped(&path).unwrap();

        let got2 = DiskEnvelope::<DummyPayload>::load_enveloped(&path, 1).unwrap();
        assert_eq!(
            got2,
            DummyPayload {
                a: 2,
                b: "second".to_string(),
                v: vec![2, 2]
            }
        );
    }

    #[test]
    fn load_err_on_corrupted_bytes() {
        let dir = tempdir().unwrap();
        let path = tmp_path_for(&dir, "corrupt.bin");

        // Write random bytes that are not valid bitcode for DiskEnvelope<DummyPayload>.
        std::fs::write(path.as_std_path(), b"this is not bitcode").unwrap();

        let err = DiskEnvelope::<DummyPayload>::load_enveloped(&path, 1).unwrap_err();
        assert!(
            matches!(err, PersistenceIoError::Bitcode(_)),
            "expected Bitcode error, got: {err:?}"
        );
    }

    #[test]
    fn load_err_on_invalid_magic() {
        let dir = tempdir().unwrap();
        let path = tmp_path_for(&dir, "bad_magic.bin");

        let env = DiskEnvelope {
            magic: *b"BADMAGIC",
            schema_version: 1,
            created_unix_s: 0,
            payload: DummyPayload {
                a: 1,
                b: "x".to_string(),
                v: vec![],
            },
        };

        let bytes = encode_bytes(&env).unwrap();
        atomic_write_utf8(&path, &bytes).unwrap();

        let err = DiskEnvelope::<DummyPayload>::load_enveloped(&path, 1).unwrap_err();
        assert!(
            matches!(
                err,
                PersistenceIoError::Envelope(EnvelopeError::InvalidMagic)
            ),
            "expected Envelope(InvalidMagic), got: {err:?}"
        );
    }

    #[test]
    fn load_err_on_unsupported_schema_version() {
        let dir = tempdir().unwrap();
        let path = tmp_path_for(&dir, "bad_version.bin");

        let env = DiskEnvelope::new(
            DummyPayload {
                a: 1,
                b: "x".to_string(),
                v: vec![],
            },
            7,
            0,
        );

        // Write with schema_version=7
        let bytes = encode_bytes(&env).unwrap();
        atomic_write_utf8(&path, &bytes).unwrap();

        // Read expecting schema_version=8 -> should fail at envelope validation.
        let err = DiskEnvelope::<DummyPayload>::load_enveloped(&path, 8).unwrap_err();

        assert!(
            matches!(
                err,
                PersistenceIoError::Envelope(EnvelopeError::UnsupportedSchemaVersion {
                    found: 7,
                    supported: 8
                })
            ),
            "expected Envelope(UnsupportedSchemaVersion), got: {err:?}"
        );
    }

    #[test]
    fn save_writes_non_empty_bytes_for_non_empty_payload() {
        let dir = tempdir().unwrap();
        let path = tmp_path_for(&dir, "bytes.bin");

        let env = DiskEnvelope::new(
            DummyPayload {
                a: 123,
                b: "bytes".to_string(),
                v: vec![0, 1, 2, 3],
            },
            1,
            0,
        );

        env.save_enveloped(&path).unwrap();

        let bytes = read_bytes(&path);
        assert!(
            !bytes.is_empty(),
            "serialized file should not be empty for a non-empty payload"
        );
    }

    // =====================================================================
    // Parquet I/O tests
    // =====================================================================

    use arrow_array::{ArrayRef, Float64Array, Int32Array, StringArray, UInt64Array};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    /// Build a simple two-column RecordBatch for test purposes.
    fn make_test_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
                Arc::new(StringArray::from(vec!["alpha", "beta", "gamma"])) as ArrayRef,
            ],
        )
        .unwrap()
    }

    #[test]
    fn save_parquet_creates_file() {
        let dir = tempdir().unwrap();
        let path = tmp_path_for(&dir, "test.parquet");

        let batch = make_test_batch();
        save_parquet(&path, &batch).unwrap();

        assert!(path.exists(), "parquet file must exist after save");

        let bytes = read_bytes(&path);
        assert!(
            bytes.len() > 4,
            "parquet file should contain more than just a header"
        );
    }

    #[test]
    fn save_parquet_creates_parent_directories() {
        let dir = tempdir().unwrap();
        let path = to_utf8_pathbuf(dir.path().join("a/b/c/nested.parquet"));

        let batch = make_test_batch();
        save_parquet(&path, &batch).unwrap();

        assert!(path.exists());
    }

    #[test]
    fn save_parquet_is_atomic_and_leaves_no_tmp_file() {
        let dir = tempdir().unwrap();
        let path = tmp_path_for(&dir, "atomic.parquet");
        let tmp_path = Utf8Path::new(&format!("{}.tmp", path)).to_path_buf();

        let batch = make_test_batch();
        save_parquet(&path, &batch).unwrap();

        assert!(path.exists());
        assert!(!tmp_path.exists(), "temporary file must be renamed away");
    }

    #[test]
    fn save_and_load_parquet_roundtrip() {
        let dir = tempdir().unwrap();
        let path = tmp_path_for(&dir, "roundtrip.parquet");

        let batch = make_test_batch();
        save_parquet(&path, &batch).unwrap();

        let (schema, batches) = load_parquet(&path).unwrap();

        // Schema should match.
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(schema.field(1).name(), "name");

        // Exactly one batch with 3 rows.
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 3);
        assert_eq!(batches[0].num_columns(), 2);

        // Verify column values.
        let ids = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("column 0 should be Int32Array");
        assert_eq!(ids.values(), &[1, 2, 3]);

        let names = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("column 1 should be StringArray");
        assert_eq!(names.value(0), "alpha");
        assert_eq!(names.value(1), "beta");
        assert_eq!(names.value(2), "gamma");
    }

    #[test]
    fn save_and_load_parquet_roundtrip_multi_type() {
        let dir = tempdir().unwrap();
        let path = tmp_path_for(&dir, "multi_type.parquet");

        let schema = Arc::new(Schema::new(vec![
            Field::new("track_id", DataType::Utf8, false),
            Field::new("dia_source_id", DataType::UInt64, false),
            Field::new("score", DataType::Float64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![
                    "TRK2026abc",
                    "TRK2026abc",
                    "TRK2026xyz",
                ])) as ArrayRef,
                Arc::new(UInt64Array::from(vec![100_u64, 200, 300])) as ArrayRef,
                Arc::new(Float64Array::from(vec![0.95, 0.87, 0.42])) as ArrayRef,
            ],
        )
        .unwrap();

        save_parquet(&path, &batch).unwrap();

        let (schema, batches) = load_parquet(&path).unwrap();
        assert_eq!(schema.fields().len(), 3);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 3);

        let track_ids = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(track_ids.value(0), "TRK2026abc");
        assert_eq!(track_ids.value(2), "TRK2026xyz");

        let dia_ids = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(dia_ids.value(0), 100);
        assert_eq!(dia_ids.value(1), 200);
        assert_eq!(dia_ids.value(2), 300);

        let scores = batches[0]
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!((scores.value(0) - 0.95).abs() < 1e-12);
        assert!((scores.value(2) - 0.42).abs() < 1e-12);
    }

    #[test]
    fn save_parquet_overwrites_existing_file() {
        let dir = tempdir().unwrap();
        let path = tmp_path_for(&dir, "overwrite.parquet");

        // Write first batch.
        let batch1 = make_test_batch();
        save_parquet(&path, &batch1).unwrap();

        // Write a different batch to the same path.
        let schema2 = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, false)]));
        let batch2 = RecordBatch::try_new(
            schema2,
            vec![Arc::new(Float64Array::from(vec![1.0, 2.0])) as ArrayRef],
        )
        .unwrap();
        save_parquet(&path, &batch2).unwrap();

        // Load should return the second batch.
        let (schema, batches) = load_parquet(&path).unwrap();
        assert_eq!(schema.fields().len(), 1);
        assert_eq!(schema.field(0).name(), "x");
        assert_eq!(batches[0].num_rows(), 2);
    }

    #[test]
    fn load_parquet_err_on_missing_file() {
        let dir = tempdir().unwrap();
        let path = tmp_path_for(&dir, "nonexistent.parquet");

        let err = load_parquet(&path).unwrap_err();
        assert!(
            matches!(err, PersistenceIoError::Io(_)),
            "expected Io error for missing file, got: {err:?}"
        );
    }

    #[test]
    fn load_parquet_err_on_corrupted_file() {
        let dir = tempdir().unwrap();
        let path = tmp_path_for(&dir, "corrupt.parquet");

        // Write garbage that is not valid Parquet.
        std::fs::write(path.as_std_path(), b"this is not parquet data").unwrap();

        let err = load_parquet(&path).unwrap_err();
        assert!(
            matches!(err, PersistenceIoError::Arrow(_)),
            "expected Arrow error for corrupted parquet, got: {err:?}"
        );
    }

    #[test]
    fn save_and_load_parquet_empty_batch() {
        let dir = tempdir().unwrap();
        let path = tmp_path_for(&dir, "empty.parquet");

        let schema = Arc::new(Schema::new(vec![Field::new("col", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(Vec::<i32>::new())) as ArrayRef],
        )
        .unwrap();

        save_parquet(&path, &batch).unwrap();

        let (schema, batches) = load_parquet(&path).unwrap();
        assert_eq!(schema.fields().len(), 1);

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 0, "empty batch should produce 0 rows on reload");
    }
}
