//! Compression algorithms for binary persistence payloads.
//!
//! Overview
//! --------
//! This module provides the [`Compression`] enum and the associated
//! [`Compression::compress`] / [`Compression::decompress`] methods used by
//! [`crate::persistence::envelope::DiskEnvelope`] when writing and reading
//! binary blobs.
//!
//! The algorithm choice is embedded directly in the binary frame so that the
//! reader can decompress without being told which algorithm was used:
//!
//! ```text
//! [ algorithm : u8 (1 byte) ]
//! [ original_len : u64 LE   (8 bytes) ]
//! [ compressed_or_raw_bytes … ]
//! ```
//!
//! Supported algorithms
//! --------------------
//! | Variant         | Crate        | Speed  | Ratio  | Intended use |
//! |-----------------|--------------|--------|--------|--------------|
//! [`Compression::None`]  | —      | ∞      | 1×     | debugging, already-compressed payloads |
//! [`Compression::Lz4`]   | `lz4_flex` | ★★★★★ | ★★★  | high-frequency I/O, edge journals |
//! [`Compression::Zstd`]  | `zstd` | ★★★★   | ★★★★★  | nightly blobs, graph snapshots |
//! [`Compression::Gzip`]  | `flate2` | ★★   | ★★★★   | interoperability with external tools |
//!
//! Frame format
//! ------------
//! Every call to [`Compression::compress`] produces a *framed* byte vector
//! whose first 9 bytes contain:
//! - 1 byte: algorithm discriminant (see [`Compression::as_byte`]),
//! - 8 bytes: original (uncompressed) length as little-endian `u64`.
//!
//! [`Compression::decompress`] reads the header to determine the algorithm
//! and the expected output size, then decompresses the remainder.
//!
//! Configuration
//! -------------
//! The active algorithm is selected via
//! [`crate::engine_config::EngineConfig::binary_compression`] and stored in
//! the [`crate::persistence::envelope::DiskEnvelope`] at construction time.

use std::io::{Read, Write};

use serde::{Deserialize, Serialize};

use crate::persistence::error::PersistenceIoError;

// =============================================================================
// Frame layout constants
// =============================================================================

/// Number of bytes occupied by the fixed-size frame header.
///
/// Layout: `[ algorithm : u8 (1 B) | original_len : u64 LE (8 B) ]`.
const FRAME_HEADER_LEN: usize = 1 /* algorithm byte */ + 8 /* original_len u64 */;

// =============================================================================
// Compression enum
// =============================================================================

/// Compression algorithm applied to binary payloads before writing to disk.
///
/// The chosen variant is embedded in the on-disk frame header so that readers
/// can decompress without needing to know the algorithm in advance.
///
/// Variant choice guide
/// --------------------
/// - [`Compression::None`]  — raw bytes, zero overhead, useful for tests or
///   already-compressed data.
/// - [`Compression::Lz4`]   — ultra-fast, moderate ratio; best for frequently
///   written artifacts (edge deltas).
/// - [`Compression::Zstd`]  — best ratio at reasonable speed; recommended for
///   nightly blobs (alerts, seeds, snapshots).
/// - [`Compression::Gzip`]  — widely interoperable, slower; useful when files
///   may be consumed by external tools.
///
/// Notes
/// -----
/// - `#[non_exhaustive]` prevents downstream code from matching exhaustively,
///   making it safe to add new variants in future releases.
/// - The default is [`Compression::None`] so that the engine works without
///   any compression dependency by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum Compression {
    /// No compression — payload bytes are stored as-is.
    #[default]
    None,

    /// LZ4 frame format (via `lz4_flex`).
    ///
    /// Excellent write/read throughput, moderate compression ratio. The
    /// `lz4_flex` crate is used with its *prepend-size* framing so that the
    /// decompressor can allocate the exact output buffer up front.
    Lz4,

    /// Zstandard compression (via `zstd`).
    ///
    /// Best compression ratio among the provided variants at a still-reasonable
    /// CPU cost. Uses the default compression level (0, which maps to 3 in the
    /// `zstd` library).
    Zstd,

    /// Gzip / DEFLATE compression (via `flate2`).
    ///
    /// Widely supported by external tools (`gzip`, Python, etc.). Slower than
    /// LZ4 or Zstd but the resulting files can be decompressed without Rust.
    Gzip,
}

impl Compression {
    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /// Encode the variant as a single discriminant byte stored in the frame.
    #[inline]
    pub(crate) fn as_byte(self) -> u8 {
        match self {
            Compression::None => 0,
            Compression::Lz4 => 1,
            Compression::Zstd => 2,
            Compression::Gzip => 3,
        }
    }

    /// Decode a variant from the discriminant byte read from the frame.
    ///
    /// Errors
    /// ------
    /// Returns [`PersistenceIoError::Compression`] for unknown byte values.
    #[inline]
    fn from_byte(b: u8) -> Result<Self, PersistenceIoError> {
        match b {
            0 => Ok(Compression::None),
            1 => Ok(Compression::Lz4),
            2 => Ok(Compression::Zstd),
            3 => Ok(Compression::Gzip),
            _ => Err(PersistenceIoError::Compression(format!(
                "unknown compression algorithm discriminant: {b:#04x}"
            ))),
        }
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /// Compress `data` and wrap the result in the binary frame.
    ///
    /// The returned buffer starts with the 9-byte header
    /// `[ algorithm (1B) | original_len_le (8B) ]` followed by the
    /// compressed (or raw) bytes.
    ///
    /// Arguments
    /// ---------
    /// * `data` — Raw bytes to compress.
    ///
    /// Return
    /// ------
    /// Framed byte vector ready to be written atomically to disk.
    ///
    /// Errors
    /// ------
    /// [`PersistenceIoError::Compression`] if the compression operation fails.
    pub fn compress(self, data: &[u8]) -> Result<Vec<u8>, PersistenceIoError> {
        let compressed: Vec<u8> = match self {
            Compression::None => data.to_vec(),

            Compression::Lz4 => lz4_flex::compress_prepend_size(data),

            Compression::Zstd => zstd::encode_all(data, 0 /* default level */)
                .map_err(|e| PersistenceIoError::Compression(e.to_string()))?,

            Compression::Gzip => {
                use flate2::{Compression as GzLevel, write::GzEncoder};
                let mut enc = GzEncoder::new(Vec::new(), GzLevel::default());
                enc.write_all(data)
                    .map_err(|e| PersistenceIoError::Compression(e.to_string()))?;
                enc.finish()
                    .map_err(|e| PersistenceIoError::Compression(e.to_string()))?
            }
        };

        // Build the framed output: [ algo:u8 | orig_len:u64 LE | compressed ]
        let original_len = data.len() as u64;
        let mut framed = Vec::with_capacity(FRAME_HEADER_LEN + compressed.len());
        framed.push(self.as_byte());
        framed.extend_from_slice(&original_len.to_le_bytes());
        framed.extend_from_slice(&compressed);
        Ok(framed)
    }

    /// Decompress framed bytes produced by [`Compression::compress`].
    ///
    /// The algorithm is read from the first byte of `framed`; callers do not
    /// need to know which algorithm was used at write time.
    ///
    /// Arguments
    /// ---------
    /// * `framed` — Byte slice beginning with the 9-byte frame header.
    ///
    /// Return
    /// ------
    /// Decompressed bytes.
    ///
    /// Errors
    /// ------
    /// - [`PersistenceIoError::Compression`] if the header is truncated, the
    ///   algorithm byte is unknown, or the decompression itself fails.
    pub fn decompress(framed: &[u8]) -> Result<Vec<u8>, PersistenceIoError> {
        if framed.len() < FRAME_HEADER_LEN {
            return Err(PersistenceIoError::Compression(format!(
                "framed payload is too short to contain the {FRAME_HEADER_LEN}-byte header \
                 (got {} bytes)",
                framed.len()
            )));
        }

        let algo = Compression::from_byte(framed[0])?;
        let original_len = u64::from_le_bytes(framed[1..9].try_into().unwrap()) as usize;
        let payload = &framed[FRAME_HEADER_LEN..];

        let out: Vec<u8> = match algo {
            Compression::None => payload.to_vec(),

            Compression::Lz4 => lz4_flex::decompress_size_prepended(payload)
                .map_err(|e| PersistenceIoError::Compression(e.to_string()))?,

            Compression::Zstd => zstd::decode_all(payload)
                .map_err(|e| PersistenceIoError::Compression(e.to_string()))?,

            Compression::Gzip => {
                use flate2::read::GzDecoder;
                let mut dec = GzDecoder::new(payload);
                let mut buf = Vec::with_capacity(original_len);
                dec.read_to_end(&mut buf)
                    .map_err(|e| PersistenceIoError::Compression(e.to_string()))?;
                buf
            }
        };

        Ok(out)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod compression_tests {
    use super::*;

    /// Round-trip helper: compress then decompress `data` with `algo`.
    fn roundtrip(algo: Compression, data: &[u8]) {
        let framed = algo.compress(data).expect("compress must succeed");
        let got = Compression::decompress(&framed).expect("decompress must succeed");
        assert_eq!(
            got,
            data,
            "roundtrip failed for {:?} with {} bytes",
            algo,
            data.len()
        );
    }

    #[test]
    fn roundtrip_empty_all_algorithms() {
        for algo in [
            Compression::None,
            Compression::Lz4,
            Compression::Zstd,
            Compression::Gzip,
        ] {
            roundtrip(algo, b"");
        }
    }

    #[test]
    fn roundtrip_small_payload_all_algorithms() {
        let data = b"Hello, Fink-FAT compression!";
        for algo in [
            Compression::None,
            Compression::Lz4,
            Compression::Zstd,
            Compression::Gzip,
        ] {
            roundtrip(algo, data);
        }
    }

    #[test]
    fn roundtrip_large_repetitive_payload() {
        let data: Vec<u8> = (0..100_000_u64).flat_map(|i| i.to_le_bytes()).collect();
        for algo in [
            Compression::None,
            Compression::Lz4,
            Compression::Zstd,
            Compression::Gzip,
        ] {
            roundtrip(algo, &data);
        }
    }

    #[test]
    fn zstd_and_lz4_produce_smaller_output_for_repetitive_data() {
        let data: Vec<u8> = vec![42u8; 100_000];
        for algo in [Compression::Lz4, Compression::Zstd, Compression::Gzip] {
            let framed = algo.compress(&data).unwrap();
            assert!(
                framed.len() < data.len(),
                "{algo:?}: framed ({}) should be smaller than raw ({})",
                framed.len(),
                data.len()
            );
        }
    }

    #[test]
    fn decompress_err_on_truncated_header() {
        // Only 4 bytes — not enough for the 9-byte header.
        let err = Compression::decompress(b"\x01\x00\x00\x00").unwrap_err();
        assert!(matches!(err, PersistenceIoError::Compression(_)));
    }

    #[test]
    fn decompress_err_on_unknown_algorithm_byte() {
        // Build a well-formed 9-byte header with an unknown algo byte (0xFF).
        let mut framed = vec![0xFF_u8];
        framed.extend_from_slice(&0u64.to_le_bytes()); // original_len = 0
        let err = Compression::decompress(&framed).unwrap_err();
        assert!(matches!(err, PersistenceIoError::Compression(_)));
    }

    #[test]
    fn none_compression_is_identity() {
        let data = b"identity test";
        let framed = Compression::None.compress(data).unwrap();
        // The payload portion (after 9-byte header) should equal `data`.
        assert_eq!(&framed[FRAME_HEADER_LEN..], data);
        let got = Compression::decompress(&framed).unwrap();
        assert_eq!(got, data);
    }

    #[test]
    fn frame_header_algorithm_byte_matches_variant() {
        for (algo, expected_byte) in [
            (Compression::None, 0u8),
            (Compression::Lz4, 1u8),
            (Compression::Zstd, 2u8),
            (Compression::Gzip, 3u8),
        ] {
            let framed = algo.compress(b"").unwrap();
            assert_eq!(
                framed[0], expected_byte,
                "algorithm byte mismatch for {algo:?}"
            );
        }
    }

    #[test]
    fn frame_header_original_len_is_correct() {
        let data = vec![7u8; 1234];
        for algo in [
            Compression::None,
            Compression::Lz4,
            Compression::Zstd,
            Compression::Gzip,
        ] {
            let framed = algo.compress(&data).unwrap();
            let stored_len = u64::from_le_bytes(framed[1..9].try_into().unwrap());
            assert_eq!(stored_len, 1234, "{algo:?}: stored original_len mismatch");
        }
    }
}
