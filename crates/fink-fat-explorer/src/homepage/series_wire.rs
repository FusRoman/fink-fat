//! Binary wire format of the (a, e) population plot's series.
//!
//! The server encodes the snapshot's [`PlotSeries`] once per build (see
//! [`crate::homepage::snapshot`]) and serves the bytes as-is from a plain GET
//! route; the wasm client decodes them back into the exact same
//! `Vec<PlotSeries>`. Encoder and decoder live side by side here, compiled for
//! both targets, so the two ends cannot drift apart.
//!
//! Compared to the former JSON payload this ships `a`/`e` as raw
//! little-endian `f32` (4 bytes per value instead of a ~10-character decimal
//! that the client had to parse back), bit-for-bit identical to what the
//! snapshot holds — the hover text, which prints both to 3 decimals, is
//! unchanged. Designations still travel with the points since hover and
//! click both need them as soon as the plot is drawn.
//!
//! Layout (all integers little-endian):
//!
//! | Field            | Size              | Content                                         |
//! |------------------|-------------------|-------------------------------------------------|
//! | magic            | 4                 | [`MAGIC`], format version included              |
//! | header length    | 4 (`u32`)         | byte length of the header that follows          |
//! | header           | header length     | JSON `[{"family", "tier", "len"}, ...]`, one entry per series |
//! | series bodies    | rest              | per series, in header order: `len` × `f32` a, `len` × `f32` e, then `len` × (`u32` byte length + UTF-8 designation) |
//!
//! The header reuses the serde names of [`DynamicalFamily`] and
//! [`QualityTier`], so the format needs no numbering of its own for either
//! enum.

use serde::{Deserialize, Serialize};

use crate::homepage::dynamic_pop_plot::PlotSeries;
use crate::homepage::family::DynamicalFamily;
use crate::homepage::quality_tier::QualityTier;

/// URL path of the plain GET route serving the encoded payload (see
/// `homepage::series_route`). Shared by the route and the client fetch.
pub const POPULATION_SERIES_PATH: &str = "/api/homepage/population-series";

/// Leading bytes of every payload: a format tag plus its version. Bump the
/// last byte on any layout change so a stale client fails loudly on
/// [`SeriesWireError::BadMagic`] instead of misreading the body.
pub const MAGIC: [u8; 4] = *b"FFP\x01";

/// Why a payload could not be decoded.
#[derive(Debug, thiserror::Error)]
pub enum SeriesWireError {
    /// The payload does not start with [`MAGIC`]: not a series payload, or
    /// one written by another format version.
    #[error("not a population series payload (bad magic or format version)")]
    BadMagic,
    /// The payload ends before a field it announces.
    #[error("population series payload truncated while reading {0}")]
    Truncated(&'static str),
    /// The JSON header is malformed.
    #[error("invalid population series header: {0}")]
    Header(#[from] serde_json::Error),
    /// A designation is not valid UTF-8.
    #[error("invalid UTF-8 in a lineage designation: {0}")]
    Utf8(#[from] std::str::Utf8Error),
    /// Bytes are left over once every announced series has been read.
    #[error("{0} unexpected trailing bytes after the population series")]
    TrailingBytes(usize),
    /// Encoding only: a length does not fit the format's `u32` length fields.
    #[error("{0} too long for the population series format")]
    TooLong(&'static str),
}

/// Per-series header entry: everything constant across a series, plus its
/// point count.
#[derive(Serialize, Deserialize)]
struct SeriesHeader {
    family: DynamicalFamily,
    tier: QualityTier,
    len: u32,
}

/// Encodes the plot series into the binary wire format.
///
/// # Arguments
/// * `series` — the series to encode; every series' `a`, `e` and
///   `lineage_designations` must have the same length (as
///   `snapshot::build_series` guarantees).
///
/// # Return
/// The encoded payload, ready to be served as-is.
///
/// # Errors
/// [`SeriesWireError::TooLong`] if a series, a designation or the header is
/// longer than `u32::MAX` (bytes or points), [`SeriesWireError::Header`] if
/// the header fails to serialize. Neither can happen at homepage scale.
pub fn encode_series(series: &[PlotSeries]) -> Result<Vec<u8>, SeriesWireError> {
    let header = series
        .iter()
        .map(|s| {
            Ok(SeriesHeader {
                family: s.family,
                tier: s.tier,
                len: len_u32(s.a.len(), "series")?,
            })
        })
        .collect::<Result<Vec<SeriesHeader>, SeriesWireError>>()?;
    let header = serde_json::to_vec(&header)?;

    let body_len: usize = series
        .iter()
        .map(|s| {
            8 * s.a.len()
                + s.lineage_designations
                    .iter()
                    .map(|d| 4 + d.len())
                    .sum::<usize>()
        })
        .sum();
    let mut out = Vec::with_capacity(MAGIC.len() + 4 + header.len() + body_len);

    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&len_u32(header.len(), "series header")?.to_le_bytes());
    out.extend_from_slice(&header);
    for s in series {
        debug_assert_eq!(s.a.len(), s.e.len());
        debug_assert_eq!(s.a.len(), s.lineage_designations.len());
        out.extend(s.a.iter().flat_map(|v| v.to_le_bytes()));
        out.extend(s.e.iter().flat_map(|v| v.to_le_bytes()));
        for designation in &s.lineage_designations {
            out.extend_from_slice(&len_u32(designation.len(), "designation")?.to_le_bytes());
            out.extend_from_slice(designation.as_bytes());
        }
    }
    Ok(out)
}

/// Decodes a payload produced by [`encode_series`].
///
/// # Arguments
/// * `bytes` — the whole payload.
///
/// # Return
/// The series, in the order they were encoded.
///
/// # Errors
/// [`SeriesWireError`] if the payload is not in this format version, is
/// truncated, has a malformed header or designation, or carries trailing
/// bytes.
pub fn decode_series(bytes: &[u8]) -> Result<Vec<PlotSeries>, SeriesWireError> {
    let mut reader = Reader { bytes };

    if reader.take(MAGIC.len(), "magic")? != MAGIC {
        return Err(SeriesWireError::BadMagic);
    }
    let header_len = reader.u32("header length")? as usize;
    let header: Vec<SeriesHeader> = serde_json::from_slice(reader.take(header_len, "header")?)?;

    let series = header
        .into_iter()
        .map(|h| {
            let len = h.len as usize;
            let a = reader.f32s(len, "semi-major axes")?;
            let e = reader.f32s(len, "eccentricities")?;
            let lineage_designations = (0..len)
                .map(|_| {
                    let n = reader.u32("designation length")? as usize;
                    let designation = std::str::from_utf8(reader.take(n, "designation")?)?;
                    Ok(designation.into())
                })
                .collect::<Result<Vec<Box<str>>, SeriesWireError>>()?;
            Ok(PlotSeries {
                family: h.family,
                tier: h.tier,
                a,
                e,
                lineage_designations,
            })
        })
        .collect::<Result<Vec<PlotSeries>, SeriesWireError>>()?;

    match reader.bytes.len() {
        0 => Ok(series),
        n => Err(SeriesWireError::TrailingBytes(n)),
    }
}

/// `usize` -> `u32` for the length field of `what`.
fn len_u32(len: usize, what: &'static str) -> Result<u32, SeriesWireError> {
    u32::try_from(len).map_err(|_| SeriesWireError::TooLong(what))
}

/// Forward-only cursor over the payload; every read names the field it is
/// reading so a truncation error says where the payload stopped.
struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize, field: &'static str) -> Result<&'a [u8], SeriesWireError> {
        if self.bytes.len() < n {
            return Err(SeriesWireError::Truncated(field));
        }
        let (head, tail) = self.bytes.split_at(n);
        self.bytes = tail;
        Ok(head)
    }

    fn u32(&mut self, field: &'static str) -> Result<u32, SeriesWireError> {
        let raw = self.take(4, field)?;
        Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
    }

    fn f32s(&mut self, len: usize, field: &'static str) -> Result<Vec<f32>, SeriesWireError> {
        let raw = self.take(
            len.checked_mul(4)
                .ok_or(SeriesWireError::Truncated(field))?,
            field,
        )?;
        // `raw.len()` is exactly `len * 4`, so the remainder is always empty.
        let (words, _) = raw.as_chunks::<4>();
        Ok(words.iter().map(|w| f32::from_le_bytes(*w)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<PlotSeries> {
        vec![
            PlotSeries {
                family: DynamicalFamily::NeaApollo,
                tier: QualityTier::Discovery,
                a: vec![1.5, 2.25],
                e: vec![0.5, 0.125],
                lineage_designations: vec!["FF2026 AB1".into(), "é/ü with spaces".into()],
            },
            // Empty series and edge-case floats must survive too.
            PlotSeries {
                family: DynamicalFamily::KboSdo,
                tier: QualityTier::Ineligible,
                a: vec![],
                e: vec![],
                lineage_designations: vec![],
            },
            PlotSeries {
                family: DynamicalFamily::Trojan,
                tier: QualityTier::NotFitted,
                a: vec![f32::MAX, f32::MIN_POSITIVE, f32::NAN],
                e: vec![0.0, -0.0, f32::INFINITY],
                lineage_designations: vec!["".into(), "a\nb".into(), "x".into()],
            },
        ]
    }

    /// `PartialEq` would fail on the NaN above, so compare the bit patterns.
    fn assert_same(decoded: &[PlotSeries], expected: &[PlotSeries]) {
        assert_eq!(decoded.len(), expected.len());
        for (d, x) in decoded.iter().zip(expected) {
            assert_eq!((d.family, d.tier), (x.family, x.tier));
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&d.a), bits(&x.a));
            assert_eq!(bits(&d.e), bits(&x.e));
            assert_eq!(d.lineage_designations, x.lineage_designations);
        }
    }

    #[test]
    fn round_trip_is_bit_exact() {
        let series = sample();
        let decoded = decode_series(&encode_series(&series).expect("encodes")).expect("decodes");
        assert_same(&decoded, &series);
    }

    #[test]
    fn empty_population_round_trips() {
        let decoded = decode_series(&encode_series(&[]).expect("encodes")).expect("decodes");
        assert!(decoded.is_empty());
    }

    #[test]
    fn encoding_is_deterministic() {
        assert_eq!(
            encode_series(&sample()).expect("encodes"),
            encode_series(&sample()).expect("encodes")
        );
    }

    #[test]
    fn other_format_version_is_rejected() {
        let mut bytes = encode_series(&sample()).expect("encodes");
        bytes[3] = bytes[3].wrapping_add(1);
        assert!(matches!(
            decode_series(&bytes),
            Err(SeriesWireError::BadMagic)
        ));
    }

    #[test]
    fn every_truncation_is_an_error_not_a_panic() {
        let bytes = encode_series(&sample()).expect("encodes");
        for cut in 0..bytes.len() {
            assert!(
                decode_series(&bytes[..cut]).is_err(),
                "payload cut at {cut} bytes decoded successfully"
            );
        }
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let mut bytes = encode_series(&sample()).expect("encodes");
        bytes.push(0);
        assert!(matches!(
            decode_series(&bytes),
            Err(SeriesWireError::TrailingBytes(1))
        ));
    }

    #[test]
    fn invalid_utf8_designation_is_rejected() {
        let series = vec![PlotSeries {
            family: DynamicalFamily::MbInner,
            tier: QualityTier::NotFitted,
            a: vec![2.5],
            e: vec![0.1],
            lineage_designations: vec!["ab".into()],
        }];
        let mut bytes = encode_series(&series).expect("encodes");
        let last = bytes.len() - 1;
        bytes[last] = 0xFF;
        assert!(matches!(
            decode_series(&bytes),
            Err(SeriesWireError::Utf8(_))
        ));
    }
}
