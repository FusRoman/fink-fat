//! Pure parsers for the sexagesimal RA/Dec strings Skybot's JSON conesearch
//! response uses (`"RA (hms)"`/`"DEC (dms)"`, e.g. `"09 57 19.2502"` /
//! `"+00 56 6.372"` — space-separated, confirmed against a live response;
//! `:`-separated input is accepted too in case a future response format
//! changes). No network or shared state is involved, so these are exercised
//! directly by the unit tests below rather than through a live request.

/// Parses a right-ascension string formatted as sexagesimal hours
/// (`"HH MM SS.ss"` or `"HH:MM:SS.ss"`) into decimal degrees.
///
/// Returns `None` if `s` doesn't split into exactly three numeric fields.
pub fn parse_ra_hms_to_deg(s: &str) -> Option<f64> {
    let (hours, minutes, seconds) = split_three_fields(s)?;
    Some((hours + minutes / 60.0 + seconds / 3600.0) * 15.0)
}

/// Parses a declination string formatted as signed sexagesimal degrees
/// (`"+DD MM SS.s"` or `"+DD:MM:SS.s"`) into decimal degrees.
///
/// Returns `None` if `s` doesn't split into exactly three numeric fields.
pub fn parse_dec_dms_to_deg(s: &str) -> Option<f64> {
    let is_negative = s.trim_start().starts_with('-');
    let (degrees, minutes, seconds) = split_three_fields(s)?;
    let magnitude = degrees.abs() + minutes / 60.0 + seconds / 3600.0;
    Some(if is_negative { -magnitude } else { magnitude })
}

/// Splits a sexagesimal string into three parsed `f64` fields. Fields may be
/// separated by whitespace, `:`, or a mix of both (Skybot's live response
/// uses spaces; `:` is accepted defensively) — `:` is normalized to a space
/// before splitting on whitespace.
fn split_three_fields(s: &str) -> Option<(f64, f64, f64)> {
    let normalized = s.trim().replace(':', " ");
    let mut parts = normalized.split_whitespace();
    let a: f64 = parts.next()?.parse().ok()?;
    let b: f64 = parts.next()?.parse().ok()?;
    let c: f64 = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((a, b, c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_space_separated_ra_hms_to_degrees() {
        // Skybot's actual live format, e.g. "09 57 19.2502".
        // (9 + 52/60 + 12.34/3600) * 15
        let deg = parse_ra_hms_to_deg("09 52 12.34").unwrap();
        assert!((deg - 148.051_416_666_666_66).abs() < 1e-9);
    }

    #[test]
    fn parses_colon_separated_ra_hms_to_degrees() {
        let deg = parse_ra_hms_to_deg("09:52:12.34").unwrap();
        assert!((deg - 148.051_416_666_666_66).abs() < 1e-9);
    }

    #[test]
    fn parses_positive_dec_dms_to_degrees() {
        // Skybot's actual live format, e.g. "+00 56 6.372".
        let deg = parse_dec_dms_to_deg("+16 23 01.2").unwrap();
        assert!((deg - 16.383_666_666_666_67).abs() < 1e-9);
    }

    #[test]
    fn parses_negative_dec_dms_to_degrees() {
        let deg = parse_dec_dms_to_deg("-16 23 01.2").unwrap();
        assert!((deg + 16.383_666_666_666_67).abs() < 1e-9);
    }

    #[test]
    fn rejects_malformed_input() {
        assert_eq!(parse_ra_hms_to_deg("not-a-time"), None);
        assert_eq!(parse_dec_dms_to_deg("12:34"), None);
        assert_eq!(parse_ra_hms_to_deg(""), None);
    }
}
