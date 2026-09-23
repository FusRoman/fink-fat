//! Builds MPC 1992 80-column ("obs80") observation records from a fink-fat
//! observation, for submission to the Minor Planet Center's Check
//! Near-Duplicates (CND) API
//! (<https://docs.minorplanetcenter.net/mpc-ops-docs/apis/cnd/>), which
//! expects observations in this fixed-width format rather than ADES.
//!
//! This is a faithful Rust port of the companion analysis project's
//! `python/mpc_obs80.py` (same column layout, same fixed TT-UTC offset, same
//! branch_id-derived placeholder designation, same rounding-carry fix — see
//! [`split_sexagesimal`]), so its regression tests are reproduced here
//! verbatim rather than re-derived.
//!
//! Column layout (1-based, 80 chars total), matching the CND docs' example
//! line (`"     K10CM6D  C2023 05 16.43686615 56 36.807-23 12 43.67         21.55wX~6o8oF51"`):
//!   * 1-5    blank (packed minor planet number — none, these are unpublished)
//!   * 6-12   packed provisional designation (7 chars)
//!   * 13     discovery asterisk (blank)
//!   * 14     Note 1 (blank)
//!   * 15     Note 2 (`"C"` — CCD)
//!   * 16-32  date of observation, UTC: `"YYYY MM DD.dddddd"` (17 chars)
//!   * 33-44  RA (J2000): `"HH MM SS.ddd"` (12 chars)
//!   * 45-56  Dec (J2000): `"sDD MM SS.ss"` (12 chars)
//!   * 57-65  blank (9 chars)
//!   * 66-70  magnitude: `"DD.dd"` (5 chars)
//!   * 71     band (1 char)
//!   * 72-77  blank (6 chars, reference field — unused here)
//!   * 78-80  observatory code (3 chars)
//!
//! fink-fat branches have no real MPC designation yet (that is the point of
//! the CND check), so columns 6-12 hold a placeholder derived from
//! `branch_id` instead of a real provisional designation — it makes each
//! obs80 line traceable back to its branch in the CND response, without
//! claiming a real MPC identity.

use chrono::{Datelike, Duration, NaiveDate, Timelike};

/// TT - UTC = 32.184s (TT - TAI) + 37s (TAI - UTC, current leap-second
/// count), same fixed offset the Python port uses — accurate to well under a
/// millisecond, since no leap second has been announced since 2017-01-01.
const TT_MINUS_UTC_S: f64 = 69.184;

const BASE36_DIGITS: &[u8; 36] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// Encodes `branch_id` as a 7-character, zero-padded base36 string, used as
/// a placeholder in the obs80 provisional-designation field (columns 6-12)
/// since these branches have no real MPC designation.
///
/// # Arguments
///
/// * `branch_id` — the branch to encode; always non-negative in practice
///   (`branches.branch_id` is a `BIGINT` but never stores a negative value).
///
/// # Return
///
/// A 7-character string. Values that don't fit in 7 base36 digits are
/// truncated to their least-significant 7 digits rather than erroring, same
/// as the Python original — `branch_id` never gets anywhere near
/// `36^7 ≈ 78` billion in practice.
pub fn pack_branch_id(branch_id: u64) -> String {
    let digits = if branch_id == 0 {
        "0".to_string()
    } else {
        let mut chars = Vec::new();
        let mut n = branch_id;
        while n > 0 {
            let r = (n % 36) as usize;
            chars.push(BASE36_DIGITS[r] as char);
            n /= 36;
        }
        chars.iter().rev().collect()
    };
    let padded = format!("{digits:0>7}");
    padded[padded.len() - 7..].to_string()
}

/// Converts `mjd_tt` to the obs80 date field: UTC `"YYYY MM DD.dddddd"` (17
/// chars).
fn format_obs80_date(mjd_tt: f64) -> String {
    let mjd_utc = mjd_tt - TT_MINUS_UTC_S / 86_400.0;
    let epoch = NaiveDate::from_ymd_opt(1858, 11, 17)
        .expect("1858-11-17 is a valid calendar date")
        .and_hms_opt(0, 0, 0)
        .expect("midnight is a valid time");
    let offset_ns = (mjd_utc * 86_400_000_000_000.0).round() as i64;
    let dt = epoch + Duration::nanoseconds(offset_ns);
    let seconds_since_midnight =
        dt.time().num_seconds_from_midnight() as f64 + dt.time().nanosecond() as f64 / 1e9;
    let day_fraction = dt.day() as f64 + seconds_since_midnight / 86_400.0;
    format!("{:04} {:02} {day_fraction:09.6}", dt.year(), dt.month())
}

/// Splits a non-negative value (hours or degrees) into `(units, minutes,
/// seconds_str)`, rounding seconds to `seconds_decimals` decimals.
///
/// Rounding a value very close to a minute/second boundary (e.g. `59.9999s`)
/// can make the *formatted* seconds field read `"60.00"` — invalid in
/// obs80 — so any such rollover is detected on the formatted string itself
/// (not by comparing the raw float to `60.0`, which would miss cases where
/// formatting alone pushes the value over the boundary) and carried into
/// minutes, then into `units` if minutes reaches 60. Single-level carry
/// only, same as the Python original: `minutes`/`seconds` are already
/// reduced into their normal ranges before formatting, so one bump can never
/// need a second.
fn split_sexagesimal(
    total_units: f64,
    seconds_width: usize,
    seconds_decimals: usize,
) -> (i64, i64, String) {
    let total_seconds = total_units * 3600.0;
    let units = (total_seconds / 3600.0).floor() as i64;
    let remainder = total_seconds - units as f64 * 3600.0;
    let mut minutes = (remainder / 60.0).floor() as i64;
    let seconds = remainder - minutes as f64 * 60.0;
    let mut seconds_str = format!("{seconds:0seconds_width$.seconds_decimals$}");
    let mut units = units;
    if seconds_str.starts_with("60") {
        seconds_str = format!("{:0seconds_width$.seconds_decimals$}", 0.0_f64);
        minutes += 1;
        if minutes == 60 {
            minutes = 0;
            units += 1;
        }
    }
    (units, minutes, seconds_str)
}

/// Converts `ra_deg` to the obs80 RA field: `"HH MM SS.ddd"` (12 chars).
fn format_obs80_ra(ra_deg: f64) -> String {
    let (hh, mm, ss_str) = split_sexagesimal(ra_deg / 15.0, 6, 3);
    format!("{hh:02} {mm:02} {ss_str}")
}

/// Converts `dec_deg` to the obs80 Dec field: `"sDD MM SS.ss"` (12 chars).
fn format_obs80_dec(dec_deg: f64) -> String {
    let sign = if dec_deg < 0.0 { '-' } else { '+' };
    let (dd, mm, ss_str) = split_sexagesimal(dec_deg.abs(), 5, 2);
    format!("{sign}{dd:02} {mm:02} {ss_str}")
}

/// Assembles one 80-column obs80 record for a single fink-fat observation.
///
/// # Arguments
///
/// * `branch_id` — packed into the placeholder designation field (see
///   [`pack_branch_id`]).
/// * `ra_deg`, `dec_deg` — J2000 equatorial position, degrees.
/// * `mjd_tt` — detection epoch, Modified Julian Date, Terrestrial Time.
/// * `magnitude` — apparent magnitude.
/// * `filter` — fink-fat's LSST `0..=5` band index
///   ([`crate::lsst_band::band_index_to_letter`]).
/// * `mpc_code_obs` — 3-character MPC observatory code.
///
/// # Return
///
/// The 80-character obs80 line.
///
/// # Panics
///
/// If `filter` is outside `0..=5` (no band letter to place in column 71) or
/// `mpc_code_obs` is not exactly 3 characters — both are invariants of the
/// `observations` table this is built from, not something a caller should
/// ever hit in practice.
pub fn build_obs80_line(
    branch_id: u64,
    ra_deg: f64,
    dec_deg: f64,
    mjd_tt: f64,
    magnitude: f64,
    filter: i16,
    mpc_code_obs: &str,
) -> String {
    let band = crate::lsst_band::band_index_to_letter(filter)
        .unwrap_or_else(|| panic!("obs80: unknown LSST filter code {filter}"));
    assert_eq!(
        mpc_code_obs.len(),
        3,
        "obs80: observatory code must be exactly 3 characters, got {mpc_code_obs:?}"
    );

    let line = format!(
        "{blank5}{designation} {discovery}{note2}{date}{ra}{dec}{blank9}{mag:>5.2}{band}{blank6}{obs_code:>3}",
        blank5 = " ".repeat(5),
        designation = pack_branch_id(branch_id),
        discovery = " ", // Note 1, blank
        note2 = "C",
        date = format_obs80_date(mjd_tt),
        ra = format_obs80_ra(ra_deg),
        dec = format_obs80_dec(dec_deg),
        blank9 = " ".repeat(9),
        mag = magnitude,
        band = band,
        blank6 = " ".repeat(6),
        obs_code = mpc_code_obs,
    );
    assert_eq!(
        line.len(),
        80,
        "obs80 line has {} chars, expected 80: {line:?}",
        line.len()
    );
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Example obs80 line from the CND API docs, used (like the Python
    /// port's tests) to pin down the exact column offsets for the
    /// date/RA/Dec/mag/band/obscode fields. Its designation ("K10CM6D") is a
    /// real MPC provisional designation, not a packed branch_id, so it's
    /// only used to derive the RA/Dec/date/mag/obscode test inputs below,
    /// not reproduced by `build_obs80_line`.
    fn utc_ymd_hms_frac_to_mjd_tt(year: i32, month: u32, day: u32, day_fraction: f64) -> f64 {
        let epoch = NaiveDate::from_ymd_opt(1858, 11, 17)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        let dt = NaiveDate::from_ymd_opt(year, month, day)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        let mjd_utc = (dt - epoch).num_seconds() as f64 / 86_400.0 + day_fraction;
        mjd_utc + TT_MINUS_UTC_S / 86_400.0
    }

    #[test]
    fn format_obs80_date_matches_the_cnd_docs_example() {
        let mjd_tt = utc_ymd_hms_frac_to_mjd_tt(2023, 5, 16, 0.436866);
        assert_eq!(format_obs80_date(mjd_tt), "2023 05 16.436866");
    }

    #[test]
    fn format_obs80_ra_matches_the_cnd_docs_example() {
        let ra_deg = 15.0 * (15.0 + 56.0 / 60.0 + 36.807 / 3600.0);
        assert_eq!(format_obs80_ra(ra_deg), "15 56 36.807");
    }

    #[test]
    fn format_obs80_dec_matches_the_cnd_docs_example() {
        let dec_deg = -(23.0 + 12.0 / 60.0 + 43.67 / 3600.0);
        assert_eq!(format_obs80_dec(dec_deg), "-23 12 43.67");
    }

    #[test]
    fn format_obs80_dec_uses_a_plus_sign_for_northern_declinations() {
        let dec_deg = 23.0 + 12.0 / 60.0 + 43.67 / 3600.0;
        assert_eq!(format_obs80_dec(dec_deg), "+23 12 43.67");
    }

    #[test]
    fn format_obs80_dec_rounds_seconds_up_to_60_carries_into_minutes() {
        // -11 17 59.9996 rounds to "60.00" at 2 decimals if not carried —
        // invalid per the MPC parser ("Invalid declination: -11 17 60.00").
        let dec_deg = -(11.0 + 17.0 / 60.0 + 59.9996 / 3600.0);
        assert_eq!(format_obs80_dec(dec_deg), "-11 18 00.00");
    }

    #[test]
    fn format_obs80_ra_rounds_seconds_up_to_60_carries_into_minutes() {
        let ra_deg = 15.0 * (21.0 + 54.0 / 60.0 + 59.9997 / 3600.0);
        assert_eq!(format_obs80_ra(ra_deg), "21 55 00.000");
    }

    #[test]
    fn format_obs80_dec_carries_minutes_up_to_60_into_degrees() {
        let dec_deg = -(11.0 + 59.0 / 60.0 + 59.9996 / 3600.0);
        assert_eq!(format_obs80_dec(dec_deg), "-12 00 00.00");
    }

    #[test]
    fn pack_branch_id_is_seven_chars_zero_padded() {
        assert_eq!(pack_branch_id(0), "0000000");
        assert_eq!(pack_branch_id(35), "000000Z");
        assert_eq!(pack_branch_id(36), "0000010");
    }

    #[test]
    fn build_obs80_line_has_expected_length_and_field_positions() {
        let mjd_tt = utc_ymd_hms_frac_to_mjd_tt(2023, 5, 16, 0.436866);
        let ra_deg = 15.0 * (15.0 + 56.0 / 60.0 + 36.807 / 3600.0);
        let dec_deg = -(23.0 + 12.0 / 60.0 + 43.67 / 3600.0);

        let line = build_obs80_line(42, ra_deg, dec_deg, mjd_tt, 21.55, 5, "X05");

        assert_eq!(line.len(), 80);
        assert_eq!(&line[5..12], pack_branch_id(42));
        assert_eq!(&line[15..32], "2023 05 16.436866");
        assert_eq!(&line[32..44], "15 56 36.807");
        assert_eq!(&line[44..56], "-23 12 43.67");
        assert_eq!(&line[65..70], "21.55");
        assert_eq!(&line[70..71], "y");
        assert_eq!(&line[77..80], "X05");
    }

    /// Round-trips `build_obs80_line`'s output through `photom`'s own obs80
    /// parser (`ObsDataset::from_mpc_80_col`) as an extra correctness check
    /// beyond the hand-picked regression cases above — photom has no obs80
    /// *writer* to call into directly (checked before writing this module),
    /// but its reader is a second, independently-implemented parser of the
    /// same spec, so a successful round-trip is good evidence the column
    /// layout is right.
    #[test]
    fn build_obs80_line_round_trips_through_photoms_obs80_parser() {
        use approx::assert_relative_eq;

        let ra_deg = 15.0 * (15.0 + 56.0 / 60.0 + 36.807 / 3600.0);
        let dec_deg = -(23.0 + 12.0 / 60.0 + 43.67 / 3600.0);
        let line = build_obs80_line(42, ra_deg, dec_deg, 60_080.5, 21.55, 5, "X05");

        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("round_trip.obs80");
        std::fs::write(&path, format!("{line}\n")).expect("write obs80 file");
        let utf8_path =
            camino::Utf8PathBuf::from_path_buf(path).expect("tempdir path is valid UTF-8");

        let dataset = photom::observation_dataset::ObsDataset::from_mpc_80_col(&utf8_path)
            .expect("photom should parse our own obs80 output");
        assert_eq!(dataset.observation_count(), 1);
        let obs = dataset.get_obs_by_index(0).expect("one observation");

        assert_relative_eq!(obs.equ_coord().ra.to_degrees(), ra_deg, epsilon = 1e-3);
        assert_relative_eq!(obs.equ_coord().dec.to_degrees(), dec_deg, epsilon = 1e-3);
        assert_relative_eq!(obs.photometry().magnitude, 21.55, epsilon = 1e-2);
    }
}
