//! # Human-friendly unit parsing for YAML configuration (`units`)
//!
//! This module provides **serde deserialization helpers** to parse quantities
//! from configuration files (typically YAML) in a human-friendly way.
//!
//! The goal is to allow end-users to write either:
//! - **raw numeric values** already expressed in the engine’s canonical units, or
//! - **strings with explicit units**, such as `"35 arcmin/day"`, `"2 arcsec/hour"`,
//!   `"90 min"`, `"1.5 deg"`.
//!
//! The engine internally uses canonical units, and the deserializers convert
//! any supported input representation into those canonical units.
//!
//! -----------------------------------------------------------------------------
//! Canonical internal units
//! -----------------------------------------------------------------------------
//!
//! All conversions performed by this module target the following internal units:
//!
//! - **Angles**: radians
//! - **Time**: days
//! - **Angular speed**: radians per day
//!
//! The public entry points (`de_*`) always return `f64` in canonical units.
//!
//! -----------------------------------------------------------------------------
//! Accepted YAML syntaxes
//! -----------------------------------------------------------------------------
//!
//! For a field declared as `deserialize_with = "...":`
//!
//! 1) Numeric input (already in canonical units)
//! --------------------------------------------
//! The YAML value can be a number:
//!
//! ```yaml
//! pairs:
//!   max_dt: 0.06
//!   max_angular_speed: 5.0e-2
//! ```
//!
//! In this case, the value is interpreted as already being in canonical units:
//! - `max_dt` is taken as **days**,
//! - `max_angular_speed` is taken as **rad/day**.
//!
//! 2) String input with units
//! --------------------------
//! The YAML value can be a string containing a number and a unit:
//!
//! ```yaml
//! pairs:
//!   max_dt: "86.4 min"
//!   max_angular_speed: "35 arcmin/day"
//! triplets:
//!   max_pair_sep: "8.6 arcmin"
//!   max_predicted_residual: "2.75 arcmin"
//! ```
//!
//! The module supports both `"value unit"` and a best-effort `"valueunit"` form:
//! - `"0.05 rad"` and `"0.05rad"` are both accepted,
//! - `"35 arcmin"` and `"35arcmin"` are both accepted.
//!
//! For angular speeds, the expected syntax is:
//!
//! ```text
//! <angle> / <time>
//! ```
//!
//! Examples:
//! - `"35 arcmin/day"`
//! - `"2 arcsec / hour"`
//! - `"0.05 rad/day"`
//!
//! Notes:
//! - Exactly one `'/'` separator must be present for angular speeds.
//! - The numerator must be a valid angle quantity (value + angle unit).
//! - The denominator must be a time unit (optionally prefixed by `"per"`).
//!
//! -----------------------------------------------------------------------------
//! Supported units
//! -----------------------------------------------------------------------------
//!
//! Angle units
//! ----------
//! Accepted angle units (case-insensitive):
//! - `"rad"`, `"radian"`, `"radians"`
//! - `"deg"`, `"degree"`, `"degrees"`
//! - `"arcmin"`, `"arcminute"`, `"arcminutes"`
//! - `"arcsec"`, `"arcsecond"`, `"arcseconds"`
//!
//! Basic French aliases (best-effort):
//! - `"degre"`, `"degres"` (treated as degrees)
//! - any string containing both `"minute"` and `"arc"` is normalized to
//!   a private `"minute-d-arc"` token and treated as arcminutes
//! - any string containing both `"seconde"` and `"arc"` is normalized to
//!   a private `"seconde-d-arc"` token and treated as arcseconds
//!
//! The French handling is intentionally minimal: it is not a full locale system,
//! only a convenience for common notations like `"minute d'arc"` / `"seconde d'arc"`.
//!
//! Time units
//! ----------
//! Accepted time units (case-insensitive):
//! - `"day"`, `"days"`, `"d"`, `"jour"`, `"jours"`
//! - `"hour"`, `"hours"`, `"h"`, `"heure"`, `"heures"`
//! - `"min"`, `"minute"`, `"minutes"`
//! - `"sec"`, `"second"`, `"seconds"`, `"s"`, `"seconde"`, `"secondes"`
//!
//! For angular speeds, the denominator supports the same time units.
//!
//! -----------------------------------------------------------------------------
//! Error handling and validation philosophy
//! -----------------------------------------------------------------------------
//!
//! This module is intended to surface configuration mistakes early.
//! When parsing fails, the returned error message attempts to be explicit about:
//! - what unit was unsupported,
//! - what the expected unit families are,
//! - or what part of the syntax is malformed.
//!
//! Typical error cases include:
//! - unknown/unsupported units (`"foobar"`),
//! - missing unit in a string quantity (`"12"` as a string instead of a number),
//! - malformed angular speed strings (missing `'/'`, too many separators, missing denominator).
//!
//! -----------------------------------------------------------------------------
//! Implementation notes
//! -----------------------------------------------------------------------------
//!
//! - The deserialization accepts `f64` numeric values or strings via an untagged
//!   enum (`NumOrStr`).
//! - `normalize_unit()` lowercases and does minimal punctuation normalization,
//!   including replacing typographic apostrophes `’` by `'`.
//! - `"valueunit"` parsing is implemented by scanning for the longest prefix
//!   that parses as `f64`, and treating the remainder as the unit string.
//!
//! -----------------------------------------------------------------------------
//! Public API
//! -----------------------------------------------------------------------------
//!
//! The intended usage is through `serde` field attributes:
//!
//! ```rust, ignore
//! #[serde(deserialize_with = "de_time_days")]
//! pub max_dt: f64;
//!
//! #[serde(deserialize_with = "de_angle_rad")]
//! pub max_pair_sep: f64;
//!
//! #[serde(deserialize_with = "de_ang_speed_rad_per_day")]
//! pub max_angular_speed: f64;
//! ```
//!
//! Each function converts the YAML value into canonical units:
//! - [`de_time_days`]: time → days
//! - [`de_angle_rad`]: angle → radians
//! - [`de_ang_speed_rad_per_day`]: angular speed → radians/day

use serde::Deserialize;
use serde::de;

/// Serde helper: accept either a number (`f64`) or a string (`String`).
///
/// This enum is used to support human-friendly YAML:
/// - numeric scalars are assumed to already be in canonical units,
/// - string scalars are parsed with unit suffixes.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum NumOrStr {
    /// Raw numeric value (already in canonical units).
    Num(f64),
    /// String quantity with explicit unit(s), e.g. `"90 min"` or `"35 arcmin/day"`.
    Str(String),
}

/* -------------------------------------------------------------------------- */
/*  Public serde entry points (deserialize_with = ...)                         */
/* -------------------------------------------------------------------------- */

/// Deserialize a time quantity into **days** (`f64`).
///
/// Accepted inputs
/// ---------------
/// - Numeric YAML scalar (assumed **days**):
///   - `0.06`
/// - String with explicit time unit:
///   - `"86.4 min"`, `"1.5 hour"`, `"30 sec"`, `"0.06 day"`, `"1 jour"`.
///
/// Errors
/// ------
/// Returns a serde error if:
/// - the string cannot be parsed as `<value> <unit>` (or `<value><unit>`),
/// - or the unit is not a supported time unit.
pub fn de_time_days<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    match NumOrStr::deserialize(deserializer)? {
        NumOrStr::Num(x) => Ok(x),
        NumOrStr::Str(s) => parse_time_days(&s).map_err(de::Error::custom),
    }
}

/// Deserialize an angle quantity into **radians** (`f64`).
///
/// Accepted inputs
/// ---------------
/// - Numeric YAML scalar (assumed **radians**):
///   - `2.5e-3`
/// - String with explicit angle unit:
///   - `"8.6 arcmin"`, `"0.1 deg"`, `"0.05 rad"`, `"1e-3deg"`.
///
/// Errors
/// ------
/// Returns a serde error if:
/// - the string cannot be parsed as `<value> <unit>` (or `<value><unit>`),
/// - or the unit is not a supported angle unit.
pub fn de_angle_rad<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    match NumOrStr::deserialize(deserializer)? {
        NumOrStr::Num(x) => Ok(x),
        NumOrStr::Str(s) => parse_angle_rad(&s).map_err(de::Error::custom),
    }
}

/// Deserialize an angular speed into **radians/day** (`f64`).
///
/// Accepted inputs
/// ---------------
/// - Numeric YAML scalar (assumed **radians/day**):
///   - `5.0e-2`
/// - String of the form `<angle>/<time>`, with optional spaces:
///   - `"35 arcmin/day"`
///   - `"2 arcsec / hour"`
///   - `"0.05 rad/day"`
///
/// The denominator may optionally start with `"per"`:
/// - `"35 arcmin/per day"`
/// - `"2 arcsec/per hour"`
///
/// Errors
/// ------
/// Returns a serde error if:
/// - the string is missing `'/'`,
/// - has more than one `'/'`,
/// - the numerator cannot be parsed as an angle quantity,
/// - or the denominator unit is not a supported time unit.
pub fn de_ang_speed_rad_per_day<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    match NumOrStr::deserialize(deserializer)? {
        NumOrStr::Num(x) => Ok(x),
        NumOrStr::Str(s) => parse_ang_speed_rad_per_day(&s).map_err(de::Error::custom),
    }
}

/* -------------------------------------------------------------------------- */
/*  Parsers                                                                    */
/* -------------------------------------------------------------------------- */

/// Parse a time quantity and return the value expressed in **days**.
///
/// Expected syntax:
/// - `"<value> <unit>"` (preferred)
/// - `"<value><unit>"` (best-effort)
///
/// Supported units are documented at the module level.
fn parse_time_days(input: &str) -> Result<f64, String> {
    let (value, unit) = split_value_unit(input)?;
    let u = normalize_unit(&unit);

    let days = match u.as_str() {
        "d" | "day" | "days" | "jour" | "jours" => value,
        "h" | "hour" | "hours" | "heure" | "heures" => value / 24.0,
        "min" | "minute" | "minutes" => value / (24.0 * 60.0),
        "s" | "sec" | "second" | "seconds" | "seconde" | "secondes" => value / (24.0 * 3600.0),
        _ => {
            return Err(format!(
                "Unsupported time unit '{unit}'. Expected one of: day, hour, min, sec (and French aliases)."
            ));
        }
    };

    Ok(days)
}

/// Parse an angle quantity and return the value expressed in **radians**.
///
/// Expected syntax:
/// - `"<value> <unit>"` (preferred)
/// - `"<value><unit>"` (best-effort)
///
/// Supported units are documented at the module level.
fn parse_angle_rad(input: &str) -> Result<f64, String> {
    let (value, unit) = split_value_unit(input)?;
    let u = normalize_unit(&unit);

    let rad = match u.as_str() {
        "rad" | "radian" | "radians" => value,
        "deg" | "degree" | "degrees" | "degre" | "degres" => value.to_radians(),
        "arcmin" | "arcminute" | "arcminutes" => (value / 60.0).to_radians(),
        "arcsec" | "arcsecond" | "arcseconds" => (value / 3600.0).to_radians(),
        // very lightweight French-ish support (optional / best effort)
        "minute-d-arc" => (value / 60.0).to_radians(),
        "seconde-d-arc" => (value / 3600.0).to_radians(),
        _ => {
            return Err(format!(
                "Unsupported angle unit '{unit}'. Expected one of: rad, deg, arcmin, arcsec (plus basic French aliases)."
            ));
        }
    };

    Ok(rad)
}

/// Parse an angular speed and return the value expressed in **radians/day**.
///
/// Expected syntax:
/// ```text
/// <angle> / <time>
/// ```
///
/// Examples:
/// - `"35 arcmin/day"`
/// - `"2 arcsec / hour"`
/// - `"0.05 rad/day"`
///
/// Numerator parsing:
/// - must be a valid angle quantity accepted by [`parse_angle_rad`].
///
/// Denominator parsing:
/// - must be a supported time unit (optionally prefixed by `"per"`).
fn parse_ang_speed_rad_per_day(input: &str) -> Result<f64, String> {
    // Accept "35 arcmin/day" or "35 arcmin / day"
    let s = input.trim();

    // Split on '/', but allow it to be missing spaces
    let mut parts = s.split('/').map(str::trim);
    let left = parts
        .next()
        .ok_or_else(|| "Invalid angular speed: missing numerator".to_string())?;
    let right = parts
        .next()
        .ok_or_else(|| "Invalid angular speed: missing denominator (e.g. '/day')".to_string())?;

    if parts.next().is_some() {
        return Err("Invalid angular speed: too many '/' separators".to_string());
    }

    // Left side is "<value> <angle_unit>"
    let angle_rad = parse_angle_rad(left)?;

    // Right side is "<time_unit>" (optionally with a leading "per", but not required)
    let time_unit = normalize_unit(right.trim_start_matches("per").trim());
    let days = match time_unit.as_str() {
        "d" | "day" | "days" | "jour" | "jours" => 1.0,
        "h" | "hour" | "hours" | "heure" | "heures" => 1.0 / 24.0,
        "min" | "minute" | "minutes" => 1.0 / (24.0 * 60.0),
        "s" | "sec" | "second" | "seconds" | "seconde" | "secondes" => 1.0 / (24.0 * 3600.0),
        _ => {
            return Err(format!(
                "Unsupported time unit '{right}' in angular speed denominator. Expected day/hour/min/sec (and French aliases)."
            ));
        }
    };

    Ok(angle_rad / days)
}

/* -------------------------------------------------------------------------- */
/*  Small helpers                                                              */
/* -------------------------------------------------------------------------- */

/// Split a quantity string into `(value, unit)`.
///
/// Accepted forms
/// --------------
/// - `"<value> <unit>"`
/// - `"<value><unit>"` (best-effort)
///
/// Examples:
/// - `"35 arcmin"` → `(35.0, "arcmin")`
/// - `"1e-3deg"` → `(1e-3, "deg")`
///
/// Errors
/// ------
/// Returns an error if the input is empty, cannot be parsed, or contains too
/// many whitespace tokens.
fn split_value_unit(input: &str) -> Result<(f64, String), String> {
    let s = input.trim();
    // Accept either "35 arcmin" or "35arcmin" (best effort).
    // Strategy:
    // 1) try whitespace split
    // 2) otherwise parse a leading float and take the rest as unit.
    let mut it = s.split_whitespace();
    let first = it
        .next()
        .ok_or_else(|| "Empty quantity string".to_string())?;
    let second = it.next();

    if let Some(unit) = second {
        if it.next().is_some() {
            return Err(format!("Invalid quantity '{input}': too many tokens"));
        }
        let v: f64 = first
            .parse()
            .map_err(|_| format!("Invalid number in quantity '{input}'"))?;
        return Ok((v, unit.to_string()));
    }

    // No whitespace: try to split number prefix
    let (num, unit) = split_leading_number(first)
        .ok_or_else(|| format!("Invalid quantity '{input}': expected '<value> <unit>'"))?;
    Ok((num, unit))
}

/// Split a string into a leading numeric prefix and a trailing unit suffix.
///
/// This function scans for the **longest prefix** that parses as `f64`.
/// It supports scientific notation.
///
/// Examples:
/// - `"0.05rad"` → `(0.05, "rad")`
/// - `"35arcmin"` → `(35.0, "arcmin")`
/// - `"1e-3deg"` → `(0.001, "deg")`
fn split_leading_number(s: &str) -> Option<(f64, String)> {
    // Find the longest prefix that parses as f64.
    // Works for: "0.05rad", "35arcmin", "1e-3deg".
    for i in (1..=s.len()).rev() {
        let (a, b) = s.split_at(i);
        if let Ok(v) = a.parse::<f64>() {
            let unit = b.trim();
            if !unit.is_empty() {
                return Some((v, unit.to_string()));
            }
        }
    }
    None
}

/// Normalize a unit token for matching.
///
/// Behavior
/// --------
/// - Lowercases.
/// - Trims whitespace.
/// - Replaces typographic apostrophe `’` by `'`.
/// - Applies a minimal French alias mapping for `"minute d'arc"` / `"seconde d'arc"`
///   by detecting `"minute"+"arc"` or `"seconde"+"arc"` and returning internal
///   tokens `"minute-d-arc"` / `"seconde-d-arc"`.
///
/// Notes
/// -----
/// The French handling is intentionally conservative and not meant to be a
/// comprehensive localization layer.
fn normalize_unit(u: &str) -> String {
    let s = u.trim().to_lowercase();

    // normalize basic punctuation / spaces
    let s = s.replace('’', "'");

    // very small French alias normalization
    if s.contains("minute") && s.contains("arc") {
        return "minute-d-arc".to_string();
    }
    if s.contains("seconde") && s.contains("arc") {
        return "seconde-d-arc".to_string();
    }

    s
}

#[cfg(test)]
mod config_units_tests {
    use super::*;

    use approx::{assert_relative_eq, assert_ulps_eq};
    use proptest::prelude::*;

    /* ---------------------------------------------------------------------- */
    /*  Deterministic unit tests                                                */
    /* ---------------------------------------------------------------------- */

    #[test]
    fn angle_parsing_smoke() {
        let r = parse_angle_rad("60 arcmin").unwrap();
        assert_relative_eq!(r, 1f64.to_radians(), epsilon = 1e-14);

        let r = parse_angle_rad("3600arcsec").unwrap();
        assert_relative_eq!(r, 1f64.to_radians(), epsilon = 1e-14);

        let r = parse_angle_rad("1 deg").unwrap();
        assert_relative_eq!(r, 1f64.to_radians(), epsilon = 1e-14);

        let r = parse_angle_rad("1rad").unwrap();
        assert_relative_eq!(r, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn time_parsing_smoke() {
        let d = parse_time_days("24 hour").unwrap();
        assert_relative_eq!(d, 1.0, epsilon = 1e-14);

        let d = parse_time_days("1440 min").unwrap();
        assert_relative_eq!(d, 1.0, epsilon = 1e-14);

        let d = parse_time_days("86400 sec").unwrap();
        assert_relative_eq!(d, 1.0, epsilon = 1e-14);

        let d = parse_time_days("1 day").unwrap();
        assert_relative_eq!(d, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn rate_parsing_smoke() {
        let w = parse_ang_speed_rad_per_day("60 arcmin/day").unwrap();
        let expected = 1f64.to_radians(); // per day
        assert_relative_eq!(w, expected, epsilon = 1e-14);

        let w = parse_ang_speed_rad_per_day("2 arcsec/hour").unwrap();
        let expected = ((2.0_f64 / 3600.0_f64).to_radians()) / (1.0 / 24.0);
        assert_relative_eq!(w, expected, epsilon = 1e-14);
    }

    #[test]
    fn invalid_inputs() {
        assert!(parse_angle_rad("").is_err());
        assert!(parse_time_days("10").is_err()); // missing unit
        assert!(parse_angle_rad("10 parsec").is_err());
        assert!(parse_time_days("10 fortnight").is_err());
        assert!(parse_ang_speed_rad_per_day("10 arcmin").is_err()); // missing denominator
        assert!(parse_ang_speed_rad_per_day("10 arcmin/day/hour").is_err()); // too many '/'
    }

    /* ---------------------------------------------------------------------- */
    /*  Serde deserialization tests (YAML)                                      */
    /* ---------------------------------------------------------------------- */

    #[derive(Debug, Deserialize)]
    struct WrapTime {
        #[serde(deserialize_with = "de_time_days")]
        dt: f64,
    }

    #[derive(Debug, Deserialize)]
    struct WrapAngle {
        #[serde(deserialize_with = "de_angle_rad")]
        a: f64,
    }

    #[derive(Debug, Deserialize)]
    struct WrapRate {
        #[serde(deserialize_with = "de_ang_speed_rad_per_day")]
        w: f64,
    }

    #[test]
    fn serde_accepts_number_or_string() {
        // Numeric stays unchanged (already in canonical internal units).
        let t: WrapTime = serde_yaml::from_str("dt: 0.5").unwrap();
        assert_relative_eq!(t.dt, 0.5, epsilon = 0.0);

        // String gets parsed.
        let t: WrapTime = serde_yaml::from_str("dt: \"12 hour\"").unwrap();
        assert_relative_eq!(t.dt, 0.5, epsilon = 1e-14);

        let a: WrapAngle = serde_yaml::from_str("a: \"180 deg\"").unwrap();
        assert_relative_eq!(a.a, std::f64::consts::PI, epsilon = 1e-14);

        let w: WrapRate = serde_yaml::from_str("w: \"60 arcmin/day\"").unwrap();
        assert_relative_eq!(w.w, 1f64.to_radians(), epsilon = 1e-14);
    }

    /* ---------------------------------------------------------------------- */
    /*  Property-based tests (proptest)                                         */
    /* ---------------------------------------------------------------------- */

    // Strategy helpers: we keep values away from 0 when used as denominators,
    // and keep ranges moderate to avoid edge cases (NaN/Inf) and silly epsilon issues.
    fn angle_value() -> impl Strategy<Value = f64> {
        // up to 1e6 in the chosen unit is plenty; we avoid tiny subnormals.
        1e-9_f64..1e6_f64
    }

    fn time_value() -> impl Strategy<Value = f64> {
        1e-9_f64..1e6_f64
    }

    fn angle_unit() -> impl Strategy<Value = &'static str> {
        prop_oneof![Just("rad"), Just("deg"), Just("arcmin"), Just("arcsec"),]
    }

    fn time_unit() -> impl Strategy<Value = &'static str> {
        prop_oneof![Just("day"), Just("hour"), Just("min"), Just("sec"),]
    }

    fn angle_scale_to_rad(unit: &str) -> f64 {
        match unit {
            "rad" => 1.0,
            "deg" => std::f64::consts::PI / 180.0,
            "arcmin" => std::f64::consts::PI / (180.0 * 60.0),
            "arcsec" => std::f64::consts::PI / (180.0 * 3600.0),
            _ => unreachable!("unit not in strategy"),
        }
    }

    fn time_scale_to_days(unit: &str) -> f64 {
        match unit {
            "day" => 1.0,
            "hour" => 1.0 / 24.0,
            "min" => 1.0 / (24.0 * 60.0),
            "sec" => 1.0 / (24.0 * 3600.0),
            _ => unreachable!("unit not in strategy"),
        }
    }

    proptest! {
        /// Parsing an angle string must match the analytical conversion into radians.
        #[test]
        fn prop_angle_matches_analytic(v in angle_value(), u in angle_unit()) {
            let s1 = format!("{v} {u}");
            let parsed = parse_angle_rad(&s1).unwrap();
            let expected = v * angle_scale_to_rad(u);

            // Use approx: relative error is relevant as v spans many orders of magnitude.
            assert_relative_eq!(parsed, expected, max_relative = 1e-12, epsilon = 1e-15);
        }

        /// Parsing a time string must match the analytical conversion into days.
        #[test]
        fn prop_time_matches_analytic(v in time_value(), u in time_unit()) {
            let s1 = format!("{v} {u}");
            let parsed = parse_time_days(&s1).unwrap();
            let expected = v * time_scale_to_days(u);

            assert_relative_eq!(parsed, expected, max_relative = 1e-12, epsilon = 1e-15);
        }

        /// For supported angle units, the "glued" format "<value><unit>" must match
        /// the whitespace format "<value> <unit>" (this exercises split_leading_number()).
        #[test]
        fn prop_angle_glued_equals_spaced(v in angle_value(), u in angle_unit()) {
            let glued = format!("{v}{u}");
            let spaced = format!("{v} {u}");

            let a = parse_angle_rad(&glued).unwrap();
            let b = parse_angle_rad(&spaced).unwrap();

            // These should be extremely close; ULP check is OK here.
            assert_ulps_eq!(a, b, max_ulps = 8);
        }

        /// For angular speed, parsing "<angle>/<time>" must be consistent with parsing
        /// the components and dividing in canonical units.
        #[test]
        fn prop_rate_consistent_with_components(
            a in angle_value(),
            au in angle_unit(),
            tu in time_unit(),
        ) {
            let rate_str = format!("{a} {au}/{tu}");
            let parsed = parse_ang_speed_rad_per_day(&rate_str).unwrap();

            let angle_rad = a * angle_scale_to_rad(au);
            let time_days = 1.0 * time_scale_to_days(tu); // denominator is "per <unit>"
            let expected = angle_rad / time_days;

            assert_relative_eq!(parsed, expected, max_relative = 1e-12, epsilon = 1e-15);

            // Also check a variant with extra spaces.
            let rate_str2 = format!("{a} {au} / {tu}");
            let parsed2 = parse_ang_speed_rad_per_day(&rate_str2).unwrap();
            assert_ulps_eq!(parsed, parsed2, max_ulps = 8);

            // And a variant with "per".
            let rate_str3 = format!("{a} {au}/per {tu}");
            let parsed3 = parse_ang_speed_rad_per_day(&rate_str3).unwrap();
            assert_ulps_eq!(parsed, parsed3, max_ulps = 8);
        }

        /// Serde YAML: the custom deserializers must accept string quantities.
        #[test]
        fn prop_serde_yaml_string_angle(v in angle_value(), u in angle_unit()) {
            let yaml = format!("a: \"{v} {u}\"");
            let w: WrapAngle = serde_yaml::from_str(&yaml).unwrap();

            let expected = v * angle_scale_to_rad(u);
            assert_relative_eq!(w.a, expected, max_relative = 1e-12, epsilon = 1e-15);
        }

        /// Serde YAML: the custom deserializers must accept raw numeric values.
        #[test]
        fn prop_serde_yaml_numeric_passthrough(v in 0.0_f64..1e6_f64) {
            let yaml = format!("dt: {v}");
            let w: WrapTime = serde_yaml::from_str(&yaml).unwrap();
            assert_ulps_eq!(w.dt, v, max_ulps = 0);
        }
    }
}
