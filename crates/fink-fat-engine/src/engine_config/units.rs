//! Human-friendly unit parsing for YAML configuration.
//!
//! Overview
//! --------
//! This module enables config fields to be specified either as:
//! - a raw number (assumed to already be in engine internal units), OR
//! - a string with explicit units (e.g. "35 arcmin/day", "2 arcsec/hour").
//!
//! Internal canonical units
//! ------------------------
//! - Angles: radians
//! - Time: days
//! - Angular speed: radians/day
//!
//! Accepted angle units
//! --------------------
//! - "rad", "radian", "radians"
//! - "deg", "degree", "degrees"
//! - "arcmin", "arcminute", "arcminutes"
//! - "arcsec", "arcsecond", "arcseconds"
//! - French aliases: "degre", "degres", "minute d'arc", "seconde d'arc" (basic)
//!
//! Accepted time units
//! -------------------
//! - "day", "days", "d", "jour", "jours"
//! - "hour", "hours", "h", "heure", "heures"
//! - "min", "minute", "minutes"
//! - "sec", "second", "seconds", "s", "seconde", "secondes"
//!
//! Accepted rate syntax
//! --------------------
//! "<angle>/<time>" with optional spaces, e.g.:
//! - "35 arcmin/day"
//! - "2 arcsec / hour"
//! - "0.05 rad/day"

use serde::Deserialize;
use serde::de;

/// Serde helper: accept either a number or a string.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum NumOrStr {
    Num(f64),
    Str(String),
}

/* -------------------------------------------------------------------------- */
/*  Public serde entry points (deserialize_with = ...)                         */
/* -------------------------------------------------------------------------- */

/// Deserialize a time quantity into **days** (f64).
pub fn de_time_days<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    match NumOrStr::deserialize(deserializer)? {
        NumOrStr::Num(x) => Ok(x),
        NumOrStr::Str(s) => parse_time_days(&s).map_err(de::Error::custom),
    }
}

/// Deserialize an angle quantity into **radians** (f64).
pub fn de_angle_rad<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    match NumOrStr::deserialize(deserializer)? {
        NumOrStr::Num(x) => Ok(x),
        NumOrStr::Str(s) => parse_angle_rad(&s).map_err(de::Error::custom),
    }
}

/// Deserialize an angular speed into **radians/day** (f64).
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
