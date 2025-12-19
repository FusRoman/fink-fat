//! CLI-friendly angle parsing utilities.
//!
//! This module provides a small helper type [`Angle`] designed for use in
//! command-line interfaces (CLI). It allows users to specify angular values
//! with optional units in a human-friendly way, while ensuring a single,
//! unambiguous internal representation in **radians**.
//!
//! The primary goal is ergonomics and robustness for configuration-driven
//! tools (e.g., threshold sweeps), without pulling in heavy unit systems.
//!
//! Typical use cases include:
//! - angular separation thresholds,
//! - spatial gating parameters,
//! - plotting or reporting unit conversions.
//!
//! The [`Angle`] type implements [`FromStr`], making it directly compatible
//! with [`clap`] argument parsing.

use std::str::FromStr;

/// Angular value with optional unit support (stored internally in radians).
///
/// This type represents an angle parsed from a string or constructed directly
/// from a numeric value. Internally, the value is always stored in **radians**
/// to avoid ambiguity and repeated conversions downstream.
///
/// Accepted input forms
/// --------------------
/// The parser accepts both space-separated and compact notations:
///
/// * `"0.00029"` – interpreted as radians (default)
/// * `"0.00029 rad"`
/// * `"2 deg"` or `"2deg"`
/// * `"40 arcmin"` or `"40arcmin"`
/// * `"30 arcsec"` or `"30arcsec"`
///
/// Supported units
/// ---------------
/// * `rad`, `radian`, `radians`
/// * `deg`, `degree`, `degrees`
/// * `arcmin`, `amin`, `arcminute`, `arcminutes`
/// * `arcsec`, `asec`, `arcsecond`, `arcseconds`
///
/// Notes
/// -----
/// * If no unit is specified, the value is assumed to be in radians.
/// * Parsing errors are reported with explicit messages suitable for CLI
///   feedback.
/// * This type intentionally does **not** perform semantic validation
///   (e.g., positivity). Such checks should be applied at the configuration
///   level depending on the intended physical meaning.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Angle {
    /// Angle value stored in radians.
    rad: f64,
}

impl Angle {
    /// Return the angle value in radians.
    ///
    /// Arguments
    /// ---------
    /// * `self` – The angle value.
    ///
    /// Return
    /// ------
    /// * `f64` – The angle expressed in radians.
    ///
    /// Notes
    /// -----
    /// This accessor performs no conversion or validation; it simply returns
    /// the internally stored value.
    #[inline]
    pub fn as_radians(self) -> f64 {
        self.rad
    }
}

impl From<f64> for Angle {
    /// Construct an [`Angle`] directly from a value in radians.
    ///
    /// Arguments
    /// ---------
    /// * `rad` – Angle value in radians.
    ///
    /// Return
    /// ------
    /// * `Angle` – A new angle storing the given value in radians.
    #[inline]
    fn from(rad: f64) -> Self {
        Self { rad }
    }
}

impl FromStr for Angle {
    type Err = anyhow::Error;

    /// Parse an [`Angle`] from a string representation.
    ///
    /// The parser accepts either:
    /// - a bare number (assumed to be in radians), or
    /// - a number followed by an angular unit, with or without whitespace.
    ///
    /// Arguments
    /// ---------
    /// * `input` – Input string to parse (e.g., `"2 deg"`, `"40arcsec"`).
    ///
    /// Return
    /// ------
    /// * `Ok(Angle)` on successful parsing.
    /// * `Err(anyhow::Error)` if the string is empty, malformed, or uses an
    ///   unsupported unit.
    ///
    /// Notes
    /// -----
    /// This implementation is optimized for CLI usage:
    /// - error messages are concise and user-oriented,
    /// - no allocation-heavy parsing or regexes are used.
    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let s = input.trim();
        anyhow::ensure!(!s.is_empty(), "empty angle");

        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts.len() == 1 {
            return parse_compact(parts[0]);
        }
        if parts.len() == 2 {
            let value: f64 = parts[0]
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid angle value: {:?}", parts[0]))?;
            let unit = parts[1];
            return Ok(Angle {
                rad: value * unit_to_rad_scale(unit)?,
            });
        }

        anyhow::bail!("invalid angle format: {input:?}");
    }
}

/// Parse a compact angle token of the form `<number><unit>`.
///
/// Arguments
/// ---------
/// * `token` – Compact angle token (e.g., `"2deg"`, `"40arcsec"`).
///
/// Return
/// ------
/// * `Ok(Angle)` if the token can be split into a numeric prefix and a unit.
/// * `Err(anyhow::Error)` if the token is malformed or uses an unknown unit.
///
/// Notes
/// -----
/// This helper is intentionally kept private and minimal. It avoids regexes
/// for performance and to keep binary size small.
fn parse_compact(token: &str) -> anyhow::Result<Angle> {
    if let Ok(v) = token.parse::<f64>() {
        return Ok(Angle { rad: v });
    }

    let mut split = token.len();
    for (i, ch) in token.char_indices() {
        if !(ch.is_ascii_digit() || ch == '.' || ch == '-' || ch == '+' || ch == 'e' || ch == 'E') {
            split = i;
            break;
        }
    }

    anyhow::ensure!(
        split > 0 && split < token.len(),
        "invalid angle token: {token:?}"
    );

    let (num, unit) = token.split_at(split);
    let value: f64 = num
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid angle value: {:?}", num))?;

    Ok(Angle {
        rad: value * unit_to_rad_scale(unit)?,
    })
}

/// Convert an angular unit string into a scale factor to radians.
///
/// Arguments
/// ---------
/// * `unit` – Unit string (case-insensitive).
///
/// Return
/// ------
/// * `Ok(f64)` – Multiplicative factor converting the unit to radians.
/// * `Err(anyhow::Error)` if the unit is unknown.
///
/// Notes
/// -----
/// Supported units are intentionally limited to a small, explicit set to
/// reduce ambiguity and improve error messages in CLI tools.
fn unit_to_rad_scale(unit: &str) -> anyhow::Result<f64> {
    let u = unit.trim().to_ascii_lowercase();
    let scale = match u.as_str() {
        // radians
        "rad" | "radian" | "radians" => 1.0,
        // degrees
        "deg" | "degree" | "degrees" => std::f64::consts::PI / 180.0,
        // arcminutes
        "arcmin" | "amin" | "arcminute" | "arcminutes" => std::f64::consts::PI / (180.0 * 60.0),
        // arcseconds
        "arcsec" | "asec" | "arcsecond" | "arcseconds" => std::f64::consts::PI / (180.0 * 3600.0),
        _ => anyhow::bail!("unknown angle unit: {unit:?} (use rad|deg|arcmin|arcsec)"),
    };
    Ok(scale)
}
