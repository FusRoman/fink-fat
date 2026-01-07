//! CLI-friendly angular speed parsing utilities (Angle / time).
//!
//! This module provides [`AngularSpeed`], designed for CLI tools where users
//! want to specify an angular *rate* with human-friendly units.
//!
//! The key design choice is to **reuse [`crate::angle::Angle`]** for the angular
//! part (rad/deg/arcmin/arcsec, with compact or spaced forms), and only parse
//! a small set of time units for the denominator.
//!
//! Internally, the value is stored in **radians per day** (`rad/day`), matching
//! the engine convention (`PairConfig.max_angular_speed`).
//!
//! Examples (all valid)
//! --------------------
//! - "0.02"                  -> 0.02 rad/day (default)
//! - "0.02 rad/day"
//! - "10 arcsec/hour"
//! - "10arcsec/hour"
//! - "2 deg/hr"
//! - "30 arcmin/min"
//! - "1 rad"                 -> 1 rad/day (implicit /day for convenience)
//!
//! Notes
//! -----
//! - If no unit is specified, the value is assumed to be **rad/day**.
//! - If only an angle unit is specified (e.g. "0.1 deg"), `/day` is assumed.
//! - No semantic validation (positivity) is performed here.

use std::str::FromStr;

use crate::angle::Angle;

/// Angular speed stored internally in radians per day.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct AngularSpeed {
    rad_per_day: f64,
}

impl AngularSpeed {
    /// Return the angular speed in radians per day.
    #[inline]
    pub fn as_rad_per_day(self) -> f64 {
        self.rad_per_day
    }
}

impl From<f64> for AngularSpeed {
    /// Construct directly from a value in radians per day.
    #[inline]
    fn from(rad_per_day: f64) -> Self {
        Self { rad_per_day }
    }
}

impl FromStr for AngularSpeed {
    type Err = anyhow::Error;

    /// Parse an [`AngularSpeed`] from a string.
    ///
    /// Accepted forms
    /// --------------
    /// - Bare number: "0.02" -> rad/day
    /// - "ANGLE/TIME": "10 arcsec/hour", "2deg/hr", "0.01 rad/day"
    /// - "ANGLE" only: "0.1 deg" -> assumes "/day"
    ///
    /// Where ANGLE is parsed by [`Angle`] (supports compact or spaced),
    /// and TIME supports day|hour|min|sec (with aliases).
    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let s = input.trim();
        anyhow::ensure!(!s.is_empty(), "empty angular speed");

        // Fast path: bare number => rad/day.
        if let Ok(v) = s.parse::<f64>() {
            return Ok(AngularSpeed { rad_per_day: v });
        }

        // Allow either:
        // - "ANGLE / TIME" (with or without spaces)
        // - "ANGLE" alone (=> /day)
        //
        // We normalize by removing outer spaces, but we must preserve inner
        // angle spaces because Angle parser accepts "10 arcsec".
        if let Some((lhs, rhs)) = split_once_slash(s) {
            let angle = Angle::from_str(lhs.trim())?;
            let time_unit = TimeUnit::from_str(rhs.trim())?;
            return Ok(AngularSpeed {
                rad_per_day: angle.as_radians() / time_unit.as_days(),
            });
        }

        // No slash: treat as angle-only => /day
        let angle = Angle::from_str(s)?;
        Ok(AngularSpeed {
            rad_per_day: angle.as_radians(), // per day
        })
    }
}

/* ----------------------------- internal helpers ---------------------------- */

/// Split on the first '/', but tolerate optional spaces around it.
/// Returns None if no slash present.
fn split_once_slash(s: &str) -> Option<(&str, &str)> {
    // We want the first slash character.
    let idx = s.find('/')?;
    Some((&s[..idx], &s[idx + 1..]))
}

/// Supported time units for rates.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum TimeUnit {
    Day,
    Hour,
    Minute,
    Second,
}

impl TimeUnit {
    /// Convert 1 unit into days.
    #[inline]
    fn as_days(self) -> f64 {
        match self {
            TimeUnit::Day => 1.0,
            TimeUnit::Hour => 1.0 / 24.0,
            TimeUnit::Minute => 1.0 / (24.0 * 60.0),
            TimeUnit::Second => 1.0 / (24.0 * 3600.0),
        }
    }
}

impl FromStr for TimeUnit {
    type Err = anyhow::Error;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let u = input.trim().to_ascii_lowercase();
        let u = u.as_str();

        // Accept common aliases used in CLIs.
        let out = match u {
            "day" | "days" | "d" => TimeUnit::Day,
            "hour" | "hours" | "hr" | "hrs" | "h" => TimeUnit::Hour,
            "min" | "mins" | "minute" | "minutes" | "m" => TimeUnit::Minute,
            "sec" | "secs" | "second" | "seconds" | "s" => TimeUnit::Second,
            _ => anyhow::bail!("unknown time unit: {input:?} (use day|hour|min|sec)"),
        };
        Ok(out)
    }
}
