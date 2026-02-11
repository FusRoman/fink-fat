//! Fundamental scalar units used across the engine.
//!
//! These are strong-typed aliases for clarity. They do **not** enforce unit
//! safety at compile time, but make the intent explicit in function signatures.

/// Modified Julian Date in TT (Terrestrial Time), in days.
///
/// Convention
/// ----------
/// - Same zero-point as standard MJD,
/// - Time scale is TT (not UTC/TAI).
pub type MJDTT = f64;

/// Angle in radians, typically used for right ascension, declination,
/// and small-angle offsets on the sky.
pub type Radians = f64;

/// Angle in arcseconds.
pub type Arcsec = f64;
