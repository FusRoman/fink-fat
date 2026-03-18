pub mod alerts;
pub mod astro_math;
pub mod display_format;
pub mod engine_config;
pub mod error;
pub mod graph;
pub mod night_id;
pub mod persistence;
pub mod pipeline;
pub mod seeding;
pub mod solver;
pub mod spacetime_bucket;
pub mod trajectory;

/// Modified Julian Date in TT (Terrestrial Time), in days.
///
/// Convention
/// ----------
/// - Same zero-point as standard MJD,
/// - Time scale is TT (not UTC/TAI).
pub type MJDTT = f64;

/// Angle in radians, typically used for right ascension, declination,
/// and small-angle offsets on the sky.
pub type Radian = f64;

/// Angle in arcseconds.
pub type Arcsec = f64;

pub use crate::alerts::{Alert, AlertKey, store::AlertStore};
