//! Analytic Sun altitude and morning twilight, without any ephemeris file.
//!
//! The night clock only needs *when the night ends* to within about a
//! minute, so a low-precision analytic Sun suffices: the NOAA solar
//! calculator's algorithm (Meeus, *Astronomical Algorithms*, ch. 25),
//! accurate to ~0.01° in position. Refraction is ignored (irrelevant below
//! the horizon, where twilight altitudes lie) and UT1 is taken equal to UTC
//! (|UT1 − UTC| < 0.9 s).
//!
//! Reference values in the tests come from `astropy` (`get_sun` → `AltAz`
//! without refraction).

use std::f64::consts::TAU;

use photom::NightId;
use serde::Deserialize;

use fink_fat_engine::engine_config::units::de_angle_rad;

/// MJD of the J2000.0 epoch (2000-01-01 12:00 TT, taken as UTC here).
const MJD_J2000: f64 = 51_544.5;

/// An observing site.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Site {
    /// Geodetic longitude, **east-positive**, radians (YAML accepts units,
    /// e.g. `"-70.749417 deg"`).
    #[serde(deserialize_with = "de_angle_rad")]
    pub lon: f64,
    /// Geodetic latitude, radians.
    #[serde(deserialize_with = "de_angle_rad")]
    pub lat: f64,
}

impl Site {
    /// Rubin Observatory, Cerro Pachón (Chile).
    pub fn cerro_pachon() -> Self {
        Self {
            lon: (-70.749_417_f64).to_radians(),
            lat: (-30.244_639_f64).to_radians(),
        }
    }
}

impl Default for Site {
    fn default() -> Self {
        Self::cerro_pachon()
    }
}

/// Apparent equatorial position of the Sun.
struct SunPosition {
    /// Right ascension, radians.
    ra: f64,
    /// Declination, radians.
    dec: f64,
}

/// Low-precision apparent Sun position at `mjd` (NOAA / Meeus ch. 25).
fn sun_position(mjd: f64) -> SunPosition {
    let t = (mjd - MJD_J2000) / 36_525.0;
    let mean_longitude = (280.466_46 + t * (36_000.769_83 + 0.000_303_2 * t)).rem_euclid(360.0);
    let mean_anomaly = (357.529_11 + t * (35_999.050_29 - 0.000_153_7 * t)).to_radians();
    let center = mean_anomaly.sin() * (1.914_602 - t * (0.004_817 + 0.000_014 * t))
        + (2.0 * mean_anomaly).sin() * (0.019_993 - 0.000_101 * t)
        + (3.0 * mean_anomaly).sin() * 0.000_289;
    let omega = (125.04 - 1_934.136 * t).to_radians();
    let apparent_longitude =
        (mean_longitude + center - 0.005_69 - 0.004_78 * omega.sin()).to_radians();
    let mean_obliquity =
        23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.000_59 - t * 0.001_813))) / 60.0) / 60.0;
    let obliquity = (mean_obliquity + 0.002_56 * omega.cos()).to_radians();

    SunPosition {
        ra: (obliquity.cos() * apparent_longitude.sin()).atan2(apparent_longitude.cos()),
        dec: (obliquity.sin() * apparent_longitude.sin()).asin(),
    }
}

/// Greenwich mean sidereal time at `mjd` (UT1 ≈ UTC), radians in `[0, 2π)`.
fn gmst(mjd: f64) -> f64 {
    let d = mjd - MJD_J2000;
    let t = d / 36_525.0;
    let degrees =
        280.460_618_37 + 360.985_647_366_29 * d + 0.000_387_933 * t * t - t * t * t / 38_710_000.0;
    degrees.to_radians().rem_euclid(TAU)
}

/// Geometric altitude of the Sun's centre seen from `site` at `mjd_utc`,
/// radians (no refraction).
pub fn sun_altitude(mjd_utc: f64, site: &Site) -> f64 {
    let sun = sun_position(mjd_utc);
    let hour_angle = gmst(mjd_utc) + site.lon - sun.ra;
    (site.lat.sin() * sun.dec.sin() + site.lat.cos() * sun.dec.cos() * hour_angle.cos()).asin()
}

/// When the Sun rises through `altitude` (radians, e.g. −12° for nautical
/// twilight) on the morning that ends `night`, MJD UTC.
///
/// `night` is a Rubin night id: the MJD of the evening's `dayObs` date. The
/// search spans local solar midnight to local noon of the next date, where
/// the Sun's altitude increases monotonically, and bisects to ~1 ms.
///
/// # Returns
/// `None` when the Sun does not cross `altitude` upwards in that window (it
/// never gets that low, or never that high — polar sites).
pub fn morning_twilight(night: NightId, altitude: f64, site: &Site) -> Option<f64> {
    let local_midnight = f64::from(night.0) + 1.0 - site.lon / TAU;
    let excess = |mjd: f64| sun_altitude(mjd, site) - altitude;

    let (mut low, mut high) = (local_midnight, local_midnight + 0.5);
    if excess(low) >= 0.0 || excess(high) <= 0.0 {
        return None;
    }
    // 0.5 day / 2^36 ≈ 0.6 µs.
    for _ in 0..36 {
        let mid = 0.5 * (low + high);
        if excess(mid) < 0.0 {
            low = mid;
        } else {
            high = mid;
        }
    }
    Some(0.5 * (low + high))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINUTE: f64 = 1.0 / 1_440.0;

    #[test]
    fn sun_altitude_matches_astropy() {
        // (MJD UTC, astropy altitude in degrees) at Cerro Pachón.
        let site = Site::cerro_pachon();
        for (mjd, expected_deg) in [
            (61_344.0, -12.119_043_536_651_84),
            (61_344.3, -30.393_415_571_161_935),
            (61_345.1, -36.493_012_358_192_715),
            (60_847.75, 33.372_533_287_030_684),
        ] {
            let altitude = sun_altitude(mjd, &site).to_degrees();
            assert!(
                (altitude - expected_deg).abs() < 0.05,
                "MJD {mjd}: {altitude}° vs astropy {expected_deg}°"
            );
        }
    }

    #[test]
    fn morning_twilight_matches_astropy_to_the_minute() {
        // (dayObs night id, astropy −12° crossing, MJD UTC).
        let site = Site::cerro_pachon();
        for (night, expected) in [
            (60_847, 60_848.446_439_107_77), // 2025-06-21, austral winter
            (61_030, 61_031.359_456_628_03), // 2025-12-21, austral summer
            (61_119, 61_120.413_352_130_54), // 2026-03-20, equinox
            (61_344, 61_345.369_788_870_57), // 2026-10-31
        ] {
            let twilight = morning_twilight(NightId(night), (-12.0_f64).to_radians(), &site)
                .expect("Cerro Pachón has a nautical twilight every night");
            assert!(
                (twilight - expected).abs() < MINUTE,
                "night {night}: {twilight} vs astropy {expected} ({:.1} s)",
                (twilight - expected) * 86_400.0
            );
        }
    }

    #[test]
    fn no_twilight_when_the_sun_never_gets_that_low() {
        // Polar summer: the Sun stays above −12° all night at 80° S.
        let site = Site {
            lon: 0.0,
            lat: (-80.0_f64).to_radians(),
        };
        assert_eq!(
            morning_twilight(NightId(61_030), (-12.0_f64).to_radians(), &site),
            None
        );
    }
}
