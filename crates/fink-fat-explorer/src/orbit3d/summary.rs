//! The focus object's orbit summary: derived quantities, landmarks and
//! planet MOIDs, built from data already at hand.
//!
//! Pure — no I/O, no ephemeris access — so it compiles on every target and
//! is unit-tested with hand-made orbits and planets.

use crate::orbit3d::geometry::{self, Keplerian};
use crate::orbit3d::types::{Body3D, BodyKind, OrbitSummary3D, PlanetMoid3D};

/// Days per Julian year, to express the period in years.
const DAYS_PER_YEAR: f64 = 365.25;

/// Builds the [`OrbitSummary3D`] of an orbit.
///
/// # Arguments
///
/// * `elems` — the object's orbit (a closed ellipse).
/// * `position_now` — the object's heliocentric position at the view's
///   epoch, AU.
/// * `planets` — the planets/perturbers at that same epoch; only
///   [`BodyKind::Planet`] entries get a MOID, and the one named `"Earth"`
///   gives the Earth distance.
///
/// # Returns
///
/// The summary; `moids` follows the order of `planets`.
pub fn build_orbit_summary(
    elems: &Keplerian,
    position_now: [f64; 3],
    planets: &[Body3D],
) -> OrbitSummary3D {
    OrbitSummary3D {
        semi_major_axis_au: elems.semi_major_axis_au,
        eccentricity: elems.eccentricity,
        inclination_deg: elems.inclination_deg,
        ascending_node_longitude_deg: elems.ascending_node_longitude_deg,
        perihelion_au: geometry::perihelion_distance_au(elems),
        aphelion_au: geometry::aphelion_distance_au(elems),
        period_years: geometry::period_days(elems) / DAYS_PER_YEAR,
        sun_distance_au: geometry::distance(position_now, [0.0; 3]),
        earth_distance_au: planets
            .iter()
            .find(|b| b.name == "Earth")
            .map(|earth| geometry::distance(position_now, earth.position)),
        landmarks: geometry::orbit_landmarks(elems),
        moids: planets
            .iter()
            .filter(|b| b.kind == BodyKind::Planet)
            .map(|b| PlanetMoid3D {
                name: b.name.clone(),
                moid: geometry::moid(elems, &b.elements),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn circle(radius: f64, inclination_deg: f64) -> Keplerian {
        Keplerian {
            epoch_mjd_tt: 60_000.0,
            semi_major_axis_au: radius,
            eccentricity: 0.0,
            inclination_deg,
            ascending_node_longitude_deg: 0.0,
            periapsis_argument_deg: 0.0,
            mean_anomaly_deg: 0.0,
        }
    }

    fn body(name: &str, kind: BodyKind, elements: Keplerian, position: [f64; 3]) -> Body3D {
        Body3D {
            name: name.to_string(),
            kind,
            position,
            orbit: Vec::new(),
            elements,
        }
    }

    #[test]
    fn summary_reports_orbit_numbers_distances_and_planet_moids() {
        let object = Keplerian {
            semi_major_axis_au: 2.5,
            eccentricity: 0.2,
            ..circle(2.5, 10.0)
        };
        let planets = [
            body("Earth", BodyKind::Planet, circle(1.0, 0.0), [1.0, 0.0, 0.0]),
            body(
                "Ceres",
                BodyKind::Perturber,
                circle(2.77, 10.6),
                [2.0, 0.0, 0.0],
            ),
        ];

        let summary = build_orbit_summary(&object, [3.0, 4.0, 0.0], &planets);

        assert!((summary.perihelion_au - 2.0).abs() < 1e-12);
        assert!((summary.aphelion_au - 3.0).abs() < 1e-12);
        assert!((summary.sun_distance_au - 5.0).abs() < 1e-12);
        let earth = summary.earth_distance_au.expect("Earth is in the list");
        assert!((earth - (4.0_f64 + 16.0).sqrt()).abs() < 1e-12);
        assert!((summary.period_years - 2.5_f64.powf(1.5)).abs() < 0.01);
    }

    /// Only the eight planets get a MOID, not the perturbers.
    #[test]
    fn summary_only_computes_moids_for_planets() {
        let planets = [
            body("Earth", BodyKind::Planet, circle(1.0, 0.0), [1.0, 0.0, 0.0]),
            body(
                "Ceres",
                BodyKind::Perturber,
                circle(2.77, 10.6),
                [2.0, 0.0, 0.0],
            ),
            body(
                "Mars",
                BodyKind::Planet,
                circle(1.52, 1.85),
                [1.5, 0.0, 0.0],
            ),
        ];

        let summary = build_orbit_summary(&circle(3.0, 0.0), [3.0, 0.0, 0.0], &planets);

        let names: Vec<&str> = summary.moids.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["Earth", "Mars"]);
        // Coplanar circles: MOID is the radius difference.
        assert!((summary.moids[0].moid.distance_au - 2.0).abs() < 1e-6);
    }

    #[test]
    fn summary_without_an_earth_has_no_earth_distance() {
        let planets = [body(
            "Mars",
            BodyKind::Planet,
            circle(1.52, 1.85),
            [1.5, 0.0, 0.0],
        )];
        let summary = build_orbit_summary(&circle(3.0, 0.0), [3.0, 0.0, 0.0], &planets);
        assert!(summary.earth_distance_au.is_none());
    }
}
