//! Pure Keplerian-orbit geometry: turning a set of osculating elements into
//! 3D heliocentric points for the orbit-visualization feature.
//!
//! Every function here is a plain, side-effect-free computation — no
//! database, no ephemeris file, no network. That is deliberate: it is what
//! lets [`ellipse_points`] and [`position_at_epoch`] be unit-tested with
//! nothing but known orbital elements, and what lets the same code serve both
//! tracked objects (elements from an `orbit_fits` row) and planets/
//! perturbers (elements derived from a JPL ephemeris query, see
//! [`super::ephem_provider`]) through one shared implementation.
//!
//! The orbit model is unperturbed two-body Keplerian motion: a fixed ellipse,
//! swept at the constant mean motion `n = sqrt(GM_sun / a^3)`. That is an
//! approximation for real bodies (their osculating elements drift under
//! perturbations), but it is the right one here — every caller already only
//! has a single osculating element set (either a fit's, or an ephemeris
//! sample's), not a full perturbed trajectory, so two-body propagation is the
//! only motion model consistent with the input.

/// One set of heliocentric osculating Keplerian elements, in the units the
/// rest of the crate already displays them in (see
/// [`crate::fit_pipeline::fit::KeplerianView`]) — angles in degrees, distance
/// in AU — so a caller can build one directly from a `KeplerianView` plus its
/// companion `reference_epoch` without a unit conversion at the call site.
/// Every angle is converted to radians internally, once, by the functions
/// below.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Keplerian {
    /// Epoch the elements are osculating at. Modified Julian Date,
    /// Terrestrial Time (MJD-TT) — the same convention `orbit_fits
    /// .reference_epoch` and every other stored epoch in this crate use (see
    /// `crate::format_epoch::iso_utc`).
    pub epoch_mjd_tt: f64,
    /// Semi-major axis, AU. Must be strictly positive and finite for the
    /// orbit to be a closed ellipse — every function here assumes so and is
    /// only meaningful for `0.0 <= eccentricity < 1.0`.
    pub semi_major_axis_au: f64,
    /// Orbital eccentricity, unitless. Must satisfy `0.0 <= eccentricity <
    /// 1.0` for a closed ellipse; parabolic/hyperbolic orbits are out of
    /// scope for this module.
    pub eccentricity: f64,
    /// Inclination to the ecliptic, degrees.
    pub inclination_deg: f64,
    /// Longitude of the ascending node, degrees.
    pub ascending_node_longitude_deg: f64,
    /// Argument of periapsis, degrees.
    pub periapsis_argument_deg: f64,
    /// Mean anomaly at `epoch_mjd_tt`, degrees.
    pub mean_anomaly_deg: f64,
}

impl Keplerian {
    fn inclination_rad(&self) -> f64 {
        self.inclination_deg.to_radians()
    }

    fn ascending_node_longitude_rad(&self) -> f64 {
        self.ascending_node_longitude_deg.to_radians()
    }

    fn periapsis_argument_rad(&self) -> f64 {
        self.periapsis_argument_deg.to_radians()
    }

    fn mean_anomaly_rad(&self) -> f64 {
        self.mean_anomaly_deg.to_radians()
    }
}

/// The Sun's gravitational parameter, in AU³/day² — the unit system two-body
/// mean motion needs when distances are in AU and time in days.
///
/// Deliberately *not* imported from the `outfit` crate: that crate is a
/// server-only dependency (it does not build for `wasm32-unknown-unknown`,
/// see `Cargo.toml`), while this module has no such restriction and is
/// exercised by plain `cargo test`. The derivation below is copied verbatim
/// from `outfit::propagator::planet_gm::GM_SUN` (DE440 mass parameter) rather
/// than inlining a single rounded constant, so the two stay trivially
/// comparable and re-derivable from the same source value.
///
/// Source: Park, R.S. et al. (2021), *The JPL Planetary and Lunar Ephemerides
/// DE440 and DE441*, AJ 161, 105.
mod gm_sun {
    /// Sun GM, km³/s² (DE440).
    const GM_SUN_KM3_S2: f64 = 1.327_124_400_41e11;
    /// AU in kilometres (IAU 2012).
    const AU_KM: f64 = 1.495_978_707e8;
    /// km³/s² → AU³/day²: `(86400 s/day)² / (AU_KM km/AU)³`.
    const KM3_S2_TO_AU3_DAY2: f64 = (86400.0 * 86400.0) / (AU_KM * AU_KM * AU_KM);

    pub(super) const GM_SUN_AU3_PER_DAY2: f64 = GM_SUN_KM3_S2 * KM3_S2_TO_AU3_DAY2;
}

/// Two-body mean motion for a heliocentric orbit.
///
/// # Arguments
///
/// * `semi_major_axis_au` — semi-major axis, AU. Must be strictly positive.
///
/// # Returns
///
/// The mean angular rate `n = sqrt(GM_sun / a^3)`, in radians/day.
fn mean_motion_rad_per_day(semi_major_axis_au: f64) -> f64 {
    (gm_sun::GM_SUN_AU3_PER_DAY2 / semi_major_axis_au.powi(3)).sqrt()
}

/// Wraps an angle into `[0, 2*PI)`.
fn wrap_2pi(angle_rad: f64) -> f64 {
    let two_pi = std::f64::consts::TAU;
    angle_rad.rem_euclid(two_pi)
}

/// Solves Kepler's equation `M = E - e*sin(E)` for the eccentric anomaly `E`,
/// by Newton-Raphson.
///
/// # Arguments
///
/// * `mean_anomaly_rad` — mean anomaly, radians. Any finite value; wrapped
///   internally, so it need not already lie in `[0, 2*PI)`.
/// * `eccentricity` — orbital eccentricity. Must satisfy `0.0 <= eccentricity
///   < 1.0`.
///
/// # Returns
///
/// The eccentric anomaly `E`, radians, in `[0, 2*PI)`, accurate to within
/// about `1e-12` radians for every eccentricity below `0.99` (the solver caps
/// at 50 iterations, which converges well inside that bound for any
/// eccentricity in the closed-ellipse range this module supports).
pub fn solve_eccentric_anomaly(mean_anomaly_rad: f64, eccentricity: f64) -> f64 {
    let m = wrap_2pi(mean_anomaly_rad);

    // Standard starting guess: exact for e = 0, and close enough for the
    // eccentricities a solar-system orbit visualization ever needs (up to
    // ~0.99) that Newton's method converges in a handful of iterations.
    let mut e_anom = m + eccentricity * m.sin();

    const MAX_ITERATIONS: u32 = 50;
    const TOLERANCE_RAD: f64 = 1e-12;

    for _ in 0..MAX_ITERATIONS {
        let f = e_anom - eccentricity * e_anom.sin() - m;
        let f_prime = 1.0 - eccentricity * e_anom.cos();
        let delta = f / f_prime;
        e_anom -= delta;
        if delta.abs() < TOLERANCE_RAD {
            break;
        }
    }

    wrap_2pi(e_anom)
}

/// Converts an eccentric anomaly to the corresponding true anomaly.
///
/// # Arguments
///
/// * `eccentricity` — orbital eccentricity, `0.0 <= eccentricity < 1.0`.
/// * `eccentric_anomaly_rad` — eccentric anomaly, radians.
///
/// # Returns
///
/// The true anomaly, radians, in `(-PI, PI]`.
fn true_anomaly_from_eccentric(eccentricity: f64, eccentric_anomaly_rad: f64) -> f64 {
    let half = eccentric_anomaly_rad / 2.0;
    let (sin_half, cos_half) = half.sin_cos();
    2.0 * ((1.0 + eccentricity).sqrt() * sin_half).atan2((1.0 - eccentricity).sqrt() * cos_half)
}

/// Heliocentric distance at a given true anomaly, from the polar equation of
/// the conic.
///
/// # Arguments
///
/// * `semi_major_axis_au` — semi-major axis, AU.
/// * `eccentricity` — orbital eccentricity, `0.0 <= eccentricity < 1.0`.
/// * `true_anomaly_rad` — true anomaly, radians.
///
/// # Returns
///
/// Distance from the Sun, AU.
fn radius_at_true_anomaly(
    semi_major_axis_au: f64,
    eccentricity: f64,
    true_anomaly_rad: f64,
) -> f64 {
    semi_major_axis_au * (1.0 - eccentricity * eccentricity)
        / (1.0 + eccentricity * true_anomaly_rad.cos())
}

/// A point on `elems`'s orbit at a given true anomaly, in heliocentric
/// ecliptic Cartesian coordinates.
///
/// Computes the point in the perifocal frame (`x` toward periapsis, `y` 90°
/// ahead in the direction of motion, `z = 0`) and rotates it into the
/// ecliptic frame by the standard `R_z(Omega) * R_x(i) * R_z(omega)`
/// composition.
///
/// # Arguments
///
/// * `elems` — osculating elements; only the shape (`a`, `e`, `i`, `Omega`,
///   `omega`) is used, not the epoch or mean anomaly.
/// * `true_anomaly_rad` — true anomaly at which to evaluate the position,
///   radians.
///
/// # Returns
///
/// `[x, y, z]` in AU, heliocentric, ecliptic mean J2000 — the same frame
/// [`outfit::jpl_ephem::EphemerisFrame::Ecliptic`] returns, so planet
/// positions (see [`super::ephem_provider`]) and tracked-object positions
/// plot in the same frame without any further rotation.
pub fn point_at_true_anomaly(elems: &Keplerian, true_anomaly_rad: f64) -> [f64; 3] {
    let r = radius_at_true_anomaly(
        elems.semi_major_axis_au,
        elems.eccentricity,
        true_anomaly_rad,
    );
    let x_pf = r * true_anomaly_rad.cos();
    let y_pf = r * true_anomaly_rad.sin();

    let (sin_i, cos_i) = elems.inclination_rad().sin_cos();
    let (sin_node, cos_node) = elems.ascending_node_longitude_rad().sin_cos();
    let (sin_peri, cos_peri) = elems.periapsis_argument_rad().sin_cos();

    let x = (cos_node * cos_peri - sin_node * sin_peri * cos_i) * x_pf
        + (-cos_node * sin_peri - sin_node * cos_peri * cos_i) * y_pf;
    let y = (sin_node * cos_peri + cos_node * sin_peri * cos_i) * x_pf
        + (-sin_node * sin_peri + cos_node * cos_peri * cos_i) * y_pf;
    let z = (sin_peri * sin_i) * x_pf + (cos_peri * sin_i) * y_pf;

    [x, y, z]
}

/// Samples `elems`'s full orbital ellipse as a closed polyline.
///
/// This is pure shape: it walks the true anomaly uniformly over one full
/// revolution, so the sampling is denser near periapsis in *time* than in
/// arc length — irrelevant for drawing the ellipse itself, since every point
/// still lies exactly on it.
///
/// # Arguments
///
/// * `elems` — osculating elements to trace.
/// * `n_samples` — number of points to emit. Must be at least 2; the curve
///   is closed by construction (the first and last sampled true anomalies
///   are `0` and `2*PI * (n_samples - 1) / n_samples`, one step short of a
///   full turn, so the caller can close the polyline by re-appending the
///   first point if a visually closed loop is wanted).
///
/// # Returns
///
/// `n_samples` heliocentric ecliptic points, AU, in the same frame as
/// [`point_at_true_anomaly`].
pub fn ellipse_points(elems: &Keplerian, n_samples: usize) -> Vec<[f64; 3]> {
    let n_samples = n_samples.max(2);
    (0..n_samples)
        .map(|i| {
            let true_anomaly_rad = std::f64::consts::TAU * (i as f64) / (n_samples as f64);
            point_at_true_anomaly(elems, true_anomaly_rad)
        })
        .collect()
}

/// The heliocentric position `elems`'s body occupies at `target_epoch_mjd_tt`,
/// propagated by unperturbed two-body Keplerian motion from `elems`'s own
/// epoch.
///
/// # Arguments
///
/// * `elems` — osculating elements, valid at `elems.epoch_mjd_tt`.
/// * `target_epoch_mjd_tt` — epoch to propagate to, MJD-TT. May be before or
///   after `elems.epoch_mjd_tt`.
///
/// # Returns
///
/// `[x, y, z]` in AU, heliocentric ecliptic mean J2000, same frame as
/// [`point_at_true_anomaly`].
pub fn position_at_epoch(elems: &Keplerian, target_epoch_mjd_tt: f64) -> [f64; 3] {
    let dt_days = target_epoch_mjd_tt - elems.epoch_mjd_tt;
    let n = mean_motion_rad_per_day(elems.semi_major_axis_au);
    let mean_anomaly_rad = elems.mean_anomaly_rad() + n * dt_days;

    let eccentric_anomaly_rad = solve_eccentric_anomaly(mean_anomaly_rad, elems.eccentricity);
    let true_anomaly_rad = true_anomaly_from_eccentric(elems.eccentricity, eccentric_anomaly_rad);

    point_at_true_anomaly(elems, true_anomaly_rad)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(p: [f64; 3]) -> f64 {
        (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
    }

    fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
        norm([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
    }

    /// A circular orbit (`e = 0`) has `E = M` exactly, at any mean anomaly.
    #[test]
    fn solve_eccentric_anomaly_is_identity_for_circular_orbits() {
        for m_deg in [0.0_f64, 45.0, 90.0, 180.0, 270.0, 359.0] {
            let m = m_deg.to_radians();
            let e = solve_eccentric_anomaly(m, 0.0);
            assert!((e - m).abs() < 1e-12, "m={m_deg} e={e} expected={m}");
        }
    }

    /// For a highly eccentric orbit, the returned eccentric anomaly must
    /// satisfy Kepler's equation to within the solver's own tolerance.
    #[test]
    fn solve_eccentric_anomaly_converges_at_high_eccentricity() {
        let eccentricity = 0.99;
        for m_deg in [0.0_f64, 10.0, 90.0, 179.0, 270.0] {
            let m = m_deg.to_radians();
            let e_anom = solve_eccentric_anomaly(m, eccentricity);
            let residual = e_anom - eccentricity * e_anom.sin() - wrap_2pi(m);
            assert!(
                residual.abs() < 1e-9,
                "m={m_deg} e_anom={e_anom} residual={residual}"
            );
        }
    }

    fn earth_like(epoch_mjd_tt: f64, mean_anomaly_deg: f64) -> Keplerian {
        Keplerian {
            epoch_mjd_tt,
            semi_major_axis_au: 1.000_000_11,
            eccentricity: 0.016_710_22,
            inclination_deg: 0.000_05,
            ascending_node_longitude_deg: -11.260_64,
            periapsis_argument_deg: 114.207_83,
            mean_anomaly_deg,
        }
    }

    /// A circular-orbit ellipse (`e = 0`) has every sampled point at exactly
    /// `a` from the Sun.
    #[test]
    fn ellipse_points_circular_orbit_is_a_perfect_circle() {
        let elems = Keplerian {
            epoch_mjd_tt: 60_000.0,
            semi_major_axis_au: 2.5,
            eccentricity: 0.0,
            inclination_deg: 12.0,
            ascending_node_longitude_deg: 40.0,
            periapsis_argument_deg: 70.0,
            mean_anomaly_deg: 0.0,
        };
        for p in ellipse_points(&elems, 64) {
            assert!(
                (norm(p) - elems.semi_major_axis_au).abs() < 1e-9,
                "point {p:?} at distance {} from a {} AU circular orbit",
                norm(p),
                elems.semi_major_axis_au
            );
        }
    }

    /// A planar orbit (`i = 0`) never leaves the ecliptic plane.
    #[test]
    fn ellipse_points_planar_orbit_has_zero_z() {
        let elems = Keplerian {
            epoch_mjd_tt: 60_000.0,
            semi_major_axis_au: 3.0,
            eccentricity: 0.3,
            inclination_deg: 0.0,
            ascending_node_longitude_deg: 25.0,
            periapsis_argument_deg: 200.0,
            mean_anomaly_deg: 0.0,
        };
        for p in ellipse_points(&elems, 64) {
            assert!(
                p[2].abs() < 1e-9,
                "point {p:?} has nonzero z on a planar orbit"
            );
        }
    }

    /// The first sampled point (true anomaly 0) is periapsis, at `a(1-e)`;
    /// the midpoint sample (true anomaly PI) is apoapsis, at `a(1+e)`.
    #[test]
    fn ellipse_points_periapsis_and_apoapsis_distances() {
        let elems = Keplerian {
            epoch_mjd_tt: 60_000.0,
            semi_major_axis_au: 4.0,
            eccentricity: 0.2,
            inclination_deg: 5.0,
            ascending_node_longitude_deg: 15.0,
            periapsis_argument_deg: 30.0,
            mean_anomaly_deg: 0.0,
        };
        let n_samples = 64;
        let points = ellipse_points(&elems, n_samples);

        let periapsis = norm(points[0]);
        let apoapsis = norm(points[n_samples / 2]);

        assert!((periapsis - elems.semi_major_axis_au * (1.0 - elems.eccentricity)).abs() < 1e-9);
        assert!((apoapsis - elems.semi_major_axis_au * (1.0 + elems.eccentricity)).abs() < 1e-9);
    }

    /// Propagating by exactly one full orbital period returns to the
    /// starting point.
    #[test]
    fn position_at_epoch_returns_to_start_after_one_period() {
        let elems = earth_like(60_000.0, 100.0);
        let period_days = std::f64::consts::TAU / mean_motion_rad_per_day(elems.semi_major_axis_au);

        let start = position_at_epoch(&elems, elems.epoch_mjd_tt);
        let after_one_period = position_at_epoch(&elems, elems.epoch_mjd_tt + period_days);

        assert!(
            dist(start, after_one_period) < 1e-6,
            "start={start:?} after_one_period={after_one_period:?}"
        );
        // Sanity check the period itself is close to a year.
        assert!(
            (period_days - 365.25).abs() < 1.0,
            "period={period_days} days"
        );
    }

    /// `position_at_epoch` at the elements' own epoch must land exactly on
    /// `point_at_true_anomaly` evaluated at the true anomaly corresponding to
    /// the elements' own mean anomaly — the zero-propagation case.
    #[test]
    fn position_at_epoch_at_own_epoch_matches_point_at_true_anomaly() {
        // Non-trivial values in the range a real `orbit_fits.keplerian` row
        // would carry (a typical outer main-belt asteroid).
        let elems = Keplerian {
            epoch_mjd_tt: 60_123.456,
            semi_major_axis_au: 2.741,
            eccentricity: 0.187,
            inclination_deg: 9.32,
            ascending_node_longitude_deg: 202.5,
            periapsis_argument_deg: 88.1,
            mean_anomaly_deg: 317.9,
        };

        let e_anom = solve_eccentric_anomaly(elems.mean_anomaly_rad(), elems.eccentricity);
        let true_anomaly_rad = true_anomaly_from_eccentric(elems.eccentricity, e_anom);
        let expected = point_at_true_anomaly(&elems, true_anomaly_rad);

        let actual = position_at_epoch(&elems, elems.epoch_mjd_tt);
        assert!(
            dist(expected, actual) < 1e-9,
            "expected={expected:?} actual={actual:?}"
        );
    }
}
