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
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
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

/// The characteristic points of an orbit, heliocentric ecliptic, AU.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Landmarks {
    /// Closest point to the Sun (true anomaly `0`).
    pub perihelion: [f64; 3],
    /// Farthest point from the Sun (true anomaly `π`).
    pub aphelion: [f64; 3],
    /// Where the orbit crosses the ecliptic going north (`z` increasing).
    pub ascending_node: [f64; 3],
    /// Where the orbit crosses the ecliptic going south.
    pub descending_node: [f64; 3],
}

/// The perihelion, aphelion and both nodes of an orbit.
///
/// The nodes sit at true anomalies `-ω` (ascending) and `π - ω`
/// (descending), where `ω` is the argument of periapsis: the line joining
/// them is the line of nodes, which passes through the Sun.
///
/// # Arguments
///
/// * `elems` — the orbit; a closed ellipse (`0 <= e < 1`).
///
/// # Returns
///
/// The four [`Landmarks`], in the same frame as [`point_at_true_anomaly`].
pub fn orbit_landmarks(elems: &Keplerian) -> Landmarks {
    let omega = elems.periapsis_argument_rad();
    Landmarks {
        perihelion: point_at_true_anomaly(elems, 0.0),
        aphelion: point_at_true_anomaly(elems, std::f64::consts::PI),
        ascending_node: point_at_true_anomaly(elems, -omega),
        descending_node: point_at_true_anomaly(elems, std::f64::consts::PI - omega),
    }
}

/// Perihelion distance $q = a(1-e)$, AU.
///
/// # Arguments
///
/// * `elems` — the orbit.
///
/// # Returns
///
/// The closest distance to the Sun, AU.
pub fn perihelion_distance_au(elems: &Keplerian) -> f64 {
    elems.semi_major_axis_au * (1.0 - elems.eccentricity)
}

/// Aphelion distance $Q = a(1+e)$, AU.
///
/// # Arguments
///
/// * `elems` — the orbit.
///
/// # Returns
///
/// The farthest distance from the Sun, AU.
pub fn aphelion_distance_au(elems: &Keplerian) -> f64 {
    elems.semi_major_axis_au * (1.0 + elems.eccentricity)
}

/// Orbital period, days, from Kepler's third law with the Sun's GM.
///
/// # Arguments
///
/// * `elems` — the orbit; `semi_major_axis_au` must be strictly positive.
///
/// # Returns
///
/// The period in days.
pub fn period_days(elems: &Keplerian) -> f64 {
    std::f64::consts::TAU / mean_motion_rad_per_day(elems.semi_major_axis_au)
}

/// Number of true-anomaly samples per orbit in [`moid`]'s coarse search —
/// $720^2$ pair distances per call, a few milliseconds.
const MOID_COARSE_SAMPLES: usize = 720;
/// Step size, radians of true anomaly, below which [`moid`]'s local
/// refinement stops.
const MOID_REFINE_TOLERANCE_RAD: f64 = 1e-10;

/// The minimum orbit intersection distance between two orbits.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Moid {
    /// The smallest distance between any point of one orbit and any point
    /// of the other, AU.
    pub distance_au: f64,
    /// The point on the first orbit where it is reached.
    pub point_a: [f64; 3],
    /// The point on the second orbit where it is reached.
    pub point_b: [f64; 3],
}

/// The minimum orbit intersection distance (MOID) between two Keplerian
/// orbits.
///
/// A coarse grid search over both true anomalies finds the neighbourhood of
/// the global minimum, then a pattern search refines it. Purely geometric:
/// it compares the two fixed ellipses, not where the bodies are at any
/// date, and takes no account of perturbations.
///
/// # Arguments
///
/// * `a`, `b` — the two orbits; closed ellipses.
///
/// # Returns
///
/// The [`Moid`]. Symmetric in `a` and `b` up to the swapped points.
pub fn moid(a: &Keplerian, b: &Keplerian) -> Moid {
    let step = std::f64::consts::TAU / MOID_COARSE_SAMPLES as f64;
    let angle = |i: usize| i as f64 * step;
    let points_a: Vec<[f64; 3]> = (0..MOID_COARSE_SAMPLES)
        .map(|i| point_at_true_anomaly(a, angle(i)))
        .collect();
    let points_b: Vec<[f64; 3]> = (0..MOID_COARSE_SAMPLES)
        .map(|i| point_at_true_anomaly(b, angle(i)))
        .collect();

    let squared = |p: [f64; 3], q: [f64; 3]| {
        (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)
    };
    let (mut best_i, mut best_j, mut best) = (0, 0, f64::INFINITY);
    for (i, &p) in points_a.iter().enumerate() {
        for (j, &q) in points_b.iter().enumerate() {
            let d = squared(p, q);
            if d < best {
                (best_i, best_j, best) = (i, j, d);
            }
        }
    }

    let separation = |nu_a: f64, nu_b: f64| {
        distance(
            point_at_true_anomaly(a, nu_a),
            point_at_true_anomaly(b, nu_b),
        )
    };
    let (mut nu_a, mut nu_b) = (angle(best_i), angle(best_j));
    let mut best = best.sqrt();
    let mut h = step;
    while h > MOID_REFINE_TOLERANCE_RAD {
        let candidate = [-1.0, 0.0, 1.0]
            .into_iter()
            .flat_map(|da| [-1.0, 0.0, 1.0].into_iter().map(move |db| (da, db)))
            .filter(|&(da, db)| da != 0.0 || db != 0.0)
            .map(|(da, db)| (nu_a + da * h, nu_b + db * h))
            .map(|(x, y)| (separation(x, y), x, y))
            .min_by(|l, r| l.0.total_cmp(&r.0));
        match candidate {
            Some((d, x, y)) if d < best => (best, nu_a, nu_b) = (d, x, y),
            _ => h /= 2.0,
        }
    }

    Moid {
        distance_au: best,
        point_a: point_at_true_anomaly(a, nu_a),
        point_b: point_at_true_anomaly(b, nu_b),
    }
}

/// The outline of a disc lying in the ecliptic plane, centred on the Sun.
///
/// # Arguments
///
/// * `radius` — disc radius, AU.
/// * `n` — number of outline points; clamped to at least 3.
///
/// # Returns
///
/// `n` points on the circle `x² + y² = radius²`, `z = 0`, in counter-clockwise
/// order, the first at `(radius, 0, 0)`.
pub fn ecliptic_disc_outline(radius: f64, n: usize) -> Vec<[f64; 3]> {
    let n = n.max(3);
    (0..n)
        .map(|i| {
            let angle = std::f64::consts::TAU * i as f64 / n as f64;
            [radius * angle.cos(), radius * angle.sin(), 0.0]
        })
        .collect()
}

/// A pie-slice showing an orbit's inclination to the ecliptic, and where to
/// label it.
#[derive(Clone, Debug, PartialEq)]
pub struct InclinationWedge {
    /// The slice's outline as one polyline: a ray along the ecliptic, back to
    /// the Sun, out along the orbital plane's steepest direction, then the
    /// arc from the orbital plane back down to the ecliptic.
    pub outline: Vec<[f64; 3]>,
    /// The middle of the arc, where an icon can sit.
    pub arc_midpoint: [f64; 3],
}

/// Builds the [`InclinationWedge`] of an orbit: the angle between the
/// ecliptic and the orbital plane, drawn where it is measured — in the plane
/// perpendicular to the line of nodes, at the Sun.
///
/// With `û = (-sin Ω, cos Ω, 0)` (the ecliptic direction perpendicular to
/// the line of nodes) and `ẑ` the ecliptic pole, the orbital plane's
/// steepest direction is `cos i · û + sin i · ẑ`, so the slice spans the
/// angle `i` from `û` to that direction.
///
/// # Arguments
///
/// * `ascending_node_longitude_deg` — longitude of the ascending node `Ω`,
///   degrees.
/// * `inclination_deg` — inclination `i`, degrees, in `[0, 180]`.
/// * `radius` — length of the slice's rays and arc, AU.
/// * `arc_samples` — number of segments of the arc; clamped to at least 1.
///
/// # Returns
///
/// The wedge; degenerate (a single ray) when `i = 0`.
pub fn inclination_wedge(
    ascending_node_longitude_deg: f64,
    inclination_deg: f64,
    radius: f64,
    arc_samples: usize,
) -> InclinationWedge {
    let node = ascending_node_longitude_deg.to_radians();
    let inclination = inclination_deg.to_radians();
    let arc_samples = arc_samples.max(1);

    let u = [-node.sin(), node.cos(), 0.0];
    let at = |angle: f64, scale: f64| {
        [
            scale * radius * angle.cos() * u[0],
            scale * radius * angle.cos() * u[1],
            scale * radius * angle.sin(),
        ]
    };

    let mut outline = vec![at(0.0, 1.0), [0.0; 3]];
    outline.extend((0..=arc_samples).map(|i| {
        let angle = inclination * (1.0 - i as f64 / arc_samples as f64);
        at(angle, 1.0)
    }));

    InclinationWedge {
        outline,
        arc_midpoint: at(inclination / 2.0, 1.0),
    }
}

/// Kilometres in one astronomical unit (the IAU 2012 definition).
pub const KM_PER_AU: f64 = 149_597_870.7;
/// Kilometres in one lunar distance (the Moon's mean orbital distance).
pub const KM_PER_LUNAR_DISTANCE: f64 = 384_400.0;

/// Euclidean distance between two points.
///
/// # Arguments
///
/// * `a`, `b` — points in the same Cartesian frame and units.
///
/// # Returns
///
/// `|a - b|`, in the points' units.
pub fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Phase angle of a body: the angle at the body between the directions to
/// the Sun and to the observer (Sun–body–observer).
///
/// The same formula as `outfit`'s (crate-private) `phase_angle`,
/// $\phi = \arccos\left(\frac{\mathbf{r}\_\mathrm{body} \cdot \mathbf{d}}{r\_\mathrm{helio}\,\rho}\right)$
/// with $\mathbf{d} = \mathbf{r}\_\mathrm{body} - \mathbf{r}\_\mathrm{obs}$, applied to
/// positions already in hand rather than to a full propagated state. The
/// cosine is clamped to `[-1, 1]` before `acos`, so an exact opposition or
/// conjunction cannot produce `NaN`.
///
/// # Arguments
///
/// * `body` — heliocentric position of the body, AU.
/// * `observer` — heliocentric position of the observer, AU, same frame.
///
/// # Returns
///
/// The phase angle in radians, in `[0, π]`: `0` when the Sun and the
/// observer lie in the same direction from the body (full illumination, as
/// at opposition), `π` when they lie in opposite directions (the observer
/// sees only the dark side). `0` in the degenerate case of a body at the
/// Sun's or the observer's position, where the angle is undefined.
pub fn phase_angle_rad(body: [f64; 3], observer: [f64; 3]) -> f64 {
    let d = [
        body[0] - observer[0],
        body[1] - observer[1],
        body[2] - observer[2],
    ];
    let r_helio = distance(body, [0.0; 3]);
    let rho = distance(body, observer);
    if r_helio <= 0.0 || rho <= 0.0 {
        return 0.0;
    }
    let dot = body[0] * d[0] + body[1] * d[1] + body[2] * d[2];
    (dot / (r_helio * rho)).clamp(-1.0, 1.0).acos()
}

/// Solar elongation of a body: the angle at the observer between the
/// directions to the Sun and to the body (Sun–observer–body).
///
/// Small values mean the body was close to the Sun on the sky — hard or
/// impossible to observe from the ground. Same convention as `outfit`'s
/// crate-private `solar_elongation`,
/// $\varepsilon = \arccos\left(\frac{-\mathbf{r}\_\mathrm{obs} \cdot \mathbf{d}}{|\mathbf{r}\_\mathrm{obs}|\,\rho}\right)$
/// with $\mathbf{d} = \mathbf{r}\_\mathrm{body} - \mathbf{r}\_\mathrm{obs}$.
///
/// # Arguments
///
/// * `body` — heliocentric position of the body, AU.
/// * `observer` — heliocentric position of the observer, AU, same frame.
///
/// # Returns
///
/// The elongation in radians, in `[0, π]`: `0` when the body lies in the
/// Sun's direction, `π` at opposition. `0` in the degenerate case of an
/// observer at the Sun or a body at the observer, where it is undefined.
pub fn solar_elongation_rad(body: [f64; 3], observer: [f64; 3]) -> f64 {
    let d = [
        body[0] - observer[0],
        body[1] - observer[1],
        body[2] - observer[2],
    ];
    let r_obs = distance(observer, [0.0; 3]);
    let rho = distance(body, observer);
    if r_obs <= 0.0 || rho <= 0.0 {
        return 0.0;
    }
    let dot = -(observer[0] * d[0] + observer[1] * d[1] + observer[2] * d[2]);
    (dot / (r_obs * rho)).clamp(-1.0, 1.0).acos()
}

/// Largest phase angle, radians, at which [`absolute_magnitude_hg`] still
/// returns a value — the H,G phase function is only calibrated up to
/// roughly this geometry.
const HG_MAX_PHASE_RAD: f64 = 2.0 * std::f64::consts::FRAC_PI_3;

/// Absolute magnitude $H$ implied by one apparent magnitude, with the
/// Bowell et al. (1989) $H,G$ phase function.
///
/// $H = m - 5\log\_{10}(r\,\Delta) + 2.5\log\_{10}\left((1-G)\,\Phi\_1(\alpha) + G\,\Phi\_2(\alpha)\right)$
/// with $\Phi\_i = \exp(-A\_i \tan^{B\_i}(\alpha/2))$, $(A\_1, B\_1) = (3.33, 0.63)$ and
/// $(A\_2, B\_2) = (1.87, 1.22)$.
///
/// Indicative only: it is $H$ in the observation's own photometric band —
/// no colour correction — and ignores rotational variability.
///
/// # Arguments
///
/// * `apparent_mag` — apparent magnitude $m$.
/// * `helio_distance_au` — heliocentric distance $r$, AU.
/// * `topo_distance_au` — observer–body distance $\Delta$, AU.
/// * `phase_rad` — phase angle $\alpha$, radians.
/// * `g` — slope parameter $G$ (`0.15` is the customary default).
///
/// # Returns
///
/// `Some(H)`, or `None` if an input is not finite, a distance is not
/// positive, or `phase_rad` is outside `[0, 120°]`.
pub fn absolute_magnitude_hg(
    apparent_mag: f64,
    helio_distance_au: f64,
    topo_distance_au: f64,
    phase_rad: f64,
    g: f64,
) -> Option<f64> {
    let inputs = [
        apparent_mag,
        helio_distance_au,
        topo_distance_au,
        phase_rad,
        g,
    ];
    if inputs.iter().any(|x| !x.is_finite())
        || helio_distance_au <= 0.0
        || topo_distance_au <= 0.0
        || !(0.0..=HG_MAX_PHASE_RAD).contains(&phase_rad)
    {
        return None;
    }

    let tan_half = (phase_rad / 2.0).tan();
    let phi_1 = (-3.33 * tan_half.powf(0.63)).exp();
    let phi_2 = (-1.87 * tan_half.powf(1.22)).exp();
    let phase_function = (1.0 - g) * phi_1 + g * phi_2;

    Some(
        apparent_mag - 5.0 * (helio_distance_au * topo_distance_au).log10()
            + 2.5 * phase_function.log10(),
    )
}

/// Unit vector of an equatorial sky direction.
///
/// # Arguments
///
/// * `ra_rad` — right ascension, radians.
/// * `dec_rad` — declination, radians.
///
/// # Returns
///
/// `[cos(dec)·cos(ra), cos(dec)·sin(ra), sin(dec)]` — a unit vector in the
/// equatorial frame the angles are expressed in (no rotation applied).
pub fn equatorial_unit_vector(ra_rad: f64, dec_rad: f64) -> [f64; 3] {
    let cos_dec = dec_rad.cos();
    [
        cos_dec * ra_rad.cos(),
        cos_dec * ra_rad.sin(),
        dec_rad.sin(),
    ]
}

/// The point on an observer's line of sight closest to a predicted position.
///
/// A sky position (ra/dec) fixes only a direction, not a distance. To draw
/// an observation in 3D anyway, this picks the distance along the measured
/// direction at which the line of sight passes nearest to where the fitted
/// orbit says the object was: the returned point therefore sits on the
/// measured line of sight, and its offset from `target` is exactly the part
/// of the fit residual perpendicular to it.
///
/// # Arguments
///
/// * `observer` — observer position, in any Cartesian frame.
/// * `direction` — **unit** vector of the measured line of sight, in the
///   same frame.
/// * `target` — the orbit-predicted object position at the observation
///   epoch, in the same frame.
///
/// # Returns
///
/// `observer + t·direction`, where `t = max(0, (target - observer)·direction)`:
/// the orthogonal projection of `target` onto the line of sight, clamped so
/// the point never lies behind the observer.
pub fn point_on_line_of_sight_nearest(
    observer: [f64; 3],
    direction: [f64; 3],
    target: [f64; 3],
) -> [f64; 3] {
    let t = (0..3)
        .map(|i| (target[i] - observer[i]) * direction[i])
        .sum::<f64>()
        .max(0.0);
    [
        observer[0] + t * direction[0],
        observer[1] + t * direction[1],
        observer[2] + t * direction[2],
    ]
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
    /// Sun and observer on the same side of the body: full illumination.
    /// Observer at 1 AU, body on the far side of the Sun: opposition of the
    /// body from the Sun's point of view.
    fn orbit(a: f64, e: f64, i: f64, node: f64, peri: f64) -> Keplerian {
        Keplerian {
            epoch_mjd_tt: 60_000.0,
            semi_major_axis_au: a,
            eccentricity: e,
            inclination_deg: i,
            ascending_node_longitude_deg: node,
            periapsis_argument_deg: peri,
            mean_anomaly_deg: 0.0,
        }
    }

    #[test]
    fn ecliptic_disc_outline_is_a_circle_in_the_ecliptic() {
        let outline = ecliptic_disc_outline(3.0, 48);
        assert_eq!(outline.len(), 48);
        assert!(dist(outline[0], [3.0, 0.0, 0.0]) < 1e-12);
        for p in &outline {
            assert_eq!(p[2], 0.0);
            assert!((norm(*p) - 3.0).abs() < 1e-12);
        }
        assert_eq!(ecliptic_disc_outline(1.0, 0).len(), 3);
    }

    /// The slice lies in the plane perpendicular to the line of nodes, and
    /// its rays make exactly the inclination angle.
    #[test]
    fn inclination_wedge_spans_the_inclination_in_the_plane_normal_to_the_nodes() {
        let (node_deg, incl_deg, r) = (80.0_f64, 25.0_f64, 1.5);
        let wedge = inclination_wedge(node_deg, incl_deg, r, 20);

        let node = node_deg.to_radians();
        let line_of_nodes = [node.cos(), node.sin(), 0.0];
        for p in &wedge.outline {
            let along_nodes: f64 = (0..3).map(|i| p[i] * line_of_nodes[i]).sum();
            assert!(along_nodes.abs() < 1e-12, "{p:?} leaves the plane");
        }

        // First ray lies in the ecliptic; the second one leaves it by `i`.
        let ecliptic_ray = wedge.outline[0];
        let orbital_ray = wedge.outline[2];
        assert_eq!(ecliptic_ray[2], 0.0);
        assert_eq!(wedge.outline[1], [0.0; 3]);
        assert!((norm(ecliptic_ray) - r).abs() < 1e-12);
        assert!((norm(orbital_ray) - r).abs() < 1e-12);
        let cos_angle: f64 = (0..3)
            .map(|i| ecliptic_ray[i] * orbital_ray[i])
            .sum::<f64>()
            / (r * r);
        assert!((cos_angle - incl_deg.to_radians().cos()).abs() < 1e-12);
        assert!((orbital_ray[2] - r * incl_deg.to_radians().sin()).abs() < 1e-12);
    }

    #[test]
    fn inclination_wedge_arc_closes_the_slice_and_stays_at_the_radius() {
        let wedge = inclination_wedge(30.0, 40.0, 2.0, 16);
        // Two rays' end points, then the arc (arc_samples + 1 points).
        assert_eq!(wedge.outline.len(), 2 + 17);
        assert!(dist(*wedge.outline.last().unwrap(), wedge.outline[0]) < 1e-12);
        for p in &wedge.outline[2..] {
            assert!((norm(*p) - 2.0).abs() < 1e-12);
        }
        // The midpoint sits on the arc, at half the inclination.
        assert!((norm(wedge.arc_midpoint) - 2.0).abs() < 1e-12);
        let mid = wedge.arc_midpoint[2] / norm(wedge.arc_midpoint);
        assert!((mid - 20.0_f64.to_radians().sin()).abs() < 1e-12);
    }

    #[test]
    fn landmarks_of_a_flat_eccentric_orbit_lie_on_the_axes() {
        // i = 0, node = 0, omega = 0: perihelion on +x, aphelion on -x.
        let l = orbit_landmarks(&orbit(2.0, 0.5, 0.0, 0.0, 0.0));
        assert!(dist(l.perihelion, [1.0, 0.0, 0.0]) < 1e-9);
        assert!(dist(l.aphelion, [-3.0, 0.0, 0.0]) < 1e-9);
    }

    /// Both nodes are on the ecliptic (`z = 0`), on opposite sides of the
    /// Sun along the line of nodes.
    #[test]
    fn landmark_nodes_are_on_the_ecliptic_and_on_the_line_of_nodes() {
        let elems = orbit(2.7, 0.3, 25.0, 80.0, 60.0);
        let l = orbit_landmarks(&elems);
        assert!(l.ascending_node[2].abs() < 1e-9);
        assert!(l.descending_node[2].abs() < 1e-9);

        // The ascending node points along the longitude of the node.
        let expected = [
            80.0_f64.to_radians().cos(),
            80.0_f64.to_radians().sin(),
            0.0,
        ];
        let r = norm(l.ascending_node);
        assert!(dist(l.ascending_node, [expected[0] * r, expected[1] * r, 0.0]) < 1e-9);

        // Collinear with the Sun: the descending node is opposite.
        let rd = norm(l.descending_node);
        assert!(
            dist(
                l.descending_node,
                [-expected[0] * rd, -expected[1] * rd, 0.0]
            ) < 1e-9
        );
    }

    #[test]
    fn perihelion_aphelion_and_period_follow_from_the_elements() {
        let elems = orbit(2.5, 0.2, 10.0, 0.0, 0.0);
        assert!((perihelion_distance_au(&elems) - 2.0).abs() < 1e-12);
        assert!((aphelion_distance_au(&elems) - 3.0).abs() < 1e-12);
        // Kepler: P[yr] = a^1.5 (the Sun's GM here is the Gaussian one).
        let years = period_days(&elems) / 365.25;
        assert!((years - 2.5_f64.powf(1.5)).abs() < 0.01, "{years}");
    }

    #[test]
    fn moid_of_concentric_coplanar_circles_is_their_radius_difference() {
        let m = moid(
            &orbit(1.0, 0.0, 0.0, 0.0, 0.0),
            &orbit(2.0, 0.0, 0.0, 0.0, 0.0),
        );
        assert!((m.distance_au - 1.0).abs() < 1e-9, "{}", m.distance_au);
        assert!((norm(m.point_a) - 1.0).abs() < 1e-9);
        assert!((norm(m.point_b) - 2.0).abs() < 1e-9);
    }

    /// Two unit circles inclined by 90° intersect at their common nodes.
    #[test]
    fn moid_of_intersecting_orbits_is_zero() {
        let m = moid(
            &orbit(1.0, 0.0, 0.0, 0.0, 0.0),
            &orbit(1.0, 0.0, 90.0, 0.0, 0.0),
        );
        assert!(m.distance_au < 1e-6, "{}", m.distance_au);
    }

    /// A tilted circle 0.3 AU outside: the closest approach is along the
    /// line of nodes, where the two circles are exactly 0.3 AU apart.
    #[test]
    fn moid_of_an_inclined_outer_circle_is_found_at_the_nodes() {
        let m = moid(
            &orbit(1.0, 0.0, 0.0, 0.0, 0.0),
            &orbit(1.3, 0.0, 20.0, 0.0, 0.0),
        );
        assert!((m.distance_au - 0.3).abs() < 1e-6, "{}", m.distance_au);
    }

    #[test]
    fn moid_is_symmetric() {
        let a = orbit(1.0, 0.02, 0.0, 0.0, 100.0);
        let b = orbit(2.4, 0.35, 12.0, 70.0, 30.0);
        let ab = moid(&a, &b);
        let ba = moid(&b, &a);
        assert!((ab.distance_au - ba.distance_au).abs() < 1e-8);
    }

    #[test]
    fn solar_elongation_is_pi_at_opposition_and_zero_towards_the_sun() {
        let observer = [1.0, 0.0, 0.0];
        assert!(
            (solar_elongation_rad([3.0, 0.0, 0.0], observer) - std::f64::consts::PI).abs() < 1e-12
        );
        assert!(solar_elongation_rad([0.5, 0.0, 0.0], observer).abs() < 1e-12);
        let quadrature = solar_elongation_rad([1.0, 2.0, 0.0], observer);
        assert!((quadrature - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn solar_elongation_of_a_degenerate_geometry_is_zero() {
        assert_eq!(solar_elongation_rad([1.0, 0.0, 0.0], [0.0; 3]), 0.0);
        assert_eq!(solar_elongation_rad([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]), 0.0);
    }

    /// At zero phase angle the phase function is 1, so only the distance
    /// term remains.
    #[test]
    fn absolute_magnitude_at_zero_phase_only_removes_the_distance_term() {
        let h = absolute_magnitude_hg(20.0, 2.0, 1.0, 0.0, 0.15).unwrap();
        assert!((h - (20.0 - 5.0 * 2.0_f64.log10())).abs() < 1e-12);
    }

    /// The same apparent magnitude at a larger phase angle means a brighter
    /// object (the phase function dims it), i.e. a smaller H.
    #[test]
    fn absolute_magnitude_decreases_with_phase_angle() {
        let phases = [0.0, 0.2, 0.5, 1.0, 1.5];
        let hs: Vec<f64> = phases
            .iter()
            .map(|&a| absolute_magnitude_hg(20.0, 2.0, 1.0, a, 0.15).unwrap())
            .collect();
        for pair in hs.windows(2) {
            assert!(pair[1] < pair[0], "{hs:?}");
        }
    }

    #[test]
    fn absolute_magnitude_is_none_outside_the_valid_domain() {
        assert!(absolute_magnitude_hg(20.0, 2.0, 1.0, 2.5, 0.15).is_none());
        assert!(absolute_magnitude_hg(20.0, 2.0, 1.0, -0.1, 0.15).is_none());
        assert!(absolute_magnitude_hg(20.0, 0.0, 1.0, 0.3, 0.15).is_none());
        assert!(absolute_magnitude_hg(f64::NAN, 2.0, 1.0, 0.3, 0.15).is_none());
    }

    #[test]
    fn phase_angle_is_zero_at_opposition() {
        // Observer at 1 AU, body at 3 AU on the same side of the Sun.
        assert!(phase_angle_rad([3.0, 0.0, 0.0], [1.0, 0.0, 0.0]).abs() < 1e-12);
    }

    /// Body between the Sun and the observer: only its dark side faces us.
    #[test]
    fn phase_angle_is_pi_when_the_body_is_between_sun_and_observer() {
        let phi = phase_angle_rad([1.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
        assert!((phi - std::f64::consts::PI).abs() < 1e-12);
    }

    /// Sun–body–observer right angle: body at (1,0,0), Sun at the origin
    /// (direction −x from the body), observer at (1,1,0) (direction +y).
    #[test]
    fn phase_angle_is_a_right_angle_for_perpendicular_directions() {
        let phi = phase_angle_rad([1.0, 0.0, 0.0], [1.0, 1.0, 0.0]);
        assert!((phi - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn phase_angle_of_a_degenerate_geometry_is_zero() {
        assert_eq!(phase_angle_rad([0.0; 3], [1.0, 0.0, 0.0]), 0.0);
        assert_eq!(phase_angle_rad([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn distance_is_euclidean_and_symmetric() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 6.0, 3.0];
        assert!((distance(a, b) - 5.0).abs() < 1e-12);
        assert_eq!(distance(a, b), distance(b, a));
        assert_eq!(distance(a, a), 0.0);
    }

    #[test]
    fn equatorial_unit_vector_points_along_the_expected_axes() {
        let x = equatorial_unit_vector(0.0, 0.0);
        let y = equatorial_unit_vector(std::f64::consts::FRAC_PI_2, 0.0);
        let z = equatorial_unit_vector(1.234, std::f64::consts::FRAC_PI_2);
        assert!(dist(x, [1.0, 0.0, 0.0]) < 1e-12);
        assert!(dist(y, [0.0, 1.0, 0.0]) < 1e-12);
        assert!(dist(z, [0.0, 0.0, 1.0]) < 1e-12);
    }

    #[test]
    fn equatorial_unit_vector_has_unit_norm() {
        for &(ra, dec) in &[(0.3, -0.7), (4.1, 1.2), (6.0, 0.0)] {
            assert!((norm(equatorial_unit_vector(ra, dec)) - 1.0).abs() < 1e-12);
        }
    }

    /// A target already on the line of sight is returned unchanged.
    #[test]
    fn line_of_sight_point_recovers_a_target_on_the_line() {
        let observer = [1.0, 0.5, -0.2];
        let direction = equatorial_unit_vector(0.8, 0.3);
        let target = [
            observer[0] + 2.5 * direction[0],
            observer[1] + 2.5 * direction[1],
            observer[2] + 2.5 * direction[2],
        ];
        assert!(
            dist(
                point_on_line_of_sight_nearest(observer, direction, target),
                target
            ) < 1e-12
        );
    }

    /// An off-axis target is projected: the result lies on the line, and the
    /// residual to the target is perpendicular to it.
    #[test]
    fn line_of_sight_point_is_the_orthogonal_projection() {
        let observer = [0.0, 0.0, 0.0];
        let direction = [1.0, 0.0, 0.0];
        let target = [3.0, 0.4, -0.1];
        let point = point_on_line_of_sight_nearest(observer, direction, target);
        assert!(dist(point, [3.0, 0.0, 0.0]) < 1e-12);
    }

    /// A target behind the observer clamps to the observer rather than
    /// producing a negative range.
    #[test]
    fn line_of_sight_point_never_lies_behind_the_observer() {
        let observer = [1.0, 1.0, 1.0];
        let direction = [1.0, 0.0, 0.0];
        let target = [-5.0, 1.0, 1.0];
        assert_eq!(
            point_on_line_of_sight_nearest(observer, direction, target),
            observer
        );
    }

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
