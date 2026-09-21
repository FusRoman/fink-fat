//! Wire types shared between the `orbit3d` server functions and the wasm
//! components that plot them. No `#[cfg]` gating: every type here is plain
//! data, safe to compile on both the server and the wasm client.

use serde::{Deserialize, Serialize};

use crate::homepage::family::DynamicalFamily;
use crate::homepage::quality_tier::QualityTier;
use crate::orbit3d::geometry::{Keplerian, Landmarks, Moid};

/// Which group a [`Body3D`] belongs to, so the client can style planets and
/// perturbers differently without string-matching on `name`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BodyKind {
    /// One of the eight major planets.
    Planet,
    /// A dwarf planet or numbered main-belt asteroid plotted for context —
    /// see [`crate::orbit3d::ephem_provider::TRACKED_BODIES`] for exactly
    /// which ones.
    Perturber,
}

/// One planet or perturber's current position and full orbital ellipse, as
/// served by `orbit3d::server_fns::get_planets_3d`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Body3D {
    pub name: String,
    pub kind: BodyKind,
    /// Heliocentric ecliptic position, AU, at the epoch the request was
    /// evaluated at (real JPL/ANISE ephemeris, not a two-body
    /// approximation).
    pub position: [f64; 3],
    /// The body's full orbital ellipse, AU, heliocentric ecliptic mean
    /// J2000 — see [`crate::orbit3d::geometry::ellipse_points`]. Derived
    /// from the osculating elements implied by `position` and the body's
    /// velocity at the same epoch, so it is exact at `position` and only
    /// approximate (unperturbed two-body) away from it.
    pub orbit: Vec<[f64; 3]>,
    /// The osculating elements `orbit` was sampled from, for comparing this
    /// body's orbit with another (see [`crate::orbit3d::geometry::moid`]).
    pub elements: Keplerian,
}

/// One tracked object's current heliocentric position, for the homepage's
/// population-wide 3D scatter. Deliberately carries no orbit curve: the
/// homepage plots positions only (see the module docs on
/// `orbit3d::server_fns` for why), while the lineage page's single-object
/// view is where a full ellipse is shown.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObjectPoint3D {
    pub lineage_id: i64,
    pub family: DynamicalFamily,
    pub tier: QualityTier,
    /// Heliocentric ecliptic position, AU, propagated by unperturbed
    /// two-body motion from the branch's latest fitted elements — see
    /// [`crate::orbit3d::geometry::position_at_epoch`].
    pub position: [f64; 3],
}

/// A single lineage's 3D view: its own orbit plus the planets/perturbers for
/// context — served by `orbit3d::server_fns::get_lineage_orbit3d`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LineageOrbit3D {
    /// The tracked object's current heliocentric position, AU.
    pub object_position: [f64; 3],
    /// The tracked object's full orbital ellipse, AU.
    pub object_orbit: Vec<[f64; 3]>,
    /// Orbit numbers, landmarks and planet MOIDs.
    pub summary: OrbitSummary3D,
    /// The orbit's uncertainty as a cloud of clones drawn from the N-body
    /// fit's covariance; `None` when that fit has no usable covariance (see
    /// [`Self::uncertainty_unavailable_reason`]).
    pub uncertainty: Option<UncertaintyCloud3D>,
    /// Why `uncertainty` is `None`, as a sentence for the user; `None` when
    /// the cloud is available.
    pub uncertainty_unavailable_reason: Option<String>,
    /// One entry per real observation of the lineage that could be placed in
    /// 3D, in observation order. An observation whose observer or ephemeris
    /// could not be resolved is omitted, so this can be shorter than the
    /// lineage's observation list.
    pub observation_points: Vec<ObservationPoint3D>,
    pub planets: Vec<Body3D>,
}

/// One real observation placed in 3D, with the numbers shown in its hover.
///
/// The position is the observer's heliocentric position plus the measured
/// line of sight, at the distance where it passes nearest to the fitted
/// orbit's predicted position (see
/// [`crate::orbit3d::geometry::point_on_line_of_sight_nearest`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationPoint3D {
    /// Heliocentric ecliptic position, AU.
    pub position: [f64; 3],
    /// The observer's heliocentric ecliptic position at the observation
    /// epoch, AU — the other end of the sight line through `position`.
    pub observer_position: [f64; 3],
    /// Observation epoch, MJD-TT.
    pub mjd_tt: f64,
    /// Apparent magnitude of the observation and its 1-sigma error.
    pub magnitude: f64,
    pub mag_err: f64,
    /// The observation's photometric filter code, as stored (LSST
    /// `u, g, r, i, z, y` = `0..=5`).
    pub filter: i16,
    /// The observing site's MPC code.
    pub mpc_code: String,
    /// Solar elongation — the angle at the observer between the Sun and the
    /// object — in degrees, `[0, 180]`.
    pub elongation_deg: f64,
    /// Absolute magnitude $H$ implied by `magnitude` at this geometry
    /// (H,G with `G = 0.15`, in the observation's own band, no colour
    /// correction); `None` when the phase angle is outside the model's
    /// domain.
    pub absolute_magnitude: Option<f64>,
    /// Distance from the Sun to `position`, AU.
    pub heliocentric_distance_au: f64,
    /// Phase angle at `position` — the angle between the directions to the
    /// Sun and to the observer (Sun–object–observer) — in degrees, `[0, 180]`.
    pub phase_angle_deg: f64,
    /// Distance from the observer to `position` — the range along the line
    /// of sight — AU.
    pub topocentric_distance_au: f64,
}

/// The minimum orbit intersection distance between the focus object's orbit
/// and one planet's.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanetMoid3D {
    pub name: String,
    pub moid: Moid,
}

/// Numbers describing the focus object's orbit, for its hover and the orbit
/// landmarks drawn on the plot — served in [`LineageOrbit3D::summary`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrbitSummary3D {
    pub semi_major_axis_au: f64,
    pub eccentricity: f64,
    pub inclination_deg: f64,
    /// Longitude of the ascending node $\Omega$, degrees — the direction of
    /// the line of nodes in the ecliptic.
    pub ascending_node_longitude_deg: f64,
    /// Perihelion distance $q$, AU.
    pub perihelion_au: f64,
    /// Aphelion distance $Q$, AU.
    pub aphelion_au: f64,
    pub period_years: f64,
    /// Distance from the Sun to the object at the view's epoch, AU.
    pub sun_distance_au: f64,
    /// Distance from the Earth to the object at the view's epoch, AU;
    /// `None` if the Earth is missing from the planet list.
    pub earth_distance_au: Option<f64>,
    /// Perihelion, aphelion and nodes, heliocentric ecliptic AU.
    pub landmarks: Landmarks,
    /// MOID against each of the eight planets, in planet order.
    pub moids: Vec<PlanetMoid3D>,
}

/// Number of points of each clone orbit in [`UncertaintyCloud3D::orbits`]
/// — a fixed size the client needs to split the flattened orbits back
/// apart.
pub const UNCERTAINTY_ORBIT_SAMPLES: usize = 90;

/// The focus object's orbit uncertainty, as clones drawn from the N-body
/// fit's covariance.
///
/// Every clone is an orbit sampled from the multivariate normal
/// `N(best solution, covariance)`; clones that are not closed ellipses are
/// discarded.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UncertaintyCloud3D {
    /// A short description of the fit the covariance comes from (method and
    /// quality).
    pub detail: String,
    /// How many clones were drawn.
    pub n_sampled: usize,
    /// How many of them were closed ellipses.
    pub n_clones: usize,
    /// How many clones are in [`Self::at_now`] after discarding the
    /// far-flung ones (see `orbit3d::uncertainty::trim_outliers`).
    pub n_kept_now: usize,
    /// The best solution's own position at the last observation's epoch, AU
    /// — the point the clones' deviations at that epoch are measured from.
    pub center_last_observation: [f64; 3],
    /// The best solution's own position at the view's epoch, AU.
    pub center_now: [f64; 3],
    /// The best solution's orbit, sampled exactly like [`Self::orbits`]
    /// ([`UNCERTAINTY_ORBIT_SAMPLES`] points, same true anomalies), so clone
    /// orbit `k` deviates from it point by point.
    pub best_orbit: Vec<[f64; 3]>,
    /// Every clone's heliocentric position at the last observation's epoch,
    /// AU.
    pub at_last_observation: Vec<[f64; 3]>,
    /// The retained clones' positions at the view's epoch, AU.
    pub at_now: Vec<[f64; 3]>,
    /// A few clone orbits, each [`UNCERTAINTY_ORBIT_SAMPLES`] points.
    pub orbits: Vec<Vec<[f64; 3]>>,
}
