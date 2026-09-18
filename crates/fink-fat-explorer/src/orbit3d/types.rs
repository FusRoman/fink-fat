//! Wire types shared between the `orbit3d` server functions and the wasm
//! components that plot them. No `#[cfg]` gating: every type here is plain
//! data, safe to compile on both the server and the wasm client.

use serde::{Deserialize, Serialize};

use crate::homepage::family::DynamicalFamily;
use crate::homepage::quality_tier::QualityTier;

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
    pub planets: Vec<Body3D>,
}
