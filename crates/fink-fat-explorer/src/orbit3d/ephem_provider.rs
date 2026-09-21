//! Planet and perturber positions, backed by the real JPL/ANISE ephemeris.
//!
//! This deliberately does **not** load its own [`outfit::JPLEphem`]: the
//! engine already loads one — [`crate::get_kalman_context`]'s
//! `ephem_state.jpl`, used today by the lineage page's Kalman replay — and
//! loading a second copy would double the ~200 MB of SPK kernel downloads
//! and parsing that backend already pays for once. [`get_planets`] just
//! reads that same cached context.
//!
//! Bodies are queried at a single requested epoch via
//! [`outfit::JPLEphem::body_ephemeris`], which returns a real
//! ephemeris-derived `(position, velocity)`. The *position* served in
//! [`Body3D::position`] is that exact value. The *orbit curve* is not: there
//! is no cheap way to ask the ephemeris for "the shape of this body's
//! orbit", so the queried `(position, velocity)` is converted into a set of
//! osculating Keplerian elements
//! ([`outfit::OrbitalElements::from_orbital_state`]) and handed to
//! [`crate::orbit3d::geometry::ellipse_points`], the same pure geometry
//! tracked objects' orbits use. That curve is exact at the queried epoch and
//! only approximate (unperturbed two-body) away from it — indistinguishable
//! from the truth at plot scale for a full solar-system view.

use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use hifitime::{Epoch, TimeScale};
use nalgebra::Vector3;
use outfit::jpl_ephem::naif::naif_ids::{
    main_belt::AsteroidNumber, planet_bary::PlanetaryBary, NaifIds,
};
use outfit::{EphemerisFrame, OrbitalElements};
use tokio::sync::OnceCell;

use crate::orbit3d::geometry::{self, Keplerian};
use crate::orbit3d::types::{Body3D, BodyKind};

/// Every planet and perturber the 3D view plots, in the order they are
/// rendered.
///
/// Coverage is limited by what the two loaded SPK kernels contain: the eight
/// planets and Pluto come from the primary `DE440` kernel, Ceres/Pallas/
/// Vesta from the supplementary `codes_300ast_20100725.bsp` kernel
/// [`crate::get_kalman_context`] already loads
/// (`EphemState::new`/`with_main_belt_asteroids`). Other large Kuiper-belt
/// objects (Eris, Haumea, Makemake, ...) are **not** covered by either
/// kernel and are intentionally left out rather than approximated — adding
/// them would need a separate dwarf-planet SPK kernel, left as a follow-up.
pub const TRACKED_BODIES: &[(&str, BodyKind, NaifIds)] = &[
    (
        "Mercury",
        BodyKind::Planet,
        NaifIds::PB(PlanetaryBary::Mercury),
    ),
    ("Venus", BodyKind::Planet, NaifIds::PB(PlanetaryBary::Venus)),
    (
        "Earth",
        BodyKind::Planet,
        NaifIds::PB(PlanetaryBary::EarthMoon),
    ),
    ("Mars", BodyKind::Planet, NaifIds::PB(PlanetaryBary::Mars)),
    (
        "Jupiter",
        BodyKind::Planet,
        NaifIds::PB(PlanetaryBary::Jupiter),
    ),
    (
        "Saturn",
        BodyKind::Planet,
        NaifIds::PB(PlanetaryBary::Saturn),
    ),
    (
        "Uranus",
        BodyKind::Planet,
        NaifIds::PB(PlanetaryBary::Uranus),
    ),
    (
        "Neptune",
        BodyKind::Planet,
        NaifIds::PB(PlanetaryBary::Neptune),
    ),
    (
        "Pluto",
        BodyKind::Perturber,
        NaifIds::PB(PlanetaryBary::Pluto),
    ),
    (
        "Ceres",
        BodyKind::Perturber,
        NaifIds::AST(AsteroidNumber::CERES),
    ),
    (
        "Pallas",
        BodyKind::Perturber,
        NaifIds::AST(AsteroidNumber::PALLAS),
    ),
    (
        "Vesta",
        BodyKind::Perturber,
        NaifIds::AST(AsteroidNumber::VESTA),
    ),
];

/// Number of points sampled per orbital ellipse — dense enough to read as a
/// smooth curve at plot scale without shipping an unnecessarily large
/// payload for twelve bodies.
const ORBIT_CURVE_SAMPLES: usize = 180;

/// How long a built planet/perturber list is served before being recomputed.
///
/// Unlike [`crate::homepage::snapshot`]'s multi-minute rebuild, recomputing
/// this list is a handful of already-loaded-kernel lookups — cheap enough to
/// do inline on cache expiry within a request, so there is no need for that
/// module's "keep serving the stale value, rebuild in the background"
/// machinery. The TTL exists purely to avoid re-querying the ephemeris on
/// every homepage/lineage request; planets move a negligible amount within
/// it.
const PLANETS_CACHE_TTL: Duration = Duration::from_secs(60 * 60);

/// A cached body list plus the [`Instant`] it was built at, so [`get_planets`]
/// can tell whether it has aged past [`PLANETS_CACHE_TTL`].
type PlanetsCacheEntry = (Instant, Arc<Vec<Body3D>>);

static PLANETS_CACHE: OnceCell<RwLock<Option<PlanetsCacheEntry>>> = OnceCell::const_new();

async fn cache_cell() -> &'static RwLock<Option<PlanetsCacheEntry>> {
    PLANETS_CACHE
        .get_or_init(|| async { RwLock::new(None) })
        .await
}

/// The current planet/perturber list, evaluated at `epoch_mjd_tt`, cached for
/// [`PLANETS_CACHE_TTL`].
///
/// A body whose ephemeris lookup fails, or whose instantaneous state is not
/// a closed ellipse (`e >= 1`, never expected for a real planet/perturber but
/// not assumed away), is skipped with a logged warning rather than failing
/// the whole call — a transient issue with one body should not blank the
/// entire plot.
///
/// # Arguments
///
/// * `epoch_mjd_tt` — evaluation epoch, Modified Julian Date, Terrestrial
///   Time.
///
/// # Returns
///
/// The tracked bodies that resolved successfully, in [`TRACKED_BODIES`]
/// order.
pub async fn get_planets(epoch_mjd_tt: f64) -> Arc<Vec<Body3D>> {
    {
        let guard = cache_cell()
            .await
            .read()
            .expect("planets cache lock poisoned");
        if let Some((built_at, bodies)) = &*guard {
            if built_at.elapsed() < PLANETS_CACHE_TTL {
                return bodies.clone();
            }
        }
    }

    let bodies = Arc::new(build_planets(epoch_mjd_tt).await);

    let mut guard = cache_cell()
        .await
        .write()
        .expect("planets cache lock poisoned");
    *guard = Some((Instant::now(), bodies.clone()));
    bodies
}

async fn build_planets(epoch_mjd_tt: f64) -> Vec<Body3D> {
    let kalman_context = crate::get_kalman_context().await;
    let jpl = &kalman_context.ephem_state.jpl;
    let epoch = Epoch::from_mjd_in_time_scale(epoch_mjd_tt, TimeScale::TT);

    let mut bodies = Vec::with_capacity(TRACKED_BODIES.len());
    for &(name, kind, naif_id) in TRACKED_BODIES {
        match jpl.body_ephemeris(naif_id, &epoch, EphemerisFrame::Ecliptic) {
            Ok((position, velocity)) => match keplerian_from_state(&position, &velocity, epoch_mjd_tt) {
                Some(elems) => bodies.push(Body3D {
                    name: name.to_string(),
                    kind,
                    position: [position.x, position.y, position.z],
                    orbit: geometry::ellipse_points(&elems, ORBIT_CURVE_SAMPLES),
                    elements: elems,
                }),
                None => tracing::warn!(
                    "orbit3d: {name}'s state at MJD-TT {epoch_mjd_tt} is not a closed ellipse, skipping"
                ),
            },
            Err(e) => tracing::warn!("orbit3d: failed to resolve {name}'s ephemeris: {e}"),
        }
    }
    bodies
}

/// Converts a heliocentric ecliptic `(position, velocity)` state into the
/// osculating [`Keplerian`] elements it implies.
///
/// # Arguments
///
/// * `position` — heliocentric position, AU, ecliptic mean J2000.
/// * `velocity` — heliocentric velocity, AU/day, ecliptic mean J2000.
/// * `epoch_mjd_tt` — the state's epoch, MJD-TT — tagged onto the result
///   unchanged, not reinterpreted (`OrbitalElements::from_orbital_state`'s
///   conversion math has no dependency on the epoch's own value or scale).
///
/// # Returns
///
/// `Some(elements)` for a closed ellipse (`0 <= e < 1`); `None` if the state
/// is parabolic or hyperbolic, which [`geometry`]'s functions do not model.
fn keplerian_from_state(
    position: &Vector3<f64>,
    velocity: &Vector3<f64>,
    epoch_mjd_tt: f64,
) -> Option<Keplerian> {
    let elems =
        OrbitalElements::from_orbital_state(position, velocity, epoch_mjd_tt).as_keplerian()?;
    Some(keplerian_from_outfit_elements(&elems, epoch_mjd_tt))
}

/// Converts a Kalman attributable state (topocentric ra/dec/rho and their
/// rates, plus the observer state it was measured against) into
/// heliocentric osculating [`Keplerian`] elements — the same conversion
/// `homepage::family::classify_from_attributable_state` does for the
/// dynamic-family classification.
///
/// # Arguments
///
/// * `state` — `(ra, dec, ra_dot, dec_dot, rho, rho_dot)`.
/// * `r_obs`, `v_obs` — the observer's heliocentric position (AU) and
///   velocity (AU/day).
/// * `epoch_mjd_tt` — the state's epoch, MJD-TT.
///
/// # Returns
///
/// `None` if the state does not resolve to a closed ellipse (`e >= 1`, or
/// non-finite), not expected for a real tracked object but not assumed away.
pub fn keplerian_from_attributable_state(
    state: &nalgebra::Vector6<f64>,
    r_obs: &Vector3<f64>,
    v_obs: &Vector3<f64>,
    epoch_mjd_tt: f64,
) -> Option<Keplerian> {
    use fink_fat_engine::topocentric_kf::conversion::attributable_to_cartesian;

    let cartesian = attributable_to_cartesian(state, r_obs, v_obs);
    keplerian_from_state(&cartesian.pos, &cartesian.vel, epoch_mjd_tt)
}

/// Converts an `outfit::KeplerianElements` (radians) into this module's
/// [`Keplerian`] (degrees) — the single conversion point shared by every
/// caller that already has `outfit`-derived elements in hand, so the
/// radians-to-degrees mapping has one definition.
///
/// # Arguments
///
/// * `elems` — elements as produced by `outfit::OrbitalElements
///   ::from_orbital_state(..).as_keplerian()`, angles in radians.
/// * `epoch_mjd_tt` — epoch to tag the result with, MJD-TT. Not read from
///   `elems.reference_epoch` because callers may want to override it (for
///   example a branch's own attributable-state epoch rather than whatever
///   value `from_orbital_state` was called with).
///
/// # Returns
///
/// The equivalent [`Keplerian`], angles in degrees.
pub fn keplerian_from_outfit_elements(
    elems: &outfit::KeplerianElements,
    epoch_mjd_tt: f64,
) -> Keplerian {
    Keplerian {
        epoch_mjd_tt,
        semi_major_axis_au: elems.semi_major_axis,
        eccentricity: elems.eccentricity,
        inclination_deg: elems.inclination.to_degrees(),
        ascending_node_longitude_deg: elems.ascending_node_longitude.to_degrees(),
        periapsis_argument_deg: elems.periapsis_argument.to_degrees(),
        mean_anomaly_deg: elems.mean_anomaly.to_degrees(),
    }
}
