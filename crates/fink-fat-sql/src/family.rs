//! Dynamical-family classification of an orbit, stored in the
//! `dynamic_family` / `semi_major_axis` / `eccentricity` columns of
//! `kf_state`, `branch_best_state` and `archived_trajectories`.
//!
//! Pure functions only: [`DynamicalFamily::classify`] from `(a, e)`, and
//! [`classify_from_attributable_state`] from a topocentric attributable
//! state (via the engine's attributable → Cartesian conversion and
//! `outfit`'s Keplerian elements).

use nalgebra::{Vector3, Vector6};
use outfit::OrbitalElements;

use fink_fat_engine::topocentric_kf::conversion::attributable_to_cartesian;

/// Dynamical family of an asteroid, classified from its (semi-major axis,
/// eccentricity) based on the IMCCE SSP population table:
/// <https://ssp.imcce.fr/webservices/skybot/>
///
/// Variant order is deterministic and roughly follows increasing heliocentric
/// distance — it is what backs the `Ord` used to sort the lineages table by
/// family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DynamicalFamily {
    /// Unclassifiable: `a < 0.08` AU, or no bound Keplerian orbit.
    Unknown,
    /// `0.08 ≤ a < 0.21` AU.
    Vulcanoid,
    /// Near-Earth, `a < 1` AU and aphelion `Q < 0.983` AU (orbit inside Earth's).
    NeaAtira,
    /// Near-Earth, `a < 1` AU and `Q ≥ 0.983` AU.
    NeaAten,
    /// Near-Earth, `1 ≤ a < 2` AU and perihelion `q < 1.017` AU.
    NeaApollo,
    /// Near-Earth, `1 ≤ a < 2` AU and `1.017 ≤ q < 1.3` AU.
    NeaAmor,
    /// Mars-crosser, `1 ≤ a < 2` AU and `1.3 ≤ q ≤ 1.58` AU.
    MarsCrosserDeep,
    /// Mars-crosser, `1 ≤ a < 2` AU and `1.58 < q ≤ 1.666` AU.
    MarsCrosserShallow,
    /// `1 ≤ a < 2` AU and `q > 1.666` AU.
    Hungaria,
    /// Inner main belt, `2 ≤ a < 2.5` AU.
    MbInner,
    /// Middle main belt, `2.5 ≤ a < 2.82` AU.
    MbMiddle,
    /// Outer main belt, `2.82 ≤ a < 3.27` AU.
    MbOuter,
    /// Cybele group, `3.27 ≤ a < 3.7` AU.
    MbCybele,
    /// Hilda group, `3.7 ≤ a < 4.6` AU.
    MbHilda,
    /// Jupiter Trojan, `4.6 ≤ a < 5.5` AU.
    Trojan,
    /// `5.5 ≤ a < 30.1` AU.
    Centaur,
    /// Scattered-disk object, `30.1 ≤ a < 2000` AU and `q ≤ 36.3` AU.
    KboSdo,
    /// Detached KBO, `30.1 ≤ a < 2000` AU, `q > 36.3` AU and `e ≥ 0.24`.
    KboDetached,
    /// Inner classical KBO, `30.1 ≤ a < 39.4` AU, `q > 36.3` AU, `e < 0.24`.
    KboClassicalInner,
    /// Main classical KBO, `39.4 ≤ a < 47.8` AU, `q > 36.3` AU, `e < 0.24`.
    KboClassicalMain,
    /// Outer classical KBO, `47.8 ≤ a < 2000` AU, `q > 36.3` AU, `e < 0.24`.
    KboClassicalOuter,
    /// Inner Oort cloud, `a ≥ 2000` AU.
    Ioc,
}

impl DynamicalFamily {
    /// Classify from orbital elements, following the IMCCE SSP table.
    ///
    /// # Arguments
    ///
    /// * `a` — semi-major axis, in AU.
    /// * `e` — eccentricity.
    ///
    /// # Returns
    ///
    /// The family whose `(a, q = a(1 − e), Q = a(1 + e), e)` bounds contain
    /// the orbit (see each variant's doc).
    pub fn classify(a: f64, e: f64) -> Self {
        let perihelion = a * (1.0 - e);
        let aphelion = a * (1.0 + e);

        // Threshold for KBO>SDO: a(1−e) ≤ 30.1 * 2^(2/3) * (1−0.24) ≈ 36.3 AU
        const KBO_SDO_THRESHOLD: f64 = 30.1 * 1.587_401_05 * (1.0 - 0.24);

        match a {
            a if a < 0.08 => Self::Unknown,
            a if a < 0.21 => Self::Vulcanoid,
            a if a < 1.0 => {
                if aphelion < 0.983 {
                    Self::NeaAtira
                } else {
                    Self::NeaAten
                }
            }
            a if a < 2.0 => {
                if perihelion < 1.017 {
                    Self::NeaApollo
                } else if perihelion < 1.3 {
                    Self::NeaAmor
                } else if perihelion <= 1.58 {
                    Self::MarsCrosserDeep
                } else if perihelion <= 1.666 {
                    Self::MarsCrosserShallow
                } else {
                    Self::Hungaria
                }
            }
            a if a < 2.5 => Self::MbInner,
            a if a < 2.82 => Self::MbMiddle,
            a if a < 3.27 => Self::MbOuter,
            a if a < 3.7 => Self::MbCybele,
            a if a < 4.6 => Self::MbHilda,
            a if a < 5.5 => Self::Trojan,
            a if a < 30.1 => Self::Centaur,
            a if a < 2000.0 => {
                if perihelion <= KBO_SDO_THRESHOLD {
                    Self::KboSdo
                } else if e >= 0.24 {
                    Self::KboDetached
                } else if a < 39.4 {
                    Self::KboClassicalInner
                } else if a < 47.8 {
                    Self::KboClassicalMain
                } else {
                    Self::KboClassicalOuter
                }
            }
            _ => Self::Ioc,
        }
    }

    /// Human-readable label, as stored in the `dynamic_family` columns
    /// (e.g. `"NEA>Apollo"`, `"KBO>Classical>Main"`).
    ///
    /// # Returns
    ///
    /// A static string; `>` separates the hierarchy levels.
    pub fn label(self) -> &'static str {
        match self {
            Self::Unknown => "Unknown",
            Self::Vulcanoid => "Vulcanoid",
            Self::NeaAtira => "NEA>Atira",
            Self::NeaAten => "NEA>Aten",
            Self::NeaApollo => "NEA>Apollo",
            Self::NeaAmor => "NEA>Amor",
            Self::MarsCrosserDeep => "Mars-Crosser>Deep",
            Self::MarsCrosserShallow => "Mars-Crosser>Shallow",
            Self::Hungaria => "Hungaria",
            Self::MbInner => "MB>Inner",
            Self::MbMiddle => "MB>Middle",
            Self::MbOuter => "MB>Outer",
            Self::MbCybele => "MB>Cybele",
            Self::MbHilda => "MB>Hilda",
            Self::Trojan => "Trojan",
            Self::Centaur => "Centaur",
            Self::KboSdo => "KBO>SDO",
            Self::KboDetached => "KBO>Detached",
            Self::KboClassicalInner => "KBO>Classical>Inner",
            Self::KboClassicalMain => "KBO>Classical>Main",
            Self::KboClassicalOuter => "KBO>Classical>Outer",
            Self::Ioc => "IOC",
        }
    }
}

impl std::fmt::Display for DynamicalFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Classify a dynamical family directly from a raw attributable state (as
/// stored in `kf_state`/`archived_trajectories`), for callers that already
/// fetched these columns as part of a larger query — avoids a second
/// DB round-trip / duplicate join just to compute the family.
///
/// # Arguments
///
/// * `ra`, `dec`, `ra_dot`, `dec_dot`, `rho`, `rho_dot` — the topocentric
///   attributable state (equatorial J2000 angles in rad, rates in rad/day,
///   range in AU, range rate in AU/day).
/// * `epoch` — epoch of the state, MJD (TT).
/// * `r_obs`, `v_obs` — heliocentric observer position (AU) and velocity
///   (AU/day) at `epoch`, ecliptic J2000.
///
/// # Returns
///
/// `(family, semi_major_axis, eccentricity)`, or `None` when the state does
/// not convert to a bound (Keplerian) orbit.
#[allow(clippy::too_many_arguments)]
pub fn classify_from_attributable_state(
    ra: f64,
    dec: f64,
    ra_dot: f64,
    dec_dot: f64,
    rho: f64,
    rho_dot: f64,
    epoch: f64,
    r_obs: Vector3<f64>,
    v_obs: Vector3<f64>,
) -> Option<(DynamicalFamily, f64, f64)> {
    let state = Vector6::new(ra, dec, ra_dot, dec_dot, rho, rho_dot);
    let cartesian = attributable_to_cartesian(&state, &r_obs, &v_obs);

    let orbit = OrbitalElements::from_orbital_state(&cartesian.pos, &cartesian.vel, epoch)
        .as_keplerian()?;

    Some((
        DynamicalFamily::classify(orbit.semi_major_axis, orbit.eccentricity),
        orbit.semi_major_axis,
        orbit.eccentricity,
    ))
}
