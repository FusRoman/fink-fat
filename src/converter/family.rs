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
    Unknown,
    Vulcanoid,
    NeaAtira,
    NeaAten,
    NeaApollo,
    NeaAmor,
    MarsCrosserDeep,
    MarsCrosserShallow,
    Hungaria,
    MbInner,
    MbMiddle,
    MbOuter,
    MbCybele,
    MbHilda,
    Trojan,
    Centaur,
    KboSdo,
    KboDetached,
    KboClassicalInner,
    KboClassicalMain,
    KboClassicalOuter,
    Ioc,
}

impl DynamicalFamily {
    /// Classify from orbital elements: a = semi-major axis (AU), e = eccentricity.
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
