//! Everything that describes *how* a fit should run: the form-facing parameter
//! types and their conversion into `outfit`'s own configuration structs.
//!
//! These types live here rather than next to either caller because both the
//! single-lineage fit (`crate::orbit_fit`) and the bulk fit
//! (`crate::bulk_orbit_fit`) must configure `outfit` identically — the two
//! paths have already drifted apart twice on details invisible from the form
//! (see `super::fit` for the latest one).

use serde::{Deserialize, Serialize};

/// Astrometric error model applied before the fit — mirrors
/// `photom::observer::error_model::ObsErrorModel`, kept as our own type here
/// so this module (and the wasm build of the form) doesn't need the
/// server-only `photom`/`outfit` dependencies just to describe the choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObsErrorModelChoice {
    Fcct14,
    Cbm10,
    Vfcc17,
    Lsst,
}

impl Default for ObsErrorModelChoice {
    fn default() -> Self {
        Self::Fcct14
    }
}

impl ObsErrorModelChoice {
    pub fn label(self) -> &'static str {
        match self {
            Self::Fcct14 => "FCCT14 (Farnocchia et al. 2014)",
            Self::Cbm10 => "CBM10 (Chesley, Baer & Monet 2010)",
            Self::Vfcc17 => "VFCC17 (Vereš et al. 2017)",
            Self::Lsst => "LSST (empirical Rubin/X05, Fink diaSource-derived)",
        }
    }
}

/// Dynamical model used to propagate the orbit during the fit. N-body is the
/// point of this feature (a real perturbed dynamical model, rather than the
/// two-body approximation the Kalman filter uses) so it's the default here,
/// even though the `outfit` crate itself defaults to two-body.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropagatorChoice {
    TwoBody,
    NBody,
}

impl Default for PropagatorChoice {
    fn default() -> Self {
        Self::NBody
    }
}

/// Planets that can be added as N-body perturbers. The Sun is always
/// included and isn't offered as a choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerturberChoice {
    Mercury,
    Venus,
    EarthMoon,
    Mars,
    Jupiter,
    Saturn,
    Uranus,
    Neptune,
    Pluto,
}

impl PerturberChoice {
    pub const ALL: [PerturberChoice; 9] = [
        Self::Mercury,
        Self::Venus,
        Self::EarthMoon,
        Self::Mars,
        Self::Jupiter,
        Self::Saturn,
        Self::Uranus,
        Self::Neptune,
        Self::Pluto,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Mercury => "Mercury",
            Self::Venus => "Venus",
            Self::EarthMoon => "Earth-Moon",
            Self::Mars => "Mars",
            Self::Jupiter => "Jupiter",
            Self::Saturn => "Saturn",
            Self::Uranus => "Uranus",
            Self::Neptune => "Neptune",
            Self::Pluto => "Pluto",
        }
    }
}

/// The 9 heaviest of the 300 main-belt asteroids in `outfit`'s ANISE
/// supplementary kernel (`codes_300ast_20100725.bsp`), by descending GM —
/// same ordering as `outfit::propagator::planet_gm::known_main_belt_asteroids_by_mass`.
/// Only resolvable with the ANISE ephemeris backend (`JPLEphem::with_main_belt_asteroids`),
/// which the whole app now uses. Kept as our own enum, mapping to a plain
/// asteroid number rather than an `outfit` type, for the same reason as
/// `ObsErrorModelChoice`/`PerturberChoice`: the wasm build of the form must
/// stay free of the server-only `outfit`/`photom` dependencies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AsteroidPerturberChoice {
    Ceres,
    Pallas,
    Juno,
    Vesta,
    Hygiea,
    Eunomia,
    Euphrosyne,
    Davida,
    Interamnia,
}

impl AsteroidPerturberChoice {
    pub const ALL: [AsteroidPerturberChoice; 9] = [
        Self::Ceres,
        Self::Pallas,
        Self::Juno,
        Self::Vesta,
        Self::Hygiea,
        Self::Eunomia,
        Self::Euphrosyne,
        Self::Davida,
        Self::Interamnia,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Ceres => "Ceres",
            Self::Pallas => "Pallas",
            Self::Juno => "Juno",
            Self::Vesta => "Vesta",
            Self::Hygiea => "Hygiea",
            Self::Eunomia => "Eunomia",
            Self::Euphrosyne => "Euphrosyne",
            Self::Davida => "Davida",
            Self::Interamnia => "Interamnia",
        }
    }

    /// The minor-planet number `outfit::jpl_ephem::naif::naif_ids::main_belt::AsteroidNumber`
    /// resolves through the main-belt supplementary kernel.
    pub fn asteroid_number(self) -> u32 {
        match self {
            Self::Ceres => 1,
            Self::Pallas => 2,
            Self::Juno => 3,
            Self::Vesta => 4,
            Self::Hygiea => 10,
            Self::Eunomia => 15,
            Self::Euphrosyne => 31,
            Self::Davida => 511,
            Self::Interamnia => 704,
        }
    }
}

/// Where the differential correction's starting orbit comes from.
///
/// The bulk fit (`crate::bulk_orbit_fit::run`) has always used
/// [`Self::SeedlessGaussIod`] — it fits every branch independently and has no
/// production orbit to seed from. The single-lineage fit defaulted to
/// [`Self::KalmanOrbit`] on the (reasonable) assumption that refining the
/// production estimate is usually what's wanted, but that means the two
/// paths can converge or diverge differently for the exact same branch and
/// observations: an N-body correction started from a two-body Kalman state
/// is a different (and not necessarily easier) basin of convergence than one
/// started from a fresh Gauss IOD solution. This option lets the
/// single-lineage page reproduce the bulk fit's strategy to tell the two
/// apart, or to unblock a fit that only diverges from the Kalman seed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeedStrategy {
    /// Seed from the lineage's current Kalman-derived orbit (default,
    /// historical behaviour).
    KalmanOrbit,
    /// Seed from a fresh Gauss IOD solution computed from the selected
    /// observations, ignoring the Kalman orbit entirely — same strategy the
    /// bulk fit uses.
    SeedlessGaussIod,
}

impl Default for SeedStrategy {
    fn default() -> Self {
        Self::KalmanOrbit
    }
}

impl SeedStrategy {
    pub fn label(self) -> &'static str {
        match self {
            Self::KalmanOrbit => "Kalman orbit (refine the production estimate)",
            Self::SeedlessGaussIod => "Gauss IOD, seedless (same as the bulk fit)",
        }
    }
}

/// All tunable parameters of an Outfit orbit fit, flattened out of
/// `outfit::IODParams` / `outfit::DifferentialCorrectionConfig` /
/// `ObsErrorModel` / `PropagatorKind` into one plain, serializable struct the
/// form can bind to and the server can convert back into the crate's own
/// types. The IOD/Gauss fields only take effect when `seed_strategy` is
/// [`SeedStrategy::SeedlessGaussIod`]; they're always shown in the form since
/// switching `seed_strategy` shouldn't reset them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrbitFitParams {
    pub error_model: ObsErrorModelChoice,
    /// `#[serde(default)]` so `fit_params` rows stored before this field
    /// existed (every fit run prior to this change) still deserialize — read
    /// back, for instance, by `latest::get_latest_fit_params` — as
    /// [`SeedStrategy::KalmanOrbit`], the behaviour they actually ran with.
    #[serde(default)]
    pub seed_strategy: SeedStrategy,

    // --- IOD / Gauss (only used when seed_strategy is SeedlessGaussIod) ---
    pub n_noise_realizations: usize,
    pub noise_scale: f64,
    pub extf: f64,
    pub dtmax: f64,
    pub dt_min: f64,
    pub dt_max_triplet: f64,
    pub optimal_interval_time: f64,
    pub max_obs_for_triplets: usize,
    pub max_triplets: u32,
    pub gap_max: f64,
    pub max_ecc: f64,
    pub max_perihelion_au: f64,
    pub min_rho2_au: f64,
    pub aberth_max_iter: u32,
    pub aberth_eps: f64,
    pub kepler_eps: f64,
    pub max_tested_solutions: usize,
    pub r2_min_au: f64,
    pub r2_max_au: f64,
    pub newton_eps: f64,
    pub newton_max_it: usize,
    pub root_imag_eps: f64,

    // --- Differential correction ---
    pub max_newton_iterations: usize,
    pub max_outlier_rejection_passes: usize,
    pub convergence_threshold: f64,
    pub convergence_before_rejection_threshold: f64,
    pub rms_stagnation_ratio: f64,
    pub rms_divergence_ratio: f64,
    pub max_stagnation_iterations: usize,
    pub enable_outlier_rejection: bool,

    // --- Dynamical model ---
    pub propagator: PropagatorChoice,
    pub perturbers: Vec<PerturberChoice>,
    pub asteroid_perturbers: Vec<AsteroidPerturberChoice>,
}

impl Default for OrbitFitParams {
    fn default() -> Self {
        Self {
            error_model: ObsErrorModelChoice::default(),
            seed_strategy: SeedStrategy::default(),

            n_noise_realizations: 20,
            noise_scale: 1.0,
            extf: -1.0,
            dtmax: 30.0,
            dt_min: 0.03,
            dt_max_triplet: 150.0,
            optimal_interval_time: 20.0,
            max_obs_for_triplets: 100,
            max_triplets: 10,
            gap_max: 8.0 / 24.0,
            max_ecc: 5.0,
            max_perihelion_au: 1.0e3,
            min_rho2_au: 0.01,
            aberth_max_iter: 50,
            aberth_eps: 1.0e-6,
            kepler_eps: 1e3 * f64::EPSILON,
            max_tested_solutions: 3,
            r2_min_au: 0.05,
            r2_max_au: 200.0,
            newton_eps: 1.0e-10,
            newton_max_it: 50,
            root_imag_eps: 1.0e-6,

            max_newton_iterations: 30,
            max_outlier_rejection_passes: 10,
            convergence_threshold: 1e-4,
            convergence_before_rejection_threshold: 2.0,
            rms_stagnation_ratio: 0.98,
            rms_divergence_ratio: 1.5,
            max_stagnation_iterations: 3,
            enable_outlier_rejection: true,

            propagator: PropagatorChoice::default(),
            perturbers: vec![PerturberChoice::Jupiter, PerturberChoice::Saturn],
            asteroid_perturbers: Vec::new(),
        }
    }
}

/// Minimum number of selected observations to attempt a fit — a differential
/// correction of 6 free elements needs at least 3 optical observations (6
/// scalar measurements).
pub const MIN_OBSERVATIONS: usize = 3;

/// Minimum time baseline (days) across the selected observations. Below
/// this, the arc is too short for the correction to reliably constrain all
/// six elements — `outfit` would likely reject it anyway, but this lets the
/// UI warn before the round trip.
pub const MIN_BASELINE_DAYS: f64 = 0.25;

/// Every branch the bulk orbit fit would attempt right now: at least
/// [`MIN_OBSERVATIONS`] observations spanning at least [`MIN_BASELINE_DAYS`].
/// Shared verbatim between `bulk_orbit_fit::run` (which fits these branches)
/// and the homepage snapshot (which needs the same set to tell a branch that
/// was never *eligible* for a fit apart from one that simply hasn't been
/// picked up by a bulk fit run yet) — a single copy of the predicate keeps
/// the two from drifting apart. Bind `$1 = MIN_OBSERVATIONS as i64`,
/// `$2 = MIN_BASELINE_DAYS`.
pub const ELIGIBLE_BRANCH_QUERY: &str = "
    SELECT bo.branch_id, b.lineage_designation
    FROM branch_observations bo
    JOIN branches b ON b.branch_id = bo.branch_id
    JOIN observations o ON o.id = bo.obs_id
    GROUP BY bo.branch_id, b.lineage_designation
    HAVING count(*) >= $1 AND (max(o.mjd_tt) - min(o.mjd_tt)) >= $2
";

/// Base seed of the per-branch RNG that drives the Gauss IOD's Monte-Carlo
/// noise realizations (`super::fit::branch_rng`).
///
/// Load-bearing for reproducibility, not a tuning knob: a branch fitted twice
/// with the same parameters only yields the same orbit because both runs draw
/// from `seed_from_u64(FIT_RNG_SEED ^ traj_id.stable_hash())`. Changing it
/// makes every past fit unreproducible, and letting the two fit paths use
/// different values would silently make the single-lineage fit disagree with
/// the bulk fit on the same branch.
pub const FIT_RNG_SEED: u64 = 42;

#[cfg(feature = "server")]
fn planetary_bary(
    choice: PerturberChoice,
) -> outfit::jpl_ephem::naif::naif_ids::planet_bary::PlanetaryBary {
    use outfit::jpl_ephem::naif::naif_ids::planet_bary::PlanetaryBary;
    match choice {
        PerturberChoice::Mercury => PlanetaryBary::Mercury,
        PerturberChoice::Venus => PlanetaryBary::Venus,
        PerturberChoice::EarthMoon => PlanetaryBary::EarthMoon,
        PerturberChoice::Mars => PlanetaryBary::Mars,
        PerturberChoice::Jupiter => PlanetaryBary::Jupiter,
        PerturberChoice::Saturn => PlanetaryBary::Saturn,
        PerturberChoice::Uranus => PlanetaryBary::Uranus,
        PerturberChoice::Neptune => PlanetaryBary::Neptune,
        PerturberChoice::Pluto => PlanetaryBary::Pluto,
    }
}

/// Maps the form's error-model choice onto `photom`'s own enum.
#[cfg(feature = "server")]
pub(crate) fn to_error_model(
    choice: ObsErrorModelChoice,
) -> photom::observer::error_model::ObsErrorModel {
    use photom::observer::error_model::ObsErrorModel;
    match choice {
        ObsErrorModelChoice::Fcct14 => ObsErrorModel::FCCT14,
        ObsErrorModelChoice::Cbm10 => ObsErrorModel::CBM10,
        ObsErrorModelChoice::Vfcc17 => ObsErrorModel::VFCC17,
        ObsErrorModelChoice::Lsst => ObsErrorModel::LSST,
    }
}

/// Builds the `outfit` differential-correction configuration (propagator,
/// perturbers, Newton/outlier-rejection tuning) from the form's flattened
/// `OrbitFitParams`.
#[cfg(feature = "server")]
pub(crate) fn build_dc_config(params: &OrbitFitParams) -> outfit::DifferentialCorrectionConfig {
    use outfit::differential_orbit_correction::OutlierRejectionConfig;
    use outfit::jpl_ephem::naif::naif_ids::main_belt::AsteroidNumber;
    use outfit::jpl_ephem::naif::naif_ids::{solar_system_bary::SolarSystemBary, NaifIds};
    use outfit::orbit_type::equinoctial_element::EquinoctialLimits;
    use outfit::propagator::{NBodyConfig, PropagatorKind};

    let mut perturbing_bodies = vec![NaifIds::SSB(SolarSystemBary::Sun)];
    perturbing_bodies.extend(
        params
            .perturbers
            .iter()
            .map(|p| NaifIds::PB(planetary_bary(*p))),
    );
    perturbing_bodies.extend(
        params
            .asteroid_perturbers
            .iter()
            .map(|a| NaifIds::AST(AsteroidNumber(a.asteroid_number()))),
    );
    let propagator = match params.propagator {
        PropagatorChoice::TwoBody => PropagatorKind::TwoBody,
        PropagatorChoice::NBody => PropagatorKind::NBody(NBodyConfig {
            perturbing_bodies,
            ..Default::default()
        }),
    };

    outfit::DifferentialCorrectionConfig {
        max_newton_iterations: params.max_newton_iterations,
        max_outlier_rejection_passes: params.max_outlier_rejection_passes,
        convergence_threshold: params.convergence_threshold,
        convergence_before_rejection_threshold: params.convergence_before_rejection_threshold,
        rms_stagnation_ratio: params.rms_stagnation_ratio,
        rms_divergence_ratio: params.rms_divergence_ratio,
        max_stagnation_iterations: params.max_stagnation_iterations,
        enable_outlier_rejection: params.enable_outlier_rejection,
        outlier_rejection_config: OutlierRejectionConfig::default(),
        orbital_limits: EquinoctialLimits::default(),
        free_elements: [true; 6],
        propagator,
    }
}

/// Builds the `outfit` IOD (Gauss) parameters from the form's flattened
/// `OrbitFitParams`. Only consulted when the fit runs seedless
/// (`SeedStrategy::SeedlessGaussIod`, always the case for the bulk fit).
#[cfg(feature = "server")]
pub(crate) fn build_iod_params(
    params: &OrbitFitParams,
) -> Result<outfit::initial_orbit_determination::IODParams, String> {
    use outfit::initial_orbit_determination::IODParams;

    IODParams::builder()
        .n_noise_realizations(params.n_noise_realizations)
        .noise_scale(params.noise_scale)
        .extf(params.extf)
        .dtmax(params.dtmax)
        .dt_min(params.dt_min)
        .dt_max_triplet(params.dt_max_triplet)
        .optimal_interval_time(params.optimal_interval_time)
        .max_obs_for_triplets(params.max_obs_for_triplets)
        .max_triplets(params.max_triplets)
        .gap_max(params.gap_max)
        .max_ecc(params.max_ecc)
        .max_perihelion_au(params.max_perihelion_au)
        .min_rho2_au(params.min_rho2_au)
        .aberth_max_iter(params.aberth_max_iter)
        .aberth_eps(params.aberth_eps)
        .kepler_eps(params.kepler_eps)
        .max_tested_solutions(params.max_tested_solutions)
        .r2_min_au(params.r2_min_au)
        .r2_max_au(params.r2_max_au)
        .newton_eps(params.newton_eps)
        .newton_max_it(params.newton_max_it)
        .root_imag_eps(params.root_imag_eps)
        .build()
        .map_err(|e| e.to_string())
}
