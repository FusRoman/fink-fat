pub mod history;
pub mod latest;
pub mod run;
pub mod status;

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

/// All tunable parameters of an Outfit orbit fit, flattened out of
/// `outfit::IODParams` / `outfit::DifferentialCorrectionConfig` /
/// `ObsErrorModel` / `PropagatorKind` into one plain, serializable struct the
/// form can bind to and the server can convert back into the crate's own
/// types. The IOD/Gauss fields are kept (and shown in the form) for
/// completeness even though the fit always seeds from the current Kalman
/// orbit rather than running Gauss IOD — see `run::start_orbit_fit`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrbitFitParams {
    pub error_model: ObsErrorModelChoice,

    // --- IOD / Gauss (unused while seeding from the Kalman orbit) ---
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
}

impl Default for OrbitFitParams {
    fn default() -> Self {
        Self {
            error_model: ObsErrorModelChoice::default(),

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

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum JobStatus {
    Running,
    Done,
    Failed,
}

/// One Keplerian orbital element with its 1-sigma uncertainty (`None` when
/// the covariance wasn't propagated, e.g. the fit failed before producing
/// one).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KeplerianView {
    pub semi_major_axis_au: f64,
    pub eccentricity: f64,
    pub inclination_deg: f64,
    pub ascending_node_longitude_deg: f64,
    pub periapsis_argument_deg: f64,
    pub mean_anomaly_deg: f64,

    pub sigma_semi_major_axis_au: Option<f64>,
    pub sigma_eccentricity: Option<f64>,
    pub sigma_inclination_deg: Option<f64>,
    pub sigma_ascending_node_longitude_deg: Option<f64>,
    pub sigma_periapsis_argument_deg: Option<f64>,
    pub sigma_mean_anomaly_deg: Option<f64>,
}

/// Difference between two Keplerian orbits (new fit minus a reference
/// orbit), angles wrapped to (-180, 180] degrees.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrbitDelta {
    pub delta_semi_major_axis_au: f64,
    pub delta_eccentricity: f64,
    pub delta_inclination_deg: f64,
    pub delta_ascending_node_longitude_deg: f64,
    pub delta_periapsis_argument_deg: f64,
    pub delta_mean_anomaly_deg: f64,
    /// Reference epoch (MJD) of the orbit this delta was computed against.
    pub reference_epoch: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObsSelectionView {
    Kept,
    Rejected,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObsResidual {
    pub obs_id: i64,
    pub mjd_tt: f64,
    pub residual_ra_arcsec: f64,
    pub residual_dec_arcsec: f64,
    pub chi: f64,
    pub selection: ObsSelectionView,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrbitFitResult {
    pub reference_epoch: f64,
    pub keplerian: KeplerianView,

    pub delta_vs_kalman: Option<OrbitDelta>,
    pub delta_vs_previous_fit: Option<OrbitDelta>,

    pub normalised_rms: f64,
    pub reduced_chi2: f64,
    pub degrees_of_freedom: i64,
    pub total_newton_iterations: usize,
    pub num_measurements: usize,
    pub n_observations_used: usize,
    pub n_observations_rejected: usize,
    pub converged: bool,

    pub residuals: Vec<ObsResidual>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrbitFitJobView {
    pub status: JobStatus,
    pub logs: Vec<String>,
    pub error: Option<String>,
    pub result: Option<OrbitFitResult>,
}

/// Server-side job entry — not sent to the client directly, `OrbitFitJobView`
/// (a plain snapshot) is what `status::get_orbit_fit_job_status` returns.
#[cfg(feature = "server")]
#[derive(Clone, Debug)]
pub struct OrbitFitJob {
    pub status: JobStatus,
    pub logs: Vec<String>,
    pub error: Option<String>,
    pub result: Option<OrbitFitResult>,
}

#[cfg(feature = "server")]
impl OrbitFitJob {
    pub fn new() -> Self {
        Self {
            status: JobStatus::Running,
            logs: Vec::new(),
            error: None,
            result: None,
        }
    }

    pub fn view(&self) -> OrbitFitJobView {
        OrbitFitJobView {
            status: self.status,
            logs: self.logs.clone(),
            error: self.error.clone(),
            result: self.result.clone(),
        }
    }
}

/// Summary row for the fit history of a lineage — used both to render the
/// "History" tab and, server-side, to compute the "vs previous fit" delta.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrbitFitSummary {
    pub id: i64,
    pub fitted_at: String,
    pub n_observations_used: i32,
    pub normalised_rms: f64,
    pub reference_epoch: f64,
    pub semi_major_axis_au: f64,
    pub eccentricity_sin_lon: f64,
    pub eccentricity_cos_lon: f64,
    pub tan_half_incl_sin_node: f64,
    pub tan_half_incl_cos_node: f64,
    pub mean_longitude: f64,
    pub covariance: Vec<f64>,
}
