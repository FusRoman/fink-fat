//! Turning observation rows read from Postgres into the corrected
//! `photom::ObsDataset` + `outfit::OutfitCache` pair every fit runs against.
//!
//! One construction path for both fits, because the details here are silent
//! when they go wrong: the `traj_id` column is what gives the dataset its
//! trajectory index, and without that index
//! `ModelCorrection::apply_batch_rms_correction` degrades to a documented
//! no-op — which is exactly how the single-lineage fit spent a while feeding
//! the solver uncorrected weights while the bulk fit used inflated ones.

use photom::io::polars::FromPolarsArgs;
use photom::observation_dataset::observation::Observation;
use photom::observation_dataset::ObsDataset;
use photom::observer::error_model::{ModelCorrection, ObsErrorModel};
use photom::TrajId;
use polars::df;

/// One observation as the fits read it out of the `observations` table.
///
/// `branch_id` doubles as the dataset's trajectory id: batch-RMS correction
/// groups strictly within a trajectory, so tagging each row with its branch is
/// what makes a branch's same-night observations share a batch — identically
/// whether the dataset holds one branch (single-lineage fit) or every eligible
/// branch at once (bulk fit).
pub struct FitObservation {
    pub id: i64,
    pub branch_id: i64,
    pub ra: f64,
    pub ra_err: f64,
    pub dec: f64,
    pub dec_err: f64,
    pub magnitude: f64,
    pub mag_err: f64,
    pub filter: i16,
    pub mjd_tt: f64,
    pub mpc_code_obs: String,
}

/// The trajectory id a branch's observations carry inside an [`ObsDataset`].
pub fn traj_id(branch_id: i64) -> TrajId {
    TrajId::from(branch_id as u32)
}

/// Builds the raw (uncorrected) dataset from `observations`.
///
/// `from_polars` resolves the observers from the `mpc_code_obs` column and
/// indexes the rows by `traj_id`; it leaves the row order untouched (its
/// default night-contiguous sort is skipped for want of a `night_id` column),
/// so callers that care about chronology sort via [`observations_of`].
///
/// # Returns
///
/// The dataset, with one trajectory per distinct `branch_id`.
///
/// # Errors
///
/// The DataFrame assembly or `photom`'s ingestion failing (unknown MPC
/// observatory code, schema mismatch), as a display string.
pub fn build_dataset(observations: &[FitObservation]) -> Result<ObsDataset, String> {
    let df = df!(
        "id" => observations.iter().map(|o| o.id as u64).collect::<Vec<_>>(),
        "ra" => observations.iter().map(|o| o.ra).collect::<Vec<_>>(),
        "ra_err" => observations.iter().map(|o| o.ra_err).collect::<Vec<_>>(),
        "dec" => observations.iter().map(|o| o.dec).collect::<Vec<_>>(),
        "dec_err" => observations.iter().map(|o| o.dec_err).collect::<Vec<_>>(),
        "magnitude" => observations.iter().map(|o| o.magnitude).collect::<Vec<_>>(),
        "mag_err" => observations.iter().map(|o| o.mag_err).collect::<Vec<_>>(),
        "filter" => observations.iter().map(|o| o.filter as u32).collect::<Vec<_>>(),
        "mjd_tt" => observations.iter().map(|o| o.mjd_tt).collect::<Vec<_>>(),
        "mpc_code_obs" => observations.iter().map(|o| o.mpc_code_obs.clone()).collect::<Vec<_>>(),
        "traj_id" => observations.iter().map(|o| o.branch_id as u32).collect::<Vec<_>>(),
    )
    .map_err(|e| e.to_string())?;

    ObsDataset::from_polars(&df, FromPolarsArgs::default()).map_err(|e| e.to_string())
}

/// Applies the astrometric error model to a raw dataset.
///
/// The three steps are one unit on purpose — `apply_model_errors` raises each
/// observation's σ to the model floor, then `apply_batch_rms_correction`
/// inflates σ by √n across each batch of observations the same observer took
/// within `gap_max` days — and the solver's weights are `1/σ²`, so skipping
/// or reordering a step quietly changes every fit result.
///
/// # Arguments
///
/// - `dataset` — a dataset from [`build_dataset`] (needs its trajectory index).
/// - `error_model` — from `params::to_error_model`.
/// - `gap_max` — batch window in days (`OrbitFitParams::gap_max`).
pub fn apply_error_model(
    dataset: ObsDataset,
    error_model: ObsErrorModel,
    gap_max: f64,
) -> ObsDataset {
    dataset
        .with_error_model(error_model)
        .apply_model_errors()
        .apply_batch_rms_correction(gap_max)
}

/// One branch's observations, chronologically sorted.
///
/// Sorting is not something the solver needs — `run_differential_correction`
/// derives its arc window by min/max and both IOD entry points sort
/// internally — but it keeps the per-observation residuals the page displays
/// and stores in a stable, readable order, which they otherwise are not: the
/// fit page hands its observation ids over from a `HashSet`.
pub fn observations_of(dataset: &ObsDataset, branch_id: i64) -> Vec<Observation> {
    let mut observations: Vec<Observation> = dataset
        .materialize_trajectory(traj_id(branch_id))
        .map(|m| m.collect_into_vec().into_iter().cloned().collect())
        .unwrap_or_default();
    observations.sort_by(|a, b| a.mjd_tt().total_cmp(&b.mjd_tt()));
    observations
}

/// Builds the observer-geometry cache the fit propagates against.
///
/// # Errors
///
/// Ephemeris or UT1 lookups failing for one of the observations, as a display
/// string.
pub fn build_cache(
    dataset: &ObsDataset,
    jpl: &outfit::JPLEphem,
    ut1_provider: &hifitime::ut1::Ut1Provider,
) -> Result<outfit::cache::OutfitCache, String> {
    outfit::cache::OutfitCache::build(dataset, jpl, ut1_provider, true).map_err(|e| e.to_string())
}
