//! The single-lineage orbit fit: its job plumbing, and the result types the
//! fit page renders.
//!
//! How a fit actually runs — parameters, dataset, solver, persistence — lives
//! in [`crate::fit_pipeline`], shared with the bulk fit.

pub mod history;
pub mod latest;
pub mod run;
pub mod status;

use serde::{Deserialize, Serialize};

use crate::fit_pipeline::fit::{FitMethod, KeplerianView, ObsResidual};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum JobStatus {
    Running,
    Done,
    Failed,
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrbitFitResult {
    /// The branch this fit ran on. `None` only for rows written before this
    /// column existed (the bulk fit has always populated it; the
    /// single-lineage fit started populating it once it began resolving an
    /// explicit branch — see `run::run_fit`).
    pub branch_id: Option<i64>,
    /// [`FitMethod::IodOnly`] means the differential correction diverged and
    /// this is the preliminary Gauss orbit — the page warns about it rather
    /// than presenting it as a least-squares solution.
    pub fit_method: FitMethod,
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
    /// See [`OrbitFitResult::branch_id`].
    pub branch_id: Option<i64>,
    /// See [`OrbitFitResult::fit_method`] — what makes an IOD-only row from a
    /// bulk run readable as such in the history table.
    pub fit_method: FitMethod,
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

/// Warns when the branch about to be fit differs from the branch the
/// lineage's most recent orbit fit (individual or bulk) actually used.
///
/// The single-lineage fit page resolves "the lineage's best branch" once,
/// up front (`lineage_page::observations_table::get_lineage_observations`),
/// but the bulk fit runs every branch of a lineage independently — so a past
/// bulk-fit success can silently belong to a different `branch_id` than the
/// one this page is currently set up to fit, producing a different
/// observation set and a different Kalman seed for what looks like "the same
/// lineage" to the user.
///
/// # Arguments
///
/// - `current_branch_id` — the branch this page is about to fit.
/// - `last_fitted_branch_id` — the branch of the lineage's most recent
///   `orbit_fits` row, if any (see [`OrbitFitResult::branch_id`]).
///
/// # Returns
///
/// `Some(message)` when the two are known and differ; `None` when they match
/// or when no previous fit (or no `branch_id` on it) is available to compare
/// against.
pub fn branch_mismatch_warning(
    current_branch_id: i64,
    last_fitted_branch_id: Option<i64>,
) -> Option<String> {
    match last_fitted_branch_id {
        Some(last) if last != current_branch_id => Some(format!(
            "The most recent fit for this lineage ran on branch #{last}; this page is \
             currently set up to fit branch #{current_branch_id} instead, so it may use \
             different observations and produce a different result."
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn branch_mismatch_warning_is_none_when_branches_match() {
        assert_eq!(branch_mismatch_warning(42, Some(42)), None);
    }

    #[test]
    fn branch_mismatch_warning_is_none_without_a_previous_fit() {
        assert_eq!(branch_mismatch_warning(42, None), None);
    }

    #[test]
    fn branch_mismatch_warning_fires_when_branches_differ() {
        let warning = branch_mismatch_warning(42, Some(7));
        assert!(warning.is_some());
        let message = warning.unwrap();
        assert!(message.contains("#7"));
        assert!(message.contains("#42"));
    }
}
