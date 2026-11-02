use dioxus::prelude::*;

use super::OrbitFitJobView;

/// Snapshot of an orbit fit job started by `run::start_orbit_fit`. The
/// frontend polls this on an interval while the job is `Running`.
#[server]
pub async fn get_orbit_fit_job_status(job_id: u64) -> Result<OrbitFitJobView, ServerFnError> {
    use crate::get_orbit_fit_jobs;

    let jobs = get_orbit_fit_jobs().await;
    let jobs = jobs
        .lock()
        .map_err(|_| ServerFnError::new("orbit fit job registry poisoned"))?;

    jobs.get(&job_id)
        .map(|job| job.view())
        .ok_or_else(|| ServerFnError::new(format!("no orbit fit job with id {job_id}")))
}
