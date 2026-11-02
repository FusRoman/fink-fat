use dioxus::prelude::*;

/// Polled by `BulkOrbitFitPage` while a bulk fit job is running.
#[server]
pub async fn get_bulk_orbit_fit_job_status(
    job_id: u64,
) -> Result<super::BulkOrbitFitJobView, ServerFnError> {
    let jobs = crate::get_bulk_orbit_fit_jobs().await;
    let jobs = jobs
        .lock()
        .map_err(|_| ServerFnError::new("bulk orbit fit job registry poisoned"))?;
    jobs.get(&job_id)
        .map(|job| job.view())
        .ok_or_else(|| ServerFnError::new(format!("no bulk orbit fit job with id {job_id}")))
}
