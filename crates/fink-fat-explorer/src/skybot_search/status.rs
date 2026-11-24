use dioxus::prelude::*;

/// Polled by the lineage page while a Skybot search job is running.
#[server]
pub async fn get_skybot_job_status(job_id: u64) -> Result<super::SkybotJobView, ServerFnError> {
    let jobs = crate::get_skybot_jobs().await;
    let jobs = jobs
        .lock()
        .map_err(|_| ServerFnError::new("skybot job registry poisoned"))?;
    jobs.get(&job_id)
        .map(|job| job.view())
        .ok_or_else(|| ServerFnError::new(format!("no skybot job with id {job_id}")))
}
