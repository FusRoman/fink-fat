use dioxus::prelude::*;

/// Snapshot of the running/finished bulk CND job — see
/// [`super::run::start_bulk_cnd_check`].
#[server]
pub async fn get_bulk_cnd_job_status(job_id: u64) -> Result<super::BulkCndJobView, ServerFnError> {
    use crate::get_bulk_cnd_jobs;

    let jobs = get_bulk_cnd_jobs().await;
    let jobs = jobs
        .lock()
        .map_err(|_| ServerFnError::new("bulk CND job registry poisoned"))?;
    let job = jobs
        .get(&job_id)
        .ok_or_else(|| ServerFnError::new("unknown bulk CND job id"))?;
    Ok(job.view())
}
