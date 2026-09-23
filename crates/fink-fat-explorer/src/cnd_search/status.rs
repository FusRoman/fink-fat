use dioxus::prelude::*;

/// Snapshot of a running/finished CND job — see [`super::run::start_cnd_search`].
///
/// # Arguments
///
/// * `job_id` — the id returned by `start_cnd_search`.
///
/// # Return
///
/// The job's current [`super::CndJobView`], or an error if it can't be read
/// back from the registry.
#[server]
pub async fn get_cnd_job_status(job_id: u64) -> Result<super::CndJobView, ServerFnError> {
    use crate::get_cnd_jobs;

    let jobs = get_cnd_jobs().await;
    let jobs = jobs
        .lock()
        .map_err(|_| ServerFnError::new("CND job registry poisoned"))?;
    let job = jobs
        .get(&job_id)
        .ok_or_else(|| ServerFnError::new("unknown CND job id"))?;
    Ok(job.view())
}
