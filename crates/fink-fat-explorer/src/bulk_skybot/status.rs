use dioxus::prelude::*;

#[cfg(feature = "server")]
use super::SkybotBulkJobStatus;
use super::SkybotBulkJobView;

/// Converts a stored job row into the client-facing view.
///
/// # Errors
///
/// If `row.status` isn't a value [`SkybotBulkJobStatus::from_column`]
/// recognizes — this column is only ever written by this module, so that
/// would mean the schema and the enum have drifted apart.
#[cfg(feature = "server")]
fn to_view(row: super::persist::JobRow) -> Result<SkybotBulkJobView, ServerFnError> {
    let status = SkybotBulkJobStatus::from_column(&row.status).ok_or_else(|| {
        ServerFnError::new(format!("unknown bulk Skybot job status {:?}", row.status))
    })?;
    Ok(SkybotBulkJobView {
        status,
        radius_arcsec: row.radius_arcsec,
        started_at: row.started_at.to_rfc3339(),
        finished_at: row.finished_at.map(|t| t.to_rfc3339()),
        total_observations: row.total_observations,
        processed_observations: row.processed_observations,
        matched_observations: row.matched_observations,
        error: row.error,
        logs: row.logs,
    })
}

/// The current (if running) or most recently finished bulk Skybot job, if
/// any has ever run. Reads straight from `skybot_bulk_jobs` — there's no
/// in-memory registry to fall back on, by design (see
/// `crate::bulk_skybot`'s module doc comment), so this is accurate across
/// page revisits and server restarts alike.
///
/// # Return
///
/// `Ok(None)` if no bulk Skybot job has ever been started.
///
/// # Errors
///
/// The query failing, or a stored status value not being recognized.
#[server]
pub async fn get_current_skybot_bulk_job_status() -> Result<Option<SkybotBulkJobView>, ServerFnError>
{
    super::run::reconcile_stale_skybot_bulk_jobs().await;
    let pool = crate::get_pool().await;
    let row = super::persist::fetch_current_job(pool)
        .await
        .map_err(ServerFnError::new)?;
    row.map(to_view).transpose()
}

/// Requests that the currently running bulk Skybot job stop early. The job
/// notices within a few seconds (see `bulk_skybot::run::KILL_POLL_INTERVAL`),
/// finishes whatever requests are already in flight, and marks itself
/// `killed` — nothing is killed synchronously by this call.
///
/// # Return
///
/// `Ok(())` regardless of whether a job was actually running (matches
/// `UPDATE ... WHERE status = 'running'`'s own no-op-if-nothing-matches
/// behavior, rather than treating "nothing to kill" as an error).
///
/// # Errors
///
/// The query failing.
#[server]
pub async fn request_kill_skybot_bulk_job() -> Result<(), ServerFnError> {
    super::run::reconcile_stale_skybot_bulk_jobs().await;
    let pool = crate::get_pool().await;
    super::persist::request_kill(pool)
        .await
        .map_err(ServerFnError::new)
}
