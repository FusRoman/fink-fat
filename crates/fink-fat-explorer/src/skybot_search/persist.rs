//! Writing checked observations to `skybot_obs_status` — the single source
//! of truth for Skybot results shared by the per-lineage search (this
//! module) and the bulk sweep (`crate::bulk_skybot`). Kept separate from
//! [`super::run`] for the same reason `cnd_search::persist` is kept apart
//! from `cnd_search::run`: the insert itself needs a real Postgres
//! connection and is worth testing/reusing in isolation from the job
//! registry/semaphore plumbing.

use sqlx::PgExecutor;

use super::SkybotHit;

/// Upserts one observation's Skybot result.
///
/// Generic over the executor (a plain `&PgPool` or `&mut Transaction`) so
/// `crate::bulk_skybot::persist::record_observation_checked` can run this
/// statement inside the same transaction as its job-progress-counter
/// update, while the per-lineage flow (which has no counters to keep in
/// sync) can just pass the pool directly.
///
/// Always upsert, even when `hits` is empty: an observation with no row in
/// `skybot_obs_status` has never been checked, which both the lineage
/// page's "last checked" display and the bulk job's resume-priority query
/// need to tell apart from "checked, no match".
///
/// # Arguments
///
/// * `executor` — the pool or transaction to run the upsert against.
/// * `obs_id` — the observation checked; `skybot_obs_status`'s primary key.
/// * `lineage_designation`, `branch_id` — denormalized onto the row so
///   downstream queries (the lineage page, the bulk priority query, the
///   cross-match dashboard) never need to join back to `branch_observations`.
/// * `radius_arcsec` — the conesearch radius this particular check used.
/// * `hits` — this observation's Skybot matches, possibly empty.
///
/// # Return
///
/// `Ok(())` once committed.
///
/// # Errors
///
/// `hits` failing to serialize, or the statement failing, as a display
/// string.
pub async fn upsert_skybot_obs_status<'e>(
    executor: impl PgExecutor<'e>,
    obs_id: i64,
    lineage_designation: &str,
    branch_id: i64,
    radius_arcsec: f64,
    hits: &[SkybotHit],
) -> Result<(), String> {
    let hits_json = serde_json::to_value(hits).map_err(|e| e.to_string())?;

    sqlx::query(
        "INSERT INTO skybot_obs_status \
         (obs_id, lineage_designation, branch_id, checked_at, radius_arcsec, hits) \
         VALUES ($1, $2, $3, now(), $4, $5) \
         ON CONFLICT (obs_id) DO UPDATE SET \
             lineage_designation = EXCLUDED.lineage_designation, \
             branch_id = EXCLUDED.branch_id, \
             checked_at = EXCLUDED.checked_at, \
             radius_arcsec = EXCLUDED.radius_arcsec, \
             hits = EXCLUDED.hits",
    )
    .bind(obs_id)
    .bind(lineage_designation)
    .bind(branch_id)
    .bind(radius_arcsec)
    .bind(hits_json)
    .execute(executor)
    .await
    .map_err(|e| e.to_string())?;

    Ok(())
}
