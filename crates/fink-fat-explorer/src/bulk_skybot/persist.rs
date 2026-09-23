//! All Postgres reads/writes for the bulk Skybot job — kept apart from
//! [`super::run`] (which owns the HTTP fan-out/semaphore/kill-watcher
//! plumbing) so the actual queries are testable in isolation, same
//! rationale as `skybot_search::persist`/`cnd_search::persist`.

use sqlx::PgPool;

use crate::skybot_search::SkybotHit;

use super::SkybotBulkJobStatus;

/// One row of the priority-ordered worklist a job processes — see
/// [`super::run`]'s doc comment for the query this comes from.
#[cfg_attr(feature = "server", derive(sqlx::FromRow))]
pub struct WorkItem {
    pub obs_id: i64,
    pub branch_id: i64,
    pub lineage_designation: String,
    pub ra: f64,
    pub dec: f64,
    pub mjd_tt: f64,
}

/// Every observation of a branch with a converged n-body fit, ordered
/// never-checked first, then oldest-checked first — the exact priority
/// [`crate::bulk_skybot::run::start_bulk_skybot_search`]'s doc comment
/// promises. `ORDER BY checked_at ASC NULLS FIRST` expresses both rules in
/// one clause: a `NULL` (never checked) sorts before any real timestamp,
/// and among real timestamps the oldest sorts first.
///
/// Duplicates `bulk_cnd::run::CONVERGED_BRANCH_QUERY`'s branch-selection CTE
/// verbatim rather than sharing it across the two bulk-job modules — small
/// enough (4 lines) that a cross-module dependency for it isn't worth it.
const WORKLIST_QUERY: &str = "
    WITH converged_branches AS (
        SELECT DISTINCT ON (branch_id) branch_id, lineage_designation
        FROM orbit_fits
        WHERE branch_id IS NOT NULL AND converged
        ORDER BY branch_id, fitted_at DESC
    )
    SELECT o.id AS obs_id, cb.branch_id, cb.lineage_designation, o.ra, o.dec, o.mjd_tt
    FROM branch_observations bo
    JOIN converged_branches cb ON cb.branch_id = bo.branch_id
    JOIN observations o ON o.id = bo.obs_id
    LEFT JOIN skybot_obs_status s ON s.obs_id = o.id
    ORDER BY s.checked_at ASC NULLS FIRST, o.id ASC
";

/// Fetches the full priority-ordered worklist (see [`WORKLIST_QUERY`]).
///
/// # Errors
///
/// The query failing, as a display string.
pub async fn fetch_worklist(pool: &PgPool) -> Result<Vec<WorkItem>, String> {
    sqlx::query_as(WORKLIST_QUERY)
        .fetch_all(pool)
        .await
        .map_err(|e| e.to_string())
}

/// Marks any job left `running` by a server process that exited before
/// reaching a terminal state as `interrupted` instead — otherwise that row
/// would trip the partial unique index on `status = 'running'` forever and
/// permanently block every future job. Idempotent; cheap to call more than
/// once (see [`super::run`]'s `OnceCell`-guarded wrapper, which only
/// actually calls this once per process).
///
/// # Errors
///
/// The query failing, as a display string.
pub async fn reconcile_stale_jobs(pool: &PgPool) -> Result<(), String> {
    sqlx::query(
        "UPDATE skybot_bulk_jobs SET status = 'interrupted', finished_at = now() \
         WHERE status = 'running'",
    )
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// Inserts a new `running` job row.
///
/// # Arguments
///
/// * `radius_arcsec` — the already-clamped conesearch radius this job uses.
/// * `total_observations` — the worklist's full length; see
///   `skybot_bulk_jobs.total_observations`'s column comment for why this is
///   the whole eligible population, not just what's left to check.
///
/// # Return
///
/// The new job's id.
///
/// # Errors
///
/// `"a bulk Skybot job is already running"` if the partial unique index on
/// `status = 'running'` rejects the insert (call
/// [`reconcile_stale_jobs`] first if a stale row might be the cause); any
/// other database error as a display string.
pub async fn insert_running_job(
    pool: &PgPool,
    radius_arcsec: f64,
    total_observations: i64,
) -> Result<i64, String> {
    let result = sqlx::query_scalar::<_, i64>(
        "INSERT INTO skybot_bulk_jobs (status, radius_arcsec, total_observations) \
         VALUES ('running', $1, $2) \
         RETURNING id",
    )
    .bind(radius_arcsec)
    .bind(total_observations)
    .fetch_one(pool)
    .await;

    match result {
        Ok(id) => Ok(id),
        Err(sqlx::Error::Database(e)) if e.is_unique_violation() => {
            Err("a bulk Skybot job is already running".to_string())
        }
        Err(e) => Err(e.to_string()),
    }
}

/// Records one observation's Skybot result and advances its job's progress
/// counters, in one transaction so a crash between the two can never leave
/// them out of sync. The write itself is
/// [`crate::skybot_search::persist::upsert_skybot_obs_status`] — the exact
/// same function the per-lineage search uses, run here against `&mut tx`
/// instead of a bare pool so it participates in this transaction — kept
/// there rather than duplicated here since `skybot_obs_status` is the one
/// table both flows share (see `skybot_search`'s module doc comment).
///
/// # Arguments
///
/// * `job_id` — whose `processed_observations`/`matched_observations` to
///   advance.
/// * `radius_arcsec` — the radius this particular check used.
/// * `hits` — this observation's Skybot matches, possibly empty.
///
/// # Errors
///
/// Either statement failing (including `hits` failing to serialize), as a
/// display string.
pub async fn record_observation_checked(
    pool: &PgPool,
    job_id: i64,
    obs_id: i64,
    lineage_designation: &str,
    branch_id: i64,
    radius_arcsec: f64,
    hits: &[SkybotHit],
) -> Result<(), String> {
    let matched = !hits.is_empty();

    let mut tx = pool.begin().await.map_err(|e| e.to_string())?;

    crate::skybot_search::persist::upsert_skybot_obs_status(
        &mut *tx,
        obs_id,
        lineage_designation,
        branch_id,
        radius_arcsec,
        hits,
    )
    .await?;

    sqlx::query(
        "UPDATE skybot_bulk_jobs \
         SET processed_observations = processed_observations + 1, \
             matched_observations = matched_observations + $1 \
         WHERE id = $2",
    )
    .bind(i64::from(matched))
    .bind(job_id)
    .execute(&mut *tx)
    .await
    .map_err(|e| e.to_string())?;

    tx.commit().await.map_err(|e| e.to_string())?;
    Ok(())
}

/// Appends one milestone line to a job's log — see `skybot_bulk_jobs.logs`'s
/// column comment for why this is called sparingly (milestones only, never
/// per observation).
///
/// # Errors
///
/// The query failing, as a display string.
pub async fn push_job_log(pool: &PgPool, job_id: i64, message: &str) -> Result<(), String> {
    sqlx::query("UPDATE skybot_bulk_jobs SET logs = array_append(logs, $1) WHERE id = $2")
        .bind(message)
        .bind(job_id)
        .execute(pool)
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Marks a job as having reached a terminal state.
///
/// # Errors
///
/// The query failing, as a display string.
pub async fn finish_job(
    pool: &PgPool,
    job_id: i64,
    status: SkybotBulkJobStatus,
    error: Option<&str>,
) -> Result<(), String> {
    sqlx::query(
        "UPDATE skybot_bulk_jobs SET status = $1, finished_at = now(), error = $2 WHERE id = $3",
    )
    .bind(status.as_column())
    .bind(error)
    .bind(job_id)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// Whether kill has been requested for a still-running job (see
/// [`super::status::request_kill_skybot_bulk_job`]) — polled by the
/// job's kill-watcher task, not by every worker task directly, to keep
/// per-observation work free of a database round-trip.
///
/// # Errors
///
/// The query failing, as a display string.
pub async fn is_kill_requested(pool: &PgPool, job_id: i64) -> Result<bool, String> {
    sqlx::query_scalar("SELECT kill_requested FROM skybot_bulk_jobs WHERE id = $1")
        .bind(job_id)
        .fetch_optional(pool)
        .await
        .map(|v| v.unwrap_or(false))
        .map_err(|e| e.to_string())
}

/// Sets `kill_requested = true` on the currently running job, if any.
///
/// # Errors
///
/// The query failing, as a display string.
pub async fn request_kill(pool: &PgPool) -> Result<(), String> {
    sqlx::query("UPDATE skybot_bulk_jobs SET kill_requested = true WHERE status = 'running'")
        .execute(pool)
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// The row [`WorkItem`]-adjacent queries read back for the client-facing
/// [`super::SkybotBulkJobView`].
#[cfg_attr(feature = "server", derive(sqlx::FromRow))]
pub struct JobRow {
    pub status: String,
    pub radius_arcsec: f64,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub finished_at: Option<chrono::DateTime<chrono::Utc>>,
    pub total_observations: i64,
    pub processed_observations: i64,
    pub matched_observations: i64,
    pub error: Option<String>,
    pub logs: Vec<String>,
}

/// Fetches the most recently started job, if any have ever run.
///
/// # Errors
///
/// The query failing, as a display string.
pub async fn fetch_current_job(pool: &PgPool) -> Result<Option<JobRow>, String> {
    sqlx::query_as("SELECT status, radius_arcsec, started_at, finished_at, total_observations, processed_observations, matched_observations, error, logs FROM skybot_bulk_jobs ORDER BY started_at DESC LIMIT 1")
        .fetch_optional(pool)
        .await
        .map_err(|e| e.to_string())
}
