use dioxus::prelude::*;

#[cfg(feature = "server")]
use std::collections::HashMap;

/// Branches this bulk job submits to CND: every branch whose *latest* n-body
/// fit converged (`orbit_fits.converged`), one row per branch.
///
/// Deliberately not `fit_pipeline::params::ELIGIBLE_BRANCH_QUERY` (used by
/// `bulk_orbit_fit`) — that query answers "eligible *to be fit*" (enough
/// observations/baseline), a different question from "has a fit that
/// actually converged". Filtering on `orbit_fits.converged` directly matches
/// what a bulk CND check should submit: branches with a real orbit, not
/// every branch that was merely fittable.
#[cfg(feature = "server")]
const CONVERGED_BRANCH_QUERY: &str = "
    SELECT DISTINCT ON (branch_id) branch_id, lineage_designation
    FROM orbit_fits
    WHERE branch_id IS NOT NULL AND converged
    ORDER BY branch_id, fitted_at DESC
";

/// Kicks off a CND check of every branch with a converged n-body fit.
/// Returns immediately with a job id; poll
/// [`super::status::get_bulk_cnd_job_status`] for progress. Only one bulk CND
/// job runs at a time (independent from `bulk_orbit_fit`'s own single-run
/// guard — the two jobs don't contend for the same resource).
///
/// # Arguments
///
/// * `time_separation_s`, `angle_separation_arcsec` — CND's match
///   thresholds, applied to every branch submitted; clamped into CND's own
///   documented bounds (see
///   `cnd_search::clamp_time_separation_s`/`cnd_search::clamp_angle_separation_arcsec`)
///   before use, for the same reason `cnd_search::run::start_cnd_search`
///   clamps them — an out-of-range value fails every batch's request-level
///   validation regardless of content, which looks identical to a
///   genuinely bad observation to the resilient batching below and wastes a
///   very long time bisecting down to nothing trying to isolate it.
///
/// # Return
///
/// The job id to poll, or an error if a bulk CND job is already running.
#[server]
pub async fn start_bulk_cnd_check(
    time_separation_s: f64,
    angle_separation_arcsec: f64,
) -> Result<u64, ServerFnError> {
    use crate::cnd_search::{clamp_angle_separation_arcsec, clamp_time_separation_s};
    use crate::{get_bulk_cnd_jobs, NEXT_BULK_CND_JOB_ID};
    use std::sync::atomic::Ordering;

    if crate::BULK_CND_RUNNING.swap(true, Ordering::SeqCst) {
        return Err(ServerFnError::new(
            "a bulk CND check is already running; wait for it to finish before starting another",
        ));
    }

    let time_separation_s = clamp_time_separation_s(time_separation_s);
    let angle_separation_arcsec = clamp_angle_separation_arcsec(angle_separation_arcsec);

    let job_id = NEXT_BULK_CND_JOB_ID.fetch_add(1, Ordering::Relaxed);
    {
        let jobs = get_bulk_cnd_jobs().await;
        jobs.lock()
            .expect("bulk CND job registry poisoned")
            .insert(job_id, super::BulkCndJob::new());
    }

    tokio::spawn(run_bulk_cnd_job(
        job_id,
        time_separation_s,
        angle_separation_arcsec,
    ));

    Ok(job_id)
}

#[cfg(feature = "server")]
async fn push_log(job_id: u64, message: impl Into<String>) {
    let jobs = crate::get_bulk_cnd_jobs().await;
    if let Ok(mut jobs) = jobs.lock() {
        if let Some(job) = jobs.get_mut(&job_id) {
            job.logs.push(message.into());
        }
    }
}

#[cfg(feature = "server")]
async fn run_bulk_cnd_job(job_id: u64, time_separation_s: f64, angle_separation_arcsec: f64) {
    let outcome = run_bulk_cnd(job_id, time_separation_s, angle_separation_arcsec).await;

    let jobs = crate::get_bulk_cnd_jobs().await;
    if let Ok(mut jobs) = jobs.lock() {
        if let Some(job) = jobs.get_mut(&job_id) {
            match outcome {
                Ok(()) => job.status = super::JobStatus::Done,
                Err(message) => {
                    job.status = super::JobStatus::Failed;
                    job.error = Some(message);
                }
            }
        }
    }

    crate::BULK_CND_RUNNING.store(false, std::sync::atomic::Ordering::SeqCst);
}

#[cfg(feature = "server")]
#[derive(sqlx::FromRow)]
struct ConvergedBranchRow {
    branch_id: i64,
    lineage_designation: String,
}

#[cfg(feature = "server")]
#[derive(sqlx::FromRow)]
struct ObsRow {
    branch_id: i64,
    id: i64,
    ra: f64,
    dec: f64,
    mjd_tt: f64,
    magnitude: f64,
    filter: i16,
    mpc_code_obs: String,
}

/// Loads every observation of `branch_ids`, ordered by branch then track
/// position so [`build_query_points`] can derive a stable per-branch
/// `source_index` while it iterates.
#[cfg(feature = "server")]
async fn load_observations(branch_ids: &[i64]) -> Result<Vec<ObsRow>, String> {
    let pool = crate::get_pool().await;
    sqlx::query_as(
        "SELECT bo.branch_id, o.id, o.ra, o.dec, o.mjd_tt, o.magnitude, o.filter, o.mpc_code_obs \
         FROM branch_observations bo \
         JOIN observations o ON o.id = bo.obs_id \
         WHERE bo.branch_id = ANY($1) \
         ORDER BY bo.branch_id, bo.position",
    )
    .bind(branch_ids)
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())
}

/// Converts observation rows (grouped by branch via `load_observations`'s
/// `ORDER BY`) into [`crate::cnd_search::CndQueryPoint`]s, RA/Dec converted
/// to degrees (stored in radians in `observations`) and `source_index` reset
/// to 0 at the start of each branch — the same "position within this
/// branch's track" meaning `source_index` has in the per-lineage job.
#[cfg(feature = "server")]
fn build_query_points(rows: &[ObsRow]) -> Vec<crate::cnd_search::CndQueryPoint> {
    let mut next_index: HashMap<i64, usize> = HashMap::new();
    rows.iter()
        .map(|row| {
            let source_index = next_index.entry(row.branch_id).or_insert(0);
            let point = crate::cnd_search::CndQueryPoint {
                source_index: *source_index,
                obs_id: row.id,
                branch_id: row.branch_id,
                ra_deg: row.ra.to_degrees(),
                dec_deg: row.dec.to_degrees(),
                mjd_tt: row.mjd_tt,
                magnitude: row.magnitude,
                filter: row.filter,
                mpc_code_obs: row.mpc_code_obs.clone(),
            };
            *source_index += 1;
            point
        })
        .collect()
}

#[cfg(feature = "server")]
async fn run_bulk_cnd(
    job_id: u64,
    time_separation_s: f64,
    angle_separation_arcsec: f64,
) -> Result<(), String> {
    use crate::cnd_search::client::{query_cnd_batch_resilient, CND_BATCH_SIZE};
    use crate::cnd_search::obs80::build_obs80_line;
    use crate::cnd_search::persist::insert_cnd_query;
    use crate::cnd_search::run::hit_for_point;
    use crate::cnd_search::CndHit;
    use std::sync::atomic::Ordering;

    push_log(job_id, "Finding branches with a converged n-body fit...").await;

    let pool = crate::get_pool().await;
    let branches: Vec<ConvergedBranchRow> = sqlx::query_as(CONVERGED_BRANCH_QUERY)
        .fetch_all(pool)
        .await
        .map_err(|e| e.to_string())?;

    if branches.is_empty() {
        return Err("no branch with a converged n-body fit found".to_string());
    }

    let lineage_by_branch: HashMap<i64, String> = branches
        .iter()
        .map(|r| (r.branch_id, r.lineage_designation.clone()))
        .collect();
    let branch_ids: Vec<i64> = branches.iter().map(|r| r.branch_id).collect();

    push_log(job_id, format!("{} converged branches.", branch_ids.len())).await;
    push_log(job_id, "Fetching their observations...").await;
    let observation_rows = load_observations(&branch_ids).await?;
    let points = build_query_points(&observation_rows);

    {
        let jobs = crate::get_bulk_cnd_jobs().await;
        if let Ok(mut jobs) = jobs.lock() {
            if let Some(job) = jobs.get_mut(&job_id) {
                job.total_branches = branch_ids.len();
                job.total_observations = points.len();
            }
        }
    }
    push_log(
        job_id,
        format!(
            "{} observations; submitting in sequential batches of <= {CND_BATCH_SIZE}...",
            points.len()
        ),
    )
    .await;

    let client = crate::get_http_client().await;
    let mut hits_by_branch: HashMap<i64, Vec<CndHit>> = HashMap::new();
    let mut skipped_lines = 0usize;

    // Cloned once so `query_cnd_batch_resilient` can advance it live as
    // sub-batches resolve instead of only once per whole top-level chunk —
    // see its doc comment; the same rationale as `cnd_search::run`'s
    // per-lineage job.
    let processed_counter = {
        let jobs = crate::get_bulk_cnd_jobs().await;
        jobs.lock().ok().and_then(|jobs| {
            jobs.get(&job_id)
                .map(|job| job.processed_observations.clone())
        })
    };
    let Some(processed_counter) = processed_counter else {
        return Err("bulk CND job vanished from the registry".to_string());
    };

    for chunk in points.chunks(CND_BATCH_SIZE) {
        let batch: Vec<(String, &crate::cnd_search::CndQueryPoint)> = chunk
            .iter()
            .map(|point| {
                let line = build_obs80_line(
                    point.branch_id as u64,
                    point.ra_deg,
                    point.dec_deg,
                    point.mjd_tt,
                    point.magnitude,
                    point.filter,
                    &point.mpc_code_obs,
                );
                (line, point)
            })
            .collect();
        let obs80_lines: Vec<String> = batch.iter().map(|(line, _)| line.clone()).collect();

        let (results, skipped) = query_cnd_batch_resilient(
            client,
            &obs80_lines,
            time_separation_s,
            angle_separation_arcsec,
            &processed_counter,
        )
        .await;
        for (obs80_line, point) in &batch {
            if let Some(hit) = hit_for_point(point, obs80_line, &results) {
                hits_by_branch.entry(point.branch_id).or_default().push(hit);
            }
        }
        for message in &skipped {
            push_log(job_id, format!("Skipped one observation: {message}")).await;
        }
        skipped_lines += skipped.len();
    }

    push_log(
        job_id,
        format!(
            "Batches done ({skipped_lines} observation(s) skipped); saving one row per branch..."
        ),
    )
    .await;

    let mut branches_with_match = 0usize;
    for branch_id in &branch_ids {
        let hits = hits_by_branch.remove(branch_id).unwrap_or_default();
        if !hits.is_empty() {
            branches_with_match += 1;
        }
        let lineage_designation = &lineage_by_branch[branch_id];
        insert_cnd_query(
            pool,
            lineage_designation,
            Some(*branch_id),
            time_separation_s,
            angle_separation_arcsec,
            &hits,
        )
        .await?;
    }

    {
        let jobs = crate::get_bulk_cnd_jobs().await;
        if let Ok(mut jobs) = jobs.lock() {
            if let Some(job) = jobs.get_mut(&job_id) {
                job.branches_with_match
                    .store(branches_with_match, Ordering::Relaxed);
            }
        }
    }

    push_log(
        job_id,
        format!(
            "Done: {branches_with_match}/{} branches have at least one MPC near-duplicate.",
            branch_ids.len()
        ),
    )
    .await;

    Ok(())
}
