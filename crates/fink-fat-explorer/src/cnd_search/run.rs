//! Orchestration for a per-lineage CND check job: build one obs80 line per
//! observation, submit them to the CND API in sequential batches of at most
//! [`super::client::CND_BATCH_SIZE`], accumulate hits into the job registry
//! as each batch completes, and persist the finished attempt.
//!
//! The actual HTTP call is [`super::client::query_cnd_batch_resilient`] and
//! obs80 line construction is [`super::obs80::build_obs80_line`] — kept
//! apart from this module for the same reason `skybot_search::run` keeps its
//! network code separate: no dependency on this module's job registry/dioxus
//! plumbing.

use dioxus::prelude::*;

use super::CndQueryPoint;

/// Kicks off a CND check for every point in `points`, batched sequentially
/// (see [`super::client::CND_BATCH_SIZE`]). Returns immediately with a job
/// id; poll [`super::status::get_cnd_job_status`] for the growing hit list.
///
/// # Arguments
///
/// * `points` — one submission per observation of the lineage's best branch.
/// * `lineage_designation` — the lineage this check is for, used both to
///   record the finished attempt and (via `points`' shared `branch_id`) to
///   tag it with a branch.
/// * `time_separation_s`, `angle_separation_arcsec` — CND's match
///   thresholds; clamped into CND's own documented bounds (see
///   [`super::clamp_time_separation_s`]/[`super::clamp_angle_separation_arcsec`])
///   before use — an out-of-range value fails the request's own parameter
///   validation for *every* batch regardless of content, which the resilient
///   batching below can't distinguish from a genuinely bad observation and
///   would waste a very long time bisecting down to nothing.
///
/// # Return
///
/// The job id to poll, or an error if the job registry couldn't be updated.
#[server]
pub async fn start_cnd_search(
    points: Vec<CndQueryPoint>,
    lineage_designation: String,
    time_separation_s: f64,
    angle_separation_arcsec: f64,
) -> Result<u64, ServerFnError> {
    use crate::{get_cnd_jobs, NEXT_CND_JOB_ID};
    use std::sync::atomic::Ordering;

    let time_separation_s = super::clamp_time_separation_s(time_separation_s);
    let angle_separation_arcsec = super::clamp_angle_separation_arcsec(angle_separation_arcsec);

    let job_id = NEXT_CND_JOB_ID.fetch_add(1, Ordering::Relaxed);
    {
        let jobs = get_cnd_jobs().await;
        jobs.lock()
            .expect("CND job registry poisoned")
            .insert(job_id, super::CndJob::new(points.len()));
    }

    tokio::spawn(run_cnd_job(
        job_id,
        points,
        lineage_designation,
        time_separation_s,
        angle_separation_arcsec,
    ));

    Ok(job_id)
}

#[cfg(feature = "server")]
async fn push_log(job_id: u64, message: impl Into<String>) {
    let jobs = crate::get_cnd_jobs().await;
    if let Ok(mut jobs) = jobs.lock() {
        if let Some(job) = jobs.get_mut(&job_id) {
            job.logs.push(message.into());
        }
    }
}

/// Turns one [`CndQueryPoint`] + the CND response keyed by obs80 line into a
/// [`super::CndHit`], or `None` if that point had no match — see
/// [`super::CndHit`]'s doc comment on why absence, not a zero-match entry, is
/// how "no match" is represented. Reused directly by `crate::bulk_cnd::run`,
/// which needs to group hits by branch rather than just collect them flat
/// (see [`hits_from_batch`] for the flat-collection case this module uses).
#[cfg(feature = "server")]
pub(crate) fn hit_for_point(
    point: &CndQueryPoint,
    obs80_line: &str,
    results: &std::collections::HashMap<String, Vec<super::client::CndMatch>>,
) -> Option<super::CndHit> {
    let matches = results.get(obs80_line)?;
    if matches.is_empty() {
        return None;
    }
    let min_time_separation_s = matches
        .iter()
        .map(|m| m.time_separation_s.abs())
        .fold(f64::INFINITY, f64::min);
    let closest = matches
        .iter()
        .min_by(|a, b| {
            a.angle_separation_arcsec
                .total_cmp(&b.angle_separation_arcsec)
        })
        .expect("matches is non-empty");
    Some(super::CndHit {
        source_index: point.source_index,
        obs_id: point.obs_id,
        ra_deg: point.ra_deg,
        dec_deg: point.dec_deg,
        n_matches: matches.len(),
        min_time_separation_s,
        min_angle_separation_arcsec: closest.angle_separation_arcsec,
        closest_match_obs80: Some(closest.obs80.clone()),
    })
}

/// Turns one batch's [`CndQueryPoint`]s + the CND response into the
/// [`super::CndHit`]s for that batch, via [`hit_for_point`].
#[cfg(feature = "server")]
fn hits_from_batch(
    batch: &[(String, CndQueryPoint)],
    results: &std::collections::HashMap<String, Vec<super::client::CndMatch>>,
) -> Vec<super::CndHit> {
    batch
        .iter()
        .filter_map(|(obs80_line, point)| hit_for_point(point, obs80_line, results))
        .collect()
}

/// Runs every point of one job to completion, batched sequentially via
/// [`super::client::query_cnd_batch_resilient`], which bisects a failing
/// batch down to the individual line(s) actually causing trouble rather
/// than discarding up to [`super::client::CND_BATCH_SIZE`] good observations
/// over one bad one (confirmed against the live service: a single specific
/// placeholder designation reliably 500s the server for whatever batch
/// contains it — see that function's doc comment). Once every batch has
/// been tried, the accumulated hits (possibly none) are recorded in
/// `cnd_queries` via [`super::persist::insert_cnd_query`] before the job is
/// marked `Done`.
#[cfg(feature = "server")]
async fn run_cnd_job(
    job_id: u64,
    points: Vec<CndQueryPoint>,
    lineage_designation: String,
    time_separation_s: f64,
    angle_separation_arcsec: f64,
) {
    use super::client::{query_cnd_batch_resilient, CND_BATCH_SIZE};
    use super::obs80::build_obs80_line;

    let branch_id = points.first().map(|p| p.branch_id);

    push_log(
        job_id,
        format!(
            "Checking {} observation(s) against MPC's published observations \
             (time_separation_s={time_separation_s:.1}, angle_separation_arcsec={angle_separation_arcsec:.1})...",
            points.len()
        ),
    )
    .await;

    let client = crate::get_http_client().await;
    let mut skipped_lines = 0usize;

    // Cloned once so `query_cnd_batch_resilient` can advance it live as sub-
    // batches resolve (see its doc comment) — the same counter the client
    // polls via `get_cnd_job_status`, rather than only updating once per
    // whole top-level chunk, which could sit still for minutes during a slow
    // bisection and make a perfectly healthy job look frozen.
    let processed_counter = {
        let jobs = crate::get_cnd_jobs().await;
        jobs.lock()
            .ok()
            .and_then(|jobs| jobs.get(&job_id).map(|job| job.processed.clone()))
    };
    let Some(processed_counter) = processed_counter else {
        return;
    };

    for chunk in points.chunks(CND_BATCH_SIZE) {
        let batch: Vec<(String, CndQueryPoint)> = chunk
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
                (line, point.clone())
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
        let hits = hits_from_batch(&batch, &results);
        for message in &skipped {
            push_log(job_id, format!("Skipped one observation: {message}")).await;
        }
        skipped_lines += skipped.len();

        let jobs = crate::get_cnd_jobs().await;
        if let Ok(mut jobs) = jobs.lock() {
            if let Some(job) = jobs.get_mut(&job_id) {
                job.hits.extend(hits);
            }
        }
    }

    push_log(
        job_id,
        format!("Done ({skipped_lines} observation(s) skipped)."),
    )
    .await;

    let hits = {
        let jobs = crate::get_cnd_jobs().await;
        jobs.lock()
            .ok()
            .and_then(|jobs| jobs.get(&job_id).map(|job| job.hits.clone()))
            .unwrap_or_default()
    };

    let pool = crate::get_pool().await;
    if let Err(message) = super::persist::insert_cnd_query(
        pool,
        &lineage_designation,
        branch_id,
        time_separation_s,
        angle_separation_arcsec,
        &hits,
    )
    .await
    {
        push_log(job_id, format!("Failed to save this attempt: {message}")).await;
    }

    let jobs = crate::get_cnd_jobs().await;
    if let Ok(mut jobs) = jobs.lock() {
        if let Some(job) = jobs.get_mut(&job_id) {
            job.status = super::JobStatus::Done;
        }
    }
}
