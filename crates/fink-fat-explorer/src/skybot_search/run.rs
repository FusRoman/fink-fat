//! Orchestration for a Skybot conesearch job: fan out one HTTP request per
//! observation epoch, bounded to [`MAX_CONCURRENT_REQUESTS`] concurrent
//! connections via a semaphore, accumulating hits into the job registry as
//! they arrive and marking the job done once every point has been tried.
//!
//! The actual network call — URL construction, sending the request, parsing
//! the response — is [`super::parsing::fetch_conesearch_hits`]. It's kept
//! there rather than here specifically so it has no dependency on this
//! module's job registry/semaphore/dioxus plumbing, which lets it (and the
//! pure URL/parsing functions it calls) be exercised directly in tests,
//! including live ones against the real service — see the tests in
//! `parsing.rs`.

use dioxus::prelude::*;

#[cfg(feature = "server")]
use super::clamp_radius_arcsec;
use super::SkybotQueryPoint;

/// Skybot is a shared public service — this caps how many conesearch
/// requests fink-fat has in flight at once, regardless of how many
/// observation points a lineage has. `pub(crate)` so
/// `crate::bulk_skybot::run` shares this one ceiling instead of a second
/// copy of the same magic number — the bulk job is the exact same public
/// service, just with many more points to get through.
#[cfg(feature = "server")]
pub(crate) const MAX_CONCURRENT_REQUESTS: usize = 10;

/// Per-request timeout: long enough for a slow response, short enough that
/// one stalled cone can't hold its semaphore permit indefinitely.
/// `pub(crate)` for the same reason as [`MAX_CONCURRENT_REQUESTS`].
#[cfg(feature = "server")]
pub(crate) const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(20);

/// Kicks off a Skybot conesearch for every point in `points`, one HTTP
/// request each, at most 10 in flight at a time (see
/// `MAX_CONCURRENT_REQUESTS`). Returns immediately with a job id; poll
/// [`super::status::get_skybot_job_status`] for the growing hit list.
///
/// # Arguments
///
/// * `points` — one query per observation epoch of the lineage.
/// * `radius_arcsec` — requested conesearch radius; clamped into
///   [`super::MIN_RADIUS_ARCSEC`, `super::MAX_RADIUS_ARCSEC`].
/// * `lineage_designation` — the lineage this search is for, carried through
///   to [`run_skybot_job`] so each checked observation can be recorded in
///   `skybot_obs_status` against the right lineage.
///
/// # Return
///
/// The job id to poll, or an error if the job registry couldn't be updated.
#[server]
pub async fn start_skybot_search(
    points: Vec<SkybotQueryPoint>,
    radius_arcsec: f64,
    lineage_designation: String,
) -> Result<u64, ServerFnError> {
    use crate::{get_skybot_jobs, NEXT_SKYBOT_JOB_ID};
    use std::sync::atomic::Ordering;

    let radius_arcsec = clamp_radius_arcsec(radius_arcsec);

    let job_id = NEXT_SKYBOT_JOB_ID.fetch_add(1, Ordering::Relaxed);
    {
        let jobs = get_skybot_jobs().await;
        jobs.lock()
            .expect("skybot job registry poisoned")
            .insert(job_id, super::SkybotJob::new(points.len()));
    }

    tokio::spawn(run_skybot_job(
        job_id,
        points,
        radius_arcsec,
        lineage_designation,
    ));

    Ok(job_id)
}

#[cfg(feature = "server")]
async fn push_log(job_id: u64, message: impl Into<String>) {
    let jobs = crate::get_skybot_jobs().await;
    if let Ok(mut jobs) = jobs.lock() {
        if let Some(job) = jobs.get_mut(&job_id) {
            job.logs.push(message.into());
        }
    }
}

/// Runs every query point of one job to completion. A single point failing
/// (timeout, HTTP error, unparseable response) is logged and skipped rather
/// than failing the whole job — the points that already succeeded stay
/// visible to the client. Each point's result is upserted into
/// `skybot_obs_status` via [`super::persist::upsert_skybot_obs_status`] as
/// soon as it completes (not batched into one final write): this is the
/// same table `crate::bulk_skybot` writes to, one row per observation, so a
/// lineage search and a bulk sweep can never disagree about "was this
/// observation checked".
///
/// # Arguments
///
/// * `job_id` — the registry entry to update as points complete.
/// * `points` — the query points to search, one HTTP request each.
/// * `radius_arcsec` — the already-clamped conesearch radius.
/// * `lineage_designation` — the lineage each checked observation is
///   recorded against.
#[cfg(feature = "server")]
async fn run_skybot_job(
    job_id: u64,
    points: Vec<SkybotQueryPoint>,
    radius_arcsec: f64,
    lineage_designation: String,
) {
    use std::sync::Arc;
    use tokio::sync::Semaphore;

    push_log(
        job_id,
        format!(
            "Searching {} point(s) within a {radius_arcsec:.1}\" radius, up to \
             {MAX_CONCURRENT_REQUESTS} at a time...",
            points.len()
        ),
    )
    .await;

    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_REQUESTS));
    let client = crate::get_http_client().await;

    let handles: Vec<_> = points
        .into_iter()
        .map(|point| {
            let semaphore = semaphore.clone();
            let client = client.clone();
            tokio::spawn(async move {
                // Held for the request's whole lifetime; released on drop.
                let _permit = semaphore
                    .acquire_owned()
                    .await
                    .expect("skybot semaphore closed unexpectedly");
                let result = super::parsing::fetch_conesearch_hits(
                    &client,
                    &point,
                    radius_arcsec,
                    REQUEST_TIMEOUT,
                )
                .await;
                (point, result)
            })
        })
        .collect();

    let pool = crate::get_pool().await;
    let mut failed_points = 0usize;
    for handle in handles {
        let (point, hits) = match handle.await {
            Ok((point, Ok(hits))) => (point, hits),
            Ok((point, Err(message))) => {
                failed_points += 1;
                push_log(job_id, format!("A query point failed: {message}")).await;
                (point, Vec::new())
            }
            Err(join_error) => {
                failed_points += 1;
                push_log(job_id, format!("A query point task panicked: {join_error}")).await;
                continue; // no `point` to record against without it
            }
        };

        if let Err(message) = super::persist::upsert_skybot_obs_status(
            pool,
            point.obs_id,
            &lineage_designation,
            point.branch_id,
            radius_arcsec,
            &hits,
        )
        .await
        {
            push_log(
                job_id,
                format!("Failed to save observation {}: {message}", point.obs_id),
            )
            .await;
        }

        let jobs = crate::get_skybot_jobs().await;
        if let Ok(mut jobs) = jobs.lock() {
            if let Some(job) = jobs.get_mut(&job_id) {
                job.hits.extend(hits);
                job.processed
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    push_log(
        job_id,
        format!("Done ({failed_points} point(s) failed and were skipped)."),
    )
    .await;

    let jobs = crate::get_skybot_jobs().await;
    if let Ok(mut jobs) = jobs.lock() {
        if let Some(job) = jobs.get_mut(&job_id) {
            job.status = super::JobStatus::Done;
        }
    }
}
