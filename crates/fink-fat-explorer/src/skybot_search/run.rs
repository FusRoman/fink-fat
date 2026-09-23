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
/// observation points a lineage has.
#[cfg(feature = "server")]
const MAX_CONCURRENT_REQUESTS: usize = 10;

/// Per-request timeout: long enough for a slow response, short enough that
/// one stalled cone can't hold its semaphore permit indefinitely.
#[cfg(feature = "server")]
const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(20);

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
///   to [`run_skybot_job`] so the finished attempt can be recorded in
///   `skybot_queries` against the right lineage.
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
/// visible to the client. Once every point has been tried, the accumulated
/// hits (possibly none) are recorded in `skybot_queries` via
/// [`super::persist::insert_skybot_query`] before the job is marked `Done`.
///
/// # Arguments
///
/// * `job_id` — the registry entry to update as points complete.
/// * `points` — the query points to search, one HTTP request each.
/// * `radius_arcsec` — the already-clamped conesearch radius.
/// * `lineage_designation` — the lineage to record the persisted attempt
///   against.
#[cfg(feature = "server")]
async fn run_skybot_job(
    job_id: u64,
    points: Vec<SkybotQueryPoint>,
    radius_arcsec: f64,
    lineage_designation: String,
) {
    use std::sync::atomic::Ordering;
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
                super::parsing::fetch_conesearch_hits(
                    &client,
                    &point,
                    radius_arcsec,
                    REQUEST_TIMEOUT,
                )
                .await
            })
        })
        .collect();

    let mut failed_points = 0usize;
    for handle in handles {
        let hits = match handle.await {
            Ok(Ok(hits)) => hits,
            Ok(Err(message)) => {
                failed_points += 1;
                push_log(job_id, format!("A query point failed: {message}")).await;
                Vec::new()
            }
            Err(join_error) => {
                failed_points += 1;
                push_log(job_id, format!("A query point task panicked: {join_error}")).await;
                Vec::new()
            }
        };

        let jobs = crate::get_skybot_jobs().await;
        if let Ok(mut jobs) = jobs.lock() {
            if let Some(job) = jobs.get_mut(&job_id) {
                job.hits.extend(hits);
                job.processed.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    push_log(
        job_id,
        format!("Done ({failed_points} point(s) failed and were skipped)."),
    )
    .await;

    // Snapshot the hits while holding the lock, then release it before the
    // `.await` on the DB write — matching this function's own convention
    // elsewhere of never holding the mutex across an await point.
    let hits = {
        let jobs = crate::get_skybot_jobs().await;
        jobs.lock()
            .ok()
            .and_then(|jobs| jobs.get(&job_id).map(|job| job.hits.clone()))
            .unwrap_or_default()
    };

    let pool = crate::get_pool().await;
    if let Err(message) =
        super::persist::insert_skybot_query(pool, &lineage_designation, radius_arcsec, &hits).await
    {
        push_log(job_id, format!("Failed to save this attempt: {message}")).await;
    }

    let jobs = crate::get_skybot_jobs().await;
    if let Ok(mut jobs) = jobs.lock() {
        if let Some(job) = jobs.get_mut(&job_id) {
            job.status = super::JobStatus::Done;
        }
    }
}
