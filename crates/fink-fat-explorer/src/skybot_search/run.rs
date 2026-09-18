//! Orchestration for a Skybot conesearch job: fan out one HTTP request per
//! observation epoch, bounded to [`MAX_CONCURRENT_REQUESTS`] concurrent
//! connections via a semaphore, accumulating hits into the job registry as
//! they arrive and marking the job done once every point has been tried.
//!
//! Request URL construction and response parsing are pure functions in
//! [`super::parsing`] — this module only adds the network I/O and job-state
//! bookkeping around them.

use dioxus::prelude::*;

use super::SkybotQueryPoint;
#[cfg(feature = "server")]
use super::{clamp_radius_arcsec, SkybotHit};

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
#[server]
pub async fn start_skybot_search(
    points: Vec<SkybotQueryPoint>,
    radius_arcsec: f64,
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

    tokio::spawn(run_skybot_job(job_id, points, radius_arcsec));

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
/// visible to the client.
#[cfg(feature = "server")]
async fn run_skybot_job(job_id: u64, points: Vec<SkybotQueryPoint>, radius_arcsec: f64) {
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
                query_one_point(&client, point, radius_arcsec).await
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

    let jobs = crate::get_skybot_jobs().await;
    if let Ok(mut jobs) = jobs.lock() {
        if let Some(job) = jobs.get_mut(&job_id) {
            job.status = super::JobStatus::Done;
        }
    }
}

/// Queries Skybot for the objects near one point and returns its hits (empty
/// if none were found).
#[cfg(feature = "server")]
async fn query_one_point(
    client: &reqwest::Client,
    point: SkybotQueryPoint,
    radius_arcsec: f64,
) -> Result<Vec<SkybotHit>, String> {
    let url = super::parsing::conesearch_url(&point, radius_arcsec);

    let response = client
        .get(&url)
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| format!("Skybot request failed: {e}"))?;

    let body = response
        .text()
        .await
        .map_err(|e| format!("failed to read Skybot response: {e}"))?;

    super::parsing::parse_conesearch_response(&body, point.source_index)
}
