//! Orchestration for the bulk Skybot job: fetch the priority-ordered
//! worklist (see [`super::persist::fetch_worklist`]), fan it out with the
//! same semaphore-bounded concurrency as the per-lineage job
//! (`crate::skybot_search::run::run_skybot_job`,
//! [`crate::skybot_search::run::MAX_CONCURRENT_REQUESTS`]), and record each
//! observation's result durably as it completes — unlike the per-lineage
//! job, which only writes its result to Postgres once, at the very end.
//! That per-observation durability is what makes a job spanning hours safe
//! to kill or lose to a server restart: nothing is only "remembered" in
//! process memory.

use dioxus::prelude::*;

#[cfg(feature = "server")]
use super::SkybotBulkJobStatus;

/// How often the kill-watcher task checks `skybot_bulk_jobs.kill_requested`
/// for the running job. A single watcher task does this polling (into a
/// shared in-memory flag every worker checks for free) rather than every
/// worker hitting the database directly — with potentially tens of
/// thousands of workers, that would turn a user's kill click into a
/// database-load spike for no benefit, when 2s of extra latency to notice
/// a kill is unnoticeable against a job that runs for hours.
#[cfg(feature = "server")]
const KILL_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);

#[cfg(feature = "server")]
static RECONCILED: tokio::sync::OnceCell<()> = tokio::sync::OnceCell::const_new();

/// Runs [`super::persist::reconcile_stale_jobs`] exactly once per process
/// (via the `OnceCell` guard — same lazy-init idiom as `crate::get_pool`),
/// so a job left `running` by a crashed/restarted server can never
/// permanently block new jobs via the partial unique index on
/// `status = 'running'`. Called at the top of every bulk-Skybot endpoint
/// ([`start_bulk_skybot_search`] and both fns in [`super::status`]) rather
/// than once at process startup, since this app has no async startup hook
/// to run it from.
#[cfg(feature = "server")]
pub(crate) async fn reconcile_stale_skybot_bulk_jobs() {
    RECONCILED
        .get_or_init(|| async {
            let pool = crate::get_pool().await;
            if let Err(message) = super::persist::reconcile_stale_jobs(pool).await {
                tracing::warn!("failed to reconcile stale bulk Skybot jobs: {message}");
            }
        })
        .await;
}

/// Starts a bulk Skybot check of every observation belonging to a branch
/// with a converged n-body fit, prioritized never-checked first, then
/// oldest-checked first (see [`super::persist::fetch_worklist`]'s doc
/// comment for the exact query). Returns as soon as the job is recorded and
/// spawned; poll [`super::status::get_current_skybot_bulk_job_status`] for
/// progress — there's no job id to hold onto, since at most one bulk
/// Skybot job exists at a time (enforced by `skybot_bulk_jobs`'s partial
/// unique index, not this function).
///
/// # Arguments
///
/// * `radius_arcsec` — conesearch radius; clamped via
///   [`crate::skybot_search::clamp_radius_arcsec`].
///
/// # Return
///
/// `Ok(())` once the job is running. `Err` if one is already running, no
/// branch is eligible, or the database is unreachable.
#[server]
pub async fn start_bulk_skybot_search(radius_arcsec: f64) -> Result<(), ServerFnError> {
    use crate::skybot_search::clamp_radius_arcsec;

    reconcile_stale_skybot_bulk_jobs().await;

    let radius_arcsec = clamp_radius_arcsec(radius_arcsec);
    let pool = crate::get_pool().await;

    let worklist = super::persist::fetch_worklist(pool)
        .await
        .map_err(ServerFnError::new)?;
    if worklist.is_empty() {
        return Err(ServerFnError::new(
            "no branch with a converged n-body fit found",
        ));
    }

    let job_id = super::persist::insert_running_job(pool, radius_arcsec, worklist.len() as i64)
        .await
        .map_err(ServerFnError::new)?;

    tokio::spawn(run_bulk_skybot_job(job_id, worklist, radius_arcsec));

    Ok(())
}

/// One worker task's outcome for one [`super::persist::WorkItem`]: either a
/// (possibly empty) hit list, or `None` if it was skipped because a kill had
/// already been requested by the time its semaphore permit came up.
#[cfg(feature = "server")]
type WorkerOutcome = (
    super::persist::WorkItem,
    Option<Result<Vec<crate::skybot_search::SkybotHit>, String>>,
);

/// Runs one [`super::persist::WorkItem`] once its semaphore permit is held:
/// skip if a kill was requested while waiting for the permit, otherwise
/// query Skybot for it.
///
/// # Arguments
///
/// * `kill_flag` — checked once, right after the permit is acquired, so a
///   kill requested while this task was queued behind the concurrency limit
///   is honored instead of firing one more request anyway.
#[cfg(feature = "server")]
async fn run_one_work_item(
    item: super::persist::WorkItem,
    client: reqwest::Client,
    radius_arcsec: f64,
    kill_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> WorkerOutcome {
    use crate::skybot_search::run::REQUEST_TIMEOUT;
    use crate::skybot_search::{parsing::fetch_conesearch_hits, SkybotQueryPoint};
    use std::sync::atomic::Ordering;

    if kill_flag.load(Ordering::Relaxed) {
        return (item, None);
    }

    let point = SkybotQueryPoint {
        source_index: 0, // unused in bulk: results are identified by `item.obs_id`, not a trajectory position.
        obs_id: item.obs_id,
        branch_id: item.branch_id,
        ra_deg: item.ra.to_degrees(),
        dec_deg: item.dec.to_degrees(),
        mjd_tt: item.mjd_tt,
    };
    let result = fetch_conesearch_hits(&client, &point, radius_arcsec, REQUEST_TIMEOUT).await;
    (item, Some(result))
}

/// Spawns a lightweight task that polls `kill_requested` for `job_id` every
/// [`KILL_POLL_INTERVAL`] and flips the returned flag once it sees `true`.
/// The caller is responsible for aborting the returned handle once the job
/// finishes on its own, so this doesn't poll forever.
#[cfg(feature = "server")]
fn spawn_kill_watcher(
    job_id: i64,
) -> (
    std::sync::Arc<std::sync::atomic::AtomicBool>,
    tokio::task::JoinHandle<()>,
) {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let kill_flag = Arc::new(AtomicBool::new(false));
    let watcher_flag = kill_flag.clone();
    let handle = tokio::spawn(async move {
        loop {
            tokio::time::sleep(KILL_POLL_INTERVAL).await;
            let pool = crate::get_pool().await;
            if super::persist::is_kill_requested(pool, job_id)
                .await
                .unwrap_or(false)
            {
                watcher_flag.store(true, Ordering::Relaxed);
                return;
            }
        }
    });
    (kill_flag, handle)
}

/// Runs `worklist` to completion (or until killed), recording each
/// observation as it's checked and the job's final status once done.
#[cfg(feature = "server")]
async fn run_bulk_skybot_job(
    job_id: i64,
    worklist: Vec<super::persist::WorkItem>,
    radius_arcsec: f64,
) {
    use crate::skybot_search::run::MAX_CONCURRENT_REQUESTS;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;
    use tokio::sync::Semaphore;

    let pool = crate::get_pool().await;
    let total = worklist.len();

    let _ = super::persist::push_job_log(
        pool,
        job_id,
        &format!(
            "Checking {total} observation(s) within a {radius_arcsec:.1}\" radius, up to \
             {MAX_CONCURRENT_REQUESTS} at a time..."
        ),
    )
    .await;

    let (kill_flag, watcher) = spawn_kill_watcher(job_id);
    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_REQUESTS));
    let client = crate::get_http_client().await;

    let handles: Vec<_> = worklist
        .into_iter()
        .map(|item| {
            let semaphore = semaphore.clone();
            let client = client.clone();
            let kill_flag = kill_flag.clone();
            tokio::spawn(async move {
                let _permit = semaphore
                    .acquire_owned()
                    .await
                    .expect("bulk skybot semaphore closed unexpectedly");
                run_one_work_item(item, client, radius_arcsec, kill_flag).await
            })
        })
        .collect();

    let mut failed = 0usize;
    let mut killed = false;
    for handle in handles {
        match handle.await {
            Ok((item, Some(Ok(hits)))) => {
                if let Err(message) = super::persist::record_observation_checked(
                    pool,
                    job_id,
                    item.obs_id,
                    &item.lineage_designation,
                    item.branch_id,
                    radius_arcsec,
                    &hits,
                )
                .await
                {
                    failed += 1;
                    tracing::warn!(
                        "bulk skybot: failed to record observation {}: {message}",
                        item.obs_id
                    );
                }
            }
            Ok((_, Some(Err(_)))) => failed += 1,
            Ok((_, None)) => killed = true,
            Err(_join_error) => failed += 1,
        }
        if kill_flag.load(Ordering::Relaxed) {
            killed = true;
        }
    }

    watcher.abort();

    let _ = super::persist::push_job_log(
        pool,
        job_id,
        &format!(
            "{} ({failed} observation(s) failed and will be retried on the next run)",
            if killed { "Killed" } else { "Done" }
        ),
    )
    .await;

    let final_status = if killed {
        SkybotBulkJobStatus::Killed
    } else {
        SkybotBulkJobStatus::Done
    };
    if let Err(message) = super::persist::finish_job(pool, job_id, final_status, None).await {
        tracing::warn!("bulk skybot: failed to finalize job {job_id}: {message}");
    }
}
