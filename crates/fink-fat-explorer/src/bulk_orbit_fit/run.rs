use dioxus::prelude::*;

use crate::fit_pipeline::params::OrbitFitParams;

/// How many rows one insert transaction carries, so a long insert can report
/// progress between chunks.
#[cfg(feature = "server")]
const INSERT_CHUNK: usize = 2000;

/// Kick off a bulk Outfit orbit fit over every eligible branch (>= 3
/// observations, >= `MIN_BASELINE_DAYS` baseline) in the database, run fully
/// independently per branch (Gauss IOD + differential correction, no Kalman
/// seed). Returns immediately with a job id; poll
/// `status::get_bulk_orbit_fit_job_status` for progress. Only one bulk fit can
/// run at a time.
#[server]
pub async fn start_bulk_orbit_fit(params: OrbitFitParams) -> Result<u64, ServerFnError> {
    use crate::{get_bulk_orbit_fit_jobs, NEXT_BULK_ORBIT_FIT_JOB_ID};
    use std::sync::atomic::Ordering;

    if crate::BULK_ORBIT_FIT_RUNNING.swap(true, Ordering::SeqCst) {
        return Err(ServerFnError::new(
            "a bulk orbit fit is already running; wait for it to finish before starting another",
        ));
    }

    let job_id = NEXT_BULK_ORBIT_FIT_JOB_ID.fetch_add(1, Ordering::Relaxed);
    {
        let jobs = get_bulk_orbit_fit_jobs().await;
        jobs.lock()
            .expect("bulk orbit fit job registry poisoned")
            .insert(job_id, super::BulkOrbitFitJob::new(0));
    }

    tokio::spawn(run_bulk_fit_job(job_id, params));

    Ok(job_id)
}

#[cfg(feature = "server")]
async fn push_log(job_id: u64, message: impl Into<String>) {
    let jobs = crate::get_bulk_orbit_fit_jobs().await;
    if let Ok(mut jobs) = jobs.lock() {
        if let Some(job) = jobs.get_mut(&job_id) {
            job.logs.push(message.into());
        }
    }
}

#[cfg(feature = "server")]
async fn run_bulk_fit_job(job_id: u64, params: OrbitFitParams) {
    let outcome = run_bulk_fit(job_id, &params).await;

    let jobs = crate::get_bulk_orbit_fit_jobs().await;
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

    crate::BULK_ORBIT_FIT_RUNNING.store(false, std::sync::atomic::Ordering::SeqCst);
}

#[cfg(feature = "server")]
#[derive(sqlx::FromRow)]
struct EligibleBranchRow {
    branch_id: i64,
    lineage_designation: String,
}

/// What happened to one branch's fit attempt: either a row ready for
/// `orbit_fits`, or just the `branch_id` to record in `orbit_fit_failures`.
/// Kept as one `par_iter().map(...)` output (rather than `filter_map`ing the
/// failures away) specifically so failures can still be persisted — that is
/// the only thing telling a branch whose fit failed apart from one that was
/// never submitted to a bulk fit.
#[cfg(feature = "server")]
enum FitOutcome {
    Success(crate::fit_pipeline::store::OrbitFitRow),
    Failure(i64),
}

/// Loads every eligible branch's observations in one query, as the pipeline's
/// row shape.
#[cfg(feature = "server")]
async fn load_observations(
    branch_ids: &[i64],
) -> Result<Vec<crate::fit_pipeline::dataset::FitObservation>, String> {
    use crate::fit_pipeline::dataset::FitObservation;

    #[derive(sqlx::FromRow)]
    struct Row {
        branch_id: i64,
        id: i64,
        ra: f64,
        ra_err: f64,
        dec: f64,
        dec_err: f64,
        magnitude: f64,
        mag_err: f64,
        filter: i16,
        mjd_tt: f64,
        mpc_code_obs: String,
    }

    let pool = crate::get_pool().await;
    let rows: Vec<Row> = sqlx::query_as(
        "SELECT bo.branch_id, o.id, o.ra, o.ra_err, o.dec, o.dec_err, o.magnitude, o.mag_err, \
         o.filter, o.mjd_tt, o.mpc_code_obs
         FROM branch_observations bo
         JOIN observations o ON o.id = bo.obs_id
         WHERE bo.branch_id = ANY($1)",
    )
    .bind(branch_ids)
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    Ok(rows
        .into_iter()
        .map(|r| FitObservation {
            id: r.id,
            branch_id: r.branch_id,
            ra: r.ra,
            ra_err: r.ra_err,
            dec: r.dec,
            dec_err: r.dec_err,
            magnitude: r.magnitude,
            mag_err: r.mag_err,
            filter: r.filter,
            mjd_tt: r.mjd_tt,
            mpc_code_obs: r.mpc_code_obs,
        })
        .collect())
}

#[cfg(feature = "server")]
async fn run_bulk_fit(job_id: u64, params: &OrbitFitParams) -> Result<(), String> {
    use crate::fit_pipeline::dataset;
    use crate::fit_pipeline::fit::{fit_seedless, Diagnostics};
    use crate::fit_pipeline::params::{
        build_dc_config, build_iod_params, to_error_model, ELIGIBLE_BRANCH_QUERY,
        MIN_BASELINE_DAYS, MIN_OBSERVATIONS,
    };
    use crate::fit_pipeline::store::{self, OrbitFitRow};
    use rayon::prelude::*;
    use std::collections::HashMap;
    use std::sync::atomic::Ordering;

    push_log(
        job_id,
        format!(
            "Finding eligible branches (>= {MIN_OBSERVATIONS} observations, >= \
             {MIN_BASELINE_DAYS} day baseline)..."
        ),
    )
    .await;

    let pool = crate::get_pool().await;
    let eligible: Vec<EligibleBranchRow> = sqlx::query_as(ELIGIBLE_BRANCH_QUERY)
        .bind(MIN_OBSERVATIONS as i64)
        .bind(MIN_BASELINE_DAYS)
        .fetch_all(pool)
        .await
        .map_err(|e| e.to_string())?;

    if eligible.is_empty() {
        return Err("no eligible trajectories found".to_string());
    }

    let lineage_by_branch: HashMap<i64, String> = eligible
        .iter()
        .map(|r| (r.branch_id, r.lineage_designation.clone()))
        .collect();
    let branch_ids: Vec<i64> = eligible.iter().map(|r| r.branch_id).collect();

    {
        let jobs = crate::get_bulk_orbit_fit_jobs().await;
        if let Ok(mut jobs) = jobs.lock() {
            if let Some(job) = jobs.get_mut(&job_id) {
                job.total = branch_ids.len();
            }
        }
    }
    push_log(job_id, format!("{} eligible branches.", branch_ids.len())).await;

    push_log(job_id, "Fetching their observations...").await;
    let observation_rows = load_observations(&branch_ids).await?;

    push_log(
        job_id,
        format!(
            "{} observations loaded; building the shared dataset...",
            observation_rows.len()
        ),
    )
    .await;

    let kalman_context = crate::get_kalman_context().await;
    let ephem = kalman_context.get_ephem();
    let jpl: &'static outfit::JPLEphem = &ephem.jpl;
    let ut1_provider: &'static hifitime::ut1::Ut1Provider = &ephem.ut1_provider;
    let error_model = to_error_model(params.error_model);
    let gap_max = params.gap_max;

    let mut stage_start = std::time::Instant::now();
    let (dataset, cache, dataset_secs, cache_build_secs) =
        tokio::task::spawn_blocking(move || -> Result<_, String> {
            let sub_start = std::time::Instant::now();
            let dataset = dataset::apply_error_model(
                dataset::build_dataset(&observation_rows)?,
                error_model,
                gap_max,
            );
            let dataset_secs = sub_start.elapsed().as_secs_f64();

            let sub_start = std::time::Instant::now();
            let cache = dataset::build_cache(&dataset, jpl, ut1_provider)?;
            let cache_build_secs = sub_start.elapsed().as_secs_f64();

            Ok((dataset, cache, dataset_secs, cache_build_secs))
        })
        .await
        .map_err(|e| e.to_string())??;
    push_log(
        job_id,
        format!(
            "ObsDataset built ({} observations) in {:.1}s.",
            dataset.observation_count(),
            dataset_secs
        ),
    )
    .await;
    push_log(
        job_id,
        format!(
            "Observer geometry cache built in {:.1}s (rayon-parallel if outfit's \
             \"parallel\" feature is enabled) — total stage time {:.1}s.",
            cache_build_secs,
            stage_start.elapsed().as_secs_f64()
        ),
    )
    .await;

    push_log(
        job_id,
        "Building IOD / differential-correction configuration...",
    )
    .await;
    let iod_params = build_iod_params(params)?;
    let dc_config = build_dc_config(params);

    let (processed_counter, succeeded_counter, failed_counter) = {
        let jobs = crate::get_bulk_orbit_fit_jobs().await;
        let jobs = jobs
            .lock()
            .map_err(|_| "bulk orbit fit job registry poisoned".to_string())?;
        let job = jobs
            .get(&job_id)
            .ok_or_else(|| "bulk orbit fit job vanished from the registry".to_string())?;
        (
            job.processed.clone(),
            job.succeeded.clone(),
            job.failed.clone(),
        )
    };

    let branch_lineages: Vec<(i64, String)> = branch_ids
        .iter()
        .map(|id| (*id, lineage_by_branch[id].clone()))
        .collect();

    push_log(
        job_id,
        format!(
            "Spawning the parallel fit: {} branches across {} rayon threads \
             (Gauss IOD + differential correction, no Kalman seed)...",
            branch_lineages.len(),
            rayon::current_num_threads()
        ),
    )
    .await;
    stage_start = std::time::Instant::now();

    let outcomes: Vec<FitOutcome> = {
        let processed_counter = processed_counter.clone();
        let succeeded_counter = succeeded_counter.clone();
        let failed_counter = failed_counter.clone();
        tokio::task::spawn_blocking(move || {
            branch_lineages
                .par_iter()
                .map(|(branch_id, lineage_designation)| {
                    let observations = dataset::observations_of(&dataset, *branch_id);
                    // Diagnostics are skipped here: recovering per-observation
                    // residuals costs a second correction pass per branch,
                    // which is affordable for one on-demand fit but not across
                    // every eligible branch.
                    let product = (!observations.is_empty())
                        .then(|| {
                            fit_seedless(
                                &observations,
                                &cache,
                                jpl,
                                &iod_params,
                                &dc_config,
                                *branch_id,
                                Diagnostics::Skip,
                            )
                            .ok()
                        })
                        .flatten();

                    processed_counter.fetch_add(1, Ordering::Relaxed);

                    match product {
                        Some(product) => {
                            succeeded_counter.fetch_add(1, Ordering::Relaxed);
                            FitOutcome::Success(bulk_row(
                                *branch_id,
                                lineage_designation.clone(),
                                &observations,
                                product,
                            ))
                        }
                        None => {
                            failed_counter.fetch_add(1, Ordering::Relaxed);
                            FitOutcome::Failure(*branch_id)
                        }
                    }
                })
                .collect()
        })
        .await
        .map_err(|e| e.to_string())?
    };

    let mut fit_rows: Vec<OrbitFitRow> = Vec::new();
    let mut failed_branch_ids: Vec<i64> = Vec::new();
    for outcome in outcomes {
        match outcome {
            FitOutcome::Success(row) => fit_rows.push(row),
            FitOutcome::Failure(branch_id) => failed_branch_ids.push(branch_id),
        }
    }

    push_log(
        job_id,
        format!(
            "Fitting done in {:.1}s: {} fitted, saving to the database...",
            stage_start.elapsed().as_secs_f64(),
            fit_rows.len()
        ),
    )
    .await;

    stage_start = std::time::Instant::now();
    for (chunk_index, chunk) in fit_rows.chunks(INSERT_CHUNK).enumerate() {
        store::insert_orbit_fits(pool, params, chunk).await?;
        let inserted = (chunk_index * INSERT_CHUNK + chunk.len()).min(fit_rows.len());
        if inserted < fit_rows.len() {
            push_log(
                job_id,
                format!(
                    "Inserted {}/{} fit rows so far...",
                    inserted,
                    fit_rows.len()
                ),
            )
            .await;
        }
    }
    push_log(
        job_id,
        format!(
            "{} rows inserted in {:.1}s.",
            fit_rows.len(),
            stage_start.elapsed().as_secs_f64()
        ),
    )
    .await;

    if !failed_branch_ids.is_empty() {
        store::record_fit_failures(pool, &failed_branch_ids).await?;
        push_log(
            job_id,
            format!(
                "{} failed attempts recorded in orbit_fit_failures.",
                failed_branch_ids.len()
            ),
        )
        .await;
    }

    push_log(
        job_id,
        format!(
            "Done: {} succeeded, {} failed, out of {} eligible branches.",
            succeeded_counter.load(Ordering::Relaxed),
            failed_counter.load(Ordering::Relaxed),
            branch_ids.len()
        ),
    )
    .await;

    Ok(())
}

/// Wraps one branch's fit product into a storable row.
///
/// The bulk job has no "previous fit" to compare against — that comparison is
/// the single-lineage page's, where the user is looking at one lineage's
/// history.
#[cfg(feature = "server")]
fn bulk_row(
    branch_id: i64,
    lineage_designation: String,
    observations: &[photom::observation_dataset::observation::Observation],
    product: crate::fit_pipeline::fit::FitProduct,
) -> crate::fit_pipeline::store::OrbitFitRow {
    crate::fit_pipeline::store::OrbitFitRow {
        lineage_designation,
        branch_id,
        observation_ids: observations.iter().map(|o| *o.id() as i64).collect(),
        delta_vs_previous_fit: None,
        product,
    }
}
