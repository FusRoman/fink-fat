use dioxus::prelude::*;

use crate::orbit_fit::OrbitFitParams;

/// Base seed for the deterministic per-branch RNGs — same role as
/// `lsst_cross_fink_fat_analysis::RNG_SEED`, kept local since no other module
/// needs it.
#[cfg(feature = "server")]
const RNG_SEED: u64 = 42;

/// Kick off a bulk Outfit orbit fit over every eligible branch (>= 3
/// observations, >= `MIN_BASELINE_DAYS` baseline) in the database, run fully
/// independently per branch (Gauss IOD + differential correction, no Kalman
/// seed) — unlike the single-lineage fit in `orbit_fit`. Returns immediately
/// with a job id; poll `status::get_bulk_orbit_fit_job_status` for progress.
/// Only one bulk fit can run at a time.
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

#[cfg(feature = "server")]
#[derive(sqlx::FromRow)]
struct BulkObsRow {
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

/// One fitted branch, ready to be inserted as a row of `orbit_fits`.
#[cfg(feature = "server")]
struct BulkFitRow {
    branch_id: i64,
    lineage_designation: String,
    observation_ids: Vec<i64>,
    n_observations_used: i32,
    fit_method: &'static str,
    reference_epoch: f64,
    semi_major_axis: f64,
    eccentricity_sin_lon: f64,
    eccentricity_cos_lon: f64,
    tan_half_incl_sin_node: f64,
    tan_half_incl_cos_node: f64,
    mean_longitude: f64,
    covariance: Vec<f64>,
    normalised_rms: f64,
    num_measurements: i32,
    converged: bool,
    keplerian_json: serde_json::Value,
}

/// Converts one branch's `outfit::FitOrbitResult` into a `BulkFitRow`.
/// `Err` means the orbit couldn't be converted to either representation —
/// treated the same as a fit failure (not inserted, counted as `failed`).
///
/// Unlike `orbit_fit::run::run_fit`'s `DifferentialCorrectionOutput`, the
/// seedless `differential_correction` wrapper used here only returns the
/// final orbital elements and a quality scalar — not the per-iteration
/// Newton count or the final observation selection — so
/// `total_newton_iterations` isn't tracked (stored as 0) and
/// `n_observations_used`/`num_measurements` reflect every observation
/// passed in, not the post-outlier-rejection subset.
#[cfg(feature = "server")]
fn build_row(
    branch_id: i64,
    lineage_designation: String,
    observations: &[photom::observation_dataset::observation::Observation],
    fit: &outfit::constants::FitOrbitResult,
) -> Result<BulkFitRow, String> {
    use outfit::constants::FitOrbitResult;
    use outfit::OrbitalElements;

    let fit_method = match fit {
        FitOrbitResult::DifferentialCorrection(_) => "differential_correction",
        FitOrbitResult::IODGauss(_) => "iod_only",
    };
    let normalised_rms = fit.orbit_quality();
    let elements = fit.orbital_elements();

    let (equinoctial, covariance) = match elements.to_equinoctial().map_err(|e| e.to_string())? {
        OrbitalElements::Equinoctial {
            elements,
            covariance,
            ..
        } => (
            elements,
            covariance
                .map(|c| c.matrix.iter().copied().collect::<Vec<f64>>())
                .unwrap_or_default(),
        ),
        _ => return Err("expected equinoctial orbital elements after conversion".to_string()),
    };

    let keplerian = elements.to_keplerian().ok().and_then(|oe| match oe {
        OrbitalElements::Keplerian {
            elements,
            uncertainty,
            ..
        } => Some(crate::orbit_fit::run::keplerian_view(
            &elements,
            uncertainty.as_ref(),
        )),
        _ => None,
    });
    let keplerian_json = keplerian
        .and_then(|view| serde_json::to_value(view).ok())
        .unwrap_or(serde_json::Value::Null);

    let converged = fit_method == "differential_correction"
        && normalised_rms.is_finite()
        && normalised_rms < 10.0;

    Ok(BulkFitRow {
        branch_id,
        lineage_designation,
        observation_ids: observations.iter().map(|o| *o.id() as i64).collect(),
        n_observations_used: observations.len() as i32,
        fit_method,
        reference_epoch: equinoctial.reference_epoch,
        semi_major_axis: equinoctial.semi_major_axis,
        eccentricity_sin_lon: equinoctial.eccentricity_sin_lon,
        eccentricity_cos_lon: equinoctial.eccentricity_cos_lon,
        tan_half_incl_sin_node: equinoctial.tan_half_incl_sin_node,
        tan_half_incl_cos_node: equinoctial.tan_half_incl_cos_node,
        mean_longitude: equinoctial.mean_longitude,
        covariance,
        normalised_rms,
        num_measurements: (observations.len() * 2) as i32,
        converged,
        keplerian_json,
    })
}

#[cfg(feature = "server")]
async fn run_bulk_fit(job_id: u64, params: &OrbitFitParams) -> Result<(), String> {
    use crate::orbit_fit::run::{build_dc_config, build_iod_params, to_error_model};
    use crate::orbit_fit::{MIN_BASELINE_DAYS, MIN_OBSERVATIONS};
    use outfit::cache::OutfitCache;
    use outfit::differential_orbit_correction::differential_correction;
    use photom::io::polars::FromPolarsArgs;
    use photom::observation_dataset::ObsDataset;
    use photom::observer::error_model::ModelCorrection;
    use photom::TrajId;
    use polars::df;
    use rand::{rngs::SmallRng, SeedableRng};
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
    let eligible: Vec<EligibleBranchRow> = sqlx::query_as(
        "SELECT bo.branch_id, b.lineage_designation
         FROM branch_observations bo
         JOIN branches b ON b.branch_id = bo.branch_id
         JOIN observations o ON o.id = bo.obs_id
         GROUP BY bo.branch_id, b.lineage_designation
         HAVING count(*) >= $1 AND (max(o.mjd_tt) - min(o.mjd_tt)) >= $2",
    )
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
    let rows: Vec<BulkObsRow> = sqlx::query_as(
        "SELECT bo.branch_id, o.id, o.ra, o.ra_err, o.dec, o.dec_err, o.magnitude, o.mag_err, \
         o.filter, o.mjd_tt, o.mpc_code_obs
         FROM branch_observations bo
         JOIN observations o ON o.id = bo.obs_id
         WHERE bo.branch_id = ANY($1)",
    )
    .bind(&branch_ids)
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    push_log(
        job_id,
        format!(
            "{} observations loaded; building the shared dataset...",
            rows.len()
        ),
    )
    .await;

    let ids: Vec<u64> = rows.iter().map(|r| r.id as u64).collect();
    let ras: Vec<f64> = rows.iter().map(|r| r.ra).collect();
    let ra_errs: Vec<f64> = rows.iter().map(|r| r.ra_err).collect();
    let decs: Vec<f64> = rows.iter().map(|r| r.dec).collect();
    let dec_errs: Vec<f64> = rows.iter().map(|r| r.dec_err).collect();
    let magnitudes: Vec<f64> = rows.iter().map(|r| r.magnitude).collect();
    let mag_errs: Vec<f64> = rows.iter().map(|r| r.mag_err).collect();
    let filters: Vec<u32> = rows.iter().map(|r| r.filter as u32).collect();
    let mjd_tts: Vec<f64> = rows.iter().map(|r| r.mjd_tt).collect();
    let mpc_codes: Vec<String> = rows.iter().map(|r| r.mpc_code_obs.clone()).collect();
    let traj_ids: Vec<u32> = rows.iter().map(|r| r.branch_id as u32).collect();
    drop(rows);

    let mut stage_start = std::time::Instant::now();
    let df = df!(
        "id" => ids,
        "ra" => ras,
        "ra_err" => ra_errs,
        "dec" => decs,
        "dec_err" => dec_errs,
        "magnitude" => magnitudes,
        "mag_err" => mag_errs,
        "filter" => filters,
        "mjd_tt" => mjd_tts,
        "mpc_code_obs" => mpc_codes,
        "traj_id" => traj_ids,
    )
    .map_err(|e| e.to_string())?;
    push_log(
        job_id,
        format!(
            "DataFrame assembled ({} rows) in {:.1}s.",
            df.height(),
            stage_start.elapsed().as_secs_f64()
        ),
    )
    .await;

    push_log(
        job_id,
        "Resolving observers and building the ObsDataset (from_polars)...",
    )
    .await;
    let kalman_context = crate::get_kalman_context().await;
    let ephem = kalman_context.get_ephem();
    let jpl: &'static outfit::JPLEphem = &ephem.jpl;
    let ut1_provider: &'static hifitime::ut1::Ut1Provider = &ephem.ut1_provider;
    let error_model = to_error_model(params.error_model);
    let gap_max = params.gap_max;

    stage_start = std::time::Instant::now();
    let (dataset, cache, from_polars_secs, cache_build_secs) =
        tokio::task::spawn_blocking(move || -> Result<_, String> {
            let sub_start = std::time::Instant::now();
            let dataset = ObsDataset::from_polars(&df, FromPolarsArgs::default())
                .map_err(|e| e.to_string())?;
            let dataset = dataset
                .with_error_model(error_model)
                .apply_model_errors()
                .apply_batch_rms_correction(gap_max);
            let from_polars_secs = sub_start.elapsed().as_secs_f64();

            let sub_start = std::time::Instant::now();
            let cache =
                OutfitCache::build(&dataset, jpl, ut1_provider, true).map_err(|e| e.to_string())?;
            let cache_build_secs = sub_start.elapsed().as_secs_f64();

            Ok((dataset, cache, from_polars_secs, cache_build_secs))
        })
        .await
        .map_err(|e| e.to_string())??;
    push_log(
        job_id,
        format!(
            "ObsDataset built ({} observations) in {:.1}s.",
            dataset.observation_count(),
            from_polars_secs
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

    let fit_rows: Vec<BulkFitRow> = {
        let processed_counter = processed_counter.clone();
        let succeeded_counter = succeeded_counter.clone();
        let failed_counter = failed_counter.clone();
        tokio::task::spawn_blocking(move || {
            branch_lineages
                .par_iter()
                .filter_map(|(branch_id, lineage_designation)| {
                    let traj_id = TrajId::from(*branch_id as u32);
                    let mut rng = SmallRng::seed_from_u64(RNG_SEED ^ traj_id.stable_hash());

                    let outcome = dataset.materialize_trajectory(traj_id.clone()).map(|m| {
                        let mut observations: Vec<_> =
                            m.collect_into_vec().into_iter().cloned().collect();
                        observations.sort_by(|a, b| a.mjd_tt().total_cmp(&b.mjd_tt()));
                        let result = differential_correction(
                            &observations,
                            &cache,
                            jpl,
                            &iod_params,
                            &dc_config,
                            None,
                            &mut rng,
                        );
                        (observations, result)
                    });

                    processed_counter.fetch_add(1, Ordering::Relaxed);

                    let row = match outcome {
                        Some((observations, Ok(fit))) => {
                            build_row(*branch_id, lineage_designation.clone(), &observations, &fit)
                                .ok()
                        }
                        _ => None,
                    };

                    if row.is_some() {
                        succeeded_counter.fetch_add(1, Ordering::Relaxed);
                    } else {
                        failed_counter.fetch_add(1, Ordering::Relaxed);
                    }
                    row
                })
                .collect()
        })
        .await
        .map_err(|e| e.to_string())?
    };

    push_log(
        job_id,
        format!(
            "Fitting done in {:.1}s: {} fitted, saving to the database...",
            stage_start.elapsed().as_secs_f64(),
            fit_rows.len()
        ),
    )
    .await;

    let error_model_str = format!("{:?}", params.error_model);
    let fit_params_json = serde_json::to_value(params).map_err(|e| e.to_string())?;

    stage_start = std::time::Instant::now();
    let mut tx = pool.begin().await.map_err(|e| e.to_string())?;
    for (i, row) in fit_rows.iter().enumerate() {
        if i > 0 && i % 2000 == 0 {
            push_log(
                job_id,
                format!("Inserted {}/{} fit rows so far...", i, fit_rows.len()),
            )
            .await;
        }
        sqlx::query(
            "INSERT INTO orbit_fits (
                lineage_designation, branch_id, observation_ids, n_observations_used,
                error_model, fit_method, fit_params, reference_epoch, semi_major_axis,
                eccentricity_sin_lon, eccentricity_cos_lon, tan_half_incl_sin_node,
                tan_half_incl_cos_node, mean_longitude, covariance, normalised_rms,
                total_newton_iterations, num_measurements, converged, keplerian,
                delta_vs_previous_fit, residuals
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, \
                $19, $20, $21, $22
            )",
        )
        .bind(&row.lineage_designation)
        .bind(row.branch_id)
        .bind(&row.observation_ids)
        .bind(row.n_observations_used)
        .bind(&error_model_str)
        .bind(row.fit_method)
        .bind(&fit_params_json)
        .bind(row.reference_epoch)
        .bind(row.semi_major_axis)
        .bind(row.eccentricity_sin_lon)
        .bind(row.eccentricity_cos_lon)
        .bind(row.tan_half_incl_sin_node)
        .bind(row.tan_half_incl_cos_node)
        .bind(row.mean_longitude)
        .bind(&row.covariance)
        .bind(row.normalised_rms)
        .bind(0_i32)
        .bind(row.num_measurements)
        .bind(row.converged)
        .bind(&row.keplerian_json)
        .bind(Option::<serde_json::Value>::None)
        .bind(serde_json::Value::Array(Vec::new()))
        .execute(&mut *tx)
        .await
        .map_err(|e| e.to_string())?;
    }
    tx.commit().await.map_err(|e| e.to_string())?;
    push_log(
        job_id,
        format!(
            "{} rows inserted in {:.1}s.",
            fit_rows.len(),
            stage_start.elapsed().as_secs_f64()
        ),
    )
    .await;

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
