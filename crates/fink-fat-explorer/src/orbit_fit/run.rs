use dioxus::prelude::*;

#[cfg(feature = "server")]
use crate::fit_pipeline::dataset::FitObservation;
use crate::fit_pipeline::params::OrbitFitParams;
#[cfg(feature = "server")]
use crate::fit_pipeline::params::{SeedStrategy, MIN_BASELINE_DAYS, MIN_OBSERVATIONS};

/// Kick off an Outfit orbit fit for `lineage_designation`, restricted to
/// `observation_ids`. Returns immediately with a job id; poll
/// `status::get_orbit_fit_job_status` for progress and the final result.
///
/// `branch_id` must be the branch `observation_ids` were drawn from (the
/// caller resolves it once via
/// `lineage_page::observations_table::get_lineage_observations` and threads
/// it through, rather than this function re-resolving "the lineage's best
/// branch" independently — see [`resolve_best_branch_id`]'s doc comment for
/// why that used to be able to disagree with the caller's own resolution).
/// It seeds the differential correction (when `params.seed_strategy` is
/// [`SeedStrategy::KalmanOrbit`]) and is stored alongside the result
/// for traceability, the same way the bulk fit already does.
#[server]
pub async fn start_orbit_fit(
    lineage_designation: String,
    branch_id: i64,
    observation_ids: Vec<i64>,
    params: OrbitFitParams,
) -> Result<u64, ServerFnError> {
    use super::OrbitFitJob;
    use crate::{get_orbit_fit_jobs, get_pool, NEXT_ORBIT_FIT_JOB_ID};
    use std::sync::atomic::Ordering;

    if observation_ids.len() < MIN_OBSERVATIONS {
        return Err(ServerFnError::new(format!(
            "at least {MIN_OBSERVATIONS} observations are required for an orbit fit, got {}",
            observation_ids.len()
        )));
    }

    let pool = get_pool().await;
    let (min_mjd, max_mjd): (Option<f64>, Option<f64>) =
        sqlx::query_as("SELECT MIN(mjd_tt), MAX(mjd_tt) FROM observations WHERE id = ANY($1)")
            .bind(&observation_ids)
            .fetch_one(pool)
            .await
            .map_err(|e| ServerFnError::new(e.to_string()))?;

    let baseline_days = match (min_mjd, max_mjd) {
        (Some(min), Some(max)) => max - min,
        _ => 0.0,
    };
    if baseline_days < MIN_BASELINE_DAYS {
        return Err(ServerFnError::new(format!(
            "selected observations span only {baseline_days:.3} days; at least \
             {MIN_BASELINE_DAYS} days are required for a reliable fit"
        )));
    }

    let job_id = NEXT_ORBIT_FIT_JOB_ID.fetch_add(1, Ordering::Relaxed);
    {
        let jobs = get_orbit_fit_jobs().await;
        jobs.lock()
            .expect("orbit fit job registry poisoned")
            .insert(job_id, OrbitFitJob::new());
    }

    tokio::spawn(run_fit_job(
        job_id,
        lineage_designation,
        branch_id,
        observation_ids,
        params,
    ));

    Ok(job_id)
}

#[cfg(feature = "server")]
async fn push_log(job_id: u64, message: impl Into<String>) {
    let jobs = crate::get_orbit_fit_jobs().await;
    if let Ok(mut jobs) = jobs.lock() {
        if let Some(job) = jobs.get_mut(&job_id) {
            job.logs.push(message.into());
        }
    }
}

#[cfg(feature = "server")]
async fn run_fit_job(
    job_id: u64,
    lineage_designation: String,
    branch_id: i64,
    observation_ids: Vec<i64>,
    params: OrbitFitParams,
) {
    let outcome = run_fit(
        job_id,
        &lineage_designation,
        branch_id,
        &observation_ids,
        &params,
    )
    .await;

    let jobs = crate::get_orbit_fit_jobs().await;
    if let Ok(mut jobs) = jobs.lock() {
        if let Some(job) = jobs.get_mut(&job_id) {
            match outcome {
                Ok(result) => {
                    job.status = super::JobStatus::Done;
                    job.result = Some(result);
                }
                Err(message) => {
                    job.status = super::JobStatus::Failed;
                    job.error = Some(message);
                }
            }
        }
    }
}

/// Resolves a lineage's best branch — highest `cumulative_llr`, `NaN`/
/// `Infinity` treated as neutral (0) rather than as the largest possible
/// value (same rationale as `homepage::snapshot::sanitize_llr` for the same
/// column). The single point of resolution for "which branch does this
/// lineage currently point at": shared by `fetch_kalman_orbit`'s "current
/// Kalman orbit" comparison in `latest::get_latest_orbit_fit_result` and by
/// `lineage_page::observations_table::get_lineage_observations`, which the
/// fit page itself reads to pin its `branch_id` — keeping this query in one
/// place is what makes those two agree.
///
/// # Returns
///
/// The lineage's best `branch_id`, or `None` if it has no branches.
#[cfg(feature = "server")]
pub(crate) async fn resolve_best_branch_id(
    lineage_designation: &str,
) -> Result<Option<i64>, String> {
    let pool = crate::get_pool().await;
    let row: Option<(i64,)> = sqlx::query_as(
        "SELECT branch_id
         FROM branches
         WHERE lineage_designation = $1
         ORDER BY (
             CASE
                 WHEN cumulative_llr = 'NaN'::double precision THEN 0
                 WHEN cumulative_llr = 'Infinity'::double precision THEN 0
                 WHEN cumulative_llr = '-Infinity'::double precision THEN 0
                 ELSE cumulative_llr
             END
         ) DESC
         LIMIT 1",
    )
    .bind(lineage_designation)
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?;

    Ok(row.map(|(branch_id,)| branch_id))
}

/// A branch's best-hypothesis attributable state, converted to Keplerian
/// orbital elements — used both as the fit's starting point (when
/// `SeedStrategy::KalmanOrbit`) and as the "vs Kalman" comparison baseline.
/// Mirrors `src/converter/family.rs::classify_from_attributable_state`.
///
/// Takes an explicit `branch_id` rather than resolving one itself — see
/// [`resolve_best_branch_id`] for callers that don't already have one.
#[cfg(feature = "server")]
pub(crate) async fn fetch_kalman_orbit(branch_id: i64) -> Result<outfit::OrbitalElements, String> {
    use fink_fat_engine::topocentric_kf::conversion::attributable_to_cartesian;
    use nalgebra::{Vector3, Vector6};

    #[derive(sqlx::FromRow)]
    struct Row {
        ra: f64,
        dec: f64,
        ra_dot: f64,
        dec_dot: f64,
        rho: f64,
        rho_dot: f64,
        epoch: f64,
        r_obs_x: f64,
        r_obs_y: f64,
        r_obs_z: f64,
        v_obs_x: f64,
        v_obs_y: f64,
        v_obs_z: f64,
    }

    let pool = crate::get_pool().await;
    let row: Option<Row> = sqlx::query_as(
        "SELECT ks.ra, ks.dec, ks.ra_dot, ks.dec_dot, ks.rho, ks.rho_dot, ks.epoch,
                ks.r_obs_x, ks.r_obs_y, ks.r_obs_z, ks.v_obs_x, ks.v_obs_y, ks.v_obs_z
         FROM hypotheses h
         JOIN kf_state ks ON ks.hypothesis_id = h.hypothesis_id
         WHERE h.branch_id = $1
         ORDER BY h.log_weight DESC
         LIMIT 1",
    )
    .bind(branch_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?;

    let row = row.ok_or_else(|| format!("branch {branch_id} has no hypotheses"))?;

    let state = Vector6::new(
        row.ra,
        row.dec,
        row.ra_dot,
        row.dec_dot,
        row.rho,
        row.rho_dot,
    );
    let r_obs = Vector3::new(row.r_obs_x, row.r_obs_y, row.r_obs_z);
    let v_obs = Vector3::new(row.v_obs_x, row.v_obs_y, row.v_obs_z);
    let cartesian = attributable_to_cartesian(&state, &r_obs, &v_obs);

    Ok(outfit::OrbitalElements::from_orbital_state(
        &cartesian.pos,
        &cartesian.vel,
        row.epoch,
    ))
}

#[cfg(feature = "server")]
struct PreviousFit {
    fitted_at: chrono::DateTime<chrono::Utc>,
    keplerian: outfit::KeplerianElements,
}

#[cfg(feature = "server")]
async fn fetch_previous_fit(lineage_designation: &str) -> Result<Option<PreviousFit>, String> {
    #[derive(sqlx::FromRow)]
    struct Row {
        fitted_at: chrono::DateTime<chrono::Utc>,
        reference_epoch: f64,
        semi_major_axis: f64,
        eccentricity_sin_lon: f64,
        eccentricity_cos_lon: f64,
        tan_half_incl_sin_node: f64,
        tan_half_incl_cos_node: f64,
        mean_longitude: f64,
    }

    let pool = crate::get_pool().await;
    let row: Option<Row> = sqlx::query_as(
        "SELECT fitted_at, reference_epoch, semi_major_axis, eccentricity_sin_lon, \
         eccentricity_cos_lon, tan_half_incl_sin_node, tan_half_incl_cos_node, mean_longitude \
         FROM orbit_fits \
         WHERE lineage_designation = $1 \
         ORDER BY fitted_at DESC \
         LIMIT 1",
    )
    .bind(lineage_designation)
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?;

    let Some(row) = row else {
        return Ok(None);
    };

    let equinoctial = outfit::EquinoctialElements {
        reference_epoch: row.reference_epoch,
        semi_major_axis: row.semi_major_axis,
        eccentricity_sin_lon: row.eccentricity_sin_lon,
        eccentricity_cos_lon: row.eccentricity_cos_lon,
        tan_half_incl_sin_node: row.tan_half_incl_sin_node,
        tan_half_incl_cos_node: row.tan_half_incl_cos_node,
        mean_longitude: row.mean_longitude,
    };
    let oe = outfit::OrbitalElements::Equinoctial {
        elements: equinoctial,
        uncertainty: None,
        covariance: None,
    };
    let keplerian = crate::fit_pipeline::fit::to_keplerian(oe)?;

    Ok(Some(PreviousFit {
        fitted_at: row.fitted_at,
        keplerian,
    }))
}
/// Wrap a difference of angles (degrees) into (-180, 180].
#[cfg(feature = "server")]
pub(crate) fn wrap_deg(x: f64) -> f64 {
    let mut y = x % 360.0;
    if y <= -180.0 {
        y += 360.0;
    }
    if y > 180.0 {
        y -= 360.0;
    }
    y
}

#[cfg(feature = "server")]
fn keplerian_delta(
    new: &outfit::KeplerianElements,
    old: &outfit::KeplerianElements,
) -> super::OrbitDelta {
    super::OrbitDelta {
        delta_semi_major_axis_au: new.semi_major_axis - old.semi_major_axis,
        delta_eccentricity: new.eccentricity - old.eccentricity,
        delta_inclination_deg: wrap_deg((new.inclination - old.inclination).to_degrees()),
        delta_ascending_node_longitude_deg: wrap_deg(
            (new.ascending_node_longitude - old.ascending_node_longitude).to_degrees(),
        ),
        delta_periapsis_argument_deg: wrap_deg(
            (new.periapsis_argument - old.periapsis_argument).to_degrees(),
        ),
        delta_mean_anomaly_deg: wrap_deg((new.mean_anomaly - old.mean_anomaly).to_degrees()),
        reference_epoch: old.reference_epoch,
    }
}

/// Loads the observations a fit was asked to use, as the pipeline's own row
/// shape, tagged with the branch they belong to.
#[cfg(feature = "server")]
async fn load_observations(
    branch_id: i64,
    observation_ids: &[i64],
) -> Result<Vec<FitObservation>, String> {
    #[derive(sqlx::FromRow)]
    struct Row {
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
        "SELECT id, ra, ra_err, dec, dec_err, magnitude, mag_err, filter, mjd_tt, mpc_code_obs \
         FROM observations WHERE id = ANY($1)",
    )
    .bind(observation_ids)
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;

    Ok(rows
        .into_iter()
        .map(|r| FitObservation {
            id: r.id,
            branch_id,
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

/// Runs one branch's fit and stores it, reporting progress through the job's
/// log as it goes.
///
/// Everything numerical is delegated to [`crate::fit_pipeline`], which the bulk
/// fit runs through as well: that is what makes a fit launched here comparable
/// to the bulk row for the same branch, rather than a second implementation
/// that drifts.
#[cfg(feature = "server")]
async fn run_fit(
    job_id: u64,
    lineage_designation: &str,
    branch_id: i64,
    observation_ids: &[i64],
    params: &OrbitFitParams,
) -> Result<super::OrbitFitResult, String> {
    use crate::fit_pipeline::dataset;
    use crate::fit_pipeline::fit::{self, Diagnostics, FitMethod};
    use crate::fit_pipeline::params::{build_dc_config, build_iod_params, to_error_model};
    use crate::fit_pipeline::store::{self, OrbitFitRow};

    push_log(job_id, "Fetching the selected observations...").await;
    let observation_rows = load_observations(branch_id, observation_ids).await?;
    if observation_rows.len() < MIN_OBSERVATIONS {
        return Err(format!(
            "only {} of the requested observations were found in the database",
            observation_rows.len()
        ));
    }

    push_log(
        job_id,
        format!(
            "Building the observation dataset ({} observations) and applying the error model...",
            observation_rows.len()
        ),
    )
    .await;
    let error_model = to_error_model(params.error_model);
    let corrected = dataset::apply_error_model(
        dataset::build_dataset(&observation_rows)?,
        error_model,
        params.gap_max,
    );
    let observations = dataset::observations_of(&corrected, branch_id);
    if observations.len() < MIN_OBSERVATIONS {
        return Err(format!(
            "the dataset built for branch {branch_id} holds only {} of its {} observations",
            observations.len(),
            observation_rows.len()
        ));
    }

    push_log(job_id, "Building the observer geometry cache...").await;
    let kalman_context = crate::get_kalman_context().await;
    let ephem = kalman_context.get_ephem();
    let cache = dataset::build_cache(&corrected, &ephem.jpl, &ephem.ut1_provider)?;

    push_log(
        job_id,
        "Reading the branch's current Kalman-derived orbit for comparison...",
    )
    .await;
    // Kept as a `Result`: under `SeedlessGaussIod` this orbit only feeds the
    // informational "vs Kalman" delta, so a branch without a Kalman hypothesis
    // shouldn't fail the fit. Only the seed branch below, where it *is* the
    // starting orbit, propagates the failure.
    let kalman_orbit = fetch_kalman_orbit(branch_id).await;
    let kalman_keplerian = kalman_orbit
        .clone()
        .ok()
        .and_then(|orbit| fit::to_keplerian(orbit).ok());

    let dc_config = build_dc_config(params);
    let product = match params.seed_strategy {
        SeedStrategy::KalmanOrbit => {
            push_log(
                job_id,
                "Running the differential correction from the Kalman orbit (this can take a \
                 while with an N-body propagator)...",
            )
            .await;
            let seed = fit::to_equinoctial(kalman_orbit?)?;
            let jpl: &'static outfit::JPLEphem = &ephem.jpl;
            tokio::task::spawn_blocking(move || {
                fit::fit_from_seed(&observations, &cache, jpl, &dc_config, &seed)
            })
            .await
            .map_err(|e| e.to_string())??
        }
        SeedStrategy::SeedlessGaussIod => {
            push_log(
                job_id,
                "Running the seedless fit — Gauss IOD then differential correction, the same \
                 strategy and the same RNG stream as the bulk fit (this can take a while with \
                 an N-body propagator)...",
            )
            .await;
            let iod_params = build_iod_params(params)?;
            let jpl: &'static outfit::JPLEphem = &ephem.jpl;
            tokio::task::spawn_blocking(move || {
                fit::fit_seedless(
                    &observations,
                    &cache,
                    jpl,
                    &iod_params,
                    &dc_config,
                    branch_id,
                    Diagnostics::Full,
                )
            })
            .await
            .map_err(|e| e.to_string())??
        }
    };

    match product.fit_method {
        FitMethod::DifferentialCorrection => {
            push_log(
                job_id,
                format!(
                    "Fit finished: normalised RMS = {:.4}, {} observations kept, {} rejected \
                     ({} Newton iterations in the pass that recomputed the residuals).",
                    product.normalised_rms,
                    product.n_observations_used,
                    product.n_observations_rejected,
                    product.total_newton_iterations
                ),
            )
            .await;
        }
        FitMethod::IodOnly => {
            push_log(
                job_id,
                format!(
                    "The differential correction diverged from the Gauss IOD seed; keeping the \
                     preliminary Gauss orbit alone (RMS = {:.4}), which is what the bulk fit \
                     stores in this situation.",
                    product.normalised_rms
                ),
            )
            .await;
        }
    }

    let keplerian = product.keplerian.clone();
    let delta_vs_kalman = kalman_keplerian
        .as_ref()
        .zip(keplerian_of(&product).ok())
        .map(|(old, new)| keplerian_delta(&new, old));

    push_log(
        job_id,
        "Looking up the previous Outfit fit for this lineage...",
    )
    .await;
    let previous_fit = fetch_previous_fit(lineage_designation).await?;
    let delta_vs_previous_fit = previous_fit
        .as_ref()
        .zip(keplerian_of(&product).ok())
        .map(|(previous, new)| keplerian_delta(&new, &previous.keplerian));
    let delta_vs_previous_fit_json =
        serde_json::to_value(&delta_vs_previous_fit).map_err(|e| e.to_string())?;

    push_log(job_id, "Saving the fit result to the database...").await;
    let result = super::OrbitFitResult {
        branch_id: Some(branch_id),
        fit_method: product.fit_method,
        reference_epoch: product.elements.reference_epoch,
        keplerian,
        delta_vs_kalman,
        delta_vs_previous_fit,
        normalised_rms: product.normalised_rms,
        reduced_chi2: product.normalised_rms * product.normalised_rms,
        degrees_of_freedom: product.num_measurements as i64 - 6,
        total_newton_iterations: product.total_newton_iterations,
        num_measurements: product.num_measurements,
        n_observations_used: product.n_observations_used,
        n_observations_rejected: product.n_observations_rejected,
        converged: product.converged(),
        residuals: product.residuals.clone(),
    };

    let pool = crate::get_pool().await;
    store::insert_orbit_fits(
        pool,
        params,
        &[OrbitFitRow {
            lineage_designation: lineage_designation.to_string(),
            branch_id,
            observation_ids: observation_ids.to_vec(),
            delta_vs_previous_fit: Some(delta_vs_previous_fit_json),
            product,
        }],
    )
    .await?;

    Ok(result)
}

/// The fitted orbit as `outfit`'s own Keplerian elements, for the "vs Kalman"
/// and "vs previous fit" comparisons — which need radians and the raw type,
/// not the display view the product already carries.
#[cfg(feature = "server")]
fn keplerian_of(
    product: &crate::fit_pipeline::fit::FitProduct,
) -> Result<outfit::KeplerianElements, String> {
    crate::fit_pipeline::fit::to_keplerian(outfit::OrbitalElements::Equinoctial {
        elements: product.elements.clone(),
        uncertainty: None,
        covariance: None,
    })
}
