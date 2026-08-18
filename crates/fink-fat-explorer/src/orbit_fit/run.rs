use dioxus::prelude::*;

use super::OrbitFitParams;
#[cfg(feature = "server")]
use super::{MIN_BASELINE_DAYS, MIN_OBSERVATIONS};

/// Kick off an Outfit orbit fit for `lineage_designation`, restricted to
/// `observation_ids`. Returns immediately with a job id; poll
/// `status::get_orbit_fit_job_status` for progress and the final result.
///
/// The fit seeds from the lineage's *current Kalman-derived orbit* rather
/// than running Gauss IOD (per product decision — the point is refining the
/// production estimate with a proper least-squares n-body fit, not
/// rediscovering the orbit from scratch).
#[server]
pub async fn start_orbit_fit(
    lineage_designation: String,
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
    observation_ids: Vec<i64>,
    params: OrbitFitParams,
) {
    let outcome = run_fit(job_id, &lineage_designation, &observation_ids, &params).await;

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

#[cfg(feature = "server")]
fn mpc_code(code: &str) -> Result<[u8; 3], String> {
    code.as_bytes()
        .try_into()
        .map_err(|_| format!("invalid MPC observatory code {code:?}"))
}

#[cfg(feature = "server")]
#[derive(sqlx::FromRow)]
struct SelectedObsRow {
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

/// The lineage's current best-hypothesis attributable state, converted to
/// Keplerian orbital elements — used both as the fit's starting point and as
/// the "vs Kalman" comparison baseline. Mirrors
/// `src/converter/family.rs::classify_from_attributable_state`.
#[cfg(feature = "server")]
pub(crate) async fn fetch_kalman_orbit(
    lineage_designation: &str,
) -> Result<outfit::OrbitalElements, String> {
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
        "WITH best_branch AS (
            SELECT branch_id
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
            LIMIT 1
        )
        SELECT ks.ra, ks.dec, ks.ra_dot, ks.dec_dot, ks.rho, ks.rho_dot, ks.epoch,
               ks.r_obs_x, ks.r_obs_y, ks.r_obs_z, ks.v_obs_x, ks.v_obs_y, ks.v_obs_z
        FROM best_branch bb
        CROSS JOIN LATERAL (
            SELECT hypothesis_id
            FROM hypotheses h
            WHERE h.branch_id = bb.branch_id
            ORDER BY h.log_weight DESC
            LIMIT 1
        ) bh
        JOIN kf_state ks ON ks.hypothesis_id = bh.hypothesis_id",
    )
    .bind(lineage_designation)
    .fetch_optional(pool)
    .await
    .map_err(|e| e.to_string())?;

    let row = row.ok_or_else(|| format!("lineage {lineage_designation:?} not found"))?;

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
    let keplerian = oe
        .to_keplerian()
        .map_err(|e| e.to_string())?
        .as_keplerian()
        .ok_or_else(|| "failed to convert the previous fit to Keplerian elements".to_string())?;

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

#[cfg(feature = "server")]
fn keplerian_view(
    elements: &outfit::KeplerianElements,
    uncertainty: Option<&outfit::orbit_type::uncertainty::KeplerianUncertainty>,
) -> super::KeplerianView {
    super::KeplerianView {
        semi_major_axis_au: elements.semi_major_axis,
        eccentricity: elements.eccentricity,
        inclination_deg: elements.inclination.to_degrees(),
        ascending_node_longitude_deg: elements.ascending_node_longitude.to_degrees(),
        periapsis_argument_deg: elements.periapsis_argument.to_degrees(),
        mean_anomaly_deg: elements.mean_anomaly.to_degrees(),
        sigma_semi_major_axis_au: uncertainty.map(|u| u.semi_major_axis),
        sigma_eccentricity: uncertainty.map(|u| u.eccentricity),
        sigma_inclination_deg: uncertainty.map(|u| u.inclination.to_degrees()),
        sigma_ascending_node_longitude_deg: uncertainty
            .map(|u| u.ascending_node_longitude.to_degrees()),
        sigma_periapsis_argument_deg: uncertainty.map(|u| u.periapsis_argument.to_degrees()),
        sigma_mean_anomaly_deg: uncertainty.map(|u| u.mean_anomaly.to_degrees()),
    }
}

#[cfg(feature = "server")]
fn planetary_bary(
    choice: super::PerturberChoice,
) -> outfit::jpl_ephem::naif::naif_ids::planet_bary::PlanetaryBary {
    use outfit::jpl_ephem::naif::naif_ids::planet_bary::PlanetaryBary;
    match choice {
        super::PerturberChoice::Mercury => PlanetaryBary::Mercury,
        super::PerturberChoice::Venus => PlanetaryBary::Venus,
        super::PerturberChoice::EarthMoon => PlanetaryBary::EarthMoon,
        super::PerturberChoice::Mars => PlanetaryBary::Mars,
        super::PerturberChoice::Jupiter => PlanetaryBary::Jupiter,
        super::PerturberChoice::Saturn => PlanetaryBary::Saturn,
        super::PerturberChoice::Uranus => PlanetaryBary::Uranus,
        super::PerturberChoice::Neptune => PlanetaryBary::Neptune,
        super::PerturberChoice::Pluto => PlanetaryBary::Pluto,
    }
}

#[cfg(feature = "server")]
async fn run_fit(
    job_id: u64,
    lineage_designation: &str,
    observation_ids: &[i64],
    params: &OrbitFitParams,
) -> Result<super::OrbitFitResult, String> {
    use outfit::differential_orbit_correction::{
        run_differential_correction, ObsFitData, ObsSelection, OutlierRejectionConfig,
    };
    use outfit::jpl_ephem::naif::naif_ids::{solar_system_bary::SolarSystemBary, NaifIds};
    use outfit::orbit_type::equinoctial_element::EquinoctialLimits;
    use outfit::propagator::{NBodyConfig, PropagatorKind};
    use outfit::{cache::OutfitCache, DifferentialCorrectionConfig};
    use photom::observation_dataset::observation::Observation;
    use photom::observation_dataset::{observation::ObservationInput, ObsDataset};
    use photom::observer::dataset::ObserverId;
    use photom::observer::error_model::{ModelCorrection, ObsErrorModel};
    use photom::{coordinates::equatorial::EquCoord, photometry::Filter, photometry::Photometry};
    use std::collections::{HashMap, HashSet};

    push_log(job_id, "Fetching the selected observations...").await;

    let pool = crate::get_pool().await;
    let mut rows: Vec<SelectedObsRow> = sqlx::query_as(
        "SELECT id, ra, ra_err, dec, dec_err, magnitude, mag_err, filter, mjd_tt, mpc_code_obs \
         FROM observations WHERE id = ANY($1)",
    )
    .bind(observation_ids)
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())?;
    rows.sort_by(|a, b| a.mjd_tt.total_cmp(&b.mjd_tt));

    if rows.len() < MIN_OBSERVATIONS {
        return Err(format!(
            "only {} of the requested observations were found in the database",
            rows.len()
        ));
    }

    push_log(
        job_id,
        format!("Resolving observatories for {} observations...", rows.len()),
    )
    .await;
    let observatories = crate::get_observatories().await;
    let mut dataset = ObsDataset::empty();
    let mut observer_ids: HashMap<[u8; 3], ObserverId> = HashMap::new();
    let codes: HashSet<[u8; 3]> = rows
        .iter()
        .map(|r| mpc_code(&r.mpc_code_obs))
        .collect::<Result<_, _>>()?;
    for code in codes {
        let observer = observatories.get(&code).ok_or_else(|| {
            format!(
                "MPC observatory code {:?} not found in the observatory list",
                std::str::from_utf8(&code).unwrap_or("?")
            )
        })?;
        let (new_dataset, id) = dataset.push_observer(observer.clone());
        dataset = new_dataset;
        observer_ids.insert(code, id);
    }

    let mut inputs = Vec::with_capacity(rows.len());
    for row in &rows {
        let code = mpc_code(&row.mpc_code_obs)?;
        inputs.push(ObservationInput::new(
            row.id as u64,
            EquCoord::new(row.ra, row.ra_err, row.dec, row.dec_err),
            Photometry {
                magnitude: row.magnitude,
                error: row.mag_err,
                filter: Filter::Int(row.filter as u32),
            },
            row.mjd_tt,
            Some(observer_ids[&code]),
        ));
    }
    let (dataset, _) = dataset
        .push_observation(inputs)
        .map_err(|e| e.to_string())?;

    let error_model = match params.error_model {
        super::ObsErrorModelChoice::Fcct14 => ObsErrorModel::FCCT14,
        super::ObsErrorModelChoice::Cbm10 => ObsErrorModel::CBM10,
        super::ObsErrorModelChoice::Vfcc17 => ObsErrorModel::VFCC17,
    };

    push_log(job_id, "Applying the observation error model...").await;
    let dataset = dataset
        .with_error_model(error_model)
        .apply_model_errors()
        .apply_batch_rms_correction(params.gap_max);

    let n = dataset.observation_count();
    let observations: Vec<Observation> = (0..n)
        .map(|i| {
            dataset
                .get_obs_by_index(i)
                .cloned()
                .expect("index within observation_count() is always present")
        })
        .collect();
    let obs_fit_data: Vec<ObsFitData> = observations
        .iter()
        .map(|obs| ObsFitData::new(obs.equ_coord().ra_error, obs.equ_coord().dec_error))
        .collect();

    push_log(job_id, "Building the observer geometry cache...").await;
    let kalman_context = crate::get_kalman_context().await;
    let ephem = kalman_context.get_ephem();
    let cache = OutfitCache::build(&dataset, &ephem.jpl, &ephem.ut1_provider, true)
        .map_err(|e| e.to_string())?;

    push_log(
        job_id,
        "Reading the lineage's current Kalman-derived orbit as the fit's starting point...",
    )
    .await;
    let kalman_orbit = fetch_kalman_orbit(lineage_designation).await?;
    let kalman_keplerian = kalman_orbit
        .clone()
        .to_keplerian()
        .map_err(|e| e.to_string())?
        .as_keplerian()
        .ok_or_else(|| "failed to convert the Kalman orbit to Keplerian elements".to_string())?;
    let equinoctial_seed = kalman_orbit
        .to_equinoctial()
        .map_err(|e| e.to_string())?
        .as_equinoctial()
        .ok_or_else(|| "failed to convert the Kalman orbit to equinoctial elements".to_string())?;

    let mut perturbing_bodies = vec![NaifIds::SSB(SolarSystemBary::Sun)];
    perturbing_bodies.extend(
        params
            .perturbers
            .iter()
            .map(|p| NaifIds::PB(planetary_bary(*p))),
    );
    let propagator = match params.propagator {
        super::PropagatorChoice::TwoBody => PropagatorKind::TwoBody,
        super::PropagatorChoice::NBody => PropagatorKind::NBody(NBodyConfig {
            perturbing_bodies,
            ..Default::default()
        }),
    };

    let dc_config = DifferentialCorrectionConfig {
        max_newton_iterations: params.max_newton_iterations,
        max_outlier_rejection_passes: params.max_outlier_rejection_passes,
        convergence_threshold: params.convergence_threshold,
        convergence_before_rejection_threshold: params.convergence_before_rejection_threshold,
        rms_stagnation_ratio: params.rms_stagnation_ratio,
        rms_divergence_ratio: params.rms_divergence_ratio,
        max_stagnation_iterations: params.max_stagnation_iterations,
        enable_outlier_rejection: params.enable_outlier_rejection,
        outlier_rejection_config: OutlierRejectionConfig::default(),
        orbital_limits: EquinoctialLimits::default(),
        free_elements: [true; 6],
        propagator,
    };

    push_log(
        job_id,
        "Running the differential correction (this can take a while with an N-body propagator)...",
    )
    .await;
    let jpl: &'static outfit::JPLEphem = &ephem.jpl;
    let dc_output = tokio::task::spawn_blocking(move || {
        run_differential_correction(
            &observations,
            &obs_fit_data,
            &equinoctial_seed,
            &cache,
            jpl,
            &dc_config,
        )
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;

    push_log(
        job_id,
        format!(
            "Fit finished: normalised RMS = {:.4}, {} Newton iterations.",
            dc_output.normalised_rms, dc_output.total_newton_iterations
        ),
    )
    .await;

    let new_elements: outfit::OrbitalElements = dc_output.clone().into();
    let new_keplerian_full = new_elements.to_keplerian().map_err(|e| e.to_string())?;
    let (new_keplerian, new_uncertainty) = match &new_keplerian_full {
        outfit::OrbitalElements::Keplerian {
            elements,
            uncertainty,
            ..
        } => (elements.clone(), uncertainty.clone()),
        _ => return Err("expected Keplerian orbital elements after conversion".to_string()),
    };

    push_log(job_id, "Comparing against the current Kalman orbit...").await;
    let delta_vs_kalman = Some(keplerian_delta(&new_keplerian, &kalman_keplerian));

    push_log(
        job_id,
        "Looking up the previous Outfit fit for this lineage...",
    )
    .await;
    let previous_fit = fetch_previous_fit(lineage_designation).await?;
    let delta_vs_previous_fit = previous_fit
        .as_ref()
        .map(|p| keplerian_delta(&new_keplerian, &p.keplerian));

    let n_observations_used = dc_output
        .final_obs_fit_data
        .iter()
        .filter(|o| o.selection == ObsSelection::Active)
        .count();
    let n_observations_rejected = dc_output.final_obs_fit_data.len() - n_observations_used;
    let degrees_of_freedom = dc_output.num_measurements as i64 - 6;
    let reduced_chi2 = dc_output.normalised_rms * dc_output.normalised_rms;

    let residuals: Vec<super::ObsResidual> = rows
        .iter()
        .zip(dc_output.final_obs_fit_data.iter())
        .map(|(row, fit_data)| super::ObsResidual {
            obs_id: row.id,
            mjd_tt: row.mjd_tt,
            residual_ra_arcsec: fit_data.residual_ra.to_degrees() * 3600.0,
            residual_dec_arcsec: fit_data.residual_dec.to_degrees() * 3600.0,
            chi: fit_data.chi,
            selection: if fit_data.selection == ObsSelection::Active {
                super::ObsSelectionView::Kept
            } else {
                super::ObsSelectionView::Rejected
            },
        })
        .collect();

    push_log(job_id, "Saving the fit result to the database...").await;
    let equinoctial = &dc_output.elements;
    let covariance: Vec<f64> = dc_output.uncertainty.covariance.iter().copied().collect();
    let fit_params_json = serde_json::to_value(params).map_err(|e| e.to_string())?;
    let converged = dc_output.normalised_rms.is_finite() && dc_output.normalised_rms < 10.0;
    let keplerian_view_result = keplerian_view(&new_keplerian, new_uncertainty.as_ref());
    let keplerian_json = serde_json::to_value(&keplerian_view_result).map_err(|e| e.to_string())?;
    let delta_vs_previous_fit_json =
        serde_json::to_value(&delta_vs_previous_fit).map_err(|e| e.to_string())?;
    let residuals_json = serde_json::to_value(&residuals).map_err(|e| e.to_string())?;

    sqlx::query(
        "INSERT INTO orbit_fits (
            lineage_designation, observation_ids, n_observations_used, error_model, fit_params,
            reference_epoch, semi_major_axis, eccentricity_sin_lon, eccentricity_cos_lon,
            tan_half_incl_sin_node, tan_half_incl_cos_node, mean_longitude, covariance,
            normalised_rms, total_newton_iterations, num_measurements, converged,
            keplerian, delta_vs_previous_fit, residuals
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)",
    )
    .bind(lineage_designation)
    .bind(observation_ids)
    .bind(n_observations_used as i32)
    .bind(format!("{:?}", params.error_model))
    .bind(fit_params_json)
    .bind(equinoctial.reference_epoch)
    .bind(equinoctial.semi_major_axis)
    .bind(equinoctial.eccentricity_sin_lon)
    .bind(equinoctial.eccentricity_cos_lon)
    .bind(equinoctial.tan_half_incl_sin_node)
    .bind(equinoctial.tan_half_incl_cos_node)
    .bind(equinoctial.mean_longitude)
    .bind(&covariance)
    .bind(dc_output.normalised_rms)
    .bind(dc_output.total_newton_iterations as i32)
    .bind(dc_output.num_measurements as i32)
    .bind(converged)
    .bind(keplerian_json)
    .bind(delta_vs_previous_fit_json)
    .bind(residuals_json)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;

    Ok(super::OrbitFitResult {
        reference_epoch: equinoctial.reference_epoch,
        keplerian: keplerian_view_result,
        delta_vs_kalman,
        delta_vs_previous_fit,
        normalised_rms: dc_output.normalised_rms,
        reduced_chi2,
        degrees_of_freedom,
        total_newton_iterations: dc_output.total_newton_iterations,
        num_measurements: dc_output.num_measurements,
        n_observations_used,
        n_observations_rejected,
        converged,
        residuals,
    })
}
