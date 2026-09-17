use dioxus::prelude::*;

#[cfg(feature = "server")]
use super::OrbitDelta;
use super::OrbitFitResult;
#[cfg(feature = "server")]
use crate::fit_pipeline::fit::{FitMethod, KeplerianView};
use crate::fit_pipeline::params::OrbitFitParams;

/// The exact `OrbitFitParams` (and the `branch_id` it ran on) of a lineage's
/// most recent orbit fit — individual or bulk, both write to `orbit_fits`
/// with the same `fit_params` shape. Backs the fit page's "Load params from
/// last fit" button, so the form can be populated with the parameters a past
/// run actually used instead of the user re-entering them from memory and
/// risking a subtle mismatch (the scenario this button exists to prevent).
#[server]
pub async fn get_latest_fit_params(
    lineage_designation: String,
) -> Result<Option<(OrbitFitParams, Option<i64>)>, ServerFnError> {
    use crate::get_pool;

    #[cfg_attr(feature = "server", derive(sqlx::FromRow))]
    struct Row {
        branch_id: Option<i64>,
        fit_params: sqlx::types::Json<OrbitFitParams>,
    }

    let pool = get_pool().await;
    let row: Option<Row> = sqlx::query_as(
        "SELECT branch_id, fit_params FROM orbit_fits \
         WHERE lineage_designation = $1 \
         ORDER BY fitted_at DESC \
         LIMIT 1",
    )
    .bind(&lineage_designation)
    .fetch_optional(pool)
    .await
    .map_err(|e| ServerFnError::new(e.to_string()))?;

    Ok(row.map(|r| (r.fit_params.0, r.branch_id)))
}

/// The most recent Outfit fit for a lineage, if any — used so the orbit-fit
/// page can land directly on a lineage's last result instead of always
/// opening on the (empty) form. `delta_vs_kalman` is recomputed live against
/// the lineage's *current* Kalman-derived orbit rather than read back from
/// storage, since that orbit may have moved since the fit was run.
#[server]
pub async fn get_latest_orbit_fit_result(
    lineage_designation: String,
) -> Result<Option<OrbitFitResult>, ServerFnError> {
    use super::run::{fetch_kalman_orbit, resolve_best_branch_id};
    use crate::fit_pipeline::fit::ObsResidual;
    use crate::get_pool;

    #[cfg_attr(feature = "server", derive(sqlx::FromRow))]
    struct Row {
        branch_id: Option<i64>,
        fit_method: String,
        observation_ids: Vec<i64>,
        n_observations_used: i32,
        reference_epoch: f64,
        normalised_rms: f64,
        total_newton_iterations: i32,
        num_measurements: i32,
        converged: bool,
        keplerian: sqlx::types::Json<KeplerianView>,
        delta_vs_previous_fit: Option<sqlx::types::Json<OrbitDelta>>,
        residuals: sqlx::types::Json<Vec<ObsResidual>>,
    }

    let pool = get_pool().await;
    let row: Option<Row> = sqlx::query_as(
        "SELECT branch_id, fit_method, observation_ids, n_observations_used, reference_epoch, normalised_rms, \
         total_newton_iterations, num_measurements, converged, keplerian, \
         delta_vs_previous_fit, residuals \
         FROM orbit_fits \
         WHERE lineage_designation = $1 \
         ORDER BY fitted_at DESC \
         LIMIT 1",
    )
    .bind(&lineage_designation)
    .fetch_optional(pool)
    .await
    .map_err(|e| ServerFnError::new(e.to_string()))?;

    let Some(row) = row else {
        return Ok(None);
    };

    let keplerian = row.keplerian.0;
    let n_observations_rejected =
        row.observation_ids.len() as usize - row.n_observations_used as usize;
    let degrees_of_freedom = row.num_measurements as i64 - 6;
    let reduced_chi2 = row.normalised_rms * row.normalised_rms;

    // Best-effort: any failure to resolve/fetch the current Kalman orbit just
    // means no "vs Kalman" delta is shown, not that the stored fit is lost.
    let delta_vs_kalman = match resolve_best_branch_id(&lineage_designation).await {
        Ok(Some(branch_id)) => match fetch_kalman_orbit(branch_id).await {
            Ok(kalman_orbit) => {
                let kalman_keplerian = kalman_orbit
                    .to_keplerian()
                    .ok()
                    .and_then(|oe| oe.as_keplerian());
                kalman_keplerian.map(|old| delta_view_vs_kalman(&keplerian, &old))
            }
            Err(_) => None,
        },
        _ => None,
    };

    Ok(Some(OrbitFitResult {
        branch_id: row.branch_id,
        fit_method: FitMethod::from_column(&row.fit_method),
        reference_epoch: row.reference_epoch,
        keplerian,
        delta_vs_kalman,
        delta_vs_previous_fit: row.delta_vs_previous_fit.map(|j| j.0),
        normalised_rms: row.normalised_rms,
        reduced_chi2,
        degrees_of_freedom,
        total_newton_iterations: row.total_newton_iterations as usize,
        num_measurements: row.num_measurements as usize,
        n_observations_used: row.n_observations_used as usize,
        n_observations_rejected,
        converged: row.converged,
        residuals: row.residuals.0,
    }))
}

/// Difference (new fit minus the current Kalman orbit), angles wrapped to
/// (-180, 180] degrees — same arithmetic as `run::keplerian_delta`, but
/// taking the fit's already-degrees `KeplerianView` (there's no `outfit`
/// orbital-elements type to reconstruct a stored fit into) against the
/// Kalman orbit's `outfit::KeplerianElements` (radians).
#[cfg(feature = "server")]
fn delta_view_vs_kalman(new: &KeplerianView, old: &outfit::KeplerianElements) -> OrbitDelta {
    use super::run::wrap_deg;

    OrbitDelta {
        delta_semi_major_axis_au: new.semi_major_axis_au - old.semi_major_axis,
        delta_eccentricity: new.eccentricity - old.eccentricity,
        delta_inclination_deg: wrap_deg(new.inclination_deg - old.inclination.to_degrees()),
        delta_ascending_node_longitude_deg: wrap_deg(
            new.ascending_node_longitude_deg - old.ascending_node_longitude.to_degrees(),
        ),
        delta_periapsis_argument_deg: wrap_deg(
            new.periapsis_argument_deg - old.periapsis_argument.to_degrees(),
        ),
        delta_mean_anomaly_deg: wrap_deg(new.mean_anomaly_deg - old.mean_anomaly.to_degrees()),
        reference_epoch: old.reference_epoch,
    }
}
