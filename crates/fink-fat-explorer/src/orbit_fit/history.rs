use dioxus::prelude::*;

use super::OrbitFitSummary;

/// Previous Outfit fits for a lineage, most recent first. Used both to
/// render the "History" tab on the fit result page and, server-side, by
/// `run::start_orbit_fit` to compute the "vs previous fit" delta.
#[server]
pub async fn get_orbit_fit_history(
    lineage_designation: String,
) -> Result<Vec<OrbitFitSummary>, ServerFnError> {
    use crate::get_pool;

    #[cfg_attr(feature = "server", derive(sqlx::FromRow))]
    struct Row {
        id: i64,
        fitted_at: chrono::DateTime<chrono::Utc>,
        n_observations_used: i32,
        normalised_rms: f64,
        reference_epoch: f64,
        semi_major_axis: f64,
        eccentricity_sin_lon: f64,
        eccentricity_cos_lon: f64,
        tan_half_incl_sin_node: f64,
        tan_half_incl_cos_node: f64,
        mean_longitude: f64,
        covariance: Vec<f64>,
    }

    let pool = get_pool().await;
    let rows: Vec<Row> = sqlx::query_as(
        "SELECT id, fitted_at, n_observations_used, normalised_rms, reference_epoch, \
         semi_major_axis, eccentricity_sin_lon, eccentricity_cos_lon, \
         tan_half_incl_sin_node, tan_half_incl_cos_node, mean_longitude, covariance \
         FROM orbit_fits \
         WHERE lineage_designation = $1 \
         ORDER BY fitted_at DESC",
    )
    .bind(&lineage_designation)
    .fetch_all(pool)
    .await
    .map_err(|e| ServerFnError::new(e.to_string()))?;

    Ok(rows
        .into_iter()
        .map(|r| OrbitFitSummary {
            id: r.id,
            fitted_at: r.fitted_at.to_rfc3339(),
            n_observations_used: r.n_observations_used,
            normalised_rms: r.normalised_rms,
            reference_epoch: r.reference_epoch,
            semi_major_axis_au: r.semi_major_axis,
            eccentricity_sin_lon: r.eccentricity_sin_lon,
            eccentricity_cos_lon: r.eccentricity_cos_lon,
            tan_half_incl_sin_node: r.tan_half_incl_sin_node,
            tan_half_incl_cos_node: r.tan_half_incl_cos_node,
            mean_longitude: r.mean_longitude,
            covariance: r.covariance,
        })
        .collect())
}
