//! Writing fit outcomes to `orbit_fits` / `orbit_fit_failures`.
//!
//! One insert for both fit paths: the column list lived in two places before,
//! and promptly drifted — the single-lineage insert was missing `branch_id`,
//! then `fit_method`, so its rows were indistinguishable from a bulk row for
//! another branch and were mislabelled as least-squares fits.

use sqlx::PgPool;

use super::fit::FitProduct;
use super::params::OrbitFitParams;

/// Column list and placeholders of [`insert_orbit_fits`], kept adjacent so
/// they cannot fall out of step.
const INSERT_ORBIT_FIT: &str = "
    INSERT INTO orbit_fits (
        lineage_designation, branch_id, observation_ids, n_observations_used, error_model,
        fit_method, fit_params, reference_epoch, semi_major_axis, eccentricity_sin_lon,
        eccentricity_cos_lon, tan_half_incl_sin_node, tan_half_incl_cos_node, mean_longitude,
        covariance, normalised_rms, total_newton_iterations, num_measurements, converged,
        keplerian, delta_vs_previous_fit, residuals
    ) VALUES (
        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19,
        $20, $21, $22
    )
";

/// One row to store: what the caller knows about the fit, plus the
/// [`FitProduct`] the pipeline produced.
pub struct OrbitFitRow {
    pub lineage_designation: String,
    pub branch_id: i64,
    /// The observations submitted to the fit, before any outlier rejection.
    pub observation_ids: Vec<i64>,
    /// Comparison against this lineage's previous fit, when the caller
    /// computed one (the single-lineage page does; the bulk job doesn't).
    pub delta_vs_previous_fit: Option<serde_json::Value>,
    pub product: FitProduct,
}

/// Inserts `rows` in a single transaction.
///
/// # Arguments
///
/// - `params` — the parameters every row was fitted with, stored verbatim in
///   `fit_params` so a past run's exact configuration can be read back (the
///   fit page's "load params from last fit").
/// - `rows` — may be empty, in which case nothing is written.
///
/// # Errors
///
/// Serializing a row's JSON payloads, or the transaction failing, as a display
/// string.
pub async fn insert_orbit_fits(
    pool: &PgPool,
    params: &OrbitFitParams,
    rows: &[OrbitFitRow],
) -> Result<(), String> {
    if rows.is_empty() {
        return Ok(());
    }

    let error_model = format!("{:?}", params.error_model);
    let fit_params = serde_json::to_value(params).map_err(|e| e.to_string())?;

    let mut tx = pool.begin().await.map_err(|e| e.to_string())?;
    for row in rows {
        let product = &row.product;
        let keplerian = serde_json::to_value(&product.keplerian).map_err(|e| e.to_string())?;
        let residuals = serde_json::to_value(&product.residuals).map_err(|e| e.to_string())?;

        sqlx::query(INSERT_ORBIT_FIT)
            .bind(&row.lineage_designation)
            .bind(row.branch_id)
            .bind(&row.observation_ids)
            .bind(product.n_observations_used as i32)
            .bind(&error_model)
            .bind(product.fit_method.as_column())
            .bind(&fit_params)
            .bind(product.elements.reference_epoch)
            .bind(product.elements.semi_major_axis)
            .bind(product.elements.eccentricity_sin_lon)
            .bind(product.elements.eccentricity_cos_lon)
            .bind(product.elements.tan_half_incl_sin_node)
            .bind(product.elements.tan_half_incl_cos_node)
            .bind(product.elements.mean_longitude)
            .bind(&product.covariance)
            .bind(product.normalised_rms)
            .bind(product.total_newton_iterations as i32)
            .bind(product.num_measurements as i32)
            .bind(product.converged())
            .bind(keplerian)
            .bind(row.delta_vs_previous_fit.clone())
            .bind(residuals)
            .execute(&mut *tx)
            .await
            .map_err(|e| e.to_string())?;
    }
    tx.commit().await.map_err(|e| e.to_string())?;

    Ok(())
}

/// Records branches whose fit produced no orbit at all.
///
/// `orbit_fits` only ever holds successes, so without this a branch whose
/// Gauss IOD failed is indistinguishable from one that was never submitted to
/// a fit — a distinction the homepage's quality tier needs.
///
/// # Errors
///
/// The insert failing, as a display string.
pub async fn record_fit_failures(pool: &PgPool, branch_ids: &[i64]) -> Result<(), String> {
    if branch_ids.is_empty() {
        return Ok(());
    }

    sqlx::query("INSERT INTO orbit_fit_failures (branch_id) SELECT * FROM UNNEST($1::bigint[])")
        .bind(branch_ids)
        .execute(pool)
        .await
        .map_err(|e| e.to_string())?;

    Ok(())
}
