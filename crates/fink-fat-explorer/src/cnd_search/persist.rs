//! Writing finished CND check attempts to `cnd_queries`.
//!
//! Kept separate from [`super::run`] for the same reason
//! `skybot_search::persist` is kept apart from `skybot_search::run`: the
//! insert itself needs a real Postgres pool and is worth testing/reusing in
//! isolation from the job registry/batching plumbing. Reused unmodified by
//! `crate::bulk_cnd::run`, which inserts one row per branch after grouping
//! the bulk job's results — no second insert function for the bulk path.

use sqlx::PgPool;

use super::CndHit;

/// Records one finished CND check attempt for a lineage/branch.
///
/// Always insert, even when `hits` is empty: a lineage with no row in
/// `cnd_queries` has never been checked, which the lineage page's "last
/// checked" display needs to tell apart from "checked and found nothing"
/// (see [`super::history::get_last_cnd_query`]).
///
/// # Arguments
///
/// * `pool` — the shared Postgres pool.
/// * `lineage_designation` — the lineage this check was run for.
/// * `branch_id` — the branch whose observations were submitted, when known
///   (the bulk job always knows it; kept `Option` for parity with
///   `orbit_fits.branch_id`, which is nullable for the same reason).
/// * `time_separation_s`, `angle_separation_arcsec` — the thresholds the
///   check actually used.
/// * `hits` — every match found, possibly empty.
///
/// # Return
///
/// `Ok(())` once the row is committed. `Err` with the failure serialized to
/// a display string if `hits` can't be encoded as JSON or the insert fails.
pub async fn insert_cnd_query(
    pool: &PgPool,
    lineage_designation: &str,
    branch_id: Option<i64>,
    time_separation_s: f64,
    angle_separation_arcsec: f64,
    hits: &[CndHit],
) -> Result<(), String> {
    let hits_json = serde_json::to_value(hits).map_err(|e| e.to_string())?;

    sqlx::query(
        "INSERT INTO cnd_queries \
         (lineage_designation, branch_id, time_separation_s, angle_separation_arcsec, hits) \
         VALUES ($1, $2, $3, $4, $5)",
    )
    .bind(lineage_designation)
    .bind(branch_id)
    .bind(time_separation_s)
    .bind(angle_separation_arcsec)
    .bind(hits_json)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;

    Ok(())
}
