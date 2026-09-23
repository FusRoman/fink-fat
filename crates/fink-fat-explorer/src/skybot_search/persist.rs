//! Writing finished Skybot search attempts to `skybot_queries`.
//!
//! Kept separate from [`super::run`] (which orchestrates the job/HTTP fan-out)
//! so the actual insert — the part that needs a real Postgres pool and is
//! worth testing in isolation from the job registry/semaphore plumbing — has
//! no dependency on either, mirroring how `fit_pipeline/store.rs` is kept
//! apart from `fit_pipeline/run.rs`.

use sqlx::PgPool;

use super::SkybotHit;

/// Records one finished Skybot search attempt for a lineage.
///
/// Always insert, even when `hits` is empty: a lineage with no row in
/// `skybot_queries` has never been searched, which the lineage page's
/// "last checked" display needs to tell apart from "searched and found
/// nothing" (see [`super::history::get_last_skybot_query`]).
///
/// # Arguments
///
/// * `pool` — the shared Postgres pool.
/// * `lineage_designation` — the lineage this search was run for.
/// * `radius_arcsec` — the conesearch radius the job actually used.
/// * `hits` — every match found across all query points, possibly empty.
///
/// # Return
///
/// `Ok(())` once the row is committed. `Err` with the failure serialized to a
/// display string if `hits` can't be encoded as JSON or the insert fails.
pub async fn insert_skybot_query(
    pool: &PgPool,
    lineage_designation: &str,
    radius_arcsec: f64,
    hits: &[SkybotHit],
) -> Result<(), String> {
    let hits_json = serde_json::to_value(hits).map_err(|e| e.to_string())?;

    sqlx::query(
        "INSERT INTO skybot_queries (lineage_designation, radius_arcsec, hits) \
         VALUES ($1, $2, $3)",
    )
    .bind(lineage_designation)
    .bind(radius_arcsec)
    .bind(hits_json)
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;

    Ok(())
}
