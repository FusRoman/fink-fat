//! Whether a lineage has an active cross-match hit (CND or Skybot), used by
//! [`crate::homepage::quality_tier`] to split its top tiers into "genuinely
//! novel" versus "matches a known object" (see
//! [`crate::homepage::quality_tier::assign_quality_tier`]'s `has_cross_match`
//! parameter).
//!
//! Neither service is read the same way, and this module deliberately keeps
//! that asymmetry rather than papering over it, since it's inherited from
//! how each service's own status page already reads its table:
//!
//! - **Skybot** (`skybot_obs_status`, one row per checked observation): hits
//!   accumulate across observations/checks and never clear, matching
//!   `skybot_search::history`'s existing fold semantics
//!   (`skybot_search::history::fold_observation_status`). A lineage counts
//!   as matched if *any* of its observations' `hits` array is non-empty.
//! - **CND** (`cnd_queries`, one row per check *attempt*, several per
//!   lineage over time): only the *most recent* attempt counts, matching
//!   `cross_match_dashboard::data::get_cross_match_status`'s existing
//!   "latest attempt is authoritative" semantics. A lineage re-checked with
//!   different parameters and no match today is not counted as matched,
//!   even if an older attempt once found something.

#[cfg(feature = "server")]
use sqlx::PgPool;

/// SQL predicate for "this `skybot_obs_status` row found at least one
/// match" — a plain non-empty JSONB array, since the table has no dedicated
/// count/boolean column.
#[cfg(feature = "server")]
const SKYBOT_HAS_HITS: &str = "jsonb_array_length(hits) > 0";

/// Whether `lineage_designation` has an active cross-match hit right now,
/// combining Skybot (accumulated across observations) and CND (latest
/// attempt only) — see the module doc for why the two differ.
///
/// # Arguments
///
/// * `pool` — the database connection pool.
/// * `lineage_designation` — the lineage to check.
///
/// # Return
///
/// `true` if either service currently reports a match for this lineage.
///
/// # Errors
///
/// The query failing.
#[cfg(feature = "server")]
pub async fn lineage_has_cross_match(
    pool: &PgPool,
    lineage_designation: &str,
) -> Result<bool, sqlx::Error> {
    let query = format!(
        "SELECT
            EXISTS (
                SELECT 1 FROM skybot_obs_status
                WHERE lineage_designation = $1 AND {SKYBOT_HAS_HITS}
            )
            OR COALESCE((
                SELECT {SKYBOT_HAS_HITS}
                FROM cnd_queries
                WHERE lineage_designation = $1
                ORDER BY queried_at DESC
                LIMIT 1
            ), false)"
    );

    sqlx::query_scalar(sqlx::AssertSqlSafe(query))
        .bind(lineage_designation)
        .fetch_one(pool)
        .await
}

/// Every lineage that currently has an active cross-match hit, for the
/// homepage snapshot's batch quality-tier computation — see
/// [`lineage_has_cross_match`] for the per-lineage equivalent and the exact
/// Skybot/CND semantics this reuses.
///
/// # Arguments
///
/// * `pool` — the database connection pool.
///
/// # Return
///
/// The set of matched lineages' designations.
///
/// # Errors
///
/// The query failing.
#[cfg(feature = "server")]
pub async fn build_cross_match_index(
    pool: &PgPool,
) -> Result<std::collections::HashSet<Box<str>>, sqlx::Error> {
    let query = format!(
        "SELECT DISTINCT lineage_designation FROM skybot_obs_status
         WHERE {SKYBOT_HAS_HITS}
         UNION
         SELECT lineage_designation FROM (
             SELECT DISTINCT ON (lineage_designation) lineage_designation, hits
             FROM cnd_queries
             ORDER BY lineage_designation, queried_at DESC
         ) latest_cnd
         WHERE {SKYBOT_HAS_HITS}"
    );

    let rows: Vec<(String,)> = sqlx::query_as(sqlx::AssertSqlSafe(query))
        .fetch_all(pool)
        .await?;

    Ok(rows
        .into_iter()
        .map(|(lineage_designation,)| lineage_designation.into_boxed_str())
        .collect())
}
