//! Reverse search: from an alert identifier to the lineages containing it.

use dioxus::fullstack::Json;
use dioxus::prelude::*;

#[cfg(feature = "server")]
use super::types::LineageMatch;
use super::types::ReverseSearchResponse;

/// One `(lineage, matching branch)` row returned by [`MATCHING_BRANCHES_QUERY`].
#[cfg(feature = "server")]
#[derive(Debug, Clone, sqlx::FromRow)]
struct MatchRow {
    lineage_id: i64,
    lineage_designation: String,
    branch_id: i64,
    best_branch_id: i64,
}

/// Branches containing the alert (`$1` = `observations.object_id`), each with
/// the best branch of its lineage. The best branch uses the same ordering as
/// [`resolve_best_branch_id`](crate::orbit_fit::run::resolve_best_branch_id):
/// `NaN`/`±Infinity` LLRs count as 0, ties broken by lowest `branch_id`.
#[cfg(feature = "server")]
const MATCHING_BRANCHES_QUERY: &str = "
    SELECT b.lineage_id, b.lineage_designation, b.branch_id, best.branch_id AS best_branch_id
    FROM (
        SELECT DISTINCT bo.branch_id
        FROM observations o
        JOIN branch_observations bo ON bo.obs_id = o.id
        WHERE o.object_id = $1
    ) m
    JOIN branches b ON b.branch_id = m.branch_id
    CROSS JOIN LATERAL (
        SELECT b2.branch_id
        FROM branches b2
        WHERE b2.lineage_designation = b.lineage_designation
        ORDER BY (
            CASE
                WHEN b2.cumulative_llr = 'NaN'::double precision THEN 0
                WHEN b2.cumulative_llr = 'Infinity'::double precision THEN 0
                WHEN b2.cumulative_llr = '-Infinity'::double precision THEN 0
                ELSE b2.cumulative_llr
            END
        ) DESC, b2.branch_id
        LIMIT 1
    ) best";

/// Groups per-branch rows into one [`LineageMatch`] per lineage.
///
/// # Arguments
///
/// * `rows` - one row per branch containing the alert, in any order.
///
/// # Return
///
/// Lineages ordered by `lineage_id`, with their matching branch ids sorted
/// ascending and deduplicated.
#[cfg(feature = "server")]
fn group_by_lineage(rows: Vec<MatchRow>) -> Vec<LineageMatch> {
    let mut grouped = std::collections::BTreeMap::<i64, LineageMatch>::new();
    for row in rows {
        let entry = grouped
            .entry(row.lineage_id)
            .or_insert_with(|| LineageMatch {
                lineage_id: row.lineage_id,
                url: format!("/lineage/{}", row.lineage_designation),
                lineage_designation: row.lineage_designation.clone(),
                best_branch_id: row.best_branch_id,
                matching_branch_ids: Vec::new(),
            });
        entry.matching_branch_ids.push(row.branch_id);
    }
    grouped
        .into_values()
        .map(|mut m| {
            m.matching_branch_ids.sort_unstable();
            m.matching_branch_ids.dedup();
            m
        })
        .collect()
}

/// Finds the lineages containing an alert.
///
/// # Arguments
///
/// * `pool` - Postgres connection pool.
/// * `object_id` - alert identifier (`observations.object_id`).
///
/// # Return
///
/// The matching lineages (possibly none if the alert is not attached to any
/// lineage), or [`ApiError::NotFound`](super::error::ApiError::NotFound) when
/// no observation has this `object_id`, or a database error.
#[cfg(feature = "server")]
pub(crate) async fn fetch_lineages_for_object(
    pool: &sqlx::PgPool,
    object_id: &str,
) -> Result<ReverseSearchResponse, super::error::ApiError> {
    let exists: Option<(i32,)> =
        sqlx::query_as("SELECT 1 FROM observations WHERE object_id = $1 LIMIT 1")
            .bind(object_id)
            .fetch_optional(pool)
            .await?;
    if exists.is_none() {
        return Err(super::error::ApiError::NotFound(object_id.to_string()));
    }
    let rows: Vec<MatchRow> = sqlx::query_as(MATCHING_BRANCHES_QUERY)
        .bind(object_id)
        .fetch_all(pool)
        .await?;
    Ok(ReverseSearchResponse {
        object_id: object_id.to_string(),
        lineages: group_by_lineage(rows),
    })
}

/// `GET /api/v1/alerts/{object_id}/lineages` — reverse search of an alert.
///
/// # Arguments
///
/// * `object_id` - alert identifier (`observations.object_id`).
///
/// # Return
///
/// JSON [`ReverseSearchResponse`]. HTTP 404 if the alert is unknown, 200 with
/// an empty `lineages` list if it belongs to no lineage, 500 on database
/// errors.
#[get("/api/v1/alerts/{object_id}/lineages")]
pub async fn reverse_search_alert(
    object_id: String,
) -> Result<Json<ReverseSearchResponse>, ServerFnError> {
    let pool = crate::get_pool().await;
    let response = fetch_lineages_for_object(pool, &object_id).await?;
    Ok(Json(response))
}

#[cfg(all(test, feature = "server"))]
mod tests {
    use super::*;

    fn row(lineage_id: i64, branch_id: i64, best: i64) -> MatchRow {
        MatchRow {
            lineage_id,
            lineage_designation: format!("L{lineage_id}"),
            branch_id,
            best_branch_id: best,
        }
    }

    #[test]
    fn groups_branches_of_the_same_lineage() {
        let out = group_by_lineage(vec![row(2, 30, 10), row(1, 11, 10), row(1, 10, 10)]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].lineage_id, 1);
        assert_eq!(out[0].matching_branch_ids, vec![10, 11]);
        assert_eq!(out[0].best_branch_id, 10);
        assert_eq!(out[0].url, "/lineage/L1");
        assert_eq!(out[1].matching_branch_ids, vec![30]);
    }

    #[test]
    fn deduplicates_and_handles_empty_input() {
        let out = group_by_lineage(vec![row(1, 5, 5), row(1, 5, 5)]);
        assert_eq!(out[0].matching_branch_ids, vec![5]);
        assert!(group_by_lineage(vec![]).is_empty());
    }
}
