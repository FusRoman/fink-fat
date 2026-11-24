//! Reverse search: from an alert identifier to the lineages containing it.

use dioxus::fullstack::Json;
use dioxus::prelude::*;

#[cfg(feature = "server")]
use super::types::LineageMatch;
use super::types::{BatchReverseSearchResponse, ReverseSearchResponse};

/// Path of the reverse-search route, as documented on the `/api-docs` page.
///
/// The `#[get]` macro below needs a string literal, so the two are kept equal
/// by the `route_path_matches_the_declared_route` test.
pub const REVERSE_SEARCH_PATH: &str = "/api/v1/alerts/{object_id}/lineages";

/// Path of the batch reverse-search route, as documented on the `/api-docs`
/// page. Kept equal to the `#[post]` literal by a test, like
/// [`REVERSE_SEARCH_PATH`].
pub const BATCH_REVERSE_SEARCH_PATH: &str = "/api/v1/alerts/lineages";

/// Maximum number of distinct alert identifiers accepted by one batch request.
pub const MAX_BATCH_SIZE: usize = 1000;

/// One `(alert, lineage, matching branch)` row returned by
/// [`MATCHING_BRANCHES_QUERY`].
#[cfg(feature = "server")]
#[derive(Debug, Clone, sqlx::FromRow)]
struct MatchRow {
    object_id: String,
    lineage_id: i64,
    lineage_designation: String,
    branch_id: i64,
    best_branch_id: i64,
}

/// Alert identifiers that exist in `observations` (`$1` = `text[]` of
/// `observations.object_id`).
#[cfg(feature = "server")]
const KNOWN_OBJECTS_QUERY: &str =
    "SELECT DISTINCT object_id FROM observations WHERE object_id = ANY($1)";

/// Branches containing each alert (`$1` = `text[]` of `observations.object_id`),
/// each with the best branch of its lineage. The best branch uses the same
/// ordering as
/// [`resolve_best_branch_id`](crate::orbit_fit::run::resolve_best_branch_id):
/// `NaN`/`±Infinity` LLRs count as 0, ties broken by lowest `branch_id`.
#[cfg(feature = "server")]
const MATCHING_BRANCHES_QUERY: &str = "
    SELECT m.object_id, b.lineage_id, b.lineage_designation, b.branch_id,
           best.branch_id AS best_branch_id
    FROM (
        SELECT DISTINCT o.object_id, bo.branch_id
        FROM observations o
        JOIN branch_observations bo ON bo.obs_id = o.id
        WHERE o.object_id = ANY($1)
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

/// Groups per-branch rows of one alert into one [`LineageMatch`] per lineage.
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

/// Removes duplicates from the requested identifiers, keeping first-seen order.
///
/// # Arguments
///
/// * `object_ids` - identifiers as received.
///
/// # Return
///
/// The distinct identifiers in request order.
#[cfg(feature = "server")]
fn dedup_preserving_order(object_ids: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    object_ids
        .into_iter()
        .filter(|id| seen.insert(id.clone()))
        .collect()
}

/// Assembles the batch response from the database results (pure).
///
/// # Arguments
///
/// * `object_ids` - the distinct requested identifiers, in request order.
/// * `known` - those of them that exist in `observations`.
/// * `rows` - the matching-branch rows of the known alerts.
///
/// # Return
///
/// Known alerts (with possibly no lineage) in request order, and the unknown
/// identifiers in request order.
#[cfg(feature = "server")]
fn assemble_batch(
    object_ids: &[String],
    known: &std::collections::HashSet<String>,
    rows: Vec<MatchRow>,
) -> BatchReverseSearchResponse {
    let mut rows_by_object = std::collections::HashMap::<String, Vec<MatchRow>>::new();
    for row in rows {
        rows_by_object
            .entry(row.object_id.clone())
            .or_default()
            .push(row);
    }
    let mut response = BatchReverseSearchResponse {
        results: Vec::new(),
        unknown_object_ids: Vec::new(),
    };
    for object_id in object_ids {
        if known.contains(object_id) {
            let rows = rows_by_object.remove(object_id).unwrap_or_default();
            response.results.push(ReverseSearchResponse {
                object_id: object_id.clone(),
                lineages: group_by_lineage(rows),
            });
        } else {
            response.unknown_object_ids.push(object_id.clone());
        }
    }
    response
}

/// Finds the lineages containing each of several alerts, with two queries
/// whatever the number of alerts.
///
/// # Arguments
///
/// * `pool` - Postgres connection pool.
/// * `object_ids` - alert identifiers (`observations.object_id`); duplicates
///   are ignored.
///
/// # Return
///
/// The [`BatchReverseSearchResponse`], or
/// [`ApiError::BadRequest`](super::error::ApiError::BadRequest) if the list is
/// empty or has more than [`MAX_BATCH_SIZE`] distinct identifiers, or a
/// database error.
#[cfg(feature = "server")]
pub(crate) async fn fetch_lineages_for_objects(
    pool: &sqlx::PgPool,
    object_ids: Vec<String>,
) -> Result<BatchReverseSearchResponse, super::error::ApiError> {
    use super::error::ApiError;

    let object_ids = dedup_preserving_order(object_ids);
    if object_ids.is_empty() {
        return Err(ApiError::BadRequest(
            "`object_ids` must not be empty".into(),
        ));
    }
    if object_ids.len() > MAX_BATCH_SIZE {
        return Err(ApiError::BadRequest(format!(
            "at most {MAX_BATCH_SIZE} object_ids per request, got {}",
            object_ids.len()
        )));
    }
    let known: std::collections::HashSet<String> = sqlx::query_scalar(KNOWN_OBJECTS_QUERY)
        .bind(&object_ids)
        .fetch_all(pool)
        .await?
        .into_iter()
        .collect();
    let rows: Vec<MatchRow> = sqlx::query_as(MATCHING_BRANCHES_QUERY)
        .bind(&object_ids)
        .fetch_all(pool)
        .await?;
    Ok(assemble_batch(&object_ids, &known, rows))
}

/// Finds the lineages containing an alert (a batch of one).
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
    let batch = fetch_lineages_for_objects(pool, vec![object_id.to_string()]).await?;
    batch
        .results
        .into_iter()
        .next()
        .ok_or_else(|| super::error::ApiError::NotFound(object_id.to_string()))
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

/// `POST /api/v1/alerts/lineages` — reverse search of several alerts at once.
///
/// The JSON body is `{"object_ids": [...]}` (see
/// [`BatchReverseSearchRequest`](super::types::BatchReverseSearchRequest)).
///
/// # Arguments
///
/// * `object_ids` - alert identifiers (`observations.object_id`), at most
///   [`MAX_BATCH_SIZE`] distinct ones.
///
/// # Return
///
/// JSON [`BatchReverseSearchResponse`]. Unknown identifiers do not fail the
/// request: they are listed in `unknown_object_ids`. HTTP 400 if the list is
/// empty or too large, 500 on database errors.
#[post("/api/v1/alerts/lineages")]
pub async fn reverse_search_alerts(
    object_ids: Vec<String>,
) -> Result<Json<BatchReverseSearchResponse>, ServerFnError> {
    let pool = crate::get_pool().await;
    let response = fetch_lineages_for_objects(pool, object_ids).await?;
    Ok(Json(response))
}

#[cfg(test)]
mod path_tests {
    use super::{BATCH_REVERSE_SEARCH_PATH, REVERSE_SEARCH_PATH};

    #[test]
    fn route_path_matches_the_declared_route() {
        let declared = format!("#[get(\"{REVERSE_SEARCH_PATH}\")]");
        assert!(include_str!("reverse_search.rs").contains(&declared));
    }

    #[test]
    fn batch_route_path_matches_the_declared_route() {
        let declared = format!("#[post(\"{BATCH_REVERSE_SEARCH_PATH}\")]");
        assert!(include_str!("reverse_search.rs").contains(&declared));
    }
}

#[cfg(all(test, feature = "server"))]
mod tests {
    use super::*;

    fn row(lineage_id: i64, branch_id: i64, best: i64) -> MatchRow {
        object_row("a", lineage_id, branch_id, best)
    }

    fn object_row(object_id: &str, lineage_id: i64, branch_id: i64, best: i64) -> MatchRow {
        MatchRow {
            object_id: object_id.to_string(),
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

    fn ids(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn dedup_keeps_first_seen_order() {
        assert_eq!(
            dedup_preserving_order(ids(&["b", "a", "b", "c", "a"])),
            ids(&["b", "a", "c"])
        );
    }

    #[test]
    fn batch_splits_known_unknown_and_lineage_less_alerts() {
        let requested = ids(&["a", "ghost", "b", "c"]);
        let known: std::collections::HashSet<String> = ids(&["a", "b", "c"]).into_iter().collect();
        let rows = vec![
            object_row("c", 2, 20, 20),
            object_row("a", 1, 10, 10),
            object_row("a", 3, 30, 30),
        ];
        let out = assemble_batch(&requested, &known, rows);
        assert_eq!(out.unknown_object_ids, ids(&["ghost"]));
        let order: Vec<&str> = out.results.iter().map(|r| r.object_id.as_str()).collect();
        assert_eq!(order, vec!["a", "b", "c"]);
        assert_eq!(out.results[0].lineages.len(), 2);
        assert!(out.results[1].lineages.is_empty());
        assert_eq!(out.results[2].lineages[0].lineage_id, 2);
    }
}
