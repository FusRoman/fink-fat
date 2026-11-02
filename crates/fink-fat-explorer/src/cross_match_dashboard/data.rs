//! Fetches every lineage's latest Skybot/CND cross-match status, straight
//! from Postgres — deliberately **not** from `crate::homepage::snapshot`,
//! whose own doc comment scopes it to data that doesn't change while the
//! server runs (it excludes `orbit_fits` for exactly this reason: the
//! explorer writes to it live). `skybot_obs_status`/`cnd_queries` are the
//! same kind of table — written by user-triggered or background searches
//! at arbitrary times, with no `fink-fat convert`-tied invalidation hook —
//! so this follows `bulk_cnd::run::CONVERGED_BRANCH_QUERY`'s precedent of a
//! small, purpose-built fresh query instead.

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

/// One lineage's cross-match status, as shown on the dashboard. Only
/// lineages with a positive match in at least one service are ever
/// returned by [`get_cross_match_status`] — see its doc comment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CrossMatchStatusRow {
    pub lineage_designation: String,
    /// `None` if none of the lineage's observations have ever been checked
    /// with Skybot, by either the per-lineage search or the bulk sweep —
    /// both write to `skybot_obs_status`, so there's a single answer.
    pub skybot_queried_at: Option<String>,
    /// Distinct matched object names (deduped via
    /// [`crate::skybot_search::dedup_hits_by_name`]), pooled across every
    /// checked observation of the lineage; empty if never checked or
    /// checked with no match.
    pub skybot_object_names: Vec<String>,
    /// `None` if the lineage has never been checked against MPC.
    pub cnd_queried_at: Option<String>,
    /// How many of the lineage's own observations had at least one MPC
    /// near-duplicate — CND hits don't carry a stable object identity worth
    /// surfacing (unlike Skybot's object names), so a count is all this
    /// shows.
    pub cnd_hit_count: usize,
}

// Only called from `get_cross_match_status`'s `#[server]` body below, which
// the macro elides entirely on the wasm32 client build — leaving these
// genuinely unused on that target, hence the explicit gate (same pattern as
// e.g. `skybot_search::history`'s `elapsed_days`).
#[cfg(feature = "server")]
impl CrossMatchStatusRow {
    fn empty(lineage_designation: String) -> Self {
        Self {
            lineage_designation,
            skybot_queried_at: None,
            skybot_object_names: Vec::new(),
            cnd_queried_at: None,
            cnd_hit_count: 0,
        }
    }

    fn has_a_match(&self) -> bool {
        !self.skybot_object_names.is_empty() || self.cnd_hit_count > 0
    }
}

/// One lineage's pooled Skybot status, folded from its `skybot_obs_status`
/// rows (one per checked observation) — an intermediate accumulator, not
/// part of the public API, kept as its own small type purely so
/// [`merge_skybot_status`] (the actual fold rule: most recent timestamp
/// wins, object names union) is a pure function testable without a
/// database.
#[cfg(feature = "server")]
#[derive(Clone, Debug, PartialEq)]
struct SkybotStatus {
    checked_at: chrono::DateTime<chrono::Utc>,
    object_names: std::collections::HashSet<String>,
}

/// Folds one more checked observation (from `skybot_obs_status`) into
/// `existing`, keeping the most recent `checked_at` and the union of object
/// names.
///
/// # Arguments
///
/// * `existing` — the lineage's status so far, `None` if this is its first
///   contribution.
/// * `checked_at`, `object_names` — the new fact to fold in.
///
/// # Return
///
/// The updated status.
#[cfg(feature = "server")]
fn merge_skybot_status(
    existing: Option<SkybotStatus>,
    checked_at: chrono::DateTime<chrono::Utc>,
    object_names: impl IntoIterator<Item = String>,
) -> SkybotStatus {
    let mut status = existing.unwrap_or(SkybotStatus {
        checked_at,
        object_names: std::collections::HashSet::new(),
    });
    if checked_at > status.checked_at {
        status.checked_at = checked_at;
    }
    status.object_names.extend(object_names);
    status
}

/// Fetches every lineage's cross-match status, keeping only lineages with a
/// positive match in Skybot and/or CND — a lineage that was searched and
/// found nothing, or never searched at all, is left out entirely rather
/// than shown as a "no match" row, since the dashboard's whole point is
/// "which lineages are worth a second look".
///
/// # Return
///
/// Rows sorted by `lineage_designation`.
///
/// # Errors
///
/// Any underlying query failing, or a stored `hits` payload failing to
/// decode, as a `ServerFnError`.
#[server]
pub async fn get_cross_match_status() -> Result<Vec<CrossMatchStatusRow>, ServerFnError> {
    use crate::cnd_search::CndHit;
    use crate::get_pool;
    use crate::skybot_search::{dedup_hits_by_name, SkybotHit};
    use std::collections::HashMap;

    #[cfg_attr(feature = "server", derive(sqlx::FromRow))]
    struct SkybotRow {
        lineage_designation: String,
        checked_at: chrono::DateTime<chrono::Utc>,
        hits: serde_json::Value,
    }
    #[cfg_attr(feature = "server", derive(sqlx::FromRow))]
    struct CndRow {
        lineage_designation: String,
        queried_at: chrono::DateTime<chrono::Utc>,
        hits: serde_json::Value,
    }

    let pool = get_pool().await;

    let skybot_rows: Vec<SkybotRow> =
        sqlx::query_as("SELECT lineage_designation, checked_at, hits FROM skybot_obs_status")
            .fetch_all(pool)
            .await
            .map_err(|e| ServerFnError::new(e.to_string()))?;

    let cnd_rows: Vec<CndRow> = sqlx::query_as(
        "SELECT DISTINCT ON (lineage_designation) lineage_designation, queried_at, hits \
         FROM cnd_queries \
         ORDER BY lineage_designation, queried_at DESC",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| ServerFnError::new(e.to_string()))?;

    let mut skybot_status: HashMap<String, SkybotStatus> = HashMap::new();
    for row in skybot_rows {
        let hits: Vec<SkybotHit> =
            serde_json::from_value(row.hits).map_err(|e| ServerFnError::new(e.to_string()))?;
        let object_names = dedup_hits_by_name(&hits).into_iter().map(|hit| hit.name);
        let entry = skybot_status.remove(&row.lineage_designation);
        skybot_status.insert(
            row.lineage_designation,
            merge_skybot_status(entry, row.checked_at, object_names),
        );
    }

    let mut rows: HashMap<String, CrossMatchStatusRow> = HashMap::new();

    for (lineage_designation, status) in skybot_status {
        let mut object_names: Vec<String> = status.object_names.into_iter().collect();
        object_names.sort();
        let entry = rows
            .entry(lineage_designation.clone())
            .or_insert_with(|| CrossMatchStatusRow::empty(lineage_designation));
        entry.skybot_queried_at = Some(status.checked_at.to_rfc3339());
        entry.skybot_object_names = object_names;
    }

    for row in cnd_rows {
        let hits: Vec<CndHit> =
            serde_json::from_value(row.hits).map_err(|e| ServerFnError::new(e.to_string()))?;
        let entry = rows
            .entry(row.lineage_designation.clone())
            .or_insert_with(|| CrossMatchStatusRow::empty(row.lineage_designation));
        entry.cnd_queried_at = Some(row.queried_at.to_rfc3339());
        entry.cnd_hit_count = hits.len();
    }

    let mut rows: Vec<CrossMatchStatusRow> = rows
        .into_values()
        .filter(CrossMatchStatusRow::has_a_match)
        .collect();
    rows.sort_by(|a, b| a.lineage_designation.cmp(&b.lineage_designation));
    Ok(rows)
}

#[cfg(all(test, feature = "server"))]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn at(hour: u32) -> chrono::DateTime<chrono::Utc> {
        chrono::Utc
            .with_ymd_and_hms(2026, 9, 23, hour, 0, 0)
            .single()
            .expect("valid test timestamp")
    }

    #[test]
    fn merge_skybot_status_starts_from_nothing() {
        let status = merge_skybot_status(None, at(10), ["Ceres".to_string()]);
        assert_eq!(status.checked_at, at(10));
        assert!(status.object_names.contains("Ceres"));
    }

    #[test]
    fn merge_skybot_status_keeps_the_most_recent_timestamp() {
        let first = merge_skybot_status(None, at(10), []);
        let merged = merge_skybot_status(Some(first), at(8), []);
        // An older fact folded in afterwards must not roll the timestamp back.
        assert_eq!(merged.checked_at, at(10));

        let merged = merge_skybot_status(Some(merged), at(12), []);
        assert_eq!(merged.checked_at, at(12));
    }

    #[test]
    fn merge_skybot_status_unions_object_names() {
        let first = merge_skybot_status(None, at(10), ["Ceres".to_string()]);
        let merged = merge_skybot_status(
            Some(first),
            at(11),
            ["Vesta".to_string(), "Ceres".to_string()],
        );
        let mut names: Vec<&String> = merged.object_names.iter().collect();
        names.sort();
        assert_eq!(names, vec!["Ceres", "Vesta"]);
    }
}
