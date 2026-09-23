//! Fetches every lineage's latest Skybot/CND cross-match status, straight
//! from Postgres — deliberately **not** from `crate::homepage::snapshot`,
//! whose own doc comment scopes it to data that doesn't change while the
//! server runs (it excludes `orbit_fits` for exactly this reason: the
//! explorer writes to it live). `skybot_queries`/`cnd_queries` are the same
//! kind of table — written by user-triggered searches at arbitrary times,
//! with no `fink-fat convert`-tied invalidation hook — so this follows
//! `bulk_cnd::run::CONVERGED_BRANCH_QUERY`'s precedent of a small,
//! purpose-built fresh query instead.

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

/// One lineage's cross-match status, as shown on the dashboard. Only
/// lineages with a positive match in at least one service are ever
/// returned by [`get_cross_match_status`] — see its doc comment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CrossMatchStatusRow {
    pub lineage_designation: String,
    /// `None` if the lineage has never been searched with Skybot.
    pub skybot_queried_at: Option<String>,
    /// Distinct matched object names (deduped via
    /// [`crate::skybot_search::dedup_hits_by_name`]), empty if never
    /// searched or searched with no match.
    pub skybot_object_names: Vec<String>,
    /// `None` if the lineage has never been checked against MPC.
    pub cnd_queried_at: Option<String>,
    /// How many of the lineage's own observations had at least one MPC
    /// near-duplicate — CND hits don't carry a stable object identity worth
    /// surfacing (unlike Skybot's object names), so a count is all this
    /// shows.
    pub cnd_hit_count: usize,
}

// Both methods are only called from `get_cross_match_status`'s `#[server]`
// body below, which the macro elides entirely on the wasm32 client build —
// leaving these genuinely unused on that target, hence the explicit gate
// (same pattern as e.g. `skybot_search::history`'s `elapsed_days`).
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
/// Either underlying query failing, or a stored `hits` payload failing to
/// decode, as a `ServerFnError`.
#[server]
pub async fn get_cross_match_status() -> Result<Vec<CrossMatchStatusRow>, ServerFnError> {
    use crate::cnd_search::CndHit;
    use crate::get_pool;
    use crate::skybot_search::{dedup_hits_by_name, SkybotHit};
    use std::collections::HashMap;

    #[cfg_attr(feature = "server", derive(sqlx::FromRow))]
    struct Row {
        lineage_designation: String,
        queried_at: chrono::DateTime<chrono::Utc>,
        hits: serde_json::Value,
    }

    let pool = get_pool().await;

    let skybot_rows: Vec<Row> = sqlx::query_as(
        "SELECT DISTINCT ON (lineage_designation) lineage_designation, queried_at, hits \
         FROM skybot_queries \
         ORDER BY lineage_designation, queried_at DESC",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| ServerFnError::new(e.to_string()))?;

    let cnd_rows: Vec<Row> = sqlx::query_as(
        "SELECT DISTINCT ON (lineage_designation) lineage_designation, queried_at, hits \
         FROM cnd_queries \
         ORDER BY lineage_designation, queried_at DESC",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| ServerFnError::new(e.to_string()))?;

    let mut rows: HashMap<String, CrossMatchStatusRow> = HashMap::new();

    for row in skybot_rows {
        let hits: Vec<SkybotHit> =
            serde_json::from_value(row.hits).map_err(|e| ServerFnError::new(e.to_string()))?;
        let object_names: Vec<String> = dedup_hits_by_name(&hits)
            .into_iter()
            .map(|hit| hit.name)
            .collect();
        let entry = rows
            .entry(row.lineage_designation.clone())
            .or_insert_with(|| CrossMatchStatusRow::empty(row.lineage_designation));
        entry.skybot_queried_at = Some(row.queried_at.to_rfc3339());
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
