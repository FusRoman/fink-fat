//! Reading back the most recent persisted CND check for a lineage.
//!
//! Mirrors `crate::skybot_search::history`: a client-safe summary type (no
//! `chrono` on the wire) plus one `#[server]` query function.

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

use super::CndHit;

/// The most recent CND check attempt recorded for a lineage, as returned by
/// [`get_last_cnd_query`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CndQueryRecord {
    /// RFC 3339 timestamp of the attempt, for display.
    pub queried_at: String,
    /// Days elapsed between `queried_at` and the moment this record was
    /// fetched, computed server-side by [`elapsed_days`] so the client never
    /// needs its own date-math.
    pub delta_days: f64,
    pub time_separation_s: f64,
    pub angle_separation_arcsec: f64,
    /// Every match found by that attempt; empty if none were.
    pub hits: Vec<CndHit>,
}

/// Days between `queried_at` and `now`, negative if `queried_at` is somehow
/// in the future (clock skew) rather than clamped — same rationale as
/// `skybot_search::history`'s identical helper: let the caller decide how to
/// display that case instead of hiding it here.
#[cfg(feature = "server")]
fn elapsed_days(
    queried_at: chrono::DateTime<chrono::Utc>,
    now: chrono::DateTime<chrono::Utc>,
) -> f64 {
    now.signed_duration_since(queried_at).num_seconds() as f64 / 86_400.0
}

/// Fetches the most recent `cnd_queries` row for `lineage_designation`, if
/// the lineage has ever been checked.
///
/// # Arguments
///
/// * `lineage_designation` — the lineage to look up.
///
/// # Return
///
/// `Ok(None)` if the lineage has no recorded check yet, `Ok(Some(_))` with
/// the latest attempt otherwise. `Err` if the query fails or a stored `hits`
/// payload can't be decoded.
#[server]
pub async fn get_last_cnd_query(
    lineage_designation: String,
) -> Result<Option<CndQueryRecord>, ServerFnError> {
    use crate::get_pool;

    #[cfg_attr(feature = "server", derive(sqlx::FromRow))]
    struct Row {
        queried_at: chrono::DateTime<chrono::Utc>,
        time_separation_s: f64,
        angle_separation_arcsec: f64,
        hits: serde_json::Value,
    }

    let pool = get_pool().await;
    let row: Option<Row> = sqlx::query_as(
        "SELECT queried_at, time_separation_s, angle_separation_arcsec, hits FROM cnd_queries \
         WHERE lineage_designation = $1 \
         ORDER BY queried_at DESC \
         LIMIT 1",
    )
    .bind(&lineage_designation)
    .fetch_optional(pool)
    .await
    .map_err(|e| ServerFnError::new(e.to_string()))?;

    let Some(row) = row else {
        return Ok(None);
    };
    let hits: Vec<CndHit> =
        serde_json::from_value(row.hits).map_err(|e| ServerFnError::new(e.to_string()))?;

    Ok(Some(CndQueryRecord {
        queried_at: row.queried_at.to_rfc3339(),
        delta_days: elapsed_days(row.queried_at, chrono::Utc::now()),
        time_separation_s: row.time_separation_s,
        angle_separation_arcsec: row.angle_separation_arcsec,
        hits,
    }))
}

#[cfg(all(test, feature = "server"))]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn at(year: i32, month: u32, day: u32, hour: u32) -> chrono::DateTime<chrono::Utc> {
        chrono::Utc
            .with_ymd_and_hms(year, month, day, hour, 0, 0)
            .single()
            .expect("valid test timestamp")
    }

    #[test]
    fn elapsed_days_is_zero_for_the_same_instant() {
        let t = at(2026, 9, 23, 12);
        assert_eq!(elapsed_days(t, t), 0.0);
    }

    #[test]
    fn elapsed_days_counts_whole_and_partial_days() {
        let queried_at = at(2026, 9, 20, 0);
        let now = at(2026, 9, 23, 12);
        assert_eq!(elapsed_days(queried_at, now), 3.5);
    }

    #[test]
    fn elapsed_days_is_negative_when_queried_at_is_in_the_future() {
        let queried_at = at(2026, 9, 24, 0);
        let now = at(2026, 9, 23, 0);
        assert_eq!(elapsed_days(queried_at, now), -1.0);
    }
}
