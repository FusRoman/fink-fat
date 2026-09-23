//! Reading back the most recent persisted Skybot search for a lineage.
//!
//! Mirrors [`crate::orbit_fit::history`]: a client-safe summary type (no
//! `chrono` on the wire, matching `OrbitFitSummary::fitted_at: String`) plus
//! one `#[server]` query function.

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

use super::SkybotHit;

/// The most recent Skybot search attempt recorded for a lineage, as returned
/// by [`get_last_skybot_query`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SkybotQueryRecord {
    /// RFC 3339 timestamp of the attempt, for display.
    pub queried_at: String,
    /// Days elapsed between `queried_at` and the moment this record was
    /// fetched, computed server-side by [`elapsed_days`] so the client never
    /// needs its own date-math — it just formats a number.
    pub delta_days: f64,
    pub radius_arcsec: f64,
    /// Every match found by that attempt; empty if none were.
    pub hits: Vec<SkybotHit>,
}

/// Days between `queried_at` and `now`, negative if `queried_at` is somehow
/// in the future (clock skew) rather than clamped, so a caller can decide
/// how to display that case instead of it being silently hidden here.
///
/// # Arguments
///
/// * `queried_at` — when the search was recorded.
/// * `now` — the reference instant to measure against.
///
/// # Return
///
/// The elapsed time in fractional days.
#[cfg(feature = "server")]
fn elapsed_days(
    queried_at: chrono::DateTime<chrono::Utc>,
    now: chrono::DateTime<chrono::Utc>,
) -> f64 {
    now.signed_duration_since(queried_at).num_seconds() as f64 / 86_400.0
}

/// Fetches the most recent `skybot_queries` row for `lineage_designation`, if
/// the lineage has ever been searched.
///
/// # Arguments
///
/// * `lineage_designation` — the lineage to look up.
///
/// # Return
///
/// `Ok(None)` if the lineage has no recorded search yet, `Ok(Some(_))` with
/// the latest attempt otherwise. `Err` if the query fails or a stored `hits`
/// payload can't be decoded.
#[server]
pub async fn get_last_skybot_query(
    lineage_designation: String,
) -> Result<Option<SkybotQueryRecord>, ServerFnError> {
    use crate::get_pool;

    #[cfg_attr(feature = "server", derive(sqlx::FromRow))]
    struct Row {
        queried_at: chrono::DateTime<chrono::Utc>,
        radius_arcsec: f64,
        hits: serde_json::Value,
    }

    let pool = get_pool().await;
    let row: Option<Row> = sqlx::query_as(
        "SELECT queried_at, radius_arcsec, hits FROM skybot_queries \
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
    let hits: Vec<SkybotHit> =
        serde_json::from_value(row.hits).map_err(|e| ServerFnError::new(e.to_string()))?;

    Ok(Some(SkybotQueryRecord {
        queried_at: row.queried_at.to_rfc3339(),
        delta_days: elapsed_days(row.queried_at, chrono::Utc::now()),
        radius_arcsec: row.radius_arcsec,
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
