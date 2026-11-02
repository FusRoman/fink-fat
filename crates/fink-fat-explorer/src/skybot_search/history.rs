//! Reading back a lineage's Skybot status from `skybot_obs_status` — the
//! same table both the per-lineage search ([`super::run`]) and the bulk
//! sweep (`crate::bulk_skybot`) write to, aggregated here across every
//! observation belonging to the lineage.

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

use super::SkybotHit;

/// A lineage's aggregated Skybot status, as returned by
/// [`get_last_skybot_query`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SkybotQueryRecord {
    /// RFC 3339 timestamp of the most recently checked observation, for
    /// display.
    pub queried_at: String,
    /// Days elapsed between `queried_at` and the moment this record was
    /// fetched, computed server-side by [`elapsed_days`] so the client never
    /// needs its own date-math — it just formats a number.
    pub delta_days: f64,
    /// The radius used by whichever observation was checked most recently —
    /// a lineage's observations can in principle have been checked at
    /// different radii across separate searches/sweeps, so this is a
    /// "best current guess", not a guarantee every hit below used it.
    pub radius_arcsec: f64,
    /// Every match found across every checked observation of this lineage;
    /// empty if none were.
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

/// Folds one more checked observation's row into a lineage's running
/// aggregate — the most recent `checked_at`/`radius_arcsec` win, and hits
/// accumulate across every observation. Factored out as its own pure
/// function (rather than inlined in [`get_last_skybot_query`]'s loop) so the
/// aggregation rule is unit-testable without a database.
///
/// # Arguments
///
/// * `existing` — the lineage's aggregate so far, `None` for its first row.
/// * `checked_at`, `radius_arcsec`, `hits` — one `skybot_obs_status` row's
///   data.
///
/// # Return
///
/// The updated aggregate.
#[cfg(feature = "server")]
fn fold_observation_status(
    existing: Option<SkybotQueryRecord>,
    checked_at: chrono::DateTime<chrono::Utc>,
    radius_arcsec: f64,
    hits: Vec<SkybotHit>,
) -> SkybotQueryRecord {
    match existing {
        None => SkybotQueryRecord {
            queried_at: checked_at.to_rfc3339(),
            delta_days: 0.0, // recomputed by the caller once `now` is known
            radius_arcsec,
            hits,
        },
        Some(mut record) => {
            let is_more_recent = chrono::DateTime::parse_from_rfc3339(&record.queried_at)
                .map(|prev| checked_at > prev)
                .unwrap_or(true);
            if is_more_recent {
                record.queried_at = checked_at.to_rfc3339();
                record.radius_arcsec = radius_arcsec;
            }
            record.hits.extend(hits);
            record
        }
    }
}

/// Fetches `lineage_designation`'s aggregated Skybot status: every
/// `skybot_obs_status` row belonging to any of its observations, folded
/// into one record via [`fold_observation_status`].
///
/// # Arguments
///
/// * `lineage_designation` — the lineage to look up.
///
/// # Return
///
/// `Ok(None)` if none of the lineage's observations have ever been checked,
/// `Ok(Some(_))` otherwise. `Err` if the query fails or a stored `hits`
/// payload can't be decoded.
#[server]
pub async fn get_last_skybot_query(
    lineage_designation: String,
) -> Result<Option<SkybotQueryRecord>, ServerFnError> {
    use crate::get_pool;

    #[cfg_attr(feature = "server", derive(sqlx::FromRow))]
    struct Row {
        checked_at: chrono::DateTime<chrono::Utc>,
        radius_arcsec: f64,
        hits: serde_json::Value,
    }

    let pool = get_pool().await;
    let rows: Vec<Row> = sqlx::query_as(
        "SELECT checked_at, radius_arcsec, hits FROM skybot_obs_status \
         WHERE lineage_designation = $1",
    )
    .bind(&lineage_designation)
    .fetch_all(pool)
    .await
    .map_err(|e| ServerFnError::new(e.to_string()))?;

    let mut record: Option<SkybotQueryRecord> = None;
    for row in rows {
        let hits: Vec<SkybotHit> =
            serde_json::from_value(row.hits).map_err(|e| ServerFnError::new(e.to_string()))?;
        record = Some(fold_observation_status(
            record,
            row.checked_at,
            row.radius_arcsec,
            hits,
        ));
    }

    Ok(record.map(|mut record| {
        let queried_at = chrono::DateTime::parse_from_rfc3339(&record.queried_at)
            .expect("just serialized via to_rfc3339")
            .with_timezone(&chrono::Utc);
        record.delta_days = elapsed_days(queried_at, chrono::Utc::now());
        record
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

    fn hit(name: &str) -> SkybotHit {
        SkybotHit {
            source_index: 0,
            name: name.to_string(),
            class: "Asteroid".to_string(),
            ra_deg: 0.0,
            dec_deg: 0.0,
            vmag: None,
            err_arcsec: None,
            geocentric_distance_au: None,
            heliocentric_distance_au: None,
            ssodnet_url: None,
            separation_arcsec: 0.0,
        }
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

    #[test]
    fn fold_observation_status_starts_from_the_first_row() {
        let record = fold_observation_status(None, at(10, 1, 1, 0), 5.0, vec![hit("Ceres")]);
        assert_eq!(record.hits.len(), 1);
        assert_eq!(record.radius_arcsec, 5.0);
    }

    #[test]
    fn fold_observation_status_keeps_the_most_recent_timestamp_and_radius() {
        let first = fold_observation_status(None, at(2026, 1, 1, 10), 5.0, vec![]);
        let older = fold_observation_status(Some(first), at(2026, 1, 1, 8), 10.0, vec![]);
        // An older row folded in afterwards must not roll back queried_at/radius.
        assert_eq!(older.queried_at, at(2026, 1, 1, 10).to_rfc3339());
        assert_eq!(older.radius_arcsec, 5.0);

        let newer = fold_observation_status(Some(older), at(2026, 1, 1, 12), 20.0, vec![]);
        assert_eq!(newer.queried_at, at(2026, 1, 1, 12).to_rfc3339());
        assert_eq!(newer.radius_arcsec, 20.0);
    }

    #[test]
    fn fold_observation_status_accumulates_hits_across_rows() {
        let first = fold_observation_status(None, at(2026, 1, 1, 10), 5.0, vec![hit("Ceres")]);
        let merged =
            fold_observation_status(Some(first), at(2026, 1, 1, 11), 5.0, vec![hit("Vesta")]);
        let names: Vec<&str> = merged.hits.iter().map(|h| h.name.as_str()).collect();
        assert_eq!(names, vec!["Ceres", "Vesta"]);
    }
}
