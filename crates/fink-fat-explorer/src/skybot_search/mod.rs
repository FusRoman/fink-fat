//! Background Skybot conesearch jobs for the lineage trajectory plot: query
//! IMCCE's Skybot web service
//! (<https://ssp.imcce.fr/webservices/skybot/>) once per observation epoch of
//! a lineage, rate-limited to a small number of concurrent connections (see
//! `run::MAX_CONCURRENT_REQUESTS`), and let the client poll for the growing
//! hit list while it runs — the same background-job/poll convention as
//! [`crate::orbit_fit`] and [`crate::bulk_orbit_fit`], rather than
//! SSE/WebSockets (unused anywhere else in this app).
//!
//! Request URL construction and response parsing are pure, unit-tested
//! functions in [`parsing`] and [`sexagesimal`]; only [`run`] touches the
//! network and the job registry.

/// Server-only: computing [`SkybotHit::separation_arcsec`] pulls in `photom`,
/// and sending the request pulls in `reqwest` — neither builds for `wasm32`.
#[cfg(feature = "server")]
pub mod parsing;
pub mod run;
pub mod sexagesimal;
pub mod status;

use serde::{Deserialize, Serialize};

pub use crate::orbit_fit::JobStatus;

/// One sky position + epoch to search around, built client-side from a
/// lineage's real observations (`ObservationRow::ra`/`::dec`/`::mjd_tt`,
/// converted to degrees).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SkybotQueryPoint {
    /// Index of the observation this point was built from, carried through
    /// to [`SkybotHit::source_index`] so a hit can be traced back to the
    /// trajectory point that produced it.
    pub source_index: usize,
    pub ra_deg: f64,
    pub dec_deg: f64,
    /// Modified Julian Date, TT scale — see
    /// [`parsing::mjd_tt_to_jd`] for how this becomes Skybot's `-ep`.
    pub mjd_tt: f64,
}

/// One object Skybot reported near a query point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SkybotHit {
    /// See [`SkybotQueryPoint::source_index`].
    pub source_index: usize,
    pub name: String,
    pub class: String,
    /// Skybot's own reported position for the object at the query epoch —
    /// not the query center — so the plotted marker shows whether a known
    /// object's ephemeris actually lines up with the observed track.
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub vmag: Option<f64>,
    pub err_arcsec: Option<f64>,
    /// Geocentric distance, astronomical units.
    pub geocentric_distance_au: Option<f64>,
    /// Heliocentric distance, astronomical units.
    pub heliocentric_distance_au: Option<f64>,
    /// Link into SSODNet for this object, when Skybot's response included
    /// one (`ssocard`, falling back to `quaero`).
    pub ssodnet_url: Option<String>,
    /// Great-circle distance (Vincenty formula, via `photom`) between this
    /// hit's own reported position and the real observation at
    /// [`SkybotQueryPoint::source_index`] that this point was queried
    /// around — computed server-side (see
    /// `parsing::raw_row_to_hit`) since `photom` isn't available client-side.
    pub separation_arcsec: f64,
}

/// Snapshot of a running/finished Skybot search job, returned to the client
/// by [`status::get_skybot_job_status`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SkybotJobView {
    pub status: JobStatus,
    pub total: usize,
    pub processed: usize,
    /// Every hit found so far, across all query points processed to date.
    /// The client redraws the trajectory plot with this on every poll,
    /// which is what makes matches appear incrementally rather than all at
    /// once when the job finishes.
    pub hits: Vec<SkybotHit>,
    pub logs: Vec<String>,
    pub error: Option<String>,
}

/// Server-side job entry — not sent to the client directly, [`SkybotJobView`]
/// (a plain snapshot) is what [`status::get_skybot_job_status`] returns.
#[cfg(feature = "server")]
#[derive(Clone, Debug)]
pub struct SkybotJob {
    pub status: JobStatus,
    pub total: usize,
    /// `Arc<AtomicUsize>` so per-point request tasks can report progress
    /// without taking the job-registry mutex more than once each, matching
    /// `bulk_orbit_fit::BulkOrbitFitJob`'s counters.
    pub processed: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    pub hits: Vec<SkybotHit>,
    pub logs: Vec<String>,
    pub error: Option<String>,
}

#[cfg(feature = "server")]
impl SkybotJob {
    pub fn new(total: usize) -> Self {
        Self {
            status: JobStatus::Running,
            total,
            processed: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            hits: Vec::new(),
            logs: Vec::new(),
            error: None,
        }
    }

    pub fn view(&self) -> SkybotJobView {
        use std::sync::atomic::Ordering;
        SkybotJobView {
            status: self.status,
            total: self.total,
            processed: self.processed.load(Ordering::Relaxed),
            hits: self.hits.clone(),
            logs: self.logs.clone(),
            error: self.error.clone(),
        }
    }
}

/// Minimum search radius accepted by [`run::start_skybot_search`], arcseconds.
pub const MIN_RADIUS_ARCSEC: f64 = 1.0;
/// Maximum search radius accepted by [`run::start_skybot_search`], arcseconds.
pub const MAX_RADIUS_ARCSEC: f64 = 60.0;

/// Clamps a requested search radius into
/// `[MIN_RADIUS_ARCSEC, MAX_RADIUS_ARCSEC]` — matches the UI slider's range,
/// and is re-applied server-side in case a tampered client call sends a
/// value outside it.
pub fn clamp_radius_arcsec(radius_arcsec: f64) -> f64 {
    radius_arcsec.clamp(MIN_RADIUS_ARCSEC, MAX_RADIUS_ARCSEC)
}

/// Deduplicates hits by object name, keeping the first occurrence (the
/// earliest trajectory point it was found near). Used by the SSODNet side
/// panel so an object detected near several points is listed once.
pub fn dedup_hits_by_name(hits: &[SkybotHit]) -> Vec<SkybotHit> {
    let mut seen = std::collections::HashSet::new();
    hits.iter()
        .filter(|hit| seen.insert(hit.name.clone()))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_radius_arcsec_clamps_below_and_above_range() {
        assert_eq!(clamp_radius_arcsec(0.1), MIN_RADIUS_ARCSEC);
        assert_eq!(clamp_radius_arcsec(100.0), MAX_RADIUS_ARCSEC);
        assert_eq!(clamp_radius_arcsec(15.0), 15.0);
    }

    fn hit(source_index: usize, name: &str) -> SkybotHit {
        SkybotHit {
            source_index,
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
    fn dedup_hits_by_name_keeps_first_occurrence_only() {
        let hits = vec![hit(0, "Ceres"), hit(1, "Ceres"), hit(2, "Vesta")];
        let deduped = dedup_hits_by_name(&hits);
        assert_eq!(deduped.len(), 2);
        assert_eq!(deduped[0].source_index, 0);
        assert_eq!(deduped[1].name, "Vesta");
    }
}
