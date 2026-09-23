//! Background MPC Check Near-Duplicates (CND) jobs for the lineage
//! trajectory plot: check whether a lineage's observations have already been
//! published at the MPC, using the MPC's Check Near-Duplicates API
//! (<https://docs.minorplanetcenter.net/mpc-ops-docs/apis/cnd/>) — the same
//! background-job/poll convention as [`crate::skybot_search`], just
//! submitting obs80-encoded observations in sequential batches instead of
//! one conesearch request per point (see [`client::CND_BATCH_SIZE`]).
//!
//! Obs80 line construction is pure, unit-tested code in [`obs80`]; the CND
//! HTTP request/response shape lives in [`client`]; only [`run`] touches the
//! network and the job registry.
//!
//! Every finished attempt (including one that finds nothing) is also
//! recorded in the `cnd_queries` table via [`persist::insert_cnd_query`] —
//! reused unmodified by `crate::bulk_cnd::run` — so the lineage page can show
//! its last result without re-running the check on every visit;
//! [`history::get_last_cnd_query`] reads that back.

/// Server-only: sending the request pulls in `reqwest`, which doesn't build
/// for `wasm32`.
#[cfg(feature = "server")]
pub mod client;
pub mod history;
/// Server-only: computing an obs80 line isn't itself server-specific, but
/// it's only ever called from server code — kept out of the wasm client
/// bundle the same way `skybot_search::parsing` is.
#[cfg(feature = "server")]
pub mod obs80;
/// Server-only: writing to `cnd_queries` pulls in `sqlx`.
#[cfg(feature = "server")]
pub mod persist;
pub mod run;
pub mod status;

use serde::{Deserialize, Serialize};

pub use crate::orbit_fit::JobStatus;

/// The MPC's own defaults for the CND API's match thresholds, mirrored from
/// the companion Python project's `report_mpc_cnd_check.py` so the explorer
/// UI's inputs default to the same values.
pub const DEFAULT_TIME_SEPARATION_S: f64 = 60.0;
pub const DEFAULT_ANGLE_SEPARATION_ARCSEC: f64 = 5.0;

/// CND's own hard bounds on `time_separation_s`/`angle_separation_arcsec`
/// (confirmed against the live API's request schema — `GET
/// https://data.minorplanetcenter.net/api/cnd` with no body returns it, and
/// separately by probing the exact boundary values: 60/10 succeed, anything
/// above fails, -1 fails). Note the two are **not** symmetric: time tops out
/// at a full minute, but angle tops out at 10 arcsec, well below the UI's
/// old free-form input range. Submitting a value outside these bounds fails
/// the request's own parameter validation — every single batch, regardless
/// of the observations in it — which [`run::start_cnd_search`]'s (and
/// `crate::bulk_cnd::run::start_bulk_cnd_check`'s) resilient batching can't
/// tell apart from "one specific observation is the problem": it just sees
/// every sub-batch fail identically and bisects all the way down to
/// individual lines for nothing, taking far longer than the job should.
/// Clamping here, before ever building a request, is what actually avoids
/// that (confirmed as the root cause of a real stuck bulk run: the UI's
/// angle input had been set to 60, ten times over this bound).
///
/// The lower bound is kept at `0.5` rather than the API's own documented
/// `0` — a zero-width match window is a degenerate case the API technically
/// accepts but that isn't a meaningful search (nothing can ever match an
/// exactly-zero separation) and is untested territory we'd rather not risk
/// after already hitting one undocumented edge-case failure in this API.
pub const MIN_TIME_SEPARATION_S: f64 = 0.5;
pub const MAX_TIME_SEPARATION_S: f64 = 60.0;
pub const MIN_ANGLE_SEPARATION_ARCSEC: f64 = 0.5;
pub const MAX_ANGLE_SEPARATION_ARCSEC: f64 = 10.0;

/// Clamps a requested `time_separation_s` into
/// `[MIN_TIME_SEPARATION_S, MAX_TIME_SEPARATION_S]` — matches the UI
/// inputs' range, and is re-applied server-side in case a tampered client
/// call sends a value outside it.
pub fn clamp_time_separation_s(time_separation_s: f64) -> f64 {
    time_separation_s.clamp(MIN_TIME_SEPARATION_S, MAX_TIME_SEPARATION_S)
}

/// Clamps a requested `angle_separation_arcsec` into
/// `[MIN_ANGLE_SEPARATION_ARCSEC, MAX_ANGLE_SEPARATION_ARCSEC]` — same
/// rationale as [`clamp_time_separation_s`].
pub fn clamp_angle_separation_arcsec(angle_separation_arcsec: f64) -> f64 {
    angle_separation_arcsec.clamp(MIN_ANGLE_SEPARATION_ARCSEC, MAX_ANGLE_SEPARATION_ARCSEC)
}

/// One observation to submit to CND, built client-side from a lineage's real
/// observations. Richer than [`crate::skybot_search::SkybotQueryPoint`]
/// since building an obs80 line needs magnitude/filter/observatory code/
/// branch_id too, not just sky position and epoch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CndQueryPoint {
    pub source_index: usize,
    pub obs_id: i64,
    pub branch_id: i64,
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub mjd_tt: f64,
    pub magnitude: f64,
    pub filter: i16,
    pub mpc_code_obs: String,
}

/// One of our own observations that CND found near-duplicate(s) for among
/// already-published MPC observations.
///
/// Only matched observations get a `CndHit` — an observation with no CND
/// match simply has none in the list, mirroring
/// [`crate::skybot_search::SkybotHit`]'s "absence = no match" convention,
/// rather than one entry per observation regardless of outcome.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CndHit {
    /// See [`CndQueryPoint::source_index`].
    pub source_index: usize,
    pub obs_id: i64,
    /// Our own observation's position — CND doesn't return the matched
    /// published observation's coordinates in a form worth parsing back out
    /// (see `closest_match_obs80`), so this is plotted the same way Skybot
    /// hits are, just marking our own point instead of a foreign one.
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub n_matches: usize,
    pub min_time_separation_s: f64,
    pub min_angle_separation_arcsec: f64,
    /// Raw obs80 line of whichever match had the smallest angular
    /// separation, shown as-is in the hover tooltip. Not parsed further:
    /// parsing a foreign, already-published obs80 record back into
    /// structured fields (real designation, observatory, etc.) is out of
    /// scope here — the raw line is already informative for a human to read.
    pub closest_match_obs80: Option<String>,
}

/// Snapshot of a running/finished CND job, returned to the client by
/// [`status::get_cnd_job_status`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CndJobView {
    pub status: JobStatus,
    pub total: usize,
    pub processed: usize,
    /// Every hit found so far, across all batches processed to date — grows
    /// incrementally the same way [`crate::skybot_search::SkybotJobView::hits`]
    /// does.
    pub hits: Vec<CndHit>,
    pub logs: Vec<String>,
    pub error: Option<String>,
}

/// Server-side job entry — not sent to the client directly, [`CndJobView`]
/// (a plain snapshot) is what [`status::get_cnd_job_status`] returns.
#[cfg(feature = "server")]
#[derive(Clone, Debug)]
pub struct CndJob {
    pub status: JobStatus,
    pub total: usize,
    pub processed: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    pub hits: Vec<CndHit>,
    pub logs: Vec<String>,
    pub error: Option<String>,
}

#[cfg(feature = "server")]
impl CndJob {
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

    pub fn view(&self) -> CndJobView {
        use std::sync::atomic::Ordering;
        CndJobView {
            status: self.status,
            total: self.total,
            processed: self.processed.load(Ordering::Relaxed),
            hits: self.hits.clone(),
            logs: self.logs.clone(),
            error: self.error.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_time_separation_s_clamps_below_and_above_range() {
        assert_eq!(clamp_time_separation_s(-1.0), MIN_TIME_SEPARATION_S);
        assert_eq!(clamp_time_separation_s(120.0), MAX_TIME_SEPARATION_S);
        assert_eq!(clamp_time_separation_s(30.0), 30.0);
    }

    #[test]
    fn clamp_time_separation_s_rejects_zero() {
        // The API itself accepts 0 (a zero-width match window), but we
        // deliberately don't offer it — see MIN_TIME_SEPARATION_S's doc.
        assert_eq!(clamp_time_separation_s(0.0), MIN_TIME_SEPARATION_S);
        assert!(MIN_TIME_SEPARATION_S > 0.0);
    }

    #[test]
    fn clamp_angle_separation_arcsec_clamps_below_and_above_range() {
        assert_eq!(
            clamp_angle_separation_arcsec(-1.0),
            MIN_ANGLE_SEPARATION_ARCSEC
        );
        // The real bug this guards against: a UI value copied from the
        // time-separation field's own 60-second range, well past angle's
        // 10-arcsec bound.
        assert_eq!(
            clamp_angle_separation_arcsec(60.0),
            MAX_ANGLE_SEPARATION_ARCSEC
        );
        assert_eq!(clamp_angle_separation_arcsec(3.0), 3.0);
    }

    #[test]
    fn clamp_angle_separation_arcsec_rejects_zero() {
        assert_eq!(
            clamp_angle_separation_arcsec(0.0),
            MIN_ANGLE_SEPARATION_ARCSEC
        );
        assert!(MIN_ANGLE_SEPARATION_ARCSEC > 0.0);
    }
}
