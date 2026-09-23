//! Bulk Skybot check: submit every converged branch's observations to
//! Skybot, as a single durable, resumable, DB-backed job — unlike
//! [`crate::bulk_cnd`]/[`crate::bulk_orbit_fit`], which track their job in
//! an in-memory registry that a server restart simply erases.
//!
//! Skybot is queried one observation at a time (no batching, unlike CND),
//! and at real dataset sizes (tens of thousands of observations) this can
//! run for hours, so an in-memory registry isn't good enough here: the job
//! needs to survive a restart, be resumable without re-checking already-done
//! observations, and be visible/killable from any page load. Two new
//! Postgres tables make this possible — see `src/converter/sql/create_skybot_bulk_tables.sql`:
//! `skybot_bulk_jobs` (one row per job attempt; a partial unique index on
//! `status = 'running'` is the actual "only one job at a time" guarantee,
//! enforced by Postgres rather than a process-local `AtomicBool`) and
//! `skybot_obs_status` (one row per observation ever checked, which is what
//! makes "resume, prioritizing never-checked observations first, then
//! oldest-checked first" possible at all).
//!
//! [`run`] reuses [`crate::skybot_search::parsing::fetch_conesearch_hits`]
//! for the actual network call and the same semaphore-bounded fan-out shape
//! as [`crate::skybot_search::run::run_skybot_job`] — only the work source
//! (a priority-ordered DB query instead of one lineage's points) and sink
//! (per-observation DB upserts instead of an in-memory job) differ.

/// Server-only: all Postgres access.
#[cfg(feature = "server")]
pub mod persist;
pub mod run;
pub mod status;

use serde::{Deserialize, Serialize};

/// Where a [`SkybotBulkJobView`] currently stands. A superset of
/// [`crate::orbit_fit::JobStatus`]'s three states — reused would have meant
/// bolting `Killed`/`Interrupted` onto a type every other bulk/single job
/// also uses, for states that only make sense for a job durable enough to
/// outlive a page visit or a server restart.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum SkybotBulkJobStatus {
    Running,
    Done,
    Failed,
    /// Stopped early by [`status::request_kill_skybot_bulk_job`].
    Killed,
    /// The server process that ran this job exited (crash or restart)
    /// before it reached a terminal state — set by
    /// [`run::reconcile_stale_skybot_bulk_jobs`] the next time any bulk
    /// Skybot endpoint runs, not by the job itself.
    Interrupted,
}

impl SkybotBulkJobStatus {
    /// The exact `TEXT` value stored in `skybot_bulk_jobs.status`.
    pub fn as_column(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Done => "done",
            Self::Failed => "failed",
            Self::Killed => "killed",
            Self::Interrupted => "interrupted",
        }
    }

    /// Parses a `skybot_bulk_jobs.status` value back into a
    /// [`SkybotBulkJobStatus`].
    ///
    /// # Return
    ///
    /// `None` for any value other than one [`Self::as_column`] can produce —
    /// this column is only ever written by this module, so an unrecognized
    /// value means the schema and this enum have drifted apart.
    pub fn from_column(value: &str) -> Option<Self> {
        match value {
            "running" => Some(Self::Running),
            "done" => Some(Self::Done),
            "failed" => Some(Self::Failed),
            "killed" => Some(Self::Killed),
            "interrupted" => Some(Self::Interrupted),
            _ => None,
        }
    }
}

/// Snapshot of the current (or most recent) bulk Skybot job, returned to the
/// client by [`status::get_current_skybot_bulk_job_status`]. There is no job
/// id on this type, unlike every other bulk job's view: at most one bulk
/// Skybot job is ever running, so a client polls "what's the current job"
/// unconditionally rather than holding an id across a page visit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SkybotBulkJobView {
    pub status: SkybotBulkJobStatus,
    pub radius_arcsec: f64,
    /// RFC 3339.
    pub started_at: String,
    /// RFC 3339, `None` while `status == Running`.
    pub finished_at: Option<String>,
    /// Population-wide, not "since this job started" — see this module's
    /// doc comment and `skybot_bulk_jobs.total_observations`'s column
    /// comment for why.
    pub total_observations: i64,
    pub processed_observations: i64,
    pub matched_observations: i64,
    pub error: Option<String>,
    pub logs: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_round_trips_through_its_column_encoding() {
        for status in [
            SkybotBulkJobStatus::Running,
            SkybotBulkJobStatus::Done,
            SkybotBulkJobStatus::Failed,
            SkybotBulkJobStatus::Killed,
            SkybotBulkJobStatus::Interrupted,
        ] {
            assert_eq!(
                SkybotBulkJobStatus::from_column(status.as_column()),
                Some(status)
            );
        }
    }

    #[test]
    fn from_column_rejects_an_unrecognized_value() {
        assert_eq!(SkybotBulkJobStatus::from_column("bogus"), None);
    }
}
