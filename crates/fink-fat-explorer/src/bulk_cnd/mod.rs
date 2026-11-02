pub mod run;
pub mod status;

use serde::{Deserialize, Serialize};

pub use crate::orbit_fit::JobStatus;

/// Snapshot of a running/finished bulk CND job, returned to the client by
/// `status::get_bulk_cnd_job_status`. Unlike the per-lineage job
/// (`crate::cnd_search::CndJobView`), progress is tracked per *observation*
/// (CND batches are observation-sized, not branch-sized) while the summary
/// still reports branch-level counts, since "how many trajectories got a
/// match" is the number a user actually cares about here.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BulkCndJobView {
    pub status: JobStatus,
    pub total_branches: usize,
    pub total_observations: usize,
    pub processed_observations: usize,
    pub branches_with_match: usize,
    pub elapsed_secs: f64,
    pub logs: Vec<String>,
    pub error: Option<String>,
}

/// Server-side job entry — not sent to the client directly.
/// `processed_observations`/`branches_with_match` are `Arc<AtomicUsize>` so
/// progress can be updated between sequential CND batches without
/// re-locking the whole registry for each one — same rationale as
/// `crate::bulk_orbit_fit::BulkOrbitFitJob`'s counters, even though this job
/// isn't rayon-parallel (batches are sequential — see
/// `crate::cnd_search::client::CND_BATCH_SIZE`'s doc comment for why).
#[cfg(feature = "server")]
#[derive(Clone, Debug)]
pub struct BulkCndJob {
    pub status: JobStatus,
    pub total_branches: usize,
    pub total_observations: usize,
    pub processed_observations: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    pub branches_with_match: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    pub started_at: std::time::Instant,
    pub logs: Vec<String>,
    pub error: Option<String>,
}

#[cfg(feature = "server")]
impl BulkCndJob {
    pub fn new() -> Self {
        Self {
            status: JobStatus::Running,
            total_branches: 0,
            total_observations: 0,
            processed_observations: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            branches_with_match: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            started_at: std::time::Instant::now(),
            logs: Vec::new(),
            error: None,
        }
    }

    pub fn view(&self) -> BulkCndJobView {
        use std::sync::atomic::Ordering;
        BulkCndJobView {
            status: self.status,
            total_branches: self.total_branches,
            total_observations: self.total_observations,
            processed_observations: self.processed_observations.load(Ordering::Relaxed),
            branches_with_match: self.branches_with_match.load(Ordering::Relaxed),
            elapsed_secs: self.started_at.elapsed().as_secs_f64(),
            logs: self.logs.clone(),
            error: self.error.clone(),
        }
    }
}

#[cfg(feature = "server")]
impl Default for BulkCndJob {
    fn default() -> Self {
        Self::new()
    }
}
