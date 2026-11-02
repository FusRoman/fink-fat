pub mod run;
pub mod status;

use serde::{Deserialize, Serialize};

pub use crate::orbit_fit::JobStatus;

/// Snapshot of a running/finished bulk fit job, returned to the client by
/// `status::get_bulk_orbit_fit_job_status`. Unlike the single-lineage fit
/// (`orbit_fit::OrbitFitJobView`), there is no per-fit orbit to display here
/// — results land directly in `orbit_fits` — only aggregate progress.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BulkOrbitFitJobView {
    pub status: JobStatus,
    pub total: usize,
    pub processed: usize,
    pub succeeded: usize,
    pub failed: usize,
    pub elapsed_secs: f64,
    pub logs: Vec<String>,
    pub error: Option<String>,
}

/// Server-side job entry — not sent to the client directly.
/// `processed`/`succeeded`/`failed` are `Arc<AtomicUsize>` so the rayon
/// workers fitting each branch can update progress without taking the
/// job-registry mutex on every branch; only start/finish touch the registry.
#[cfg(feature = "server")]
#[derive(Clone, Debug)]
pub struct BulkOrbitFitJob {
    pub status: JobStatus,
    pub total: usize,
    pub processed: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    pub succeeded: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    pub failed: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    pub started_at: std::time::Instant,
    pub logs: Vec<String>,
    pub error: Option<String>,
}

#[cfg(feature = "server")]
impl BulkOrbitFitJob {
    pub fn new(total: usize) -> Self {
        Self {
            status: JobStatus::Running,
            total,
            processed: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            succeeded: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            failed: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            started_at: std::time::Instant::now(),
            logs: Vec::new(),
            error: None,
        }
    }

    pub fn view(&self) -> BulkOrbitFitJobView {
        use std::sync::atomic::Ordering;
        BulkOrbitFitJobView {
            status: self.status,
            total: self.total,
            processed: self.processed.load(Ordering::Relaxed),
            succeeded: self.succeeded.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
            elapsed_secs: self.started_at.elapsed().as_secs_f64(),
            logs: self.logs.clone(),
            error: self.error.clone(),
        }
    }
}
