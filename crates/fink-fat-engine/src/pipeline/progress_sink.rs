use std::sync::Arc;

use crate::pipeline::{
    hooks::{PipelineHooks, StageMeta},
    stages::PipelineStage,
};

/// Small, UI-agnostic progress reporter.
///
/// Implementations may forward to a CLI progress bar, a logger, metrics, etc.
pub trait ProgressSink: Send + Sync {
    /// Called once when the operation starts (optional).
    fn set_total(&self, _total: u64) {}

    /// Called to report that `delta` units of work are done.
    fn inc(&self, _delta: u64) {}

    /// Called when the operation ends (optional).
    fn finish(&self) {}

    /// Create a child progress scope.
    ///
    /// Default: returns a no-op child (so existing implementations keep working).
    fn child(&self, _meta: StageMeta) -> Arc<dyn ProgressSink> {
        Arc::new(NoopProgress)
    }
}

/// Default no-op sink (engine users can ignore progress entirely).
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopProgress;

impl ProgressSink for NoopProgress {}

/// `ProgressSink` that forwards increments to `PipelineHooks::on_stage_progress`.
pub struct HookProgressSink<'h> {
    pub hooks: &'h dyn PipelineHooks,
    pub stage: PipelineStage,
}

impl<'h> ProgressSink for HookProgressSink<'h> {
    fn set_total(&self, total: u64) {
        // Le total est plutôt communiqué via on_stage_start(meta.total).
        // Ici tu peux ne rien faire.
        let _ = total;
    }

    fn inc(&self, delta: u64) {
        self.hooks.on_stage_progress(self.stage, delta);
    }

    fn finish(&self) {
        // no-op
    }
}
