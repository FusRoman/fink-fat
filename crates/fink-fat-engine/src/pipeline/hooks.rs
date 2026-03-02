use std::sync::Arc;

use crate::pipeline::stages::PipelineStage;

// ── Fine-grained progress handle ─────────────────────────────────────────────

/// A scoped, fine-grained progress handle for a running stage or sub-operation.
///
/// Returned by [`PipelineHooks::on_stage_start`]. The engine uses it to report
/// incremental progress (milestones, per-night work, per-item work, etc.) without
/// any knowledge of the UI layer.
///
/// # indicatif integration
/// Wrap a `ProgressBar` in an `Arc` and implement this trait:
/// ```ignore
/// struct IndicatifProgress(ProgressBar);
/// impl StageProgress for IndicatifProgress {
///     fn set_total(&self, n: u64) { self.0.set_length(n); }
///     fn inc(&self, d: u64)       { self.0.inc(d); }
///     fn finish(&self)            { self.0.finish(); }
///     fn child(&self, meta: StageMeta) -> Arc<dyn StageProgress> {
///         let child_pb = multi.add(ProgressBar::new(meta.total.unwrap_or(0)));
///         Arc::new(IndicatifProgress(child_pb))
///     }
/// }
/// ```
pub trait StageProgress: Send + Sync {
    /// Declare the total number of work units for this scope (optional).
    fn set_total(&self, _total: u64) {}

    /// Report that `delta` units of work have been completed.
    fn inc(&self, _delta: u64) {}

    /// Called when this progress scope is finished.
    fn finish(&self) {}

    /// Create a named child scope for a sub-operation (e.g. one night within a stage).
    ///
    /// Default: returns a no-op child so existing implementations keep working.
    fn child(&self, _meta: StageMeta) -> Arc<dyn StageProgress> {
        Arc::new(NoopProgress)
    }
}

// ── No-op implementations ─────────────────────────────────────────────────────

/// No-op [`StageProgress`]: all methods are inlined empty calls.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopProgress;

impl StageProgress for NoopProgress {}

/// No-op [`PipelineHooks`]: useful in tests and benchmarks.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopHooks;

impl PipelineHooks for NoopHooks {}

// ── Pipeline hooks ────────────────────────────────────────────────────────────

/// Unified pipeline observer.
///
/// Implement this single trait to:
/// - receive structured stage lifecycle events (`on_stage_start` / `on_stage_end`),
/// - provide per-stage (and per-sub-operation) progress handles via [`StageProgress`],
/// - signal cancellation at stage boundaries.
///
/// All methods have default no-op implementations, so you only override what you need.
///
/// # indicatif integration example
/// ```ignore
/// struct CliHooks { mp: MultiProgress }
///
/// impl PipelineHooks for CliHooks {
///     fn on_stage_start(&self, _stage: PipelineStage, meta: StageMeta) -> Arc<dyn StageProgress> {
///         let pb = self.mp.add(ProgressBar::new(meta.total.unwrap_or(0)));
///         pb.set_message(meta.label);
///         Arc::new(IndicatifProgress(pb))
///     }
///     fn on_stage_end(&self, stage: PipelineStage, report: StageReport) {
///         eprintln!("{stage} done in {}ms", report.elapsed_ms);
///     }
/// }
/// ```
pub trait PipelineHooks: Send + Sync {
    /// Called before a stage runs. Returns a [`StageProgress`] handle the engine
    /// will use for fine-grained progress reporting within that stage.
    ///
    /// The default returns a no-op handle.
    fn on_stage_start(&self, _stage: PipelineStage, _meta: StageMeta) -> Arc<dyn StageProgress> {
        Arc::new(NoopProgress)
    }

    /// Called after a stage completes (whether successfully or not).
    fn on_stage_end(&self, _stage: PipelineStage, _report: StageReport) {}

    /// Return `true` to request cancellation before the next stage starts.
    fn is_cancelled(&self) -> bool {
        false
    }
}

// ── Shared data types ─────────────────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
pub struct StageMeta {
    pub label: String,
    /// Expected number of work units, if known upfront.
    pub total: Option<u64>,
}

#[derive(Clone, Debug, Default)]
pub struct StageReport {
    pub elapsed_ms: u128,
    pub counters: Vec<(&'static str, u64)>,
}
