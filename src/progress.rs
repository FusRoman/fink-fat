//! indicatif-backed implementation of [`PipelineHooks`] and [`StageProgress`].
//!
//! This module provides a compact, terminal-friendly progress reporting
//! implementation for the pipeline stages using `indicatif::MultiProgress` and
//! `ProgressBar` primitives. It is intended to be enabled at runtime with the
//! `--progress` CLI flag; when disabled the pipeline uses
//! [`fink_fat_engine::pipeline::hooks::NoopHooks`] with zero overhead.
//!
//! Key behaviours
//! - Renders per-stage progress bars with consistent styling.
//! - Supports nested child progress scopes (via `StageProgress::child`).
//! - Exposes a `multi_progress()` handle that can be shared with the logging
//!   layer so log lines are printed above active bars using
//!   `MultiProgress::println` (avoids display corruption).
//!
//! Usage
//! ```no_run
//! let hooks = fink_fat::progress::IndicatifHooks::new();
//! runner.run(&mut ctx, &hooks)?; // runner: PipelineRunner
//! ```

use std::sync::Arc;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use fink_fat_engine::pipeline::{
    hooks::{PipelineHooks, StageMeta, StageProgress, StageReport},
    stages::PipelineStage,
};

// ── Style helpers ─────────────────────────────────────────────────────────────

/// Progress bar visual style used for determinate stages.
///
/// Returns an `indicatif::ProgressStyle` configured with a narrow message
/// column and a 40-character progress gauge.
fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{spinner:.green} {msg:<30} [{bar:40.cyan/blue}] {pos}/{len} ({elapsed})",
    )
    .unwrap()
    .progress_chars("##-")
}

/// Progress spinner style used for indeterminate stages.
///
/// Returns an `indicatif::ProgressStyle` configured with a compact
/// spinner and a short message column.
fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner:.green} {msg:<30} {elapsed}")
        .unwrap()
        .tick_strings(&["⠋", "⠙", "⠸", "⠴", "⠦", "⠇", "✔"])
}

// ── IndicatifProgress ─────────────────────────────────────────────────────────

/// A [`StageProgress`] scope backed by an indicatif [`ProgressBar`].
///
/// Holds a reference to the shared [`MultiProgress`] so it can spawn nested
/// child bars via [`StageProgress::child`].
pub struct IndicatifProgress {
    pb: ProgressBar,
    mp: Arc<MultiProgress>,
}

impl IndicatifProgress {
    /// Create a new `IndicatifProgress` scope.
    ///
    /// Parameters
    /// * `mp`: shared `MultiProgress` used to host the bar.
    /// * `meta`: stage metadata (label and optional total) used to select a
    ///   determinate bar or an indeterminate spinner.
    fn new(mp: Arc<MultiProgress>, meta: &StageMeta) -> Self {
        let pb = match meta.total {
            Some(n) => {
                let pb = mp.add(ProgressBar::new(n));
                pb.set_style(bar_style());
                pb
            }
            None => {
                let pb = mp.add(ProgressBar::new_spinner());
                pb.set_style(spinner_style());
                pb
            }
        };
        pb.set_message(meta.label.clone());
        Self { pb, mp }
    }
}

impl StageProgress for IndicatifProgress {
    fn set_total(&self, total: u64) {
        self.pb.set_length(total);
        // Switch to bar style when we learn the total.
        self.pb.set_style(bar_style());
    }

    fn inc(&self, delta: u64) {
        self.pb.inc(delta);
    }

    fn finish(&self) {
        self.pb
            .finish_with_message(format!("{} ✔", self.pb.message()));
    }

    fn child(&self, meta: StageMeta) -> Arc<dyn StageProgress> {
        Arc::new(IndicatifProgress::new(self.mp.clone(), &meta))
    }
}

// ── IndicatifHooks ────────────────────────────────────────────────────────────

/// A [`PipelineHooks`] implementation that renders stage progress using
/// indicatif's [`MultiProgress`].
///
/// # Usage
/// ```ignore
/// let hooks = IndicatifHooks::new();
/// runner.run(&mut ctx, &hooks)?;
/// ```
pub struct IndicatifHooks {
    mp: Arc<MultiProgress>,
}

impl IndicatifHooks {
    pub fn new() -> Self {
        Self {
            mp: Arc::new(MultiProgress::new()),
        }
    }

    /// Build an [`IndicatifHooks`] that uses an externally owned [`MultiProgress`].
    ///
    /// Use this when you need to share the same `MultiProgress` with the
    /// logging layer so that log lines are printed above the active bars.
    pub fn with_mp(mp: Arc<MultiProgress>) -> Self {
        Self { mp }
    }

    /// Return a clone of the shared [`MultiProgress`] handle.
    ///
    /// Pass this to [`crate::logging::init_logging`] so that log lines are
    /// routed through [`MultiProgress::println`] instead of raw stderr.
    pub fn multi_progress(&self) -> Arc<MultiProgress> {
        self.mp.clone()
    }
}

impl Default for IndicatifHooks {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineHooks for IndicatifHooks {
    fn on_stage_start(&self, _stage: PipelineStage, meta: StageMeta) -> Arc<dyn StageProgress> {
        Arc::new(IndicatifProgress::new(self.mp.clone(), &meta))
    }

    fn on_stage_end(&self, _stage: PipelineStage, report: StageReport) {
        // Print a compact summary line below the finished bar.
        let counters: Vec<String> = report
            .counters
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        let summary = if counters.is_empty() {
            format!("{}ms", report.elapsed_ms)
        } else {
            format!("{}ms  {}", report.elapsed_ms, counters.join("  "))
        };
        // `println_*` prints above the active bars to avoid flickering.
        let _ = self
            .mp
            .println(format!(" (Stage {}) → {summary}", _stage.label()));
    }
}
