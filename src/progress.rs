//! Progress reporting utilities based on [`indicatif`].
//!
//! This module provides ergonomic helpers around [`MultiProgress`] and
//! [`ProgressBar`] for long-running computations, with special care for:
//!
//! - **pytest compatibility**: bars are rendered on `stderr` with a capped
//!   refresh rate to avoid overwhelming test logs.
//! - **throttling**: update frequency can be limited to reduce rendering
//!   overhead when tracking millions of iterations.
//! - **optional progress**: `maybe_*` helpers allow enabling/disabling
//!   progress reporting without cluttering the call sites.
//!
//! Typical usage:
//!
//! ```rust
//! use fink_fat::util::progress::*;
//!
//! let mp = make_multi_progress();
//! let pb = make_bar(&mp, 1_000, "Processing");
//!
//! let mut last = 0;
//! for i in 0..1_000 {
//!     // ... heavy work ...
//!     throttled_inc(&pb, i, &mut last, 100);
//! }
//! maybe_progress_finish(Some(&pb), 1_000, "Done");
//! ```

use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};

/// Create a [`MultiProgress`] suitable for unit tests or batch jobs.
///
/// This uses [`ProgressDrawTarget::stderr_with_hz`] with a low refresh rate
/// (10 Hz) to keep logs readable under `pytest` or CI environments.
///
/// Returns
/// -------
/// A [`MultiProgress`] instance ready to spawn child bars.
pub fn make_multi_progress() -> MultiProgress {
    MultiProgress::with_draw_target(ProgressDrawTarget::stderr_with_hz(10))
}

/// Create and register a child progress bar with custom style.
///
/// Parameters
/// ----------
/// * `mp` – Parent [`MultiProgress`] handle.
/// * `len` – Expected length (number of items).
/// * `msg` – Label shown to the left of the bar.
///
/// Style
/// -----
/// - Spinner + left-aligned message (width 24).
/// - Elapsed time, cyan/blue bar (40 chars), right-aligned counters.
/// - Unicode block characters for smooth animation.
///
/// Returns
/// -------
/// A configured [`ProgressBar`] registered in the given `mp`.
pub fn make_bar(mp: &MultiProgress, len: u64, msg: &str) -> ProgressBar {
    let pb = mp.add(ProgressBar::new(len));
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} {msg:24} [{elapsed_precise}] \
             [{bar:40.cyan/blue}] {pos:>9}/{len:<9} ({percent:>3}%)",
        )
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏  "),
    );
    pb.set_message(msg.to_string());
    pb
}

/// Increment the bar with throttling to reduce overhead.
///
/// Parameters
/// ----------
/// * `pb` – Progress bar to update.
/// * `processed` – Current number of processed items.
/// * `last_drawn` – Mutable counter tracking last drawn position.
/// * `chunk` – Minimum increment step before updating the bar.
///
/// Notes
/// -----
/// For very large `N`, calling `ProgressBar::inc` at each iteration is
/// expensive. This helper ensures updates happen only every `chunk`
/// increments.
pub fn throttled_inc(pb: &ProgressBar, processed: u64, last_drawn: &mut u64, chunk: u64) {
    if processed.saturating_sub(*last_drawn) >= chunk {
        pb.set_position(processed);
        *last_drawn = processed;
    }
}

/// Conditionally initialize a progress bar (no-op if `None`).
///
/// This allows ergonomically disabling progress tracking by passing `None`.
#[inline]
pub fn maybe_progress_start(pb: Option<&ProgressBar>, len: u64, msg: &str) {
    if let Some(pb) = pb {
        pb.set_message(msg.to_string());
        pb.set_length(len);
        pb.set_position(0);
    }
}

/// Conditionally finish a progress bar (no-op if `None`).
///
/// Ensures the bar is marked complete and replaced with a final message.
#[inline]
pub fn maybe_progress_finish(pb: Option<&ProgressBar>, len: u64, msg: &str) {
    if let Some(pb) = pb {
        pb.set_position(len);
        pb.finish_with_message(msg.to_string());
    }
}

/// Conditionally update a progress bar with throttling (no-op if `None`).
///
/// Parameters
/// ----------
/// * `pb` – Optional progress bar.
/// * `current` – Current count.
/// * `last` – Last drawn count (updated in place).
/// * `step` – Update only when at least `step` increments have passed.
///
/// Notes
/// -----
/// This is a safe variant of [`throttled_inc`] that works even when the
/// caller has disabled progress reporting.
#[inline]
pub fn maybe_progress_throttled_set(
    pb: Option<&ProgressBar>,
    current: u64,
    last: &mut u64,
    step: u64,
) {
    if let Some(pb) = pb {
        if current.wrapping_sub(*last) >= step {
            pb.set_position(current);
            *last = current;
        }
    }
}
