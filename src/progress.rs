// src/util/progress.rs (par ex.)
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};

/// Crée une MultiProgress adaptée à l’exécution sous pytest (stderr, refresh limité).
pub fn make_multi_progress() -> MultiProgress {
    let mp = MultiProgress::with_draw_target(ProgressDrawTarget::stderr_with_hz(10));
    mp
}

pub fn make_bar(mp: &MultiProgress, len: u64, msg: &str) -> ProgressBar {
    let pb = mp.add(ProgressBar::new(len));
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} {msg:24} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>9}/{len:<9} ({percent:>3}%)",
        )
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏  "),
    );
    pb.set_message(msg.to_string());
    pb
}

/// Incrémente la barre par paquets (throttle) pour éviter le surcoût indicatif sur très gros N.
pub fn throttled_inc(pb: &ProgressBar, processed: u64, last_drawn: &mut u64, chunk: u64) {
    if processed.saturating_sub(*last_drawn) >= chunk {
        pb.set_position(processed);
        *last_drawn = processed;
    }
}

/// Internal throttled progress helper (no-op when `pb` is `None`).
#[inline]
pub fn maybe_progress_start(pb: Option<&ProgressBar>, len: u64, msg: &str) {
    if let Some(pb) = pb {
        pb.set_message(msg.to_string());
        pb.set_length(len);
        pb.set_position(0);
    }
}

#[inline]
pub fn maybe_progress_finish(pb: Option<&ProgressBar>, len: u64, msg: &str) {
    if let Some(pb) = pb {
        pb.set_position(len);
        pb.finish_with_message(msg.to_string());
    }
}

#[inline]
pub fn maybe_progress_throttled_set(pb: Option<&ProgressBar>, current: u64, last: &mut u64, step: u64) {
    if let Some(pb) = pb {
        // Update only every `step` increments (or on exact multiples).
        if current.wrapping_sub(*last) >= step {
            pb.set_position(current);
            *last = current;
        }
    }
}
