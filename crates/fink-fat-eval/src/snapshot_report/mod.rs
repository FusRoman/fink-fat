//! Point-in-time statistics/plots for a single `BranchCollection` snapshot
//! loaded from disk (the `.rkyv` file the real `fink-fat` engine binary
//! writes under `storage_path` after every night) — unlike
//! [`crate::tracking_report`], which drives the full night-by-night
//! simulation and needs per-night history, this module only ever sees the
//! final state, so it reports structural/distributional stats and (if
//! ground truth is present) a reconstruction-efficacy classification per
//! ground-truth trajectory, rather than any per-night trend.
//!
//! Entry point: `bin/tracking_analysis.rs`'s `--from-snapshot` mode.

pub mod efficacy;
pub mod plots;
pub mod stats;
