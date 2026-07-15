//! Per-object outcome classification across the whole run.
//!
//! [`night_stats::compute_night_tracking_stats`](super::night_stats::compute_night_tracking_stats)'s
//! completeness metrics only look at the *final* night's snapshot: an
//! object that had a perfectly pure, fully-covering branch on night 10 but
//! lost it by the end of the run is indistinguishable, in that metric, from
//! an object the tracker never captured at all. [`ObjectOutcomeTracker`]
//! watches every night's pure branches to tell those cases apart, so a low
//! `final_completeness_pct` can be attributed to a specific failure mode
//! (never associated, never pure, fragmented into competing tracks, or
//! captured then lost) instead of a single opaque percentage.

use ahash::{AHashMap, AHashSet};
use photom::TrajId;

use fink_fat_engine::topocentric_kf::branching::Branch;

use crate::{
    seed_bank_report::ground_truth::{ObsTrajLookup, SeedPurity},
    tracking_report::gold_trajectory::GoldTrajectoryTracker,
};

/// Final classification of a single multi-detection ground-truth
/// trajectory, derived from [`ObjectOutcomeTracker`]'s bookkeeping once the
/// whole run has been observed. Checked in this order — e.g. a trajectory
/// that was both fragmented and eventually lost is reported as
/// [`Self::Fragmented`], since that's the earlier/more actionable failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectOutcome {
    /// No branch ever contained one of this object's observations —
    /// seeding/gating never associated it with anything.
    NeverTouched,
    /// Touched by some branch, but never by a pure one — always mixed in
    /// with another object's observations.
    TouchedNeverPure,
    /// More than one pure branch for this object existed at the same time
    /// at some point in the run — the tracker split it into competing
    /// tracks instead of keeping a single one.
    Fragmented,
    /// A pure branch existed at some point but none exists anymore by the
    /// end of the run (lost along the way, e.g. pruned).
    PureThenLost,
    /// A pure branch is alive at the end of the run but covers less than
    /// the relaxed coverage threshold of the observations seen so far.
    PureAliveBelowCoverage,
    /// A pure branch is alive at the end of the run and covers at least the
    /// relaxed coverage threshold, but not exactly every observation.
    CompleteRelaxed,
    /// A pure branch is alive at the end of the run and covers exactly
    /// every observation seen for this object.
    CompleteStrict,
}

impl ObjectOutcome {
    /// Stable, human-readable label for console/plot output.
    pub fn label(self) -> &'static str {
        match self {
            ObjectOutcome::NeverTouched => "never touched",
            ObjectOutcome::TouchedNeverPure => "touched, never pure",
            ObjectOutcome::Fragmented => "fragmented",
            ObjectOutcome::PureThenLost => "pure then lost",
            ObjectOutcome::PureAliveBelowCoverage => "pure, alive, below coverage",
            ObjectOutcome::CompleteRelaxed => "complete (relaxed)",
            ObjectOutcome::CompleteStrict => "complete (strict)",
        }
    }

    /// All variants, in the fixed order used for reporting.
    pub fn all() -> [ObjectOutcome; 7] {
        [
            ObjectOutcome::NeverTouched,
            ObjectOutcome::TouchedNeverPure,
            ObjectOutcome::Fragmented,
            ObjectOutcome::PureThenLost,
            ObjectOutcome::PureAliveBelowCoverage,
            ObjectOutcome::CompleteRelaxed,
            ObjectOutcome::CompleteStrict,
        ]
    }
}

/// Cumulative per-object bookkeeping, fed one night at a time via
/// [`Self::observe_night`].
#[derive(Default)]
pub struct ObjectOutcomeTracker {
    ever_touched: AHashSet<TrajId>,
    best_pure_coverage: AHashMap<TrajId, usize>,
    last_pure_night: AHashMap<TrajId, usize>,
    max_simultaneous_pure_branches: AHashMap<TrajId, usize>,
}

impl ObjectOutcomeTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one night's live branches into the per-object bookkeeping. Call
    /// once per night, with that night's live `branches` (i.e. the
    /// `BranchCollection` *after* `advance_one_night`).
    pub fn observe_night(
        &mut self,
        step: usize,
        branches: &[Branch<'_, '_>],
        ground_truth: &ObsTrajLookup,
    ) {
        let mut pure_branches_tonight: AHashMap<TrajId, usize> = AHashMap::default();

        for branch in branches {
            match ground_truth.classify(branch.track_ids()) {
                SeedPurity::Pure(traj_id) => {
                    self.ever_touched.insert(traj_id.clone());
                    let coverage = branch.track_ids().len();
                    self.best_pure_coverage
                        .entry(traj_id.clone())
                        .and_modify(|best| *best = (*best).max(coverage))
                        .or_insert(coverage);
                    self.last_pure_night.insert(traj_id.clone(), step);
                    *pure_branches_tonight.entry(traj_id).or_insert(0) += 1;
                }
                SeedPurity::Mixed => {
                    for &obs_id in branch.track_ids() {
                        if let Some(traj_id) = ground_truth.traj_of(obs_id) {
                            self.ever_touched.insert(traj_id.clone());
                        }
                    }
                }
                SeedPurity::Unknown => {}
            }
        }

        for (traj_id, count) in pure_branches_tonight {
            self.max_simultaneous_pure_branches
                .entry(traj_id)
                .and_modify(|best| *best = (*best).max(count))
                .or_insert(count);
        }
    }

    /// Classify every *trackable* trajectory known to `gold_tracker` (i.e.
    /// [`GoldTrajectoryTracker::is_trackable`] — had >= 2 observations
    /// within a single night at some point; objects that never did are
    /// structurally unreachable by this pipeline's intra-night seeding and
    /// are excluded here rather than reported as tracking failures) into an
    /// [`ObjectOutcome`], plus its best-ever pure coverage ratio
    /// (`best_pure_coverage / n_obs_so_far`, `None` if never pure).
    ///
    /// `last_step` is the index of the last night processed — a pure branch
    /// whose `last_pure_night` is earlier than this is one that existed and
    /// then disappeared before the run ended.
    pub fn classify_all(
        &self,
        gold_tracker: &GoldTrajectoryTracker,
        last_step: usize,
        completeness_coverage_threshold: f64,
    ) -> Vec<(TrajId, ObjectOutcome, Option<f64>)> {
        gold_tracker
            .trackable_traj_ids()
            .map(|traj_id| {
                let n_seen = gold_tracker.n_obs_so_far(traj_id).unwrap_or(0);
                let coverage_ratio = self
                    .best_pure_coverage
                    .get(traj_id)
                    .map(|&c| c as f64 / n_seen.max(1) as f64);
                let outcome =
                    self.classify_one(traj_id, n_seen, last_step, completeness_coverage_threshold);
                (traj_id.clone(), outcome, coverage_ratio)
            })
            .collect()
    }

    fn classify_one(
        &self,
        traj_id: &TrajId,
        n_seen: usize,
        last_step: usize,
        completeness_coverage_threshold: f64,
    ) -> ObjectOutcome {
        if !self.ever_touched.contains(traj_id) {
            return ObjectOutcome::NeverTouched;
        }
        let Some(&best_coverage) = self.best_pure_coverage.get(traj_id) else {
            return ObjectOutcome::TouchedNeverPure;
        };
        if self
            .max_simultaneous_pure_branches
            .get(traj_id)
            .copied()
            .unwrap_or(0)
            > 1
        {
            return ObjectOutcome::Fragmented;
        }
        let last_pure = self.last_pure_night.get(traj_id).copied().unwrap_or(0);
        if last_pure < last_step {
            return ObjectOutcome::PureThenLost;
        }
        if n_seen > 0 && best_coverage == n_seen {
            return ObjectOutcome::CompleteStrict;
        }
        if n_seen > 0 && best_coverage as f64 / n_seen as f64 >= completeness_coverage_threshold {
            return ObjectOutcome::CompleteRelaxed;
        }
        ObjectOutcome::PureAliveBelowCoverage
    }
}
