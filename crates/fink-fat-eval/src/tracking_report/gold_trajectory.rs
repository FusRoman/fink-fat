//! Cumulative ground-truth bookkeeping across nights.
//!
//! [`seed_bank_report::night_stats::compute_night_stats`](crate::seed_bank_report::night_stats::compute_night_stats)
//! only judges a night's branches against *that night's* multi-detection
//! objects. Once a `BranchCollection` persists across nights, a stronger
//! question becomes answerable: does some branch's association history
//! exactly match *every* observation of a gold trajectory seen so far (not
//! just tonight's), with none missing and none extra? [`GoldTrajectoryTracker`]
//! accumulates the per-trajectory observation sets needed to answer that.

use ahash::{AHashMap, AHashSet};
use photom::{
    TrajId,
    observation_dataset::{ObsId, observation::Observation},
};

use crate::seed_bank_report::ground_truth::ObsTrajLookup;

/// Every ground-truth trajectory's observation set, as seen through the
/// nights processed so far.
#[derive(Default)]
pub struct GoldTrajectoryTracker {
    seen: AHashMap<TrajId, AHashSet<ObsId>>,
}

impl GoldTrajectoryTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a single observation against its ground-truth trajectory.
    fn insert(&mut self, traj_id: TrajId, obs_id: ObsId) {
        self.seen.entry(traj_id).or_default().insert(obs_id);
    }

    /// Fold this night's observations into the cumulative per-trajectory
    /// sets. Call once per night, before computing that night's stats.
    pub fn observe_night(&mut self, night_obs: &[&Observation], ground_truth: &ObsTrajLookup) {
        for obs in night_obs {
            if let Some(traj_id) = ground_truth.traj_of(*obs.id()) {
                self.insert(traj_id.clone(), *obs.id());
            }
        }
    }

    /// Number of ground-truth trajectories with >= 2 observations seen so
    /// far — the population a "fully successful" run should have completely
    /// reconstructed by the end.
    pub fn n_multi_detection_so_far(&self) -> usize {
        self.seen.values().filter(|obs| obs.len() >= 2).count()
    }

    /// Total observation count seen so far for `traj_id`, or `None` if it
    /// has never been observed.
    pub fn n_obs_so_far(&self, traj_id: &TrajId) -> Option<usize> {
        self.seen.get(traj_id).map(|obs| obs.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_across_multiple_nights() {
        let mut tracker = GoldTrajectoryTracker::new();
        assert_eq!(tracker.n_obs_so_far(&TrajId::Int(10)), None);

        tracker.insert(TrajId::Int(10), 1);
        assert_eq!(tracker.n_obs_so_far(&TrajId::Int(10)), Some(1));
        assert_eq!(tracker.n_multi_detection_so_far(), 0);

        tracker.insert(TrajId::Int(10), 2);
        assert_eq!(tracker.n_obs_so_far(&TrajId::Int(10)), Some(2));
        assert_eq!(tracker.n_multi_detection_so_far(), 1);
    }

    #[test]
    fn inserting_the_same_obs_id_twice_does_not_double_count() {
        let mut tracker = GoldTrajectoryTracker::new();
        tracker.insert(TrajId::Int(10), 1);
        tracker.insert(TrajId::Int(10), 1);
        assert_eq!(tracker.n_obs_so_far(&TrajId::Int(10)), Some(1));
    }

    #[test]
    fn distinct_trajectories_are_tracked_independently() {
        let mut tracker = GoldTrajectoryTracker::new();
        tracker.insert(TrajId::Int(10), 1);
        tracker.insert(TrajId::Int(20), 2);
        tracker.insert(TrajId::Int(20), 3);

        assert_eq!(tracker.n_obs_so_far(&TrajId::Int(10)), Some(1));
        assert_eq!(tracker.n_obs_so_far(&TrajId::Int(20)), Some(2));
        assert_eq!(tracker.n_multi_detection_so_far(), 1);
    }
}
