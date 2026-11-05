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
    /// Largest number of this trajectory's observations ever seen within a
    /// *single* night, over the whole run — see [`Self::is_trackable`].
    max_same_night_count: AHashMap<TrajId, usize>,
    /// `step` of the night that achieved [`Self::max_same_night_count`] for
    /// this trajectory, needed to go back and re-fetch that night's
    /// observations for gate diagnostics.
    best_same_night_step: AHashMap<TrajId, usize>,
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
    /// sets, and update each trajectory's same-night observation record.
    /// Call once per night, before computing that night's stats.
    pub fn observe_night(
        &mut self,
        step: usize,
        night_obs: &[&Observation],
        ground_truth: &ObsTrajLookup,
    ) {
        let mut tonight_counts: AHashMap<TrajId, usize> = AHashMap::default();
        for obs in night_obs {
            if let Some(traj_id) = ground_truth.traj_of(*obs.id()) {
                self.insert(traj_id.clone(), *obs.id());
                *tonight_counts.entry(traj_id.clone()).or_insert(0) += 1;
            }
        }
        for (traj_id, count) in tonight_counts {
            let is_new_best = self
                .max_same_night_count
                .get(&traj_id)
                .is_none_or(|&best| count > best);
            if is_new_best {
                self.max_same_night_count.insert(traj_id.clone(), count);
                self.best_same_night_step.insert(traj_id, step);
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

    /// Ids of every trajectory with >= 2 observations seen so far — see
    /// [`Self::n_multi_detection_so_far`].
    pub fn multi_detection_traj_ids(&self) -> impl Iterator<Item = &TrajId> {
        self.seen
            .iter()
            .filter(|(_, obs)| obs.len() >= 2)
            .map(|(traj_id, _)| traj_id)
    }

    /// Whether `traj_id` ever had >= 2 observations within a *single* night
    /// over the run so far. Seeding in this pipeline is strictly intra-night
    /// (see `fink_fat_engine::topocentric_kf::branching::discovery`), so an
    /// object that never satisfies this can never be picked up by any
    /// lineage no matter how its total observation count grows across
    /// nights — it is structurally unreachable, not a tracking failure.
    pub fn is_trackable(&self, traj_id: &TrajId) -> bool {
        self.max_same_night_count
            .get(traj_id)
            .is_some_and(|&c| c >= 2)
    }

    /// Ids of every trajectory that is [`Self::is_trackable`] so far — the
    /// population whose "never touched" outcome is an actual seeding/gating
    /// failure rather than an architectural impossibility.
    pub fn trackable_traj_ids(&self) -> impl Iterator<Item = &TrajId> {
        self.max_same_night_count
            .iter()
            .filter(|&(_, &c)| c >= 2)
            .map(|(traj_id, _)| traj_id)
    }

    /// Number of trackable trajectories so far — see [`Self::is_trackable`].
    pub fn n_trackable_so_far(&self) -> usize {
        self.trackable_traj_ids().count()
    }

    /// `step` of the night where `traj_id` reached its best same-night
    /// observation count, if it was ever observed at all.
    pub fn best_same_night_step(&self, traj_id: &TrajId) -> Option<usize> {
        self.best_same_night_step.get(traj_id).copied()
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
