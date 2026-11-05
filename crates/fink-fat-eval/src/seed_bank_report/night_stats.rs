//! Per-night intra-night seeding statistics.

use ahash::{AHashMap, AHashSet};
use fink_fat_engine::topocentric_kf::branching::Branch;
use photom::{NightId, TrajId, observation_dataset::observation::Observation};

use crate::seed_bank_report::ground_truth::{ObsTrajLookup, SeedPurity};

/// Seeding recall/purity/volume statistics for a single night.
#[derive(Debug, Clone)]
pub struct NightSeedingStats {
    pub night_id: NightId,
    pub n_observations: usize,
    /// Ground-truth trajectories with ≥2 observations on this night — the
    /// set a "fully successful" night should produce at least one pure seed
    /// for.
    pub n_multi_detection_objects: usize,
    /// Of `n_multi_detection_objects`, how many have ≥1 **pure** seed
    /// branch (a genuinely correct, uncontaminated candidate lineage).
    pub n_objects_recalled: usize,
    /// Of `n_multi_detection_objects`, how many appear in *any* seed
    /// branch, pure or mixed — a looser "did we get a candidate at all"
    /// signal.
    pub n_objects_touched: usize,
    /// Total seed branches (= Kalman banks) produced this night.
    pub n_branches: usize,
    /// Of `n_branches`, how many are pure (single true object).
    pub n_pure_branches: usize,
    /// Live hypothesis count of each produced bank, in branch order.
    pub hypotheses_per_branch: Vec<usize>,
}

impl NightSeedingStats {
    pub fn recall_pct(&self) -> f64 {
        percentage(self.n_objects_recalled, self.n_multi_detection_objects)
    }

    pub fn loose_recall_pct(&self) -> f64 {
        percentage(self.n_objects_touched, self.n_multi_detection_objects)
    }

    pub fn purity_pct(&self) -> f64 {
        percentage(self.n_pure_branches, self.n_branches)
    }
}

/// `100 * numerator / denominator`, or `f64::NAN` when there is nothing to
/// measure (`denominator == 0`) — callers format `NAN` as "n/a" rather than
/// a misleading `0%` or `100%`. Shared with [`super::report::SeedingReport`]'s
/// dataset-wide percentages.
pub(crate) fn percentage(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        f64::NAN
    } else {
        100.0 * numerator as f64 / denominator as f64
    }
}

/// Compute [`NightSeedingStats`] for one night's seeding output.
///
/// `night_obs` is this night's full observation set (used to determine
/// which ground-truth trajectories had ≥2 detections that night);
/// `branches` is the seed set [`BranchCollection::advance_one_night`]
/// produced from it.
pub fn compute_night_stats(
    night_id: NightId,
    night_obs: &[&Observation],
    branches: &[Branch],
    ground_truth: &ObsTrajLookup,
) -> NightSeedingStats {
    let multi_detection_objects = multi_detection_trajectories(night_obs, ground_truth);

    let mut recalled: AHashSet<&TrajId> = AHashSet::default();
    let mut touched: AHashSet<&TrajId> = AHashSet::default();
    let mut n_pure_branches = 0;

    for branch in branches {
        let is_pure = matches!(
            ground_truth.classify(branch.track_ids()),
            SeedPurity::Pure(_)
        );
        n_pure_branches += is_pure as usize;

        // Track "touched"/"recalled" via the ground-truth reference
        // directly (not the owned clone `classify` returns), so both sets
        // can borrow from `ground_truth` with a single shared lifetime.
        for obs_id in branch.track_ids() {
            let Some(traj_id) = ground_truth.traj_of(*obs_id) else {
                continue;
            };
            if !multi_detection_objects.contains(traj_id) {
                continue;
            }
            touched.insert(traj_id);
            if is_pure {
                recalled.insert(traj_id);
            }
        }
    }

    NightSeedingStats {
        night_id,
        n_observations: night_obs.len(),
        n_multi_detection_objects: multi_detection_objects.len(),
        n_objects_recalled: recalled.len(),
        n_objects_touched: touched.len(),
        n_branches: branches.len(),
        n_pure_branches,
        hypotheses_per_branch: branches.iter().map(|b| b.bank.len()).collect(),
    }
}

/// Ground-truth trajectories with ≥2 observations within `night_obs`.
fn multi_detection_trajectories<'a>(
    night_obs: &[&Observation],
    ground_truth: &'a ObsTrajLookup,
) -> AHashSet<&'a TrajId> {
    let mut counts: AHashMap<&TrajId, usize> = AHashMap::default();
    for obs in night_obs {
        if let Some(traj_id) = ground_truth.traj_of(*obs.id()) {
            *counts.entry(traj_id).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .filter(|&(_, count)| count >= 2)
        .map(|(traj_id, _)| traj_id)
        .collect()
}
