//! Reconstruction-efficacy classification: how well a single
//! `BranchCollection` snapshot accounts for every known ground-truth
//! trajectory, restricted to the nights the snapshot actually covers (see
//! [`determine_last_processed_night`]) and to trajectories that are
//! structurally reachable by this pipeline's intra-night-only seeding (see
//! [`GoldTrajectoryTracker::is_trackable`]). Only meaningful when
//! [`ObsTrajLookup::has_ground_truth`] is true.

use ahash::{AHashMap, AHashSet};
use anyhow::{Context, Result};
use camino::Utf8Path;
use photom::{
    NightId, TrajId,
    observation_dataset::{ObsDataset, ObsId, observation::Observation},
};
use serde::{Deserialize, Serialize};

use fink_fat_engine::topocentric_kf::branching::BranchCollection;

use crate::{
    seed_bank_report::ground_truth::{ObsTrajLookup, SeedPurity},
    tracking_report::gold_trajectory::GoldTrajectoryTracker,
};

/// Final classification of one ground-truth trajectory against the
/// snapshot's branches. Checked in this order (most-actionable first):
/// `Fragmented` > `Contaminated` > `Partial` > `Perfect` > `NotReconstructed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconstructionOutcome {
    /// No branch's `track_ids()` contains any observation of this
    /// trajectory.
    NotReconstructed,
    /// This trajectory's observations are split across >= 2 distinct
    /// branches.
    Fragmented,
    /// Touched by exactly one branch, and that branch is impure (mixes in
    /// another trajectory's observations).
    Contaminated,
    /// Touched by exactly one, pure, branch, covering < 100% of this
    /// trajectory's known observations (restricted to processed nights).
    Partial,
    /// Touched by exactly one, pure, branch, covering exactly 100% of this
    /// trajectory's known observations (restricted to processed nights).
    Perfect,
}

impl ReconstructionOutcome {
    pub fn label(self) -> &'static str {
        match self {
            Self::NotReconstructed => "not reconstructed",
            Self::Fragmented => "fragmented",
            Self::Contaminated => "contaminated",
            Self::Partial => "partial",
            Self::Perfect => "perfect",
        }
    }

    pub fn all() -> [ReconstructionOutcome; 5] {
        [
            Self::NotReconstructed,
            Self::Fragmented,
            Self::Contaminated,
            Self::Partial,
            Self::Perfect,
        ]
    }
}

/// Dataset-wide reconstruction-efficacy report for one snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconstructionEfficacy {
    /// One row per classified ground-truth trajectory:
    /// `(traj_id.to_string(), outcome.label(), coverage_ratio)`. `TrajId`
    /// isn't serde-friendly in this crate (the `photom` `serde` feature
    /// isn't enabled here), but it implements `Display`, so the id is
    /// stored as its string form.
    pub per_trajectory: Vec<(String, String, Option<f64>)>,
    pub counts: [usize; 5],
    /// Multi-detection trajectories excluded from `counts` because they
    /// never had >= 2 observations within a single night — structurally
    /// unreachable by this pipeline's intra-night-only seeding, not a
    /// tracking failure (see [`GoldTrajectoryTracker::is_trackable`]).
    pub n_excluded_unreachable: usize,
    pub coverage_samples: Vec<f64>,
    pub n_pure_branches: usize,
    pub n_mixed_branches: usize,
    pub n_unknown_branches: usize,
    pub last_processed_night: Option<u32>,
}

/// Last night actually referenced by any live branch's `track_ids()` — the
/// snapshot itself carries no night metadata, only opaque `ObsId`s, so this
/// is the closest proxy for "which night was this snapshot produced after".
///
/// Limitation: if the true last processed night had every one of its
/// objects pruned away (no live branch references any of its observations
/// anymore), this underestimates it by one night or more — accepted as a
/// known limitation since the information doesn't exist anywhere else in
/// the `.rkyv` file.
pub fn determine_last_processed_night(
    collection: &BranchCollection<'_, '_>,
    obs_to_night: &AHashMap<ObsId, NightId>,
) -> Option<NightId> {
    collection
        .branches
        .iter()
        .flat_map(|b| b.track_ids())
        .filter_map(|id| obs_to_night.get(id))
        .copied()
        .max()
}

/// Build a [`GoldTrajectoryTracker`] restricted to the nights the pipeline
/// actually processed to produce this snapshot: every night in
/// `obs_dataset` with `night_id <= last_processed_night`, in order — this
/// assumes (matching how the real engine binary runs) that nights are
/// processed sequentially from the dataset's first night, with no gaps.
pub fn build_gold_tracker_for_processed_nights(
    obs_dataset: &ObsDataset,
    ground_truth: &ObsTrajLookup,
    last_processed_night: NightId,
) -> GoldTrajectoryTracker {
    let mut night_ids: Vec<NightId> = obs_dataset
        .iter_night_id()
        .into_iter()
        .flatten()
        .copied()
        .filter(|id| *id <= last_processed_night)
        .collect();
    night_ids.sort_unstable();

    let mut gold_tracker = GoldTrajectoryTracker::new();
    for (step, night_id) in night_ids.iter().enumerate() {
        let night_obs: Vec<&Observation> = obs_dataset
            .iter_night_observations(night_id)
            .into_iter()
            .flatten()
            .collect();
        gold_tracker.observe_night(step, &night_obs, ground_truth);
    }
    gold_tracker
}

pub fn compute_reconstruction_efficacy(
    collection: &BranchCollection<'_, '_>,
    ground_truth: &ObsTrajLookup,
    gold_tracker: &GoldTrajectoryTracker,
    last_processed_night: Option<NightId>,
) -> ReconstructionEfficacy {
    // Branch purity + which branches touch which trajectory.
    let mut touching_branches: AHashMap<TrajId, AHashSet<usize>> = AHashMap::default();
    let mut covered_obs_by_traj: AHashMap<TrajId, AHashSet<ObsId>> = AHashMap::default();
    let (mut n_pure, mut n_mixed, mut n_unknown) = (0usize, 0usize, 0usize);

    for (branch_idx, branch) in collection.branches.iter().enumerate() {
        match ground_truth.classify(branch.track_ids()) {
            SeedPurity::Pure(traj_id) => {
                n_pure += 1;
                touching_branches
                    .entry(traj_id.clone())
                    .or_default()
                    .insert(branch_idx);
                covered_obs_by_traj
                    .entry(traj_id)
                    .or_default()
                    .extend(branch.track_ids().iter().copied());
            }
            SeedPurity::Mixed => {
                n_mixed += 1;
                for &obs_id in branch.track_ids() {
                    if let Some(traj_id) = ground_truth.traj_of(obs_id) {
                        touching_branches
                            .entry(traj_id.clone())
                            .or_default()
                            .insert(branch_idx);
                    }
                }
            }
            SeedPurity::Unknown => n_unknown += 1,
        }
    }

    let trackable_ids: AHashSet<TrajId> = gold_tracker.trackable_traj_ids().cloned().collect();
    let n_excluded_unreachable = gold_tracker
        .multi_detection_traj_ids()
        .filter(|id| !trackable_ids.contains(id))
        .count();

    let mut per_trajectory = Vec::new();
    let mut counts = [0usize; 5];
    let mut coverage_samples = Vec::new();

    for traj_id in &trackable_ids {
        let Some(total) = gold_tracker.n_obs_so_far(traj_id) else {
            continue;
        };
        let branches_touching = touching_branches.get(traj_id);
        let n_distinct = branches_touching.map(|s| s.len()).unwrap_or(0);

        let outcome = if n_distinct == 0 {
            ReconstructionOutcome::NotReconstructed
        } else if n_distinct >= 2 {
            ReconstructionOutcome::Fragmented
        } else {
            let branch_idx = *branches_touching.unwrap().iter().next().unwrap();
            let is_pure = matches!(
                ground_truth.classify(collection.branches[branch_idx].track_ids()),
                SeedPurity::Pure(ref t) if t == traj_id
            );
            if !is_pure {
                ReconstructionOutcome::Contaminated
            } else {
                let covered = covered_obs_by_traj
                    .get(traj_id)
                    .map(|s| s.len())
                    .unwrap_or(0);
                if covered >= total {
                    ReconstructionOutcome::Perfect
                } else {
                    ReconstructionOutcome::Partial
                }
            }
        };

        let coverage = match outcome {
            ReconstructionOutcome::Perfect | ReconstructionOutcome::Partial => {
                let covered = covered_obs_by_traj
                    .get(traj_id)
                    .map(|s| s.len())
                    .unwrap_or(0);
                let ratio = covered as f64 / total.max(1) as f64;
                coverage_samples.push(ratio);
                Some(ratio)
            }
            _ => {
                coverage_samples.push(0.0);
                None
            }
        };

        let idx = ReconstructionOutcome::all()
            .iter()
            .position(|o| *o == outcome)
            .unwrap();
        counts[idx] += 1;
        per_trajectory.push((traj_id.to_string(), outcome.label().to_string(), coverage));
    }

    ReconstructionEfficacy {
        per_trajectory,
        counts,
        n_excluded_unreachable,
        coverage_samples,
        n_pure_branches: n_pure,
        n_mixed_branches: n_mixed,
        n_unknown_branches: n_unknown,
        last_processed_night: last_processed_night.map(|n| n.0),
    }
}

impl ReconstructionEfficacy {
    pub fn print_summary(&self) {
        let total: usize = self.counts.iter().sum();
        let sep = "=".repeat(78);
        println!("\n{sep}");
        println!(
            "Reconstruction efficacy — {total} trackable trajectories \
             (last processed night: {:?}; {} multi-detection trajectories \
             excluded as structurally unreachable — never had >= 2 \
             observations within a single night)",
            self.last_processed_night, self.n_excluded_unreachable
        );
        println!("{sep}");
        for (i, outcome) in ReconstructionOutcome::all().into_iter().enumerate() {
            let count = self.counts[i];
            let pct = if total == 0 {
                0.0
            } else {
                100.0 * count as f64 / total as f64
            };
            println!("  {:<20} : {count:>6} ({pct:>5.1}%)", outcome.label());
        }
        let n_branches = self.n_pure_branches + self.n_mixed_branches + self.n_unknown_branches;
        let pure_pct = if n_branches == 0 {
            0.0
        } else {
            100.0 * self.n_pure_branches as f64 / n_branches as f64
        };
        println!(
            "  branch purity: {} pure / {} mixed / {} unknown ({pure_pct:.1}% pure)",
            self.n_pure_branches, self.n_mixed_branches, self.n_unknown_branches
        );
        println!("{sep}\n");
    }

    pub fn write_json(&self, path: &Utf8Path) -> Result<()> {
        let file =
            std::fs::File::create(path).with_context(|| format!("failed to create {path}"))?;
        serde_json::to_writer_pretty(file, self)
            .with_context(|| format!("failed to write JSON to {path}"))
    }
}
