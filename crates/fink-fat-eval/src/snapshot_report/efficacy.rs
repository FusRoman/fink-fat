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
    population::Population,
    seed_bank_report::ground_truth::{ObsTrajLookup, SeedPurity},
    tracking_report::gold_trajectory::GoldTrajectoryTracker,
    trajectory_processing::{MetricStats, fmt_stats, metric_stats},
};

/// Final classification of one ground-truth trajectory against the
/// snapshot's branches. Checked in this order (most-actionable first):
/// `Fragmented` > `Contaminated` > `Partial` > `Perfect` > `NotReconstructed`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconstructionOutcome {
    /// No branch's `track_ids()` contains any observation of this
    /// trajectory.
    NotReconstructed,
    /// This trajectory's observations are split across >= 2 distinct
    /// branches.
    Fragmented(Box<ReconstructionOutcome>),
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
    pub fn label(&self) -> String {
        match self {
            Self::NotReconstructed => "not reconstructed".to_string(),
            Self::Fragmented(sub) => format!("fragmented({})", sub.label()),
            Self::Contaminated => "contaminated".to_string(),
            Self::Partial => "partial".to_string(),
            Self::Perfect => "perfect".to_string(),
        }
    }

    pub fn all() -> [ReconstructionOutcome; 6] {
        [
            Self::NotReconstructed,
            Self::Fragmented(Box::new(Self::Perfect)),
            Self::Fragmented(Box::new(Self::Contaminated)),
            Self::Contaminated,
            Self::Partial,
            Self::Perfect,
        ]
    }
}

/// Reconstruction efficacy restricted to one orbital-class population — the
/// per-population breakdown of the global `counts`, so exotic (non-MBA)
/// completeness/contamination is read directly instead of drowned in the MBA
/// majority.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationEfficacy {
    /// Human-readable population name (see [`Population::label`]).
    pub label: String,
    /// Per-outcome counts, same 6-slot layout as [`ReconstructionOutcome::all`].
    pub counts: [usize; 6],
    /// Sum of distinct touching branches over *touched* trajectories of this
    /// population (mean over-generation = `branches_touching_sum / n_touched`).
    pub branches_touching_sum: usize,
    /// Number of touched trajectories of this population (over-gen denominator).
    pub n_touched: usize,
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
    pub counts: [usize; 6],
    /// Multi-detection trajectories excluded from `counts` because they
    /// never had >= 2 observations within a single night — structurally
    /// unreachable by this pipeline's intra-night-only seeding, not a
    /// tracking failure (see [`GoldTrajectoryTracker::is_trackable`]).
    pub n_excluded_unreachable: usize,
    pub coverage_samples: Vec<f64>,
    pub n_pure_branches: usize,
    pub n_mixed_branches: usize,
    pub n_unknown_branches: usize,
    /// Distinct branches touching each trajectory that was touched by at
    /// least one branch (`NotReconstructed` trajectories are excluded, not
    /// zero-filled) — the basis for the per-trajectory over-generation
    /// stat, distinct from the global `n_branches / n_trackable` ratio
    /// which also folds in branches that never touch any trackable
    /// trajectory at all.
    pub branches_per_trajectory_samples: Vec<usize>,
    pub branches_per_trajectory: MetricStats,
    /// Per-orbital-class breakdown of `counts`, in [`Population::all`] order.
    /// Empty populations are still listed (all-zero) so the table shape is
    /// stable across runs.
    pub per_population: Vec<PopulationEfficacy>,
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
    traj_population: &AHashMap<TrajId, Population>,
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
    let mut counts = [0usize; 6];
    let mut coverage_samples = Vec::new();
    let mut branches_per_trajectory_samples = Vec::new();
    // Per-population accumulators (counts + over-generation sums).
    let mut pop_counts: AHashMap<Population, [usize; 6]> = AHashMap::default();
    let mut pop_branches_sum: AHashMap<Population, usize> = AHashMap::default();
    let mut pop_n_touched: AHashMap<Population, usize> = AHashMap::default();

    for traj_id in &trackable_ids {
        let Some(total) = gold_tracker.n_obs_so_far(traj_id) else {
            continue;
        };
        let branches_touching = touching_branches.get(traj_id);
        let n_distinct = branches_touching.map(|s| s.len()).unwrap_or(0);
        if n_distinct >= 1 {
            branches_per_trajectory_samples.push(n_distinct);
        }

        let outcome = if n_distinct == 0 {
            ReconstructionOutcome::NotReconstructed
        } else {
            let branch_idx = *branches_touching.unwrap().iter().next().unwrap();
            let is_pure = matches!(
                ground_truth.classify(collection.branches[branch_idx].track_ids()),
                SeedPurity::Pure(ref t) if t == traj_id
            );

            if is_pure {
                if n_distinct >= 2 {
                    ReconstructionOutcome::Fragmented(Box::new(ReconstructionOutcome::Perfect))
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
            } else {
                if n_distinct >= 2 {
                    ReconstructionOutcome::Fragmented(Box::new(ReconstructionOutcome::Contaminated))
                } else {
                    ReconstructionOutcome::Contaminated
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

        // Per-population accumulation.
        let pop = traj_population
            .get(traj_id)
            .copied()
            .unwrap_or(Population::Unknown);
        pop_counts.entry(pop).or_insert([0; 6])[idx] += 1;
        if n_distinct >= 1 {
            *pop_branches_sum.entry(pop).or_insert(0) += n_distinct;
            *pop_n_touched.entry(pop).or_insert(0) += 1;
        }

        per_trajectory.push((traj_id.to_string(), outcome.label().to_string(), coverage));
    }

    let per_population = Population::all()
        .into_iter()
        .map(|pop| PopulationEfficacy {
            label: pop.label().to_string(),
            counts: pop_counts.get(&pop).copied().unwrap_or([0; 6]),
            branches_touching_sum: pop_branches_sum.get(&pop).copied().unwrap_or(0),
            n_touched: pop_n_touched.get(&pop).copied().unwrap_or(0),
        })
        .collect();

    let branches_per_trajectory = metric_stats(&branches_per_trajectory_samples, |&v| v as f64);

    ReconstructionEfficacy {
        per_trajectory,
        counts,
        n_excluded_unreachable,
        coverage_samples,
        n_pure_branches: n_pure,
        n_mixed_branches: n_mixed,
        n_unknown_branches: n_unknown,
        branches_per_trajectory_samples,
        branches_per_trajectory,
        per_population,
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

        let n_complete: usize = [
            ReconstructionOutcome::Perfect,
            ReconstructionOutcome::Partial,
            ReconstructionOutcome::Fragmented(Box::new(ReconstructionOutcome::Perfect)),
        ]
        .iter()
        .map(|o| {
            let idx = ReconstructionOutcome::all()
                .iter()
                .position(|x| x == o)
                .unwrap();
            self.counts[idx]
        })
        .sum();
        let completeness_pct = if total == 0 {
            0.0
        } else {
            100.0 * n_complete as f64 / total as f64
        };
        println!(
            "  completeness (pure-only outcomes): {n_complete} / {total} ({completeness_pct:.1}%)"
        );

        let overgen_global = if total == 0 {
            0.0
        } else {
            n_branches as f64 / total as f64
        };
        println!(
            "  branch over-generation, global: {n_branches} branches / {total} trackable trajectories ({overgen_global:.2}x)"
        );
        println!(
            "  branch over-generation, per touched trajectory: {}",
            fmt_stats(&self.branches_per_trajectory)
        );

        // Per-population breakdown — the exotic-vs-MBA readout the whole
        // evaluation is really about. completeness = pure-only outcomes
        // (Perfect + Partial + Fragmented(Perfect)); contaminated = any
        // mixed-branch outcome (Contaminated + Fragmented(Contaminated)).
        let idx_of = |o: &ReconstructionOutcome| {
            ReconstructionOutcome::all()
                .iter()
                .position(|x| x == o)
                .unwrap()
        };
        let complete_idx = [
            idx_of(&ReconstructionOutcome::Perfect),
            idx_of(&ReconstructionOutcome::Partial),
            idx_of(&ReconstructionOutcome::Fragmented(Box::new(
                ReconstructionOutcome::Perfect,
            ))),
        ];
        let contam_idx = [
            idx_of(&ReconstructionOutcome::Contaminated),
            idx_of(&ReconstructionOutcome::Fragmented(Box::new(
                ReconstructionOutcome::Contaminated,
            ))),
        ];
        println!("  per-population breakdown:");
        println!(
            "    {:<20} {:>9} {:>16} {:>16} {:>9}",
            "population", "trackable", "completeness", "contaminated", "over-gen"
        );
        for pe in &self.per_population {
            let pt: usize = pe.counts.iter().sum();
            if pt == 0 {
                continue;
            }
            let comp: usize = complete_idx.iter().map(|&i| pe.counts[i]).sum();
            let contam: usize = contam_idx.iter().map(|&i| pe.counts[i]).sum();
            let overgen = if pe.n_touched == 0 {
                0.0
            } else {
                pe.branches_touching_sum as f64 / pe.n_touched as f64
            };
            println!(
                "    {:<20} {:>9} {:>7} ({:>5.1}%) {:>7} ({:>5.1}%) {:>7.2}x",
                pe.label,
                pt,
                comp,
                100.0 * comp as f64 / pt as f64,
                contam,
                100.0 * contam as f64 / pt as f64,
                overgen,
            );
        }
        println!("{sep}\n");
    }

    pub fn write_json(&self, path: &Utf8Path) -> Result<()> {
        let file =
            std::fs::File::create(path).with_context(|| format!("failed to create {path}"))?;
        serde_json::to_writer_pretty(file, self)
            .with_context(|| format!("failed to write JSON to {path}"))
    }
}
