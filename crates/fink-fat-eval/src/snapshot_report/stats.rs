//! Basic structural/distributional stats over a loaded `BranchCollection`,
//! independent of ground truth.

use ahash::{AHashMap, AHashSet};
use anyhow::{Context, Result};
use camino::Utf8Path;
use photom::{
    NightId,
    observation_dataset::{ObsDataset, ObsId},
};
use serde::{Deserialize, Serialize};

use fink_fat_engine::topocentric_kf::branching::BranchCollection;

use crate::trajectory_processing::{MetricStats, fmt_stats, metric_stats};

/// `ObsId -> NightId` reverse lookup, built once from `obs_dataset` — there
/// is no such accessor on `ObsDataset` itself (only `iter_night_id`/
/// `iter_night_observations`/`iter_full_night`).
pub fn build_obs_to_night_map(obs_dataset: &ObsDataset) -> AHashMap<ObsId, NightId> {
    let Some(pairs) = obs_dataset.iter_full_night() else {
        return AHashMap::default();
    };
    pairs.map(|(night_id, obs)| (*obs.id(), night_id)).collect()
}

/// Structural/distributional snapshot stats — no ground truth required.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotStats {
    pub n_branches: usize,
    pub hypotheses_per_branch: MetricStats,
    pub hypotheses_per_branch_samples: Vec<f64>,
    pub track_length_per_branch: MetricStats,
    pub track_length_per_branch_samples: Vec<f64>,
    pub n_unique_observations_referenced: usize,
    pub n_distinct_nights_spanned: usize,
    pub min_night: Option<u32>,
    pub max_night: Option<u32>,
    /// Per-night observation count, restricted to nights actually
    /// referenced by some branch's `track_ids()` — a cross-sectional
    /// breakdown of *this snapshot's* footprint, not the whole dataset's.
    pub observations_per_night: Vec<(u32, usize)>,
}

pub fn compute_snapshot_stats(
    collection: &BranchCollection<'_, '_>,
    obs_to_night: &AHashMap<ObsId, NightId>,
) -> SnapshotStats {
    let hyp_samples: Vec<f64> = collection
        .branches
        .iter()
        .map(|b| b.bank.hypotheses().len() as f64)
        .collect();
    let len_samples: Vec<f64> = collection
        .branches
        .iter()
        .map(|b| b.track_ids().len() as f64)
        .collect();

    let mut unique_obs: AHashSet<ObsId> = AHashSet::default();
    let mut per_night_counts: AHashMap<NightId, usize> = AHashMap::default();
    for branch in &collection.branches {
        for &obs_id in branch.track_ids() {
            unique_obs.insert(obs_id);
            if let Some(&night_id) = obs_to_night.get(&obs_id) {
                *per_night_counts.entry(night_id).or_insert(0) += 1;
            }
        }
    }

    let mut observations_per_night: Vec<(u32, usize)> = per_night_counts
        .into_iter()
        .map(|(n, c)| (n.0, c))
        .collect();
    observations_per_night.sort_unstable_by_key(|&(n, _)| n);

    SnapshotStats {
        n_branches: collection.branches.len(),
        hypotheses_per_branch: metric_stats(&hyp_samples, |&v| v),
        hypotheses_per_branch_samples: hyp_samples,
        track_length_per_branch: metric_stats(&len_samples, |&v| v),
        track_length_per_branch_samples: len_samples,
        n_unique_observations_referenced: unique_obs.len(),
        n_distinct_nights_spanned: observations_per_night.len(),
        min_night: observations_per_night.first().map(|&(n, _)| n),
        max_night: observations_per_night.last().map(|&(n, _)| n),
        observations_per_night,
    }
}

impl SnapshotStats {
    pub fn write_json(&self, path: &Utf8Path) -> Result<()> {
        let file =
            std::fs::File::create(path).with_context(|| format!("failed to create {path}"))?;
        serde_json::to_writer_pretty(file, self)
            .with_context(|| format!("failed to write JSON to {path}"))
    }

    /// Console summary, matching `AggregatedTrackingStats::print_summary`'s
    /// style.
    pub fn print_summary(&self) {
        let sep = "=".repeat(78);
        println!("\n{sep}");
        println!("Snapshot structural stats");
        println!("{sep}");
        println!(
            "  Branches                                    : {}",
            self.n_branches
        );
        println!(
            "  Hypotheses per branch                       : {}",
            fmt_stats(&self.hypotheses_per_branch)
        );
        println!(
            "  Observations per branch                     : {}",
            fmt_stats(&self.track_length_per_branch)
        );
        println!(
            "  Unique observations referenced               : {}",
            self.n_unique_observations_referenced
        );
        println!(
            "  Nights spanned (min, max)                   : {} ({:?}, {:?})",
            self.n_distinct_nights_spanned, self.min_night, self.max_night
        );
        println!("{sep}\n");
    }
}
