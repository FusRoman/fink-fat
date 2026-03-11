use std::fmt;

use ahash::{AHashMap, AHashSet};
use anyhow::Result;
use fink_fat_engine::{night_id::NightId, pipeline::PipelineContext};

use crate::truth_sso::{SeedClass, TrajId, TruthSSOMap, classify_seed, get_truth_traj_id};

// ─────────────────────────────────────────────────────────────────────────────
// Seed quality statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Quality statistics for a collection of seeds.
#[derive(Debug, Default, Clone, Copy)]
pub struct SeedStats {
    /// Total number of seeds.
    pub n_seeds: usize,
    /// Seeds where every member alert belongs to the same ground-truth trajectory.
    pub n_true_positive: usize,
    /// Seeds where member alerts belong to more than one ground-truth trajectory.
    pub n_false_positive: usize,
    /// Seeds where at least one member alert is absent from the truth map.
    pub n_unknown: usize,
    /// Number of ground-truth trajectories with ≥ 2 alerts on this night
    /// (i.e., trajectories that *could* have produced at least one seed).
    pub n_recoverable_trajs: usize,
    /// Number of recoverable trajectories for which at least one TP seed was produced.
    pub n_recovered_trajs: usize,
}

impl SeedStats {
    /// Purity: fraction of classifiable seeds that are true positives.
    ///
    /// Seeds with unknown truth are excluded from the denominator.
    /// Returns `f64::NAN` when no classifiable seed exists.
    pub fn purity(&self) -> f64 {
        let classifiable = self.n_true_positive + self.n_false_positive;
        if classifiable == 0 {
            f64::NAN
        } else {
            self.n_true_positive as f64 / classifiable as f64
        }
    }

    /// Recall: fraction of recoverable ground-truth trajectories covered by
    /// at least one TP seed.
    ///
    /// Returns `f64::NAN` when there are no recoverable trajectories.
    pub fn recall(&self) -> f64 {
        if self.n_recoverable_trajs == 0 {
            f64::NAN
        } else {
            self.n_recovered_trajs as f64 / self.n_recoverable_trajs as f64
        }
    }

    fn add_assign(&mut self, other: SeedStats) {
        self.n_seeds += other.n_seeds;
        self.n_true_positive += other.n_true_positive;
        self.n_false_positive += other.n_false_positive;
        self.n_unknown += other.n_unknown;
        self.n_recoverable_trajs += other.n_recoverable_trajs;
        self.n_recovered_trajs += other.n_recovered_trajs;
    }
}

impl fmt::Display for SeedStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "seeds={n:>8}  TP={tp:>8}  FP={fp:>8}  unknown={unk:>8}  purity={pur:.4}  recall={rec:.4} ({rec_n}/{rec_d})",
            n = self.n_seeds,
            tp = self.n_true_positive,
            fp = self.n_false_positive,
            unk = self.n_unknown,
            pur = self.purity(),
            rec = self.recall(),
            rec_n = self.n_recovered_trajs,
            rec_d = self.n_recoverable_trajs,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Evaluation entry-point
// ─────────────────────────────────────────────────────────────────────────────

pub fn seeding_evaluation(ctx: &PipelineContext, truth: &TruthSSOMap) -> Result<()> {
    let seed_store = &ctx.runtime_state.seed_store;
    let alert_store = &ctx.runtime_state.alert_store;

    let mut nights: Vec<&NightId> = seed_store.nights().collect();
    nights.sort();

    let mut global = SeedStats::default();

    tracing::info!("Seeding evaluation");
    tracing::info!("{:-<72}", "");
    tracing::info!(
        "{:<10}  {}",
        "night",
        "seeds        TP        FP   unknown    purity"
    );
    tracing::info!("{:-<72}", "");

    for night_id in &nights {
        let seeds = match seed_store.get(night_id) {
            Some(s) => s,
            None => continue,
        };

        // ── Recoverable trajectories ──────────────────────────────────────────
        // Count alerts per traj_id on this night; any traj with ≥ 2 alerts
        // could in principle produce at least one seed.
        let mut traj_alert_count: AHashMap<TrajId, usize> = AHashMap::new();
        if let Some(night_alerts) = alert_store.get(night_id) {
            for alert in night_alerts {
                if let Some(traj_id) = get_truth_traj_id(truth, alert) {
                    *traj_alert_count.entry(traj_id).or_insert(0) += 1;
                }
            }
        }
        let recoverable: AHashSet<TrajId> = traj_alert_count
            .into_iter()
            .filter_map(|(traj_id, count)| (count >= 2).then_some(traj_id))
            .collect();

        // ── Classify seeds and collect recovered trajectories ─────────────────
        let mut night_stats = SeedStats {
            n_recoverable_trajs: recoverable.len(),
            ..SeedStats::default()
        };
        let mut recovered: AHashSet<TrajId> = AHashSet::new();

        for seed in seeds {
            night_stats.n_seeds += 1;

            let resolved = seed.resolve_members(alert_store).unwrap_or_default();

            let class = classify_seed(truth, &resolved);

            match class {
                SeedClass::TruePositive => {
                    night_stats.n_true_positive += 1;
                    // Record which trajectory this TP seed covers.
                    if let Some(traj_id) =
                        resolved.first().and_then(|a| get_truth_traj_id(truth, a))
                    {
                        recovered.insert(traj_id);
                    }
                }
                SeedClass::FalsePositive => night_stats.n_false_positive += 1,
                SeedClass::Unknown => night_stats.n_unknown += 1,
            }
        }

        // Only count trajectories that were actually recoverable.
        night_stats.n_recovered_trajs = recovered.intersection(&recoverable).count();

        tracing::info!("{:<10}  {}", night_id, night_stats);
        global.add_assign(night_stats);
    }

    tracing::info!("{:-<72}", "");
    tracing::info!("{:<10}  {}", "TOTAL", global);

    Ok(())
}
