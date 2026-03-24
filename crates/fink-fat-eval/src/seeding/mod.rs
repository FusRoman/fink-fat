pub mod export;
pub mod plots;

use std::fmt;

use ahash::AHashSet;
use anyhow::Result;
use fink_fat_engine::{night_id::NightId, pipeline::PipelineContext};

use crate::truth_sso::{TrajId, TruthClass, TruthSSO};

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
// Stats computation (shared by logging and plotting)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute per-night seed statistics for all nights in the seed store.
///
/// Returns `(global, per_night)` where `per_night` is sorted by night ID and
/// each element is `(night_label, SeedStats)`.
///
/// This is the shared core used by both [`seeding_evaluation`] (for logging)
/// and [`plots::seeding_plots`] (for plotting).
pub fn compute_seeding_stats(
    ctx: &PipelineContext,
    truth: &TruthSSO,
) -> Result<(SeedStats, Vec<(String, SeedStats)>)> {
    let seed_store = &ctx.runtime_state.seed_store;
    let alert_store = &ctx.runtime_state.alert_store;

    let mut nights: Vec<&NightId> = seed_store.nights().collect();
    nights.sort();

    let mut global = SeedStats::default();
    let mut per_night: Vec<(String, SeedStats)> = Vec::with_capacity(nights.len());

    for night_id in nights {
        let seeds = match seed_store.get(night_id) {
            Some(s) => s,
            None => continue,
        };

        // ── Recoverable trajectories ──────────────────────────────────────────
        let recoverable: AHashSet<TrajId> = truth.recoverable_seeds(*night_id, 2).collect();

        // ── Classify seeds ────────────────────────────────────────────────────
        let mut night_stats = SeedStats {
            n_recoverable_trajs: recoverable.len(),
            ..SeedStats::default()
        };
        let mut recovered: AHashSet<TrajId> = AHashSet::new();

        for seed in seeds {
            night_stats.n_seeds += 1;
            let resolved = match seed.resolve_members(alert_store) {
                Ok(resolved) => resolved,
                Err(_) => {
                    night_stats.n_unknown += 1;
                    continue;
                }
            };
            let class = truth.classify(&resolved);
            match class {
                TruthClass::TruePositive => {
                    night_stats.n_true_positive += 1;
                    if let Some(traj_id) = resolved.first().and_then(|a| truth.get_truth_traj_id(a))
                    {
                        recovered.insert(traj_id);
                    }
                }
                TruthClass::FalsePositive => night_stats.n_false_positive += 1,
                TruthClass::Unknown => night_stats.n_unknown += 1,
            }
        }
        night_stats.n_recovered_trajs = recovered.intersection(&recoverable).count();
        global.add_assign(night_stats);
        per_night.push((night_id.to_string(), night_stats));
    }

    Ok((global, per_night))
}

// ─────────────────────────────────────────────────────────────────────────────
// Logging entry-point
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate seeding quality and write a summary to the tracing log.
///
/// Calls [`compute_seeding_stats`] internally.  For plot output, use
/// [`plots::seeding_plots`] in addition.
pub fn seeding_evaluation(ctx: &PipelineContext, truth: &TruthSSO) -> Result<()> {
    let (global, per_night) = compute_seeding_stats(ctx, truth)?;

    tracing::info!("Seeding evaluation");
    tracing::info!("{:-<72}", "");
    tracing::info!(
        "{:<10}  {}",
        "night",
        "seeds        TP        FP   unknown    purity"
    );
    tracing::info!("{:-<72}", "");

    for (label, stats) in &per_night {
        tracing::info!("{:<10}  {}", label, stats);
    }

    tracing::info!("{:-<72}", "");
    tracing::info!("{:<10}  {}", "TOTAL", global);

    Ok(())
}
