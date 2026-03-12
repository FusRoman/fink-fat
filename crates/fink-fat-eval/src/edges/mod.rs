use std::fmt;

use ahash::AHashSet;
use anyhow::{Context, Result};
use fink_fat_engine::pipeline::PipelineContext;

use crate::truth_sso::{TrajId, TruthClass, TruthSSO};

// ─────────────────────────────────────────────────────────────────────────────
// Edge quality statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Quality statistics for a collection of edges.
#[derive(Debug, Default, Clone, Copy)]
pub struct EdgeStats {
    /// Total number of edges.
    pub n_edges: usize,
    /// Edges where every member alert belongs to the same ground-truth trajectory.
    pub n_true_positive: usize,
    /// Edges where member alerts belong to more than one ground-truth trajectory.
    pub n_false_positive: usize,
    /// Edges where at least one member alert is absent from the truth map.
    pub n_unknown: usize,
    /// Number of ground-truth trajectories with ≥ 2 alerts on this night
    /// (i.e., trajectories that *could* have produced at least one edge).
    pub n_recoverable_trajs: usize,
    /// Number of recoverable trajectories for which at least one TP edge was produced.
    pub n_recovered_trajs: usize,
}

impl EdgeStats {
    /// Purity: fraction of classifiable edges that are true positives.
    ///
    /// Edges with unknown truth are excluded from the denominator.
    /// Returns `f64::NAN` when no classifiable edge exists.
    pub fn purity(&self) -> f64 {
        let classifiable = self.n_true_positive + self.n_false_positive;
        if classifiable == 0 {
            f64::NAN
        } else {
            self.n_true_positive as f64 / classifiable as f64
        }
    }

    /// Recall: fraction of recoverable ground-truth trajectories covered by
    /// at least one TP edge.
    ///
    /// Returns `f64::NAN` when there are no recoverable trajectories.
    pub fn recall(&self) -> f64 {
        if self.n_recoverable_trajs == 0 {
            f64::NAN
        } else {
            self.n_recovered_trajs as f64 / self.n_recoverable_trajs as f64
        }
    }
}

impl fmt::Display for EdgeStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "edges={n:>8}  TP={tp:>8}  FP={fp:>8}  unknown={unk:>8}  purity={pur:.4}  recall={rec:.4} ({rec_n}/{rec_d})",
            n = self.n_edges,
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

/// Compute per-night edge statistics for all nights in the seed store.
///
/// Returns `(global, per_night)` where `per_night` is sorted by night ID and
/// each element is `(night_label, EdgeStats)`.
///
/// This is the shared core used by both [`edge_evaluation`] (for logging)
/// and [`plots::edge_plots`] (for plotting).
pub fn compute_edge_stats(ctx: &PipelineContext, truth: &TruthSSO) -> Result<EdgeStats> {
    let seed_store = &ctx.runtime_state.seed_store;
    let alert_store = &ctx.runtime_state.alert_store;

    // ── Recoverable trajectories ──────────────────────────────────────────
    let max_gap = ctx.engine_config.max_gap_nights();
    let recoverable: AHashSet<TrajId> = truth.recoverable_edges(2, max_gap).collect();

    let mut global = EdgeStats {
        n_edges: ctx.runtime_state.graph.edges.len(),
        n_recoverable_trajs: recoverable.len(),
        ..EdgeStats::default()
    };

    let mut recovered: AHashSet<TrajId> = AHashSet::new();

    for edge in ctx.runtime_state.graph.edges.iter() {
        // ── Classify edge ───────────────────────────────────────────────────────
        let from_seed = seed_store
            .try_get_seed(edge.from)
            .context("from seed not found in the seed store")?;
        let to_seed = seed_store
            .try_get_seed(edge.to)
            .context("to seed not found in the seed store")?;

        let alerts: Vec<_> = from_seed
            .resolve_members(alert_store)
            .context("from seed")?
            .into_iter()
            .chain(
                to_seed
                    .resolve_members(alert_store)
                    .context("to seed")?
                    .into_iter(),
            )
            .collect();

        let edge_class = truth.classify(&alerts);

        match edge_class {
            TruthClass::TruePositive => {
                global.n_true_positive += 1;
                if let Some(traj_id) = alerts.first().and_then(|a| truth.get_truth_traj_id(a)) {
                    recovered.insert(traj_id);
                }
            }
            TruthClass::FalsePositive => global.n_false_positive += 1,
            TruthClass::Unknown => global.n_unknown += 1,
        }
    }

    global.n_recovered_trajs = recovered.intersection(&recoverable).count();
    Ok(global)
}

/// Evaluate edge quality and write a summary to the tracing log.
///
/// Calls [`compute_edge_stats`] internally.  For plot output, use
/// [`plots::edge_plots`] in addition.
pub fn edge_evaluation(ctx: &PipelineContext, truth: &TruthSSO) -> Result<()> {
    let stats = compute_edge_stats(ctx, truth)?;

    tracing::info!("Edge evaluation");
    tracing::info!("{:-<72}", "");
    tracing::info!("TOTAL       {}", stats);

    Ok(())
}
