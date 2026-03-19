use std::fmt;

use ahash::{AHashMap, AHashSet};
use anyhow::Result;
use camino::Utf8Path;
use fink_fat_engine::pipeline::PipelineContext;

use crate::{
    edges::compute_edge_stats,
    truth_sso::{TrajId, TruthClass, TruthSSO},
};

// ─────────────────────────────────────────────────────────────────────────────
// Edge quality statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Minimum fraction of a trajectory's alerts that a single TP track must cover
/// for that trajectory to be counted as partially recovered.
pub const PARTIAL_RECALL_THRESHOLD: f64 = 0.5;

/// Quality statistics for a collection of edges.
#[derive(Debug, Default, Clone, Copy)]
pub struct SolverStats {
    /// Total number of trajectories.
    pub n_traj: usize,
    /// Trajectories where every member alert belongs to the same ground-truth trajectory.
    pub n_true_positive: usize,
    /// Trajectories where member alerts belong to more than one ground-truth trajectory.
    pub n_false_positive: usize,
    /// Trajectories where at least one member alert is absent from the truth map.
    pub n_unknown: usize,
    /// Number of ground-truth trajectories with ≥ 2 alerts on this night
    /// (i.e., trajectories that *could* have produced at least one edge).
    pub n_recoverable_trajs: usize,
    /// Number of recoverable trajectories for which at least one TP edge was produced.
    pub n_recovered_trajs: usize,
    /// Threshold used for partial recall (fraction in `[0, 1]`).
    ///
    /// A trajectory is partially recovered when at least one TP track covers
    /// ≥ `partial_recall_threshold` of its total alerts.
    pub partial_recall_threshold: f64,
    /// Number of recoverable trajectories partially recovered above the threshold.
    pub n_recovered_trajs_partial: usize,
}

impl SolverStats {
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

    /// Partial recall: fraction of recoverable trajectories for which at least
    /// one TP track covers ≥ [`Self::partial_recall_threshold`] of their total alerts.
    ///
    /// Returns `f64::NAN` when there are no recoverable trajectories.
    pub fn partial_recall(&self) -> f64 {
        if self.n_recoverable_trajs == 0 {
            f64::NAN
        } else {
            self.n_recovered_trajs_partial as f64 / self.n_recoverable_trajs as f64
        }
    }
}

impl fmt::Display for SolverStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "track={n:>8}  TP={tp:>8}  FP={fp:>8}  unknown={unk:>8}  purity={pur:.4}  recall={rec:.4} ({rec_n}/{rec_d})  partial_recall={prec:.4} ({prec_n}/{prec_d}, ≥{thr:.0}%)",
            n = self.n_traj,
            tp = self.n_true_positive,
            fp = self.n_false_positive,
            unk = self.n_unknown,
            pur = self.purity(),
            rec = self.recall(),
            rec_n = self.n_recovered_trajs,
            rec_d = self.n_recoverable_trajs,
            prec = self.partial_recall(),
            prec_n = self.n_recovered_trajs_partial,
            prec_d = self.n_recoverable_trajs,
            thr = self.partial_recall_threshold * 100.0,
        )
    }
}

/// Compute per-night solver statistics for all nights in the seed store.
///
/// Returns `(global, per_night)` where `per_night` is sorted by night ID and
/// each element is `(night_label, SolverStats)`.
///
/// This is the shared core used by both [`solver_evaluation`] (for logging)
/// and the plot functions (for plotting).
pub fn compute_solver_stats(ctx: &PipelineContext, truth: &TruthSSO) -> Result<SolverStats> {
    let seed_store = &ctx.runtime_state.seed_store;
    let alert_store = &ctx.runtime_state.alert_store;

    // ── Recoverable trajectories ──────────────────────────────────────────
    let max_gap = ctx.engine_config.max_gap_nights();
    let min_nodes = ctx.engine_config.solver_config.bounded_beam.min_nodes;
    let recoverable: AHashSet<TrajId> = truth.recoverable_traj(2, max_gap, min_nodes).collect();
    let n_recoverable_trajs = recoverable.len();

    let mut global = SolverStats {
        n_traj: ctx.runtime_state.track_hypotheses.len(),
        n_recoverable_trajs,
        partial_recall_threshold: PARTIAL_RECALL_THRESHOLD,
        ..SolverStats::default()
    };

    let mut recovered: AHashSet<TrajId> = AHashSet::new();
    // Tracks, per trajectory, the maximum number of alerts covered by any single TP track.
    let mut max_coverage: AHashMap<TrajId, usize> = AHashMap::new();

    for (_, track) in ctx.runtime_state.track_hypotheses.iter() {
        let alerts = track.get_alerts(alert_store, seed_store)?;
        let alert_refs: Vec<&_> = alerts.iter().collect();
        let track_class = truth.classify(&alert_refs);

        match track_class {
            TruthClass::TruePositive => {
                global.n_true_positive += 1;
                if let Some(traj_id) = alerts.first().and_then(|a| truth.get_truth_traj_id(a)) {
                    if recoverable.contains(&traj_id) {
                        recovered.insert(traj_id);
                    }
                    // Track max coverage for partial recall.
                    let entry = max_coverage.entry(traj_id).or_insert(0);
                    *entry = (*entry).max(alerts.len());
                }
            }
            TruthClass::FalsePositive => global.n_false_positive += 1,
            TruthClass::Unknown => global.n_unknown += 1,
        }
    }

    global.n_recovered_trajs = recovered.len();
    global.n_recovered_trajs_partial = recoverable
        .iter()
        .filter(|&&traj_id| {
            let n = max_coverage.get(&traj_id).copied().unwrap_or(0);
            truth.coverage_fraction(traj_id, n) >= PARTIAL_RECALL_THRESHOLD
        })
        .count();
    Ok(global)
}

/// Evaluate edge quality and write a summary to the tracing log.
///
/// Calls [`compute_edge_stats`] internally.  When `plot_dir` is provided,
/// feature distribution charts (TP vs FP) are also written via
/// [`crate::edges::plots::edge_plots`].
pub fn solver_evaluation(
    ctx: &PipelineContext,
    truth: &TruthSSO,
    _: Option<&Utf8Path>,
) -> Result<()> {
    let edges_stats = compute_edge_stats(ctx, truth)?;
    let solver_stats = compute_solver_stats(ctx, truth)?;

    tracing::info!("Edge evaluation");
    tracing::info!("{:-<72}", "");
    tracing::info!("TOTAL       {}", edges_stats);

    tracing::info!("");
    tracing::info!("Solver evaluation");
    tracing::info!("{:-<72}", "");
    tracing::info!("TOTAL       {}", solver_stats);

    Ok(())
}
