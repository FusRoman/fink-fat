pub mod export;
pub mod plots;

use std::fmt;

use ahash::AHashSet;
use anyhow::{Context, Result};
use camino::Utf8Path;
use fink_fat_engine::night_id::NightId;
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
    ///
    /// **Lower-bound recall**: a trajectory is counted only when the engine
    /// produced a TP edge whose exact `(from_night, to_night)` pair appears in
    /// the pre-built `recoverable_edges` set (built with `windows(2)`).
    ///
    /// This may under-count trajectories recovered by *skip edges* — TP edges
    /// that hop over an intermediate seeded night.
    /// See also [`Self::n_recovered_trajs_upper`].
    pub n_recovered_trajs: usize,

    /// Number of recoverable trajectories for which at least one TP edge was
    /// produced — computed with the direct, per-night seeding check.
    ///
    /// **Upper-bound recall**: a trajectory is counted when the engine produced
    /// a TP edge `(from_night → to_night)` such that:
    /// - both nights have ≥ 2 alerts for that trajectory (both are seeded nights),
    /// - the gap `to_night − from_night ≤ max_gap`.
    ///
    /// This is the *semantic* definition of a recovered edge and captures skip
    /// edges that [`Self::n_recovered_trajs`] misses because `recoverable_edges`
    /// is built with `windows(2)` (consecutive-pairs only).
    ///
    /// # Relationship between the two metrics
    ///
    /// By construction:
    /// - Every edge counted in [`Self::n_recovered_trajs`] also satisfies the
    ///   upper-bound condition → `n_recovered_trajs ≤ n_recovered_trajs_upper`.
    /// - The two values coincide when the engine never produces skip edges, i.e.
    ///   every TP edge connects *consecutive* seeded nights.
    /// - The gap between them quantifies how many trajectories were recovered
    ///   via skip edges that the lower-bound metric silently missed.
    pub n_recovered_trajs_upper: usize,
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

    /// Lower-bound recall: fraction of recoverable trajectories counted as
    /// recovered using the exact-pair check against `recoverable_edges`.
    ///
    /// A trajectory is counted only when the engine produced a TP edge whose
    /// exact `(from_night, to_night)` pair appears in the pre-built recoverable
    /// set. Because that set is built with `windows(2)` (consecutive seeded-night
    /// pairs only), *skip edges* over an intermediate seeded night are silently
    /// excluded → the result is a lower bound on the true recall.
    ///
    /// Returns `f64::NAN` when there are no recoverable trajectories.
    ///
    /// See also [`Self::recall_upper`].
    pub fn recall(&self) -> f64 {
        if self.n_recoverable_trajs == 0 {
            f64::NAN
        } else {
            self.n_recovered_trajs as f64 / self.n_recoverable_trajs as f64
        }
    }

    /// Upper-bound recall: fraction of recoverable trajectories counted as
    /// recovered using the direct per-night seeding check.
    ///
    /// A trajectory is counted when the engine produced a TP edge
    /// `(from_night → to_night)` such that both nights are seeded nights for
    /// that trajectory (≥ 2 alerts each) and the gap is ≤ `max_gap`. This
    /// is the semantic definition of a recovered edge and captures skip edges
    /// that [`Self::recall`] misses.
    ///
    /// Use the gap between `recall()` and `recall_upper()` to measure how many
    /// trajectories are recovered exclusively via skip edges.
    ///
    /// Returns `f64::NAN` when there are no recoverable trajectories.
    pub fn recall_upper(&self) -> f64 {
        if self.n_recoverable_trajs == 0 {
            f64::NAN
        } else {
            self.n_recovered_trajs_upper as f64 / self.n_recoverable_trajs as f64
        }
    }
}

impl fmt::Display for EdgeStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "edges={n:>8}  TP={tp:>8}  FP={fp:>8}  unknown={unk:>8}  purity={pur:.4}  recall_lb={rlb:.4}  recall_ub={rub:.4} ({rec_n}/{rec_d})",
            n = self.n_edges,
            tp = self.n_true_positive,
            fp = self.n_false_positive,
            unk = self.n_unknown,
            pur = self.purity(),
            rlb = self.recall(),
            rub = self.recall_upper(),
            rec_n = self.n_recovered_trajs_upper,
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
    let recoverable: AHashSet<(TrajId, NightId, NightId)> =
        truth.recoverable_edges(2, max_gap).collect();
    let n_recoverable_trajs: usize = recoverable
        .iter()
        .map(|(t, _, _)| *t)
        .collect::<AHashSet<_>>()
        .len();

    let mut global = EdgeStats {
        n_edges: ctx.runtime_state.graph.edges.len(),
        n_recoverable_trajs,
        ..EdgeStats::default()
    };

    // Lower-bound set: trajectories recovered via an exact pair in `recoverable`.
    let mut recovered: AHashSet<TrajId> = AHashSet::new();
    // Upper-bound set: trajectories recovered by any TP edge whose endpoint
    // nights are both seeded nights for that trajectory within max_gap.
    let mut recovered_upper: AHashSet<TrajId> = AHashSet::new();

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
                    let from_night = from_seed.night_id();
                    let to_night = to_seed.night_id();

                    // Lower-bound: requires the exact pair in the pre-built
                    // recoverable set (built with windows(2) — may miss skip edges).
                    if recoverable.contains(&(traj_id, from_night, to_night)) {
                        recovered.insert(traj_id);
                    }

                    // Upper-bound: directly checks the three conditions that
                    // define a recoverable edge, bypassing the windows(2) limit.
                    // For a TP edge both seeded-night conditions hold by definition
                    // (each seed already contains ≥ 2 alerts from the trajectory);
                    // the gap check guards against edges that span > max_gap.
                    let from_seeded = truth.traj_count_for_night(traj_id, from_night) >= 2;
                    let to_seeded = truth.traj_count_for_night(traj_id, to_night) >= 2;
                    let gap_ok = to_night.0.saturating_sub(from_night.0) <= max_gap as u32;
                    if from_seeded && to_seeded && gap_ok {
                        recovered_upper.insert(traj_id);
                    }
                }
            }
            TruthClass::FalsePositive => global.n_false_positive += 1,
            TruthClass::Unknown => global.n_unknown += 1,
        }
    }

    global.n_recovered_trajs = recovered.len();
    global.n_recovered_trajs_upper = recovered_upper.len();
    Ok(global)
}

/// Evaluate edge quality and write a summary to the tracing log.
///
/// Calls [`compute_edge_stats`] internally.  When `plot_dir` is provided,
/// feature distribution charts (TP vs FP) are also written via
/// [`plots::edge_plots`].  When `export_features_path` is provided, a Parquet
/// file with all edge features and truth labels is written via
/// [`export::export_edge_features_parquet`].
pub fn edge_evaluation(
    ctx: &PipelineContext,
    truth: &TruthSSO,
    plot_dir: Option<&Utf8Path>,
    export_features_path: Option<&Utf8Path>,
) -> Result<()> {
    let stats = compute_edge_stats(ctx, truth)?;

    tracing::info!("Edge evaluation");
    tracing::info!("{:-<72}", "");
    tracing::info!("TOTAL       {}", stats);

    if let Some(dir) = plot_dir {
        plots::edge_plots(ctx, truth, dir)?;
    }

    if let Some(path) = export_features_path {
        export::export_edge_features_parquet(ctx, truth, path)?;
    }

    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod edge_stats_tests {
    use super::*;

    /// Convenience constructor for test fixtures.
    ///
    /// `recovered_upper` defaults to `recovered` (lower == upper when there
    /// are no skip-edge ambiguities in the test scenario).
    fn make_stats(
        n_edges: usize,
        tp: usize,
        fp: usize,
        unk: usize,
        recoverable: usize,
        recovered: usize,
    ) -> EdgeStats {
        EdgeStats {
            n_edges,
            n_true_positive: tp,
            n_false_positive: fp,
            n_unknown: unk,
            n_recoverable_trajs: recoverable,
            n_recovered_trajs: recovered,
            n_recovered_trajs_upper: recovered,
        }
    }

    /// Convenience constructor that sets both lower and upper recall counters
    /// independently, to test the skip-edge scenario.
    fn make_stats_with_upper(
        n_edges: usize,
        tp: usize,
        fp: usize,
        unk: usize,
        recoverable: usize,
        recovered: usize,
        recovered_upper: usize,
    ) -> EdgeStats {
        EdgeStats {
            n_edges,
            n_true_positive: tp,
            n_false_positive: fp,
            n_unknown: unk,
            n_recoverable_trajs: recoverable,
            n_recovered_trajs: recovered,
            n_recovered_trajs_upper: recovered_upper,
        }
    }

    // ── recall ────────────────────────────────────────────────────────────────

    #[test]
    fn recall_nan_when_no_recoverable_trajs() {
        assert!(
            make_stats(0, 0, 0, 0, 0, 0).recall().is_nan(),
            "recall with no recoverable trajectories should be NaN"
        );
    }

    #[test]
    fn recall_zero_when_nothing_recovered() {
        assert_eq!(
            make_stats(5, 0, 5, 0, 3, 0).recall(),
            0.0,
            "recall should be 0 when no trajectory was recovered"
        );
    }

    #[test]
    fn recall_one_when_all_recovered() {
        let r = make_stats(3, 3, 0, 0, 3, 3).recall();
        assert!((r - 1.0).abs() < 1e-10, "expected recall = 1.0, got {r}");
    }

    #[test]
    fn recall_half() {
        let r = make_stats(10, 5, 5, 0, 4, 2).recall();
        assert!((r - 0.5).abs() < 1e-10, "expected recall = 0.5, got {r}");
    }

    #[test]
    fn recall_three_out_of_four() {
        let r = make_stats(8, 6, 2, 0, 4, 3).recall();
        assert!((r - 0.75).abs() < 1e-10, "expected recall = 0.75, got {r}");
    }

    /// The unit of recall is trajectories, not edges.
    ///
    /// A trajectory that generated 5 TP edges must be counted ONCE in
    /// `n_recovered_trajs`.  Counting edges would give recall > 1.0 here.
    #[test]
    fn recall_counts_trajectories_not_edges() {
        // 1 recoverable trajectory, 1 recovered trajectory, 5 TP edges.
        let s = make_stats(5, 5, 0, 0, 1, 1);
        let r = s.recall();
        assert!((r - 1.0).abs() < 1e-10, "expected recall = 1.0, got {r}");
        assert!(r <= 1.0 + f64::EPSILON, "recall must never exceed 1.0");
    }

    /// Unknown edges have no bearing on the recall denominator.
    ///
    /// Recall is defined over recoverable *trajectories*, not over edges.
    /// Adding or removing unknown edges must leave recall unchanged.
    #[test]
    fn recall_independent_of_unknown_edges() {
        let with_unknown = make_stats(8, 2, 2, 4, 2, 1);
        let without_unknown = make_stats(4, 2, 2, 0, 2, 1);
        let diff = (with_unknown.recall() - without_unknown.recall()).abs();
        assert!(
            diff < 1e-10,
            "recall changed when unknowns were added (diff = {diff})"
        );
    }

    // ── purity ────────────────────────────────────────────────────────────────

    #[test]
    fn purity_nan_when_no_classifiable_edges() {
        assert!(make_stats(0, 0, 0, 0, 0, 0).purity().is_nan());
        // Unknown-only: still no classifiable edge.
        assert!(make_stats(4, 0, 0, 4, 0, 0).purity().is_nan());
    }

    #[test]
    fn purity_one_when_all_tp() {
        let p = make_stats(5, 5, 0, 0, 2, 2).purity();
        assert!((p - 1.0).abs() < 1e-10, "expected purity = 1.0, got {p}");
    }

    #[test]
    fn purity_zero_when_all_fp() {
        assert_eq!(make_stats(5, 0, 5, 0, 2, 0).purity(), 0.0);
    }

    #[test]
    fn purity_excludes_unknowns_from_denominator() {
        // 3 TP, 3 FP, 4 unknown → classifiable = 6 → purity = 0.5.
        let p = make_stats(10, 3, 3, 4, 0, 0).purity();
        assert!((p - 0.5).abs() < 1e-10, "expected purity = 0.5, got {p}");
    }

    // ── recall_upper ──────────────────────────────────────────────────────────
    //
    // The upper-bound recall uses the same denominator as recall() but a wider
    // numerator: any TP edge over two seeded nights within max_gap is credited,
    // even when the exact pair is absent from the windows(2)-based recoverable set.

    #[test]
    fn recall_upper_nan_when_no_recoverable_trajs() {
        assert!(
            make_stats(0, 0, 0, 0, 0, 0).recall_upper().is_nan(),
            "recall_upper with no recoverable trajectories should be NaN"
        );
    }

    #[test]
    fn recall_upper_equals_recall_when_no_skip_edges() {
        // When lower == upper (no skip-edge ambiguities), both methods agree.
        let s = make_stats(6, 4, 2, 0, 3, 2);
        let diff = (s.recall_upper() - s.recall()).abs();
        assert!(
            diff < 1e-10,
            "recall_upper should equal recall when lb == ub; diff = {diff}"
        );
    }

    /// Upper-bound is always ≥ lower-bound.
    ///
    /// A trajectory counted by the lower-bound check always satisfies the
    /// upper-bound conditions, so the upper-bound set is a superset.
    #[test]
    fn recall_upper_is_at_least_recall() {
        let s = make_stats_with_upper(10, 6, 4, 0, 4, 2, 3);
        assert!(
            s.recall_upper() >= s.recall() - 1e-10,
            "recall_upper ({}) must be >= recall ({})",
            s.recall_upper(),
            s.recall(),
        );
    }

    /// When skip edges recover extra trajectories, recall_upper > recall.
    ///
    /// Scenario: 4 recoverable trajectories, 2 recovered by exact-pair edges
    /// (counted by both metrics), 1 additional trajectory recovered only via a
    /// skip edge (counted by recall_upper but not recall).
    #[test]
    fn recall_upper_exceeds_recall_for_skip_edge_scenario() {
        // lower: 2/4 = 0.50,  upper: 3/4 = 0.75
        let s = make_stats_with_upper(10, 5, 5, 0, 4, 2, 3);
        let lb = s.recall();
        let ub = s.recall_upper();
        assert!(
            (lb - 0.5).abs() < 1e-10,
            "expected lower recall = 0.50, got {lb}"
        );
        assert!(
            (ub - 0.75).abs() < 1e-10,
            "expected upper recall = 0.75, got {ub}"
        );
        assert!(
            ub > lb,
            "upper recall must strictly exceed lower recall here"
        );
    }

    #[test]
    fn recall_upper_one_when_all_recovered_via_skip_edges() {
        // All 3 trajectories recovered, but only via skip edges (lb = 0).
        let s = make_stats_with_upper(3, 3, 0, 0, 3, 0, 3);
        let ub = s.recall_upper();
        assert!(
            (ub - 1.0).abs() < 1e-10,
            "expected upper recall = 1.0, got {ub}"
        );
        assert_eq!(
            s.recall(),
            0.0,
            "lower recall should be 0.0 when only skip edges"
        );
    }
}
