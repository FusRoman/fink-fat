//! Evaluate one [`CalibrationParams`] candidate against a set of
//! ground-truth trajectories: run the same materialize → Kalman filter bank
//! → summarize pipeline as
//! [`trajectory_processing::process_all_trajectories`](crate::trajectory_processing::process_all_trajectories),
//! parallelized over trajectories with rayon, but against an explicit
//! `&[TrajId]` subset (one round's sample) instead of the whole dataset, and
//! reduced to the `(recall, cost)` pair [`crate::kf_calibration::search`]'s
//! coordinate descent compares candidates on.

use ahash::AHashSet;
use fink_fat_engine::{
    engine_config::{kalman_context::KalmanContext, main_config::EngineConfig},
    topocentric_kf::single_kalman::KFState,
};
use photom::{
    TrajId,
    observation_dataset::{ObsDataset, observation::Observation},
};
use rayon::prelude::*;

use crate::{
    kalman_traj::{ObserverGeometryCache, recompute_search_region_metrics, study_kalman_asteroid},
    kf_calibration::params::CalibrationParams,
    trajectory_processing::{TrajSummary, materialize_contiguous_traj, summarize_trajectory},
};

/// Whether a trajectory counts as "correctly reconstructed" for recall
/// purposes: the bank tracked it to the end (`completion_fraction`) *and*
/// the true next observation genuinely fell inside the searched region at
/// each step (`pct_within_search_radius`) — the second condition guards
/// against a lax parameter set inflating `completion_fraction` by locking
/// onto whatever's nearby rather than the real object.
pub fn is_reconstructed(
    summary: &TrajSummary,
    completion_threshold: f64,
    within_radius_threshold: f64,
) -> bool {
    summary.completion_fraction >= completion_threshold
        && summary.pct_within_search_radius >= within_radius_threshold
}

/// Outcome of evaluating one [`CalibrationParams`] candidate over one round's
/// trajectory sample.
#[derive(Debug, Clone)]
pub struct RoundEval {
    /// Number of trajectory ids the candidate was evaluated against
    /// (`traj_ids.len()`, the recall denominator — includes trajectories
    /// that failed to even materialize/bootstrap a bank, counted as
    /// failures, not silently dropped).
    pub n_traj: usize,
    /// Fraction of `n_traj` for which [`is_reconstructed`] holds.
    pub recall: f64,
    /// Mean of [`TrajSummary::mean_search_radius_arcsec`] over trajectories
    /// that produced a summary. `f64::INFINITY` if none did (worst possible
    /// cost, so a candidate that reconstructs nothing never wins a
    /// comparison on cost alone).
    pub cost: f64,
    /// Ids that did *not* satisfy [`is_reconstructed`] (including ones that
    /// never produced a summary at all) — carried forward by
    /// [`crate::kf_calibration::search`] into the next round's sample.
    pub failing_ids: Vec<TrajId>,
}

/// Evaluate `params` against `traj_ids`: build the config/context this
/// candidate implies (see [`CalibrationParams::apply`]/
/// [`CalibrationParams::build_context`] — cheap, no ephemeris reload), run
/// [`study_kalman_asteroid`] on every id in parallel, and reduce to a
/// [`RoundEval`].
#[allow(clippy::too_many_arguments)]
pub fn evaluate(
    params: &CalibrationParams,
    base_config: &EngineConfig,
    base_context: &KalmanContext,
    obs_dataset: &ObsDataset,
    traj_ids: &[TrajId],
    geometry_cache: &ObserverGeometryCache,
    completion_threshold: f64,
    within_radius_threshold: f64,
) -> RoundEval {
    let config = params.apply(base_config);
    let context = params.build_context(base_context);

    let summaries: Vec<TrajSummary> = traj_ids
        .par_iter()
        .filter_map(|traj_id| {
            let traj = materialize_contiguous_traj(obs_dataset, traj_id).ok()?;
            if traj.len() < 3 {
                return None;
            }
            let study_outcome = study_kalman_asteroid(
                &traj,
                obs_dataset,
                &context,
                &config.kfbank_config,
                &config.seeding_grid_config,
                &config.advance_params,
                geometry_cache,
                None,
            );
            summarize_trajectory(traj_id.clone(), traj.len(), &study_outcome)
        })
        .collect();

    let n_traj = traj_ids.len();

    let success_ids: AHashSet<TrajId> = summaries
        .iter()
        .filter(|s| is_reconstructed(s, completion_threshold, within_radius_threshold))
        .map(|s| s.traj_id.clone())
        .collect();

    let recall = if n_traj == 0 {
        0.0
    } else {
        success_ids.len() as f64 / n_traj as f64
    };

    let cost = {
        let radii: Vec<f64> = summaries
            .iter()
            .map(|s| s.mean_search_radius_arcsec)
            .filter(|v| v.is_finite())
            .collect();
        if radii.is_empty() {
            f64::INFINITY
        } else {
            radii.iter().sum::<f64>() / radii.len() as f64
        }
    };

    let failing_ids: Vec<TrajId> = traj_ids
        .iter()
        .filter(|id| !success_ids.contains(*id))
        .cloned()
        .collect();

    RoundEval {
        n_traj,
        recall,
        cost,
        failing_ids,
    }
}

/// One trajectory's recorded reference run — the output of running the full
/// Kalman filter loop once for a fixed set of dynamics-affecting parameters
/// (`gate_chi2`/`weight_floor`/`q0`/`dt_ref`, see
/// [`crate::kf_calibration::params::ParamSpec::is_diagnostic_only`]),
/// kept around so [`evaluate_diagnostic_only`] can cheaply re-derive
/// [`RoundEval`] for any diagnostic-only-parameter candidate without
/// re-running that loop.
///
/// `completion_fraction` is fixed regardless of which diagnostic-only
/// candidate is later evaluated — only the trajectory's *dynamics* (not
/// touched by diagnostic-only params) determine whether/when the bank
/// collapses. `steps` carries what's needed to cheaply re-derive the
/// search-region-dependent metrics (`pct_within_search_radius`,
/// `mean_search_radius_arcsec`) per candidate — see
/// [`recompute_search_region_metrics`].
pub struct ReferenceRun<'a> {
    pub traj_id: TrajId,
    pub completion_fraction: f64,
    pub steps: Vec<(Vec<(f64, KFState<'a>)>, Observation)>,
}

/// Run the full Kalman filter loop once per trajectory in `round_ids`,
/// recording each one's [`ReferenceRun`] — the expensive part
/// (`study_kalman_asteroid`, dominated by per-hypothesis Kepler propagation
/// and covariance updates) [`evaluate_diagnostic_only`] then amortizes over
/// every diagnostic-only-parameter candidate tried against this same
/// `(gate_chi2, weight_floor, q0, dt_ref)` combination.
///
/// `context` must already reflect `dynamics_config`'s `q0`/`dt_ref` (i.e.
/// the caller has already called
/// [`CalibrationParams::apply`]/[`CalibrationParams::build_context`] on the
/// dynamics-affecting values it's currently holding fixed) and must outlive
/// every [`ReferenceRun`] returned — the caller keeps it alive in its own
/// stack frame for as long as the diagnostic-only sweep that uses these
/// reference runs is in progress.
pub fn build_reference_runs<'a>(
    dynamics_config: &EngineConfig,
    context: &'a KalmanContext,
    obs_dataset: &ObsDataset,
    round_ids: &[TrajId],
    geometry_cache: &ObserverGeometryCache,
) -> Vec<ReferenceRun<'a>> {
    round_ids
        .par_iter()
        .filter_map(|traj_id| {
            let traj = materialize_contiguous_traj(obs_dataset, traj_id).ok()?;
            if traj.len() < 3 {
                return None;
            }
            let mut steps = Vec::new();
            let study_outcome = study_kalman_asteroid(
                &traj,
                obs_dataset,
                context,
                &dynamics_config.kfbank_config,
                &dynamics_config.seeding_grid_config,
                &dynamics_config.advance_params,
                geometry_cache,
                Some(&mut steps),
            );
            let summary = summarize_trajectory(traj_id.clone(), traj.len(), &study_outcome)?;
            Some(ReferenceRun {
                traj_id: traj_id.clone(),
                completion_fraction: summary.completion_fraction,
                steps,
            })
        })
        .collect()
}

/// Cheap counterpart to [`evaluate`] for a diagnostic-only-parameter
/// candidate: reuses each [`ReferenceRun`]'s fixed `completion_fraction`
/// and only recomputes the search-region-dependent metrics (via
/// [`recompute_search_region_metrics`], no Kepler solve, no gating) —
/// otherwise identical semantics to [`evaluate`] (same `is_reconstructed`
/// definition, same recall/cost/failing-ids conventions, including
/// counting a `round_ids` entry missing from `reference_runs` — e.g. it
/// failed to materialize/bootstrap when [`build_reference_runs`] ran — as a
/// failure rather than silently dropping it).
pub fn evaluate_diagnostic_only(
    candidate: &CalibrationParams,
    base_config: &EngineConfig,
    reference_runs: &[ReferenceRun],
    round_ids: &[TrajId],
    completion_threshold: f64,
    within_radius_threshold: f64,
) -> RoundEval {
    let config = candidate.apply(base_config);
    let search_region_chi2 = config.kfbank_config.search_region_chi2;

    let per_run: Vec<(TrajId, bool, f64)> = reference_runs
        .par_iter()
        .map(|run| {
            let (pct_within_search_radius, mean_search_radius_arcsec) =
                recompute_search_region_metrics(
                    &run.steps,
                    &config.advance_params,
                    search_region_chi2,
                );
            let reconstructed = run.completion_fraction >= completion_threshold
                && pct_within_search_radius >= within_radius_threshold;
            (
                run.traj_id.clone(),
                reconstructed,
                mean_search_radius_arcsec,
            )
        })
        .collect();

    let success_ids: AHashSet<TrajId> = per_run
        .iter()
        .filter(|(_, reconstructed, _)| *reconstructed)
        .map(|(id, _, _)| id.clone())
        .collect();

    let n_traj = round_ids.len();
    let recall = if n_traj == 0 {
        0.0
    } else {
        success_ids.len() as f64 / n_traj as f64
    };

    let cost = {
        let radii: Vec<f64> = per_run
            .iter()
            .map(|(_, _, mean_radius)| *mean_radius)
            .filter(|v| v.is_finite())
            .collect();
        if radii.is_empty() {
            f64::INFINITY
        } else {
            radii.iter().sum::<f64>() / radii.len() as f64
        }
    };

    let failing_ids: Vec<TrajId> = round_ids
        .iter()
        .filter(|id| !success_ids.contains(*id))
        .cloned()
        .collect();

    RoundEval {
        n_traj,
        recall,
        cost,
        failing_ids,
    }
}

/// Lexicographic comparison key for two [`RoundEval`]s: candidates that
/// reach `target_recall` always beat ones that don't; among those that do
/// (or don't), higher recall wins first, then lower cost. Returned as a
/// `(bool, ordered recall, ordered -cost)` tuple so callers can just
/// `max_by_key`/`>` two keys with the standard tuple `Ord` — no external
/// multi-objective-optimization dependency needed.
pub fn comparison_key(
    eval: &RoundEval,
    target_recall: f64,
) -> (bool, ordered_recall::Key, ordered_recall::Key) {
    (
        eval.recall >= target_recall,
        ordered_recall::Key(eval.recall),
        ordered_recall::Key(-eval.cost),
    )
}

/// A tiny `f64` newtype with a total order (`NaN` sorts lowest), just
/// expressive enough to make [`comparison_key`]'s tuple `Ord`-comparable
/// without pulling in a crate like `ordered-float` for two call sites.
pub mod ordered_recall {
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Key(pub f64);

    impl Eq for Key {}

    impl PartialOrd for Key {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for Key {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0.total_cmp(&other.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(recall: f64, cost: f64) -> RoundEval {
        RoundEval {
            n_traj: 10,
            recall,
            cost,
            failing_ids: Vec::new(),
        }
    }

    #[test]
    fn candidate_reaching_target_recall_always_wins() {
        let below_target = comparison_key(&eval(0.80, 10.0), 0.95);
        let at_target_high_cost = comparison_key(&eval(0.96, 1000.0), 0.95);
        assert!(at_target_high_cost > below_target);
    }

    #[test]
    fn ties_on_target_broken_by_lower_cost() {
        let cheap = comparison_key(&eval(1.0, 10.0), 0.95);
        let expensive = comparison_key(&eval(1.0, 100.0), 0.95);
        assert!(cheap > expensive);
    }

    #[test]
    fn below_target_broken_by_higher_recall() {
        let higher = comparison_key(&eval(0.90, 500.0), 0.95);
        let lower = comparison_key(&eval(0.80, 10.0), 0.95);
        assert!(higher > lower);
    }
}
