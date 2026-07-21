//! The progressive-sampling + coordinate-descent calibration loop.
//!
//! Round `k` samples a trajectory subset (previous round's failures, topped
//! up with a fresh random draw), runs [`coordinate_descent`] — sweeping
//! [`crate::kf_calibration::params::default_param_specs`] one field at a
//! time, keeping whichever candidate value scores best — over that subset,
//! then grows the sample for round `k+1`. See [`calibrate`]'s doc for the
//! stopping conditions.

use ahash::AHashSet;
use fink_fat_engine::engine_config::{kalman_context::KalmanContext, main_config::EngineConfig};
use indicatif::{ProgressBar, ProgressStyle};
use photom::{TrajId, observation_dataset::ObsDataset};
use rand::{SeedableRng, rngs::StdRng, seq::index};

use crate::{
    kalman_traj::ObserverGeometryCache,
    kf_calibration::{
        objective::{
            RoundEval, build_reference_runs, comparison_key, evaluate, evaluate_diagnostic_only,
        },
        params::{CalibrationParams, ParamSpec, default_param_specs},
        report::{CalibrationReport, RoundResult},
    },
};

/// Tuning knobs for [`calibrate`] itself (as opposed to
/// [`CalibrationParams`], the engine knobs being calibrated).
#[derive(Debug, Clone)]
pub struct CalibrationOptions {
    /// Seed for the round-sampling RNG — same seed, same alerts/config in →
    /// same [`CalibrationReport`] out.
    pub seed: u64,
    /// Trajectory sample size for round 0.
    pub initial_sample_size: usize,
    /// Multiplicative growth applied to the sample size after each round
    /// that doesn't yet cover the whole dataset.
    pub growth_factor: f64,
    /// Recall (fraction of a round's sample satisfying
    /// [`crate::kf_calibration::objective::is_reconstructed`]) at which
    /// [`coordinate_descent`] switches from "maximize recall" to "minimize
    /// cost", and at which [`calibrate`] is allowed to stop early.
    pub target_recall: f64,
    /// Hard cap on the number of rounds, regardless of convergence.
    pub max_rounds: usize,
    /// Cap on how many failing ids are carried into the next round's
    /// sample — without this the failing pool could grow to dominate every
    /// later round's sample, crowding out fresh trajectories.
    pub max_failing_pool: usize,
    /// [`TrajSummary::completion_fraction`](crate::trajectory_processing::TrajSummary::completion_fraction)
    /// threshold for "reconstructed".
    pub completion_threshold: f64,
    /// [`TrajSummary::pct_within_search_radius`](crate::trajectory_processing::TrajSummary::pct_within_search_radius)
    /// threshold (0-100) for "reconstructed".
    pub within_radius_threshold: f64,
    /// Restrict calibration to these [`ParamSpec::name`]s (`None` = every
    /// default param — see
    /// [`default_param_specs`]).
    pub param_names: Option<Vec<String>>,
}

impl Default for CalibrationOptions {
    fn default() -> Self {
        Self {
            seed: 42,
            initial_sample_size: 50,
            growth_factor: 4.0,
            target_recall: 0.98,
            max_rounds: 20,
            max_failing_pool: 200,
            completion_threshold: 0.98,
            within_radius_threshold: 95.0,
            param_names: None,
        }
    }
}

/// Run the full progressive-sampling calibration loop.
///
/// Starting from `base_config`/`base_context`'s own values (see
/// [`CalibrationParams::from_engine_config`]), each round:
/// 1. samples a trajectory subset (`opts.max_failing_pool`-capped failures
///    from the previous round, topped up with a fresh random draw up to the
///    round's target size — see [`sample_round_ids`]);
/// 2. runs [`coordinate_descent`] over that subset;
/// 3. records a [`RoundResult`].
///
/// Stops when a round both reaches `opts.target_recall` *and* its sample
/// already covers the whole dataset, or after `opts.max_rounds` regardless.
/// Empty `all_traj_ids` short-circuits to a zero-round report.
pub fn calibrate(
    obs_dataset: &ObsDataset,
    base_config: &EngineConfig,
    base_context: &KalmanContext,
    all_traj_ids: &[TrajId],
    geometry_cache: &ObserverGeometryCache,
    opts: &CalibrationOptions,
) -> CalibrationReport {
    let mut params = CalibrationParams::from_engine_config(base_config, base_context);

    if all_traj_ids.is_empty() {
        return CalibrationReport {
            rounds: Vec::new(),
            final_params: params,
        };
    }

    let specs = select_specs(opts);
    let mut rng = StdRng::seed_from_u64(opts.seed);
    let mut failing_pool: Vec<TrajId> = Vec::new();
    let mut rounds: Vec<RoundResult> = Vec::new();
    let mut target_size = opts.initial_sample_size.max(1).min(all_traj_ids.len());

    let max_rounds = opts.max_rounds.max(1);
    let progress = ProgressBar::new(max_rounds as u64);
    progress.set_style(
        ProgressStyle::with_template("{bar:40.cyan/blue} round {pos}/{len}  {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    progress.set_message(format!(
        "sample={target_size} starting from {}",
        format_params_line(&params)
    ));

    for round_index in 0..max_rounds {
        let round_ids = sample_round_ids(all_traj_ids, &failing_pool, target_size, &mut rng);
        let covers_full_dataset = round_ids.len() >= all_traj_ids.len();

        let (new_params, eval, param_deltas) = coordinate_descent(
            params,
            &specs,
            obs_dataset,
            base_config,
            base_context,
            &round_ids,
            geometry_cache,
            opts,
            &progress,
            round_index,
            max_rounds,
        );
        params = new_params;

        failing_pool = cap_pool(eval.failing_ids.clone(), opts.max_failing_pool, &mut rng);

        let reached_target = eval.recall >= opts.target_recall;

        progress.println(format!(
            "[round {round_index}] sample={} recall={:.1}% cost={:.1}\" failing={}  best: {}",
            round_ids.len(),
            eval.recall * 100.0,
            eval.cost,
            eval.failing_ids.len(),
            format_params_line(&params),
        ));

        rounds.push(RoundResult {
            round_index,
            sample_size: round_ids.len(),
            n_failing: eval.failing_ids.len(),
            recall: eval.recall,
            cost_mean_search_radius_arcsec: eval.cost,
            params,
            param_deltas,
        });

        progress.inc(1);

        if reached_target && covers_full_dataset {
            progress.finish_with_message(format!(
                "target recall reached on the full dataset — {}",
                format_params_line(&params)
            ));
            break;
        }

        target_size = if covers_full_dataset {
            all_traj_ids.len()
        } else {
            let grown = (target_size as f64 * opts.growth_factor).ceil() as usize;
            grown.max(target_size + 1).min(all_traj_ids.len())
        };
    }

    if !progress.is_finished() {
        progress.finish_with_message(format!(
            "max rounds reached — {}",
            format_params_line(&params)
        ));
    }

    CalibrationReport {
        rounds,
        final_params: params,
    }
}

/// Compact one-line dump of every calibrated field, used for round-by-round
/// progress output (see [`calibrate`]) — deliberately terser than
/// [`crate::kf_calibration::report::to_yaml_snippet`], which is for the
/// final, copy-pasteable recommendation.
fn format_params_line(p: &CalibrationParams) -> String {
    format!(
        "max_arcsec={:.1}\" obs_σ={:.3}\" wthr={:.4} gate_χ²={:.1} search_χ²={:.1} wfloor={:.1e} q0={:.2e} dt_ref={:.2}",
        p.max_arcsec,
        p.obs_noise_sigma_arcsec,
        p.weight_threshold,
        p.gate_chi2,
        p.search_region_chi2,
        p.weight_floor,
        p.q0,
        p.dt_ref,
    )
}

fn select_specs(opts: &CalibrationOptions) -> Vec<ParamSpec> {
    let all = default_param_specs();
    match &opts.param_names {
        None => all,
        Some(names) => all
            .into_iter()
            .filter(|spec| names.iter().any(|n| n == spec.name))
            .collect(),
    }
}

/// One trajectory subset for a round: `failing_pool` in full, topped up
/// with a fresh random draw from `all_ids` (excluding `failing_pool`) up to
/// `target_size`. If `failing_pool` alone already exceeds `target_size`, a
/// random subset of it is kept instead (see [`cap_pool`]) — this only
/// happens if the caller passed an already-oversized `failing_pool` (`calibrate`
/// caps it every round via `opts.max_failing_pool`, so in practice this is a
/// safety net, not the normal path).
pub fn sample_round_ids(
    all_ids: &[TrajId],
    failing_pool: &[TrajId],
    target_size: usize,
    rng: &mut StdRng,
) -> Vec<TrajId> {
    if failing_pool.len() >= target_size {
        return cap_pool(failing_pool.to_vec(), target_size, rng);
    }

    let mut seen: AHashSet<TrajId> = failing_pool.iter().cloned().collect();
    let mut result: Vec<TrajId> = failing_pool.to_vec();

    let remaining_target = target_size - result.len();
    let candidates: Vec<&TrajId> = all_ids.iter().filter(|id| !seen.contains(*id)).collect();
    let n_pick = remaining_target.min(candidates.len());

    if n_pick > 0 {
        for i in index::sample(rng, candidates.len(), n_pick).into_iter() {
            let id = candidates[i].clone();
            seen.insert(id.clone());
            result.push(id);
        }
    }

    result
}

/// Keep at most `max_size` entries of `pool`, chosen uniformly at random
/// (deterministic given `rng`'s state).
fn cap_pool(pool: Vec<TrajId>, max_size: usize, rng: &mut StdRng) -> Vec<TrajId> {
    if pool.len() <= max_size {
        return pool;
    }
    index::sample(rng, pool.len(), max_size)
        .into_iter()
        .map(|i| pool[i].clone())
        .collect()
}

/// Sweep every [`ParamSpec`] (up to 3 full sweeps, stopping early once a
/// sweep makes no change), keeping whichever candidate value scores best by
/// [`comparison_key`].
///
/// Each sweep runs in two phases:
/// 1. Dynamics-affecting specs (`is_diagnostic_only == false`) first, each
///    candidate evaluated via the full [`evaluate`] — these can change which
///    observations the bank actually associates, so there's no shortcut.
/// 2. Diagnostic-only specs (`is_diagnostic_only == true`) next, against a
///    single [`ReferenceRun`](crate::kf_calibration::objective::ReferenceRun)
///    set built *once* per sweep (via
///    [`build_reference_runs`](crate::kf_calibration::objective::build_reference_runs))
///    for phase 1's now-settled dynamics values, then reused for every
///    diagnostic-only candidate via the cheap
///    [`evaluate_diagnostic_only`](crate::kf_calibration::objective::evaluate_diagnostic_only)
///    — safe because nothing in phase 2 can change the dynamics-affecting
///    values the reference runs were built from (see
///    [`crate::kf_calibration::params::ParamSpec::is_diagnostic_only`]'s doc).
///
/// [`default_param_specs`](crate::kf_calibration::params::default_param_specs)
/// is already ordered dynamics-affecting-first so `specs`' own order lines
/// up with these two phases; [`coordinate_descent`] still filters by
/// `is_diagnostic_only` rather than assuming a prefix split, so a caller
/// passing a reordered/filtered `specs` (e.g. `--params`) stays correct.
///
/// Returns the final params, their [`RoundEval`], and the last sweep's
/// per-parameter score deltas (see [`scalar_score`] for what "score" means)
/// — the latter feeds [`RoundResult::param_deltas`]
/// (`crate::kf_calibration::report`).
#[allow(clippy::too_many_arguments)]
pub fn coordinate_descent(
    initial_params: CalibrationParams,
    specs: &[ParamSpec],
    obs_dataset: &ObsDataset,
    base_config: &EngineConfig,
    base_context: &KalmanContext,
    round_ids: &[TrajId],
    geometry_cache: &ObserverGeometryCache,
    opts: &CalibrationOptions,
    progress: &ProgressBar,
    round_index: usize,
    max_rounds: usize,
) -> (CalibrationParams, RoundEval, Vec<(String, f64)>) {
    let mut params = initial_params;
    let mut current_eval = evaluate(
        &params,
        base_config,
        base_context,
        obs_dataset,
        round_ids,
        geometry_cache,
        opts.completion_threshold,
        opts.within_radius_threshold,
    );
    let mut last_sweep_deltas: Vec<(String, f64)> = Vec::new();

    const MAX_SWEEPS: usize = 3;
    for sweep in 0..MAX_SWEEPS {
        let mut sweep_deltas = Vec::with_capacity(specs.len());
        let mut improved = false;

        // Phase 1: dynamics-affecting specs — full KF re-run per candidate.
        for spec in specs.iter().filter(|s| !s.is_diagnostic_only) {
            let (name, delta, spec_improved) = sweep_one_spec(
                &mut params,
                &mut current_eval,
                spec,
                opts.target_recall,
                progress,
                round_index,
                max_rounds,
                sweep,
                |trial| {
                    evaluate(
                        trial,
                        base_config,
                        base_context,
                        obs_dataset,
                        round_ids,
                        geometry_cache,
                        opts.completion_threshold,
                        opts.within_radius_threshold,
                    )
                },
            );
            sweep_deltas.push((name, delta));
            improved |= spec_improved;
        }

        // Phase 2: diagnostic-only specs — one reference run for the whole
        // phase, reused across every candidate of every diagnostic-only spec.
        let diagnostic_specs: Vec<&ParamSpec> =
            specs.iter().filter(|s| s.is_diagnostic_only).collect();
        if !diagnostic_specs.is_empty() {
            let dynamics_config = params.apply(base_config);
            let dynamics_context = params.build_context(base_context);
            let reference_runs = build_reference_runs(
                &dynamics_config,
                &dynamics_context,
                obs_dataset,
                round_ids,
                geometry_cache,
            );

            for spec in diagnostic_specs {
                let (name, delta, spec_improved) = sweep_one_spec(
                    &mut params,
                    &mut current_eval,
                    spec,
                    opts.target_recall,
                    progress,
                    round_index,
                    max_rounds,
                    sweep,
                    |trial| {
                        evaluate_diagnostic_only(
                            trial,
                            base_config,
                            &reference_runs,
                            round_ids,
                            opts.completion_threshold,
                            opts.within_radius_threshold,
                        )
                    },
                );
                sweep_deltas.push((name, delta));
                improved |= spec_improved;
            }
        }

        last_sweep_deltas = sweep_deltas;
        if !improved {
            break;
        }
    }

    (params, current_eval, last_sweep_deltas)
}

/// One [`ParamSpec`]'s worth of candidate trial-and-keep-best, factored out
/// of [`coordinate_descent`] so its two phases (full re-run vs. cheap
/// reference-run reuse) can share the exact same candidate-selection logic,
/// differing only in `eval_fn` — `|trial_params| ...` calling either
/// [`evaluate`] or
/// [`evaluate_diagnostic_only`](crate::kf_calibration::objective::evaluate_diagnostic_only).
///
/// Mutates `params`/`current_eval` in place when a candidate beats the
/// current value (by [`comparison_key`]); returns `(spec name, score delta,
/// whether it changed)` for the caller's sweep-level bookkeeping.
#[allow(clippy::too_many_arguments)]
fn sweep_one_spec(
    params: &mut CalibrationParams,
    current_eval: &mut RoundEval,
    spec: &ParamSpec,
    target_recall: f64,
    progress: &ProgressBar,
    round_index: usize,
    max_rounds: usize,
    sweep: usize,
    mut eval_fn: impl FnMut(&CalibrationParams) -> RoundEval,
) -> (String, f64, bool) {
    let current_value = (spec.get)(params);
    let before_score = scalar_score(current_eval, target_recall);

    let mut best_value = current_value;
    let mut best_eval = current_eval.clone();

    let candidates = spec.candidates(current_value);
    for (candidate_index, candidate_value) in candidates.iter().enumerate() {
        let candidate_value = *candidate_value;
        if (candidate_value - current_value).abs() < f64::EPSILON {
            continue;
        }
        progress.set_message(format!(
            "round {round_index}/{max_rounds} · sweep {} · {} ({}/{}) · best recall={:.1}% cost={:.1}\"",
            sweep + 1,
            spec.name,
            candidate_index + 1,
            candidates.len(),
            current_eval.recall * 100.0,
            current_eval.cost,
        ));
        let mut trial_params = *params;
        (spec.set)(&mut trial_params, candidate_value);
        let trial_eval = eval_fn(&trial_params);
        if comparison_key(&trial_eval, target_recall) > comparison_key(&best_eval, target_recall) {
            best_value = candidate_value;
            best_eval = trial_eval;
        }
    }

    let after_score = scalar_score(&best_eval, target_recall);
    let delta = after_score - before_score;

    let improved = best_value != current_value;
    if improved {
        (spec.set)(params, best_value);
        *current_eval = best_eval;
    }

    (spec.name.to_string(), delta, improved)
}

/// Scalar score used to report a sweep's per-parameter gain (see
/// [`coordinate_descent`]): recall itself while still below
/// `target_recall` (bigger is better progress toward the goal), `-cost`
/// once at/above it (bigger — i.e. smaller cost — is better once recall is
/// no longer the bottleneck). Only used for the human-readable delta in
/// [`RoundResult::param_deltas`]; candidate selection itself always goes
/// through [`comparison_key`], not this.
fn scalar_score(eval: &RoundEval, target_recall: f64) -> f64 {
    if eval.recall >= target_recall {
        -eval.cost
    } else {
        eval.recall
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(n: usize) -> Vec<TrajId> {
        (0..n as u32).map(TrajId::Int).collect()
    }

    #[test]
    fn sample_round_ids_is_reproducible_with_same_seed() {
        let all_ids = ids(100);
        let mut rng_a = StdRng::seed_from_u64(7);
        let mut rng_b = StdRng::seed_from_u64(7);

        let a = sample_round_ids(&all_ids, &[], 10, &mut rng_a);
        let b = sample_round_ids(&all_ids, &[], 10, &mut rng_b);

        assert_eq!(a, b);
    }

    #[test]
    fn sample_round_ids_always_includes_the_whole_failing_pool() {
        let all_ids = ids(100);
        let failing = vec![TrajId::Int(3), TrajId::Int(41)];
        let mut rng = StdRng::seed_from_u64(1);

        let round = sample_round_ids(&all_ids, &failing, 10, &mut rng);

        assert_eq!(round.len(), 10);
        for f in &failing {
            assert!(round.contains(f));
        }
    }

    #[test]
    fn sample_round_ids_caps_an_oversized_failing_pool() {
        let all_ids = ids(100);
        let failing = ids(50);
        let mut rng = StdRng::seed_from_u64(1);

        let round = sample_round_ids(&all_ids, &failing, 10, &mut rng);

        assert_eq!(round.len(), 10);
    }

    #[test]
    fn cap_pool_is_a_no_op_under_the_limit() {
        let pool = ids(5);
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(cap_pool(pool.clone(), 10, &mut rng).len(), 5);
    }

    #[test]
    fn cap_pool_truncates_over_the_limit() {
        let pool = ids(50);
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(cap_pool(pool, 10, &mut rng).len(), 10);
    }
}
