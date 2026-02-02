use std::fmt;

use fink_fat_engine::engine_config::EngineConfig;
use fink_fat_engine::engine_config::score_config::ScoreConfig;
use fink_fat_engine::graph::edge::Edge;
use fink_fat_engine::graph::edge::edge_id::EdgeId;
use rand::Rng;
use rand::rngs::StdRng;

use crate::night_seeds::{LabeledEdge, SeedStore};
use crate::scoring::frozen_pairs::FrozenPair;
use crate::scoring::optimization_metrics::EdgeSeparationMetrics;
use anyhow::Result;
use fink_fat_engine::graph::edge::score::ScoredEdge;

/// Result of a single optimisation run.
#[derive(Debug, Clone)]
pub struct OptimisationResult {
    /// Best scoring configuration found.
    pub best_cfg: Option<ScoreConfig>,
    /// False positive rate at the target true positive rate for the best configuration.
    pub best_fpr: f64,
    /// Separation metrics of the best configuration (useful for diagnostics).
    pub best_metrics: Option<EdgeSeparationMetrics>,
}

// -----------------------------------------------------------------------------
// Random-search driver (refactor)
// -----------------------------------------------------------------------------

/// Bookkeeping counters for the optimization loop.
#[derive(Debug, Default, Clone)]
pub struct OptStats {
    pub invalid: usize,
    pub unreachable: usize,
    pub valid_eval: usize,
}

/// Best-so-far accumulator.
#[derive(Debug, Clone)]
pub struct BestSoFar {
    pub fpr: f64,
    pub cfg: Option<ScoreConfig>,
    pub metrics: Option<EdgeSeparationMetrics>,
}

impl Default for BestSoFar {
    fn default() -> Self {
        Self {
            fpr: f64::INFINITY,
            cfg: None,
            metrics: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EvalStats {
    pub tot_good: usize,
    pub tot_bad: usize,
    pub acc_good: usize,
    pub acc_bad: usize,
    pub acc_cost_good: usize,
    pub acc_cost_bad: usize,
}

impl fmt::Display for EvalStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tot_good={} tot_bad={} acc_good={} acc_bad={} acc_cost_good={} acc_cost_bad={}",
            self.tot_good,
            self.tot_bad,
            self.acc_good,
            self.acc_bad,
            self.acc_cost_good,
            self.acc_cost_bad
        )
    }
}

/// Sample one candidate `ScoreConfig` from a baseline `engine_cfg.scoring`.
///
/// Sampling policy
/// ---------------
/// - `numeric.min_variance`: log-uniform in [1e-12, 1e-4]
/// - `predict.noise.*`: log-uniform with broad priors
/// - `position.max_d2`: uniform in [1, 50]
/// - `position.w_pos`: uniform in [0.1, 5]
pub fn sample_candidate_cfg(rng: &mut StdRng, engine_cfg: &EngineConfig) -> ScoreConfig {
    let mut cfg = engine_cfg.edges.score_config.clone();

    let min_var_pow = rng.random_range(-12.0..-4.0);
    cfg.numeric.min_variance = 10f64.powf(min_var_pow);

    let var_floor_pow = rng.random_range(-13.0..-5.0);
    let drift_pow = rng.random_range(-13.0..-7.0);
    let curv_pow = rng.random_range(-15.0..-9.0);
    cfg.predict.noise.variance_floor = 10f64.powf(var_floor_pow);
    cfg.predict.noise.drift_per_day = 10f64.powf(drift_pow);
    cfg.predict.noise.curvature_per_day2 = 10f64.powf(curv_pow);

    // cfg.position.max_d2 = rng.random_range(1.0..50.0);
    cfg.position.w_pos = rng.random_range(0.1..5.0);

    cfg
}

/// Evaluate a scoring configuration on a frozen set of pairs.
///
/// For each frozen pair, calls `ScoredEdge::score` with `delta=1` and, when
/// scoring succeeds, records the resulting `ScoredEdge` along with the class
/// label. Returns the FPR at the given TPR and optional separation metrics.
fn evaluate_cfg_on_frozen_pairs(
    cfg: &ScoreConfig,
    seed_store: &SeedStore,
    frozen_pairs: &[FrozenPair],
) -> Option<(Option<EdgeSeparationMetrics>, usize, EvalStats)> {
    let mut edges: Vec<LabeledEdge> = Vec::with_capacity(frozen_pairs.len());

    let mut tot_good = 0usize;
    let mut tot_bad = 0usize;
    let mut acc_good = 0usize;
    let mut acc_bad = 0usize;
    let mut acc_cost_good = 0usize;
    let mut acc_cost_bad = 0usize;

    let mut accepted_costs: Vec<(f64, bool)> = Vec::with_capacity(frozen_pairs.len());

    let mut edge_id = 0u64;
    for p in frozen_pairs {
        if p.same {
            tot_good += 1
        } else {
            tot_bad += 1
        }

        let a_seeds = &seed_store.get(&p.a_nid)?.seeds;
        let b_seeds = &seed_store.get(&p.b_nid)?.seeds;
        let si = &a_seeds[p.a_idx];
        let sj = &b_seeds[p.b_idx];

        if let Some(edge) = ScoredEdge::score(si, sj, cfg, p.delta) {
            if p.same {
                acc_good += 1
            } else {
                acc_bad += 1
            }

            let cost = edge.cost;
            if cost.is_finite() {
                accepted_costs.push((cost, p.same));
                if p.same {
                    acc_cost_good += 1
                } else {
                    acc_cost_bad += 1
                }
            }

            let edge = Edge::new(EdgeId(edge_id), si, sj, edge.cost, edge.dt_days);
            edge_id += 1;

            edges.push(LabeledEdge { same: p.same, edge });
        }
    }

    let stats = EvalStats {
        tot_good,
        tot_bad,
        acc_good,
        acc_bad,
        acc_cost_good,
        acc_cost_bad,
    };

    // Hard failure: nothing was accepted at all.
    if edges.is_empty() {
        return None;
    }

    let metrics = EdgeSeparationMetrics::edge_metrics(&edges, tot_good);
    Some((metrics, edges.len(), stats))
}

/// Validate + evaluate a candidate on frozen pairs.
///
/// Returns `Ok(Some(...))` for a valid evaluation,
/// `Ok(None)` when evaluation is "unreachable" (no edges survive, or metric can't be computed),
/// and `Err(...)` when config validation fails.
pub fn validate_and_eval_candidate(
    cfg: &ScoreConfig,
    seed_store: &SeedStore,
    frozen_pairs: &[FrozenPair],
) -> Result<Option<(Option<EdgeSeparationMetrics>, usize, EvalStats)>, anyhow::Error> {
    cfg.validate()?; // treat validation errors as "hard invalid"
    Ok(evaluate_cfg_on_frozen_pairs(cfg, seed_store, frozen_pairs))
}

// Optimise a subset of scoring parameters to minimise `FPR@TPR`.
//
// (Refactored version: smaller helpers, same semantics as before.)
// pub fn optimize_scoring_params(
//     seed_store: &SeedStore,
//     engine_cfg: &EngineConfig,
//     sampling: &EdgeSampling,
// ) -> Result<OptimisationResult> {
//     println!("\n\n === Starting optimization ===\n");

//     let frozen_pairs = freeze_balanced_edge_pairs(&seed_store, &sampling);

//     println!("\n\n === Summary of frozen balanced pairs ===");

//     println!(
//         "[score-opt] total frozen balanced pairs: {}",
//         frozen_pairs.len()
//     );

//     let nb_good = frozen_pairs.iter().filter(|p| p.same).count();
//     let nb_bad = frozen_pairs.len() - nb_good;

//     println!("[score-opt]   good pairs: {}", nb_good);
//     println!("[score-opt]   bad pairs:  {}", nb_bad);

//     // 2) Initialize RNG, stats, and best accumulator.
//     let mut rng = rand::rngs::StdRng::seed_from_u64(42_u64);
//     let mut stats = OptStats::default();
//     let mut best = BestSoFar::default();

//     println!("\n\n === Random search iterations ===");
//     let num_iters = 5000;

//     // 3) Random search loop (parallel evaluation).
//     //
//     // IMPORTANT: we keep the same RNG semantics by sampling candidates sequentially,
//     // then evaluating them in parallel.
//     let candidates: Vec<_> = (0..num_iters)
//         .map(|_| sample_candidate_cfg(&mut rng, &engine_cfg))
//         .collect();

//     println!(
//         "[score-opt] sampled {} candidate configurations.",
//         candidates.len()
//     );

//     println!("[score-opt] evaluating candidates in parallel...");

//     // Parallel evaluation only (no shared mutation, no prints here).
//     let mut outcomes: Vec<IterOutcome> = candidates
//         .par_iter()
//         .enumerate()
//         .map(|(iter0, cfg)| {
//             let eval = match validate_and_eval_candidate(cfg, &seed_store, &frozen_pairs) {
//                 Ok(x) => x,
//                 Err(err) => {
//                     return IterOutcome::Invalid {
//                         iter0,
//                         // If your error type isn't anyhow::Error, store it directly.
//                         err: err.into(),
//                     };
//                 }
//             };

//             let Some((metrics_opt, _, _)) = eval else {
//                 return IterOutcome::Unreachable { iter0 };
//             };

//             IterOutcome::Ok {
//                 iter0,
//                 cfg: cfg.clone(),
//                 metrics_opt,
//             }
//         })
//         .collect();

//     println!("[score-opt] processing evaluation results...");

//     // To preserve the original "online" behavior (stats/logging/best update),
//     // we process results in iteration order.
//     outcomes.sort_by_key(|o| match o {
//         IterOutcome::Invalid { iter0, .. } => *iter0,
//         IterOutcome::Unreachable { iter0 } => *iter0,
//         IterOutcome::Ok { iter0, .. } => *iter0,
//     });

//     for outcome in outcomes {
//         let iter0 = match &outcome {
//             IterOutcome::Invalid { iter0, .. } => *iter0,
//             IterOutcome::Unreachable { iter0 } => *iter0,
//             IterOutcome::Ok { iter0, .. } => *iter0,
//         };
//         let iter = iter0 + 1;

//         match outcome {
//             IterOutcome::Invalid { iter0, err } => {
//                 stats.invalid += 1;
//                 // Keep the same "occasional log" behavior.
//                 if (iter0 % 50) == 0 {
//                     eprintln!(
//                         "[score-opt][{:>4}/{}] invalid candidate (skipped): {:?}",
//                         iter, num_iters, err
//                     );
//                 }
//             }
//             IterOutcome::Unreachable { .. } => {
//                 stats.unreachable += 1;
//                 println!(
//                     "[score-opt][{:>4}/{}] candidate unreachable (skipped)",
//                     iter, num_iters
//                 );
//             }
//             IterOutcome::Ok {
//                 cfg, metrics_opt, ..
//             } => {
//                 let fpr_99 = metrics_opt.unwrap().edge_metrics.fpr_at_tpr_99;
//                 let fpr_95 = metrics_opt.unwrap().edge_metrics.fpr_at_tpr_95;

//                 if fpr_99.is_finite() && fpr_99 < best.fpr {
//                     let prev = best.fpr;

//                     best.cfg = Some(cfg.clone());
//                     best.metrics = metrics_opt;

//                     println!(
//                         "[score-opt][{:>4}/{}] NEW BEST! \n{} \n\t(prev {:.6}, Δ {:.6}) \n\tFPR@0.95:\n{}",
//                         iter,
//                         num_iters,
//                         fpr_99,
//                         prev,
//                         prev - fpr_99,
//                         fpr_95
//                     );
//                 }
//             }
//         }
//     }

//     println!("\n\n === Optimization complete ===\n");

//     println!("\n\n Best params found:\n{:#?}", best.cfg);
//     if let Some(metrics) = &best.metrics {
//         println!("\nBest metrics:\n{:#?}", metrics);
//     }

//     Ok(OptimisationResult {
//         best_cfg: best.cfg,
//         best_fpr: best.fpr,
//         best_metrics: best.metrics,
//     })
// }
