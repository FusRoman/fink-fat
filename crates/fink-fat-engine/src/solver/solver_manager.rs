//! Solver orchestration: routing policy, planning, and execution.
//!
//! Overview
//! --------
//! Solving the inter-night graph is performed **component by component**.
//! Components can differ widely in size, density, and night span, so routing each
//! component to an appropriate solver strategy helps keep runtime predictable.
//!
//! This module defines:
//! - [`SolverChoice`]: which solver strategy to use for one component,
//! - [`SolverRoutingMode`] and [`SolverPolicy`]: parameters and heuristics for routing,
//! - [`SolvePlan`]: a list of work items `(component_id, choice)`,
//! - [`SolverManager`]: orchestration that builds a plan and executes it.
//!
//! Conceptual pipeline
//! -------------------
//! 1) Build [`ConnectedComponents`] from the global `RuntimeGraph` and [`SeedStore`].
//!
//! 2) For each component, compute a routing decision ([`SolverChoice`]).
//!
//! 3) Execute the selected solver on each component and collect [`SolverOutput`].
//!
//! Routing modes
//! -------------
//! The routing policy supports two modes:
//! - `Heuristics`
//!   - choose a solver based on component-level statistics,
//!   - intended for heterogeneous workloads.
//! - `Force(choice)`
//!   - route every component to the same solver,
//!   - intended for debugging, benchmarking, and evaluation.
//!
//! Heuristic signals
//! -----------------
//! Heuristics rely on summary statistics exposed by [`ConnectedComponents`]:
//! - number of nodes (`n_nodes`),
//! - number of active intra-component edges (`m_active_edges`),
//! - night span (`max_night - min_night`).
//!
//! The policy also contains a simple time model for estimating the cost of
//! running a min-cost flow solver:
//!
//! ```text
//! t_est ~ k_mcf_s_per_edge_logn * m_edges * log2(n_nodes + 1)
//! ```
//!
//! This model is intentionally simple and designed to be calibrated empirically.
//!
//! Notes
//! -----
//! - This file currently instantiates and runs only the bounded beam solver.
//! - Other solver choices are wired as `todo!()` placeholders.

use crate::{
    engine_config::solver_config::{
        bounded_beam_config::BoundedBeamConfig,
        solver_policy::{SolverChoice, SolverPolicy, SolverRoutingMode},
    },
    graph::AlertLinkageDAG,
    pipeline::hooks::StageProgress,
    seeding::store::SeedStore,
    solver::{
        Solver, SolverOutput, bounded_beam::BoundedBeamSolver, components::ConnectedComponents,
    },
};

/// A single work item produced by the planner.
///
/// A work item defines:
/// - which component to solve (`component_id`),
/// - which solver family to use (`choice`).
#[derive(Clone, Debug)]
pub struct WorkItem {
    /// Dense component id.
    pub component_id: u32,

    /// Solver family selected for this component.
    pub choice: SolverChoice,
}

/// A plan for one solver pass.
///
/// The plan is a list of work items. Execution order is the plan order.
///
/// Notes
/// -----
/// Keeping an explicit plan makes runs more reproducible and debuggable:
/// - routing can be inspected separately from execution,
/// - the plan can be reused for benchmarking multiple solver implementations.
#[derive(Clone, Debug, Default)]
pub struct SolvePlan {
    /// Work items to execute.
    pub items: Vec<WorkItem>,
}

/// Manager object holding the routing policy.
///
/// The manager is responsible for:
/// - producing a [`SolvePlan`] (routing decision per component),
/// - executing the plan and collecting outputs.
///
/// Notes
/// -----
/// - This object does not own the graph or components; it only orchestrates.
#[derive(Clone, Debug, Default)]
pub struct SolverManager {
    /// Routing policy controlling solver selection.
    pub policy: SolverPolicy,

    /// Configuration for the bounded beam solver.
    pub bounded_beam_config: BoundedBeamConfig,
}

impl SolverManager {
    /// Build a solve plan from component statistics and the current policy.
    ///
    /// In `Heuristics` mode, solver selection is delegated to:
    /// `ConnectedComponents::classify(component_id, &policy)`.
    ///
    /// In `Force(choice)` mode, the forced choice is used for every component.
    ///
    /// Arguments
    /// ---------
    /// * `comps` – Connected components object used to access component stats.
    ///
    /// Return
    /// ------
    /// `SolvePlan` containing one [`WorkItem`] per component.
    ///
    /// Notes
    /// -----
    /// - The plan order is `component_id` ascending (`0..n_components`).
    /// - Keeping this plan explicit makes routing easy to inspect and reuse.
    pub fn make_plan(&self, comps: &ConnectedComponents) -> SolvePlan {
        let routing_mode = match self.policy.routing {
            SolverRoutingMode::Heuristics => "heuristics",
            SolverRoutingMode::Force(_) => "force",
        };
        tracing::debug!(
            n_components = comps.n_components,
            routing_mode,
            "make_plan starting",
        );

        let mut items = Vec::with_capacity(comps.n_components as usize);
        let mut n_bounded_beam = 0u32;
        let mut n_mcf = 0u32;
        let mut n_blob_breaker = 0u32;

        for cid in 0..comps.n_components {
            let cid_u32 = cid;

            let choice = match self.policy.routing {
                SolverRoutingMode::Heuristics => comps.classify(cid_u32, &self.policy),
                SolverRoutingMode::Force(c) => c,
            };

            match choice {
                SolverChoice::BoundedBeam => n_bounded_beam += 1,
                SolverChoice::MinCostFlow => n_mcf += 1,
                SolverChoice::BlobBreaker => n_blob_breaker += 1,
            }

            items.push(WorkItem {
                component_id: cid_u32,
                choice,
            });
        }

        tracing::debug!(n_bounded_beam, n_mcf, n_blob_breaker, "make_plan complete");

        SolvePlan { items }
    }

    /// Execute the plan and collect one [`SolverOutput`] per work item.
    ///
    /// Each work item is executed independently on its component.
    ///
    /// Arguments
    /// ---------
    /// * `comps` – Connected components object providing per-component restricted views.
    /// * `graph` – Global runtime graph backing edges/nodes referenced in outputs.
    /// * `_seed_store` – Seed store (currently unused by implemented solvers; reserved for future use).
    /// * `plan` – Plan generated by [`SolverManager::make_plan`].
    ///
    /// Return
    /// ------
    /// `Vec<SolverOutput>` – Outputs in the same order as `plan.items`.
    ///
    /// Notes
    /// -----
    /// - Only the bounded beam solver is currently implemented here.
    /// - Other solver choices are placeholders.
    /// - The returned vector preserves plan order for reproducible downstream processing.
    pub fn run_plan<'edge_lf, 'seed_lf>(
        &self,
        comps: &'edge_lf ConnectedComponents<'edge_lf, 'seed_lf>,
        graph: &'edge_lf AlertLinkageDAG,
        _seed_store: &'seed_lf SeedStore,
        plan: &SolvePlan,
        progress_sink: &dyn StageProgress,
    ) -> Vec<SolverOutput>
    where
        'edge_lf: 'seed_lf,
    {
        // Solver instances used by this manager.
        let bounded_beam = BoundedBeamSolver::new(self.bounded_beam_config.clone());

        tracing::debug!(n_items = plan.items.len(), "run_plan starting");

        let mut outputs: Vec<SolverOutput> = Vec::with_capacity(plan.items.len());

        for item in &plan.items {
            // Dispatch by solver family.
            let out = match item.choice {
                SolverChoice::BoundedBeam => bounded_beam.solve(graph, comps, item.component_id),

                SolverChoice::MinCostFlow => {
                    todo!("MinCostFlow solver not implemented yet")
                }

                SolverChoice::BlobBreaker => {
                    todo!("BlobBreaker solver not implemented yet")
                }
            };

            if out.diag.n_selected > 0 {
                tracing::trace!(
                    component_id = item.component_id,
                    solver = out.diag.solver_name,
                    n_nodes = out.diag.n_nodes,
                    n_tracks = out.diag.n_selected,
                    n_candidates = out.diag.n_candidates,
                    n_expansions = out.diag.n_expansions,
                    "component solved",
                );
            }

            outputs.push(out);
            progress_sink.inc(1);
        }

        let total_tracks: u32 = outputs.iter().map(|o| o.diag.n_selected).sum();
        let n_with_tracks = outputs.iter().filter(|o| o.diag.n_selected > 0).count();
        tracing::debug!(
            n_outputs = outputs.len(),
            total_tracks,
            n_with_tracks,
            "run_plan complete",
        );

        log_diag_stats(&outputs);

        outputs
    }
}

// ---------------------------------------------------------------------------
// Aggregated diagnostics — logged once at the end of every run_plan call
// ---------------------------------------------------------------------------

/// Summary statistics over a numeric sample (all fields are pre-sorted).
struct DiagStats {
    mean: f64,
    median: f64,
    p90: f64,
    p99: f64,
    max: f64,
}

impl DiagStats {
    /// Compute stats from an owned `Vec<f64>`. Returns `None` for empty input.
    fn compute(mut vals: Vec<f64>) -> Option<Self> {
        if vals.is_empty() {
            return None;
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = vals.len();
        let mean = vals.iter().sum::<f64>() / n as f64;
        let median = sorted_percentile(&vals, 0.50);
        let p90 = sorted_percentile(&vals, 0.90);
        let p99 = sorted_percentile(&vals, 0.99);
        let max = vals[n - 1];
        Some(DiagStats {
            mean,
            median,
            p90,
            p99,
            max,
        })
    }
}

/// Nearest-rank percentile on an already-sorted slice (p in [0.0, 1.0]).
#[inline]
fn sorted_percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    let idx = ((n as f64 * p).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    sorted[idx]
}

/// Emit aggregated `SolverDiagnostics` statistics to the tracing log at DEBUG
/// level.
///
/// One `DEBUG` event is emitted per tracked field (n_nodes, m_active_edges,
/// n_candidates, n_selected, n_expansions, candidates/selected ratio), plus
/// one event for tuning signals (zero-track components, ratio diagnostics).
///
/// ### Tuning signals surfaced
///
/// | Log field               | Meaning                                        |
/// |-------------------------|------------------------------------------------|
/// | `n_components_0_tracks` | components that produced no track (missed)     |
/// | `n_selected` stats      | if median ≪ max_tracks → beam width too small  |
/// | `n_candidates` stats    | if p99 ≫ p99(n_selected) → heavy pruning       |
/// | `n_expansions` stats    | if p99 is large → budget exhaustion risk       |
/// | `cand_sel_ratio` stats  | candidates examined per selected track         |
fn log_diag_stats(outputs: &[SolverOutput]) {
    if outputs.is_empty() {
        return;
    }

    // Collect per-field sample vectors.
    let n_nodes_v: Vec<f64> = outputs.iter().map(|o| o.diag.n_nodes as f64).collect();
    let m_edges_v: Vec<f64> = outputs
        .iter()
        .map(|o| o.diag.m_active_edges as f64)
        .collect();
    let n_cands_v: Vec<f64> = outputs.iter().map(|o| o.diag.n_candidates as f64).collect();
    let n_sel_v: Vec<f64> = outputs.iter().map(|o| o.diag.n_selected as f64).collect();
    let n_exp_v: Vec<f64> = outputs.iter().map(|o| o.diag.n_expansions as f64).collect();

    // candidates-per-selected ratio (only for components that produced tracks).
    let ratio_v: Vec<f64> = outputs
        .iter()
        .filter(|o| o.diag.n_selected > 0)
        .map(|o| o.diag.n_candidates as f64 / o.diag.n_selected as f64)
        .collect();

    // Tuning signal: components with zero tracks.
    let n_zero = outputs.iter().filter(|o| o.diag.n_selected == 0).count();
    let pct_zero = 100.0 * n_zero as f64 / outputs.len() as f64;

    // Helper macro: emit one DEBUG event per field if stats are available.
    macro_rules! log_stat {
        ($label:expr, $vals:expr) => {
            if let Some(s) = DiagStats::compute($vals) {
                tracing::debug!(
                    stat = $label,
                    mean = s.mean.round() as u64,
                    median = s.median.round() as u64,
                    p90 = s.p90.round() as u64,
                    p99 = s.p99.round() as u64,
                    max = s.max.round() as u64,
                    "run_plan diag stats",
                );
            }
        };
    }

    log_stat!("n_nodes", n_nodes_v);
    log_stat!("m_active_edges", m_edges_v);
    log_stat!("n_candidates", n_cands_v);
    log_stat!("n_selected", n_sel_v);
    log_stat!("n_expansions", n_exp_v);

    // Ratio gets one decimal place — keep it as f64 rounded to 1 dp.
    if let Some(s) = DiagStats::compute(ratio_v) {
        let (mean, median, p90, p99, max) = (
            (s.mean * 10.0).round() / 10.0,
            (s.median * 10.0).round() / 10.0,
            (s.p90 * 10.0).round() / 10.0,
            (s.p99 * 10.0).round() / 10.0,
            (s.max * 10.0).round() / 10.0,
        );
        tracing::debug!(
            stat = "cand_sel_ratio",
            mean,
            median,
            p90,
            p99,
            max,
            "run_plan diag stats",
        );
    }

    tracing::debug!(
        n_components = outputs.len(),
        n_components_0_tracks = n_zero,
        pct_components_0_tracks = pct_zero.round() as u64,
        // hint: if n_selected p99 is close to max_tracks, raise max_tracks
        // hint: if n_expansions p99 is close to max_expansions, raise budget
        // hint: if cand_sel_ratio p99 is very high, raise max_out_per_node
        "run_plan tuning signals",
    );
}

// ---------------------------------------------------------------------------
// Unit tests — stats helpers
// ---------------------------------------------------------------------------

#[cfg(test)]
mod diag_stats_tests {
    use super::*;
    use proptest::prelude::*;

    // -----------------------------------------------------------------------
    // sorted_percentile — deterministic unit tests
    // -----------------------------------------------------------------------

    /// Any p on a single-element vector returns that element.
    #[test]
    fn percentile_single_element_returns_it_for_any_p() {
        let v = vec![42.0_f64];
        for p in [0.0, 0.01, 0.25, 0.50, 0.75, 0.99, 1.0] {
            assert_eq!(sorted_percentile(&v, p), 42.0, "p={p}");
        }
    }

    /// p=1.0 always returns the last (maximum) element of a sorted slice.
    #[test]
    fn percentile_p100_returns_last_element() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 100.0];
        assert_eq!(sorted_percentile(&v, 1.0), 100.0);
    }

    /// Median (p=0.50) of a five-element sorted vector.
    ///
    /// ceil(5 × 0.5) = 3  →  idx = 2  →  v[2] = 3.0
    #[test]
    fn percentile_median_five_element_sorted_vec() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(sorted_percentile(&v, 0.50), 3.0);
    }

    /// p90 of a ten-element vector [1..=10].
    ///
    /// ceil(10 × 0.9) = 9  →  idx = 8  →  v[8] = 9.0
    #[test]
    fn percentile_p90_ten_element_vec() {
        let v: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        assert_eq!(sorted_percentile(&v, 0.90), 9.0);
    }

    /// Percentile values are monotonically non-decreasing as p increases.
    #[test]
    fn percentile_monotone_non_decreasing_over_increasing_p() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ps = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0];
        let values: Vec<f64> = ps.iter().map(|&p| sorted_percentile(&v, p)).collect();
        for w in values.windows(2) {
            assert!(w[0] <= w[1], "not monotone: {w:?}");
        }
    }

    // -----------------------------------------------------------------------
    // DiagStats::compute — deterministic unit tests
    // -----------------------------------------------------------------------

    /// Empty input returns None.
    #[test]
    fn compute_empty_returns_none() {
        assert!(DiagStats::compute(vec![]).is_none());
    }

    /// Single-element input: all five fields equal that element.
    #[test]
    fn compute_single_element_all_fields_equal() {
        let s = DiagStats::compute(vec![7.0]).unwrap();
        assert_eq!(s.mean, 7.0, "mean");
        assert_eq!(s.median, 7.0, "median");
        assert_eq!(s.p90, 7.0, "p90");
        assert_eq!(s.p99, 7.0, "p99");
        assert_eq!(s.max, 7.0, "max");
    }

    /// Unsorted input is sorted internally; result is independent of input order.
    #[test]
    fn compute_is_order_independent() {
        let asc = DiagStats::compute(vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let desc = DiagStats::compute(vec![5.0, 4.0, 3.0, 2.0, 1.0]).unwrap();
        let rand = DiagStats::compute(vec![3.0, 1.0, 5.0, 2.0, 4.0]).unwrap();
        for (a, b) in [
            (asc.mean, desc.mean),
            (asc.median, desc.median),
            (asc.p90, desc.p90),
            (asc.p99, desc.p99),
            (asc.max, desc.max),
        ] {
            assert_eq!(a, b, "asc != desc");
        }
        for (a, b) in [
            (asc.mean, rand.mean),
            (asc.median, rand.median),
            (asc.p90, rand.p90),
            (asc.p99, rand.p99),
            (asc.max, rand.max),
        ] {
            assert_eq!(a, b, "asc != rand");
        }
    }

    /// Known ten-element vector [1..=10]:
    ///   mean=5.5, median=5, p90=9, p99=10, max=10.
    #[test]
    fn compute_known_ten_element_vector() {
        let vals: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let s = DiagStats::compute(vals).unwrap();
        assert_eq!(s.mean, 5.5, "mean");
        assert_eq!(s.median, 5.0, "median");
        assert_eq!(s.p90, 9.0, "p90");
        assert_eq!(s.p99, 10.0, "p99");
        assert_eq!(s.max, 10.0, "max");
    }

    /// Ordering invariant: median ≤ p90 ≤ p99 ≤ max for a random-looking vec.
    #[test]
    fn compute_ordering_invariant_on_unsorted_input() {
        let s = DiagStats::compute(vec![9.0, 1.0, 5.0, 3.0, 7.0, 2.0, 8.0]).unwrap();
        assert!(s.median <= s.p90, "median({}) <= p90({})", s.median, s.p90);
        assert!(s.p90 <= s.p99, "p90({})    <= p99({})", s.p90, s.p99);
        assert!(s.p99 <= s.max, "p99({})    <= max({})", s.p99, s.max);
    }

    // -----------------------------------------------------------------------
    // sorted_percentile — property-based tests
    // -----------------------------------------------------------------------

    proptest! {
        /// For any non-empty vector and any p ∈ [0, 1], the result lies within
        /// [min(input), max(input)].
        #[test]
        fn prop_percentile_result_between_min_and_max(
            elems in proptest::collection::vec(0.0f64..1_000.0, 1..=50),
            p in 0.0f64..=1.0,
        ) {
            let min_val = elems.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_val = elems.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut sorted = elems.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let result = sorted_percentile(&sorted, p);
            prop_assert!(
                result >= min_val && result <= max_val,
                "percentile(p={p}) = {result} outside [{min_val}, {max_val}]"
            );
        }

        /// The returned value is always one of the elements in the input.
        #[test]
        fn prop_percentile_result_is_one_of_the_inputs(
            elems in proptest::collection::vec(0.0f64..1_000.0, 1..=30),
            p in 0.0f64..=1.0,
        ) {
            let mut sorted = elems.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let result = sorted_percentile(&sorted, p);
            prop_assert!(
                sorted.iter().any(|&v| v == result),
                "result {result} not found in sorted input {sorted:?}"
            );
        }

        // -----------------------------------------------------------------------
        // DiagStats — property-based tests
        // -----------------------------------------------------------------------

        /// Ordering: median ≤ p90 ≤ p99 ≤ max for any non-empty input.
        #[test]
        fn prop_diag_stats_ordering(
            elems in proptest::collection::vec(0.0f64..1_000.0, 1..=50),
        ) {
            let s = DiagStats::compute(elems.clone()).unwrap();
            prop_assert!(s.median <= s.p90, "median({}) > p90({}) for {:?}", s.median, s.p90, elems);
            prop_assert!(s.p90    <= s.p99, "p90({})    > p99({}) for {:?}", s.p90,    s.p99, elems);
            prop_assert!(s.p99    <= s.max, "p99({})    > max({}) for {:?}", s.p99,    s.max, elems);
        }

        /// mean lies within [min(input), max(input)].
        #[test]
        fn prop_diag_stats_mean_in_range(
            elems in proptest::collection::vec(0.0f64..1_000.0, 1..=50),
        ) {
            let min_val = elems.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_val = elems.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let s = DiagStats::compute(elems.clone()).unwrap();
            prop_assert!(
                s.mean >= min_val - 1e-9 && s.mean <= max_val + 1e-9,
                "mean ({}) outside [{min_val}, {max_val}] for {:?}",
                s.mean, elems
            );
        }

        /// max equals the actual maximum element of the input.
        #[test]
        fn prop_diag_stats_max_is_true_maximum(
            elems in proptest::collection::vec(0.0f64..1_000.0, 1..=50),
        ) {
            let true_max = elems.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let s = DiagStats::compute(elems).unwrap();
            prop_assert_eq!(s.max, true_max);
        }

        /// For a constant vector every field equals that constant.
        #[test]
        fn prop_diag_stats_constant_vector(
            val in 0.0f64..1_000.0,
            n   in 1usize..=20,
        ) {
            let elems = vec![val; n];
            let s = DiagStats::compute(elems).unwrap();
            prop_assert!((s.mean   - val).abs() < 1e-9, "mean   != {val}");
            prop_assert!((s.median - val).abs() < 1e-9, "median != {val}");
            prop_assert!((s.p90    - val).abs() < 1e-9, "p90    != {val}");
            prop_assert!((s.p99    - val).abs() < 1e-9, "p99    != {val}");
            prop_assert!((s.max    - val).abs() < 1e-9, "max    != {val}");
        }
    }
}
