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
//! 1) Build [`ConnectedComponents`] from the global [`RuntimeGraph`] and [`SeedStore`].
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
    graph::RuntimeGraph,
    pipeline::seed_store::SeedStore,
    solver::{
        Solver, SolverOutput, bounded_beam::BoundedBeamSolver, components::ConnectedComponents,
    },
};

/// Solver strategy selected to process a connected component.
///
/// Variants represent solver families. Routing decisions are produced by
/// [`SolverPolicy`] (heuristics) or by forced routing ([`SolverRoutingMode::Force`]).
///
/// Notes
/// -----
/// - The meaning of each variant is algorithmic (a family of approaches), not
///   necessarily tied to one specific implementation.
/// - [`SolverManager`] is responsible for instantiating the concrete solver
///   corresponding to each choice.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SolverChoice {
    /// Use a bounded beam-search solver suitable for small components.
    ///
    /// Typical properties of components routed here:
    /// - small number of nodes,
    /// - moderate branching (kept under control by beam width and local pruning),
    /// - cheap to enumerate a bounded set of candidate tracks.
    BoundedBeam,

    /// Use a min-cost flow solver (global optimum within the component).
    ///
    /// Intended for medium-sized components where a global optimization is feasible
    /// under a time budget.
    MinCostFlow,

    /// Use a blob-breaker strategy: windowing + partitioning + local peeling.
    ///
    /// Intended for large or difficult components that exceed the budget or
    /// violate assumptions (e.g. very large night span).
    BlobBreaker,
}

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

/// Routing mode for solver selection.
///
/// This controls how [`SolverPolicy`] assigns [`SolverChoice`] to components.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SolverRoutingMode {
    /// Use component summary statistics and heuristics to pick a solver.
    Heuristics,

    /// Force the same solver for every component.
    ///
    /// This is useful for:
    /// - debugging one solver in isolation,
    /// - benchmarking,
    /// - ablation studies.
    Force(SolverChoice),
}

/// Policy parameters for routing.
///
/// The policy is intentionally simple and aims to keep routing decisions:
/// - transparent,
/// - easy to tune,
/// - based on stable component-level summary statistics.
///
/// Routing decision inputs
/// -----------------------
/// The policy relies on:
/// - `n_nodes`: number of nodes in the component,
/// - `m_active_edges`: number of active intra-component edges,
/// - `night_span`: `max_night - min_night`.
///
/// The policy additionally contains a time estimate model for min-cost flow.
///
/// Notes
/// -----
/// - Constants should be tuned with empirical measurements.
/// - The policy does not embed solver-internal parameters; it only decides which
///   solver family should be used.
#[derive(Copy, Clone, Debug)]
pub struct SolverPolicy {
    /// Global routing mode: heuristics vs forced routing.
    pub routing: SolverRoutingMode,

    /// If `n_nodes <= trivial_max_nodes` (and other guardrails permit), route to
    /// a bounded beam solver.
    pub trivial_max_nodes: u32,

    /// If `m_active_edges <= trivial_max_active_edges`, bounded beam solving is
    /// usually cheap.
    ///
    /// Notes
    /// -----
    /// This threshold is available for heuristics even if classification also
    /// uses other signals such as night span or a time budget model.
    pub trivial_max_active_edges: u32,

    /// Time budget (seconds) for the min-cost flow solver.
    ///
    /// If the estimated MCF time exceeds this budget, routing falls back to
    /// blob-breaker.
    pub mcf_budget_s: f64,

    /// Time model coefficient for min-cost flow:
    /// seconds per `(edge * log2(nodes+1))`.
    ///
    /// This coefficient is intended to be calibrated on representative workloads.
    pub k_mcf_s_per_edge_logn: f64,

    /// If a component spans more nights than this, route to blob-breaker.
    ///
    /// Motivation
    /// ----------
    /// Large night spans often imply many alternative connections and complex
    /// combinatorics. Even if `n_nodes` is moderate, the structure may exceed
    /// optimization budgets.
    pub max_night_span_for_mcf: u32,
}

impl Default for SolverPolicy {
    /// Default heuristic routing policy.
    ///
    /// The defaults are conservative:
    /// - very small components route to bounded beam,
    /// - min-cost flow is allowed only under a strict time budget,
    /// - large night spans route to blob-breaker.
    fn default() -> Self {
        Self {
            routing: SolverRoutingMode::Heuristics,

            trivial_max_nodes: 8,
            trivial_max_active_edges: 16,

            mcf_budget_s: 0.05,          // 50 ms
            k_mcf_s_per_edge_logn: 1e-8, // placeholder (calibrate later)
            max_night_span_for_mcf: 4,
        }
    }
}

impl SolverPolicy {
    /// Estimate the expected runtime of the min-cost flow solver (seconds).
    ///
    /// The estimator uses a simple scaling model:
    ///
    /// ```text
    /// t_est = k * m_edges * log2(n_nodes + 1)
    /// ```
    ///
    /// where:
    /// - `m_edges` is the number of active edges inside the component,
    /// - `n_nodes` is the number of nodes,
    /// - `k` is `k_mcf_s_per_edge_logn`.
    ///
    /// Arguments
    /// ---------
    /// * `n_nodes` – Number of nodes in the component.
    /// * `m_edges` – Number of active intra-component edges.
    ///
    /// Return
    /// ------
    /// Estimated runtime in seconds.
    ///
    /// Notes
    /// -----
    /// - This is a coarse model intended for routing decisions, not profiling.
    /// - `k_mcf_s_per_edge_logn` should be calibrated empirically.
    #[inline]
    pub fn estimate_mcf_time_s(&self, n_nodes: u32, m_edges: u32) -> f64 {
        let logn = ((n_nodes as f64) + 1.0).log2();
        self.k_mcf_s_per_edge_logn * (m_edges as f64) * logn
    }

    /// Build a policy that forces a single solver family for all components.
    ///
    /// Arguments
    /// ---------
    /// * `choice` – Solver family to use for every component.
    ///
    /// Return
    /// ------
    /// A policy configured in `Force(choice)` mode.
    #[inline]
    pub fn forced(choice: SolverChoice) -> Self {
        Self {
            routing: SolverRoutingMode::Force(choice),
            ..Self::default()
        }
    }

    /// Build a policy in explicit heuristics mode.
    ///
    /// Return
    /// ------
    /// A policy configured in `Heuristics` mode.
    #[inline]
    pub fn heuristics() -> Self {
        Self {
            routing: SolverRoutingMode::Heuristics,
            ..Self::default()
        }
    }
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
#[derive(Clone, Debug)]
pub struct SolverManager {
    /// Routing policy controlling solver selection.
    pub policy: SolverPolicy,
}

impl Default for SolverManager {
    fn default() -> Self {
        Self {
            policy: SolverPolicy::default(),
        }
    }
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
        let mut items = Vec::with_capacity(comps.n_components as usize);

        for cid in 0..comps.n_components {
            let cid_u32 = cid as u32;

            let choice = match self.policy.routing {
                SolverRoutingMode::Heuristics => comps.classify(cid_u32, &self.policy),
                SolverRoutingMode::Force(c) => c,
            };

            items.push(WorkItem {
                component_id: cid_u32,
                choice,
            });
        }

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
    pub fn run_plan<'edge_lf, 'seed_lf, 'alert_lf>(
        &self,
        comps: &'edge_lf ConnectedComponents<'edge_lf, 'seed_lf, 'alert_lf>,
        graph: &'edge_lf RuntimeGraph<'seed_lf, 'alert_lf>,
        _seed_store: &'seed_lf SeedStore<'alert_lf>,
        plan: &SolvePlan,
    ) -> Vec<SolverOutput<'edge_lf, 'seed_lf, 'alert_lf>>
    where
        'edge_lf: 'seed_lf,
    {
        // Solver instances used by this manager.
        // These can later become fields or be constructed lazily if needed.
        let bounded_beam = BoundedBeamSolver::default();

        let mut outputs: Vec<SolverOutput<'edge_lf, 'seed_lf, 'alert_lf>> =
            Vec::with_capacity(plan.items.len());

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

            outputs.push(out);
        }

        outputs
    }
}
