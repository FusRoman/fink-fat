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
    engine_config::solver_config::solver_policy::{SolverChoice, SolverPolicy, SolverRoutingMode},
    graph::AlertLinkageDAG,
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
    pub fn run_plan<'edge_lf, 'seed_lf>(
        &self,
        comps: &'edge_lf ConnectedComponents<'edge_lf, 'seed_lf>,
        graph: &'edge_lf AlertLinkageDAG,
        _seed_store: &'seed_lf SeedStore,
        plan: &SolvePlan,
    ) -> Vec<SolverOutput>
    where
        'edge_lf: 'seed_lf,
    {
        // Solver instances used by this manager.
        // These can later become fields or be constructed lazily if needed.
        let bounded_beam = BoundedBeamSolver::default();

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

            outputs.push(out);
        }

        outputs
    }
}
