use crate::{
    graph::InterNightGraph,
    pipeline::seed_store::SeedStore,
    solver::{
        components::ConnectedComponents,
        trivial_solver::{TrivialSolver, TrivialSolverOutput},
    },
};

/// Solver choice for a component.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SolverChoice {
    /// Skip or solve with a tiny direct method.
    Trivial,
    /// Use min-cost flow (global optimum within the component).
    MinCostFlow,
    /// Use blob-breaker: window + partition + local peeling.
    BlobBreaker,
}

/// A single work item produced by the planner.
#[derive(Clone, Debug)]
pub struct WorkItem {
    pub component_id: u32,
    pub choice: SolverChoice,
}

/// A plan for one solver pass.
#[derive(Clone, Debug, Default)]
pub struct SolvePlan {
    pub items: Vec<WorkItem>,
}

/// Policy parameters for routing.
///
/// Intentionally simple for now.
#[derive(Copy, Clone, Debug)]
pub struct SolverPolicy {
    pub trivial_max_nodes: u32,
    pub trivial_max_active_edges: u32,

    /// If estimated MCF cost exceeds this, route to blob-breaker.
    pub mcf_budget_s: f64,

    /// Seconds per (edge * log2(nodes+1)) for the MCF solver.
    pub k_mcf_s_per_edge_logn: f64,

    /// If a component spans more nights than this, route to blob-breaker.
    pub max_night_span_for_mcf: u32,
}

impl Default for SolverPolicy {
    fn default() -> Self {
        Self {
            trivial_max_nodes: 8,
            trivial_max_active_edges: 16,
            mcf_budget_s: 0.05,          // 50 ms
            k_mcf_s_per_edge_logn: 1e-8, // placeholder (calibrate later)
            max_night_span_for_mcf: 4,
        }
    }
}

impl SolverPolicy {
    #[inline]
    pub fn estimate_mcf_time_s(&self, n_nodes: u32, m_edges: u32) -> f64 {
        let logn = ((n_nodes as f64) + 1.0).log2();
        self.k_mcf_s_per_edge_logn * (m_edges as f64) * logn
    }
}

/// Manager object holding the routing policy.
#[derive(Clone, Debug)]
pub struct SolverManager {
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
    /// Build a solve plan from per-component stats.
    pub fn make_plan(&self, comps: &ConnectedComponents) -> SolvePlan {
        let mut items = Vec::with_capacity(comps.n_components as usize);
        for cid in 0..comps.n_components {
            let choice = comps.classify(cid as u32, &self.policy);
            items.push(WorkItem {
                component_id: cid as u32,
                choice,
            });
        }
        SolvePlan { items }
    }

    /// Execute the plan (Trivial solver implemented, others TODO).
    ///
    /// Returns one `TrivialSolverOutput` per planned item (in plan order).
    ///
    /// Note: When you implement MCF / blob-breaker, you can unify the output type
    /// into your existing `SolverOutput`.
    pub fn run_plan<'seed_lf, 'alert_lf>(
        &self,
        comps: &'seed_lf ConnectedComponents<'seed_lf, 'alert_lf>,
        graph: &InterNightGraph<'seed_lf, 'alert_lf>,
        _seed_store: &'seed_lf SeedStore<'alert_lf>,
        plan: &SolvePlan,
    ) -> Vec<TrivialSolverOutput<'seed_lf, 'alert_lf>> {
        // Trivial solver instance
        let trivial = TrivialSolver::default();

        let mut outputs: Vec<TrivialSolverOutput<'seed_lf, 'alert_lf>> =
            Vec::with_capacity(plan.items.len());

        for item in &plan.items {
            // For "trivial", we usually want active edges only.
            let out = match item.choice {
                SolverChoice::Trivial => {
                    trivial.solve_component(
                        &graph.edges,
                        comps.component_nodes(item.component_id),
                        /*active_only=*/ true,
                    )
                }
                SolverChoice::MinCostFlow => {
                    todo!("MinCostFlowSolver not implemented yet (new API)")
                }
                SolverChoice::BlobBreaker => {
                    todo!("BlobBreaker not implemented yet (new API)")
                }
            };

            outputs.push(out);
        }

        outputs
    }
}
