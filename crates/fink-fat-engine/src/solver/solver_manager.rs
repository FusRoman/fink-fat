use crate::{
    graph::InterNightGraph,
    pipeline::seed_store::SeedStore,
    solver::{
        components::{
            ConnectedComponents, component_stats::ComponentStats, seed_index::SeedGlobalIndex,
        },
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
    pub fn make_plan(&self, stats: &[ComponentStats]) -> SolvePlan {
        let mut items = Vec::with_capacity(stats.len());
        for (cid, st) in stats.iter().enumerate() {
            let choice = self.classify(*st);
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
        _graph: &InterNightGraph<'seed_lf, 'alert_lf>,
        _seed_store: &'seed_lf SeedStore<'alert_lf>,
        cc: &ConnectedComponentsRuntime<'seed_lf, 'alert_lf>,
        plan: &SolvePlan,
    ) -> Vec<TrivialSolverOutput<'seed_lf, 'alert_lf>> {
        // Trivial solver instance
        let trivial = TrivialSolver::default();

        let mut outputs: Vec<TrivialSolverOutput<'seed_lf, 'alert_lf>> =
            Vec::with_capacity(plan.items.len());

        for item in &plan.items {
            let cid = item.component_id as usize;

            let nodes = &cc.components[cid];
            // For "trivial", we usually want active edges only.
            let out = match item.choice {
                SolverChoice::Trivial => {
                    trivial.solve_component(cc.edges, nodes, /*active_only=*/ true)
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

    /// Classify a component into a solver choice using the current policy.
    #[inline]
    pub fn classify(&self, st: ComponentStats) -> SolverChoice {
        // Trivial fast path.
        if st.n_nodes <= self.policy.trivial_max_nodes
            || st.m_active_edges <= self.policy.trivial_max_active_edges
        {
            return SolverChoice::Trivial;
        }

        // Night span guardrail.
        if st.night_span() > self.policy.max_night_span_for_mcf {
            return SolverChoice::BlobBreaker;
        }

        // Budget-aware MCF vs BlobBreaker.
        let t_est = estimate_mcf_time_s(self.policy, st.n_nodes, st.m_active_edges);
        if t_est <= self.policy.mcf_budget_s {
            SolverChoice::MinCostFlow
        } else {
            SolverChoice::BlobBreaker
        }
    }
}

#[inline]
fn estimate_mcf_time_s(policy: SolverPolicy, n_nodes: u32, m_edges: u32) -> f64 {
    let logn = ((n_nodes as f64) + 1.0).log2();
    policy.k_mcf_s_per_edge_logn * (m_edges as f64) * logn
}

//
// -----------------------------------------------------------------------------
// Runtime connected components view used by SolverManager
// -----------------------------------------------------------------------------
//

use crate::{graph::edge::Edge, seeding::seed_node::SeedNode};

/// Connected components materialized as `Vec<Vec<&SeedNode>>` for solver consumption.
///
/// This is built once per solve pass from:
/// - `SeedStore` (for nodes)
/// - `SeedGlobalIndex` (for global dense indexing)
/// - `comp_of_node` (DSU result)
/// - global `edges` slice (for stats and trivial solver)
pub struct ConnectedComponentsRuntime<'seed_lf, 'alert_lf> {
    /// Global edge slice (borrowed)
    pub edges: &'seed_lf [Edge<'seed_lf, 'alert_lf>],
    /// Component node lists (borrowed seed refs)
    pub components: Vec<Vec<&'seed_lf SeedNode<'alert_lf>>>,
    /// Per-component stats for routing
    pub stats: Vec<ComponentStats>,
}

impl<'seed_lf, 'alert_lf> ConnectedComponentsRuntime<'seed_lf, 'alert_lf> {
    /// Materialize components as seed references + compute stats.
    ///
    /// Inputs:
    /// - `seed_store` provides all nodes
    /// - `index` maps SeedKey -> global index
    /// - `components` is the CC result (comp_of_node + n_components)
    /// - `edges` is the global edge slice
    pub fn build(
        seed_store: &'seed_lf SeedStore<'alert_lf>,
        index: &SeedGlobalIndex,
        components: ConnectedComponents,
        edges: &'seed_lf [Edge<'seed_lf, 'alert_lf>],
    ) -> Self {
        // Single pass over SeedStore: build node lists + min/max night bounds.
        let (components_ref, night_bounds) =
            components.materialize_components_with_bounds(seed_store, index);

        // Stats: nodes + bounds
        let mut stats = vec![ComponentStats::default(); components.n_components as usize];
        for cid in 0..(components.n_components as usize) {
            stats[cid].n_nodes = components_ref[cid].len() as u32;
            stats[cid].night_bounds = night_bounds[cid];
        }

        // Count internal active directed edges per component.
        for e in edges {
            if !e.core.active {
                continue;
            }
            let cu = components.compid_of_node(e.from.core.key);
            let cv = components.compid_of_node(e.to.core.key);
            if cu == cv {
                stats[cu as usize].m_active_edges += 1;
            }
        }

        Self {
            edges,
            components: components_ref,
            stats,
        }
    }
}
