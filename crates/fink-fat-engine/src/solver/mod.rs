pub mod components;
pub mod solver_manager;
pub mod solver_trivial;

use crate::graph::edge_id::EdgeId;
use crate::graph::graph::InterNightGraph;
use crate::solver::components::ComponentStats;
use crate::trajectory::track_hypothesis::TrackHypothesis;

#[derive(Clone, Debug, Default)]
pub struct SolverDiagnostics {
    pub component_id: u32,
    pub solver_name: &'static str,

    pub n_nodes: u32,
    pub m_active_edges: u32,

    /// Wall-clock estimate / budget usage tracking (optional).
    pub time_est_s: f64,
    pub time_spent_s: f64,

    /// Solver-specific counters (keep it simple).
    pub n_candidates: u32,
    pub n_selected: u32,
}

/// Output of a solver pass over a connected component.
#[derive(Clone, Debug, Default)]
pub struct SolverOutput {
    /// Candidate tracks (usually sorted by increasing cost).
    pub tracks: Vec<TrackHypothesis>,

    /// Edges that the solver suggests deactivating immediately
    /// (optional: you might only deactivate after IOD validation).
    pub proposed_deactivations: Vec<EdgeId>,

    /// Diagnostics for monitoring and tuning.
    pub diag: SolverDiagnostics,
}

/// A solver that extracts trajectory hypotheses from a connected component.
pub trait Solver {
    /// A short stable name for logs/metrics.
    fn name(&self) -> &'static str;

    /// Solve a single connected component and return trajectory hypotheses.
    ///
    /// Parameters
    /// ----------
    /// graph : &InterNightGraph
    ///     Global inter-night graph (read-only for solving).
    /// component_nodes : &[NodeId]
    ///     Node ids belonging to the connected component.
    /// stats : ComponentStats
    ///     Precomputed stats for routing and solver heuristics.
    ///
    /// Returns
    /// -------
    /// SolverOutput
    ///     Candidate tracks and optional deactivation suggestions.
    fn solve(
        &self,
        graph: &InterNightGraph,
        component_nodes: &[crate::graph::node_id::NodeId],
        stats: ComponentStats,
    ) -> SolverOutput;
}
