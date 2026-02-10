use crate::{graph::RuntimeGraph, seeding::seed_node::SeedNode, trajectory::TrackHypothesis};

pub mod components;
pub mod min_cost_flow;
pub mod solver_manager;
pub mod trivial_solver;

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
pub struct SolverOutput<'edge_lf, 'seed_lf, 'alert_lf> {
    /// Candidate tracks (usually sorted by increasing cost).
    pub tracks: Vec<TrackHypothesis<'edge_lf, 'seed_lf, 'alert_lf>>,

    /// Diagnostics for monitoring and tuning.
    pub diag: SolverDiagnostics,
}

/// A solver that extracts trajectory hypotheses from a connected component.
pub trait Solver<'edge_lf, 'seed_lf, 'alert_lf> {
    /// A short stable name for logs/metrics.
    fn name(&self) -> &'static str;

    /// Solve a single connected component and return trajectory hypotheses.
    ///
    /// Parameters
    /// ----------
    /// graph : &RuntimeGraph
    ///     Global inter-night graph (read-only for solving).
    /// component_nodes : &[&'seed_lf SeedNode<'alert_lf>]
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
        graph: &'edge_lf RuntimeGraph<'seed_lf, 'alert_lf>,
        component_nodes: &[&'seed_lf SeedNode<'alert_lf>],
    ) -> SolverOutput<'edge_lf, 'seed_lf, 'alert_lf>;
}
