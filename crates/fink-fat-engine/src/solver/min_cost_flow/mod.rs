//! Min-cost-flow path-cover solver for medium-sized connected components.
//!
//! # Motivation
//! In the inter-night graph, each node represents a seed (pair/triplet/tracklet)
//! and directed edges represent plausible temporal continuations.
//! For a given connected component, we want to extract a set of **clean,
//! non-branching trajectory hypotheses** suitable for downstream orbit
//! determination (IOD).
//!
//! A very practical representation of such hypotheses is a **set of disjoint
//! directed chains** (a *path cover*): each node belongs to at most one chain,
//! and within a chain the direction follows time.
//!
//! # Problem statement
//! Given the directed subgraph induced by a connected component, choose a subset
//! of **ACTIVE** edges such that each node has:
//! - at most one selected predecessor (**1-in**),
//! - at most one selected successor   (**1-out**),
//! while minimizing a global objective.
//!
//! # Objective
//! The solver minimizes:
//! - the sum of selected edge costs, plus
//! - a penalty (`break_penalty`) for each missing predecessor and each missing
//!   successor (i.e., explicit costs for starting/ending chains).
//!
//! Intuitively, `break_penalty` controls the “link vs. split” trade-off:
//! - high penalty ⇒ prefer longer tracks, fewer breaks,
//! - low penalty  ⇒ allow splitting into many short tracks.
//!
//! # Reduction to a bipartite assignment (min-cost flow)
//! The constraints (1-in, 1-out) suggest an assignment structure.
//! We build a bipartite network:
//! - Left side `L(u)` models the choice of a **successor** for node `u`,
//! - Right side `R(v)` models the choice of a **predecessor** for node `v`,
//! - Each candidate edge `u -> v` becomes an arc `L(u) -> R(v)` with cost
//!   equal to the graph edge cost.
//! - Break options are represented as arcs paying `break_penalty`.
//!
//! Solving a min-cost max-flow with integral capacities yields a minimum-cost
//! matching with optional breaks, which corresponds to a valid path cover.
//!
//! # Multiple global solutions (K-best approximation)
//! A strict global optimum can be too rigid in ambiguous components.
//! Rather than relaxing degree constraints, this solver can generate up to `K`
//! **global** alternatives:
//! 1. Compute the best solution,
//! 2. Generate alternatives by forbidding one arc selected in the best solution
//!    (“one-edge deviation”), and re-solving.
//!
//! The resulting solutions are converted to track hypotheses and merged
//! (deduplicated) before being returned.
//!
//! This strategy is a pragmatic approximation of K-best assignment:
//! - deterministic,
//! - inexpensive for medium components,
//! - friendly to downstream IOD (still produces clean chains).
//!
//! # Design notes
//! - This solver is intended for “medium” components where:
//!   - trivial greedy chaining is insufficient,
//!   - but heavy blob-breaking heuristics are unnecessary.
//! - It assumes the component graph has already been pruned (e.g. Top-K edges per
//!   node) to keep the candidate set manageable.
//!
//! See also
//! --------
//! - `candidate_arcs` for extracting the cheapest `(u, v)` arcs inside a component.
//! - `assignment` for building and solving the bipartite min-cost-flow instance.
//! - `tracks` for reconstructing disjoint chains into `TrackHypothesis` objects.

mod assignment;
mod candidate_arcs;
mod diagnostics;
mod k_best;
mod mcf;
mod mcf_config;
mod solution;
mod tracks;

use std::{cmp::Ordering, time::Instant};

pub use mcf_config::MinCostFlowConfig;

use crate::{
    graph::{graph::InterNightGraph, node_id::NodeId},
    solver::{
        Solver, SolverOutput,
        components::ComponentStats,
        min_cost_flow::{
            candidate_arcs::build_candidate_arcs, diagnostics::build_diagnostics,
            k_best::compute_k_solutions, tracks::solutions_to_tracks,
        },
    },
};

/// Min-cost-flow solver producing disjoint-chain track hypotheses.
///
/// # When to use
/// This solver targets connected components where a global, degree-constrained
/// selection is beneficial:
/// - too large/ambiguous for trivial greedy chaining,
/// - but not so large that blob-breaking or expensive heuristics are required.
///
/// # Output semantics
/// The solver returns a set of candidate [`TrackHypothesis`](crate::trajectory::track_hypothesis::TrackHypothesis)
/// objects representing disjoint directed chains. Depending on configuration,
/// it may also propose a list of edges to deactivate (typically *after* IOD
/// validation).
#[derive(Clone, Debug, Default)]
pub struct MinCostFlowSolver {
    /// Solver configuration knobs (break penalty, K solutions, etc.).
    pub config: MinCostFlowConfig,
}

impl MinCostFlowSolver {
    /// Create a new [`MinCostFlowSolver`] with the provided configuration.
    ///
    /// Parameters
    /// ----------
    /// config : MinCostFlowConfig
    ///     Solver configuration (track length threshold, break penalty, K solutions).
    pub fn new(config: MinCostFlowConfig) -> Self {
        Self { config }
    }
}

impl Solver for MinCostFlowSolver {
    /// Return a short stable name for logs/metrics.
    fn name(&self) -> &'static str {
        "min_cost_flow"
    }

    /// Solve a single connected component and return trajectory hypotheses.
    ///
    /// Pipeline
    /// --------
    /// 1. **Extract candidate arcs** inside the component, restricted to ACTIVE edges,
    ///    keeping only the cheapest edge per ordered node pair `(u, v)`.
    /// 2. **Solve** the min-cost assignment:
    ///    - best solution (always),
    ///    - plus up to `K-1` alternatives via one-edge deviation (optional).
    /// 3. **Reconstruct tracks** as disjoint chains and apply the `min_nodes` filter.
    /// 4. **Deduplicate** track hypotheses across solutions.
    /// 5. Optionally output proposed edge deactivations.
    ///
    /// Parameters
    /// ----------
    /// graph : &InterNightGraph
    ///     Global inter-night graph storage (read-only during solving).
    /// component_nodes : &[NodeId]
    ///     Node ids belonging to the connected component.
    /// stats : ComponentStats
    ///     Precomputed component statistics (used for diagnostics and routing).
    ///
    /// Returns
    /// -------
    /// SolverOutput
    ///     Contains:
    ///     - `tracks`: candidate tracks (sorted by increasing cost),
    ///     - `proposed_deactivations`: optional edge ids suggested for deactivation,
    ///     - `diag`: solver diagnostics (timing and counters).
    ///
    /// Notes
    /// -----
    /// - This solver assumes the input graph has already been pruned (e.g. Top-K
    ///   outgoing edges per node). If not, candidate arc extraction can be large
    ///   and solving may become expensive.
    /// - Costs are treated as additive and “smaller is better”.
    /// - If `propose_deactivations` is enabled, the solver suggests deactivating
    ///   edges used by returned tracks. Many pipelines should only apply that
    ///   after downstream IOD validation.
    fn solve(
        &self,
        graph: &InterNightGraph,
        component_nodes: &[NodeId],
        stats: ComponentStats,
    ) -> SolverOutput {
        let start_wall = Instant::now();

        let mut diagnostics = build_diagnostics(self.name(), stats, component_nodes.len());

        // Defensive early exit.
        if component_nodes.is_empty() {
            diagnostics.time_spent_s = start_wall.elapsed().as_secs_f64();
            return SolverOutput {
                diag: diagnostics,
                ..Default::default()
            };
        }

        // 1) Candidate arcs restricted to this component:
        //    (local_u, local_v) -> (EdgeId, edge_cost)
        let candidate_arcs = build_candidate_arcs(graph, component_nodes);

        // 2) Solve best + alternatives.
        let requested_solutions = self.config.n_solutions.max(1);
        let solutions = compute_k_solutions(
            component_nodes.len(),
            &candidate_arcs,
            self.config.break_penalty,
            requested_solutions,
            self.config.max_alt_attempts,
        );

        // 3) Convert solutions into tracks and deduplicate across solutions.
        let (mut track_hypotheses, mut proposed_deactivations) = solutions_to_tracks(
            graph,
            component_nodes,
            &candidate_arcs,
            &solutions,
            self.config.min_nodes,
            self.config.propose_deactivations,
        );

        // Ensure best-first (nice invariant for downstream).
        track_hypotheses
            .sort_by(|lhs, rhs| lhs.cost.partial_cmp(&rhs.cost).unwrap_or(Ordering::Equal));

        if self.config.propose_deactivations {
            proposed_deactivations.sort();
            proposed_deactivations.dedup();
        } else {
            proposed_deactivations.clear();
        }

        diagnostics.n_selected = track_hypotheses.len() as u32;
        diagnostics.time_spent_s = start_wall.elapsed().as_secs_f64();

        SolverOutput {
            tracks: track_hypotheses,
            proposed_deactivations,
            diag: diagnostics,
        }
    }
}
