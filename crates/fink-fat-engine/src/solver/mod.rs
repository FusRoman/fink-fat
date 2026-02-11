//! Solver subsystem: common interfaces, diagnostics, and outputs.
//!
//! Overview
//! --------
//! This module defines the **shared API** used by all solvers operating on the
//! inter-night runtime graph.
//!
//! A solver consumes a **connected component** (identified by [`ComponentId`])
//! and produces a set of **trajectory hypotheses** ([`TrackHypothesis`]),
//! along with lightweight diagnostics ([`SolverDiagnostics`]) used for monitoring
//! and tuning.
//!
//! Solvers are intended to operate on:
//! - the global runtime graph ([`RuntimeGraph`]) as the backing storage for nodes/edges,
//! - a component-local view provided by [`ConnectedComponents`] (nodes, restricted
//!   adjacency, sources/sinks, degrees, etc.).
//!
//! This module does not implement a specific solving strategy.
//! Concrete implementations live in submodules (e.g. `bounded_beam`, `min_cost_flow`).
//!
//! Key concepts
//! ------------
//!
//! ### Connected component
//! Solvers typically work component-by-component to keep the combinatorial
//! problem tractable.
//!
//! Components are computed on the **undirected view** of the graph (connectivity),
//! but solvers operate on the **directed, time-forward** edges restricted to the
//! component. The component-local directed subgraph is exposed by
//! [`ConnectedComponents`].
//!
//! ### Track hypothesis
//! A [`TrackHypothesis`] represents a candidate multi-night association:
//! - ordered list of seed nodes (in time order),
//! - list of edges linking consecutive nodes,
//! - aggregated cost and metadata (e.g. night span).
//!
//! Tracks are solver outputs and typically feed downstream steps such as
//! trajectory fitting, scoring, or persistence.
//!
//! ### Diagnostics
//! [`SolverDiagnostics`] is intentionally small and generic:
//! - sufficient to monitor performance and tune policies,
//! - without forcing solvers to expose complex internal state.
//!
//! Diagnostics can be used to:
//! - drive solver routing policies,
//! - enforce budgets,
//! - log component-level summary statistics,
//! - track pruning behavior (candidates vs selected).
//!
//! Lifetime model
//! --------------
//! The solver API is designed to avoid copying large graph structures.
//!
//! - [`RuntimeGraph`] stores edges referencing seed nodes.
//! - [`TrackHypothesis`] typically borrows those same edges and nodes.
//!
//! Therefore, [`SolverOutput`] is generic over lifetimes:
//! - `'edge_lf` for edge references,
//! - `'seed_lf` for seed references,
//! - `'alert_lf` for alert references inside seeds.
//!
//! The bound `'edge_lf: 'seed_lf` ensures that borrowed edges outlive the seed
//! references they contain.
//!
//! Submodules
//! ----------
//! - `components` – connected components API and component-local views.
//! - `solver_manager` – routing and orchestration between solver implementations.
//! - `bounded_beam` – bounded beam-search enumeration inside one component.
//! - `min_cost_flow` – (optional) global optimization solver for larger components.

use crate::{
    graph::RuntimeGraph,
    solver::components::{ComponentId, ConnectedComponents},
    trajectory::TrackHypothesis,
};

pub mod bounded_beam;
pub mod components;
pub mod min_cost_flow;
pub mod solver_manager;

/// Lightweight diagnostics emitted by solvers for monitoring and tuning.
///
/// This structure is intended to be:
/// - small,
/// - easy to log/serialize,
/// - generic across very different solver strategies.
///
/// The fields are a mix of:
/// - component identifiers and sizes,
/// - optional timing/budget tracking,
/// - simple counters for candidate generation and selection.
///
/// Notes
/// -----
/// - Not all solvers will interpret `n_candidates` and `n_selected` identically.
///   The recommended convention is:
///   - `n_candidates`: number of intermediate candidates considered (edges, states,
///     paths, assignments, depending on solver),
///   - `n_selected`: number of final hypotheses emitted (or kept after final ranking).
#[derive(Clone, Debug, Default)]
pub struct SolverDiagnostics {
    /// Identifier of the connected component being solved.
    ///
    /// This should match the `ComponentId` passed to [`Solver::solve`].
    pub component_id: u32,

    /// Stable solver name used in logs and metrics.
    ///
    /// The name is intended to be:
    /// - human-readable,
    /// - stable across versions,
    /// - suitable for metric labels.
    ///
    /// Examples: `"bounded_beam"`, `"min_cost_flow"`.
    pub solver_name: &'static str,

    /// Number of nodes in the component.
    ///
    /// This value is typically used for routing heuristics:
    /// small components can be handled by bounded enumerators, while larger ones
    /// may require optimization methods.
    pub n_nodes: u32,

    /// Number of active edges considered in the component.
    ///
    /// The definition of "active" is graph-dependent (typically an edge flag used
    /// to ignore deactivated links during solving).
    pub m_active_edges: u32,

    /// Optional wall-clock estimate or budget target (seconds).
    ///
    /// This can be used by orchestration code to:
    /// - record predicted runtime,
    /// - compare expected vs actual runtime,
    /// - implement budget-aware routing policies.
    ///
    /// Notes
    /// -----
    /// - This field may remain `0.0` if not used.
    pub time_est_s: f64,

    /// Optional wall-clock time spent (seconds).
    ///
    /// Notes
    /// -----
    /// - This can be filled by the solver itself or by the caller.
    /// - The API keeps it as a scalar to avoid imposing a timing framework.
    pub time_spent_s: f64,

    /// Solver-specific candidate counter (generic).
    ///
    /// Recommended meaning
    /// -------------------
    /// Count the number of intermediate objects examined before final selection.
    ///
    /// Examples
    /// --------
    /// - Bounded beam search: number of edge candidates seen during adjacency
    ///   preparation, number of state expansions, or similar.
    /// - Flow solver: number of arcs processed, iterations performed, etc.
    /// - Greedy solver: number of candidate edges evaluated.
    ///
    /// Notes
    /// -----
    /// This is intentionally generic and should be interpreted in the context
    /// of `solver_name`.
    pub n_candidates: u32,

    /// Solver-specific selected counter (generic).
    ///
    /// Recommended meaning
    /// -------------------
    /// Number of final hypotheses emitted (or kept after final ranking).
    ///
    /// Notes
    /// -----
    /// - For solvers returning tracks, this is typically `tracks.len()`.
    pub n_selected: u32,
}

/// Output of a solver pass over a connected component.
///
/// This is the common return type of the [`Solver`] trait.
///
/// It includes:
/// - `tracks`: candidate trajectory hypotheses,
/// - `diag`: diagnostics for monitoring and tuning.
///
/// Typical conventions
/// -------------------
/// - `tracks` are often sorted from best to worst (e.g. increasing cost),
///   but this module does not enforce ordering.
/// - Callers may apply additional global ranking, deduplication, or truncation.
///
/// Lifetimes
/// ---------
/// Tracks typically borrow edges and nodes from the runtime graph,
/// so the output is parameterized by:
/// - `'edge_lf`: lifetime of borrowed edges,
/// - `'seed_lf`: lifetime of borrowed seeds,
/// - `'alert_lf`: lifetime of borrowed alerts inside seeds.
#[derive(Clone, Debug, Default)]
pub struct SolverOutput<'edge_lf, 'seed_lf, 'alert_lf> {
    /// Candidate tracks returned by the solver.
    ///
    /// In most solvers, tracks are sorted from best to worst, but callers should
    /// not rely on that unless explicitly documented by the solver implementation.
    pub tracks: Vec<TrackHypothesis<'edge_lf, 'seed_lf, 'alert_lf>>,

    /// Diagnostics for monitoring and tuning.
    pub diag: SolverDiagnostics,
}

/// A solver that extracts trajectory hypotheses from a connected component.
///
/// A solver implementation is responsible for:
/// - consuming the component-local view (via [`ConnectedComponents`]),
/// - producing one or more plausible [`TrackHypothesis`] objects,
/// - returning lightweight diagnostics describing its work.
///
/// The solver operates on a connected component identified by `component_id`.
/// The component-local directed subgraph (nodes, adjacency, sources/sinks, degrees,
/// etc.) is obtained through [`ConnectedComponents`].
///
/// Design goals
/// ------------
/// - **No copying** of large graph structures: solvers borrow edges/nodes.
/// - **Component-local** operation: solvers should avoid global rescans.
/// - **Pluggable**: multiple solver strategies can implement this trait.
///
/// Notes on thread-safety
/// ----------------------
/// The trait does not impose `Send`/`Sync`. Threading concerns are handled at
/// higher levels (e.g. solver manager) depending on the broader architecture.
pub trait Solver<'edge_lf, 'seed_lf, 'alert_lf> {
    /// A short stable name for logs and metrics.
    ///
    /// The name should be:
    /// - stable across versions,
    /// - suitable for metric labels,
    /// - descriptive of the solver strategy.
    fn name(&self) -> &'static str;

    /// Solve a single connected component and return trajectory hypotheses.
    ///
    /// The solver is given read-only access to:
    /// - the global runtime graph (`graph`), which owns the edges and provides
    ///   the backing storage for borrowed references,
    /// - the connected components object (`cc`), which provides a component-local
    ///   directed subgraph view,
    /// - a `component_id` selecting the component to solve.
    ///
    /// Expected behavior
    /// -----------------
    /// - Use `cc` to retrieve the component-local subgraph view.
    /// - Enumerate / optimize / select plausible track hypotheses.
    /// - Return a [`SolverOutput`] containing:
    ///   - a list of tracks,
    ///   - diagnostics ([`SolverDiagnostics`]) describing the work performed.
    ///
    /// Arguments
    /// ---------
    /// * `graph` – Global runtime graph (read-only for solving).
    /// * `cc` – Connected components object providing component-local views.
    /// * `component_id` – Identifier of the connected component to solve.
    ///
    /// Return
    /// ------
    /// * `SolverOutput` – Candidate tracks and diagnostics.
    ///
    /// Notes
    /// -----
    /// - The bound `'edge_lf: 'seed_lf` ensures edges outlive the seed references they contain.
    /// - Solvers should treat `graph` as immutable unless explicitly designed to
    ///   mutate edge flags elsewhere in the pipeline.
    fn solve(
        &self,
        graph: &'edge_lf RuntimeGraph<'seed_lf, 'alert_lf>,
        cc: &'edge_lf ConnectedComponents<'edge_lf, 'seed_lf, 'alert_lf>,
        component_id: ComponentId,
    ) -> SolverOutput<'edge_lf, 'seed_lf, 'alert_lf>
    where
        'edge_lf: 'seed_lf;
}
