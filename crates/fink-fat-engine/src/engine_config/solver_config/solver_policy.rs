//! Solver routing policy and family selection.
//!
//! Overview
//! --------
//! This module defines the **high-level routing layer** of the solver subsystem.
//!
//! Its responsibility is to decide *which solver family* should be applied to
//! each connected component of the inter-night graph.
//!
//! The routing decision is based on:
//! - component-level summary statistics (size, density, night span),
//! - a simple time-estimate model for expensive solvers,
//! - configurable thresholds exposed through [`SolverPolicy`].
//!
//! This module does **not** implement any solver itself.
//! It only encodes:
//! - solver families ([`SolverChoice`]),
//! - routing mode ([`SolverRoutingMode`]),
//! - and the routing parameters ([`SolverPolicy`]).
//!
//! Architectural role
//! -------------------
//! In the full solving pipeline:
//!
//! 1. The global inter-night graph is partitioned into connected components.
//! 2. Each component is summarized (node count, edge count, night span).
//! 3. [`SolverPolicy`] classifies the component into a [`SolverChoice`].
//! 4. A higher-level orchestrator instantiates and runs the corresponding solver.
//!
//! This separation ensures:
//! - solver implementations remain independent from routing logic,
//! - routing remains transparent and tunable,
//! - experimentation (forcing one solver family) is straightforward.
//!
//! Solver families
//! ---------------
//! The available solver families are:
//!
//! - [`SolverChoice::BoundedBeam`]
//!   - bounded beam-search enumeration,
//!   - suitable for small components,
//!   - predictable and inexpensive.
//!
//! - [`SolverChoice::MinCostFlow`]
//!   - global optimization within a component,
//!   - used when the estimated runtime fits within a time budget.
//!
//! - [`SolverChoice::BlobBreaker`]
//!   - fallback strategy for large or structurally complex components,
//!   - used when global optimization is too expensive.
//!
//! Routing modes
//! -------------
//! [`SolverRoutingMode`] controls how decisions are made:
//!
//! - `Heuristics`
//!   - automatic classification using thresholds and runtime estimates.
//!
//! - `Force(choice)`
//!   - override routing and use a single solver family for all components.
//!   - useful for debugging, benchmarking, and ablation studies.
//!
//! Time estimate model
//! -------------------
//! For the min-cost flow solver, a coarse runtime estimate is used:
//!
//! ```text
//! t_est = k_mcf_s_per_edge_logn * m_edges * log2(n_nodes + 1)
//! ```
//!
//! where:
//! - `m_edges` is the number of active intra-component edges,
//! - `n_nodes` is the number of nodes,
//! - `k_mcf_s_per_edge_logn` is a calibration coefficient.
//!
//! This model is intentionally simple and meant for routing decisions,
//! not precise profiling.
//!
//! Design principles
//! -----------------
//! - Keep routing logic simple and explainable.
//! - Avoid embedding solver-internal configuration here.
//! - Base decisions only on stable component-level statistics.
//! - Make policies serializable for reproducibility and experimentation.
//!
//! This module is part of the solver orchestration layer and is typically
//! used together with a solver manager responsible for execution.
use serde::{Deserialize, Serialize};

/// Solver family selected to process a connected component.
///
/// Overview
/// --------
/// This enum represents the **algorithmic family** used to solve one connected
/// component of the inter-night graph.
///
/// A routing decision is produced by [`SolverPolicy`] (heuristics mode)
/// or by forced routing ([`SolverRoutingMode::Force`]).
///
/// Design intent
/// -------------
/// - Variants represent *families of approaches*, not specific implementations.
/// - The mapping from [`SolverChoice`] to a concrete solver object is handled
///   by a higher-level orchestration layer (e.g. `SolverManager`).
/// - Routing decisions rely only on component-level statistics, not on
///   solver-internal state.
///
/// Typical usage
/// -------------
/// - Small components → [`SolverChoice::BoundedBeam`]
/// - Medium components within budget → [`SolverChoice::MinCostFlow`]
/// - Large or structurally difficult components → [`SolverChoice::BlobBreaker`]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolverChoice {
    /// Use a bounded beam-search solver.
    ///
    /// Intended for relatively small components where enumerating a bounded
    /// set of candidate paths is computationally cheap.
    ///
    /// Typical properties:
    /// - small number of nodes,
    /// - manageable branching factor,
    /// - no need for global optimization.
    BoundedBeam,

    /// Use a min-cost flow solver.
    ///
    /// Intended for medium-sized components where a global optimization is
    /// feasible within a time budget.
    ///
    /// Properties:
    /// - computes a globally optimal assignment under a flow model,
    /// - runtime grows with both nodes and edges,
    /// - controlled by a time estimate model.
    MinCostFlow,

    /// Use a blob-breaker strategy.
    ///
    /// Intended for large or structurally complex components that:
    /// - exceed the min-cost flow budget,
    /// - span many nights,
    /// - or exhibit combinatorial explosion.
    ///
    /// This strategy typically relies on partitioning, windowing,
    /// and local resolution techniques.
    BlobBreaker,
}

/// Routing mode controlling how solver families are assigned to components.
///
/// Overview
/// --------
/// [`SolverRoutingMode`] determines whether solver selection is:
///
/// - automatic (heuristic-based), or
/// - globally forced to a single family.
///
/// This mechanism enables both production routing and controlled experimentation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolverRoutingMode {
    /// Select solver families using component-level heuristics.
    ///
    /// This mode uses:
    /// - component size,
    /// - number of active edges,
    /// - night span,
    /// - a time estimate model,
    ///
    /// to determine the most appropriate solver family.
    Heuristics,

    /// Force the same solver family for all components.
    ///
    /// Useful for:
    /// - benchmarking,
    /// - debugging,
    /// - ablation studies,
    /// - deterministic evaluation runs.
    Force(SolverChoice),
}

/// Policy parameters controlling solver routing decisions.
///
/// Overview
/// --------
/// [`SolverPolicy`] contains the thresholds and model parameters used to
/// classify connected components into solver families.
///
/// The policy is intentionally:
/// - transparent,
/// - lightweight,
/// - based only on stable component-level summary statistics.
///
/// Classification signals
/// ----------------------
/// Routing decisions rely on:
///
/// - `n_nodes` — number of nodes in the component,
/// - `m_active_edges` — number of active intra-component edges,
/// - `night_span` — `max_night - min_night`,
/// - a runtime estimate model for min-cost flow.
///
/// Time estimate model
/// -------------------
/// The min-cost flow solver runtime is estimated as:
///
/// ```text
/// t_est = k_mcf_s_per_edge_logn * m_active_edges * log2(n_nodes + 1)
/// ```
///
/// where `k_mcf_s_per_edge_logn` must be calibrated empirically.
///
/// Design constraints
/// ------------------
/// - This structure does *not* contain solver-internal configuration.
/// - It only decides which solver family should be used.
/// - Thresholds should be tuned using representative workloads.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct SolverPolicy {
    /// Global routing mode.
    pub routing: SolverRoutingMode,

    /// Maximum number of nodes for routing to [`SolverChoice::BoundedBeam`].
    ///
    /// Small components are efficiently handled by bounded enumeration.
    pub trivial_max_nodes: u32,

    /// Maximum number of active edges typically considered cheap for
    /// bounded beam solving.
    ///
    /// This acts as a density guardrail.
    pub trivial_max_active_edges: u32,

    /// Maximum allowed estimated runtime (seconds) for min-cost flow.
    ///
    /// If the estimate exceeds this budget, routing falls back to
    /// [`SolverChoice::BlobBreaker`].
    pub mcf_budget_s: f64,

    /// Time model coefficient for min-cost flow.
    ///
    /// Units: seconds per `(edge * log2(nodes+1))`.
    ///
    /// This value should be calibrated empirically.
    pub k_mcf_s_per_edge_logn: f64,

    /// Maximum allowed night span for min-cost flow.
    ///
    /// Components spanning more nights than this threshold are routed to
    /// [`SolverChoice::BlobBreaker`] to avoid large combinatorial structures.
    pub max_night_span_for_mcf: u32,
}

impl Default for SolverPolicy {
    /// Conservative default heuristic routing policy.
    ///
    /// Defaults are chosen to:
    /// - route very small components to bounded beam,
    /// - allow min-cost flow only under a strict time budget,
    /// - route wide night-span components to blob-breaker.
    fn default() -> Self {
        Self {
            routing: SolverRoutingMode::Heuristics,
            trivial_max_nodes: 8,
            trivial_max_active_edges: 16,
            mcf_budget_s: 0.05,
            k_mcf_s_per_edge_logn: 1e-8,
            max_night_span_for_mcf: 4,
        }
    }
}

impl SolverPolicy {
    /// Estimate the expected runtime of the min-cost flow solver.
    ///
    /// Model
    /// -----
    /// ```text
    /// t_est = k * m_edges * log2(n_nodes + 1)
    /// ```
    ///
    /// Parameters
    /// ----------
    /// * `n_nodes` – Number of nodes in the component.
    /// * `m_edges` – Number of active intra-component edges.
    ///
    /// Returns
    /// -------
    /// Estimated runtime in seconds.
    ///
    /// Notes
    /// -----
    /// - This model is coarse and intended only for routing decisions.
    /// - Calibration should be performed on representative datasets.
    #[inline]
    pub fn estimate_mcf_time_s(&self, n_nodes: u32, m_edges: u32) -> f64 {
        let logn = ((n_nodes as f64) + 1.0).log2();
        self.k_mcf_s_per_edge_logn * (m_edges as f64) * logn
    }

    /// Create a policy that forces a single solver family.
    ///
    /// This is equivalent to setting `routing = Force(choice)`
    /// while keeping other parameters at default values.
    #[inline]
    pub fn forced(choice: SolverChoice) -> Self {
        Self {
            routing: SolverRoutingMode::Force(choice),
            ..Self::default()
        }
    }

    /// Create a policy explicitly configured for heuristic routing.
    ///
    /// Equivalent to setting `routing = Heuristics`
    /// while keeping other parameters at default values.
    #[inline]
    pub fn heuristics() -> Self {
        Self {
            routing: SolverRoutingMode::Heuristics,
            ..Self::default()
        }
    }
}
