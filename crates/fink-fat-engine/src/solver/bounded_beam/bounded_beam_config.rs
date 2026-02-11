//! Bounded beam solver configuration.
//!
//! Overview
//! --------
//! This module defines the configuration structure used by the
//! [`BoundedBeamSolver`], which enumerates candidate directed paths inside a
//! time-forward DAG component using a **bounded beam search**.
//!
//! The solver is intentionally controlled and predictable: although path
//! enumeration can grow combinatorially in branching graphs, this configuration
//! enforces strict guardrails on exploration breadth, output size, and total
//! computational effort.
//!
//! Exploration model
//! -----------------
//! The `BoundedBeamSolver` performs the following high-level steps:
//!
//! 1. Build a restricted directed adjacency inside the component.
//! 2. Optionally prune outgoing edges per node (`max_out_per_node`).
//! 3. Initialize beam states from component sources.
//! 4. Iteratively expand partial paths (beam search).
//! 5. Collect terminal paths (ending at sinks).
//! 6. Rank and truncate results.
//!
//! Without guardrails, the number of candidate paths may grow exponentially.
//! This configuration makes the solver bounded and safe.
//!
//! Controlled dimensions
//! ---------------------
//! The configuration exposes independent limits over:
//!
//! - **output size** (`max_tracks`, `max_tracks_per_source`),
//! - **search breadth** (`beam_width`),
//! - **local branching factor** (`max_out_per_node`),
//! - **minimum track depth** (`min_nodes`),
//! - **global exploration budget** (`max_expansions`).
//!
//! Interactions between knobs
//! --------------------------
//! The parameters interact multiplicatively:
//!
//! - `beam_width` controls how many partial hypotheses survive each expansion.
//! - `max_out_per_node` limits the effective branching factor.
//! - `max_expansions` caps total work regardless of topology.
//! - `max_tracks` and `max_tracks_per_source` bound final output.
//!
//! Tuning guidelines
//! -----------------
//!
//! If runtime is too high:
//! - decrease `beam_width`,
//! - decrease `max_out_per_node`,
//! - decrease `max_expansions`.
//!
//! If recall is too low:
//! - increase `beam_width`,
//! - increase `max_out_per_node`,
//! - increase `max_expansions`,
//! - increase `max_tracks`.
//!
//! If diversity is low (one source dominates):
//! - decrease `max_tracks_per_source`,
//! - increase `max_tracks`.
//!
//! Safety assumptions
//! ------------------
//! The solver assumes:
//! - a time-forward DAG (no backward edges),
//! - component-restricted adjacency,
//! - deterministic edge costs.
//!
//! This configuration does not alter graph correctness assumptions;
//! it only bounds exploration and output.

/// Configuration knobs for the [`BoundedBeamSolver`].
///
/// This structure controls both:
/// - beam search exploration,
/// - output filtering and truncation.
///
/// The solver enumerates candidate directed paths inside one connected
/// component. Without constraints, path enumeration can become expensive
/// in highly branching structures. These parameters ensure bounded behavior.
#[derive(Clone, Debug)]
pub struct BoundedBeamConfig {
    /// Maximum number of tracks returned per component.
    ///
    /// This is a global output cap applied *after ranking*.
    ///
    /// Effects
    /// -------
    /// - Controls downstream workload (trajectory fitting, persistence).
    /// - Guarantees bounded output size.
    ///
    /// If too small:
    /// - high-quality tracks may be truncated.
    pub max_tracks: usize,

    /// Minimum number of nodes required for a returned track.
    ///
    /// Interpretation:
    /// - `min_nodes = 2` → at least one edge.
    /// - `min_nodes = 3` → at least two edges (typical minimum for
    ///   multi-night stability).
    ///
    /// Applied:
    /// - during terminal emission,
    /// - and again during final filtering.
    pub min_nodes: usize,

    /// Beam width: maximum number of partial hypotheses kept per iteration.
    ///
    /// At each expansion step:
    /// - candidate next states are generated,
    /// - sorted by total cost,
    /// - truncated to `beam_width`.
    ///
    /// Larger values:
    /// - increase recall,
    /// - increase runtime and memory.
    ///
    /// Smaller values:
    /// - reduce cost,
    /// - increase pruning aggressiveness.
    pub beam_width: usize,

    /// Local pruning: keep only the best K outgoing edges per node.
    ///
    /// During adjacency preparation:
    /// - outgoing edges are sorted by increasing cost,
    /// - truncated to `max_out_per_node`.
    ///
    /// This reduces local branching before beam search starts.
    ///
    /// Larger values:
    /// - increase recall,
    /// - increase branching.
    ///
    /// Smaller values:
    /// - reduce branching drastically,
    /// - may prune valid hypotheses early.
    pub max_out_per_node: usize,

    /// Maximum number of tracks emitted per source node.
    ///
    /// Prevents a single source from dominating output.
    ///
    /// Encourages diversity across different entry points.
    pub max_tracks_per_source: usize,

    /// Hard cap on total edge expansions in one component.
    ///
    /// Each state expansion increments a counter.
    /// Once `max_expansions` is reached:
    /// - exploration stops,
    /// - remaining beam states are evaluated as-is.
    ///
    /// Guarantees strict upper bound on work per component.
    ///
    /// Important:
    /// - This is a safety guardrail.
    /// - It may reduce recall if set too low.
    pub max_expansions: usize,
}

impl Default for BoundedBeamConfig {
    /// Default configuration for the [`BoundedBeamSolver`].
    ///
    /// These defaults provide:
    /// - moderate beam width,
    /// - moderate local pruning,
    /// - limited output size,
    /// - generous global expansion cap.
    ///
    /// They are intended to be safe for small-to-medium components.
    fn default() -> Self {
        Self {
            max_tracks: 16,
            min_nodes: 3,
            beam_width: 64,
            max_out_per_node: 8,
            max_tracks_per_source: 8,
            max_expansions: 50_000,
        }
    }
}
