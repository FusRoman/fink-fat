//! Solution model for the min-cost-flow (assignment) solver.
//!
//! # Overview
//! This module defines the **internal data structures** used to represent a
//! *global* solution of the bipartite assignment underlying the min-cost-flow
//! formulation of the path-cover problem.
//!
//! In the solver, each connected component is re-indexed locally as
//! `0..n_nodes` (dense indexing). The assignment chooses, for each local node:
//! - **at most one successor** (1-out), encoded by `successor_of[u]`,
//! - **at most one predecessor** (1-in), encoded by `predecessor_of[v]`.
//!
//! A solution is therefore a *set of directed arcs* `u -> v` between local
//! indices. Each selected arc corresponds to an actual graph edge
//! ([`EdgeId`]) and carries a scalar cost.
//!
//! # Why keep both pointer arrays and an explicit arc list?
//! The solver needs two complementary views:
//! - `successor_of` / `predecessor_of` for **fast chain reconstruction**
//!   (pointer chasing into disjoint tracks),
//! - `selected_arcs` for **deterministic alternative generation**
//!   (e.g., "forbid one selected arc") and for diagnostics.
//!
//! # Signatures and deduplication
//! When generating multiple global solutions (K-best approximation), we must
//! deduplicate identical solutions that may arise from different forbidden-arc
//! attempts. This module provides a stable [`signature_from_selected_arcs`]
//! helper that maps a solution to a canonical string key.
//!
//! The signature is intentionally based on **sorted backing edge ids**:
//! - it is deterministic,
//! - it is cheap to compute,
//! - it is stable across runs as long as `EdgeId` assignment is stable.
//!
//! Notes
//! -----
//! - These types are `pub(super)` because they are intended to be used only by
//!   the parent `min_cost_flow` module (and its submodules), not as a public API.
//! - Costs are stored as `f64` because the upstream scoring pipeline uses
//!   floating-point scores.
//!
//! See also
//! --------
//! - `assignment::extract_matching` for how `SelectedArc` is built from the residual network.
//! - `tracks::walk_chain` for how `successor_of` / `predecessor_of` are used.

use crate::graph::edge_id::EdgeId;

/// One selected assignment arc `local_from -> local_to`, backed by a graph edge.
///
/// This is the *explicit* representation of a chosen link in a global solution.
/// It is mostly used for:
/// - alternative generation (e.g. "forbid this arc and re-solve"),
/// - deterministic ordering and debugging,
/// - building stable solution signatures.
///
/// Fields
/// ------
/// local_from : usize
///     Local index of the predecessor node `u` in `0..n_nodes`.
/// local_to : usize
///     Local index of the successor node `v` in `0..n_nodes`.
/// edge_id : EdgeId
///     Backing graph edge id for the selected arc.
/// cost : f64
///     Scalar cost of this arc (already includes all gating / scoring terms).
#[derive(Clone, Debug)]
pub(super) struct SelectedArc {
    /// Local index of the predecessor node.
    pub(super) local_from: usize,
    /// Local index of the successor node.
    pub(super) local_to: usize,
    /// Graph edge id backing this selected arc.
    pub(super) edge_id: EdgeId,
    /// Edge cost (already scalar score).
    pub(super) cost: f64,
}

/// One global solution of the min-cost assignment.
///
/// A solution is a **partial path cover** over the component nodes: each node
/// has at most one successor and at most one predecessor. Unmatched ends
/// ("breaks") are allowed and are accounted for in `total_cost` via the break
/// penalty in the flow formulation.
///
/// Representation
/// --------------
/// The solution is stored in two redundant-but-useful forms:
/// - `successor_of[u] = Some(v)` means the arc `u -> v` is selected.
/// - `predecessor_of[v] = Some(u)` means the arc `u -> v` is selected.
/// - `selected_arcs` lists the same selected arcs explicitly, along with their
///   backing [`EdgeId`] and arc costs.
///
/// This redundancy is intentional:
/// - pointer arrays make chain reconstruction trivial and fast,
/// - the explicit list makes K-best generation deterministic and cheap.
///
/// Fields
/// ------
/// successor_of : Vec<Option<usize>>
///     Successor pointer per local node index.
/// predecessor_of : Vec<Option<usize>>
///     Predecessor pointer per local node index.
/// selected_arcs : Vec<SelectedArc>
///     Explicit list of chosen arcs (for K-best and diagnostics).
/// total_cost : f64
///     Objective value of the solution: sum(selected arc costs) + sum(break penalties).
/// signature : String
///     Canonical stable key derived from `selected_arcs`, used to deduplicate solutions.
///
/// Notes
/// -----
/// - Local indices are dense indices into the component's node list.
///   The mapping `local -> global NodeId` is maintained by the caller.
/// - `signature` is stored to avoid recomputing it when evaluating uniqueness.
#[derive(Clone, Debug)]
pub(super) struct Solution {
    /// Successor pointer per local node index.
    ///
    /// `successor_of[u] = Some(v)` means u -> v is selected.
    pub(super) successor_of: Vec<Option<usize>>,

    /// Predecessor pointer per local node index.
    pub(super) predecessor_of: Vec<Option<usize>>,

    /// Selected arcs (for alternative generation & diagnostics).
    pub(super) selected_arcs: Vec<SelectedArc>,

    /// Objective value (edges + breaks).
    pub(super) total_cost: f64,

    /// Stable signature used to deduplicate solutions.
    pub(super) signature: String,
}

/// Build a stable solution signature from a set of selected arcs.
///
/// The signature is the comma-separated list of **sorted backing edge ids**.
/// It is designed to be:
/// - **deterministic** (order-independent),
/// - **stable** as long as `EdgeId` assignment is stable,
/// - **cheap** to compute,
/// - suitable as a key for solution deduplication in K-best generation.
///
/// Parameters
/// ----------
/// selected_arcs : &[SelectedArc]
///     The selected arcs of a solution. The input order does not matter.
///
/// Returns
/// -------
/// String
///     A canonical signature string, e.g. `"12,58,104"`.
///
/// Notes
/// -----
/// - This signature ignores arc direction and node indices, relying solely on
///   the set of backing edge ids. This is sufficient if each solution’s chosen
///   arcs map uniquely to graph edges (which holds here because we keep only the
///   cheapest arc per `(u, v)` pair when building candidate arcs).
/// - If you later allow multiple distinct edges for the same `(u, v)` pair, the
///   signature remains valid because it keys on the actual `EdgeId`s.
/// - If you prefer an allocation-light alternative, you can replace the string
///   with a 64-bit hash computed from the sorted ids (e.g., xxhash). The current
///   string form is intentionally transparent for debugging.
pub(super) fn signature_from_selected_arcs(selected_arcs: &[SelectedArc]) -> String {
    let mut edge_ids: Vec<u64> = selected_arcs
        .iter()
        .map(|arc| arc.edge_id.idx() as u64)
        .collect();

    edge_ids.sort_unstable();

    edge_ids
        .into_iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",")
}
