//! Candidate-arc extraction for the min-cost-flow (assignment) solver.
//!
//! # Overview
//! The min-cost-flow formulation used by this solver operates on a **dense**
//! local indexing of a connected component: each component node is mapped to
//! an integer in `0..n_nodes`.
//!
//! This module builds the *candidate arc set* used by the assignment:
//! `candidate_arcs[(local_u, local_v)] = (edge_id, cost)`
//!
//! where:
//! - `(local_u, local_v)` are local indices inside the component,
//! - `edge_id` is the backing [`EdgeId`] in the global [`InterNightGraph`],
//! - `cost` is the scalar score associated with that edge.
//!
//! # Why "cheapest per (u, v)"?
//! The global inter-night graph may contain multiple edges between the same
//! ordered pair of nodes (e.g. produced by different heuristics, scoring modes,
//! or intermediate candidates). For the assignment reduction we keep only the
//! **cheapest** edge per ordered pair `(u, v)` to:
//! - avoid parallel edges in the bipartite network,
//! - reduce problem size,
//! - preserve a clear interpretation of "choosing u -> v" as a single decision.
//!
//! # Filtering
//! We restrict candidates to:
//! - edges marked `ACTIVE` (`edge.active == true`),
//! - edges whose target node stays **within the same connected component**.
//!
//! This ensures the solver only reasons about the induced subgraph of the
//! component and respects upstream pruning.
//!
//! See also
//! --------
//! - `assignment::build_network` for how these arcs become `L(u) -> R(v)` edges.
//! - `k_best::compute_k_solutions` for how candidate arcs are reused across re-solves.

use ahash::{AHashMap, AHashSet};

use crate::graph::{edge_id::EdgeId, graph::InterNightGraph, node_id::NodeId};

/// Extract candidate arcs inside one connected component (cheapest per ordered node pair).
///
/// This function builds a dense local indexing for the component and scans the
/// outgoing adjacency lists of the component nodes. For each ACTIVE edge
/// `from -> to` whose target is also in the component, it inserts a candidate
/// arc keyed by `(local_from, local_to)`.
///
/// If several global edges map to the same `(local_from, local_to)` pair, only
/// the **lowest-cost** one is kept.
///
/// Parameters
/// ----------
/// graph : &InterNightGraph
///     Global inter-night graph storage (read-only).
/// component_nodes : &[NodeId]
///     List of node ids belonging to the component. The order of this slice
///     defines the local indexing: `component_nodes[local]` is the global node
///     corresponding to local index `local`.
///
/// Returns
/// -------
/// AHashMap<(usize, usize), (EdgeId, f64)>
///     Map `(local_u, local_v) -> (edge_id, cost)` containing the cheapest
///     candidate arc for each ordered local node pair.
///
/// Notes
/// -----
/// - Local indexing is created once per call via a hash map `NodeId -> usize`.
/// - Membership in the component is tested with a hash set for O(1) average lookups.
/// - The returned map is intentionally sparse and directly usable to build the
///   bipartite assignment network.
/// - The cost is taken from `edge.cost` and assumed to be a scalar score where
///   "smaller is better".
///
/// Complexity
/// ----------
/// Let `E_out` be the total number of outgoing adjacency entries across nodes in
/// the component. Then:
/// - time is O(E_out) expected (hash lookups),
/// - space is O(A) where `A` is the number of distinct `(u, v)` pairs retained.
pub(super) fn build_candidate_arcs(
    graph: &InterNightGraph,
    component_nodes: &[NodeId],
) -> AHashMap<(usize, usize), (EdgeId, f64)> {
    // Local indexing for compact 0..n-1 representations.
    let local_index_of: AHashMap<NodeId, usize> = component_nodes
        .iter()
        .copied()
        .enumerate()
        .map(|(local_idx, node_id)| (node_id, local_idx))
        .collect();

    // Membership test for "is target in component?"
    let is_in_component: AHashSet<NodeId> = component_nodes.iter().copied().collect();

    let mut candidate_arcs: AHashMap<(usize, usize), (EdgeId, f64)> = AHashMap::default();

    component_nodes
        .iter()
        .copied()
        // Iterate all outgoing edges from all component nodes.
        .flat_map(|from_node| graph.out_adj[from_node.idx()].iter().copied())
        .filter_map(|edge_id| {
            let edge = &graph.edges[edge_id.idx() as usize];

            // Only consider ACTIVE edges.
            if !edge.active {
                return None;
            }

            // Only consider edges whose target stays within the component.
            if !is_in_component.contains(&edge.to) {
                return None;
            }

            // Convert global node ids to local indices.
            let local_from = *local_index_of.get(&edge.from)?;
            let local_to = *local_index_of.get(&edge.to)?;

            Some(((local_from, local_to), (edge_id, edge.cost)))
        })
        .for_each(|(key, (edge_id, edge_cost))| {
            keep_cheapest_arc(&mut candidate_arcs, key, edge_id, edge_cost)
        });

    candidate_arcs
}

/// Insert an arc into the candidate map, keeping only the cheapest for each `(u, v)` key.
///
/// Parameters
/// ----------
/// candidate_arcs : &mut AHashMap<(usize, usize), (EdgeId, f64)>
///     Mutable candidate arc map.
/// node_pair : (usize, usize)
///     Local ordered pair `(local_from, local_to)`.
/// edge_id : EdgeId
///     Backing global edge id.
/// edge_cost : f64
///     Candidate arc cost.
///
/// Notes
/// -----
/// - If `node_pair` is absent, this inserts `(edge_id, edge_cost)`.
/// - If present, it overwrites the stored value only when `edge_cost` is strictly
///   smaller than the currently stored cost.
/// - Tie-breaking for equal costs is intentionally left as "keep first seen"
///   (stable with respect to iteration order). If you need deterministic
///   tie-breaking across different iteration orders, consider comparing
///   `edge_id` as a secondary key.
pub(super) fn keep_cheapest_arc(
    candidate_arcs: &mut AHashMap<(usize, usize), (EdgeId, f64)>,
    node_pair: (usize, usize),
    edge_id: EdgeId,
    edge_cost: f64,
) {
    candidate_arcs
        .entry(node_pair)
        .and_modify(|best| {
            if edge_cost < best.1 {
                *best = (edge_id, edge_cost);
            }
        })
        .or_insert((edge_id, edge_cost));
}
