//! Assignment reduction for the min-cost-flow path-cover solver.
//!
//! # Overview
//! This module turns a set of candidate directed arcs `(u -> v)` into a
//! **bipartite min-cost flow** instance (an assignment with optional breaks),
//! solves it once, and extracts a `Solution` representation.
//!
//! Concretely, for a connected component re-indexed locally as `0..n_nodes`:
//! - each node `u` must choose **one successor** or **a break** (1-out with break),
//! - each node `v` must receive **one predecessor** or **a break** (1-in with break).
//!
//! This is implemented as a min-cost max-flow on a bipartite network:
//! - Left partition `L(u)` represents the decision "successor of `u`",
//! - Right partition `R(v)` represents the decision "predecessor of `v`",
//! - candidate arc `u -> v` becomes an edge `L(u) -> R(v)` with cost = arc cost,
//! - breaks are encoded by edges paying `break_penalty`.
//!
//! The flow is forced to send exactly `2 * n_nodes` units:
//! - `n_nodes` units ensure each `L(u)` is used once (successor or break),
//! - `n_nodes` units ensure each `R(v)` is satisfied once (predecessor or break).
//!
//! If the required flow cannot be sent, the instance is considered infeasible
//! (should be rare for well-formed inputs).
//!
//! # Inputs and assumptions
//! The solver operates on `candidate_arcs: (u, v) -> (EdgeId, cost)` where:
//! - `u` and `v` are *local* indices in `0..n_nodes`,
//! - `EdgeId` refers to the backing edge in the global inter-night graph,
//! - only the **cheapest** arc per `(u, v)` pair should be present upstream.
//!
//! A `forbidden_arcs` set can be provided to exclude specific `(u, v)` pairs;
//! this is used by the simple K-best strategy (one-edge deviation).
//!
//! # Outputs
//! The result is a [`Solution`] containing:
//! - pointer arrays (`successor_of`, `predecessor_of`) for chain reconstruction,
//! - an explicit `selected_arcs` list for determinism/diagnostics,
//! - `total_cost` (edges + breaks),
//! - `signature` (stable solution key for deduplication).
//!
//! See also
//! --------
//! - `mcf::MinCostFlow` for the generic SSAP + potentials implementation.
//! - `solution::{Solution, SelectedArc}` for the in-memory representation.
//! - `k_best::compute_k_solutions` for alternative generation using `forbidden_arcs`.

use ahash::{AHashMap, AHashSet};

use crate::graph::edge_id::EdgeId;

use super::{
    mcf::MinCostFlow,
    solution::{signature_from_selected_arcs, SelectedArc, Solution},
};

/// Solve one assignment instance and return a global [`Solution`].
///
/// This function:
/// 1. builds the bipartite min-cost flow network (see [`build_network`]),
/// 2. sends exactly `2 * n_nodes` units of flow at minimum cost,
/// 3. extracts the selected `L(u)->R(v)` arcs as a matching,
/// 4. converts that matching into a [`Solution`] with a stable signature.
///
/// Parameters
/// ----------
/// n_nodes : usize
///     Number of nodes in the connected component (dense local indexing).
/// candidate_arcs : &AHashMap<(usize, usize), (EdgeId, f64)>
///     Candidate directed arcs for this component. Keys are local index pairs `(u, v)`.
///     Values are `(edge_id, cost)`, where `edge_id` references the backing edge
///     in the global graph and `cost` is the scalar arc cost.
/// break_penalty : f64
///     Penalty paid for leaving a node without a predecessor and/or successor.
///     This is implemented via break edges in the flow network.
/// forbidden_arcs : &AHashSet<(usize, usize)>
///     Set of local pairs `(u, v)` to exclude from the candidate set.
///     Used for K-best alternatives ("forbid one selected arc").
///
/// Returns
/// -------
/// Option<Solution>
///     - `Some(solution)` if the network can send exactly `2 * n_nodes` units.
///     - `None` if the required flow cannot be sent (infeasible instance).
///
/// Notes
/// -----
/// - The required flow is `2 * n_nodes` (not `n_nodes`) because we encode
///   both constraints:
///   - each `u` chooses a successor/break (left side),
///   - each `v` gets a predecessor/break (right side).
/// - Infeasibility is typically a sign of inconsistent network construction
///   or a bug upstream (e.g., wrong vertex counts), not "lack of arcs".
pub(super) fn solve_once(
    n_nodes: usize,
    candidate_arcs: &AHashMap<(usize, usize), (EdgeId, f64)>,
    break_penalty: f64,
    forbidden_arcs: &AHashSet<(usize, usize)>,
) -> Option<Solution> {
    let (mut flow_network, source, sink) =
        build_network(n_nodes, candidate_arcs, break_penalty, forbidden_arcs);

    // We target exactly 2*n units:
    // - n units ensure each L(u) chooses successor or break,
    // - n units ensure each R(v) is satisfied by predecessor or break.
    let required_flow = (2 * n_nodes) as i32;

    let (sent_flow, total_cost) = flow_network.min_cost_max_flow(source, sink, required_flow);
    if sent_flow != required_flow {
        return None;
    }

    let (successor_of, predecessor_of, selected_arcs) =
        extract_matching(n_nodes, candidate_arcs, &flow_network);

    let signature = signature_from_selected_arcs(&selected_arcs);

    Some(Solution {
        successor_of,
        predecessor_of,
        selected_arcs,
        total_cost,
        signature,
    })
}

/// Build the bipartite min-cost flow network encoding the assignment with breaks.
///
/// # Network layout
///
/// Vertices
/// --------
/// We use a compact contiguous indexing:
/// - `S = 0` (source)
/// - `L(u) = 1 + u` for `u in 0..n_nodes`
/// - `R(v) = 1 + n_nodes + v` for `v in 0..n_nodes`
/// - `T = 1 + 2*n_nodes` (sink)
///
/// Edges
/// -----
/// - `S -> L(u)` capacity 1, cost 0
///     Forces one unit of flow to leave each `L(u)` (choose successor or break).
/// - `L(u) -> R(v)` capacity 1, cost = arc_cost
///     Encodes choosing `v` as successor of `u`.
/// - `L(u) -> T` capacity 1, cost = break_penalty
///     Encodes "no successor" for `u` (break on the successor side).
/// - `S -> R(v)` capacity 1, cost = break_penalty
///     Encodes "no predecessor" for `v` (break on the predecessor side).
/// - `R(v) -> T` capacity 1, cost 0
///     Forces one unit of flow to exit each `R(v)` into the sink.
///
/// # Parameters
/// n_nodes : usize
///     Number of local nodes in the component.
/// candidate_arcs : &AHashMap<(usize, usize), (EdgeId, f64)>
///     Candidate arcs `(u, v)` with costs.
/// break_penalty : f64
///     Break penalty used for `L(u)->T` and `S->R(v)` edges.
/// forbidden_arcs : &AHashSet<(usize, usize)>
///     Local pairs `(u, v)` to exclude from the candidate arcs.
///
/// # Returns
/// (MinCostFlow, usize, usize)
///     The constructed network, plus `(source, sink)` vertex indices.
///
/// Notes
/// -----
/// - Break edges ensure the instance remains feasible even when a node has no
///   admissible successor/predecessor, at the expense of paying penalties.
/// - Candidate arcs are filtered by `forbidden_arcs` to support K-best generation.
pub(super) fn build_network(
    n_nodes: usize,
    candidate_arcs: &AHashMap<(usize, usize), (EdgeId, f64)>,
    break_penalty: f64,
    forbidden_arcs: &AHashSet<(usize, usize)>,
) -> (MinCostFlow, usize, usize) {
    let source = 0usize;
    let sink = 1 + 2 * n_nodes;

    let mut network = MinCostFlow::new(sink + 1);

    add_source_to_left_part(&mut network, source, n_nodes);
    add_candidate_arcs(&mut network, n_nodes, candidate_arcs, forbidden_arcs);
    add_break_arcs(&mut network, n_nodes, sink, break_penalty);
    add_right_part_to_sink(&mut network, n_nodes, sink);

    (network, source, sink)
}

/// Add `S -> L(u)` edges (one unit per left node).
///
/// This enforces the "each node chooses a successor or break" constraint.
fn add_source_to_left_part(network: &mut MinCostFlow, source: usize, n_nodes: usize) {
    (0..n_nodes).for_each(|local_u| {
        network.add_edge(source, 1 + local_u, 1, 0.0);
    });
}

/// Add candidate matching edges `L(u) -> R(v)` with the provided arc costs.
///
/// Any `(u, v)` present in `forbidden_arcs` is skipped.
fn add_candidate_arcs(
    network: &mut MinCostFlow,
    n_nodes: usize,
    candidate_arcs: &AHashMap<(usize, usize), (EdgeId, f64)>,
    forbidden_arcs: &AHashSet<(usize, usize)>,
) {
    candidate_arcs
        .iter()
        .filter(|(pair, _)| !forbidden_arcs.contains(pair))
        .for_each(|(&(local_u, local_v), &(_edge_id, arc_cost))| {
            let left_u = 1 + local_u;
            let right_v = 1 + n_nodes + local_v;
            network.add_edge(left_u, right_v, 1, arc_cost);
        });
}

/// Add break edges implementing optional predecessor/successor.
/// - `L(u) -> T` is "no successor for u".
/// - `S -> R(v)` is "no predecessor for v".
fn add_break_arcs(network: &mut MinCostFlow, n_nodes: usize, sink: usize, break_penalty: f64) {
    // Break successor: L(u) -> T
    (0..n_nodes).for_each(|local_u| {
        network.add_edge(1 + local_u, sink, 1, break_penalty);
    });

    // Break predecessor: S -> R(v)
    (0..n_nodes).for_each(|local_v| {
        network.add_edge(0, 1 + n_nodes + local_v, 1, break_penalty);
    });
}

/// Add `R(v) -> T` edges (one unit per right node).
///
/// This enforces the "each node receives a predecessor or break" constraint.
fn add_right_part_to_sink(network: &mut MinCostFlow, n_nodes: usize, sink: usize) {
    (0..n_nodes).for_each(|local_v| {
        network.add_edge(1 + n_nodes + local_v, sink, 1, 0.0);
    });
}

/// Extract the selected matching from the solved residual network.
///
/// We look for saturated edges `L(u) -> R(v)`:
/// - they connect the left partition to the right partition,
/// - their residual capacity is zero after sending the full required flow.
///
/// Parameters
/// ----------
/// n_nodes : usize
///     Number of local nodes in the component.
/// candidate_arcs : &AHashMap<(usize, usize), (EdgeId, f64)>
///     Candidate arc map used to translate `(u, v)` back to `(EdgeId, cost)`.
/// network : &MinCostFlow
///     The residual network after calling `min_cost_max_flow`.
///
/// Returns
/// -------
/// (Vec<Option<usize>>, Vec<Option<usize>>, Vec<SelectedArc>)
///     - `successor_of[u] = Some(v)` if `u -> v` is selected.
///     - `predecessor_of[v] = Some(u)` if `u -> v` is selected.
///     - `selected_arcs` explicit arc list (one entry per selected `u -> v`).
///
/// Notes
/// -----
/// - The extraction assumes the standard vertex layout from [`build_network`].
/// - We only treat saturated `L(u)->R(v)` edges as matches; break edges are ignored.
/// - The mapping back to `(EdgeId, cost)` uses `candidate_arcs` and should always
///   succeed for valid matches. If it does not, it indicates a mismatch between
///   network construction and the candidate arc set.
pub(super) fn extract_matching(
    n_nodes: usize,
    candidate_arcs: &AHashMap<(usize, usize), (EdgeId, f64)>,
    network: &MinCostFlow,
) -> (Vec<Option<usize>>, Vec<Option<usize>>, Vec<SelectedArc>) {
    let mut successor_of = vec![None; n_nodes];
    let mut predecessor_of = vec![None; n_nodes];
    let mut selected_arcs: Vec<SelectedArc> = Vec::new();

    for local_u in 0..n_nodes {
        let left_u = 1 + local_u;

        for edge in &network.g[left_u] {
            // Match edges have:
            // - destination in R-part
            // - and are saturated (cap == 0 after sending flow).
            if edge.cap != 0 {
                continue;
            }
            if edge.to < 1 + n_nodes || edge.to >= 1 + 2 * n_nodes {
                continue;
            }

            let local_v = edge.to - (1 + n_nodes);

            if let Some(&(edge_id, cost)) = candidate_arcs.get(&(local_u, local_v)) {
                successor_of[local_u] = Some(local_v);
                predecessor_of[local_v] = Some(local_u);

                selected_arcs.push(SelectedArc {
                    local_from: local_u,
                    local_to: local_v,
                    edge_id,
                    cost,
                });
            }
        }
    }

    (successor_of, predecessor_of, selected_arcs)
}
