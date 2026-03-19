//! Track reconstruction from assignment solutions.
//!
//! # Overview
//! The min-cost-flow assignment produces a global [`Solution`] over a connected
//! component using dense local indices `0..n_nodes`. A solution encodes a set of
//! selected directed arcs `u -> v` such that each node has at most one successor
//! and at most one predecessor (a *partial path cover* with breaks).
//!
//! This module converts one or more global solutions into concrete
//! [`TrackHypothesis`] objects:
//! - each track is a **disjoint directed chain**,
//! - tracks are extracted by pointer-chasing `successor_of`,
//! - short tracks are filtered (`min_nodes`),
//! - tracks are deduplicated across solutions,
//! - and (optionally) the module returns a list of edge ids that could be
//!   deactivated if these tracks are accepted downstream.
//!
//! # Deduplication strategy
//! Tracks are deduplicated across multiple solutions using the **exact sequence
//! of backing `EdgeId`s** as a key. This is:
//! - stable and transparent,
//! - cheap to implement,
//! - but allocation-heavy (`Vec<EdgeId>` stored in a hash set).
//!
//! If this becomes a bottleneck, you can replace the key with a compact hash
//! signature (e.g., 64-bit hash of the edge id sequence).
//!
//! # Reconstruction details
//! To extract chains from a solution we do two passes:
//! 1. Start from nodes with **no predecessor** (typical chain heads).
//! 2. Visit any leftover nodes (cycles, isolated nodes, or components where every
//!    node has a predecessor), ensuring full coverage.
//!
//! See also
//! --------
//! - `solution::Solution` for the pointer representation used here.
//! - `candidate_arcs` for mapping `(u, v)` to backing `(EdgeId, cost)`.
//! - `assignment::extract_matching` for how `successor_of`/`predecessor_of` are formed.

use std::cmp::Ordering;

use ahash::{AHashMap, AHashSet};

use crate::{
    graph::{edge_id::EdgeId, graph::InterNightGraph, node_id::NodeId},
    trajectory::track_hypothesis::TrackHypothesis,
};

use super::solution::Solution;

/// Convert a set of global solutions into track hypotheses and deduplicate them.
///
/// Parameters
/// ----------
/// graph : &InterNightGraph
///     Global graph storage (read-only). Used here to compute track metadata
///     such as `night_span`.
/// component_nodes : &[NodeId]
///     Mapping between local and global nodes. The index in this slice is the
///     local node id used by the assignment solution.
/// candidate_arcs : &AHashMap<(usize, usize), (EdgeId, f64)>
///     Candidate arc lookup table produced by `build_candidate_arcs`.
///     This is used to translate each selected successor pointer `(u, v)` into
///     the backing edge id and arc cost.
/// solutions : &[Solution]
///     List of global solutions (best and optional alternatives).
/// min_nodes : usize
///     Minimum number of nodes required for a track hypothesis to be returned.
///     Tracks shorter than this are dropped.
/// propose_deactivations : bool
///     If true, return a list of edge ids used by returned tracks, suitable as a
///     *proposal* to deactivate these edges in the global graph.
///
/// Returns
/// -------
/// (Vec<TrackHypothesis>, Vec<EdgeId>)
///     `(tracks, proposed_deactivations)` where:
///     - `tracks` are sorted by increasing track cost,
///     - `proposed_deactivations` is deduplicated and sorted when enabled,
///       otherwise it is empty.
///
/// Notes
/// -----
/// - Deduplication is performed across solutions using the exact `EdgeId`
///   sequence of each track.
/// - Costs returned in `TrackHypothesis.cost` are the sum of arc costs along the
///   chain (break penalties are not included here).
pub(super) fn solutions_to_tracks(
    graph: &InterNightGraph,
    component_nodes: &[NodeId],
    candidate_arcs: &AHashMap<(usize, usize), (EdgeId, f64)>,
    solutions: &[Solution],
    min_nodes: usize,
    propose_deactivations: bool,
) -> (Vec<TrackHypothesis>, Vec<EdgeId>) {
    // Dedup key: the exact EdgeId sequence.
    // (Cheap and stable, but alloc-heavy; can later be replaced with a hash signature.)
    let mut seen_edge_sequences: AHashSet<Vec<EdgeId>> = AHashSet::new();

    let mut tracks: Vec<TrackHypothesis> = Vec::new();
    let mut deactivations: Vec<EdgeId> = Vec::new();

    solutions.iter().for_each(|solution| {
        let (solution_tracks, solution_edges) =
            solution_to_tracks(graph, component_nodes, candidate_arcs, solution, min_nodes);

        solution_tracks
            .into_iter()
            .filter(|track| seen_edge_sequences.insert(track.edges.clone()))
            .for_each(|track| tracks.push(track));

        if propose_deactivations {
            deactivations.extend(solution_edges);
        }
    });

    tracks.sort_by(|lhs, rhs| lhs.cost.partial_cmp(&rhs.cost).unwrap_or(Ordering::Equal));

    if propose_deactivations {
        deactivations.sort();
        deactivations.dedup();
    } else {
        deactivations.clear();
    }

    (tracks, deactivations)
}

/// Convert a single global assignment [`Solution`] into concrete track hypotheses.
///
/// # Purpose
/// This function reconstructs **disjoint directed chains** (track hypotheses)
/// from a min-cost-flow assignment solution. The assignment encodes successor
/// and predecessor pointers over a dense local indexing of the component;
/// this routine turns that pointer-based representation into explicit
/// [`TrackHypothesis`] objects suitable for downstream processing (e.g. IOD).
///
/// # Reconstruction strategy
/// Tracks are extracted by **pointer chasing** over `solution.successor_of`
/// using a two-pass approach:
///
/// 1. **Chain heads first**  
///    Start from local nodes that have **no predecessor** in the solution.
///    These are natural beginnings of open chains.
///
/// 2. **Leftover nodes**  
///    Any node not yet visited after pass #1 is processed in a second pass.
///    This ensures coverage of:
///    - cycles (where every node has a predecessor),
///    - isolated nodes,
///    - or degenerate components.
///
/// Each local node is visited at most once, guaranteeing that reconstructed
/// tracks are disjoint.
///
/// # Filtering
/// Tracks shorter than `min_nodes` are discarded *after* reconstruction.
/// This filtering does **not** affect traversal order or visited marking:
/// nodes belonging to short tracks are still considered consumed.
///
/// # Edge collection
/// For each accepted track, the backing [`EdgeId`]s are accumulated and returned
/// as a flat list (`edges_to_deactivate`). This list represents the union of all
/// edges used by accepted tracks and can be used by the caller to propose
/// deactivation in the global graph.
///
/// # Parameters
/// graph : &InterNightGraph
///     Global graph storage. Used here only to compute track metadata such as
///     the night span.
/// component_nodes : &[NodeId]
///     Mapping from local indices (`0..n_nodes`) to global node identifiers.
/// candidate_arcs : &AHashMap<(usize, usize), (EdgeId, f64)>
///     Lookup table mapping `(local_from, local_to)` to the backing edge id and
///     arc cost. This is assumed to be consistent with the assignment solution.
/// solution : &Solution
///     A single global min-cost-flow assignment solution.
/// min_nodes : usize
///     Minimum number of nodes required for a track to be returned.
///
/// # Returns
/// (Vec<TrackHypothesis>, Vec<EdgeId>)
///     - `tracks`: reconstructed track hypotheses satisfying the length filter,
///     - `edges_to_deactivate`: union of all edge ids used by those tracks.
///
/// # Invariants and assumptions
/// - `solution.successor_of` and `candidate_arcs` are consistent
///   (every selected successor pair must exist in `candidate_arcs`).
/// - Each local node belongs to at most one returned track.
/// - The function is deterministic given fixed inputs.
///
/// # Notes
/// - Break penalties are **not** included in the returned track costs; each
///   track cost is simply the sum of its arc costs.
/// - This function is intentionally private and tightly coupled to the internal
///   representation of `Solution`.
fn solution_to_tracks(
    graph: &InterNightGraph,
    component_nodes: &[NodeId],
    candidate_arcs: &AHashMap<(usize, usize), (EdgeId, f64)>,
    solution: &Solution,
    min_nodes: usize,
) -> (Vec<TrackHypothesis>, Vec<EdgeId>) {
    let n_nodes = component_nodes.len();

    let mut visited_local = vec![false; n_nodes];
    let mut tracks: Vec<TrackHypothesis> = Vec::new();
    let mut edges_to_deactivate: Vec<EdgeId> = Vec::new();

    // Pass #1: start from nodes with no predecessor (typical chain heads).
    for start_local in 0..n_nodes {
        if solution.predecessor_of[start_local].is_some() || visited_local[start_local] {
            continue;
        }

        if let Some(track) = walk_chain(
            graph,
            component_nodes,
            candidate_arcs,
            solution,
            start_local,
            &mut visited_local,
        ) {
            if track.nodes.len() >= min_nodes {
                edges_to_deactivate.extend(track.edges.iter().copied());
                tracks.push(track);
            }
        }
    }

    // Pass #2: leftover nodes (cycles, isolated nodes, etc.)
    for start_local in 0..n_nodes {
        if visited_local[start_local] {
            continue;
        }

        if let Some(track) = walk_chain(
            graph,
            component_nodes,
            candidate_arcs,
            solution,
            start_local,
            &mut visited_local,
        ) {
            if track.nodes.len() >= min_nodes {
                edges_to_deactivate.extend(track.edges.iter().copied());
                tracks.push(track);
            }
        }
    }

    (tracks, edges_to_deactivate)
}

/// Walk one directed chain `start -> succ -> succ -> ...` until it stops.
///
/// The walk stops when:
/// - the current node has no successor in the solution (break/end),
/// - the `(u, v)` successor pair is missing from `candidate_arcs` (inconsistent input),
/// - or the successor points into an already visited node (cycle protection).
///
/// Parameters
/// ----------
/// graph : &InterNightGraph
///     Global graph storage (used for metadata such as `night_span`).
/// component_nodes : &[NodeId]
///     Local-to-global mapping slice.
/// candidate_arcs : &AHashMap<(usize, usize), (EdgeId, f64)>
///     Arc lookup table `(u, v) -> (edge_id, cost)`.
/// solution : &Solution
///     The assignment solution providing `successor_of`.
/// start_local : usize
///     Local node index to start the walk from.
/// visited_local : &mut [bool]
///     Per-node visited marker to ensure each local node is assigned to at most
///     one reconstructed chain.
///
/// Returns
/// -------
/// Option<TrackHypothesis>
///     The reconstructed track, or `None` if no nodes were collected (should
///     not happen in normal usage).
///
/// Notes
/// -----
/// This is intentionally a `while` loop:
/// - it directly expresses pointer-chasing semantics,
/// - it avoids iterator gymnastics that would reduce clarity,
/// - it makes cycle handling explicit.
fn walk_chain(
    graph: &InterNightGraph,
    component_nodes: &[NodeId],
    candidate_arcs: &AHashMap<(usize, usize), (EdgeId, f64)>,
    solution: &Solution,
    start_local: usize,
    visited_local: &mut [bool],
) -> Option<TrackHypothesis> {
    let n_nodes = component_nodes.len();

    let mut local_nodes_in_chain: Vec<usize> = Vec::new();
    let mut edges_in_chain: Vec<EdgeId> = Vec::new();
    let mut sum_cost = 0.0;

    let mut current_local = start_local;

    while current_local < n_nodes && !visited_local[current_local] {
        visited_local[current_local] = true;
        local_nodes_in_chain.push(current_local);

        let Some(next_local) = solution.successor_of[current_local] else {
            break;
        };

        // Translate the chosen successor pointer back to a concrete backing edge + cost.
        let Some(&(edge_id, edge_cost)) = candidate_arcs.get(&(current_local, next_local)) else {
            // This should not happen if `candidate_arcs` and the assignment network
            // were built consistently. Treat as a hard stop to preserve safety.
            break;
        };

        edges_in_chain.push(edge_id);
        sum_cost += edge_cost;

        current_local = next_local;
    }

    if local_nodes_in_chain.is_empty() {
        return None;
    }

    // Convert local indices to global NodeIds.
    let chain_nodes: Vec<NodeId> = local_nodes_in_chain
        .iter()
        .map(|&local_idx| component_nodes[local_idx])
        .collect();

    // Compute the span in nights covered by the chain (metadata for filtering/diagnostics).
    let first_night = graph.nodes[chain_nodes[0].idx()].night.0;
    let last_night = graph.nodes[chain_nodes.last().unwrap().idx()].night.0;
    let night_span = last_night.saturating_sub(first_night);

    Some(TrackHypothesis {
        nodes: chain_nodes,
        edges: edges_in_chain,
        cost: sum_cost,
        night_span,
        n_nodes: local_nodes_in_chain.len() as u32,
    })
}
