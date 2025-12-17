//! Trivial greedy solver for *very small* connected components.
//!
//! Overview
//! --------
//! This module provides [`TrivialSolver`], a deterministic and cheap solver
//! intended for tiny connected components where heavier machinery (min-cost
//! flow, blob-breaker, k-best path covers, etc.) would be unnecessary overhead.
//!
//! The solver operates on a connected component of the inter-night graph and
//! extracts a small set of track hypotheses as **directed chains**.
//!
//! Strategy
//! --------
//! 1. Restrict the component to **ACTIVE** directed edges only.
//! 2. Compute the **in-degree inside the component** for each node
//!    (count only ACTIVE edges whose endpoints are both inside the component).
//! 3. Select candidate starts by increasing in-degree (prefer *heads*).
//! 4. For each start, greedily follow the **best outgoing ACTIVE edge**
//!    (lowest edge cost) while:
//!    - staying inside the component,
//!    - avoiding revisits / near-duplicates via a conservative "claimed node" set,
//!    - respecting [`TrivialSolverConfig::max_nodes`].
//! 5. Keep only chains of length ≥ [`TrivialSolverConfig::min_nodes`],
//!    stop after [`TrivialSolverConfig::max_tracks`] tracks.
//!
//! Determinism & invariants
//! ------------------------
//! - The solver is deterministic given a deterministic graph (stable adjacency
//!   iteration order and stable edge costs).
//! - Tracks are returned **sorted by increasing total cost**.
//! - When enabled, `proposed_deactivations` is the **deduplicated union** of all
//!   edges used by returned tracks.
//!
//! Limitations (by design)
//! -----------------------
//! - The "claimed node" guard is intentionally conservative: once a node is used
//!   in a track, it cannot appear in another track for the same component. This
//!   reduces near-duplicates but may miss alternative hypotheses.
//! - Tie-breaking between equal-cost outgoing edges follows the iteration order
//!   of `graph.out_adj[cur]`. If you need a strict tie-breaker (e.g. by `EdgeId`),
//!   implement it inside [`TrivialSolver::best_outgoing_active_edge`].
//!
//! See also
//! --------
//! - [`crate::solver::min_cost_flow`] for medium-sized components.
//! - [`crate::solver::blob_breaker`] for ambiguous dense components.
//! - [`TrackHypothesis`] for the downstream IOD payload.
pub mod trivial_config;

use std::time::Instant;

use ahash::{AHashMap, AHashSet};

use crate::{
    graph::{edge_id::EdgeId, graph::InterNightGraph, node_id::NodeId},
    solver::{
        ComponentStats, Solver, SolverDiagnostics, SolverOutput,
        trivial_solver::trivial_config::TrivialSolverConfig,
    },
    trajectory::track_hypothesis::TrackHypothesis,
};

/// A tiny greedy solver for very small connected components.
///
/// This solver is intended as a fast fallback when components are so small that
/// global optimization would cost more than it saves. It extracts a bounded
/// number of short chains using only local information (edge cost + activity).
///
/// Notes
/// -----
/// The solver is *not* guaranteed to find a globally optimal set of tracks.
/// It aims instead to produce a few plausible, non-overlapping hypotheses
/// quickly and deterministically.
#[derive(Clone, Debug, Default)]
pub struct TrivialSolver {
    /// Configuration knobs (track limits, chain lengths, deactivation proposal).
    pub config: TrivialSolverConfig,
}

impl TrivialSolver {
    /// Create a new [`TrivialSolver`] with the given configuration.
    pub fn new(config: TrivialSolverConfig) -> Self {
        Self { config }
    }

    /// Build a membership set for O(1) "is this node inside the component?" checks.
    ///
    /// Parameters
    /// ----------
    /// component_nodes : &[NodeId]
    ///     Node ids belonging to the connected component currently being solved.
    ///
    /// Return
    /// ------
    /// AHashSet<NodeId>
    ///     Hash set containing exactly the nodes in `component_nodes`.
    fn build_membership_set(component_nodes: &[NodeId]) -> AHashSet<NodeId> {
        component_nodes.iter().copied().collect()
    }

    /// Compute the in-degree inside the component using only ACTIVE edges.
    ///
    /// A node's in-degree is the count of ACTIVE incoming edges `u -> node`
    /// such that both `u` and `node` are inside the component.
    ///
    /// Parameters
    /// ----------
    /// graph : &InterNightGraph
    ///     Global inter-night graph (adjacency + edges + node metadata).
    /// component_nodes : &[NodeId]
    ///     Nodes belonging to this connected component.
    /// in_comp : &AHashSet<NodeId>
    ///     Membership set for `component_nodes`.
    ///
    /// Return
    /// ------
    /// AHashMap<NodeId, u32>
    ///     Map node -> in-degree (restricted to ACTIVE edges within the component).
    fn compute_indeg_active_in_component(
        graph: &InterNightGraph,
        component_nodes: &[NodeId],
        in_comp: &AHashSet<NodeId>,
    ) -> AHashMap<NodeId, u32> {
        component_nodes
            .iter()
            .copied()
            .map(|n| {
                let deg = graph.in_adj[n.idx()]
                    .iter()
                    .filter(|&&eid| {
                        let e = &graph.edges[eid.idx() as usize];
                        e.active && in_comp.contains(&e.from) && in_comp.contains(&e.to)
                    })
                    .count() as u32;

                (n, deg)
            })
            .collect()
    }

    /// Order candidate start nodes by increasing restricted in-degree.
    ///
    /// Intuition
    /// ---------
    /// Nodes with `indeg == 0` (within the component, considering only ACTIVE edges)
    /// are natural chain heads. Sorting by in-degree provides a deterministic
    /// priority among starts.
    fn sorted_starts(component_nodes: &[NodeId], indeg: &AHashMap<NodeId, u32>) -> Vec<NodeId> {
        let mut starts = component_nodes.to_vec();
        starts.sort_by_key(|n| indeg.get(n).copied().unwrap_or(u32::MAX));
        starts
    }

    /// Select the best outgoing ACTIVE edge from a node, constrained to the component.
    ///
    /// The "best" edge is the ACTIVE outgoing edge with minimal cost among those
    /// that:
    /// - end inside the component,
    /// - do not point to a node already in `claimed` (cycle/dup guard).
    ///
    /// Parameters
    /// ----------
    /// graph : &InterNightGraph
    ///     Graph containing adjacency lists and edge metadata.
    /// cur : NodeId
    ///     Current chain tail.
    /// in_comp : &AHashSet<NodeId>
    ///     Component membership set.
    /// claimed : &AHashSet<NodeId>
    ///     Nodes already used by previously produced tracks or earlier in this chain.
    ///
    /// Return
    /// ------
    /// Option<(EdgeId, NodeId, f64)>
    ///     `(edge_id, next_node, edge_cost)` for the selected edge, or `None` if
    ///     no admissible edge exists.
    ///
    /// Notes
    /// -----
    /// - If multiple edges have identical cost, the selected edge follows the
    ///   iteration order of `graph.out_adj[cur]`.
    /// - Adapt the `ecost` extraction if your edge cost field is not `e.cost`.
    fn best_outgoing_active_edge(
        graph: &InterNightGraph,
        cur: NodeId,
        in_comp: &AHashSet<NodeId>,
        claimed: &AHashSet<NodeId>,
    ) -> Option<(EdgeId, NodeId, f64)> {
        graph.out_adj[cur.idx()]
            .iter()
            .copied()
            .filter_map(|eid| {
                let e = &graph.edges[eid.idx() as usize];
                if !e.active {
                    return None;
                }
                if !in_comp.contains(&e.to) {
                    return None;
                }
                if claimed.contains(&e.to) {
                    return None;
                }
                let ecost = e.cost;

                Some((eid, e.to, ecost))
            })
            .min_by(|(eid_a, _, cost_a), (eid_b, _, cost_b)| {
                cost_a
                    .partial_cmp(cost_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| eid_a.cmp(eid_b))
            })
    }

    /// Greedily grow a chain from `start` by repeatedly selecting the best outgoing edge.
    ///
    /// The chain grows until:
    /// - no admissible outgoing edge exists (dead end), or
    /// - [`TrivialSolverConfig::max_nodes`] is reached.
    ///
    /// Parameters
    /// ----------
    /// start : NodeId
    ///     Chain head candidate (must not already be in `claimed`).
    /// claimed : &mut AHashSet<NodeId>
    ///     Mutable set of nodes "claimed" by already-built tracks (and this track).
    ///     This is used to reduce near-duplicates and avoid cycles.
    ///
    /// Return
    /// ------
    /// Option<TrackHypothesis>
    ///     A [`TrackHypothesis`] if the chain length is ≥ [`TrivialSolverConfig::min_nodes`],
    ///     otherwise `None`.
    fn build_track_from_start(
        &self,
        graph: &InterNightGraph,
        start: NodeId,
        in_comp: &AHashSet<NodeId>,
        claimed: &mut AHashSet<NodeId>,
    ) -> Option<TrackHypothesis> {
        // If the start node is already used by another track, skip it.
        if claimed.contains(&start) {
            return None;
        }

        // Pre-allocate to avoid repeated reallocation for short chains.
        let mut chain_nodes: Vec<NodeId> = Vec::with_capacity(self.config.max_nodes);
        let mut chain_edges: Vec<EdgeId> =
            Vec::with_capacity(self.config.max_nodes.saturating_sub(1));
        let mut cost_sum: f64 = 0.0;

        // Claim the start node immediately to prevent duplicates.
        chain_nodes.push(start);
        claimed.insert(start);

        // Greedy forward extension.
        while chain_nodes.len() < self.config.max_nodes {
            let cur = *chain_nodes.last().unwrap();

            let Some((eid, next, ecost)) =
                Self::best_outgoing_active_edge(graph, cur, in_comp, claimed)
            else {
                break; // dead end (no admissible outgoing edge)
            };

            chain_edges.push(eid);
            chain_nodes.push(next);
            claimed.insert(next);
            cost_sum += ecost;
        }

        // Reject too-short chains.
        if chain_nodes.len() < self.config.min_nodes {
            return None;
        }

        // Compute a cheap night span proxy for later prioritization/diagnostics.
        // (NightId is assumed monotonic in the graph.)
        let first_night = graph.nodes[chain_nodes[0].idx()].night.0;
        let last_night = graph.nodes[chain_nodes.last().unwrap().idx()].night.0;
        let night_span = last_night.saturating_sub(first_night);

        Some(TrackHypothesis {
            nodes: chain_nodes,
            edges: chain_edges,
            cost: cost_sum,
            night_span,
            n_nodes: 0, // filled later by `finalize_tracks`
        })
    }

    /// Build tracks by scanning start nodes in priority order.
    ///
    /// This is the main "extraction loop": it tries starts one by one and stops
    /// after producing `max_tracks` hypotheses.
    ///
    /// Notes
    /// -----
    /// The `claimed` set makes this intentionally conservative: it prevents node
    /// reuse across tracks, which reduces near-duplicates but can reduce recall.
    fn build_tracks(
        &self,
        graph: &InterNightGraph,
        starts: &[NodeId],
        in_comp: &AHashSet<NodeId>,
    ) -> Vec<TrackHypothesis> {
        let mut tracks: Vec<TrackHypothesis> = Vec::new();
        let mut claimed: AHashSet<NodeId> = AHashSet::new();

        for &start in starts {
            if tracks.len() >= self.config.max_tracks {
                break;
            }

            if let Some(tr) = self.build_track_from_start(graph, start, in_comp, &mut claimed) {
                tracks.push(tr);
            }
        }

        tracks
    }

    /// Finalize track metadata and enforce solver output invariants.
    ///
    /// Invariants enforced
    /// -------------------
    /// - `track.n_nodes` is filled consistently with `track.nodes.len()`.
    /// - Tracks are sorted best-first by total cost.
    fn finalize_tracks(_graph: &InterNightGraph, tracks: &mut [TrackHypothesis]) {
        tracks.iter_mut().for_each(|tr| {
            tr.n_nodes = tr.nodes.len() as u32;
        });

        tracks.sort_by(|a, b| {
            a.cost
                .partial_cmp(&b.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Build the optional list of edges suggested for immediate deactivation.
    ///
    /// When enabled, this returns the deduplicated union of edges used by all
    /// tracks, sorted by `EdgeId` for stable downstream handling.
    fn propose_deactivations_if_enabled(&self, tracks: &[TrackHypothesis]) -> Vec<EdgeId> {
        if !self.config.propose_deactivations {
            return Vec::new();
        }

        let mut proposed: Vec<EdgeId> = tracks
            .iter()
            .flat_map(|tr| tr.edges.iter().copied())
            .collect();

        proposed.sort();
        proposed.dedup();
        proposed
    }

    /// Build solver diagnostics for monitoring and benchmarking.
    fn build_diagnostics(
        &self,
        stats: ComponentStats,
        n_candidates: usize,
        n_selected: usize,
        time_spent_s: f64,
    ) -> SolverDiagnostics {
        let mut diag = SolverDiagnostics::default();
        diag.solver_name = self.name();
        diag.n_nodes = stats.n_nodes;
        diag.m_active_edges = stats.m_active_edges;
        diag.n_candidates = n_candidates as u32;
        diag.n_selected = n_selected as u32;
        diag.time_spent_s = time_spent_s;
        diag
    }
}

impl Solver for TrivialSolver {
    fn name(&self) -> &'static str {
        "trivial"
    }

    /// Solve a connected component by extracting a few greedy chain hypotheses.
    ///
    /// Parameters
    /// ----------
    /// graph : &InterNightGraph
    ///     Global graph containing nodes and (active/inactive) directed edges.
    /// component_nodes : &[NodeId]
    ///     Nodes belonging to the connected component to solve.
    /// stats : ComponentStats
    ///     Precomputed component-level statistics (nodes, active edges, etc.).
    ///
    /// Return
    /// ------
    /// SolverOutput
    ///     Track hypotheses (sorted by cost), optional deactivation proposals,
    ///     and diagnostics.
    fn solve(
        &self,
        graph: &InterNightGraph,
        component_nodes: &[NodeId],
        stats: ComponentStats,
    ) -> SolverOutput {
        let t0 = Instant::now();

        // 1) Precompute membership for fast "in component?" checks.
        let in_comp = Self::build_membership_set(component_nodes);

        // 2) Rank candidate starts using restricted in-degree on ACTIVE edges.
        let indeg = Self::compute_indeg_active_in_component(graph, component_nodes, &in_comp);
        let starts = Self::sorted_starts(component_nodes, &indeg);

        // 3) Greedily extract a bounded set of short, non-overlapping chains.
        let mut tracks = self.build_tracks(graph, &starts, &in_comp);

        // 4) Fill derived fields and enforce ordering invariants.
        Self::finalize_tracks(graph, &mut tracks);

        // 5) Optional immediate edge deactivation suggestion.
        let proposed_deactivations = self.propose_deactivations_if_enabled(&tracks);

        // 6) Diagnostics (timing + component metadata).
        let diag = self.build_diagnostics(
            stats,
            component_nodes.len(),
            tracks.len(),
            t0.elapsed().as_secs_f64(),
        );

        SolverOutput {
            tracks,
            proposed_deactivations,
            diag,
        }
    }
}
