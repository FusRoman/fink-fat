//! Trivial solver for very small connected components.
//!
//! This solver is designed for tiny components where sophisticated solvers
//! (min-cost flow, blob-breaker) would be overkill.
//!
//! Strategy
//! --------
//! - Treat the component as a directed graph restricted to ACTIVE edges.
//! - Identify candidate start nodes (lowest in-degree inside the component).
//! - From each start, greedily follow the best outgoing ACTIVE edge
//!   (lowest edge cost) while:
//!   - staying inside the component,
//!   - avoiding node revisits (cycle guard),
//!   - respecting a max chain length.
//!
//! The output is a small set of deterministic `TrackHypothesis` objects,
//! suitable for immediate downstream IOD validation.

use std::time::Instant;

use ahash::{AHashMap, AHashSet};

use crate::{
    graph::{edge_id::EdgeId, graph::InterNightGraph, node_id::NodeId},
    solver::{ComponentStats, Solver, SolverDiagnostics, SolverOutput},
    trajectory::track_hypothesis::TrackHypothesis,
};

/// Configuration knobs for the trivial solver.
#[derive(Clone, Debug)]
pub struct TrivialSolverConfig {
    /// Maximum number of tracks returned per component.
    pub max_tracks: usize,
    /// Minimum number of nodes for a returned track.
    pub min_nodes: usize,
    /// Maximum number of nodes for a returned track.
    pub max_nodes: usize,
    /// If true, suggest immediate deactivation of edges used by returned tracks.
    pub propose_deactivations: bool,
}

impl Default for TrivialSolverConfig {
    fn default() -> Self {
        Self {
            max_tracks: 8,
            min_nodes: 2,
            max_nodes: 8,
            propose_deactivations: false,
        }
    }
}

/// A tiny greedy solver for small components.
#[derive(Clone, Debug, Default)]
pub struct TrivialSolver {
    pub config: TrivialSolverConfig,
}

impl TrivialSolver {
    pub fn new(config: TrivialSolverConfig) -> Self {
        Self { config }
    }
}

impl Solver for TrivialSolver {
    fn name(&self) -> &'static str {
        "trivial"
    }

    fn solve(
        &self,
        graph: &InterNightGraph,
        component_nodes: &[NodeId],
        stats: ComponentStats,
    ) -> SolverOutput {
        let t0 = Instant::now();

        // Membership set for O(1) "in component?"
        let in_comp: AHashSet<NodeId> = component_nodes.iter().copied().collect();

        // Compute in-degree inside the component using ACTIVE edges.
        let mut indeg: AHashMap<NodeId, u32> = AHashMap::with_capacity(component_nodes.len());
        for &n in component_nodes {
            indeg.insert(n, 0);
        }

        for &n in component_nodes {
            // We rely on graph.in_adj being indexed by node index.
            for &eid in &graph.in_adj[n.idx()] {
                let e = &graph.edges[eid.idx() as usize];
                if !e.active {
                    continue;
                }
                // Only count edges whose endpoints are inside the component.
                if in_comp.contains(&e.from) && in_comp.contains(&e.to) {
                    *indeg.get_mut(&n).unwrap() += 1;
                }
            }
        }

        // Candidate starts: prioritize indeg=0, then low indeg.
        let mut starts: Vec<NodeId> = component_nodes.to_vec();
        starts.sort_by_key(|n| indeg.get(n).copied().unwrap_or(u32::MAX));

        let mut tracks: Vec<TrackHypothesis> = Vec::new();
        let mut proposed_deactivations: Vec<EdgeId> = Vec::new();

        // We avoid producing many near-duplicates by "claiming" nodes once used
        // as part of a track (very conservative).
        let mut claimed: AHashSet<NodeId> = AHashSet::new();

        for &start in &starts {
            if tracks.len() >= self.config.max_tracks {
                break;
            }
            if claimed.contains(&start) {
                continue;
            }

            let mut chain_nodes: Vec<NodeId> = Vec::with_capacity(self.config.max_nodes);
            let mut chain_edges: Vec<EdgeId> =
                Vec::with_capacity(self.config.max_nodes.saturating_sub(1));
            let mut cost_sum: f64 = 0.0;

            chain_nodes.push(start);
            claimed.insert(start);

            while chain_nodes.len() < self.config.max_nodes {
                let cur = *chain_nodes.last().unwrap();

                // Pick best outgoing ACTIVE edge staying in component.
                let mut best: Option<(EdgeId, NodeId, f64)> = None;

                for &eid in &graph.out_adj[cur.idx()] {
                    let e = &graph.edges[eid.idx() as usize];
                    if !e.active {
                        continue;
                    }
                    if !in_comp.contains(&e.to) {
                        continue;
                    }
                    if claimed.contains(&e.to) {
                        continue; // cycle/dup guard in trivial mode
                    }

                    // ---- ADAPT THIS LINE IF YOUR EDGE FIELD IS NOT `cost` ----
                    let ecost = e.cost;

                    match best {
                        None => best = Some((eid, e.to, ecost)),
                        Some((_, _, best_cost)) => {
                            if ecost < best_cost {
                                best = Some((eid, e.to, ecost));
                            }
                        }
                    }
                }

                let Some((eid, next, ecost)) = best else {
                    break; // dead end
                };

                chain_edges.push(eid);
                chain_nodes.push(next);
                claimed.insert(next);
                cost_sum += ecost;
            }

            if chain_nodes.len() < self.config.min_nodes {
                continue;
            }

            // Night span from first/last node (cheap, deterministic).
            let first_night = graph.nodes[chain_nodes[0].idx()].night.0;
            let last_night = graph.nodes[chain_nodes.last().unwrap().idx()].night.0;
            let night_span = last_night.saturating_sub(first_night);

            tracks.push(TrackHypothesis {
                nodes: chain_nodes,
                edges: chain_edges,
                cost: cost_sum,
                night_span,
                n_nodes: 0, // filled below
            });
        }

        for tr in &mut tracks {
            tr.n_nodes = tr.nodes.len() as u32;
        }

        // Optional immediate deactivation suggestion.
        if self.config.propose_deactivations {
            for tr in &tracks {
                proposed_deactivations.extend(tr.edges.iter().copied());
            }
            proposed_deactivations.sort();
            proposed_deactivations.dedup();
        }

        // Sort best-first by cost (nice invariant).
        tracks.sort_by(|a, b| {
            a.cost
                .partial_cmp(&b.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut diag = SolverDiagnostics::default();
        diag.solver_name = self.name();
        diag.n_nodes = stats.n_nodes;
        diag.m_active_edges = stats.m_active_edges;
        diag.n_candidates = component_nodes.len() as u32;
        diag.n_selected = tracks.len() as u32;
        diag.time_spent_s = t0.elapsed().as_secs_f64();

        SolverOutput {
            tracks,
            proposed_deactivations,
            diag,
        }
    }
}
