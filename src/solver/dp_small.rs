// src/solver/dp_small.rs

//! Tiny/path-like component solver using **linear DP on a DAG** with greedy peeling.

use ahash::AHashMap;
use std::collections::HashSet;

use crate::graph::{graph::InterNightGraph, NodeId};

#[inline]
fn out_neighbors_in_alive<'a>(
    g: &'a InterNightGraph,
    alive: &'a HashSet<NodeId>,
    v: NodeId,
) -> impl Iterator<Item = NodeId> + 'a {
    g.out_neighbors(v).filter(move |&u| alive.contains(&u))
}

#[inline]
fn in_deg_in_alive(g: &InterNightGraph, alive: &HashSet<NodeId>, v: NodeId) -> usize {
    g.in_neighbors(v).filter(|&u| alive.contains(&u)).count()
}

/// Return (paths, total_cost) with `paths` as sequences of node ids (time-ordered).
pub fn solve_trivial_dp(
    g: &InterNightGraph,
    comp_nodes: &[NodeId],
    min_obs: usize,
) -> Vec<(Vec<NodeId>, f32)> {
    if comp_nodes.is_empty() {
        return Vec::new();
    }

    // Induced-subgraph bookkeeping
    let set: HashSet<NodeId> = comp_nodes.iter().copied().collect();
    let mut alive: HashSet<NodeId> = set;

    let mut results = Vec::new();

    loop {
        // Topological order by (night, id) is valid since graph is layered forward in time.
        let mut topo = alive.iter().copied().collect::<Vec<_>>();
        topo.sort_by_key(|&v| (g.nodes[v as usize].night, v));

        // --- DP phase (no mutation of `alive`) -----------------------------------------
        let mut best_cost: AHashMap<NodeId, f32> = AHashMap::with_capacity(topo.len());
        let mut next: AHashMap<NodeId, Option<NodeId>> = AHashMap::with_capacity(topo.len());

        // Reverse pass: compute best suffix cost from each v
        for &v in topo.iter().rev() {
            let mut best = f32::INFINITY;
            let mut best_next = None;

            for u in out_neighbors_in_alive(g, &alive, v) {
                // cost(v→u) + best_cost[u]
                // Find edge cost for (v,u)
                if let Some(&eid) = g.out_adj[v as usize]
                    .iter()
                    .find(|&&eid| g.edges[eid as usize].to == u)
                {
                    let tail = *best_cost.get(&u).unwrap_or(&0.0);
                    let c = g.edges[eid as usize].cost + tail;
                    if c < best {
                        best = c;
                        best_next = Some(u);
                    }
                }
            }

            if best.is_finite() {
                best_cost.insert(v, best);
                next.insert(v, best_next);
            } else {
                // No outgoing neighbor in the alive set: suffix cost = 0.
                best_cost.insert(v, 0.0);
                next.insert(v, None);
            }
        }

        // Choose best start among in-degree-zero nodes (computed against `alive`)
        let mut best_start = None;
        let mut best_score = f32::INFINITY;
        for &v in &topo {
            if in_deg_in_alive(g, &alive, v) == 0 {
                if let Some(&c) = best_cost.get(&v) {
                    if c < best_score {
                        best_score = c;
                        best_start = Some(v);
                    }
                }
            }
        }

        // If none found (e.g., exhausted), stop.
        let Some(start) = best_start else {
            break;
        };

        // Extract the path from `start` following `next` (still **no mutation** of `alive`)
        let mut path = Vec::new();
        let mut cost_sum = 0.0f32;
        let mut v = start;
        path.push(v);
        while let Some(&Some(u)) = next.get(&v) {
            // find edge cost v→u (exists by construction)
            if let Some(eid) = g.out_adj[v as usize]
                .iter()
                .find(|&&eid| g.edges[eid as usize].to == u)
                .copied()
            {
                cost_sum += g.edges[eid as usize].cost;
            }
            v = u;
            path.push(v);
        }

        // Enforce minimum number of observations (proxy by node count)
        if path.len() < min_obs {
            // If the best remaining chain is too short, stop peeling.
            break;
        }

        // --- Peeling phase (now we mutate `alive`, after all immutable borrows ended) ---
        for &v in &path {
            alive.remove(&v);
        }
        results.push((path, cost_sum));

        if alive.is_empty() {
            break;
        }
    }

    results
}
