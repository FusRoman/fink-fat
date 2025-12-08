// src/graph/solvers/mcf_ssp.rs

//! Multi-night solver using a **Simplified Successive Shortest Path** (SSP).
//!
//! Overview
//! --------
//! We want a set of **vertex-disjoint** time-respecting paths of minimal total
//! cost in the component. A full-blown min-cost flow (with node-splitting,
//! source/sink, potentials, etc.) is ideal, but a simpler and robust approach is:
//!
//! 1. Build the induced DAG (component).
//! 2. Repeatedly find a **shortest path** from any in-degree-zero node to any
//!    sink (out-degree-zero) using non-negative edge costs.
//! 3. Enforce *exclusivity* by removing the selected path's nodes ("peeling").
//! 4. Stop when no path respecting `min_obs` remains.
//!
//! This approximates a 1-capacity node-constrained MCF. On LSST-like sparse
//! graphs, it is often close to the true MCF while being much simpler to debug.
//!
//! Arguments
//! ---------
//! * `g` – Global inter-night graph.
//! * `comp_nodes` – Node ids in the component (induced subgraph).
//! * `min_obs` – Minimal number of nodes per path (proxy for ≥3 observations).
//!
//! Returns
//! -------
//! * `Vec<(Vec<NodeId>, f32)>` – Disjoint paths and their cumulative edge cost.
//!
//! Complexity
//! ----------
//! * Each iteration runs a Dijkstra on a sparse DAG: ~**O(E log V)** per path.
//!
//! Notes
//! -----
//! * You can later swap the inner shortest-path with a **potential-based SSP**
//!   if you need exact MCF with better scaling.
//! * Costs are expected **non-negative** and strictly **> 0** per edge.
//!
//! See also
//! --------
//! * [`crate::graph::solvers::dp_small`] – faster on path-like tiny components.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use crate::graph::{graph::InterNightGraph, NodeId};

/// Shortest path from any current source (in-degree=0 in alive set) to any sink.
fn shortest_path_on_alive(
    g: &InterNightGraph,
    alive: &HashSet<NodeId>,
) -> Option<(Vec<NodeId>, f32)> {
    // Collect current sources and sinks within the alive induced subgraph.
    let mut is_source = ahash::AHashSet::with_capacity(alive.len());
    let mut is_sink = ahash::AHashSet::with_capacity(alive.len());
    for &v in alive.iter() {
        let indeg = g.in_neighbors(v).filter(|u| alive.contains(u)).count();
        let outdeg = g.out_neighbors(v).filter(|u| alive.contains(u)).count();
        if indeg == 0 {
            is_source.insert(v);
        }
        if outdeg == 0 {
            is_sink.insert(v);
        }
    }
    if is_source.is_empty() || is_sink.is_empty() {
        return None;
    }

    // Multi-source Dijkstra (non-negative costs).
    #[derive(Copy, Clone, PartialEq)]
    struct State {
        cost: f32,
        v: NodeId,
    }
    impl Eq for State {}
    impl Ord for State {
        fn cmp(&self, other: &Self) -> Ordering {
            // reverse for min-heap behavior
            other
                .cost
                .partial_cmp(&self.cost)
                .unwrap_or(Ordering::Equal)
        }
    }
    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut dist = ahash::AHashMap::with_capacity(alive.len());
    let mut prev = ahash::AHashMap::with_capacity(alive.len());
    let mut pq = BinaryHeap::new();

    for &s in is_source.iter() {
        dist.insert(s, 0.0f32);
        pq.push(State { cost: 0.0, v: s });
        prev.insert(s, None::<NodeId>);
    }

    let mut best_sink = None;
    let mut best_cost = f32::INFINITY;

    while let Some(State { cost, v }) = pq.pop() {
        if cost > *dist.get(&v).unwrap_or(&f32::INFINITY) {
            continue;
        }
        if is_sink.contains(&v) {
            best_sink = Some(v);
            best_cost = cost;
            break;
        }
        for &eid in &g.out_adj[v as usize] {
            let e = &g.edges[eid as usize];
            if !alive.contains(&e.to) {
                continue;
            }
            let nd = cost + e.cost;
            if nd < *dist.get(&e.to).unwrap_or(&f32::INFINITY) {
                dist.insert(e.to, nd);
                prev.insert(e.to, Some(v));
                pq.push(State { cost: nd, v: e.to });
            }
        }
    }

    let sink = best_sink?;
    let mut path = Vec::new();
    let mut cur = sink;
    path.push(cur);
    while let Some(Some(p)) = prev.get(&cur) {
        path.push(*p);
        cur = *p;
    }
    path.reverse();
    Some((path, best_cost))
}

pub fn solve_mcf_ssp(
    g: &InterNightGraph,
    comp_nodes: &[NodeId],
    min_obs: usize,
) -> Vec<(Vec<NodeId>, f32)> {
    if comp_nodes.is_empty() {
        return Vec::new();
    }
    let mut alive: HashSet<NodeId> = comp_nodes.iter().copied().collect();
    let mut out = Vec::new();

    loop {
        let Some((path, cost)) = shortest_path_on_alive(g, &alive) else {
            break;
        };
        if path.len() < min_obs {
            // No sufficiently long path remains under the current sparsity.
            break;
        }
        for &v in &path {
            alive.remove(&v);
        }
        out.push((path, cost));
        if alive.is_empty() {
            break;
        }
    }

    out
}
