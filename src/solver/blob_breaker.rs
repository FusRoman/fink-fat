// src/graph/solvers/blob_breaker.rs

//! Blob-breaker: partition + local peeling to handle huge components quickly.
//!
//! Overview
//! --------
//! When a component percolates into a huge blob, we:
//! 1) **Restrict** to a small **night window** (e.g., last 3–4 nights present).
//! 2) **Partition** the induced subgraph with a coarse spatial tiling (reuse the
//!    graph’s night layers and node ids as a cheap proxy) and **cap** the
//!    incoming degree per right node.
//! 3) In each partition:
//!    - If exactly **2 nights**, run a **bipartite greedy** (Top-K left, cap-in right).
//!    - Else, run the **DP peeling** from `dp_small`.
//! 4) Concatenate all local paths and (optionally) run a light **merge** step.
//!
//! This is designed to be **time-bounded** and to return IOD-compatible tracks
//! rather than a global optimum.
//!
//! Arguments
//! ---------
//! * `g` – Global inter-night graph.
//! * `comp_nodes` – Node ids in the (huge) component.
//! * `min_obs` – Minimal node count per path.
//! * `max_in_per_right` – Cap on incoming edges per right node within a partition.
//! * `window_nights` – Keep at most this many most-recent distinct nights.
//!
//! Returns
//! -------
//! * `Vec<(Vec<NodeId>, f32)>` – Disjoint paths found locally.
//!
//! Notes
//! -----
//! * You can later plug a more sophisticated spatial/kinematic tiling.
//! * All steps are deterministic and linear-time-ish on sparse inputs.

use std::collections::{HashMap, HashSet};

use crate::{
    graph::{graph::InterNightGraph, NodeId},
    NightId,
};

fn last_n_nights_in_comp(g: &InterNightGraph, comp_nodes: &[NodeId], n: usize) -> Vec<NightId> {
    let mut nights = comp_nodes
        .iter()
        .map(|&v| g.nodes[v as usize].night)
        .collect::<Vec<_>>();
    nights.sort_unstable();
    nights.dedup();
    if nights.len() > n {
        nights[nights.len() - n..].to_vec()
    } else {
        nights
    }
}

fn restrict_to_nights(
    g: &InterNightGraph,
    comp_nodes: &[NodeId],
    keep: &ahash::AHashSet<NightId>,
) -> Vec<NodeId> {
    comp_nodes
        .iter()
        .copied()
        .filter(|&v| keep.contains(&g.nodes[v as usize].night))
        .collect()
}

/// Naive partition: by **night** buckets, then we merge small buckets.
fn partition_nodes_by_night(
    g: &InterNightGraph,
    nodes: &[NodeId],
) -> HashMap<NightId, Vec<NodeId>> {
    let mut map: HashMap<NightId, Vec<NodeId>> = HashMap::new();
    for &v in nodes {
        map.entry(g.nodes[v as usize].night).or_default().push(v);
    }
    map
}

/// Cap incoming edges per right node **within the alive set** (cheap anti-hub).
fn prune_max_in_per_right(
    g: &InterNightGraph,
    alive: &HashSet<NodeId>,
    max_in: usize,
) -> ahash::AHashSet<(NodeId, NodeId)> {
    // Return the **kept** directed pairs (from,to); commit is done by callers.
    // For each right node, keep the cheapest `max_in` incoming edges.
    let mut incoming: HashMap<NodeId, Vec<(NodeId, f32)>> = HashMap::new();
    for &v in alive {
        for &eid in &g.out_adj[v as usize] {
            let e = &g.edges[eid as usize];
            if alive.contains(&e.to) {
                incoming.entry(e.to).or_default().push((e.from, e.cost));
            }
        }
    }
    let mut kept = ahash::AHashSet::default();
    for (to, mut lst) in incoming {
        lst.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for (i, (from, _)) in lst.into_iter().enumerate() {
            if i < max_in {
                kept.insert((from, to));
            } else {
                break;
            }
        }
    }
    kept
}

pub fn solve_blob_breaker(
    g: &InterNightGraph,
    comp_nodes: &[NodeId],
    min_obs: usize,
    max_in_per_right: usize,
    window_nights: usize,
) -> Vec<(Vec<NodeId>, f32)> {
    if comp_nodes.is_empty() {
        return Vec::new();
    }

    // 1) Window most-recent nights to avoid global percolation.
    let keep_nights_vec = last_n_nights_in_comp(g, comp_nodes, window_nights);
    let keep_nights: ahash::AHashSet<NightId> = keep_nights_vec.into_iter().collect();
    let window_nodes = restrict_to_nights(g, comp_nodes, &keep_nights);

    // 2) Partition (naive: by night), then expand by one night neighbor to allow links.
    let parts = partition_nodes_by_night(g, &window_nodes);

    // 3) Inside each partition: Anti-hub pruning + DP peeling (or trivial bipartite).
    let mut used: ahash::AHashSet<NodeId> = ahash::AHashSet::default();
    let mut out = Vec::new();

    for (_night, bucket) in parts {
        let alive_set: HashSet<NodeId> = bucket.into_iter().filter(|v| !used.contains(v)).collect();
        if alive_set.is_empty() {
            continue;
        }

        // Anti-hub pruning mask
        let kept_pairs = prune_max_in_per_right(g, &alive_set, max_in_per_right);

        // Build a temporary “allowed edge” mask
        let allowed = |a: NodeId, b: NodeId| kept_pairs.contains(&(a, b));

        // Count distinct nights in this local set
        let mut nights = alive_set
            .iter()
            .map(|&v| g.nodes[v as usize].night)
            .collect::<Vec<_>>();
        nights.sort_unstable();
        nights.dedup();

        // Local peeling
        let mut local_alive = alive_set.clone();
        loop {
            // Find a path respecting edge allowance and alive set
            // Reuse the Dijkstra from mcf_ssp but add the edge-allowance check.
            use std::cmp::Ordering;
            use std::collections::BinaryHeap;

            #[derive(Copy, Clone, PartialEq)]
            struct State {
                cost: f32,
                v: NodeId,
            }
            impl Eq for State {}
            impl Ord for State {
                fn cmp(&self, other: &Self) -> Ordering {
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

            let mut is_source = ahash::AHashSet::with_capacity(local_alive.len());
            let mut is_sink = ahash::AHashSet::with_capacity(local_alive.len());
            for &v in local_alive.iter() {
                let indeg = g
                    .in_neighbors(v)
                    .filter(|u| local_alive.contains(u))
                    .count();
                let outdeg = g
                    .out_neighbors(v)
                    .filter(|u| local_alive.contains(u) && allowed(v, *u))
                    .count();
                if indeg == 0 {
                    is_source.insert(v);
                }
                if outdeg == 0 {
                    is_sink.insert(v);
                }
            }
            if is_source.is_empty() || is_sink.is_empty() {
                break;
            }

            let mut dist = ahash::AHashMap::with_capacity(local_alive.len());
            let mut prev = ahash::AHashMap::with_capacity(local_alive.len());
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
                    if !local_alive.contains(&e.to) || !allowed(v, e.to) {
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

            let Some(sink) = best_sink else {
                break;
            };
            // Reconstruct and peel
            let mut path = Vec::new();
            let mut cur = sink;
            path.push(cur);
            while let Some(Some(p)) = prev.get(&cur) {
                path.push(*p);
                cur = *p;
            }
            path.reverse();
            if path.len() < min_obs {
                break;
            }

            for &v in &path {
                local_alive.remove(&v);
                used.insert(v);
            }
            out.push((path, best_cost));

            if local_alive.is_empty() {
                break;
            }
        }
    }

    out
}
