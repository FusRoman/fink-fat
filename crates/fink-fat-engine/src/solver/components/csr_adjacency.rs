use crate::graph::{graph::InterNightGraph, node_id::NodeId};

/// Simple CSR adjacency (undirected, local, active edges only).
///
/// For a node `u`, its neighbors are in:
///   neighbors[offsets[u] .. offsets[u+1]]
#[derive(Debug)]
pub struct CsrAdj {
    pub offsets: Vec<usize>,
    pub neighbors: Vec<NodeId>,
}

impl CsrAdj {
    /// Build an undirected CSR adjacency restricted to a subset of nodes,
    /// using ACTIVE edges only.
    pub fn build_active_undirected(graph: &InterNightGraph, nodes: &[NodeId]) -> Self {
        let n = graph.nodes.len();

        // Mark nodes in subset for O(1) membership test
        let mut in_subset = vec![false; n];
        for &nid in nodes {
            in_subset[nid.idx()] = true;
        }

        // First pass: count degrees
        let mut degree = vec![0usize; n];

        for &u in nodes {
            let uidx = u.idx();

            // out edges
            for &eid in &graph.out_adj[uidx] {
                let e = &graph.edges[eid.idx()];
                if !e.active {
                    continue;
                }
                let v = e.to;
                if in_subset[v.idx()] {
                    degree[uidx] += 1;
                }
            }

            // in edges (undirected view)
            for &eid in &graph.in_adj[uidx] {
                let e = &graph.edges[eid.idx()];
                if !e.active {
                    continue;
                }
                let v = e.from;
                if in_subset[v.idx()] {
                    degree[uidx] += 1;
                }
            }
        }

        // Build offsets
        let mut offsets = vec![0usize; n + 1];
        for i in 0..n {
            offsets[i + 1] = offsets[i] + degree[i];
        }

        // Allocate neighbors
        let mut neighbors = vec![NodeId::from(0u64); offsets[n]];
        let mut cursor = offsets.clone();

        // Second pass: fill neighbors
        for &u in nodes {
            let uidx = u.idx();

            for &eid in &graph.out_adj[uidx] {
                let e = &graph.edges[eid.idx()];
                if !e.active {
                    continue;
                }
                let v = e.to;
                if in_subset[v.idx()] {
                    let pos = cursor[uidx];
                    neighbors[pos] = v;
                    cursor[uidx] += 1;
                }
            }

            for &eid in &graph.in_adj[uidx] {
                let e = &graph.edges[eid.idx()];
                if !e.active {
                    continue;
                }
                let v = e.from;
                if in_subset[v.idx()] {
                    let pos = cursor[uidx];
                    neighbors[pos] = v;
                    cursor[uidx] += 1;
                }
            }
        }

        Self { offsets, neighbors }
    }

    #[inline]
    pub fn neighbors(&self, u: NodeId) -> &[NodeId] {
        let i = u.idx();
        &self.neighbors[self.offsets[i]..self.offsets[i + 1]]
    }
}
