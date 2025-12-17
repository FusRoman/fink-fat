//! Connected components utilities (undirected view) for the inter-night graph.
//!
//! This module is used as a fast pre-filter for downstream solvers:
//! - tiny components: can be handled trivially,
//! - medium components: can be sent to min-cost flow,
//! - huge components: can trigger a blob-breaker strategy.
//!
//! The graph is directed, but for connected components we treat edges as
//! undirected links between nodes.

use ahash::AHashMap;

use crate::{
    graph::{edge::Edge, graph::InterNightGraph, node_id::NodeId},
    solver::{UnionFind, csr_adjacency::CsrAdj},
};

/// Connected components result (undirected view).
///
/// Fields
/// ------
/// components
///     List of components, each as a list of node ids.
/// comp_of_node
///     For each node index `i`, the component id in `components`.
/// sizes
///     Size (number of nodes) of each component in `components`.
#[derive(Debug, Clone)]
pub struct ConnectedComponents {
    pub components: Vec<Vec<NodeId>>,
    pub comp_of_node: Vec<u32>,
    pub sizes: Vec<u32>,
}

impl ConnectedComponents {
    /// Compute connected components from node count and directed edges.
    ///
    /// Notes
    /// -----
    /// We treat each directed edge `u -> v` as an undirected link `{u, v}`.
    pub fn compute(n_nodes: usize, edges: &[Edge]) -> Self {
        let mut uf = UnionFind::new(n_nodes);

        // 1) Union all endpoints (undirected connectivity).
        for e in edges {
            uf.union(e.from.idx(), e.to.idx());
        }

        // 2) Assign compact component ids 0..C-1.
        let mut root_to_comp: AHashMap<usize, u32> = AHashMap::default();
        let mut comp_of_node: Vec<u32> = vec![0; n_nodes];
        let mut components: Vec<Vec<NodeId>> = Vec::new();

        for i in 0..n_nodes {
            let r = uf.find(i);
            let cid = *root_to_comp.entry(r).or_insert_with(|| {
                let new_id = components.len() as u32;
                components.push(Vec::new());
                new_id
            });

            comp_of_node[i] = cid;
            components[cid as usize].push(NodeId::from(i as u64));
        }

        // 3) Sizes (for routing decisions).
        let mut sizes = Vec::with_capacity(components.len());
        for c in &components {
            sizes.push(c.len() as u32);
        }

        Self {
            components,
            comp_of_node,
            sizes,
        }
    }

    /// Build components from an existing Union-Find (union-only, may be coarse after deletions).
    pub fn from_union_find(n_nodes: usize, uf: &mut UnionFind) -> Self {
        debug_assert_eq!(
            n_nodes,
            uf.len(),
            "UnionFind size must match graph node count"
        );

        let mut root_to_comp: AHashMap<usize, u32> = AHashMap::default();
        let mut comp_of_node: Vec<u32> = vec![0; n_nodes];
        let mut components: Vec<Vec<NodeId>> = Vec::new();

        for i in 0..n_nodes {
            let r = uf.find(i);
            let cid = *root_to_comp.entry(r).or_insert_with(|| {
                let new_id = components.len() as u32;
                components.push(Vec::new());
                new_id
            });

            comp_of_node[i] = cid;
            components[cid as usize].push(NodeId::from(i as u64));
        }

        let mut sizes = Vec::with_capacity(components.len());
        for c in &components {
            sizes.push(c.len() as u32);
        }

        Self {
            components,
            comp_of_node,
            sizes,
        }
    }

    /// Recompute exact connected components on a subset of nodes, using only ACTIVE edges.
    ///
    /// Parameters
    /// ----------
    /// graph
    ///     Full graph storage (nodes/edges + adjacency).
    /// nodes
    ///     Subset of nodes to consider (typically: nodes in one coarse DSU component
    ///     or impacted region after deactivations).
    ///
    /// Returns
    /// -------
    /// Vec<Vec<NodeId>>
    ///     Exact components within `nodes` (undirected view, ACTIVE edges only).
    pub fn recompute_local_exact(graph: &InterNightGraph, nodes: &[NodeId]) -> Vec<Vec<NodeId>> {
        if nodes.is_empty() {
            return Vec::new();
        }

        let csr = CsrAdj::build_active_undirected(graph, nodes);

        let mut visited = vec![false; graph.nodes.len()];
        let mut out: Vec<Vec<NodeId>> = Vec::new();
        let mut stack: Vec<NodeId> = Vec::new();

        for &start in nodes {
            if visited[start.idx()] {
                continue;
            }

            let mut comp = Vec::new();
            visited[start.idx()] = true;
            stack.push(start);

            while let Some(u) = stack.pop() {
                comp.push(u);

                for &v in csr.neighbors(u) {
                    if !visited[v.idx()] {
                        visited[v.idx()] = true;
                        stack.push(v);
                    }
                }
            }

            out.push(comp);
        }

        out
    }
}
