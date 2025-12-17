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

use crate::graph::{edge::Edge, node_id::NodeId};

/// Disjoint Set Union (Union-Find) with path compression and union by size.
///
/// Notes
/// -----
/// - Indices are `usize` and refer to node indices in `InterNightGraph::nodes`.
/// - This DSU is intended for *undirected* connectivity.
#[derive(Debug, Clone)]
pub struct UnionFind {
    parent: Vec<usize>,
    size: Vec<u32>,
}

impl UnionFind {
    /// Create a DSU over `n` elements: 0..n-1.
    pub fn new(n: usize) -> Self {
        let mut parent = Vec::with_capacity(n);
        let mut size = Vec::with_capacity(n);
        for i in 0..n {
            parent.push(i);
            size.push(1);
        }
        Self { parent, size }
    }

    /// Find the representative (root) of `x` with path compression.
    #[inline]
    pub fn find(&mut self, mut x: usize) -> usize {
        // Iterative path compression (two-pass).
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        while self.parent[x] != x {
            let p = self.parent[x];
            self.parent[x] = root;
            x = p;
        }
        root
    }

    /// Union the sets containing `a` and `b`.
    #[inline]
    pub fn union(&mut self, a: usize, b: usize) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return;
        }

        // Union by size: attach smaller tree under larger tree.
        if self.size[ra] < self.size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
    }
}

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
}
