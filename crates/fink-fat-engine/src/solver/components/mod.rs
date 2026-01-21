// //! Connected components utilities (undirected view) for the inter-night graph.
// //!
// //! This module is used as a fast pre-filter for downstream solvers:
// //! - tiny components: can be handled trivially,
// //! - medium components: can be sent to min-cost flow,
// //! - huge components: can trigger a blob-breaker strategy.
// //!
// //! The graph is directed, but for connected components we treat edges as
// //! undirected links between nodes.

// pub mod csr_adjacency;
// pub mod union_find;

// use ahash::AHashMap;

// use crate::{
//     graph::{edge::Edge, graph::InterNightGraph, node_id::NodeId},
//     // solver::components::{csr_adjacency::CsrAdj, union_find::UnionFind},
// };

// /// Connected components result (undirected view).
// ///
// /// Fields
// /// ------
// /// components
// ///     List of components, each as a list of node ids.
// /// comp_of_node
// ///     For each node index `i`, the component id in `components`.
// /// sizes
// ///     Size (number of nodes) of each component in `components`.
// #[derive(Debug, Clone)]
// pub struct ConnectedComponents {
//     pub components: Vec<Vec<NodeId>>,
//     pub comp_of_node: Vec<u32>,
//     pub sizes: Vec<u32>,
// }

// /// Per-component statistics used for solver routing and diagnostics.
// ///
// /// This module is solver-agnostic: it only describes structural properties
// /// of connected components in the inter-night graph.
// #[derive(Copy, Clone, Debug, Default)]
// pub struct ComponentStats {
//     /// Number of nodes in the component.
//     pub n_nodes: u32,
//     /// Number of ACTIVE directed edges internal to the component.
//     pub m_active_edges: u32,
//     /// Night span = max(night) - min(night).
//     pub night_span: u32,
// }

// impl ConnectedComponents {
//     /// Compute connected components from node count and directed edges.
//     ///
//     /// Notes
//     /// -----
//     /// We treat each directed edge `u -> v` as an undirected link `{u, v}`.
//     pub fn compute(n_nodes: usize, edges: &[Edge]) -> Self {
//         let mut uf = UnionFind::new(n_nodes);

//         // 1) Union all endpoints (undirected connectivity).
//         for e in edges {
//             uf.union(e.from.seed_id.idx(), e.to.seed_id.idx());
//         }

//         // 2) Assign compact component ids 0..C-1.
//         let mut root_to_comp: AHashMap<usize, u32> = AHashMap::default();
//         let mut comp_of_node: Vec<u32> = vec![0; n_nodes];
//         let mut components: Vec<Vec<NodeId>> = Vec::new();

//         for i in 0..n_nodes {
//             let r = uf.find(i);
//             let cid = *root_to_comp.entry(r).or_insert_with(|| {
//                 let new_id = components.len() as u32;
//                 components.push(Vec::new());
//                 new_id
//             });

//             comp_of_node[i] = cid;
//             components[cid as usize].push(NodeId::from(i as u64));
//         }

//         // 3) Sizes (for routing decisions).
//         let mut sizes = Vec::with_capacity(components.len());
//         for c in &components {
//             sizes.push(c.len() as u32);
//         }

//         Self {
//             components,
//             comp_of_node,
//             sizes,
//         }
//     }

//     /// Build components from an existing Union-Find (union-only, may be coarse after deletions).
//     pub fn from_union_find(n_nodes: usize, uf: &mut UnionFind) -> Self {
//         debug_assert_eq!(
//             n_nodes,
//             uf.len(),
//             "UnionFind size must match graph node count"
//         );

//         let mut root_to_comp: AHashMap<usize, u32> = AHashMap::default();
//         let mut comp_of_node: Vec<u32> = vec![0; n_nodes];
//         let mut components: Vec<Vec<NodeId>> = Vec::new();

//         for i in 0..n_nodes {
//             let r = uf.find(i);
//             let cid = *root_to_comp.entry(r).or_insert_with(|| {
//                 let new_id = components.len() as u32;
//                 components.push(Vec::new());
//                 new_id
//             });

//             comp_of_node[i] = cid;
//             components[cid as usize].push(NodeId::from(i as u64));
//         }

//         let mut sizes = Vec::with_capacity(components.len());
//         for c in &components {
//             sizes.push(c.len() as u32);
//         }

//         Self {
//             components,
//             comp_of_node,
//             sizes,
//         }
//     }

//     /// Recompute exact connected components on a subset of nodes, using only ACTIVE edges.
//     ///
//     /// Parameters
//     /// ----------
//     /// graph
//     ///     Full graph storage (nodes/edges + adjacency).
//     /// nodes
//     ///     Subset of nodes to consider (typically: nodes in one coarse DSU component
//     ///     or impacted region after deactivations).
//     ///
//     /// Returns
//     /// -------
//     /// Vec<Vec<NodeId>>
//     ///     Exact components within `nodes` (undirected view, ACTIVE edges only).
//     pub fn recompute_local_exact(graph: &InterNightGraph, nodes: &[NodeId]) -> Vec<Vec<NodeId>> {
//         if nodes.is_empty() {
//             return Vec::new();
//         }

//         let csr = CsrAdj::build_active_undirected(graph, nodes);

//         let mut visited = vec![false; graph.nodes.len()];
//         let mut out: Vec<Vec<NodeId>> = Vec::new();
//         let mut stack: Vec<NodeId> = Vec::new();

//         for &start in nodes {
//             if visited[start.idx()] {
//                 continue;
//             }

//             let mut comp = Vec::new();
//             visited[start.idx()] = true;
//             stack.push(start);

//             while let Some(u) = stack.pop() {
//                 comp.push(u);

//                 for &v in csr.neighbors(u) {
//                     if !visited[v.idx()] {
//                         visited[v.idx()] = true;
//                         stack.push(v);
//                     }
//                 }
//             }

//             out.push(comp);
//         }

//         out
//     }

//     /// Compute cheap, solver-agnostic statistics for each component.
//     pub fn compute_stats(&self, graph: &InterNightGraph) -> Vec<ComponentStats> {
//         let n_comp = self.components.len();
//         let mut stats = vec![ComponentStats::default(); n_comp];

//         // Node counts + night span
//         for (cid, nodes) in self.components.iter().enumerate() {
//             let mut min_night = u32::MAX;
//             let mut max_night = 0u32;

//             for &nid in nodes {
//                 let night = graph.nodes[nid.idx()].night.0;
//                 min_night = min_night.min(night);
//                 max_night = max_night.max(night);
//             }

//             stats[cid].n_nodes = nodes.len() as u32;
//             stats[cid].night_span = if min_night == u32::MAX {
//                 0
//             } else {
//                 max_night - min_night
//             };
//         }

//         // Active internal edges
//         for e in &graph.edges {
//             if !e.active {
//                 continue;
//             }
//             let cu = self.comp_of_node[e.from.idx()] as usize;
//             let cv = self.comp_of_node[e.to.idx()] as usize;
//             if cu == cv {
//                 stats[cu].m_active_edges += 1;
//             }
//         }

//         stats
//     }
// }
