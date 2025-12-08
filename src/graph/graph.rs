//! Incremental inter-night layered graph with horizon checks.

use std::ops::Range;

use ahash::AHashMap;

use crate::{
    graph::{EdgeId, Horizon, NodeId},
    propagation::features::SeedId,
    NightId,
};

use super::{edge::Edge, layer::NightLayer, node::Node};

/// A grow-only, layered graph of hypothesis links across nights.
///
/// Design
/// ------
/// - Nodes are appended in **night order** (strictly increasing `night`).
/// - Edges always go **forward** in time and must respect the **horizon**.
/// - We maintain CSR-like adjacency lists for fast neighborhood access.
///
/// See also
/// --------
/// - [`components::ConnectedComponents`] for per-component processing.
#[derive(Debug)]
pub struct InterNightGraph {
    /// Horizon constraint for edges.
    pub horizon: Horizon,
    /// All nodes, grouped by layers (one per night).
    pub nodes: Vec<Node>,
    /// All edges (forward in time).
    pub edges: Vec<Edge>,
    /// Index: node -> outgoing edge ids.
    pub out_adj: Vec<Vec<EdgeId>>,
    /// Index: node -> incoming edge ids.
    pub in_adj: Vec<Vec<EdgeId>>,
    /// Layers in strictly increasing `night` order.
    pub layers: Vec<NightLayer>,
    /// Night → layer index for quick resolution.
    night_to_layer_idx: AHashMap<NightId, usize>,
}

impl InterNightGraph {
    /// Create an empty layered graph with a given horizon (in nights).
    pub fn new(horizon: Horizon) -> Self {
        Self {
            horizon,
            nodes: Vec::new(),
            edges: Vec::new(),
            out_adj: Vec::new(),
            in_adj: Vec::new(),
            layers: Vec::new(),
            night_to_layer_idx: AHashMap::default(),
        }
    }

    /// Get nodes by their ids.
    ///
    /// Parameters
    /// ----------
    /// * `node_ids` – slice of `NodeId` to retrieve.
    ///
    /// Returns
    /// -------
    /// Vector of references to `Node` in the same order as `node_ids`.
    pub fn get_nodes_by_ids(&self, node_ids: &[NodeId]) -> Vec<&Node> {
        node_ids
            .iter()
            .map(|&nid| &self.nodes[nid as usize])
            .collect()
    }

    /// Append a **new night layer** with a batch of seeds.
    ///
    /// Parameters
    /// ----------
    /// * `night` – strictly greater than the previous layer's night.
    /// * `seeds` – list of `SeedId` for that night; one node per seed.
    pub fn add_night_layer(&mut self, night: NightId, seeds: &[SeedId]) {
        if let Some(last) = self.layers.last() {
            assert!(night > last.night, "Night must be strictly increasing.");
        }

        let start: NodeId = self.nodes.len() as u32;
        let mut layer = NightLayer::new(night, start..start + seeds.len() as u32);

        println!(
            "Adding night layer for night {} with {} seeds",
            night,
            seeds.len()
        );

        for (i, &seed) in seeds.iter().enumerate() {
            let nid = start + i as u32;
            self.nodes.push(Node::new(nid, night, seed));
            self.out_adj.push(Vec::new());
            self.in_adj.push(Vec::new());
            layer.seed_to_node.insert(seed, nid);
        }

        println!("  added {} nodes", seeds.len());

        let idx = self.layers.len();
        self.night_to_layer_idx.insert(night, idx);
        self.layers.push(layer);
    }

    /// Resolve a `(night, seed)` into a `NodeId` if it exists in the graph.
    #[inline]
    pub fn node_id_of(&self, night: NightId, seed: SeedId) -> Option<NodeId> {
        let &lidx = self.night_to_layer_idx.get(&night)?;
        self.layers[lidx].seed_to_node.get(&seed).copied()
    }

    /// Add a **forward** edge with horizon check, referencing nodes by id.
    pub fn add_edge(&mut self, from: NodeId, to: NodeId, cost: f32, dt_days: f32) -> EdgeId {
        let nf = self.nodes[from as usize].night;
        let nt = self.nodes[to as usize].night;
        assert!(nt > nf, "Edges must go forward in time.");
        assert!(
            self.horizon.within(nf, nt),
            "Edge exceeds horizon: from night {nf} to {nt}."
        );

        let eid = self.edges.len() as u32;
        self.edges.push(Edge::new(eid, from, to, cost, dt_days));
        self.out_adj[from as usize].push(eid);
        self.in_adj[to as usize].push(eid);
        eid
    }

    /// Convenience: add an edge by `(night, seed)` handles.
    pub fn add_edge_by_seed(
        &mut self,
        from_night: NightId,
        from_seed: SeedId,
        to_night: NightId,
        to_seed: SeedId,
        cost: f32,
        dt_days: f32,
    ) -> Option<EdgeId> {
        let from = self.node_id_of(from_night, from_seed)?;
        let to = self.node_id_of(to_night, to_seed)?;
        Some(self.add_edge(from, to, cost, dt_days))
    }

    /// Return the **directed** out-neighbors of a node.
    #[inline]
    pub fn out_neighbors(&self, v: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.out_adj[v as usize]
            .iter()
            .map(|&e| self.edges[e as usize].to)
    }

    /// Return the **directed** in-neighbors of a node.
    #[inline]
    pub fn in_neighbors(&self, v: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.in_adj[v as usize]
            .iter()
            .map(|&e| self.edges[e as usize].from)
    }

    /// Get the node range for a given night, if present.
    pub fn layer_range(&self, night: NightId) -> Option<Range<NodeId>> {
        let &idx = self.night_to_layer_idx.get(&night)?;
        Some(self.layers[idx].node_range.clone())
    }

    /// Current min night and max night present in the graph.
    pub fn night_span(&self) -> Option<(NightId, NightId)> {
        if self.layers.is_empty() {
            return None;
        }
        Some((
            self.layers.first().unwrap().night,
            self.layers.last().unwrap().night,
        ))
    }

    /// Evict layers strictly older than `min_keep_night`, compacting ids in-place.
    ///
    /// Notes
    /// -----
    /// - This performs an **in-place compaction** and rebuilds all indices.
    /// - Call **between** nightly ingestions to bound memory usage.
    pub fn evict_older_than(&mut self, min_keep_night: NightId) {
        let keep_from = self.find_keep_from(min_keep_night);

        // Nothing to do (already keeping everything)
        if keep_from == 0 {
            return;
        }
        // Everything is older than the threshold → drop all.
        if keep_from >= self.layers.len() {
            self.drop_everything();
            return;
        }

        let kept_node_range = self.compute_kept_node_range(keep_from);

        // 1) Remap nodes (build new nodes + adj lists, and an old→new map).
        let (new_nodes, mut new_out, mut new_in, old_to_new_node) =
            self.remap_nodes(&kept_node_range);

        // 2) Remap edges that survive (both endpoints kept).
        let new_edges = self.remap_edges(&old_to_new_node, &mut new_out, &mut new_in);

        // 3) Rebuild layers and the night index.
        let (new_layers, night_to_layer_idx) =
            self.rebuild_layers_and_index(keep_from, &kept_node_range, &old_to_new_node);

        // 4) Swap in the compacted state.
        self.nodes = new_nodes;
        self.edges = new_edges;
        self.out_adj = new_out;
        self.in_adj = new_in;
        self.layers = new_layers;
        self.night_to_layer_idx = night_to_layer_idx;
    }

    /* ------------------------------ helpers -------------------------------- */

    /// Find the first layer index to keep (first with `night >= min_keep_night`).
    #[inline]
    fn find_keep_from(&self, min_keep_night: NightId) -> usize {
        self.layers
            .iter()
            .position(|l| l.night >= min_keep_night)
            .unwrap_or(self.layers.len())
    }

    /// Drop all graph content (used when everything is older than the threshold).
    #[inline]
    fn drop_everything(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.out_adj.clear();
        self.in_adj.clear();
        self.layers.clear();
        self.night_to_layer_idx.clear();
    }

    /// Compute the **global** node index range to keep, based on the first kept layer.
    #[inline]
    fn compute_kept_node_range(&self, keep_from: usize) -> Range<usize> {
        self.layers[keep_from].node_range.start as usize..self.nodes.len()
    }

    /// Build compacted node arrays and an old→new node id mapping for the kept range.
    ///
    /// Return
    /// ------
    /// - `new_nodes`: compacted node vector with renumbered `NodeId`.
    /// - `new_out`, `new_in`: fresh adj lists sized for the kept nodes.
    /// - `old_to_new_node`: map from old global node idx → new NodeId (or None if evicted).
    fn remap_nodes(
        &self,
        kept_node_range: &Range<usize>,
    ) -> (
        Vec<Node>,
        Vec<Vec<EdgeId>>,
        Vec<Vec<EdgeId>>,
        Vec<Option<NodeId>>,
    ) {
        let mut old_to_new_node: Vec<Option<NodeId>> = vec![None; self.nodes.len()];

        let mut new_nodes = Vec::with_capacity(kept_node_range.len());
        let mut new_out: Vec<Vec<EdgeId>> = Vec::with_capacity(kept_node_range.len());
        let mut new_in: Vec<Vec<EdgeId>> = Vec::with_capacity(kept_node_range.len());

        for old_nid in kept_node_range.clone() {
            let new_nid = new_nodes.len() as u32;
            old_to_new_node[old_nid] = Some(new_nid);

            let mut n = self.nodes[old_nid].clone();
            n.id = new_nid;

            new_nodes.push(n);
            new_out.push(Vec::new());
            new_in.push(Vec::new());
        }

        (new_nodes, new_out, new_in, old_to_new_node)
    }

    /// Rebuild edges whose endpoints are both kept, updating the new adjacency lists.
    fn remap_edges(
        &self,
        old_to_new_node: &[Option<NodeId>],
        new_out: &mut [Vec<EdgeId>],
        new_in: &mut [Vec<EdgeId>],
    ) -> Vec<Edge> {
        let mut new_edges = Vec::new();

        for e in &self.edges {
            if let (Some(f), Some(t)) = (
                old_to_new_node[e.from as usize],
                old_to_new_node[e.to as usize],
            ) {
                let new_eid = new_edges.len() as u32;
                let mut ne = e.clone();
                ne.id = new_eid;
                ne.from = f;
                ne.to = t;

                new_out[f as usize].push(new_eid);
                new_in[t as usize].push(new_eid);
                new_edges.push(ne);
            }
        }

        new_edges
    }

    /// Rebuild the kept layers and the night→layer index using the node id mapping.
    fn rebuild_layers_and_index(
        &self,
        keep_from: usize,
        kept_node_range: &Range<usize>,
        old_to_new_node: &[Option<NodeId>],
    ) -> (Vec<NightLayer>, AHashMap<NightId, usize>) {
        let mut new_layers = Vec::new();
        let mut night_to_layer_idx = AHashMap::default();
        let mut cursor: u32 = 0;

        for l in &self.layers[keep_from..] {
            let start_old = l.node_range.start as usize;
            let end_old = l.node_range.end as usize;

            // Only layers fully included in the kept node range survive.
            if start_old >= kept_node_range.start && end_old <= kept_node_range.end {
                let len = (end_old - start_old) as u32;
                let mut nl = NightLayer::new(l.night, cursor..cursor + len);

                // Rebuild the seed→node map for the compacted ids.
                for old_nid in start_old..end_old {
                    let new_nid = old_to_new_node[old_nid].expect("compact map must exist");
                    let seed = self.nodes[old_nid].seed;
                    nl.seed_to_node.insert(seed, new_nid);
                }

                night_to_layer_idx.insert(l.night, new_layers.len());
                new_layers.push(nl);
                cursor += len;
            }
        }

        (new_layers, night_to_layer_idx)
    }
}
