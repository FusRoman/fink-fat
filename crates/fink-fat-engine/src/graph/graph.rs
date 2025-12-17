use ahash::AHashMap;

use crate::{
    engine_config::edge_config::EdgeConfig,
    graph::{
        edge::{Edge, EdgeId},
        layer::NightLayer,
        node::Node,
        node_id::NodeId,
    },
    night_id::NightId,
    seeding::{seed_id::SeedId, seed_node::SeedNode, seed_spatial_index::SeedSpatialIndex},
    spacetime_bucket::spatial_binner::SpatialBinner,
};

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
    pub fn new() -> Self {
        Self {
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
        node_ids.iter().map(|&nid| &self.nodes[nid.idx()]).collect()
    }

    /// Resolve a `(night, seed)` into a `NodeId` if it exists in the graph.
    #[inline]
    pub fn node_id_of(&self, night: NightId, seed: SeedId) -> Option<NodeId> {
        let &lidx = self.night_to_layer_idx.get(&night)?;
        self.layers[lidx].seed_to_node.get(&seed).copied()
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

        let start: NodeId = self.nodes.len().into();
        let mut layer = NightLayer::new(night, start..start + seeds.len() as u64);

        for (i, &seed) in seeds.iter().enumerate() {
            let nid = start + i as u64;
            self.nodes.push(Node::new(nid, night, seed));
            self.out_adj.push(Vec::new());
            self.in_adj.push(Vec::new());
            layer.seed_to_node.insert(seed, nid);
        }

        let idx = self.layers.len();
        self.night_to_layer_idx.insert(night, idx);
        self.layers.push(layer);
    }

    /// Add a new night layer (if needed) and generate Top-K inter-night edges
    /// from the previous night layer to this one.
    ///
    /// This method:
    /// - appends the night layer for `right_nodes` if it does not exist yet,
    /// - generates Top-K edges using [`Edge::generate_topk_edges`],
    /// - updates adjacency lists (`out_adj`, `in_adj`) consistently.
    ///
    /// Parameters
    /// ----------
    /// left_nodes : &[SeedNode]
    ///     Seed nodes from the previous night (sources).
    /// right_nodes : &[SeedNode]
    ///     Seed nodes from the current night (targets).
    /// edge_config : &EdgeConfig
    ///     Configuration controlling Top-K, cost limits, and scoring.
    /// binner : &B
    ///     Spatial binner used for cone searches.
    /// index_right : &SeedSpatialIndex
    ///     Spatial index built on `right_nodes`.
    /// t_right_med : f64
    ///     Median epoch (TT, MJD) of the right night.
    pub fn add_inter_night_edges<B: SpatialBinner>(
        &mut self,
        left_nodes: &[SeedNode],
        right_nodes: &[SeedNode],
        edge_config: &EdgeConfig,
        binner: &B,
        index_right: &SeedSpatialIndex,
        t_right_med: f64,
    ) {
        assert!(!left_nodes.is_empty(), "left_nodes must not be empty");
        assert!(!right_nodes.is_empty(), "right_nodes must not be empty");

        let night_right = right_nodes[0].night_id;

        /* ---------- ensure right night layer exists ---------- */

        if !self.night_to_layer_idx.contains_key(&night_right) {
            let seeds: Vec<_> = right_nodes.iter().map(|s| s.seed_id).collect();
            self.add_night_layer(night_right, &seeds);
        }

        /* ---------- build SeedId -> index mapping ---------- */

        let mut right_id_to_index = AHashMap::with_capacity(right_nodes.len());
        for (i, sn) in right_nodes.iter().enumerate() {
            right_id_to_index.insert(sn.seed_id, i);
        }

        /* ---------- generate edges ---------- */

        let id_start = EdgeId::from(self.edges.len());

        let new_edges = Edge::generate_topk_edges(
            id_start,
            left_nodes,
            right_nodes,
            edge_config,
            binner,
            index_right,
            t_right_med,
            &right_id_to_index,
        );

        /* ---------- insert edges into graph ---------- */

        for edge in new_edges {
            let eid = edge.id;

            let from_nid = self
                .node_id_of(left_nodes[0].night_id, edge.from)
                .expect("source node not found in graph");
            let to_nid = self
                .node_id_of(night_right, edge.to)
                .expect("target node not found in graph");

            self.out_adj[from_nid.idx()].push(eid);
            self.in_adj[to_nid.idx()].push(eid);

            self.edges.push(edge);
        }
    }
}
