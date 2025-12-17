use ahash::AHashMap;

use crate::{
    graph::{
        edge::{Edge, EdgeId},
        layer::NightLayer,
        node::Node,
        node_id::NodeId,
    },
    night_id::NightId,
    seeding::seed_id::SeedId,
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

    
}
