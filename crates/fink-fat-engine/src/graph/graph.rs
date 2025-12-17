use ahash::AHashMap;

use crate::{
    engine_config::edge_config::EdgeConfig, graph::{
        edge::{Edge, EdgeId},
        layer::NightLayer,
        node::Node,
        node_id::NodeId,
    }, night_id::NightId, seeding::{seed_id::SeedId, seed_node::SeedNode, seed_spatial_index::SeedSpatialIndex}, solver::components::ConnectedComponents, spacetime_bucket::spatial_binner::SpatialBinner
};

#[derive(Debug)]
pub struct InterNightGraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub out_adj: Vec<Vec<EdgeId>>,
    pub in_adj: Vec<Vec<EdgeId>>,
    pub layers: Vec<NightLayer>,
    night_to_layer_idx: AHashMap<NightId, usize>,
}

impl InterNightGraph {
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

    pub fn get_nodes_by_ids(&self, node_ids: &[NodeId]) -> Vec<&Node> {
        node_ids.iter().map(|&nid| &self.nodes[nid.idx()]).collect()
    }

    #[inline]
    pub fn node_id_of(&self, night: NightId, seed: SeedId) -> Option<NodeId> {
        let &lidx = self.night_to_layer_idx.get(&night)?;
        self.layers[lidx].seed_to_node.get(&seed).copied()
    }

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

        debug_assert!(
            left_nodes
                .iter()
                .all(|s| s.night_id == left_nodes[0].night_id),
            "left_nodes must all belong to the same night"
        );
        debug_assert!(
            right_nodes
                .iter()
                .all(|s| s.night_id == right_nodes[0].night_id),
            "right_nodes must all belong to the same night"
        );

        let night_left = left_nodes[0].night_id;
        let night_right = right_nodes[0].night_id;

        /* ---------- ensure both layers exist ---------- */
        // (right)
        if !self.night_to_layer_idx.contains_key(&night_right) {
            let seeds: Vec<_> = right_nodes.iter().map(|s| s.seed_id).collect();
            self.add_night_layer(night_right, &seeds);
        }
        // (left) — normally already present, but keep it robust.
        if !self.night_to_layer_idx.contains_key(&night_left) {
            let seeds: Vec<_> = left_nodes.iter().map(|s| s.seed_id).collect();
            self.add_night_layer(night_left, &seeds);
        }

        /* ---------- build right SeedId -> index mapping ---------- */
        let mut right_id_to_index = AHashMap::with_capacity(right_nodes.len());
        for (i, sn) in right_nodes.iter().enumerate() {
            right_id_to_index.insert(sn.seed_id, i);
        }

        /* ---------- borrow seed->node maps from layers ---------- */
        let left_lidx = self.night_to_layer_idx[&night_left];
        let right_lidx = self.night_to_layer_idx[&night_right];

        let left_seed_to_node = &self.layers[left_lidx].seed_to_node;
        let right_seed_to_node = &self.layers[right_lidx].seed_to_node;

        /* ---------- generate NodeId-based edges ---------- */
        let id_start = EdgeId::from(self.edges.len());

        let new_edges = Edge::generate_topk_edges(
            id_start,
            left_nodes,
            right_nodes,
            edge_config,
            binner,
            index_right,
            t_right_med,
            left_seed_to_node,
            right_seed_to_node,
            &right_id_to_index,
        );

        /* ---------- insert edges into graph without node_id_of() ---------- */
        for edge in new_edges {
            let eid = edge.id;

            self.out_adj[edge.from.idx()].push(eid);
            self.in_adj[edge.to.idx()].push(eid);

            self.edges.push(edge);
        }
    }

    /// Compute connected components (undirected view) using DSU.
    ///
    /// Notes
    /// -----
    /// This is typically used to route components to different solvers
    /// (trivial / min-cost flow / blob-breaker).
    pub fn connected_components(&self) -> ConnectedComponents {
        ConnectedComponents::compute(self.nodes.len(), &self.edges)
    }
}
