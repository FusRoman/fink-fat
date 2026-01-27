use crate::{
    engine_config::edge_config::EdgeConfig,
    graph::{edge::Edge, edge::edge_id::EdgeId, node::Node},
    seeding::seed_node::SeedNode,
    spacetime_bucket::{spatial_binner::SpatialBinner, time_binner::TimeBinner},
};

#[derive(Debug)]
pub struct InterNightGraph<'a> {
    pub nodes: Vec<Node<'a>>,
    pub edges: Vec<Edge<'a>>,
}

impl<'a> InterNightGraph<'a> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub fn add_inter_night_edges<B: SpatialBinner, T: TimeBinner>(
        &mut self,
        left_nodes: &'a [SeedNode],
        right_nodes: &'a mut [SeedNode],
        edge_config: &EdgeConfig,
        spatial_binner: &B,
        time_binner: &T,
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

        /* ---------- generate NodeId-based edges ---------- */
        let id_start = EdgeId::from(self.edges.len());

        /* ---------- Sort right nodes by epoch mid time ---------- */
        // required by generate_topk_edges()
        right_nodes.sort_by(|a, b| {
            a.plane
                .epoch_mid
                .partial_cmp(&b.plane.epoch_mid)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let new_edges = Edge::generate_topk_edges(
            id_start,
            left_nodes,
            right_nodes,
            edge_config,
            spatial_binner,
            time_binner,
        );

        /* ---------- insert edges into graph without node_id_of() ---------- */
        for edge in new_edges {
            self.edges.push(edge);
        }
    }

    pub fn deactivate_edges(&mut self, eids: &[EdgeId]) {
        for &eid in eids {
            if let Some(e) = self.edges.get_mut(eid.idx()) {
                e.active = false;
            }
        }
    }
}
