pub mod edge;

use ahash::AHashMap;

use crate::{
    engine_config::edge_config::EdgeConfig,
    graph::edge::{
        Edge,
        edge_prediction::{EdgeModelError, EdgeRankingModelPool},
    },
    persistence::{graph::GraphOwned, seed_node::SeedKey},
    seeding::seed_node::SeedNode,
    spacetime_bucket::{spatial_binner::SpatialBinner, time_binner::TimeBinner},
};

#[derive(Debug, Clone)]
pub struct GraphCore {
    pub in_deg: AHashMap<SeedKey, usize>,
    pub out_deg: AHashMap<SeedKey, usize>,
}

#[derive(Debug)]
pub struct RuntimeGraph<'seed_lf, 'alert_lf> {
    pub core: GraphCore,
    pub edges: Vec<Edge<'seed_lf, 'alert_lf>>,
}

impl<'seed_lf, 'alert_lf> RuntimeGraph<'seed_lf, 'alert_lf> {
    pub fn new() -> Self {
        Self {
            core: GraphCore {
                in_deg: AHashMap::new(),
                out_deg: AHashMap::new(),
            },
            edges: Vec::new(),
        }
    }

    pub fn to_owned(&self) -> GraphOwned {
        GraphOwned {
            core: self.core.clone(),
            edges: self.edges.iter().map(|e| e.to_owned()).collect(),
        }
    }

    pub fn add_inter_night_edges<B: SpatialBinner, T: TimeBinner>(
        &mut self,
        left_nodes: &'seed_lf [SeedNode<'alert_lf>],
        right_nodes: &'seed_lf mut [SeedNode<'alert_lf>],
        edge_config: &EdgeConfig,
        spatial_binner: &B,
        time_binner: &T,
        model_pool: Option<&EdgeRankingModelPool>,
    ) -> Result<(), EdgeModelError> {
        assert!(!left_nodes.is_empty(), "left_nodes must not be empty");
        assert!(!right_nodes.is_empty(), "right_nodes must not be empty");

        debug_assert!(
            left_nodes
                .iter()
                .all(|s| s.night_id() == left_nodes[0].night_id()),
            "left_nodes must all belong to the same night"
        );
        debug_assert!(
            right_nodes
                .iter()
                .all(|s| s.night_id() == right_nodes[0].night_id()),
            "right_nodes must all belong to the same night"
        );

        /* ---------- Sort right nodes by epoch mid time ---------- */
        // required by generate_topk_edges()
        right_nodes.sort_by(|a, b| {
            a.plane
                .epoch_mid
                .partial_cmp(&b.plane.epoch_mid)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let new_edges = Edge::build_edges(
            left_nodes,
            right_nodes,
            edge_config,
            spatial_binner,
            time_binner,
            model_pool,
        )?;

        /* ---------- insert edges into graph without node_id_of() ---------- */
        for edge in new_edges {

            let from = edge.from.core.key;
            let to = edge.to.core.key;

            *self.core.out_deg.entry(from).or_insert(0) += 1;
            *self.core.in_deg.entry(to).or_insert(0) += 1;

            self.edges.push(edge);
        }
        Ok(())
    }
}
