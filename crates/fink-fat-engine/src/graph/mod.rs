pub mod edge;

use ahash::AHashMap;

use crate::{
    MJDTT,
    engine_config::edge_config::EdgeConfig,
    graph::edge::{Edge, edge_prediction::EdgeRankingModelPool, error::EdgeBuilderError},
    persistence::graph::GraphOwned,
    pipeline::progress_sink::ProgressSink,
    seeding::{SeedKey, SeedNode},
    spacetime_bucket::spatial_binner::SpatialBinner,
};

#[derive(Debug, Clone)]
pub struct GraphCore {
    pub in_deg: AHashMap<SeedKey, usize>,
    pub out_deg: AHashMap<SeedKey, usize>,
}

#[derive(Debug)]
pub struct RuntimeGraph<'seed_lf> {
    pub core: GraphCore,
    pub edges: Vec<Edge<'seed_lf>>,
}

impl<'seed_lf> RuntimeGraph<'seed_lf> {
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

    pub fn add_inter_night_edges<B: SpatialBinner>(
        &mut self,
        left_nodes: &'seed_lf [SeedNode],
        right_nodes: &'seed_lf [SeedNode],
        edge_config: &EdgeConfig,
        spatial_binner: &B,
        time_binner_width: MJDTT,
        model_pool: Option<&EdgeRankingModelPool>,
        progress_sink: &dyn ProgressSink,
    ) -> Result<(), EdgeBuilderError> {
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

        // Invariant: right_nodes are sorted by epoch_mid (and tie-breakers) already.
        //
        // This is required by generate_topk_edges() / candidate search logic that relies
        // on monotonic epoch ordering.
        debug_assert!(
            right_nodes.windows(2).all(|w| w[0] <= w[1]),
            "right_nodes must be sorted (SeedNode Ord: epoch_mid primary key)"
        );

        let new_edges = Edge::build_edges(
            left_nodes,
            right_nodes,
            edge_config,
            spatial_binner,
            time_binner_width,
            model_pool,
            progress_sink,
        )?;

        for edge in new_edges {
            let from = edge.from.key();
            let to = edge.to.key();

            *self.core.out_deg.entry(from).or_insert(0) += 1;
            *self.core.in_deg.entry(to).or_insert(0) += 1;

            self.edges.push(edge);
        }

        Ok(())
    }
}
