pub mod edge;

use ahash::AHashMap;

use crate::{
    MJDTT,
    engine_config::edge_config::EdgeConfig,
    graph::edge::{Edge, edge_prediction::EdgeRankingModelPool, error::EdgeBuilderError},
    pipeline::progress_sink::ProgressSink,
    seeding::{SeedKey, SeedNode},
    spacetime_bucket::spatial_binner::SpatialBinner,
};

#[derive(Debug)]
pub struct AlertLinkageDAG {
    pub in_deg: AHashMap<SeedKey, usize>,
    pub out_deg: AHashMap<SeedKey, usize>,
    pub edges: Vec<Edge>,
}

impl AlertLinkageDAG {
    pub fn new() -> Self {
        Self {
            in_deg: AHashMap::new(),
            out_deg: AHashMap::new(),
            edges: Vec::new(),
        }
    }

    pub fn from_edges(edges: Vec<Edge>) -> Self {
        let mut in_deg = AHashMap::new();
        let mut out_deg = AHashMap::new();

        for edge in &edges {
            let from = edge.from;
            let to = edge.to;

            *out_deg.entry(from).or_insert(0) += 1;
            *in_deg.entry(to).or_insert(0) += 1;
        }

        Self {
            in_deg,
            out_deg,
            edges,
        }
    }

    pub fn add_inter_night_edges<B: SpatialBinner>(
        &mut self,
        left_nodes: &[SeedNode],
        right_nodes: &[SeedNode],
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
            let from = edge.from;
            let to = edge.to;

            *self.out_deg.entry(from).or_insert(0) += 1;
            *self.in_deg.entry(to).or_insert(0) += 1;

            self.edges.push(edge);
        }

        Ok(())
    }
}
