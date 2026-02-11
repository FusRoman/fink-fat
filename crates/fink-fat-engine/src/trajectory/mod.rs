pub mod track_id;

use crate::{
    graph::edge::Edge,
    seeding::seed_node::SeedNode,
    trajectory::track_id::{TrackId, track_id_from_nodes},
};

/// One trajectory hypothesis produced by a solver.
///
/// A hypothesis is an ordered chain in time.
/// Storing both nodes and edges makes post-processing easier:
/// - edges are useful to deactivate selected links,
/// - nodes are useful for inspection and downstream building blocks.
#[derive(Clone, Debug)]
pub struct TrackHypothesis<'edge_lf, 'seed_lf, 'alert_lf> {
    /// Nodes in strictly increasing time (night / epoch order).
    pub nodes: Vec<&'seed_lf SeedNode<'alert_lf>>,
    /// Edges used to connect the nodes (typically len = nodes.len() - 1).
    pub edges: Vec<&'edge_lf Edge<'seed_lf, 'alert_lf>>,
    /// Additive cost / score returned by the solver (lower is better if cost).
    pub cost: f64,

    /// Optional quick metadata (useful for routing / debug).
    pub night_span: u32,
}

impl<'edge_lf, 'seed_lf, 'alert_lf> TrackHypothesis<'edge_lf, 'seed_lf, 'alert_lf> {
    /// Get the number of nodes in the hypothesis.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges in the hypothesis.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn track_id(&self) -> TrackId {
        track_id_from_nodes(&self.nodes)
    }
}
