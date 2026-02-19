pub mod error;
pub mod track_id;

use crate::{
    Alert, AlertStore,
    graph::edge::Edge,
    seeding::SeedNode,
    trajectory::{
        error::TrackError,
        track_id::{TrackId, track_id_from_nodes},
    },
};

/// One trajectory hypothesis produced by a solver.
///
/// A hypothesis is an ordered chain in time.
/// Storing both nodes and edges makes post-processing easier:
/// - edges are useful to deactivate selected links,
/// - nodes are useful for inspection and downstream building blocks.
#[derive(Clone, Debug)]
pub struct TrackHypothesis<'edge_lf, 'seed_lf> {
    /// Nodes in strictly increasing time (night / epoch order).
    pub nodes: Vec<&'seed_lf SeedNode>,
    /// Edges used to connect the nodes (typically len = nodes.len() - 1).
    pub edges: Vec<&'edge_lf Edge>,

    /// Additive cost / score returned by the solver (lower is better if cost).
    pub cost: f64,

    /// Optional quick metadata (useful for routing / debug).
    pub night_span: u32,
}

impl<'edge_lf, 'seed_lf> TrackHypothesis<'edge_lf, 'seed_lf> {
    /// Get the number of nodes in the hypothesis.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges in the hypothesis.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn track_id(&self, store: &AlertStore) -> Result<TrackId, TrackError> {
        track_id_from_nodes(store, &self.nodes)
    }

    /// Get owned copies of all alerts in this track.
    ///
    /// Arguments
    /// ---------
    /// * `store` – Alert storage to query.
    ///
    /// Return
    /// ------
    /// * `Ok(Vec<Alert>)` – Cloned alerts in observation time order.
    /// * `Err(TrackError::AlertKeyNotFound)` – If any alert key is missing from store.
    pub fn get_alerts(&self, store: &AlertStore) -> Result<Vec<Alert>, TrackError> {
        self.nodes
            .iter()
            .flat_map(|seed| seed.members.iter())
            .map(|&alert_id| {
                store
                    .get_by_key(alert_id)
                    .cloned()
                    .ok_or_else(|| TrackError::AlertKeyNotFound(alert_id.into()))
            })
            .collect::<Result<Vec<_>, _>>()
    }
}
