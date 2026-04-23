pub mod error;
pub mod track_id;

use photom::observation_dataset::{ObsDataset, observation::Observation};

use crate::{
    graph::edge::EdgeKey,
    seeding::{SeedKey, error::SeedingError, store::SeedStore},
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
pub struct TrackHypothesis {
    /// Nodes in strictly increasing time (night / epoch order).
    pub nodes: Vec<SeedKey>,
    /// Edges used to connect the nodes (typically len = nodes.len() - 1).
    pub edges: Vec<EdgeKey>,

    /// Additive cost / score returned by the solver (lower is better if cost).
    pub cost: f64,

    /// Optional quick metadata (useful for routing / debug).
    pub night_span: u32,
}

impl TrackHypothesis {
    /// Get the number of nodes in the hypothesis.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges in the hypothesis.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn track_id(
        &self,
        obs_dataset: &ObsDataset,
        seed_store: &SeedStore,
    ) -> Result<TrackId, TrackError> {
        let seeds = self
            .nodes
            .iter()
            .map(|&seed_key| {
                seed_store
                    .try_get_seed(seed_key)
                    .ok_or(TrackError::SeedingError(SeedingError::SeedKeyNotFound(
                        seed_key,
                    )))
            })
            .collect::<Result<Vec<_>, _>>()?;

        track_id_from_nodes(obs_dataset, &seeds)
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
    pub fn get_alerts<'obs>(
        &self,
        obs_dataset: &'obs ObsDataset,
        seed_store: &SeedStore,
    ) -> Result<Vec<&'obs Observation>, TrackError> {
        self.nodes
            .iter()
            .map(|&seed_key| {
                let seed = seed_store
                    .try_get_seed(seed_key)
                    .ok_or(SeedingError::SeedKeyNotFound(seed_key))?;

                Ok(seed.resolve_members(obs_dataset)?)
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|nested| nested.into_iter().flatten().collect())
    }
}
