//! Node model (one per **seed** in a given night/layer).

use crate::{graph::NodeId, propagation::features::SeedId, NightId};

/// Graph node representing one **seed** detection/tracklet within a night.
///
/// Notes
/// -----
/// We keep the node minimal to keep memory flat and cache-friendly.
/// Per-seed kinematic features live upstream; link scoring is already done
/// when edges get created.
#[derive(Clone, Debug)]
pub struct Node {
    pub id: NodeId,
    pub night: NightId,
    pub seed: SeedId,
}

impl Node {
    /// Create a node for a seed at a given night. `id` is assigned by the graph.
    pub fn new(id: NodeId, night: NightId, seed: SeedId) -> Self {
        Self { id, night, seed }
    }
}
