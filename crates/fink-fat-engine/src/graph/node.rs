//! Node model (one per **seed** in a given night/layer).

use crate::{
    graph::node_id::NodeId,
    night_id::NightId,
    seeding::{seed_node::SeedNode},
};

/// Graph node representing one **seed** detection/tracklet within a night.
///
/// Notes
/// -----
/// We keep the node minimal to keep memory flat and cache-friendly.
/// Per-seed kinematic features live upstream; link scoring is already done
/// when edges get created.
#[derive(Clone, Debug)]
pub struct Node<'a> {
    pub id: NodeId,
    pub night: NightId,
    pub seed: &'a SeedNode,
}

impl<'a> Node<'a> {
    /// Create a node for a seed at a given night. `id` is assigned by the graph.
    pub fn new(id: NodeId, night: NightId, seed: &'a SeedNode) -> Self {
        Self { id, night, seed }
    }
}
