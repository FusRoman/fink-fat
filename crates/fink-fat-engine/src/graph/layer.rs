//! Layer metadata: index range of nodes for a single night.

use std::ops::Range;

use ahash::AHashMap;

use crate::{graph::node_id::NodeId, night_id::NightId, seeding::seed_id::SeedId};

/// A **layer** groups the nodes of a single night and provides fast lookup.
#[derive(Clone, Debug)]
pub struct NightLayer {
    /// Night identifier (strictly increasing ordering across layers).
    pub night: NightId,
    /// Half-open range [start, end) of node IDs for this night in the graph.
    pub node_range: Range<NodeId>,
    /// Optional dense mapping from `SeedId` → `NodeId` for quick resolution.
    pub seed_to_node: AHashMap<SeedId, NodeId>,
}

impl NightLayer {
    pub fn new(night: NightId, node_range: Range<NodeId>) -> Self {
        Self {
            night,
            node_range,
            seed_to_node: AHashMap::default(),
        }
    }

    #[inline]
    pub fn contains_node(&self, nid: NodeId) -> bool {
        self.node_range.start <= nid && nid < self.node_range.end
    }
}
