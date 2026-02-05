//! Layer metadata: index range of nodes for a single night.

use std::ops::Range;

use crate::{graph::node_id::NodeId, night_id::NightId};

/// A **layer** groups the nodes of a single night and provides fast lookup.
#[derive(Clone, Debug)]
pub struct NightLayer {
    /// Night identifier (strictly increasing ordering across layers).
    pub night: NightId,
    /// Half-open range [start, end) of node IDs for this night in the graph.
    pub node_range: Range<NodeId>,
}

impl NightLayer {
    pub fn new(night: NightId, node_range: Range<NodeId>) -> Self {
        Self {
            night,
            node_range,
        }
    }

    #[inline]
    pub fn contains_node(&self, nid: NodeId) -> bool {
        self.node_range.start <= nid && nid < self.node_range.end
    }
}
