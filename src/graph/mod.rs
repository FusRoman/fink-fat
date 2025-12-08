use crate::NightId;

pub mod components;
pub mod edge;
pub mod graph;
pub mod ingest;
pub mod layer;
pub mod node;
pub mod stats;

/// Index into the graph's node vector.
pub type NodeId = u32;

/// Index into the graph's edge vector.
pub type EdgeId = u32;

/// Maximum number of nights the graph keeps link **eligibility** for.
/// Links can only span at most `horizon_nights` (strictly positive).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Horizon {
    pub horizon_nights: u32,
}

impl Horizon {
    pub fn new(h: u32) -> Self {
        assert!(h > 0, "Horizon must be ≥ 1 night.");
        Self { horizon_nights: h }
    }

    #[inline]
    pub fn within(&self, from: NightId, to: NightId) -> bool {
        to > from && (to - from) <= self.horizon_nights
    }
}
