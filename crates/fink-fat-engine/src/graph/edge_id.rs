use std::fmt;

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

/// Identifier for an edge inside a graph edge array.
///
/// - 0 ≤ id < M where M is the number of edges in the graph.
/// - Permits O(1) access into `Vec<Edge>` via `.idx()`.
#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    Encode,
    Decode,
    PartialOrd,
    Ord,
    Default,
)]
pub struct EdgeId(pub u64);

impl EdgeId {
    /// Return the edge index into a `Vec<Edge>`.
    #[inline]
    pub fn idx(self) -> usize {
        self.0 as usize
    }

    /// Convenience constructor.
    #[inline]
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

impl fmt::Display for EdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EdgeId({})", self.0)
    }
}

impl From<u64> for EdgeId {
    #[inline]
    fn from(value: u64) -> Self {
        EdgeId(value)
    }
}

impl From<usize> for EdgeId {
    #[inline]
    fn from(value: usize) -> Self {
        EdgeId(value as u64)
    }
}
