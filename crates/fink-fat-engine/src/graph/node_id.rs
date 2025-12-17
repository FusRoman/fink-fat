use std::{
    fmt,
    ops::{Add, AddAssign, Sub, SubAssign},
};

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

/// Identifier for a node inside a graph node array.
///
/// - 0 ≤ id < N where N is the number of nodes in the graph.
/// - Permits O(1) access into `Vec<Node>` via `.idx()`.
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
pub struct NodeId(pub u64);

impl NodeId {
    /// Return the node index into a `Vec<Node>`.
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

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeId({})", self.0)
    }
}

impl From<u64> for NodeId {
    #[inline]
    fn from(value: u64) -> Self {
        NodeId(value)
    }
}

impl From<usize> for NodeId {
    #[inline]
    fn from(value: usize) -> Self {
        NodeId(value as u64)
    }
}

impl Add<u64> for NodeId {
    type Output = NodeId;

    #[inline]
    fn add(self, rhs: u64) -> Self::Output {
        NodeId(self.0 + rhs)
    }
}

impl AddAssign<u64> for NodeId {
    #[inline]
    fn add_assign(&mut self, rhs: u64) {
        self.0 += rhs;
    }
}

impl Sub<u64> for NodeId {
    type Output = NodeId;

    #[inline]
    fn sub(self, rhs: u64) -> Self::Output {
        NodeId(self.0 - rhs)
    }
}

impl SubAssign<u64> for NodeId {
    #[inline]
    fn sub_assign(&mut self, rhs: u64) {
        self.0 -= rhs;
    }
}
