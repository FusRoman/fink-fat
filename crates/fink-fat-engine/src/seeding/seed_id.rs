//! Lightweight identifier for a `SeedNode`.
//!
//! Semantics
//! ---------
//! A `SeedId` is **local to a single night**. For cross-night uniqueness,
//! combine it with the associated `NightId`.
//!
//! Invariant
//! ---------
//! When loading seeds from disk, seeds must be stored contiguously so that
//! `id.idx()` matches the index in the `Vec<SeedNode>` returned by `NightStore`.

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Identifier for a seed inside a single-night seed array.
///
/// - 0 ≤ id < N where N is the number of seeds in that night.
/// - Permits O(1) access into `Vec<SeedNode>` via `.idx()`.
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
pub struct SeedId(pub u64);

impl SeedId {
    /// Return the seed index into a `Vec<SeedNode>`.
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

impl fmt::Display for SeedId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SeedId({})", self.0)
    }
}

impl From<u64> for SeedId {
    #[inline]
    fn from(value: u64) -> Self {
        SeedId(value)
    }
}

impl From<usize> for SeedId {
    #[inline]
    fn from(value: usize) -> Self {
        SeedId(value as u64)
    }
}
