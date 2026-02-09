pub mod delta_chunk;
pub mod edge_journal;
pub mod edge_op;

use serde::{Deserialize, Serialize};

use crate::{
    graph::edge::{Edge, EdgeCore},
    persistence::seed_node::SeedKey,
    pipeline::seed_store::SeedStore,
};

/// Stable identity for an edge in the persisted graph.
///
/// Notes
/// -----
/// This assumes there is at most one edge per `(from, to)` pair.
/// That matches the usual Fink-FAT semantics: a directed link between two seeds.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct EdgeKey {
    /// Source seed (older epoch).
    pub from: SeedKey,
    /// Target seed (newer epoch).
    pub to: SeedKey,
}

impl EdgeKey {
    /// Construct an [`EdgeKey`] from two seed keys.
    #[inline]
    pub fn new(from: SeedKey, to: SeedKey) -> Self {
        Self { from, to }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EdgeOwned {
    pub core: EdgeCore,
    pub from: SeedKey,
    pub to: SeedKey,
}

impl EdgeOwned {
    pub fn to_borrowed<'seed, 'alert>(
        &self,
        seeds: &'seed SeedStore<'alert>,
    ) -> Result<Edge<'seed, 'alert>, String> {
        let from = seeds
            .get_by_key(self.from)
            .ok_or_else(|| format!("Missing seed for key {:?}", self.from))?;
        let to = seeds
            .get_by_key(self.to)
            .ok_or_else(|| format!("Missing seed for key {:?}", self.to))?;

        Ok(Edge {
            core: self.core.clone(),
            from,
            to,
        })
    }
}
