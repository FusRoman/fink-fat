use ahash::AHashMap;

use crate::{night_id::NightId, persistence::seed_node::SeedKey, pipeline::seed_store::SeedStore};

pub type SeedId=u32;

/// Map (night_id, idx_in_night) -> global dense index [0..N_total).
///
/// This avoids any pointer-based indexing and scales to millions of seeds.
/// Requires that `SeedKey.idx_in_night` is dense within each night.
#[derive(Debug, Clone)]
pub struct SeedGlobalIndex {
    base: AHashMap<NightId, SeedId>,
    n_total: u32,
}

impl SeedGlobalIndex {
    /// Build global offsets in a deterministic order (sorted NightId).
    pub fn build(seed_store: &SeedStore) -> Self {
        // If SeedStore's inner map is private, add a `night_ids()` + `len_for_night(nid)` API.
        let nights: Vec<NightId> = seed_store.night_ids_sorted();

        let mut base: AHashMap<NightId, SeedId> = AHashMap::default();
        let mut cursor: u32 = 0;

        for nid in nights {
            base.insert(nid, cursor);
            let len = seed_store.len_for_night(&nid).unwrap_or(0) as u32;
            cursor += len;
        }

        Self {
            base,
            n_total: cursor,
        }
    }

    #[inline]
    pub fn n_total(&self) -> usize {
        self.n_total as usize
    }

    /// Global dense index of a seed identified by its `SeedKey`.
    #[inline]
    pub fn idx_of_key(&self, key: SeedKey) -> usize {
        (self.base[&key.night_id] + key.idx_in_night) as usize
    }
}
