use ahash::AHashMap;

use crate::{
    night_id::NightId,
    persistence::{
        seed_node::{SeedKey, SeedNodeOwned},
        seed_store::SeedStoreOwned,
    },
    seeding::seed_node::SeedNode,
};

pub struct SeedStore<'alert_lf>(AHashMap<NightId, Vec<SeedNode<'alert_lf>>>);

impl<'alert_lf> SeedStore<'alert_lf> {
    /// Create a new empty `SeedStore`.
    pub fn new() -> Self {
        Self(AHashMap::new())
    }

    /// Create a `SeedStore` from a pre-constructed map of night IDs to seed nodes.
    pub fn from_map(map: AHashMap<NightId, Vec<SeedNode<'alert_lf>>>) -> Self {
        Self(map)
    }

    /// Convert this `SeedStore` into an owned version that can be serialized.
    pub fn to_owned(&self) -> SeedStoreOwned {
        let mut map = AHashMap::with_capacity(self.0.len());

        for (night_id, seeds) in self.0.iter() {
            let owned_seeds: Vec<SeedNodeOwned> = seeds.iter().map(|s| s.to_owned()).collect();

            map.insert(*night_id, owned_seeds);
        }

        SeedStoreOwned(map)
    }

    /// Insert a vector of seed nodes for a given night.
    pub fn insert(&mut self, night_id: NightId, seeds: Vec<SeedNode<'alert_lf>>) {
        self.0.insert(night_id, seeds);
    }

    /// Get the vector of seed nodes for a given night, if it exists.
    pub fn get(&self, night_id: &NightId) -> Option<&Vec<SeedNode<'alert_lf>>> {
        self.0.get(night_id)
    }

    /// Get a specific seed node by its key (night ID + index in night).
    pub fn get_by_key(&self, key: SeedKey) -> Option<&SeedNode<'alert_lf>> {
        let vec = self.0.get(&key.night_id)?;
        vec.get(key.idx_in_night as usize)
    }
}
