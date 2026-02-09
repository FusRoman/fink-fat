use ahash::AHashMap;
use serde::{Deserialize, Serialize};

use crate::{
    night_id::NightId,
    persistence::{
        alert_store::AlertStore, envelope::PersistenceIoError, layout::PersistenceLayout,
        manifest::Manifest, seed_node::SeedNodeOwned, seed_node::SeedNodeOwnedSlice,
    },
    pipeline::seed_store::SeedStore,
    seeding::seed_node::SeedNode,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SeedStoreOwned(pub AHashMap<NightId, Vec<SeedNodeOwned>>);

impl SeedStoreOwned {
    pub fn new() -> Self {
        Self(AHashMap::new())
    }

    pub fn insert(&mut self, night_id: NightId, seeds: Vec<SeedNodeOwned>) {
        self.0.insert(night_id, seeds);
    }

    pub fn to_borrowed<'a>(&self, alerts: &'a AlertStore) -> Result<SeedStore<'a>, String> {
        let mut map: AHashMap<NightId, Vec<SeedNode<'a>>> = AHashMap::with_capacity(self.0.len());

        for (night_id, seeds_owned) in self.0.iter() {
            let mut seeds_borrowed = Vec::with_capacity(seeds_owned.len());
            for s in seeds_owned {
                seeds_borrowed.push(s.to_borrowed(alerts)?);
            }
            map.insert(*night_id, seeds_borrowed);
        }

        Ok(SeedStore::from_map(map))
    }

    pub fn save_seeds_night(
        &self,
        layout: &PersistenceLayout,
        manifest: &mut Manifest,
        night_id: NightId,
    ) -> Result<(), PersistenceIoError> {
        if let Some(seeds) = self.0.get(&night_id) {
            seeds
                .as_slice()
                .save_seeds_night(layout, manifest, night_id)?;
        }
        Ok(())
    }
}
