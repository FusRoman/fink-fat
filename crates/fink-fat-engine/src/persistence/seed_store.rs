use serde::{Deserialize, Serialize};

use crate::{
    night_id::NightId,
    persistence::seed_node::SeedNodeOwned,
    pipeline::{alert_store::AlertStore, seed_store::SeedStore},
    seeding::seed_node::SeedNode,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NightSeedsOwned {
    pub night_id: NightId,
    pub seeds: Vec<SeedNodeOwned>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SeedStoreOwned {
    pub nights: Vec<NightSeedsOwned>,
}

use ahash::AHashMap;

impl SeedStoreOwned {
    pub fn to_borrowed<'a>(&self, alerts: &'a AlertStore) -> Result<SeedStore<'a>, String> {
        let mut map: AHashMap<NightId, Vec<SeedNode<'a>>> =
            AHashMap::with_capacity(self.nights.len());

        for night in &self.nights {
            let mut seeds_borrowed = Vec::with_capacity(night.seeds.len());
            for s in &night.seeds {
                seeds_borrowed.push(s.to_borrowed(alerts)?);
            }
            map.insert(night.night_id, seeds_borrowed);
        }

        Ok(SeedStore::from_map(map))
    }
}
