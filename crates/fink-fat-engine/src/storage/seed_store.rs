use ahash::AHashMap;

use crate::{night_id::NightId, seeding::seed_node::SeedNode};

pub struct SeedStore(AHashMap<NightId, Vec<SeedNode>>);

impl SeedStore {
    pub fn new() -> Self {
        Self(AHashMap::new())
    }

    pub fn insert(&mut self, night_id: NightId, seeds: Vec<SeedNode>) {
        self.0.insert(night_id, seeds);
    }

    pub fn get(&self, night_id: &NightId) -> Option<&Vec<SeedNode>> {
        self.0.get(night_id)
    }
}
