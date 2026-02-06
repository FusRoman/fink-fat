use ahash::AHashMap;

use crate::{night_id::NightId, seeding::seed_node::SeedNode};

pub struct SeedStore<'alert_lf>(AHashMap<NightId, Vec<SeedNode<'alert_lf>>>);

impl<'alert_lf> SeedStore<'alert_lf> {
    pub fn new() -> Self {
        Self(AHashMap::new())
    }

    pub fn insert(&mut self, night_id: NightId, seeds: Vec<SeedNode<'alert_lf>>) {
        self.0.insert(night_id, seeds);
    }

    pub fn get(&self, night_id: &NightId) -> Option<&Vec<SeedNode<'alert_lf>>> {
        self.0.get(night_id)
    }
}
