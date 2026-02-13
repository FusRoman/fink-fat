use ahash::AHashMap;
use serde::{Deserialize, Serialize};

use crate::{
    night_id::{NightId, NightWindow},
    persistence::{
        alert_store::AlertStore,
        error::{BorrowError, PersistenceIoError},
        layout::PersistenceLayout,
        manifest::Manifest,
        seed_node::{SeedNodeOwned, SeedNodeOwnedSlice},
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

    /// Shared implementation for borrowing seeds, optionally restricted to a `NightWindow`.
    fn to_borrowed_impl<'alert_lf>(
        &self,
        alerts: &'alert_lf AlertStore,
        night_window: Option<NightWindow>,
    ) -> Result<SeedStore<'alert_lf>, BorrowError> {
        let mut map: AHashMap<NightId, Vec<SeedNode<'alert_lf>>> =
            AHashMap::with_capacity(self.0.len());

        for (night_id, seeds_owned) in self.0.iter() {
            if let Some(w) = night_window {
                if !w.contains(*night_id) {
                    continue;
                }
            }

            let mut seeds_borrowed = Vec::with_capacity(seeds_owned.len());
            for s in seeds_owned {
                seeds_borrowed.push(s.to_borrowed(alerts)?);
            }
            map.insert(*night_id, seeds_borrowed);
        }

        Ok(SeedStore::from_map(map))
    }

    pub fn to_borrowed<'alert_lf>(
        &self,
        alerts: &'alert_lf AlertStore,
    ) -> Result<SeedStore<'alert_lf>, BorrowError> {
        self.to_borrowed_impl(alerts, None)
    }

    pub fn to_borrowed_window<'alert_lf>(
        &self,
        alerts: &'alert_lf AlertStore,
        night_window: NightWindow,
    ) -> Result<SeedStore<'alert_lf>, BorrowError> {
        self.to_borrowed_impl(alerts, Some(night_window))
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
