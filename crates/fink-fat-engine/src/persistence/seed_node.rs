use std::ops::Deref;

use camino::Utf8PathBuf;
use serde::{Deserialize, Serialize};

use crate::{
    Alert, alerts::{AlertKey, store::AlertStore}, night_id::NightId, persistence::{
        SEED_STORE_SCHEMA_VERSION,
        envelope::DiskEnvelope,
        error::{BorrowError, PersistenceIoError},
        layout::PersistenceLayout,
        manifest::Manifest,
    }, seeding::seed_node::{SeedNode, SeedNodeCore}
};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize, Default)]
pub struct SeedKey {
    pub night_id: NightId,
    pub idx_in_night: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct SeedNodeOwned {
    pub core: SeedNodeCore,
    pub members: Vec<AlertKey>,
}

impl Deref for SeedNodeOwned {
    type Target = SeedNodeCore;
    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl SeedNodeOwned {
    pub fn to_borrowed<'alert_lf>(
        &self,
        alerts: &'alert_lf AlertStore,
    ) -> Result<SeedNode<'alert_lf>, BorrowError> {
        let mut members: Vec<&'alert_lf Alert> = Vec::with_capacity(self.members.len());

        for k in &self.members {
            let a = alerts
                .get_by_key(*k)
                .ok_or_else(|| BorrowError::MissingAlert(*k))?;
            members.push(a);
        }

        Ok(SeedNode {
            core: self.core.clone(),
            members,
        })
    }
}

pub trait SeedNodeOwnedSlice {
    fn save_seeds_night(
        &self,
        layout: &PersistenceLayout,
        manifest: &Manifest,
        night_id: NightId,
    ) -> Result<Utf8PathBuf, PersistenceIoError>;
}

impl SeedNodeOwnedSlice for &[SeedNodeOwned] {
    fn save_seeds_night(
        &self,
        layout: &PersistenceLayout,
        manifest: &Manifest,
        night_id: NightId,
    ) -> Result<Utf8PathBuf, PersistenceIoError> {
        let abs_path = layout.seeds_night_path(night_id);

        // Write payload (enveloped).
        let env = DiskEnvelope::new(
            self.to_vec(),
            SEED_STORE_SCHEMA_VERSION,
            manifest.created_unix_s,
        );
        env.save_enveloped(&abs_path)?;
        Ok(abs_path)
    }
}
