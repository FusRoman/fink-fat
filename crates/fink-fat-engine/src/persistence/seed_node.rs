use std::ops::Deref;

use serde::{Deserialize, Serialize};

use crate::{
    Alert,
    night_id::NightId,
    persistence::alert::AlertKey,
    pipeline::alert_store::AlertStore,
    seeding::seed_node::{SeedNode, SeedNodeCore},
};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SeedKey {
    pub night_id: NightId,
    pub idx_in_night: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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
    ) -> Result<SeedNode<'alert_lf>, String> {
        let mut members: Vec<&'alert_lf Alert> = Vec::with_capacity(self.members.len());

        for k in &self.members {
            let a = alerts
                .get_by_key(*k)
                .ok_or_else(|| format!("Missing alert for key {:?}", k))?;
            members.push(a);
        }

        Ok(SeedNode {
            core: self.core.clone(),
            members,
        })
    }
}
