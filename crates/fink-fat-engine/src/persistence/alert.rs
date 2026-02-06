use serde::{Deserialize, Serialize};

use crate::{night_id::NightId};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct AlertKey {
    pub night_id: NightId,
    pub idx_in_night: u32,
}
