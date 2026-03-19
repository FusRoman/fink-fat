use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum PersistPolicy {
    None,
    Minimal,
    Full,
}
