use thiserror::Error;

use crate::{night_id::NightId, seeding::SeedKey};

#[derive(Debug)]
pub enum SeedOrigin {
    Store,
    Index,
}

#[derive(Debug, Error)]
pub enum ComponentError {
    #[error("Seed key not found: {key:?} (origin: {origin:?})")]
    SeedKeyNotFound { key: SeedKey, origin: SeedOrigin },
    #[error("Night not found in seed store: {0:?}")]
    NightNotFound(NightId),
}
