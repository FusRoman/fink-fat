use thiserror::Error;

use crate::{night_id::NightId, seeding::SeedKey};

#[derive(Debug, Error)]
pub enum ComponentError {
    #[error("Seed key not found in index: {0:?}")]
    SeedKeyInIndexNotFound(SeedKey),
    #[error("Night not found in seed store: {0:?}")]
    NightNotFound(NightId),
}
