use thiserror::Error;

use crate::{AlertKey, seeding::SeedKey};

#[derive(Debug, Error)]
pub enum SeedingError {
    #[error("Alert key not found: {0:?}")]
    AlertKeyNotFound(AlertKey),
    #[error("Seed key not found: {0:?}")]
    SeedKeyNotFound(SeedKey),
}
