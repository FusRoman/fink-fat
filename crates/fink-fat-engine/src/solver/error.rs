use thiserror::Error;

use crate::seeding::SeedKey;

#[derive(Debug, Error)]
pub enum SolverError {
    #[error("OrbitFit conversion error: {0:?}")]
    OrbitFitConversionError(String),
    #[error("Seed key not found: {0:?}")]
    SeedKeyNotFound(SeedKey),
}
