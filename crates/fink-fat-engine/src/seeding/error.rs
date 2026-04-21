use photom::observation_dataset::ObsId;
use thiserror::Error;

use crate::seeding::SeedKey;

#[derive(Debug, Error)]
pub enum SeedingError {
    #[error("Observation index not found: {0:?}")]
    ObservationIndexNotFound(ObsId),
    #[error("Seed key not found: {0:?}")]
    SeedKeyNotFound(SeedKey),
}
