use photom::observation_dataset::ObsId;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SeedingError {
    #[error("Observation index not found: {0:?}")]
    ObservationIndexNotFound(ObsId),
}
