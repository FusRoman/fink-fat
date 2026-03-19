use thiserror::Error;

use crate::seeding::error::SeedingError;

#[derive(Debug, Error)]
pub enum TrackError {
    #[error(transparent)]
    SeedingError(#[from] SeedingError),
    #[error("Track must at least contain one node")]
    TrackNodesEmpty,
}
