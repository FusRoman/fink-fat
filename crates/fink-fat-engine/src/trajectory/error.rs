use thiserror::Error;

use crate::alerts::AlertKey;

#[derive(Debug, Error)]
pub enum TrackError {
    #[error("Alert key not found: {0:?}")]
    AlertKeyNotFound(AlertKey),
    #[error("Track must at least contain one node")]
    TrackNodesEmpty,
}
