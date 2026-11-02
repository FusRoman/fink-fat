use thiserror::Error;

#[derive(Error, Debug)]
pub enum FinkFatError {
    #[error(
        "Snapshot file not found, run fink-fat in tracking mode to generate a snapshot containing track object."
    )]
    NoSnapshot,
    #[error("{0}")]
    Message(String),
}
