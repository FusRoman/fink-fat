use thiserror::Error;

#[derive(Debug, Error)]
pub enum SolverError {
    #[error("Alert key not found: {0:?}")]
    OrbitFitConversionError(String),
}
