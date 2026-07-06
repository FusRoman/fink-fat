use thiserror::Error;

/// Errors that can occur while loading the model or running ONNX inference.
#[derive(Debug, Error)]
pub enum EdgeModelError {
    /// ONNX Runtime / `ort` error (session build, tensor extraction, run failure, etc.).
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),

    /// ONNX Runtime session builder error.
    ///
    /// Separate from [`EdgeModelError::Ort`] because `ort::Error<SessionBuilder>`
    /// does not coerce to `ort::Error<()>` automatically.
    #[error("ONNX Runtime session builder error: {0}")]
    OrtBuilder(String),

    /// The provided model path does not exist.
    ///
    /// This is returned early to provide a clear user-facing message instead of
    /// a lower-level ORT error that may be harder to interpret.
    #[error("ONNX model file not found: {0}")]
    ModelNotFound(String),

    /// ML ranking was requested but no `EdgeRankingModel` was provided.
    ///
    /// This is useful when ranking is optional and the caller explicitly enables it
    /// in a configuration, but forgets to provide a loaded model.
    #[error("ML ranking requested but no EdgeRankingModel was provided")]
    MissingModel,
}

#[derive(Debug, Error)]
pub enum EdgeBuilderError {
    /// Errors related to the input seeds (e.g., empty seed lists, invalid epochs).
    #[error("Invalid input seeds: {0}")]
    InvalidSeeds(String),

    /// Errors that occur during edge construction (e.g., spatial indexing, feature computation).
    #[error("Edge construction error: {0}")]
    ConstructionError(String),

    /// Errors from the ML model during edge ranking.
    #[error("Edge model error: {0}")]
    ModelError(#[from] EdgeModelError),
}
