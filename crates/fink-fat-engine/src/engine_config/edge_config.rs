use serde::{Deserialize, Serialize};

use crate::engine_config::{error::EdgeConfigError, propagator_config::PredictorParams};

fn default_false() -> bool {
    false
}

/// Runtime configuration actually used by the engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EdgeConfig {
    pub edge_ranking_model_path: String,
    pub top_k_per_left: usize,
    pub onnx_batch_size: usize,

    #[serde(default = "default_false")]
    pub emit_all_edges: bool,

    /// Enable parallel processing of left seeds in Rayon using `par_chunks`.
    #[serde(default = "default_false")]
    pub parallel_left_batches: bool,

    /// Chunk size for left-side parallel processing.
    ///
    /// Notes
    /// -----
    /// - Used by `left.par_chunks(parallel_left_batch_size)`.
    /// - Values <= 0 are clamped to 1 at runtime.
    pub parallel_left_batch_size: usize,

    pub predictor_config: PredictorParams,
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            edge_ranking_model_path: "model.onnx".to_string(),
            top_k_per_left: 32,
            onnx_batch_size: 128,
            emit_all_edges: false,
            parallel_left_batches: false,
            parallel_left_batch_size: 512,
            predictor_config: PredictorParams::default(),
        }
    }
}

impl EdgeConfig {
    pub fn validate(&self) -> Result<(), EdgeConfigError> {
        if self.top_k_per_left == 0 {
            return Err(EdgeConfigError::TopKPerLeftZero);
        }
        Ok(())
    }
}
