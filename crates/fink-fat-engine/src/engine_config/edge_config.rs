use serde::{Deserialize, Serialize};

use crate::engine_config::{
    error::EdgeConfigError, propagator_config::PredictorParams, score_config::ScoreConfig,
};

fn default_false() -> bool { false }

/// Runtime configuration actually used by the engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EdgeConfig {
    pub edge_ranking_model_path: String,
    pub top_k_per_left: usize,
    pub onnx_batch_size: usize,

    #[serde(default = "default_false")]
    pub emit_all_edges: bool,

    pub predictor_config: PredictorParams,
    pub score_config: ScoreConfig,
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            edge_ranking_model_path: "model.onnx".to_string(),
            top_k_per_left: 32,
            onnx_batch_size: 128,
            emit_all_edges: false,
            predictor_config: PredictorParams::default(),
            score_config: ScoreConfig::default(),
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
