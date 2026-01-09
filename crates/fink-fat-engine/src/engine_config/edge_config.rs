use serde::{Deserialize, Serialize};

use crate::engine_config::{
    error::EdgeConfigError, propagator_config::PredictorParams, score_config::ScoreConfig,
};

/// File-level configuration for graph edge generation.
///
/// Keep this minimal: only the knobs that belong to "edge creation".
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EdgeConfigFile {
    pub top_k_per_left: usize,
    pub max_total_edges: Option<usize>,
}

impl Default for EdgeConfigFile {
    fn default() -> Self {
        Self {
            top_k_per_left: 32,
            max_total_edges: None,
        }
    }
}

impl EdgeConfigFile {
    pub fn validate(&self) -> Result<(), EdgeConfigError> {
        if self.top_k_per_left == 0 {
            return Err(EdgeConfigError::TopKPerLeftZero);
        }
        if let Some(m) = self.max_total_edges {
            if m == 0 {
                return Err(EdgeConfigError::MaxTotalEdgesZero);
            }
        }
        Ok(())
    }
}

/// Runtime configuration actually used by the engine.
pub struct EdgeConfig {
    pub top_k_per_left: usize,
    pub max_total_edges: Option<usize>,
    pub predictor_config: PredictorParams,
    pub score_config: ScoreConfig,
}
