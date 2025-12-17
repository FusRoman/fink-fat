use crate::engine_config::{
    propagator_config::PredictorParams, score_config::InterNightScoreConfig,
};

pub struct EdgeConfig {
    pub top_k_per_left: usize,
    pub max_total_edges: Option<usize>,
    pub predictor_config: PredictorParams,
    pub score_config: InterNightScoreConfig,
}
