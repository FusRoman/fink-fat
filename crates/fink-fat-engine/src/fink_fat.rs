use camino::Utf8Path;

use crate::{
    engine_config::{EngineConfig, load_engine_config_validated},
    graph::edge::edge_prediction::EdgeRankingModel,
};

pub struct FinkFat {
    pub engine_config: EngineConfig,
    pub edge_ranking_model: EdgeRankingModel,
}

impl FinkFat {
    pub fn new(engine_config_path: impl AsRef<Utf8Path>) -> Self {
        let Ok(engine_config) = load_engine_config_validated(engine_config_path.as_ref()) else {
            panic!(
                "Failed to load engine config from path: {}",
                engine_config_path.as_ref()
            );
        };

        let model_path = &engine_config.edges.edge_ranking_model_path;
        let Ok(edge_ranking_model) = EdgeRankingModel::load_edge_ranking_model(model_path) else {
            panic!(
                "Failed to load edge ranking model from path: {}",
                model_path
            );
        };
        Self {
            engine_config,
            edge_ranking_model,
        }
    }
}
