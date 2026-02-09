use camino::Utf8Path;

use crate::{
    engine_config::{EngineConfig, load_engine_config_validated},
    graph::{InterNightGraph, edge::edge_prediction::EdgeRankingModelPool},
    pipeline::{alert_store::AlertStore, seed_store::SeedStore},
};

pub struct FinkFat<'seed_lf, 'alert_lf> {
    pub engine_config: EngineConfig,
    pub edge_ranking_models: EdgeRankingModelPool,
    pub alert_store: AlertStore,
    pub seed_store: SeedStore<'alert_lf>,
    pub graph: InterNightGraph<'seed_lf, 'alert_lf>,
}

impl<'seed_lf, 'alert_lf> FinkFat<'seed_lf, 'alert_lf> {
    pub fn new(engine_config_path: impl AsRef<Utf8Path>) -> Self {
        let Ok(engine_config) = load_engine_config_validated(engine_config_path.as_ref()) else {
            panic!(
                "Failed to load engine config from path: {}",
                engine_config_path.as_ref()
            );
        };

        let model_path = &engine_config.edges.edge_ranking_model_path;

        // No heavy load here: models are created lazily per Rayon worker thread.
        let edge_ranking_models = EdgeRankingModelPool::new(model_path);

        Self {
            engine_config,
            edge_ranking_models,
            alert_store: AlertStore::new(),
            seed_store: SeedStore::new(),
            graph: InterNightGraph::new(),
        }
    }
}
