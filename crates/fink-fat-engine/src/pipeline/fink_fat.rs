use camino::Utf8Path;

use crate::{
    engine_config::{EngineConfig, load_engine_config_validated},
    graph::{InterNightGraph, edge::edge_prediction::EdgeRankingModelPool},
    persistence::alert_store::AlertStore,
    pipeline::seed_store::SeedStore,
};

/// Runtime engine façade borrowing a loaded state.
///
/// This struct is intentionally **not self-referential**:
/// it borrows the `AlertStore`, and the `SeedStore`/`InterNightGraph` are
/// temporary borrowed views built from owned persisted payloads.
pub struct FinkFat<'seed_lf, 'alert_lf> {
    pub engine_config: EngineConfig,
    pub edge_ranking_models: EdgeRankingModelPool,

    /// Backing storage for alerts (owned elsewhere, typically `RuntimeState`).
    pub alert_store: &'alert_lf AlertStore,

    /// Borrowed seed view built from persisted seeds + `alert_store`.
    pub seed_store: &'seed_lf SeedStore<'alert_lf>,

    /// Borrowed graph view built from persisted edges + `seed_store`.
    pub graph: InterNightGraph<'seed_lf, 'alert_lf>,
}

impl<'seed_lf, 'alert_lf> FinkFat<'seed_lf, 'alert_lf> {
    /// Load the engine config and build the lazy per-thread model pool.
    pub fn load_core(
        engine_config_path: impl AsRef<Utf8Path>,
    ) -> (EngineConfig, EdgeRankingModelPool) {
        let Ok(engine_config) = load_engine_config_validated(engine_config_path.as_ref()) else {
            panic!(
                "Failed to load engine config from path: {}",
                engine_config_path.as_ref()
            );
        };

        let model_path = &engine_config.edges.edge_ranking_model_path;

        // No heavy load here: models are created lazily per Rayon worker thread.
        let edge_ranking_models = EdgeRankingModelPool::new(model_path);

        (engine_config, edge_ranking_models)
    }

    /// Build a `FinkFat` runtime instance from already-loaded stores/views.
    pub fn from_parts(
        engine_config: EngineConfig,
        edge_ranking_models: EdgeRankingModelPool,
        alert_store: &'alert_lf AlertStore,
        seed_store: &'seed_lf SeedStore<'alert_lf>,
        graph: InterNightGraph<'seed_lf, 'alert_lf>,
    ) -> Self {
        Self {
            engine_config,
            edge_ranking_models,
            alert_store,
            seed_store,
            graph,
        }
    }
}
