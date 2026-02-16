use crate::{
    engine_config::EngineConfig,
    graph::edge::edge_prediction::EdgeRankingModelPool,
    night_id::PairingMode,
    persistence::{
        alert_store::AlertStore, error::PersistenceError, graph::GraphOwned, manifest::Manifest,
        seed_store::SeedStoreOwned,
    },
    pipeline::fink_fat::FinkFat,
};

/// Loaded runtime state built from persisted artifacts.
///
/// This matches the runtime needs:
/// - `AlertStore` owns the alert vectors (per night).
/// - `SeedStoreOwned` owns `SeedNodeOwned`.
/// - `InterNightGraph<'seed,'alert>` owns `Edge<'seed,'alert>` that borrow seeds.
///
/// Notes
/// -----
/// This struct is meant to be created and then moved into your runtime engine
/// (or into a `FinkFat` instance).
pub struct RuntimeState {
    pub manifest: Manifest,
    pub window: Option<PairingMode>,
    pub alert_store: AlertStore,
    pub seed_store: SeedStoreOwned,
    pub graph: GraphOwned,
}

impl RuntimeState {
    /// Build borrowed runtime views and run a closure with a temporary `FinkFat`.
    ///
    /// This avoids self-referential structs:
    /// - `RuntimeState` owns persisted payloads (`AlertStore`, `SeedStoreOwned`, `GraphOwned`)
    /// - we build borrowed views (`SeedStore<'alert>`, `InterNightGraph<'seed,'alert>`)
    /// - we instantiate `FinkFat` only for the duration of `f`.
    ///
    /// Notes
    /// -----
    /// - The returned `FinkFat` **must not escape** the closure: it contains borrows
    ///   tied to `self`.
    pub fn with_borrowed<R>(
        &self,
        engine_config: EngineConfig,
        edge_ranking_models: EdgeRankingModelPool,
        f: impl for<'seed_lf, 'alert_lf> FnOnce(FinkFat<'seed_lf, 'alert_lf>) -> R,
    ) -> Result<R, PersistenceError> {
        let alert_store = &self.alert_store;

        let seed_store = self.seed_store.to_borrowed(alert_store)?;

        let graph = self.graph.to_borrowed(&seed_store)?;

        let ff = FinkFat::from_parts(
            engine_config,
            edge_ranking_models,
            alert_store,
            &seed_store,
            graph,
        );

        Ok(f(ff))
    }
}
