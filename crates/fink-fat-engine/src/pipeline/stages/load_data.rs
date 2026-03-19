use crate::{
    error::EngineError,
    pipeline::{
        PipelineContext,
        hooks::{PipelineHooks, StageMeta, StageReport},
        stages::{PipelineStage, run_stage},
    },
};

pub fn run(
    ctx: &mut PipelineContext<'_>,
    hooks: &dyn PipelineHooks,
) -> Result<StageReport, EngineError> {
    run_stage(
        PipelineStage::LoadPersistedData,
        hooks,
        StageMeta {
            label: PipelineStage::LoadPersistedData.label().to_string(),
            total: Some(6), // We don't know the total number of nights/alerts/seeds/edges until we load the manifest and window.
        },
        |stage_sink| {
            tracing::debug!("LoadPersistedData starting");

            // Delegate to PersistenceManager::load_runtime_state which:
            //   1. Loads/initializes the manifest.
            //   2. Computes the sliding window from engine config.
            //   3. Loads alerts and seeds for nights in the window.
            //   4. Replays the edge journal (snapshot + deltas) into a graph.
            let state = ctx
                .persistence
                .load_runtime_state(ctx.engine_config, stage_sink)?;

            let n_nights = state.alert_store.n_nights() as u64;
            let n_alerts = state.alert_store.n_alerts() as u64;
            let n_seeds = state
                .seed_store
                .iter()
                .map(|(_, v)| v.len() as u64)
                .sum::<u64>();
            let n_edges = state.graph.edges.len() as u64;

            tracing::debug!(
                n_nights,
                n_alerts,
                n_seeds,
                n_edges,
                "LoadPersistedData complete",
            );

            // Populate the runtime state with loaded data.
            ctx.runtime_state.manifest = state.manifest;
            ctx.runtime_state.alert_store = state.alert_store;
            ctx.runtime_state.seed_store = state.seed_store;
            ctx.runtime_state.graph = state.graph;

            stage_sink.inc(1);

            Ok(vec![
                ("nights_loaded", n_nights),
                ("alerts_loaded", n_alerts),
                ("seeds_loaded", n_seeds),
                ("edges_loaded", n_edges),
            ])
        },
    )
}
