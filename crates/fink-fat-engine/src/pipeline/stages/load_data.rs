use std::time::{SystemTime, UNIX_EPOCH};

use crate::{
    error::EngineError,
    pipeline::{
        PipelineContext,
        hooks::{PipelineHooks, StageMeta, StageReport},
        progress_sink::ProgressSink,
        stages::{PipelineStage, run_stage},
    },
};

/// Return the current Unix timestamp (seconds).
fn now_unix_s() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

pub fn run(
    ctx: &mut PipelineContext<'_>,
    hooks: &dyn PipelineHooks,
    stage_sink: &dyn ProgressSink,
) -> Result<StageReport, EngineError> {
    run_stage(
        PipelineStage::LoadPersistedData,
        hooks,
        StageMeta {
            label: PipelineStage::LoadPersistedData.label().to_string(),
            total: Some(1),
        },
        stage_sink,
        |stage_sink| {
            stage_sink.set_total(1);

            let created_unix_s = now_unix_s();

            // Delegate to PersistenceManager::load_runtime_state which:
            //   1. Loads/initializes the manifest.
            //   2. Computes the sliding window from engine config.
            //   3. Loads alerts and seeds for nights in the window.
            //   4. Replays the edge journal (snapshot + deltas) into a graph.
            let state = ctx
                .persistence
                .load_runtime_state(ctx.engine_config, created_unix_s)?;

            let n_nights = state.alert_store.n_nights() as u64;
            let n_alerts = state.alert_store.n_alerts() as u64;
            let n_seeds = state.seed_store.iter().map(|(_, v)| v.len() as u64).sum::<u64>();
            let n_edges = state.graph.edges.len() as u64;

            // Populate the runtime state with loaded data.
            ctx.runtime_state.manifest = state.manifest;
            ctx.runtime_state.window = state.window;
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
