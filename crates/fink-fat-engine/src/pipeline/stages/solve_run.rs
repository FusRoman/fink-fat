use crate::{
    error::EngineError,
    night_id::PairingMode,
    pipeline::{
        PipelineContext,
        hooks::{PipelineHooks, StageMeta, StageReport},
        progress_sink::ProgressSink,
        stages::{PipelineStage, run_stage},
    },
    solver::{SolverOutput, components::ConnectedComponents},
};

pub fn run(
    ctx: &mut PipelineContext<'_>,
    hooks: &dyn PipelineHooks,
    stage_sink: &dyn ProgressSink,
) -> Result<StageReport, EngineError> {
    run_stage(
        PipelineStage::Solve,
        hooks,
        StageMeta {
            label: PipelineStage::Solve.label().to_string(),
            total: None,
        },
        stage_sink,
        |stage_sink| {
            // -----------------------------------------------------------------
            // 0) Preconditions
            // -----------------------------------------------------------------
            let window: PairingMode =
                ctx.runtime_state
                    .window
                    .ok_or_else(|| EngineError::StageFailed {
                        stage: PipelineStage::Solve,
                        message: "missing RuntimeState.window (PairingMode)".to_string(),
                    })?;

            let edge_config = &ctx.engine_config.edges;

            // -----------------------------------------------------------------
            // 1) Build borrowed seeds for this window
            // -----------------------------------------------------------------
            // IMPORTANT: `seed_store` must stay alive until `graph_rt.to_owned()`,
            // because `RuntimeGraph` edges borrow `SeedNode` references.
            let seed_store = ctx
                .runtime_state
                .seed_store
                .to_borrowed_window(&ctx.runtime_state.alert_store, window)
                .map_err(|e| EngineError::StageFailed {
                    stage: PipelineStage::Solve,
                    message: format!("seed_store.to_borrowed_window failed: {e:?}"),
                })?;

            // -----------------------------------------------------------------
            // 2) Prepare runtime graph (borrowed)
            // -----------------------------------------------------------------
            let graph = ctx
                .runtime_state
                .graph
                .to_borrowed(&seed_store)
                .map_err(|e| EngineError::StageFailed {
                    stage: PipelineStage::Solve,
                    message: format!("graph.to_borrowed failed: {e:?}"),
                })?;

            // -----------------------------------------------------------------
            // 3) Construct connected components
            // -----------------------------------------------------------------
            let components = ConnectedComponents::compute(&seed_store, &graph, true);

            // -----------------------------------------------------------------
            // 4) Make solve plan
            // -----------------------------------------------------------------
            let plan = ctx.solver_manager.make_plan(&components);

            let results = ctx
                .solver_manager
                .run_plan(&components, &graph, &seed_store, &plan);

            let all_hypothesis = SolverOutput::merge_solver_output(&results);

            Ok(vec![])
        },
    )
}
