use crate::{
    error::EngineError,
    pipeline::{
        PipelineContext,
        hooks::{PipelineHooks, StageMeta, StageReport},
        stages::{PipelineStage, run_stage},
    },
    solver::{SolverOutput, components::ConnectedComponents},
};

pub fn run(
    ctx: &mut PipelineContext<'_>,
    hooks: &dyn PipelineHooks,
) -> Result<StageReport, EngineError> {
    run_stage(
        PipelineStage::Solve,
        hooks,
        StageMeta {
            label: PipelineStage::Solve.label().to_string(),
            total: Some(3),
        },
        |stage_sink| {
            // -----------------------------------------------------------------
            // Early exit: no edges means nothing to link across nights.
            // -----------------------------------------------------------------
            if ctx.runtime_state.graph.edges.is_empty() {
                tracing::debug!("Solve: no edges in graph, skipping solver");
                return Ok(vec![
                    ("components", 0),
                    ("plan_items", 0),
                    ("hypotheses", 0),
                ]);
            }

            // -----------------------------------------------------------------
            // 1) Construct connected components
            // -----------------------------------------------------------------
            let components = ConnectedComponents::compute(
                &ctx.runtime_state.seed_store,
                &ctx.runtime_state.graph,
                true,
            )?;

            stage_sink.inc(1);

            // -----------------------------------------------------------------
            // 2) Make solve plan
            // -----------------------------------------------------------------
            let plan = ctx.solver_manager.make_plan(&components);

            let running_plan_sink = stage_sink.child(StageMeta {
                label: "running solver plan ...".to_string(),
                total: Some(plan.items.len() as u64),
            });

            let results = ctx.solver_manager.run_plan(
                &components,
                &ctx.runtime_state.graph,
                &ctx.runtime_state.seed_store,
                &plan,
                running_plan_sink.as_ref(),
            );
            running_plan_sink.finish();

            stage_sink.inc(1);

            ctx.runtime_state.track_hypotheses = SolverOutput::merge_solver_output(&results);

            stage_sink.inc(1);

            Ok(vec![
                ("components", components.n_components as u64),
                ("plan_items", plan.items.len() as u64),
                (
                    "hypotheses",
                    ctx.runtime_state.track_hypotheses.len() as u64,
                ),
            ])
        },
    )
}
