use crate::{
    error::EngineError,
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
            stage_sink.set_total(3); // 3 main steps: components, plan, run

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
