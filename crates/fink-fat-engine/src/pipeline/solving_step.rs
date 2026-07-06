use crate::{
    error::EngineError,
    graph::AlertLinkageDAG,
    pipeline::hooks::{StageMeta, StageProgress},
    seeding::store::SeedStore,
    solver::{components::ConnectedComponents, solver_manager::SolverManager},
};

pub fn solve_graph(
    solver_manager: &SolverManager,
    seed_store: &SeedStore,
    graph: &AlertLinkageDAG,
    stage_sink: &dyn StageProgress,
) -> Result<(), EngineError> {
    // -----------------------------------------------------------------
    // 1) Construct connected components
    // -----------------------------------------------------------------
    let components = ConnectedComponents::compute(seed_store, graph, true)?;

    // -----------------------------------------------------------------
    // 2) Make solve plan
    // -----------------------------------------------------------------
    let plan = solver_manager.make_plan(&components);

    // -----------------------------------------------------------------
    // 3) Run solver plan
    // -----------------------------------------------------------------
    let running_plan_sink = stage_sink.child(StageMeta {
        label: "running solver plan ...".to_string(),
        total: Some(plan.items.len() as u64),
    });
    let results = solver_manager.run_plan(
        &components,
        graph,
        seed_store,
        &plan,
        running_plan_sink.as_ref(),
    );
    running_plan_sink.finish();

    Ok(())
}
