use fink_fat_engine::{
    error::EngineError,
    graph::edge::edge_prediction::EdgeRankingModelPool,
    persistence::{PersistenceManager, runtime_state::RuntimeState},
    pipeline::{
        PipelineContext, PipelineInputs, PipelinePlan, PipelineRunner,
        hooks::{NoopHooks, PipelineHooks},
        stages::PipelineStage,
    },
    solver::solver_manager::SolverManager,
};

use crate::{init_cli::NightRunArgs, load_config, progress::IndicatifHooks};

/// Full persistence one night round-trip:
/// `LoadPersistedData → Ingest → Seeds → Edges → Solve → FitOrbit → Save`.
const FULL_WITH_PERSISTENCE: &[PipelineStage] = &[
    PipelineStage::LoadPersistedData,
    PipelineStage::IngestNights,
    PipelineStage::BuildSeeds,
    PipelineStage::BuildEdges,
    PipelineStage::Solve,
    PipelineStage::FitOrbit,
    PipelineStage::SavePersistedData,
];

pub fn run_single_night(cli_args: NightRunArgs) -> Result<(), EngineError> {
    let engine_config = load_config(&cli_args.config)?;
    let persistence = PersistenceManager::open_or_create(engine_config.clone().storage_path_buf())?;
    let model_pool = engine_config
        .edges
        .edge_ranking_model_path
        .clone()
        .map(|path_model| EdgeRankingModelPool::new(&path_model));

    let plan = PipelinePlan {
        stages: FULL_WITH_PERSISTENCE.to_vec(),
        persist: engine_config.pipeline_policy,
        inputs: PipelineInputs {
            alerts_uri: cli_args.alerts,
        },
    };

    let runner = PipelineRunner { plan: plan.clone() };

    let hooks: Box<dyn PipelineHooks> = if cli_args.progress {
        Box::new(IndicatifHooks::new())
    } else {
        Box::new(NoopHooks)
    };

    let solver_manager = SolverManager {
        policy: engine_config.solver_config.solver_policy,
        bounded_beam_config: engine_config.solver_config.bounded_beam.clone(),
    };

    let mut runtime_state = RuntimeState::new();
    let mut ctx = PipelineContext {
        plan: &plan,
        persistence: &persistence,
        runtime_state: &mut runtime_state,
        engine_config: &engine_config,
        edge_models: &model_pool,
        solver_manager: &solver_manager,
    };

    let run_results = runner.run(&mut ctx, hooks.as_ref())?;

    println!("Pipeline run complete. Final stage reports:");
    for (stage_meta, stage_report) in run_results.reports {
        println!("  Stage: {}", stage_meta.label());
        println!("    elapsed_time: {:?} ms", stage_report.elapsed_ms);
        for (str_counter, counter) in stage_report.counters {
            println!("      {}: {}", str_counter, counter);
        }
    }

    Ok(())
}
