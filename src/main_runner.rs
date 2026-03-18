use std::sync::Arc;

use chrono::Utc;
use fink_fat_engine::{
    engine_config::log_level::LogLevel,
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
use indicatif::MultiProgress;

use crate::{
    init_cli::FinkFatCliArgs, load_config, logging::init_logging, progress::IndicatifHooks,
};

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

pub fn fink_fat_runner(cli_args: FinkFatCliArgs) -> Result<(), EngineError> {
    let engine_config = load_config(&cli_args.config)?;
    let persistence = PersistenceManager::open_or_create(engine_config.clone().storage_path_buf())?;
    let model_pool = engine_config
        .edges
        .edge_ranking_model_path
        .clone()
        .map(|path_model| EdgeRankingModelPool::new(&path_model));

    // ── Progress hooks ────────────────────────────────────────────────────────
    // When both `--progress` and `--logs` are active we share one MultiProgress
    // so the logging layer can route lines through `mp.println()` instead of
    // writing directly to stderr (which would smear the progress bars).
    let (hooks, hooks_mp): (Box<dyn PipelineHooks>, Option<Arc<MultiProgress>>) =
        if cli_args.progress {
            let hooks = IndicatifHooks::new();
            let mp = hooks.multi_progress();
            (Box::new(hooks), Some(mp))
        } else {
            (Box::new(NoopHooks), None)
        };

    // ── Logging setup ─────────────────────────────────────────────────────────
    // The `_logging_guard` must stay alive until the process exits: dropping it
    // signals the background file-writer thread to flush and terminate.
    let _logging_guard = if cli_args.logs {
        let level = log_level_to_tracing(engine_config.log_level);

        // Use the current UTC time as a filesystem-safe session identifier.
        let run_id = Utc::now().format("%Y-%m-%dT%H-%M-%S").to_string();
        let log_path = persistence.layout().log_run_path(&run_id);

        let guard =
            init_logging(level, &log_path, hooks_mp).map_err(|e| EngineError::StageFailed {
                stage: PipelineStage::LoadPersistedData,
                message: format!("failed to initialise logging: {e}"),
            })?;

        tracing::info!(
            log_file = %log_path,
            level = %engine_config.log_level,
            "logging initialised",
        );

        Some(guard)
    } else {
        None
    };

    // ── Pipeline plan ─────────────────────────────────────────────────────────
    let plan = PipelinePlan {
        stages: FULL_WITH_PERSISTENCE.to_vec(),
        persist: engine_config.pipeline_policy,
        inputs: PipelineInputs {
            alerts_uri: cli_args.alerts,
        },
    };

    let runner = PipelineRunner { plan: plan.clone() };

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

    runner.run(&mut ctx, hooks.as_ref())?;

    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn log_level_to_tracing(level: LogLevel) -> tracing::Level {
    match level {
        LogLevel::Trace => tracing::Level::TRACE,
        LogLevel::Debug => tracing::Level::DEBUG,
        LogLevel::Info => tracing::Level::INFO,
        LogLevel::Warn => tracing::Level::WARN,
        LogLevel::Error => tracing::Level::ERROR,
    }
}
