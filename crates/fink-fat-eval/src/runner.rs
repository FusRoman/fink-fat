use anyhow::{Context, Result};
use camino::Utf8Path;
use chrono::Utc;
use fink_fat_engine::{
    engine_config::{EngineConfig, load_engine_config_validated, log_level::LogLevel},
    graph::edge::edge_prediction::EdgeRankingModelPool,
    persistence::{PersistenceManager, runtime_state::RuntimeState},
    pipeline::{
        PipelineContext, PipelineInputs, PipelinePlan, PipelineRunner,
        hooks::{NoopHooks, PipelineHooks},
        stages::PipelineStage,
    },
    solver::solver_manager::SolverManager,
};

use crate::{cli::CommonArgs, logging, truth_sso::TruthSSO};

pub fn load_config(config_path: &Utf8Path) -> Result<EngineConfig> {
    load_engine_config_validated(config_path).context("failed to load engine config")
}

fn log_level_to_tracing(level: LogLevel) -> tracing::Level {
    match level {
        LogLevel::Trace => tracing::Level::TRACE,
        LogLevel::Debug => tracing::Level::DEBUG,
        LogLevel::Info => tracing::Level::INFO,
        LogLevel::Warn => tracing::Level::WARN,
        LogLevel::Error => tracing::Level::ERROR,
    }
}

pub fn run_fink_fat(
    cli_args: CommonArgs,
    pipeline_stages: &[PipelineStage],
    evaluation_postprocess: impl Fn(&PipelineContext, &TruthSSO) -> Result<()>,
) -> Result<()> {
    let engine_config = load_config(&cli_args.config)?;

    let alerts_url = cli_args
        .alerts
        .parse()
        .context("failed to parse alerts URI")?;
    let alerts_path = Utf8Path::new(alerts_url.path());
    let truth_sso =
        TruthSSO::load(alerts_path).context("failed to load truth SSO map from alerts URI")?;

    let persistence = PersistenceManager::open_or_create(engine_config.clone().storage_path_buf())
        .context("failed to open persistence")?;

    // ── Logging setup ─────────────────────────────────────────────────────────
    // The `_logging_guard` must stay alive until the process exits: dropping it
    // signals the background file-writer thread to flush and terminate.
    let run_id = Utc::now().format("%Y-%m-%dT%H-%M-%S").to_string();
    let log_path = persistence.layout().log_run_path(&run_id);
    let _logging_guard =
        logging::init_logging(log_level_to_tracing(engine_config.log_level), &log_path)
            .map_err(|e| anyhow::anyhow!("failed to initialise logging: {e}"))?;
    tracing::info!(
        log_file = %log_path,
        level = %engine_config.log_level,
        "logging initialised",
    );

    let model_pool = engine_config
        .edges
        .edge_ranking_model_path
        .clone()
        .map(|path| EdgeRankingModelPool::new(&path));

    // ── Progress hooks ────────────────────────────────────────────────────────
    // When both `--progress` and `--logs` are active we share one MultiProgress
    // so the logging layer can route lines through `mp.println()` instead of
    // writing directly to stderr (which would smear the progress bars).
    let hooks: Box<dyn PipelineHooks> = Box::new(NoopHooks);

    // ── Pipeline plan ─────────────────────────────────────────────────────────
    let plan = PipelinePlan {
        stages: pipeline_stages.to_vec(),
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

    runner
        .run(&mut ctx, hooks.as_ref())
        .context("pipeline failed")?;

    evaluation_postprocess(&ctx, &truth_sso).context("post-processing failed")?;

    Ok(())
}
