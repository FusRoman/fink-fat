pub mod hooks;
pub mod progress_sink;
pub mod stages;

use crate::{
    engine_config::EngineConfig,
    error::EngineError,
    graph::edge::edge_prediction::EdgeRankingModelPool,
    night_id::PairingMode,
    persistence::{PersistenceManager, runtime_state::RuntimeState},
    pipeline::{
        hooks::{PipelineHooks, StageMeta, StageReport},
        progress_sink::{NoopProgress, ProgressSink},
        stages::{PipelineStage, alert_inputs::input_uri::InputUri},
    },
    solver::solver_manager::SolverManager,
};

#[derive(Clone, Debug)]
pub struct PipelineInputs {
    /// Input alert batch (typically a Parquet file).
    pub alerts_uri: InputUri,
}

#[derive(Clone, Debug)]
pub struct PipelinePlan {
    pub window: Option<PairingMode>,
    pub stages: Vec<PipelineStage>,
    pub persist: PersistPolicy,
    pub inputs: PipelineInputs,
}

#[derive(Clone, Copy, Debug)]
pub enum PersistPolicy {
    None,
    Minimal,
    Full,
}

pub struct PipelineContext<'rt> {
    pub plan: &'rt PipelinePlan,
    pub persistence: &'rt PersistenceManager,
    pub runtime_state: &'rt mut RuntimeState,
    pub engine_config: &'rt EngineConfig,
    pub edge_models: &'rt EdgeRankingModelPool,
    pub solver_manager: &'rt SolverManager,
}

pub struct PipelineRunner {
    pub plan: PipelinePlan,
}

impl PipelineRunner {
    /// Run the pipeline described by `self.plan`.
    ///
    /// Overview
    /// --------
    /// This method is the synchronous orchestration entrypoint for the engine pipeline.
    /// It validates the plan, creates a **root progress scope** for the full pipeline,
    /// then executes each requested [`PipelineStage`] in plan order.
    ///
    /// Progress reporting
    /// ------------------
    /// Progress is reported through a hierarchical [`ProgressSink`] model:
    ///
    /// - A pipeline-level scope is created first (`label = "pipeline"`), with
    ///   `total = number of stages`.
    /// - Each stage, when executed, creates its own child scope via `run_stage(...)`
    ///   (inside the stage implementation) and may create additional nested scopes.
    /// - The runner increments the pipeline-level scope by `1` after each completed stage.
    ///
    /// Hooks and cancellation
    /// ----------------------
    /// This method delegates stage lifecycle reporting (`start/progress/end`) to the
    /// provided [`PipelineHooks`] through the stage implementations and `run_stage`.
    ///
    /// If `hooks.is_cancelled()` is implemented by the caller, this method checks it:
    /// - before starting each stage,
    /// - and returns early with `EngineError::Cancelled` if cancellation is requested.
    ///
    /// Runtime state initialization
    /// ----------------------------
    /// The runtime `NightWindow` is resolved as follows:
    ///
    /// - If `ctx.plan.window` is `Some`, it is used as an explicit override and written
    ///   to `ctx.runtime_state.window` (unless already set by a previous stage).
    /// - Otherwise, stages are expected to populate it (typically `IngestNights`).
    ///
    /// Notes
    /// -----
    /// - The runner itself does not perform persistence; persistence is handled by
    ///   dedicated stages (e.g. `PersistOutputs`) and governed by `PersistPolicy`.
    /// - The root progress sink is currently a no-op. A CLI can replace it with a
    ///   concrete implementation (e.g. `indicatif`) without changing engine code.
    pub fn run(
        &self,
        ctx: &mut PipelineContext<'_>,
        hooks: &dyn PipelineHooks,
    ) -> Result<PipelineOutput, EngineError> {
        self.validate_plan()?;

        // ---------------------------------------------------------------------
        // 0) Optional: seed/override the runtime window from the plan.
        // ---------------------------------------------------------------------
        //
        // The plan-level window is an optional constraint. If provided, store it
        // in runtime state so stages can consistently use it.
        if let Some(w) = self.plan.window {
            ctx.runtime_state.window = Some(w);
        }

        // ---------------------------------------------------------------------
        // 1) Root progress scope (pipeline-level).
        // ---------------------------------------------------------------------
        //
        // The runner creates a pipeline-level scope with "1 unit = 1 completed stage".
        // Stages are responsible for creating their own child scopes and finer-grained
        // progress reporting via `run_stage(...)` and nested `ProgressSink::child(...)`.
        let root_progress: NoopProgress = NoopProgress;

        let pipeline_sink = root_progress.child(StageMeta {
            label: "pipeline".to_string(),
            total: Some(self.plan.stages.len() as u64),
        });
        pipeline_sink.set_total(self.plan.stages.len() as u64);

        // ---------------------------------------------------------------------
        // 2) Execute stages in plan order.
        // ---------------------------------------------------------------------
        let mut reports: Vec<(PipelineStage, StageReport)> =
            Vec::with_capacity(self.plan.stages.len());

        for &stage in &self.plan.stages {
            // Optional cancellation point: before starting each stage.
            if hooks.is_cancelled() {
                pipeline_sink.finish();
                return Err(EngineError::Cancelled);
            }

            // Execute stage, propagating the pipeline-level progress sink.
            // The stage implementation will create its own child scope.
            let report = stage.run(ctx, hooks, pipeline_sink.as_ref())?;

            // Record report for the pipeline output.
            reports.push((stage, report));

            // 1 unit = 1 stage completed
            pipeline_sink.inc(1);
        }

        pipeline_sink.finish();

        Ok(PipelineOutput { reports })
    }

    fn validate_plan(&self) -> Result<(), EngineError> {
        if self.plan.stages.is_empty() {
            return Err(EngineError::InvalidPlan("no stages specified"));
        }

        // Stages must be in strictly increasing canonical order.
        if self.plan.stages.windows(2).any(|w| w[0] >= w[1]) {
            return Err(EngineError::InvalidPlan(
                "stages must be in strictly increasing order \
                 (Load < Ingest < Seeds < Edges < Solve < FitOrbit < Save)",
            ));
        }

        // Avoid SavePersistedData without persistence policy.
        if self.plan.stages.contains(&PipelineStage::SavePersistedData)
            && matches!(self.plan.persist, PersistPolicy::None)
        {
            return Err(EngineError::InvalidPlan(
                "SavePersistedData stage requires persist != None",
            ));
        }

        Ok(())
    }
}
pub struct PipelineOutput {
    pub reports: Vec<(PipelineStage, StageReport)>,
    // éventuellement: ids de runs, paths persistés, stats globales
}
