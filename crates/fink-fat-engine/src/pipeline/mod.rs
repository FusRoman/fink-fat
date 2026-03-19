pub mod hooks;
pub mod stages;

use crate::{
    engine_config::{EngineConfig, pipeline_policy::PersistPolicy},
    error::EngineError,
    graph::edge::edge_prediction::EdgeRankingModelPool,
    persistence::{PersistenceManager, runtime_state::RuntimeState},
    pipeline::{
        hooks::{PipelineHooks, StageReport},
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
    pub stages: Vec<PipelineStage>,
    pub persist: PersistPolicy,
    pub inputs: PipelineInputs,
}

#[derive(Debug)]
pub struct PipelineContext<'rt> {
    pub plan: &'rt PipelinePlan,
    pub persistence: &'rt PersistenceManager,
    pub runtime_state: &'rt mut RuntimeState,
    pub engine_config: &'rt EngineConfig,
    pub edge_models: &'rt Option<EdgeRankingModelPool>,
    pub solver_manager: &'rt SolverManager,
}

pub struct PipelineRunner {
    pub plan: PipelinePlan,
}

impl PipelineRunner {
    pub fn run(
        &self,
        ctx: &mut PipelineContext<'_>,
        hooks: &dyn PipelineHooks,
    ) -> Result<PipelineOutput, EngineError> {
        self.validate_plan()?;

        tracing::info!(
            stages = ?self.plan.stages.iter().map(|s| s.label()).collect::<Vec<_>>(),
            "pipeline starting",
        );

        // ---------------------------------------------------------------------
        // Execute stages in plan order.
        // ---------------------------------------------------------------------
        // Each stage receives the hooks and creates its own progress scope via
        // `hooks.on_stage_start(stage, meta)` inside `run_stage`.
        // Cancellation is checked before each stage.
        let mut reports: Vec<(PipelineStage, StageReport)> =
            Vec::with_capacity(self.plan.stages.len());

        for &stage in &self.plan.stages {
            // Optional cancellation point: before starting each stage.
            if hooks.is_cancelled() {
                return Err(EngineError::Cancelled);
            }

            let report = stage.run(ctx, hooks)?;
            reports.push((stage, report));
        }

        tracing::info!("pipeline complete");

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
