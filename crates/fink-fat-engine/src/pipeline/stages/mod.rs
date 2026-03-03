pub mod alert_inputs;
pub mod edge_builder;
pub mod fit_orbit;
pub mod load_data;
pub mod save_data;
pub mod seed_builder;
pub mod solve_run;

use std::{fmt, time::Instant};

use crate::{
    error::EngineError,
    pipeline::{
        PipelineContext,
        hooks::{PipelineHooks, StageMeta, StageProgress, StageReport},
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineStage {
    LoadPersistedData,
    IngestNights,
    BuildSeeds,
    BuildEdges,
    Solve,
    FitOrbit,
    SavePersistedData,
}

impl PipelineStage {
    /// Ordinal position in the canonical pipeline ordering.
    #[inline]
    const fn ordinal(self) -> u8 {
        match self {
            Self::LoadPersistedData => 0,
            Self::IngestNights => 1,
            Self::BuildSeeds => 2,
            Self::BuildEdges => 3,
            Self::Solve => 4,
            Self::FitOrbit => 5,
            Self::SavePersistedData => 6,
        }
    }
}

impl PartialOrd for PipelineStage {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PipelineStage {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.ordinal().cmp(&other.ordinal())
    }
}

pub(super) fn run_stage(
    stage: PipelineStage,
    hooks: &dyn PipelineHooks,
    meta: StageMeta,
    stage_run: impl FnOnce(&dyn StageProgress) -> Result<Vec<(&'static str, u64)>, EngineError>,
) -> Result<StageReport, EngineError> {
    let sink = hooks.on_stage_start(stage, meta);

    let t0 = Instant::now();
    let counters = stage_run(sink.as_ref())?;
    sink.finish();

    let report = StageReport {
        elapsed_ms: t0.elapsed().as_millis(),
        counters,
    };
    hooks.on_stage_end(stage, report.clone());
    Ok(report)
}

impl PipelineStage {
    pub fn run(
        self,
        ctx: &mut PipelineContext<'_>,
        hooks: &dyn PipelineHooks,
    ) -> Result<StageReport, EngineError> {
        match self {
            PipelineStage::IngestNights => alert_inputs::run(ctx, hooks),
            PipelineStage::BuildSeeds => seed_builder::run(ctx, hooks),
            PipelineStage::BuildEdges => edge_builder::run(ctx, hooks),
            PipelineStage::Solve => solve_run::run(ctx, hooks),
            PipelineStage::FitOrbit => fit_orbit::run(ctx, hooks),
            PipelineStage::SavePersistedData => save_data::run(ctx, hooks),
            PipelineStage::LoadPersistedData => load_data::run(ctx, hooks),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            PipelineStage::IngestNights => "Ingest nights",
            PipelineStage::BuildSeeds => "Build seeds",
            PipelineStage::BuildEdges => "Build edges",
            PipelineStage::Solve => "Solve",
            PipelineStage::FitOrbit => "Fit orbit",
            PipelineStage::LoadPersistedData => "Load persisted data",
            PipelineStage::SavePersistedData => "Save persisted data",
        }
    }
}

impl fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}
