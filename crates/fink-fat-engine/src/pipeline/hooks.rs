use crate::pipeline::PipelineStage;

pub trait PipelineHooks: Send + Sync {
    fn on_stage_start(&self, stage: PipelineStage, meta: StageMeta);
    fn on_stage_progress(&self, stage: PipelineStage, delta: u64);
    fn on_stage_end(&self, stage: PipelineStage, report: StageReport);

    fn is_cancelled(&self) -> bool {
        false
    } // optionnel
}

#[derive(Clone, Debug, Default)]
pub struct StageMeta {
    pub label: String,
    pub total: Option<u64>,
}

#[derive(Clone, Debug, Default)]
pub struct StageReport {
    pub elapsed_ms: u128,
    pub counters: Vec<(&'static str, u64)>,
}
