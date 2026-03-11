use anyhow::Result;
use fink_fat_engine::pipeline::PipelineContext;

use crate::truth_sso::TruthSSOMap;

pub fn edge_evaluation(runtime_context: &PipelineContext, _truth: &TruthSSOMap) -> Result<()> {
    println!("Running edge evaluation with context: {runtime_context:#?}");
    Ok(())
}
