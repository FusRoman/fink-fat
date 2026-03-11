use anyhow::Result;
use fink_fat_engine::pipeline::PipelineContext;

pub fn edge_evaluation(runtime_context: &PipelineContext) -> Result<()> {
    println!("Running edge evaluation with context: {runtime_context:#?}");
    Ok(())
}
