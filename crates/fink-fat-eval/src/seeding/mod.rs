use anyhow::Result;
use fink_fat_engine::pipeline::PipelineContext;

pub fn seeding_evaluation(runtime_context: &PipelineContext) -> Result<()> {
    println!("Running seeding evaluation with context: {runtime_context:#?}");
    Ok(())
}
