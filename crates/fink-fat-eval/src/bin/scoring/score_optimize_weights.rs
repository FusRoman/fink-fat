use anyhow::Result;
use clap::Parser;
use fink_fat_eval::scoring::optimizer_pipeline_weights::{Cli, run};

fn main() -> Result<()> {
    let cli = Cli::parse();
    run(&cli)
}
