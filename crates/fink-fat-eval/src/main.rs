use anyhow::Result;

use clap::Parser;
use fink_fat_engine::pipeline::stages::PipelineStage;

pub mod cli;
pub mod edges;
pub mod logging;
pub mod runner;
pub mod seeding;
pub mod truth_sso;

use cli::{Cli, Commands};

use crate::{edges::edge_evaluation, runner::run_fink_fat};

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::SeedingEval(args) => {
            let plot_dir = args.plot_dir.clone();
            run_fink_fat(
                args.common,
                &[PipelineStage::IngestNights, PipelineStage::BuildSeeds],
                move |ctx, truth| {
                    seeding::seeding_evaluation(ctx, truth)?;
                    if let Some(ref dir) = plot_dir {
                        seeding::plots::seeding_plots(ctx, truth, dir)?;
                    }
                    Ok(())
                },
            )?
        }
        Commands::EdgeEval(args) => run_fink_fat(
            args.common,
            &[
                PipelineStage::IngestNights,
                PipelineStage::BuildSeeds,
                PipelineStage::BuildEdges,
            ],
            edge_evaluation,
        )?,
    };

    Ok(())
}
