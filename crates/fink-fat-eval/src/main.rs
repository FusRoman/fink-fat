use anyhow::Result;

use clap::Parser;
use fink_fat_engine::pipeline::stages::PipelineStage;

pub mod cli;
pub mod edges;
pub mod logging;
pub mod model_eval;
pub mod progress;
pub mod runner;
pub mod seeding;
pub mod solver;
pub mod truth_sso;

use cli::{Cli, Commands};

use crate::{runner::run_fink_fat, solver::solver_evaluation};

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::SeedingEval(args) => {
            let plot_dir = args.common.plot_dir.clone();
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
        Commands::EdgeEval(args) => {
            let plot_dir = args.common.plot_dir.clone();
            let export_features = args.export_features.clone();
            run_fink_fat(
                args.common,
                &[
                    PipelineStage::IngestNights,
                    PipelineStage::BuildSeeds,
                    PipelineStage::BuildEdges,
                ],
                move |ctx, truth| {
                    edges::edge_evaluation(
                        ctx,
                        truth,
                        plot_dir.as_deref(),
                        export_features.as_deref(),
                    )
                },
            )?
        }
        Commands::SolverEval(args) => {
            let plot_dir = args.common.plot_dir.clone();
            run_fink_fat(
                args.common,
                &[
                    PipelineStage::IngestNights,
                    PipelineStage::BuildSeeds,
                    PipelineStage::BuildEdges,
                    PipelineStage::Solve,
                ],
                move |ctx, truth| {
                    solver_evaluation(ctx, truth, plot_dir.as_deref())?;
                    Ok(())
                },
            )?
        }
        Commands::ModelEval(args) => {
            model_eval::model_evaluation(
                &args.features_parquet,
                &args.xgb_params,
                args.plot_dir.as_deref(),
                args.onnx_intra_threads,
            )?;
        }
    };

    Ok(())
}
