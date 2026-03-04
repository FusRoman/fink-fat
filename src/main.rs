pub mod init_cli;
pub mod logging;
pub mod progress;
pub mod single_night_runner;

use camino::Utf8Path;
use clap::Parser;
use fink_fat_engine::{
    engine_config::{EngineConfig, load_engine_config_validated},
    error::EngineError,
};
use init_cli::{Cli, Commands};

use crate::single_night_runner::run_single_night;

pub fn load_config(config_path: &Utf8Path) -> Result<EngineConfig, EngineError> {
    Ok(load_engine_config_validated(config_path)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::NightRun(args) => {
            run_single_night(args)?;
        }
        Commands::Reprocessing(args) => {
            println!("Reprocessing mode");
            println!("  alerts : {}", args.alerts);
            println!("  config : {}", args.config);
            let engine_config = load_config(&args.config)?;
            println!("Loaded engine config: {:#?}", engine_config);
            // TODO: call fink-fat reprocessing logic here
        }
    }
    Ok(())
}
