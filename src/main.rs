pub mod init_cli;
pub mod logging;
pub mod main_runner;
pub mod progress;

use camino::Utf8Path;
use fink_fat_engine::{
    engine_config::{EngineConfig, load_engine_config_validated},
    error::EngineError,
};

use crate::{init_cli::cli_builder, main_runner::fink_fat_runner};

pub fn load_config(config_path: &Utf8Path) -> Result<EngineConfig, EngineError> {
    Ok(load_engine_config_validated(config_path)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = cli_builder();

    fink_fat_runner(cli)?;

    Ok(())
}
