use camino::Utf8PathBuf;

use clap::{Args, Parser, Subcommand};
use fink_fat_engine::pipeline::stages::alert_inputs::input_uri::InputUri;

/// FINK-FAT: Fink Asteroid Tracker
#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Run fink-fat once for a single night of alerts
    NightRun(NightRunArgs),
    /// Run fink-fat in reprocessing mode over a set of alerts
    Reprocessing(ReprocessingArgs),
}

/// Arguments for the `night-run` subcommand
#[derive(Debug, Args)]
pub struct NightRunArgs {
    /// Path to the file containing the night's alerts
    #[arg(short, long, value_name = "ALERTS_FILE")]
    pub alerts: InputUri,

    /// Path to the fink-fat configuration file
    #[arg(short, long, value_name = "CONFIG_FILE")]
    pub config: Utf8PathBuf,

    /// Display indicatif progress bars during the pipeline run
    #[arg(long, default_value_t = false)]
    pub progress: bool,

    /// Enable logging to file and terminal (log level is read from the config file)
    #[arg(long, default_value_t = false)]
    pub logs: bool,
}

/// Arguments for the `reprocessing` subcommand
#[derive(Debug, Args)]
pub struct ReprocessingArgs {
    /// Path to the file containing the alerts to reprocess
    #[arg(short, long, value_name = "ALERTS_FILE")]
    pub alerts: InputUri,

    /// Path to the fink-fat configuration file
    #[arg(short, long, value_name = "CONFIG_FILE")]
    pub config: Utf8PathBuf,

    /// Display indicatif progress bars during the pipeline run
    #[arg(long, default_value_t = false)]
    pub progress: bool,
}

pub fn cli_builder() -> Cli {
    Cli::parse()
}
