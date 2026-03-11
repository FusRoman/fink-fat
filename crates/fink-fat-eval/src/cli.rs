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
    /// Run fink-fat in seeding evaluation mode over a set of alerts
    SeedingEval(SeedingArgs),
    /// Run fink-fat in edge evaluation mode over a set of alerts
    EdgeEval(EdgeArgs),
}

/// Arguments for the `seeding-eval` subcommand
#[derive(Debug, Args)]
pub struct CommonArgs {
    /// Path to the file containing the night's alerts
    #[arg(short, long, value_name = "ALERTS_FILE")]
    pub alerts: InputUri,

    /// Path to the fink-fat configuration file
    #[arg(short, long, value_name = "CONFIG_FILE")]
    pub config: Utf8PathBuf,
}

/// Arguments for the `seeding-eval` subcommand
#[derive(Debug, Args)]
pub struct SeedingArgs {
    #[command(flatten)]
    pub common: CommonArgs,
}

/// Arguments for the `edge-eval` subcommand
#[derive(Debug, Args)]
pub struct EdgeArgs {
    #[command(flatten)]
    pub common: CommonArgs,
}

pub fn cli_builder() -> Cli {
    Cli::parse()
}
