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

    /// Output directory for evaluation plots (PNG files).
    ///
    /// When set, a set of distribution and result charts is written to this
    /// directory after evaluation completes.  The directory is created
    /// automatically if it does not exist.
    #[arg(long, value_name = "PLOT_DIR")]
    pub plot_dir: Option<Utf8PathBuf>,
}

/// Arguments for the `edge-eval` subcommand
#[derive(Debug, Args)]
pub struct EdgeArgs {
    #[command(flatten)]
    pub common: CommonArgs,

    /// Output directory for edge evaluation plots (PNG files).
    ///
    /// When set, feature distribution charts (TP vs FP) are written here.
    /// The directory is created automatically if it does not exist.
    #[arg(long, value_name = "PLOT_DIR")]
    pub plot_dir: Option<Utf8PathBuf>,
}

pub fn cli_builder() -> Cli {
    Cli::parse()
}
