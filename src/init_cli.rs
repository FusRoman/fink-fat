use camino::Utf8PathBuf;

use clap::Parser;
use fink_fat_engine::pipeline::stages::alert_inputs::input_uri::InputUri;

/// FINK-FAT: Fink Asteroid Tracker
#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct FinkFatCliArgs {
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

pub fn cli_builder() -> FinkFatCliArgs {
    FinkFatCliArgs::parse()
}
