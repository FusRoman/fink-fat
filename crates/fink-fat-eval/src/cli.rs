use anyhow::{Context, Result};
use camino::{Utf8Path, Utf8PathBuf};
use clap::Parser;
use fink_fat_engine::engine_config::main_config::EngineConfig;
use photom::{
    io::polars::{ContiguousChoice, FromPolarsArgs},
    observation_dataset::ObsDataset,
    observer::error_model::ObsErrorModel,
};
use polars::{
    frame::DataFrame,
    lazy::frame::{LazyFrame, ScanArgsParquet},
};

pub fn load_data(parquet_path: impl AsRef<Utf8Path>) -> (DataFrame, ObsDataset) {
    let path = parquet_path.as_ref().as_str();
    let args = ScanArgsParquet {
        rechunk: true,
        ..Default::default()
    };
    let lf = LazyFrame::scan_parquet(path.into(), args).expect("scan_parquet must succeed");
    let obs_dataset = ObsDataset::from_lazy(
        lf.clone(),
        FromPolarsArgs {
            do_rechunk: Some(false),
            error_model: Some(ObsErrorModel::FCCT14),
            contiguous_choice: Some(ContiguousChoice::ContiguousNight),
        },
    )
    .expect("from_lazy must succeed for int file");

    let df = lf.collect().expect("collect must succeed");
    (df, obs_dataset)
}

/// FINK-FAT: Fink Asteroid Tracker
#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Path to the file containing the night's alerts
    #[arg(short, long, value_name = "ALERTS_FILE")]
    pub alerts: Utf8PathBuf,

    /// Path to the fink-fat configuration file
    #[arg(short, long, value_name = "CONFIG_FILE")]
    pub config: Utf8PathBuf,

    /// Path of the output directory if any results should be save on disk
    #[arg(short, long, value_name = "OUTPUT_DIR")]
    pub output_result: Option<Utf8PathBuf>,
}

pub fn load_config(config_path: impl AsRef<Utf8Path>) -> Result<EngineConfig> {
    EngineConfig::load_engine_config_validated(config_path).context("failed to load engine config")
}
