use anyhow::{Context, Result};
use camino::{Utf8Path, Utf8PathBuf};
use clap::Parser;
use fink_fat_engine::engine_config::main_config::EngineConfig;
use photom::{
    io::polars::{ContiguousChoice, FromPolarsArgs},
    observation_dataset::ObsDataset,
    observer::error_model::ObsErrorModel,
};
use polars::lazy::{
    dsl::lit,
    frame::{LazyFrame, ScanArgsParquet},
};

/// Load an alerts dataset, optionally overriding every observation's
/// `ra_err`/`dec_err` (radians) with a single uniform value — used to sweep
/// candidate astrometric-noise assumptions (see
/// `test_exp/prep_alert.py`'s hardcoded 1″) against dataset-wide NIS/NEES
/// calibration without regenerating the source Parquet for each candidate.
///
/// Only the lazy frame is materialized into `ObsDataset`'s row-oriented
/// layout — no caller needs the Polars `DataFrame` form, and collecting one
/// used to hold a second full copy of the dataset in memory for the whole
/// run.
pub fn load_data(
    parquet_path: impl AsRef<Utf8Path>,
    override_obs_error_arcsec: Option<f64>,
) -> ObsDataset {
    let path = parquet_path.as_ref().as_str();
    let args = ScanArgsParquet {
        rechunk: true,
        ..Default::default()
    };
    let mut lf = LazyFrame::scan_parquet(path.into(), args).expect("scan_parquet must succeed");

    if let Some(arcsec) = override_obs_error_arcsec {
        let err_rad = arcsec * (std::f64::consts::PI / (180.0 * 3600.0));
        lf = lf.with_columns([lit(err_rad).alias("ra_err"), lit(err_rad).alias("dec_err")]);
    }

    ObsDataset::from_lazy(
        lf,
        FromPolarsArgs {
            do_rechunk: Some(false),
            error_model: Some(ObsErrorModel::FCCT14),
            contiguous_choice: Some(ContiguousChoice::ContiguousNight),
        },
    )
    .expect("from_lazy must succeed for int file")
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

    /// Path to a ground-truth topocentric-state Parquet file (see
    /// `test_exp/solar_system_data/build_ground_truth.py`), used to compute
    /// NEES/RMSE metrics against the true trajectory. If omitted, those
    /// metrics are skipped.
    #[arg(long, value_name = "GROUND_TRUTH_FILE")]
    pub ground_truth: Option<Utf8PathBuf>,

    /// Path to write a per-step metrics Parquet file (one row per Kalman
    /// filter step across all processed trajectories).
    #[arg(long, value_name = "STEPS_PARQUET_FILE")]
    pub steps_parquet_out: Option<Utf8PathBuf>,

    /// Path to write a per-trajectory summary metrics Parquet file.
    #[arg(long, value_name = "SUMMARY_PARQUET_FILE")]
    pub summary_parquet_out: Option<Utf8PathBuf>,

    /// Override every observation's ra_err/dec_err (arcsec) with this single
    /// uniform value, without regenerating the source Parquet file — used to
    /// sweep candidate astrometric-noise assumptions against dataset-wide
    /// NIS/NEES calibration (see `test_exp/prep_alert.py`).
    #[arg(long, value_name = "ARCSEC")]
    pub override_obs_error_arcsec: Option<f64>,
}

pub fn load_config(config_path: impl AsRef<Utf8Path>) -> Result<EngineConfig> {
    EngineConfig::load_engine_config_validated(config_path).context("failed to load engine config")
}
