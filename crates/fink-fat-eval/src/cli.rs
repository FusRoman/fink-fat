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
    /// Run fink-fat in solver evaluation mode over a set of alerts
    SolverEval(SolverArgs),
    ModelEval(ModelEvalArgs),
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

    /// Output directory for evaluation plots (PNG files).
    ///
    /// When set, a set of distribution and result charts is written to this
    /// directory after evaluation completes.  The directory is created
    /// automatically if it does not exist.
    #[arg(long, value_name = "PLOT_DIR")]
    pub plot_dir: Option<Utf8PathBuf>,
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

    /// Output path for the edge feature Parquet file (ML training dataset).
    ///
    /// When set, a Parquet file containing all 17 edge features
    /// (`position.*`, `velocity.*`, `uncertainty.*`, `photometry.*`),
    /// the truth label (`is_true_edge`), the group column (`from_seed_id`),
    /// and the debug columns (`left_nid`, `right_nid`, `gap_nights`) is
    /// written to this path.  Parent directories are created automatically.
    #[arg(long, value_name = "PARQUET_PATH")]
    pub export_features: Option<Utf8PathBuf>,
}

/// Arguments for the `solver-eval` subcommand
#[derive(Debug, Args)]
pub struct SolverArgs {
    #[command(flatten)]
    pub common: CommonArgs,
}

/// Arguments for the `model-eval` subcommand
#[derive(Debug, Args)]
pub struct ModelEvalArgs {
    /// Path to the edge features Parquet file produced by `edge-eval --export-features`.
    ///
    /// This file must contain the 17 canonical feature columns
    /// (`position.*`, `velocity.*`, `uncertainty.*`, `photometry.*`) and the
    /// truth label column `is_true_edge` (1 = true positive, 0 otherwise).
    #[arg(short = 'f', long, value_name = "PARQUET_PATH")]
    pub features_parquet: Utf8PathBuf,

    /// Path to the XGBoost training config file (`xgb_params.yml`).
    ///
    /// The `training.onnx_output` field in this file is read to locate the
    /// ONNX model to evaluate.  The path stored in the config is interpreted
    /// relative to the directory of this YAML file.
    #[arg(short = 'x', long, value_name = "XGB_PARAMS")]
    pub xgb_params: Utf8PathBuf,

    /// Output directory for evaluation plots (PNG files).
    ///
    /// Written: `roc_curve.png`, `pr_curve.png`, `score_distribution.png`.
    #[arg(short = 'p', long, value_name = "PLOT_DIR")]
    pub plot_dir: Option<Utf8PathBuf>,
}

pub fn cli_builder() -> Cli {
    Cli::parse()
}
