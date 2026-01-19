use camino::Utf8PathBuf;
use clap::{ArgAction, Parser};

use crate::cli::common::{CommonBinningArgs, CommonScanArgs};

/// Inspect scoring components + seed tangent-plane values.
///
/// This CLI mirrors the options from the original monolithic binary.
#[derive(Parser, Debug)]
#[command(
    name = "inspect_scoring_components",
    author,
    version,
    about = "Inspect inter-night scoring components and seed kinematics (QA plots).",
    disable_help_subcommand = true,
    verbatim_doc_comment
)]
pub struct Cli {
    #[command(flatten)]
    pub scan: CommonScanArgs,

    #[command(flatten)]
    pub binning: CommonBinningArgs,

    /// YAML engine configuration (pairs/triplets/predictor/scoring/edges).
    #[arg(
        long,
        value_name = "YAML",
        help_heading = "Config",
        default_value = "engine.yaml"
    )]
    pub engine_config: Utf8PathBuf,

    /// Optional explicit list of night ids to process (comma-separated).
    #[arg(long, value_name = "NID1,NID2,...", help_heading = "Scan")]
    pub nids: Option<String>,

    /// Limit number of nights processed (after sorting / selection).
    #[arg(long, value_name = "N", help_heading = "Scan")]
    pub max_nights: Option<usize>,

    /// Disable triplet generation (pairs-only).
    #[arg(long, default_value_t = false, help_heading = "Seeding")]
    pub pairs_only: bool,

    /// Keep only edges where both seeds have a valid truth id (`trajectory_id > 0`)
    /// and each seed is "pure" (all members share the same traj_id).
    #[arg(long, default_value_t = true, action = ArgAction::Set, help_heading = "Truth")]
    pub only_truth: bool,

    /// Number of histogram bins.
    #[arg(long, default_value_t = 200, help_heading = "Plot")]
    pub bins: usize,

    /// If set, clip histogram x-range to [0, xmax] (applies to positive-valued plots).
    #[arg(long, help_heading = "Plot")]
    pub clip_xmax: Option<f64>,

    /// If set, use log scale on Y axis (counts).
    #[arg(long, action = ArgAction::SetTrue, help_heading = "Plot")]
    pub log_y: bool,

    /// Maximum number of seeds kept per night (random subsample if exceeded).
    #[arg(long, default_value_t = 8000, help_heading = "Sampling")]
    pub max_seeds_per_night: usize,

    /// For each left seed, sample this many right seeds (candidate evaluation budget).
    #[arg(long, default_value_t = 512, help_heading = "Sampling")]
    pub sample_right_per_left: usize,

    /// Optional cap on total edges sampled per night-pair (after filtering).
    #[arg(long, default_value_t = 500_000, help_heading = "Sampling")]
    pub max_edges_per_pair: usize,

    /// Random seed for reproducibility.
    #[arg(long, default_value_t = 42, help_heading = "Sampling")]
    pub rng_seed: u64,

    /// Number of parallel jobs (threads). Defaults to number of CPU cores.
    #[arg(long, help_heading = "Scan")]
    pub jobs: Option<usize>,

    /// Silence all human-readable output (JSON / scalar only).
    #[arg(long, action = clap::ArgAction::SetTrue)]
    pub quiet: bool,

    /// Increase verbosity (-v, -vv).
    #[arg(short = 'v', long, action = clap::ArgAction::Count)]
    pub verbose: u8,
}
