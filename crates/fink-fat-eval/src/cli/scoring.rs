use camino::Utf8PathBuf;
use clap::{ArgAction, Parser};
use fink_fat_engine::engine_config::EngineConfig;

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

    /// Limit number of nights processed (after sorting / selection).
    #[arg(long, value_name = "N", help_heading = "Scan", default_value_t = 1)]
    pub consecutive_window: usize,

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

    // -------------------------------------------------------------------------
    // Optuna / tuning controls
    // -------------------------------------------------------------------------
    /// Evaluate only the first N frozen pairs (deterministic prefix).
    ///
    /// This enables multi-fidelity optimization (e.g. 50k → 200k → 1M).
    #[arg(long, value_name = "N", help_heading = "Optimization")]
    pub budget: Option<usize>,

    /// Target true positive rate (TPR) used for FPR@TPR objective.
    #[arg(
        long,
        value_name = "TPR",
        default_value_t = 0.95,
        help_heading = "Optimization"
    )]
    pub target_tpr: f64,

    /// Minimum accepted-good fraction required to avoid degenerate solutions.
    ///
    /// Penalty is applied when:
    /// `accept_good_rate < min_accept_good`.
    #[arg(
        long,
        value_name = "RATE",
        default_value_t = 0.50,
        help_heading = "Optimization"
    )]
    pub min_accept_good: f64,

    /// Weight of the acceptance penalty term.
    #[arg(
        long,
        value_name = "W",
        default_value_t = 0.50,
        help_heading = "Optimization"
    )]
    pub penalty_weight: f64,

    // -------------------------------------------------------------------------
    // ScoreConfig overrides (gates)
    // -------------------------------------------------------------------------
    #[arg(long, value_name = "variance_floor", help_heading = "Overrides")]
    pub variance_floor: Option<f64>,

    #[arg(long, value_name = "drift_per_day", help_heading = "Overrides")]
    pub drift_per_day: Option<f64>,

    #[arg(long, value_name = "curvature_per_day", help_heading = "Overrides")]
    pub curvature_per_day: Option<f64>,

    #[arg(long, value_name = "theta_zero", help_heading = "Overrides")]
    pub theta_zero: Option<f64>,

    #[arg(long, value_name = "v_zero", help_heading = "Overrides")]
    pub v_zero: Option<f64>,

    #[arg(long, value_name = "vel_eps_day", help_heading = "Overrides")]
    pub vel_eps_day: Option<f64>,

    #[arg(long, value_name = "w_dir", help_heading = "Overrides")]
    pub w_dir: Option<f64>,

    #[arg(long, value_name = "w_norm", help_heading = "Overrides")]
    pub w_norm: Option<f64>,

    #[arg(long, value_name = "w_flux", help_heading = "Overrides")]
    pub w_flux: Option<f64>,
    #[arg(long, value_name = "flux_sigma_floor", help_heading = "Overrides")]
    pub flux_sigma_floor: Option<f64>,

    #[arg(long, value_name = "w_gap", help_heading = "Overrides")]
    pub w_gap: Option<f64>,

    #[arg(long, value_name = "rho", help_heading = "Overrides")]
    pub rho: Option<f64>,

    #[arg(long, value_name = "w_band_mismatch", help_heading = "Overrides")]
    pub w_band_mismatch: Option<f64>,

    // -------------------------------------------------------------------------
    // Logging
    // -------------------------------------------------------------------------
    /// Silence all human-readable output (machine-friendly output only).
    #[arg(long, action = clap::ArgAction::SetTrue, help_heading = "Logging")]
    pub quiet: bool,

    /// Increase verbosity (-v, -vv).
    #[arg(short = 'v', long, action = clap::ArgAction::Count, help_heading = "Logging")]
    pub verbose: u8,
}

pub fn update_score_config(cli: &Cli, engine_config: &EngineConfig) -> EngineConfig {
    let mut updated_cfg = engine_config.clone();

    if let Some(max_d2) = cli.variance_floor {
        updated_cfg.edges.score_config.predict.noise.variance_floor = max_d2;
    }
    if let Some(drift_per_day) = cli.drift_per_day {
        updated_cfg.edges.score_config.predict.noise.drift_per_day = drift_per_day;
    }
    if let Some(curvature_per_day) = cli.curvature_per_day {
        updated_cfg
            .edges
            .score_config
            .predict
            .noise
            .curvature_per_day2 = curvature_per_day;
    }

    if let Some(theta_zero) = cli.theta_zero {
        updated_cfg.edges.score_config.velocity.theta0 = theta_zero;
    }
    if let Some(v_zero) = cli.v_zero {
        updated_cfg.edges.score_config.velocity.v0 = v_zero;
    }
    if let Some(vel_eps_day) = cli.vel_eps_day {
        updated_cfg.edges.score_config.velocity.vel_eps_days = vel_eps_day;
    }

    if let Some(w_dir) = cli.w_dir {
        updated_cfg.edges.score_config.velocity.w_dir = w_dir;
    }
    if let Some(w_norm) = cli.w_norm {
        updated_cfg.edges.score_config.velocity.w_norm = w_norm;
    }

    if let Some(w_flux) = cli.w_flux {
        updated_cfg.edges.score_config.photometry.w_flux = w_flux;
    }
    if let Some(flux_sigma_floor) = cli.flux_sigma_floor {
        updated_cfg.edges.score_config.photometry.flux_sigma_floor = flux_sigma_floor;
    }

    if let Some(w_gap) = cli.w_gap {
        updated_cfg.edges.score_config.gap.w_gap = w_gap;
    }
    if let Some(rho) = cli.rho {
        updated_cfg.edges.score_config.gap.rho = rho;
    }

    if let Some(w_band_mismatch) = cli.w_band_mismatch {
        updated_cfg.edges.score_config.band.w_band_mismatch = w_band_mismatch;
    }

    updated_cfg
}
