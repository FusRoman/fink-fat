//! Entry point for automatic KF-bank parameter calibration.
//!
//! Starting from the config passed via `--config`, searches for
//! search-region/gating/process-noise parameter values (see
//! `fink_fat_eval::kf_calibration::params::CalibrationParams`) that keep
//! recall high (see `kf_calibration::objective::is_reconstructed`) while
//! shrinking the cost proxy (mean predicted search-region radius) — using a
//! progressive-sampling coordinate descent over the labelled trajectory
//! population (see `kf_calibration::search::calibrate`).
//!
//! Thin orchestration only: all the actual work lives in
//! `fink_fat_eval::kf_calibration`, reusing
//! `fink_fat_eval::kalman_traj`/`trajectory_processing` (the same
//! single-trajectory predict/update loop and per-trajectory metrics the
//! other `kalman_*_eval` binaries use).

use anyhow::Result;
use camino::Utf8PathBuf;
use clap::Parser;

use fink_fat_eval::{
    cli::{Cli, load_config, load_data},
    kalman_traj::ObserverGeometryCache,
    kf_calibration::{
        report::print_report,
        search::{CalibrationOptions, calibrate},
    },
    seed_bank_report::ground_truth::ObsTrajLookup,
};
use photom::TrajId;

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
struct KfCalibrateCli {
    #[command(flatten)]
    common: Cli,

    /// Seed for the round-sampling RNG — same seed, same inputs, same report.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Trajectory sample size for round 0.
    #[arg(long, default_value_t = 50)]
    initial_sample_size: usize,

    /// Multiplicative sample-size growth applied after each round that
    /// doesn't yet cover the whole dataset.
    #[arg(long, default_value_t = 4.0)]
    growth_factor: f64,

    /// Recall target (0-1) at which the search stops preferring recall
    /// gains over cost reductions, and at which a full-dataset round is
    /// allowed to stop the run early.
    #[arg(long, default_value_t = 0.98)]
    target_recall: f64,

    /// Hard cap on the number of rounds.
    #[arg(long, default_value_t = 20)]
    max_rounds: usize,

    /// Cap on how many failing trajectory ids are carried into the next
    /// round's sample.
    #[arg(long, default_value_t = 200)]
    max_failing_pool: usize,

    /// `completion_fraction` threshold (0-1) for "trajectory reconstructed".
    #[arg(long, default_value_t = 0.98)]
    completion_threshold: f64,

    /// `pct_within_search_radius` threshold (0-100) for "trajectory
    /// reconstructed".
    #[arg(long, default_value_t = 95.0)]
    within_radius_threshold: f64,

    /// Restrict calibration to these parameter names (comma-separated,
    /// e.g. `max_arcsec,gate_chi2`). Defaults to every parameter in
    /// `kf_calibration::params::default_param_specs`.
    #[arg(long, value_delimiter = ',')]
    params: Option<Vec<String>>,

    /// Minimum number of observations a trajectory must have to be
    /// included in calibration. Below this there's no meaningful bank
    /// bootstrap + predict/update arc to optimize over (bootstrap alone
    /// consumes the first same-night pair — see
    /// `kalman_traj::init_bank_from_first_pair` — so a 2-3 point trajectory
    /// leaves nothing to score).
    #[arg(long, default_value_t = 4)]
    min_traj_points: usize,
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("FINKFAT_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_target(true)
        .without_time()
        .with_ansi(false)
        .init();
}

fn main() -> Result<()> {
    init_tracing();

    let cli = KfCalibrateCli::parse();

    let (_, obs_dataset) = load_data(&cli.common.alerts);
    let engine_config = load_config(&cli.common.config)?;
    let kalman_ctx = engine_config.build_context();

    let obs_counts = ObsTrajLookup::build(&obs_dataset).obs_counts_by_traj();
    let all_ids: Vec<TrajId> = obs_dataset
        .iter_traj_id()
        .expect("dataset must expose at least one trajectory")
        .cloned()
        .collect();
    let n_before_filter = all_ids.len();
    let traj_ids: Vec<TrajId> = all_ids
        .into_iter()
        .filter(|id| obs_counts.get(id).copied().unwrap_or(0) >= cli.min_traj_points)
        .collect();

    println!(
        "Calibrating over {}/{} trajectories with >= {} points (seed {}, initial sample {})…",
        traj_ids.len(),
        n_before_filter,
        cli.min_traj_points,
        cli.seed,
        cli.initial_sample_size
    );

    // Observer heliocentric geometry depends only on (obs_dataset, obs,
    // kalman_ctx's ephemeris) — never on the 8 calibrated parameters — so
    // it's resolved once here and reused for every candidate/round instead
    // of being recomputed on every one of the ~90 `evaluate()` calls a
    // round's coordinate descent performs. See `ObserverGeometryCache`'s doc.
    println!("Precomputing observer geometry for every trajectory…");
    let geometry_cache = ObserverGeometryCache::build(&obs_dataset, &kalman_ctx, &traj_ids);

    let opts = CalibrationOptions {
        seed: cli.seed,
        initial_sample_size: cli.initial_sample_size,
        growth_factor: cli.growth_factor,
        target_recall: cli.target_recall,
        max_rounds: cli.max_rounds,
        max_failing_pool: cli.max_failing_pool,
        completion_threshold: cli.completion_threshold,
        within_radius_threshold: cli.within_radius_threshold,
        param_names: cli.params,
    };

    let report = calibrate(
        &obs_dataset,
        &engine_config,
        &kalman_ctx,
        &traj_ids,
        &geometry_cache,
        &opts,
    );

    print_report(&report);

    if let Some(output_dir) = &cli.common.output_result {
        write_report(output_dir, &report)?;
    }

    Ok(())
}

fn write_report(
    output_dir: &Utf8PathBuf,
    report: &fink_fat_eval::kf_calibration::report::CalibrationReport,
) -> Result<()> {
    std::fs::create_dir_all(output_dir)?;

    let json_path = output_dir.join("kf_calibration_report.json");
    std::fs::write(&json_path, serde_json::to_string_pretty(report)?)?;
    println!("\nFull report written to {json_path}");

    let yaml_path = output_dir.join("kf_calibration_params.yaml");
    std::fs::write(
        &yaml_path,
        fink_fat_eval::kf_calibration::report::to_yaml_snippet(&report.final_params),
    )?;
    println!("Recommended config snippet written to {yaml_path}");

    Ok(())
}
