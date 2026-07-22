//! Dataset-wide evaluation of intra-night Kalman-bank seeding.
//!
//! Runs [`BranchCollection::advance_one_night`] independently on every
//! night in the input dataset (each night treated as its own "night 0"
//! seeding pass, matching how [`fink_fat_engine::topocentric_kf::branching::discovery::seed_new_lineages_from_leftovers`]
//! is exercised on unclaimed observations), then reports recall (did every
//! real multi-detection object get a seed?), purity (was the seed
//! uncontaminated by another object?), and volume (banks and hypotheses
//! produced) — see [`fink_fat_eval::seed_bank_report`] for the statistics
//! themselves.

use anyhow::Result;
use camino::Utf8Path;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

use fink_fat_engine::{engine_config::EngineConfig, topocentric_kf::branching::BranchCollection};
use fink_fat_eval::{
    cli::{Cli, load_data},
    seed_bank_report::{
        ground_truth::ObsTrajLookup,
        night_stats::compute_night_stats,
        plots::{plot_branches_per_night, plot_hypotheses_histogram, plot_recall_purity_per_night},
        report::SeedingReport,
    },
};
use photom::observation_dataset::observation::Observation;

/// Directory plots are written to, relative to the current working
/// directory (matching how this binary is normally invoked, from
/// `crates/fink-fat-eval/`).
const OUTPUT_DIR: &str = "seed_to_bank_report";

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("FINKFAT_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_target(true)
        .without_time()
        .with_ansi(false)
        .init();

    let cli = Cli::parse();

    let (_, obs_dataset) = load_data(&cli.alerts, None);

    let engine_config = EngineConfig::load_engine_config_validated(cli.config)?;
    let kalman_ctx = engine_config.build_context();

    let ground_truth = ObsTrajLookup::build(&obs_dataset);

    let mut night_ids: Vec<_> = obs_dataset
        .iter_night_id()
        .expect("dataset must have a night index")
        .copied()
        .collect();
    night_ids.sort_unstable();

    let progress = ProgressBar::new(night_ids.len() as u64);
    progress.set_style(
        ProgressStyle::with_template("{bar:40.cyan/blue} {pos}/{len} nights  {msg}").unwrap(),
    );

    let mut report = SeedingReport::default();
    for night_id in night_ids {
        let Some(night_iter) = obs_dataset.iter_night_observations(&night_id) else {
            progress.inc(1);
            continue;
        };
        let night_obs: Vec<&Observation> = night_iter.collect();
        if night_obs.is_empty() {
            progress.inc(1);
            continue;
        }

        let new_collection = BranchCollection::empty().advance_one_night(
            &night_obs,
            &obs_dataset,
            &engine_config,
            &kalman_ctx,
            0,
        )?;

        let stats = compute_night_stats(
            night_id,
            &night_obs,
            &new_collection.branches,
            &ground_truth,
        );
        progress.set_message(format!(
            "night {night_id}: {} branches, recall {:.0}%",
            stats.n_branches,
            stats.recall_pct()
        ));
        report.push(stats);
        progress.inc(1);
    }
    progress.finish_with_message("done");

    report.print_summary();

    let output_dir = cli
        .output_result
        .unwrap_or(Utf8Path::new(OUTPUT_DIR).to_path_buf());
    std::fs::create_dir_all(&output_dir)?;

    let branches_plot = output_dir.join("branches_per_night.png");
    let recall_purity_plot = output_dir.join("recall_purity_per_night.png");
    let hypotheses_plot = output_dir.join("hypotheses_histogram.png");

    plot_branches_per_night(&report, &branches_plot)?;
    plot_recall_purity_per_night(&report, &recall_purity_plot)?;
    plot_hypotheses_histogram(&report, &hypotheses_plot)?;

    println!("Plots written to:");
    println!("  {branches_plot}");
    println!("  {recall_purity_plot}");
    println!("  {hypotheses_plot}");

    Ok(())
}
