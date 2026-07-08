//! Dataset-wide evaluation of night-after-night multi-hypothesis tracking.
//!
//! Unlike [`fink_fat_eval::seed_bank_report`] (see `bin/seed_to_bank_analysis.rs`),
//! which re-seeds from an empty [`BranchCollection`] every night to isolate
//! the seeding step, this binary actually **advances** a `BranchCollection`
//! night after night (`collection = collection.advance_one_night(...)`),
//! evaluating the full branching/pruning/LLR/Kalman pipeline over the
//! duration of a run — see [`fink_fat_eval::tracking_report`] for the
//! statistics themselves.

use std::time::Instant;

use anyhow::{Result, bail};
use camino::{Utf8Path, Utf8PathBuf};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

use fink_fat_engine::{
    engine_config::EngineConfig, spacetime_bucket::healpix_binner::HealpixBinner,
    topocentric_kf::branching::BranchCollection,
};
use fink_fat_eval::{
    cli::{Cli, load_data},
    seed_bank_report::ground_truth::ObsTrajLookup,
    tracking_report::{
        gold_trajectory::GoldTrajectoryTracker,
        lineage_lifecycle::LineageTracker,
        night_stats::compute_night_tracking_stats,
        plots::{
            plot_aggregated_histograms, plot_branches_and_lineages_per_night,
            plot_error_box_radius_per_night, plot_llr_and_ess_per_night,
            plot_observations_in_box_per_night, plot_recall_purity_completeness_per_night,
            plot_timing_per_night,
        },
        report::TrackingReport,
    },
};
use photom::{NightId, observation_dataset::observation::Observation};

/// Directory the JSON report and plots are written to, relative to the
/// current working directory (matching how this binary is normally
/// invoked, from `crates/fink-fat-eval/`).
const OUTPUT_DIR: &str = "tracking_report";

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct TrackingAnalysisCli {
    #[command(flatten)]
    common: Cli,

    /// Regenerate plots from a previously written tracking_report.json,
    /// skipping the engine run entirely.
    #[arg(long, value_name = "JSON_FILE")]
    from_json: Option<Utf8PathBuf>,

    /// List every night id and its observation count, then exit. Only
    /// needs --alerts (--config is still required by --alerts's shared CLI
    /// struct, but is unused in this mode).
    #[arg(long)]
    list_nights: bool,

    /// First night to process (inclusive). Defaults to the dataset's
    /// earliest night.
    ///
    /// Note: tracking always starts from an *empty* `BranchCollection` at
    /// this night, even when it isn't the dataset's first night — there is
    /// no "resume tracking state from a prior run" support, so starting
    /// mid-dataset evaluates tracking "from zero knowledge" on that
    /// sub-range, not a continuation of what came before it.
    #[arg(long)]
    start_night: Option<u32>,

    /// Last night to process (inclusive). Conflicts with --n-nights.
    #[arg(long, conflicts_with = "n_nights")]
    end_night: Option<u32>,

    /// Process exactly this many nights starting at --start-night.
    /// Conflicts with --end-night; requires --start-night.
    #[arg(long, conflicts_with = "end_night")]
    n_nights: Option<usize>,
}

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

    let cli = TrackingAnalysisCli::parse();

    if let Some(json_path) = &cli.from_json {
        let report = TrackingReport::read_json(json_path)?;
        let aggregated = report.finalize();
        aggregated.print_summary();

        let output_dir = resolve_output_dir(&cli.common);
        std::fs::create_dir_all(&output_dir)?;
        write_all_plots(&report, &output_dir)?;
        return Ok(());
    }

    let (_, obs_dataset) = load_data(&cli.common.alerts);

    if cli.list_nights {
        let mut night_ids: Vec<NightId> = obs_dataset
            .iter_night_id()
            .expect("dataset must have a night index")
            .copied()
            .collect();
        night_ids.sort_unstable();
        for night_id in night_ids {
            println!(
                "{night_id}\t{}",
                obs_dataset.len_night(&night_id).unwrap_or(0)
            );
        }
        return Ok(());
    }

    if cli.n_nights.is_some() && cli.start_night.is_none() {
        bail!("--n-nights requires --start-night");
    }

    let engine_config = EngineConfig::load_engine_config_validated(&cli.common.config)?;
    let kalman_ctx = engine_config.build_context();
    let spatial_binner = HealpixBinner::new(engine_config.healpix_depth);

    let ground_truth = ObsTrajLookup::build(&obs_dataset);

    let mut night_ids: Vec<NightId> = obs_dataset
        .iter_night_id()
        .expect("dataset must have a night index")
        .copied()
        .collect();
    night_ids.sort_unstable();
    let night_ids = resolve_night_range(night_ids, cli.start_night, cli.end_night, cli.n_nights);

    let progress = ProgressBar::new(night_ids.len() as u64);
    progress.set_style(
        ProgressStyle::with_template("{bar:40.cyan/blue} {pos}/{len} nights  {msg}").unwrap(),
    );

    let mut report = TrackingReport::default();
    let mut collection = BranchCollection::empty();
    let mut gold_tracker = GoldTrajectoryTracker::new();
    let mut lineage_tracker = LineageTracker::new();

    for (step, &night_id) in night_ids.iter().enumerate() {
        let Some(night_iter) = obs_dataset.iter_night_observations(&night_id) else {
            progress.inc(1);
            continue;
        };
        let night_obs: Vec<&Observation> = night_iter.collect();

        let next_night_obs: Option<Vec<&Observation>> = night_ids
            .get(step + 1)
            .and_then(|next_id| obs_dataset.iter_night_observations(next_id))
            .map(|it| it.collect());

        gold_tracker.observe_night(&night_obs, &ground_truth);

        let t0 = Instant::now();
        let prev_collection = collection;
        let new_collection = prev_collection.advance_one_night(
            &night_obs,
            &obs_dataset,
            &engine_config,
            &kalman_ctx,
            step,
        )?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let stats = compute_night_tracking_stats(
            step,
            night_id,
            &night_obs,
            next_night_obs.as_deref(),
            &prev_collection,
            &new_collection,
            &ground_truth,
            &gold_tracker,
            &mut lineage_tracker,
            &obs_dataset,
            &kalman_ctx,
            &engine_config,
            &spatial_binner,
            elapsed_ms,
        );

        progress.set_message(format!(
            "night {night_id}: {} branches, recall {:.0}%, completeness {:.0}%",
            stats.n_branches, stats.recall_pct_tonight, stats.completeness_pct_so_far
        ));
        report.push(stats);
        progress.inc(1);

        collection = new_collection;
    }
    progress.finish_with_message("done");

    let aggregated = report.finalize();
    aggregated.print_summary();

    let output_dir = resolve_output_dir(&cli.common);
    std::fs::create_dir_all(&output_dir)?;

    let json_path = output_dir.join("tracking_report.json");
    report.write_json(&json_path)?;
    println!("Report written to {json_path}");

    write_all_plots(&report, &output_dir)?;

    Ok(())
}

fn resolve_output_dir(common: &Cli) -> Utf8PathBuf {
    common
        .output_result
        .clone()
        .unwrap_or_else(|| Utf8Path::new(OUTPUT_DIR).to_path_buf())
}

/// Restrict `night_ids` (already sorted) to `[start_night, end_night]`, or
/// to the first `n_nights` nights from `start_night` — whichever bound was
/// given. All three are optional; an unbounded call returns `night_ids`
/// unchanged.
fn resolve_night_range(
    night_ids: Vec<NightId>,
    start_night: Option<u32>,
    end_night: Option<u32>,
    n_nights: Option<usize>,
) -> Vec<NightId> {
    let filtered: Vec<NightId> = night_ids
        .into_iter()
        .filter(|id| start_night.is_none_or(|s| id.0 >= s))
        .filter(|id| end_night.is_none_or(|e| id.0 <= e))
        .collect();

    match n_nights {
        Some(n) => filtered.into_iter().take(n).collect(),
        None => filtered,
    }
}

fn write_all_plots(report: &TrackingReport, output_dir: &Utf8Path) -> Result<()> {
    let aggregated = report.finalize();

    let branches_plot = output_dir.join("branches_and_lineages_per_night.png");
    let recall_plot = output_dir.join("recall_purity_completeness_per_night.png");
    let llr_plot = output_dir.join("llr_per_night.png");
    let error_box_plot = output_dir.join("error_box_radius_per_night.png");
    let in_box_plot = output_dir.join("observations_in_box_per_night.png");
    let timing_plot = output_dir.join("timing_per_night.png");

    plot_branches_and_lineages_per_night(report, &branches_plot)?;
    plot_recall_purity_completeness_per_night(report, &recall_plot)?;
    plot_llr_and_ess_per_night(report, &llr_plot)?;
    plot_error_box_radius_per_night(report, &error_box_plot)?;
    plot_observations_in_box_per_night(report, &in_box_plot)?;
    plot_timing_per_night(report, &timing_plot)?;

    let histogram_paths = plot_aggregated_histograms(&aggregated, output_dir)?;

    println!("Plots written to:");
    for p in [
        branches_plot,
        recall_plot,
        llr_plot,
        error_box_plot,
        in_box_plot,
        timing_plot,
    ] {
        println!("  {p}");
    }
    for p in histogram_paths {
        println!("  {p}");
    }
    Ok(())
}
