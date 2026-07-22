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

use ahash::AHashSet;
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
    snapshot_report::{
        efficacy::{
            ReconstructionEfficacy, build_gold_tracker_for_processed_nights,
            compute_reconstruction_efficacy, determine_last_processed_night,
        },
        plots::{
            plot_hypotheses_per_branch_histogram,
            plot_observations_per_night as plot_snapshot_observations_per_night,
            plot_reconstruction_coverage_histogram, plot_reconstruction_outcome_breakdown,
            plot_track_length_histogram,
        },
        stats::{SnapshotStats, build_obs_to_night_map, compute_snapshot_stats},
    },
    tracking_report::{
        gold_trajectory::GoldTrajectoryTracker,
        lineage_lifecycle::LineageTracker,
        night_stats::compute_night_tracking_stats,
        object_outcome::{ObjectOutcome, ObjectOutcomeTracker},
        plots::{
            plot_aggregated_histograms, plot_best_pure_coverage_histogram,
            plot_branches_and_lineages_per_night, plot_error_box_radius_per_night,
            plot_llr_and_ess_per_night, plot_object_outcome_breakdown,
            plot_object_outcome_per_night, plot_observations_in_box_per_night,
            plot_recall_purity_completeness_per_night, plot_timing_per_night,
        },
        report::TrackingReport,
        seeding_gate_diagnosis::{count_by_failure, diagnose_never_touched_gating},
    },
};
use photom::{NightId, observation_dataset::ObsId, observation_dataset::observation::Observation};

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

    /// Load a previously-written BranchCollection `.rkyv` snapshot (as
    /// written by the real `fink-fat` engine binary under its
    /// `storage_path`) and report statistics/plots about that single
    /// point-in-time state, instead of re-running the night-by-night
    /// simulation. The path is resolved automatically from `--config`'s
    /// `storage_path` — pass this flag with no value to use it.
    #[arg(long, conflicts_with = "from_json")]
    from_snapshot: bool,

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

    /// Minimum fraction (0.0-1.0) of a gold trajectory's cumulative
    /// observations that a currently-live pure branch must cover to count
    /// toward the *relaxed* completeness metric — unlike the strict metric,
    /// this tolerates a branch that dropped a few points along the way as
    /// long as it was never contaminated by another object.
    #[arg(long, default_value_t = 0.8)]
    completeness_coverage_threshold: f64,
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

    if cli.from_snapshot {
        return run_snapshot_analysis(&cli);
    }

    let (_, obs_dataset) = load_data(&cli.common.alerts, None);

    if cli.list_nights {
        let mut night_ids: Vec<NightId> = obs_dataset
            .iter_night_id()
            .expect("dataset must have a night index")
            .copied()
            .collect();
        night_ids.sort_unstable();
        for night_id in &night_ids {
            println!(
                "{night_id}\t{}",
                obs_dataset.len_night(&night_id).unwrap_or(0)
            );
        }
        println!("Number of nights : {}", night_ids.len());
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
    let mut object_outcome_tracker = ObjectOutcomeTracker::new();
    let mut consumed_then_pruned_ids: AHashSet<ObsId> = AHashSet::default();
    let mut last_step = 0;

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

        gold_tracker.observe_night(step, &night_obs, &ground_truth);

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

        let mut stats = compute_night_tracking_stats(
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
            cli.completeness_coverage_threshold,
        );

        progress.set_message(format!(
            "night {night_id}: {} branches, recall {:.0}%, completeness {:.0}%",
            stats.n_branches, stats.recall_pct_tonight, stats.completeness_pct_so_far
        ));
        object_outcome_tracker.observe_night(step, &new_collection.branches, &ground_truth);
        consumed_then_pruned_ids.extend(
            new_collection
                .last_night_consumed_then_pruned_ids
                .iter()
                .copied(),
        );
        last_step = step;

        // Per-night snapshot of the ObjectOutcome breakdown, for the
        // per-night plot — same classification as the final console
        // summary, just re-run at every step instead of once at the end.
        let outcomes_tonight = object_outcome_tracker.classify_all(
            &gold_tracker,
            step,
            cli.completeness_coverage_threshold,
        );
        for (i, outcome) in ObjectOutcome::all().into_iter().enumerate() {
            stats.object_outcome_counts[i] = outcomes_tonight
                .iter()
                .filter(|(_, o, _)| *o == outcome)
                .count();
        }

        report.push(stats);
        progress.inc(1);

        collection = new_collection;
    }
    progress.finish_with_message("done");

    let aggregated = report.finalize();
    aggregated.print_summary();

    let outcomes = object_outcome_tracker.classify_all(
        &gold_tracker,
        last_step,
        cli.completeness_coverage_threshold,
    );
    print_object_outcome_summary(&outcomes, &gold_tracker);

    let never_touched_ids: Vec<photom::TrajId> = outcomes
        .iter()
        .filter(|(_, o, _)| *o == ObjectOutcome::NeverTouched)
        .map(|(traj_id, _, _)| traj_id.clone())
        .collect();
    let gate_records = diagnose_never_touched_gating(
        &never_touched_ids,
        &obs_dataset,
        &ground_truth,
        &gold_tracker,
        &night_ids,
        &engine_config.pairs,
        &consumed_then_pruned_ids,
        &spatial_binner,
    );
    print_gate_diagnosis_summary(&gate_records, never_touched_ids.len());

    let output_dir = resolve_output_dir(&cli.common);
    std::fs::create_dir_all(&output_dir)?;

    let json_path = output_dir.join("tracking_report.json");
    report.write_json(&json_path)?;
    println!("Report written to {json_path}");

    write_all_plots(&report, &output_dir)?;
    write_object_outcome_plots(&outcomes, &output_dir)?;

    Ok(())
}

/// Print how many trackable ground-truth objects (see
/// [`GoldTrajectoryTracker::is_trackable`]) fall into each [`ObjectOutcome`]
/// category, giving a breakdown of *why* completeness is low instead of a
/// single opaque percentage — see `fink_fat_eval::tracking_report::object_outcome`
/// for what each category means. Also reports how many multi-detection
/// objects were excluded as structurally unreachable by this pipeline's
/// intra-night-only seeding, so the denominator change isn't silent.
fn print_object_outcome_summary(
    outcomes: &[(photom::TrajId, ObjectOutcome, Option<f64>)],
    gold_tracker: &GoldTrajectoryTracker,
) {
    let total = outcomes.len();
    let n_multi_detection = gold_tracker.n_multi_detection_so_far();
    let n_excluded = n_multi_detection.saturating_sub(total);
    println!(
        "Object outcome breakdown ({total} trackable objects; {n_excluded} of \
         {n_multi_detection} multi-detection objects excluded as structurally \
         unreachable — never had >= 2 observations within a single night):"
    );
    for outcome in ObjectOutcome::all() {
        let count = outcomes.iter().filter(|(_, o, _)| *o == outcome).count();
        let pct = if total == 0 {
            0.0
        } else {
            100.0 * count as f64 / total as f64
        };
        println!("  {:<28} : {count:>6} ({pct:>5.1}%)", outcome.label());
    }
    println!();
}

/// Print why the `NeverTouched` trackable objects' same-night pairwise gate
/// rejected every candidate pair — see
/// `fink_fat_eval::tracking_report::seeding_gate_diagnosis` for what each
/// cause means and its limitations (only the object's first two same-night
/// observations are checked, matching the real linker's "no fit yet" path).
fn print_gate_diagnosis_summary(
    records: &[fink_fat_eval::tracking_report::seeding_gate_diagnosis::GateDiagnosisRecord],
    n_never_touched: usize,
) {
    println!(
        "Never-touched seeding gate diagnosis ({}/{n_never_touched} objects analyzed; \
         counts are not mutually exclusive):",
        records.len()
    );
    for (label, count) in count_by_failure(records) {
        println!("  {label:<38} : {count:>6}");
    }
    println!();
}

/// Write the object-outcome breakdown bar chart and the best-pure-coverage
/// histogram alongside the other tracking plots.
fn write_object_outcome_plots(
    outcomes: &[(photom::TrajId, ObjectOutcome, Option<f64>)],
    output_dir: &Utf8Path,
) -> Result<()> {
    let counts: Vec<(&str, usize)> = ObjectOutcome::all()
        .into_iter()
        .map(|outcome| {
            let count = outcomes.iter().filter(|(_, o, _)| *o == outcome).count();
            (outcome.label(), count)
        })
        .collect();
    let breakdown_path = output_dir.join("object_outcome_breakdown.png");
    plot_object_outcome_breakdown(&counts, &breakdown_path)?;

    let coverage_samples: Vec<f64> = outcomes
        .iter()
        .map(|(_, _, coverage)| coverage.unwrap_or(0.0))
        .collect();
    let coverage_path = output_dir.join("best_pure_coverage_ratio_histogram.png");
    plot_best_pure_coverage_histogram(&coverage_samples, &coverage_path)?;

    println!("  {breakdown_path}");
    println!("  {coverage_path}");
    Ok(())
}

/// `--from-snapshot` entry point: load the `.rkyv` `BranchCollection`
/// snapshot the real `fink-fat` engine binary wrote under `--config`'s
/// `storage_path`, and report structural stats plus (if the alerts file
/// carries ground truth) a reconstruction-efficacy breakdown — a
/// point-in-time audit of a real run, without re-running the simulation.
fn run_snapshot_analysis(cli: &TrackingAnalysisCli) -> Result<()> {
    let (_, obs_dataset) = load_data(&cli.common.alerts, None);
    let engine_config = EngineConfig::load_engine_config_validated(&cli.common.config)?;
    let kalman_ctx = engine_config.build_context();

    let snapshot_path = engine_config.snapshot_path();
    let collection =
        BranchCollection::load_snapshot_from_disk(&snapshot_path, &kalman_ctx, &engine_config)?;
    println!("Loaded snapshot from {snapshot_path}");

    let ground_truth = ObsTrajLookup::build(&obs_dataset);
    let obs_to_night = build_obs_to_night_map(&obs_dataset);

    let stats = compute_snapshot_stats(&collection, &obs_to_night);
    stats.print_summary();

    let efficacy = if ground_truth.has_ground_truth() {
        match determine_last_processed_night(&collection, &obs_to_night) {
            Some(last_processed_night) => {
                let gold_tracker = build_gold_tracker_for_processed_nights(
                    &obs_dataset,
                    &ground_truth,
                    last_processed_night,
                );
                let efficacy = compute_reconstruction_efficacy(
                    &collection,
                    &ground_truth,
                    &gold_tracker,
                    Some(last_processed_night),
                );
                efficacy.print_summary();
                Some(efficacy)
            }
            None => {
                println!(
                    "Ground truth present, but the snapshot has no branches referencing any \
                     observation — skipping reconstruction efficacy."
                );
                None
            }
        }
    } else {
        None
    };

    let output_dir = resolve_output_dir(&cli.common).join("snapshot_report");
    std::fs::create_dir_all(&output_dir)?;

    let stats_json_path = output_dir.join("snapshot_stats.json");
    stats.write_json(&stats_json_path)?;
    println!("Report written to {stats_json_path}");

    if let Some(efficacy) = &efficacy {
        let efficacy_json_path = output_dir.join("reconstruction_efficacy.json");
        efficacy.write_json(&efficacy_json_path)?;
        println!("Report written to {efficacy_json_path}");
    }

    write_snapshot_plots(&stats, efficacy.as_ref(), &output_dir)?;

    Ok(())
}

/// Write the snapshot-mode plots (structural histograms always, plus the
/// reconstruction-efficacy breakdown/coverage histogram when ground truth
/// was available).
fn write_snapshot_plots(
    stats: &SnapshotStats,
    efficacy: Option<&ReconstructionEfficacy>,
    output_dir: &Utf8Path,
) -> Result<()> {
    let hyp_path = output_dir.join("hypotheses_per_branch_histogram.png");
    let len_path = output_dir.join("track_length_per_branch_histogram.png");
    let night_path = output_dir.join("observations_per_night.png");
    plot_hypotheses_per_branch_histogram(stats, &hyp_path)?;
    plot_track_length_histogram(stats, &len_path)?;
    plot_snapshot_observations_per_night(stats, &night_path)?;

    println!("Plots written to:");
    for p in [&hyp_path, &len_path, &night_path] {
        println!("  {p}");
    }

    if let Some(efficacy) = efficacy {
        let breakdown_path = output_dir.join("reconstruction_outcome_breakdown.png");
        let coverage_path = output_dir.join("coverage_histogram.png");
        plot_reconstruction_outcome_breakdown(efficacy, &breakdown_path)?;
        plot_reconstruction_coverage_histogram(efficacy, &coverage_path)?;
        println!("  {breakdown_path}");
        println!("  {coverage_path}");
    }

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
    let object_outcome_plot = output_dir.join("object_outcome_per_night.png");
    let llr_plot = output_dir.join("llr_per_night.png");
    let error_box_plot = output_dir.join("error_box_radius_per_night.png");
    let in_box_plot = output_dir.join("observations_in_box_per_night.png");
    let timing_plot = output_dir.join("timing_per_night.png");

    plot_branches_and_lineages_per_night(report, &branches_plot)?;
    plot_recall_purity_completeness_per_night(report, &recall_plot)?;
    plot_object_outcome_per_night(report, &object_outcome_plot)?;
    plot_llr_and_ess_per_night(report, &llr_plot)?;
    plot_error_box_radius_per_night(report, &error_box_plot)?;
    plot_observations_in_box_per_night(report, &in_box_plot)?;
    plot_timing_per_night(report, &timing_plot)?;

    let histogram_paths = plot_aggregated_histograms(&aggregated, output_dir)?;

    println!("Plots written to:");
    for p in [
        branches_plot,
        recall_plot,
        object_outcome_plot,
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
