use std::time::Instant;

/// Command-line tool to optimize position scoring parameters
/// using balanced inter-night edge samples.
//// The tool performs a random search over position scoring parameters,
/// evaluating each candidate configuration on a set of balanced inter-night edges
/// frozen from the provided dataset. The goal is to minimize the false positive rate (FPR)
/// at a target true positive rate (TPR) by adjusting scoring parameters related to position differences.

/// Example command to run the tool:
/// ```bash
/// clear && cargo run     \
///     --release     \
///     -p fink-fat-eval     \
///     --bin evaluate-scoring \
///     ../../test_exp/ztf_dataset_2025.parquet \
///     --engine-config src/bin/scoring/config_engine_best.yaml \
///     --mode fink-truth \
///     --max-nights 5 \
///     --jobs 1 \
///     --quiet \
///     --out-dir seed_store
/// ```
use anyhow::{Context, Result};
use camino::Utf8Path;
use clap::Parser;
use fink_fat_engine::{
    engine_config::{EngineConfig, load_engine_config_validated},
    graph::edge::{Edge, edge_id::EdgeId},
    night_id::NightId,
    spacetime_bucket::{healpix_binner::HealpixBinner, uniform_time_binner::UniformTimeBinner},
};
use fink_fat_eval::{
    bin_utils::resolve_nids,
    buflog, buflog_timing, buflog2,
    cli::scoring::{Cli, update_score_config},
    log, log_section, log_timing, log2,
    night_seeds::{LabeledEdge, NightSeeds, SeedStore},
    scoring::{
        edges_diagnostics::EdgesStatsDisplay,
        optimization_metrics::{
            EdgeSeparationMetrics,
            objective::{LinkingObjectiveConfig, objective_linking_quality},
        },
    },
};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::ThreadPoolBuilder;

use rayon::prelude::*;

/// Build the list of (left, right, gap) pairs to evaluate.
///
/// - `consecutive_window`: evaluates i -> i+1..i+W
/// - `gap_min..=gap_max`: evaluates i -> i+gap for each gap in the range
///
/// Returned `gap` is the number of nights between `left` and `right` (>= 1).
fn build_night_pairs(
    nids: &[NightId],
    consecutive_window: usize,
    gap_min: usize,
    gap_max: usize,
) -> Vec<(NightId, NightId, usize)> {
    let mut out = Vec::new();
    if nids.len() < 2 {
        return out;
    }

    let w = consecutive_window.max(1);

    // 1) Sliding consecutive window: i -> i+1..i+w
    for i in 0..nids.len().saturating_sub(1) {
        for k in 1..=w {
            let j = i + k;
            if j >= nids.len() {
                break;
            }
            out.push((nids[i], nids[j], k));
        }
    }

    // 2) Explicit gaps: i -> i+gap_min..i+gap_max
    if gap_min >= 1 && gap_max >= gap_min {
        for i in 0..nids.len() {
            for g in gap_min..=gap_max {
                let j = i + g;
                if j >= nids.len() {
                    break;
                }
                out.push((nids[i], nids[j], g));
            }
        }
    }

    out
}

/// Convert resolved i32 night ids into typed NightId.
fn to_night_ids(nids: Vec<u32>) -> Vec<NightId> {
    nids.into_iter().map(NightId).collect()
}

/// Step 1: build the time binner (anchored on `right`) and generate Top-K edges.
fn step_generate_topk_edges<'a>(
    logbuf: &mut String,
    cli: &Cli,
    left: &'a NightSeeds,
    right: &'a NightSeeds,
    engine_cfg: &EngineConfig,
    spatial_binner: &HealpixBinner,
) -> (UniformTimeBinner, Vec<Edge<'a>>) {
    // Build time binner anchored on the right night (stable).
    let t_timebin = Instant::now();
    let min_time = right
        .seeds
        .iter()
        .map(|s| s.plane.epoch_mid)
        .fold(f64::INFINITY, f64::min);

    let time_binner = UniformTimeBinner::new(min_time, 30.0 / 60.0 / 24.0); // 30 minutes
    buflog_timing!(logbuf, cli, "build time binner", t_timebin.elapsed());

    // Edge generation
    let t_edges = Instant::now();
    let edges = Edge::generate_topk_edges(
        EdgeId(0),
        &left.seeds,
        &right.seeds,
        &engine_cfg.edges,
        spatial_binner,
        &time_binner,
    );
    buflog_timing!(logbuf, cli, "generate_topk_edges", t_edges.elapsed());

    buflog!(
        logbuf,
        cli,
        "Edges: total={}, active={} (left seeds={}, right seeds={})",
        edges.len(),
        edges.iter().filter(|e| e.active).count(),
        left.seeds.len(),
        right.seeds.len()
    );

    // Optional detailed view
    buflog2!(
        logbuf,
        cli,
        "{}",
        EdgesStatsDisplay::new(&edges).top_k(20).only_active(false)
    );

    (time_binner, edges)
}

fn main() -> Result<()> {
    let t0 = std::time::Instant::now();
    let cli = Cli::parse();

    // -------------------------------------------------------------------------
    // Engine config
    // -------------------------------------------------------------------------
    let engine_cfg: EngineConfig = load_engine_config_validated(&cli.engine_config)
        .with_context(|| format!("load engine config {}", cli.engine_config))?;

    let updated_config = update_score_config(&cli, &engine_cfg);

    let jobs = cli.jobs.unwrap_or_else(num_cpus::get);

    log!(cli, "Using {jobs} parallel jobs (threads)");

    ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build_global()
        .ok();

    log_timing!(cli, "Script initialized", t0.elapsed());

    // -------------------------------------------------------------------------
    // Load data
    // -------------------------------------------------------------------------
    log_section!(cli, "Loading data");

    let seed_store_path = Utf8Path::new(&cli.scan.out_dir).join("seed_store.bin");
    let seed_store = SeedStore::read(&seed_store_path)?;

    log!(cli, "Seed store: {seed_store}");
    log_timing!(cli, "Time to load data", t0.elapsed());

    log!(&cli, "SeedStore nights: {}", seed_store.len());
    log2!(&cli, "{seed_store}");

    log_section!(&cli, "Night pair selection");

    let t_resolve = Instant::now();
    let nids: Vec<u32> = resolve_nids(&cli.scan.parquet, cli.nids.as_deref(), cli.max_nights)?;
    log_timing!(&cli, "resolve_nids", t_resolve.elapsed());
    anyhow::ensure!(nids.len() >= 2, "need at least 2 nights to evaluate edges");

    let consecutive_window: usize = 5;
    let gap_min: usize = 0;
    let gap_max: usize = 0;

    let t_pairs = Instant::now();
    let pairs = build_night_pairs(&to_night_ids(nids), consecutive_window, gap_min, gap_max);
    log_timing!(&cli, "build_night_pairs", t_pairs.elapsed());
    anyhow::ensure!(!pairs.is_empty(), "no night pairs to evaluate");

    log!(
        &cli,
        "Pairs to evaluate: {} (consecutive_window={}, gap=[{},{}])",
        pairs.len(),
        consecutive_window,
        gap_min,
        gap_max
    );

    let spatial_binner = HealpixBinner::new(cli.binning.healpix_depth);

    log_section!(&cli, "Evaluation loop");

    let nb_pairs = pairs.len();

    let pb = if cli.quiet {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new(nb_pairs as u64);
        pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta_precise}) {msg}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
        pb.enable_steady_tick(std::time::Duration::from_millis(120));
        pb.set_message("starting…");
        pb
    };

    // -------------------------------------------------------------------------
    // Evaluate
    // -------------------------------------------------------------------------

    // 1) Compute in parallel (no stdout).
    let outs: Vec<Option<EdgeSeparationMetrics>> = pairs
        .par_iter()
        .enumerate()
        .map(
            |(idx, (left_nid, right_nid, gap))| -> Option<EdgeSeparationMetrics> {
                let mut logbuf = String::new();
                pb.set_message(format!("{:?} -> {:?} (gap={})", left_nid, right_nid, gap));

                buflog!(
                    &mut logbuf,
                    &cli,
                    "[{}/{}] evaluating {:?} -> {:?} (gap={})",
                    idx + 1,
                    nb_pairs,
                    left_nid,
                    right_nid,
                    gap
                );

                let left = seed_store.get(left_nid).unwrap();
                let right = seed_store.get(right_nid).unwrap();

                let t_pair = Instant::now();

                // Step 1: generate top-k edges
                let (_, edges) = step_generate_topk_edges(
                    &mut logbuf,
                    &cli,
                    left,
                    right,
                    &updated_config,
                    &spatial_binner,
                );

                let t_metrics = Instant::now();
                let labeled_edges: Vec<LabeledEdge> = edges
                    .clone()
                    .into_iter()
                    .map(|e| LabeledEdge::from_edge(e, left, right))
                    .collect();

                let metrics = EdgeSeparationMetrics::edge_metrics(
                    &labeled_edges,
                    left.nb_true_possible_edges(right) as usize,
                );

                buflog_timing!(
                    logbuf,
                    cli,
                    "compute optimization metrics",
                    t_metrics.elapsed()
                );

                buflog_timing!(&mut logbuf, &cli, "pair total", t_pair.elapsed());

                // nice separator (buffered)
                if !cli.quiet {
                    logbuf.push_str(
                        "\n ____________________________________________________________\n\n",
                    );
                }

                pb.inc(1);
                metrics
            },
        )
        .collect();

    pb.finish_with_message("evaluation complete");

    // 2) Aggregate results and print to stdout.
    let agg_metrics = EdgeSeparationMetrics::aggregate(&outs).unwrap();

    log_section!(&cli, "Final aggregated metrics");
    log!(&cli, "{}", agg_metrics);

    let cfg = LinkingObjectiveConfig::default();
    let obj = objective_linking_quality(&agg_metrics, cfg);

    println!("{obj}");

    Ok(())
}
