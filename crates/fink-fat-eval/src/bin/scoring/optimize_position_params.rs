/// Command-line tool to optimize position scoring parameters
/// using balanced inter-night edge samples.
//// The tool performs a random search over position scoring parameters,
/// evaluating each candidate configuration on a set of balanced inter-night edges
/// frozen from the provided dataset. The goal is to minimize the false positive rate (FPR)
/// at a target true positive rate (TPR) by adjusting scoring parameters related to position differences.

/// Example command to run the tool:
/// ```bash
/// clear && cargo run \
///     --release \
///     -p fink-fat-eval \
///     --bin optimize-position-params \
///     ../../test_exp/ztf_dataset_2025.parquet \
///     --engine-config src/bin/scoring/config_engine.yaml \
///     --jobs 8 \
///     --only-truth false \
///     --mode fink-truth \
///     --rng-seed 04071997
/// ```
use std::fs;

use anyhow::{Context, Result};
use clap::Parser;
use fink_fat_engine::engine_config::{EngineConfig, load_engine_config_validated};
use fink_fat_eval::{
    bin_utils::resolve_nids,
    cli::scoring::Cli,
    dataset::{
        ParquetSource,
        ingest_config::AlertIngestConfig,
        ztf_alerts::{NightStore, collect_nights},
    },
    night_seeds::SeedStore,
    scoring::{
        frozen_pairs::EdgeSampling, optim_writer::write_best_scoring_yamls,
        optimizer_position_params::optimize_scoring_params,
    },
};
use rayon::ThreadPoolBuilder;

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Ensure output directory exists
    fs::create_dir_all(&cli.scan.out_dir)
        .with_context(|| format!("create {}", cli.scan.out_dir))?;

    // Load engine configuration
    let engine_cfg: EngineConfig = load_engine_config_validated(&cli.engine_config)
        .with_context(|| format!("load engine config {}", cli.engine_config))?;

    // Dataset source
    let source = ParquetSource::new(&cli.scan.parquet)
        .with_context(|| format!("failed to open parquet source: {}", cli.scan.parquet))?;
    let ingest_cfg = AlertIngestConfig::default();

    // Determine night IDs
    let nids = resolve_nids(&cli.scan.parquet, cli.nids.as_deref(), cli.max_nights)?;
    anyhow::ensure!(
        nids.len() >= 2,
        "need at least 2 nights to inspect inter-night edges"
    );

    let jobs = cli.jobs.unwrap_or_else(num_cpus::get);
    println!("Using {jobs} parallel jobs (threads)");

    ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build_global()
        .ok();

    println!("Loading nights...");

    let night_store: NightStore =
        collect_nights(&source, cli.scan.mode.into(), cli.scan.minimal, &ingest_cfg)?;

    println!("Generating seeds...");

    let seed_store =
        SeedStore::seed_store_from_night_store_truth(&night_store, true, 1.0, Some(42), None);

    println!("{seed_store}");

    let sampling = EdgeSampling {
        sample_right_per_left: 2048,
        max_night_jump: Some(15),
        target_per_class: 300_000,
        only_truth: false,
        base_seed: cli.rng_seed,
        max_tested_pairs_per_night_pair: 1_000_000,
    };

    println!("\n\n === Starting optimization ===\n");

    let optim_result = optimize_scoring_params(&seed_store, &engine_cfg, &sampling)?;

    println!("\n\n === Optimization complete ===\n");
    println!("Best config found:");
    println!("{:?}", optim_result.best_cfg);
    println!("\n\nBest FPR at target TPR: \n{:.6}", optim_result.best_fpr);
    println!(
        "\n\nBest separation metrics: \n{:?}",
        optim_result.best_metrics
    );

    write_best_scoring_yamls(
        &cli.scan.out_dir,
        &engine_cfg,
        optim_result.best_cfg.unwrap(),
    )?;

    Ok(())
}
