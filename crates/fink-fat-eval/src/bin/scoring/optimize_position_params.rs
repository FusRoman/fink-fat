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
    night_seeds::{LabeledEdgesByDelta, LabeledEdgesByDeltaDisplay, SeedStore},
    scoring::frozen_pairs::FrozenPair,
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
        SeedStore::seed_store_from_night_store_truth(&night_store, true, 1.1, Some(42), None);

    println!("{seed_store}");

    let frozen_pairs =
        FrozenPair::frozen_pairs(&seed_store, &engine_cfg.scoring, 10, 1.0, Some(42));

    let edges =
        FrozenPair::eval_cfg_on_frozen_pairs(&seed_store, &frozen_pairs, &engine_cfg.scoring);

    println!("Total frozen pairs evaluated: {}", edges.len());
    println!(
        "  - good edges: {}",
        edges.iter().filter(|e| e.same).count()
    );
    println!(
        "  - bad edges: {}",
        edges.iter().filter(|e| !e.same).count()
    );

    let buckets: LabeledEdgesByDelta = seed_store.labeled_edges_by_delta(
        &engine_cfg.scoring,
        10,
        false,
        Some(2_000_000),
        Some(42),
    );

    println!("{}", LabeledEdgesByDeltaDisplay(&buckets));

    Ok(())
}
