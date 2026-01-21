/// Command-line tool to generate frozen pairs
/// from a dataset of alerts.
/// The tool processes alerts from multiple nights,
/// generates seeds based on truth information,
/// and creates frozen pairs for scoring evaluation.
///
/// Example command to run the tool:
/// ```bash
/// clear && cargo run \
///     --release \
///     -p fink-fat-eval \
///     --bin generate-frozen-pairs \
///     ../../test_exp/ztf_dataset_2025.parquet \
///     --engine-config src/bin/scoring/config_engine.yaml \
///     --jobs 8 \
///     --only-truth false \
///     --mode fink-truth \
///     --rng-seed 04071997
/// ```
use std::fs;

use anyhow::{Context, Result};
use camino::Utf8Path;
use clap::Parser;
use fink_fat_engine::engine_config::{EngineConfig, load_engine_config_validated};
use fink_fat_eval::{
    cli::scoring::Cli,
    dataset::{
        ParquetSource,
        ingest_config::AlertIngestConfig,
        ztf_alerts::{NightStore, collect_nights},
    },
    night_seeds::SeedStore,
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
        SeedStore::seed_store_from_night_store_truth(&night_store, true, 1., Some(42), None);

    println!("{seed_store}");

    println!("Generating frozen pairs...");

    let frozen_pairs = FrozenPair::frozen_pairs(&seed_store, &engine_cfg.edges.score_config, 15, 1., Some(42));

    println!("Total frozen pairs generated: {}", frozen_pairs.len());

    println!("Writing results...");
    let frozen_pair_path = Utf8Path::new(&cli.scan.out_dir).join("frozen_pairs_by_delta.bin");
    FrozenPair::write(&frozen_pair_path, &frozen_pairs)
        .with_context(|| format!("write frozen pairs to {}", frozen_pair_path))?;

    let seed_store_path = Utf8Path::new(&cli.scan.out_dir).join("seed_store.bin");
    seed_store
        .write(&seed_store_path)
        .with_context(|| "write seed store")?;

    println!("Done.");
    Ok(())
}
