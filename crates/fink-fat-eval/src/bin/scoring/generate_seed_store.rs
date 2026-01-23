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
use std::{fs, time::Instant};

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
    log_timing,
    night_seeds::SeedStore,
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
    let t_src = Instant::now();
    let source = ParquetSource::new(&cli.scan.parquet)
        .with_context(|| format!("failed to open parquet source: {}", cli.scan.parquet))?;
    log_timing!(&cli, "open parquet source", t_src.elapsed());
    let ingest_cfg = AlertIngestConfig::default();

    let jobs = cli.jobs.unwrap_or_else(num_cpus::get);
    println!("Using {jobs} parallel jobs (threads)");

    ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build_global()
        .ok();

    println!("Loading nights...");

    let t_nights = Instant::now();
    let night_store: NightStore =
        collect_nights(&source, cli.scan.mode.into(), cli.scan.minimal, &ingest_cfg)?;
    log_timing!(&cli, "collect_nights", t_nights.elapsed());
    println!("Generating seeds...");

    let t_seeds = Instant::now();
    let seed_store = SeedStore::generate_nightseed_store(&night_store, &engine_cfg, 8, 1.0, false)?;
    log_timing!(&cli, "generate_nightseed_store", t_seeds.elapsed());

    println!("{seed_store}");

    println!("Writing results...");

    let seed_store_path = Utf8Path::new(&cli.scan.out_dir).join("seed_store.bin");
    seed_store
        .write(&seed_store_path)
        .with_context(|| "write seed store")?;

    println!("Done.");
    Ok(())
}
