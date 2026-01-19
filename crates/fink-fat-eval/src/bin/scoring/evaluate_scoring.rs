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
use anyhow::{Context, Result};
use camino::Utf8Path;
use clap::Parser;
use fink_fat_engine::engine_config::{EngineConfig, load_engine_config_validated};
use fink_fat_eval::{
    cli::scoring::Cli,
    night_seeds::SeedStore,
    scoring::frozen_pairs::{FrozenPair, compute_fast_objective},
};
use rayon::ThreadPoolBuilder;

fn main() -> Result<()> {
    let t0 = std::time::Instant::now();

    let cli = Cli::parse();

    // Load engine configuration
    let engine_cfg: EngineConfig = load_engine_config_validated(&cli.engine_config)
        .with_context(|| format!("load engine config {}", cli.engine_config))?;

    let jobs = cli.jobs.unwrap_or_else(num_cpus::get);
    println!("Using {jobs} parallel jobs (threads)");

    ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build_global()
        .ok();

    let t_init= std::time::Instant::now();
    println!("Script initialized in {:.1?}", t_init.duration_since(t0));

    println!("\n=============== Loading data ==================\n");

    let seed_store_path = Utf8Path::new(&cli.scan.out_dir).join("seed_store.bin");
    let seed_store = SeedStore::read(&seed_store_path)
        .with_context(|| format!("read seed store from {}", seed_store_path))?;

    let frozen_pair_path = Utf8Path::new(&cli.scan.out_dir).join("frozen_pairs_by_delta.bin");
    let frozen_pairs = FrozenPair::read(&frozen_pair_path)
        .with_context(|| format!("read frozen pairs from {}", frozen_pair_path))?;

    println!("Seed store: {seed_store}");
    println!("Total frozen pairs loaded: {}", frozen_pairs.len());

    let t_load = std::time::Instant::now();
    println!("Time to load data: {:.1?}", t_load.duration_since(t0));

    println!("\n=============== Evaluating pairs ==================\n");

    if let Some((items, stats)) = FrozenPair::eval_cfg_fast_items(
        &seed_store,
        &frozen_pairs,
        &engine_cfg.scoring,
        Some(500_000),
    ) {
        let t_eval = std::time::Instant::now();
        println!(
            "Time to evaluate frozen pairs: {:.1?}",
            t_eval.duration_since(t_load)
        );

        println!("\n=============== Computing metrics ==================\n");

        let metrics = compute_fast_objective(&items, stats, 0.95, 0.5, 0.5);

        println!("FPR at 95% TPR:");
        println!("{metrics}");

        let t_metrics = std::time::Instant::now();
        println!("Time to compute metrics: {:.1?}", t_metrics.duration_since(t_eval));

        println!("\nTotal time: {:.1?}", t_metrics.duration_since(t0));

        Ok(())
    } else {
        anyhow::bail!("no frozen pairs could be evaluated with the provided configuration");
    }
}
