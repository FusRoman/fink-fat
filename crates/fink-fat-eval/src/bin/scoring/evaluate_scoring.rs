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
    cli::scoring::{Cli, update_score_config},
    log, log_section, log_timing,
    night_seeds::SeedStore,
    scoring::frozen_pairs::{FrozenPair, compute_fast_objective},
};
use rayon::ThreadPoolBuilder;

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
    let seed_store = SeedStore::read(&seed_store_path)
        .with_context(|| format!("read seed store from {}", seed_store_path))?;

    let frozen_pair_path = Utf8Path::new(&cli.scan.out_dir).join("frozen_pairs_by_delta.bin");
    let frozen_pairs = FrozenPair::read(&frozen_pair_path)
        .with_context(|| format!("read frozen pairs from {}", frozen_pair_path))?;

    log!(cli, "Seed store: {seed_store}");
    log!(cli, "Total frozen pairs loaded: {}", frozen_pairs.len());
    log_timing!(cli, "Time to load data", t0.elapsed());

    // -------------------------------------------------------------------------
    // Evaluate
    // -------------------------------------------------------------------------
    log_section!(cli, "Evaluating pairs");

    let t_eval_start = std::time::Instant::now();

    let eval = FrozenPair::eval_cfg_fast_items(
        &seed_store,
        &frozen_pairs,
        &updated_config.edges.score_config,
        cli.budget,
    );

    let Some((items, stats)) = eval else {
        anyhow::bail!("no frozen pairs could be evaluated with the provided configuration");
    };

    log_timing!(cli, "Time to evaluate frozen pairs", t_eval_start.elapsed());

    // -------------------------------------------------------------------------
    // Metrics
    // -------------------------------------------------------------------------
    log_section!(cli, "Computing metrics");

    let t_metrics_start = std::time::Instant::now();

    let objective = compute_fast_objective(
        &items, stats, cli.target_tpr, // target TPR
        cli.min_accept_good,  // min accept good rate
        cli.penalty_weight,  // penalty weight
    );

    if cli.quiet {
        // Machine-friendly output (Optuna)
        println!("{objective}");
    } else {
        println!("Objective (FPR@TPR={} + penalties):", cli.target_tpr);
        println!("{objective}");
    }

    log_timing!(cli, "Time to compute metrics", t_metrics_start.elapsed());
    log_timing!(cli, "Total time", t0.elapsed());

    Ok(())
}
