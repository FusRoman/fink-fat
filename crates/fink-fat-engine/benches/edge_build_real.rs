//! Benchmark replicating the `BuildEdges` pipeline stage with real alert data.
//!
//! Designed for profiling with `perf` + Firefox Profiler or `samply`.
//!
//! # Quick start
//!
//! ```bash
//! # Option A – samply (output opens directly in Firefox Profiler)
//! cargo install samply
//! samply record cargo bench --profile evaluation -p fink-fat-engine --bench edge_build_real -- --profile-time 30
//!
//! # Option B – perf + manual upload
//! cargo build --profile evaluation -p fink-fat-engine --benches
//! BENCH=$(ls target/evaluation/deps/edge_build_real-* | grep -v '\.d' | head -1)
//! perf record -F 997 --call-graph dwarf -g -- "$BENCH" --bench --profile-time 30
//! perf script | gzip > edge_build_profile.gz
//! # then upload edge_build_profile.gz to https://profiler.firefox.com
//! ```
//!
//! # Environment variables
//!
//! | Variable                  | Default                                              |
//! |---------------------------|------------------------------------------------------|
//! | `FINK_FAT_ALERTS_PARQUET` | `test_exp/sso_dataset_eval.parquet`                  |
//! | `FINK_FAT_CONFIG`         | `crates/fink-fat-eval/eval_config_best.yml`          |
//!
//! The paths are resolved relative to the **workspace root** (run the bench
//! from there, or set the variables to absolute paths).

use std::hint::black_box;
use std::time::{Duration, Instant};

use camino::Utf8PathBuf;
use criterion::{Criterion, criterion_group, criterion_main};

use fink_fat_engine::{
    engine_config::{load_engine_config_validated, pipeline_policy::PersistPolicy},
    graph::{AlertLinkageDAG, edge::edge_prediction::EdgeRankingModelPool},
    persistence::{PersistenceManager, runtime_state::RuntimeState},
    pipeline::{
        PipelineContext, PipelineInputs, PipelinePlan,
        hooks::NoopHooks,
        stages::{PipelineStage, alert_inputs::input_uri::InputUri},
    },
    solver::solver_manager::SolverManager,
};

const DEFAULT_ALERTS: &str = "test_exp/sso_dataset_eval.parquet";
const DEFAULT_CONFIG: &str = "crates/fink-fat-eval/eval_config_best.yml";

fn bench_build_edges_real(c: &mut Criterion) {
    // ── 1. Resolve paths ─────────────────────────────────────────────────────
    let alerts_path =
        std::env::var("FINK_FAT_ALERTS_PARQUET").unwrap_or_else(|_| DEFAULT_ALERTS.to_string());
    let config_path =
        std::env::var("FINK_FAT_CONFIG").unwrap_or_else(|_| DEFAULT_CONFIG.to_string());

    // ── 2. Load engine config ─────────────────────────────────────────────────
    let engine_config = load_engine_config_validated(camino::Utf8Path::new(&config_path))
        .unwrap_or_else(|e| panic!("failed to load engine config from {config_path}: {e}"));

    // ── 3. Build model pool (None when ml_post_filter is false) ───────────────
    let model_pool: Option<EdgeRankingModelPool> = engine_config
        .edges
        .edge_ranking_model_path
        .as_deref()
        .map(|p| EdgeRankingModelPool::new(camino::Utf8Path::new(p)));

    // ── 4. Minimal persistence (tempdir, never written to disk) ───────────────
    let tmpdir = tempfile::tempdir().expect("failed to create tempdir");
    let tmpdir_utf8 = Utf8PathBuf::from_path_buf(tmpdir.path().to_path_buf())
        .expect("tempdir path is not valid UTF-8");
    let persistence =
        PersistenceManager::open_or_create(tmpdir_utf8).expect("failed to open persistence");

    // ── 5. Solver manager ─────────────────────────────────────────────────────
    let solver_manager = SolverManager {
        policy: engine_config.solver_config.solver_policy,
        bounded_beam_config: engine_config.solver_config.bounded_beam.clone(),
    };

    // ── 6. Build pipeline plan ────────────────────────────────────────────────
    let alerts_uri = format!(
        "file://{}",
        std::fs::canonicalize(&alerts_path)
            .unwrap_or_else(|e| panic!("cannot canonicalize {alerts_path}: {e}"))
            .display()
    )
    .parse::<InputUri>()
    .expect("invalid alerts URI");

    let plan = PipelinePlan {
        stages: vec![
            PipelineStage::IngestNights,
            PipelineStage::BuildSeeds,
            PipelineStage::BuildEdges,
        ],
        persist: PersistPolicy::None,
        inputs: PipelineInputs { alerts_uri },
    };

    // ── 7. One-time setup: load alerts and build seeds ────────────────────────
    let mut runtime_state = RuntimeState::new();
    {
        let mut ctx = PipelineContext {
            plan: &plan,
            persistence: &persistence,
            runtime_state: &mut runtime_state,
            engine_config: &engine_config,
            edge_models: &model_pool,
            solver_manager: &solver_manager,
        };
        PipelineStage::IngestNights
            .run(&mut ctx, &NoopHooks)
            .expect("IngestNights failed");
        PipelineStage::BuildSeeds
            .run(&mut ctx, &NoopHooks)
            .expect("BuildSeeds failed");
    }

    let n_nights = runtime_state.seed_store.nights().count();
    let n_seeds: usize = runtime_state
        .seed_store
        .nights()
        .filter_map(|id| runtime_state.seed_store.get(id))
        .map(|s| s.len())
        .sum();
    eprintln!("Setup: {n_nights} nights, {n_seeds} seeds total");

    // ── 8. Benchmark: repeated BuildEdges (graph reset between iterations) ────
    let mut group = c.benchmark_group("build_edges_real");
    // Keep wall-clock time short: 10 samples (criterion minimum) with a brief
    // warm-up.  Each BuildEdges call can be several seconds, so this is enough
    // to get a stable estimate without waiting for the default 100 samples.
    // Override from the CLI if needed:
    //   -- --sample-size 10 --warm-up-time 1 --measurement-time 5
    // For profiling use `--profile-time 30` instead (single sample):
    //   samply record cargo bench --profile evaluation --bench edge_build_real -- --profile-time 30
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(30));

    group.bench_function("all_pairs", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                // Reset the graph so each iteration starts from an empty DAG.
                // This is nearly free (just drops a HashMap) and matches what
                // fink-fat-eval does between calls in a streaming pipeline.
                runtime_state.graph = AlertLinkageDAG::new();

                let mut ctx = PipelineContext {
                    plan: &plan,
                    persistence: &persistence,
                    runtime_state: &mut runtime_state,
                    engine_config: &engine_config,
                    edge_models: &model_pool,
                    solver_manager: &solver_manager,
                };

                let t = Instant::now();
                black_box(
                    PipelineStage::BuildEdges
                        .run(&mut ctx, &NoopHooks)
                        .expect("BuildEdges failed"),
                );
                total += t.elapsed();
            }
            total
        })
    });

    group.finish();
}

criterion_group!(benches, bench_build_edges_real);
criterion_main!(benches);
