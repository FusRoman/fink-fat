//! Integration test suite for the fink-fat-engine pipeline stages.
//!
//! This module contains shared test infrastructure (helpers, configs, hooks)
//! used by all pipeline integration tests, plus the test submodules themselves.

mod build_edges_test;
mod build_seeds_test;
mod fit_orbit_test;
mod ingest_alerts_test;
mod persistence_test;
mod solver_stage_test;

// ===========================================================================
// Shared test infrastructure
// ===========================================================================
//
// The items below are used by multiple integration test files to avoid
// duplicating boilerplate (NoopHooks, config builders, empty RuntimeState, …).

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, Float64Array, RecordBatch, StringArray, UInt8Array, UInt32Array, UInt64Array,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::parquet::arrow::ArrowWriter;
use fink_fat_engine::engine_config::pipeline_policy::PersistPolicy;
use tempfile::TempDir;

use fink_fat_engine::{
    Alert, AlertStore,
    engine_config::{
        EngineConfig,
        solver_config::{
            bounded_beam_config::BoundedBeamConfig,
            solver_policy::{SolverChoice, SolverPolicy},
        },
    },
    graph::edge::edge_prediction::EdgeRankingModelPool,
    night_id::NightId,
    persistence::{PersistenceManager, runtime_state::RuntimeState},
    pipeline::{
        PipelineContext, PipelineInputs, PipelineOutput, PipelinePlan, PipelineRunner,
        stages::{PipelineStage, alert_inputs::input_uri::InputUri},
    },
    seeding::{SeedKey, store::SeedStore},
    solver::{HypothesisSet, solver_manager::SolverManager},
    trajectory::TrackHypothesis,
};

use crate::synthetic_alerts::{SyntheticDataset, TrajectoryTruth};

// ---------------------------------------------------------------------------
// No-op pipeline hooks
// ---------------------------------------------------------------------------

// Re-export the library's built-in no-op hooks so test modules can use it.
pub(crate) use fink_fat_engine::pipeline::hooks::NoopHooks;

// ---------------------------------------------------------------------------
// EngineConfig builders
// ---------------------------------------------------------------------------

/// Build a minimal `EngineConfig` with only a `storage_path`.
///
/// Suitable for tests that only run `IngestNights` and/or `BuildSeeds`
/// (no edge builder or solver).
pub(crate) fn engine_config_minimal(storage_dir: &TempDir) -> EngineConfig {
    let storage_path = storage_dir.path().to_str().unwrap();
    let yaml = format!(
        r#"
version: 1
storage_path: "{storage_path}"
"#
    );
    serde_yaml::from_str(&yaml).expect("deserialize minimal EngineConfig")
}

/// Build an `EngineConfig` with `emit_all_edges: true` and a custom
/// `max_gap_nights` so that the edge builder works without an ONNX model.
///
/// Suitable for tests that run `BuildEdges` and/or `Solve`.
pub(crate) fn engine_config_with_edges(storage_dir: &TempDir, max_gap_nights: u8) -> EngineConfig {
    let storage_path = storage_dir.path().to_str().unwrap();
    let yaml = format!(
        r#"
version: 1
storage_path: "{storage_path}"
max_gap_nights: {max_gap_nights}
edges:
  emit_all_edges: true
"#
    );
    serde_yaml::from_str(&yaml).expect("deserialize EngineConfig with emit_all_edges")
}

/// Build an `EngineConfig` with `emit_all_edges: true`, a custom
/// `max_gap_nights`, and a custom `compact_graph_every_delta` threshold.
///
/// Suitable for tests that verify edge journal compaction behavior.
pub(crate) fn engine_config_with_compaction(
    storage_dir: &TempDir,
    max_gap_nights: u8,
    compact_every: usize,
) -> EngineConfig {
    let storage_path = storage_dir.path().to_str().unwrap();
    let yaml = format!(
        r#"
version: 1
storage_path: "{storage_path}"
max_gap_nights: {max_gap_nights}
compact_graph_every_delta: {compact_every}
edges:
  emit_all_edges: true
"#
    );
    serde_yaml::from_str(&yaml).expect("deserialize EngineConfig with compaction threshold")
}

// ---------------------------------------------------------------------------
// Parquet writer for alert subsets
// ---------------------------------------------------------------------------

/// Arrow schema compatible with `AlertParquetColumns::default()`.
fn parquet_alert_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("night_id", DataType::UInt32, false),
        Field::new("dia_source_id", DataType::UInt64, false),
        Field::new("ra", DataType::Float64, false),
        Field::new("ra_err", DataType::Float64, false),
        Field::new("dec", DataType::Float64, false),
        Field::new("dec_err", DataType::Float64, false),
        Field::new("mjd_tt", DataType::Float64, false),
        Field::new("flux", DataType::Float64, false),
        Field::new("flux_err", DataType::Float64, false),
        Field::new("band", DataType::UInt8, false),
        Field::new("observer_mpc_code", DataType::Utf8, false),
    ]))
}

/// Write a slice of `Alert` references to a Parquet file and return the URI.
///
/// This is useful for incremental (night-by-night) tests that need to write
/// one night of alerts at a time, as opposed to
/// `SyntheticDataset::write_parquet` which writes the entire dataset.
pub(crate) fn write_alerts_parquet(alerts: &[&Alert], path: &Path) -> InputUri {
    let schema = parquet_alert_schema();
    let n = alerts.len();

    let mut night_ids = Vec::with_capacity(n);
    let mut dia_source_ids = Vec::with_capacity(n);
    let mut ras = Vec::with_capacity(n);
    let mut ra_errs = Vec::with_capacity(n);
    let mut decs = Vec::with_capacity(n);
    let mut dec_errs = Vec::with_capacity(n);
    let mut mjd_tts = Vec::with_capacity(n);
    let mut fluxes = Vec::with_capacity(n);
    let mut flux_errs = Vec::with_capacity(n);
    let mut bands = Vec::with_capacity(n);
    let mut observer_codes: Vec<String> = Vec::with_capacity(n);

    for alert in alerts {
        night_ids.push(alert.key.night_id.0);
        dia_source_ids.push(alert.key.dia_source_id);
        ras.push(alert.ra);
        ra_errs.push(alert.ra_err);
        decs.push(alert.dec);
        dec_errs.push(alert.dec_err);
        mjd_tts.push(alert.mjd_tt);
        fluxes.push(alert.flux);
        flux_errs.push(alert.flux_err);
        bands.push(alert.band);
        observer_codes.push((*alert.observer_mpc_code).clone());
    }

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(night_ids)) as ArrayRef,
            Arc::new(UInt64Array::from(dia_source_ids)) as ArrayRef,
            Arc::new(Float64Array::from(ras)) as ArrayRef,
            Arc::new(Float64Array::from(ra_errs)) as ArrayRef,
            Arc::new(Float64Array::from(decs)) as ArrayRef,
            Arc::new(Float64Array::from(dec_errs)) as ArrayRef,
            Arc::new(Float64Array::from(mjd_tts)) as ArrayRef,
            Arc::new(Float64Array::from(fluxes)) as ArrayRef,
            Arc::new(Float64Array::from(flux_errs)) as ArrayRef,
            Arc::new(UInt8Array::from(bands)) as ArrayRef,
            Arc::new(StringArray::from(observer_codes)) as ArrayRef,
        ],
    )
    .expect("build record batch");

    let file = std::fs::File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("create ArrowWriter");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close parquet writer");

    InputUri(format!("file://{}", path.to_str().unwrap()))
}

// ---------------------------------------------------------------------------
// Ground-truth matching utilities
// ---------------------------------------------------------------------------

/// Collect the set of `dia_source_id` values for a track hypothesis by
/// resolving its seed nodes through the alert and seed stores.
pub(crate) fn track_dia_source_ids(
    track: &TrackHypothesis,
    alert_store: &AlertStore,
    seed_store: &SeedStore,
) -> HashSet<u64> {
    let mut ids = HashSet::new();
    for &seed_key in &track.nodes {
        let seed = seed_store
            .try_get_seed(seed_key)
            .expect("seed key must exist in store");
        for &alert_key in &seed.members {
            if let Some(alert) = alert_store.get_by_key(alert_key) {
                ids.insert(alert.key.dia_source_id);
            }
        }
    }
    ids
}

/// Compute the Jaccard overlap between two sets: $|A \cap B| / |A \cup B|$.
pub(crate) fn jaccard(a: &HashSet<u64>, b: &HashSet<u64>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    intersection as f64 / union as f64
}

/// For each ground-truth trajectory, find the best-matching solver hypothesis
/// based on Jaccard overlap of `dia_source_id` sets.
///
/// Returns a list of `(truth_idx, best_hypothesis_id, jaccard_score)`.
pub(crate) fn match_truth_to_hypotheses(
    ground_truth: &[TrajectoryTruth],
    hypotheses: &HypothesisSet,
    alert_store: &AlertStore,
    seed_store: &SeedStore,
) -> Vec<(usize, Option<u32>, f64)> {
    let hyp_sets: HashMap<u32, HashSet<u64>> = hypotheses
        .iter()
        .map(|(&hid, track)| (hid, track_dia_source_ids(track, alert_store, seed_store)))
        .collect();

    ground_truth
        .iter()
        .enumerate()
        .map(|(tidx, truth)| {
            let truth_set: HashSet<u64> = truth.dia_source_ids.iter().copied().collect();
            let mut best_id: Option<u32> = None;
            let mut best_score: f64 = 0.0;

            for (&hid, hset) in &hyp_sets {
                let score = jaccard(&truth_set, hset);
                if score > best_score {
                    best_score = score;
                    best_id = Some(hid);
                }
            }

            (tidx, best_id, best_score)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Common factory functions
// ---------------------------------------------------------------------------

/// Build a `SolverManager` suitable for tests (bounded beam, `min_nodes = 2`).
pub(crate) fn test_solver_manager() -> SolverManager {
    test_solver_manager_with_min_nodes(2)
}

/// Build a `SolverManager` with a custom `min_nodes` threshold.
pub(crate) fn test_solver_manager_with_min_nodes(min_nodes: usize) -> SolverManager {
    SolverManager {
        policy: SolverPolicy::forced(SolverChoice::BoundedBeam),
        bounded_beam_config: BoundedBeamConfig {
            min_nodes,
            ..Default::default()
        },
    }
}

/// Create an `EdgeRankingModelPool` pointing to a dummy ONNX path.
///
/// Tests that use `emit_all_edges = true` never evaluate the model, so this
/// is safe even though the file does not exist.
pub(crate) fn test_edge_models() -> Option<EdgeRankingModelPool> {
    Some(EdgeRankingModelPool::new("unused.onnx"))
}

/// A dummy `InputUri` suitable for pipelines that start with `LoadPersistedData`
/// (no actual alert ingestion).
pub(crate) fn dummy_input_uri() -> InputUri {
    InputUri("file:///dev/null".to_string())
}

// ---------------------------------------------------------------------------
// RuntimeState inspection helpers
// ---------------------------------------------------------------------------

/// Collect all `SeedKey`s present in the seed store (across all nights).
pub(crate) fn collect_seed_keys(state: &RuntimeState) -> HashSet<SeedKey> {
    let mut keys = HashSet::new();
    for (_nid, seeds) in state.seed_store.iter() {
        for seed in seeds {
            keys.insert(seed.key());
        }
    }
    keys
}

/// Collect all `dia_source_id`s present in the alert store.
pub(crate) fn collect_dia_source_ids(state: &RuntimeState) -> HashSet<u64> {
    state
        .alert_store
        .iter()
        .map(|a| a.key.dia_source_id)
        .collect()
}

/// Collect the set of `(from, to)` seed-key pairs from all edges.
pub(crate) fn collect_edge_endpoints(state: &RuntimeState) -> HashSet<(SeedKey, SeedKey)> {
    state.graph.edges.iter().map(|e| (e.from, e.to)).collect()
}

/// Collect night IDs from the alert store (sorted).
pub(crate) fn collect_night_ids(state: &RuntimeState) -> Vec<NightId> {
    state.alert_store.nights_sorted()
}

// ---------------------------------------------------------------------------
// Common pipeline stage sequences
// ---------------------------------------------------------------------------

/// `IngestNights` only.
pub(crate) const INGEST_ONLY: &[PipelineStage] = &[PipelineStage::IngestNights];

/// `IngestNights → BuildSeeds`.
pub(crate) const THROUGH_SEEDS: &[PipelineStage] =
    &[PipelineStage::IngestNights, PipelineStage::BuildSeeds];

/// `IngestNights → BuildSeeds → BuildEdges`.
pub(crate) const THROUGH_EDGES: &[PipelineStage] = &[
    PipelineStage::IngestNights,
    PipelineStage::BuildSeeds,
    PipelineStage::BuildEdges,
];

/// `IngestNights → BuildSeeds → BuildEdges → Solve`.
pub(crate) const THROUGH_SOLVE: &[PipelineStage] = &[
    PipelineStage::IngestNights,
    PipelineStage::BuildSeeds,
    PipelineStage::BuildEdges,
    PipelineStage::Solve,
];

/// `IngestNights → BuildSeeds → BuildEdges → Solve → FitOrbit`.
pub(crate) const THROUGH_ORBIT: &[PipelineStage] = &[
    PipelineStage::IngestNights,
    PipelineStage::BuildSeeds,
    PipelineStage::BuildEdges,
    PipelineStage::Solve,
    PipelineStage::FitOrbit,
];

/// Full persistence round-trip:
/// `LoadPersistedData → Ingest → Seeds → Edges → Solve → FitOrbit → Save`.
pub(crate) const FULL_WITH_PERSISTENCE: &[PipelineStage] = &[
    PipelineStage::LoadPersistedData,
    PipelineStage::IngestNights,
    PipelineStage::BuildSeeds,
    PipelineStage::BuildEdges,
    PipelineStage::Solve,
    PipelineStage::FitOrbit,
    PipelineStage::SavePersistedData,
];

// ---------------------------------------------------------------------------
// Pipeline result container
// ---------------------------------------------------------------------------

/// Output from a test pipeline run.
pub(crate) struct PipelineTestResult {
    pub output: PipelineOutput,
    pub state: RuntimeState,
    pub engine_config: EngineConfig,
}

// ---------------------------------------------------------------------------
// One-shot pipeline runners
// ---------------------------------------------------------------------------

/// Run a pipeline with the default test solver (`BoundedBeam`, `min_nodes = 2`)
/// and no persistence.
///
/// Uses `engine_config_with_edges`, which enables the edge builder in
/// `emit_all_edges` mode.
pub(crate) fn run_pipeline(
    dataset: &SyntheticDataset,
    data_dir: &TempDir,
    storage_dir: &TempDir,
    stages: &[PipelineStage],
    max_gap_nights: u8,
) -> PipelineTestResult {
    let solver_manager = test_solver_manager();
    run_pipeline_with(
        dataset,
        data_dir,
        storage_dir,
        stages,
        max_gap_nights,
        &solver_manager,
        PersistPolicy::None,
    )
}

/// Run a pipeline with a custom solver and persist policy.
///
/// Uses `engine_config_with_edges`, which enables the edge builder in
/// `emit_all_edges` mode.
pub(crate) fn run_pipeline_with(
    dataset: &SyntheticDataset,
    data_dir: &TempDir,
    storage_dir: &TempDir,
    stages: &[PipelineStage],
    max_gap_nights: u8,
    solver_manager: &SolverManager,
    persist: PersistPolicy,
) -> PipelineTestResult {
    let parquet_path = data_dir.path().join("test_alerts.parquet");
    let alerts_uri = dataset.write_parquet(&parquet_path);

    let engine_config = engine_config_with_edges(storage_dir, max_gap_nights);
    let persistence = PersistenceManager::open_or_create(engine_config.storage_path_buf())
        .expect("open persistence");
    let edge_models = test_edge_models();

    let plan = PipelinePlan {
        stages: stages.to_vec(),
        persist,
        inputs: PipelineInputs { alerts_uri },
    };

    let mut runtime_state = RuntimeState::new();
    let runner = PipelineRunner { plan: plan.clone() };
    let hooks = NoopHooks;

    let mut ctx = PipelineContext {
        plan: &plan,
        persistence: &persistence,
        runtime_state: &mut runtime_state,
        engine_config: &engine_config,
        edge_models: &edge_models,
        solver_manager,
    };

    let output = runner
        .run(&mut ctx, &hooks)
        .expect("pipeline should succeed");
    drop(ctx);

    PipelineTestResult {
        output,
        state: runtime_state,
        engine_config,
    }
}

/// Run a pipeline with a minimal engine config (no edge builder setup).
///
/// Uses `SolverManager::default()` and `PersistPolicy::None`.
pub(crate) fn run_pipeline_minimal(
    dataset: &SyntheticDataset,
    data_dir: &TempDir,
    storage_dir: &TempDir,
    stages: &[PipelineStage],
) -> PipelineTestResult {
    let parquet_path = data_dir.path().join("test_alerts.parquet");
    let alerts_uri = dataset.write_parquet(&parquet_path);

    let engine_config = engine_config_minimal(storage_dir);
    let persistence = PersistenceManager::open_or_create(engine_config.storage_path_buf())
        .expect("open persistence");
    let edge_models = test_edge_models();
    let solver_manager = SolverManager::default();

    let plan = PipelinePlan {
        stages: stages.to_vec(),
        persist: PersistPolicy::None,
        inputs: PipelineInputs { alerts_uri },
    };

    let mut runtime_state = RuntimeState::new();
    let runner = PipelineRunner { plan: plan.clone() };
    let hooks = NoopHooks;

    let mut ctx = PipelineContext {
        plan: &plan,
        persistence: &persistence,
        runtime_state: &mut runtime_state,
        engine_config: &engine_config,
        edge_models: &edge_models,
        solver_manager: &solver_manager,
    };

    let output = runner
        .run(&mut ctx, &hooks)
        .expect("pipeline should succeed");
    drop(ctx);

    PipelineTestResult {
        output,
        state: runtime_state,
        engine_config,
    }
}

// ---------------------------------------------------------------------------
// Incremental (night-by-night) pipeline runner
// ---------------------------------------------------------------------------

/// Run the pipeline incrementally — once per unique night in the dataset —
/// preserving `RuntimeState` across iterations.
///
/// `stages_fn(is_last_night)` returns the stage list for each iteration.
/// This allows including extra stages (e.g. `FitOrbit`) only on the last night.
///
/// Returns the pipeline output from the **last** iteration and the final
/// cumulative runtime state.
pub(crate) fn run_incremental_pipeline(
    dataset: &SyntheticDataset,
    data_dir: &TempDir,
    storage_dir: &TempDir,
    max_gap_nights: u8,
    solver_manager: &SolverManager,
    stages_fn: impl Fn(bool) -> Vec<PipelineStage>,
) -> PipelineTestResult {
    let engine_config = engine_config_with_edges(storage_dir, max_gap_nights);
    let persistence = PersistenceManager::open_or_create(engine_config.storage_path_buf())
        .expect("open persistence");
    let edge_models = test_edge_models();

    let mut runtime_state = RuntimeState::new();

    let mut night_ids: Vec<u32> = dataset.alerts().iter().map(|a| a.key.night_id.0).collect();
    night_ids.sort_unstable();
    night_ids.dedup();

    let n_total = night_ids.len();
    let mut last_output = None;

    for (run_idx, &nid) in night_ids.iter().enumerate() {
        let night_alerts: Vec<&Alert> = dataset
            .alerts()
            .iter()
            .filter(|a| a.key.night_id.0 == nid)
            .collect();

        let parquet_path = data_dir
            .path()
            .join(format!("night_{nid}_run{run_idx}.parquet"));
        let alerts_uri = write_alerts_parquet(&night_alerts, &parquet_path);

        let is_last = run_idx + 1 == n_total;
        let stages = stages_fn(is_last);

        let plan = PipelinePlan {
            stages,
            persist: PersistPolicy::None,
            inputs: PipelineInputs { alerts_uri },
        };

        let runner = PipelineRunner { plan: plan.clone() };
        let hooks = NoopHooks;

        let mut ctx = PipelineContext {
            plan: &plan,
            persistence: &persistence,
            runtime_state: &mut runtime_state,
            engine_config: &engine_config,
            edge_models: &edge_models,
            solver_manager,
        };

        let output = runner
            .run(&mut ctx, &hooks)
            .unwrap_or_else(|e| panic!("pipeline run for night {nid} failed: {e}"));

        if is_last {
            last_output = Some(output);
        }
    }

    PipelineTestResult {
        output: last_output.expect("dataset must contain at least one night"),
        state: runtime_state,
        engine_config,
    }
}
