//! Integration test suite for the fink-fat-engine pipeline stages.
//!
//! This module contains shared test infrastructure (helpers, configs, hooks)
//! used by all pipeline integration tests, plus the test submodules themselves.

mod build_edges_test;
mod build_seeds_test;
mod fit_orbit_test;
mod ingest_alerts_test;
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
    ArrayRef, Float64Array, RecordBatch, StringArray, UInt32Array, UInt64Array, UInt8Array,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::parquet::arrow::ArrowWriter;
use outfit::FullOrbitResult;
use tempfile::TempDir;

use fink_fat_engine::{
    Alert, AlertStore,
    engine_config::EngineConfig,
    graph::AlertLinkageDAG,
    persistence::{manifest::Manifest, runtime_state::RuntimeState},
    pipeline::{
        hooks::{PipelineHooks, StageMeta, StageReport},
        stages::{PipelineStage, alert_inputs::input_uri::InputUri},
    },
    seeding::store::SeedStore,
    solver::HypothesisSet,
    trajectory::TrackHypothesis,
};

use crate::synthetic_alerts::TrajectoryTruth;

// ---------------------------------------------------------------------------
// No-op pipeline hooks
// ---------------------------------------------------------------------------

/// A no-op implementation of [`PipelineHooks`] for tests that don't need
/// progress reporting.
pub(crate) struct NoopHooks;

impl PipelineHooks for NoopHooks {
    fn on_stage_start(&self, _stage: PipelineStage, _meta: StageMeta) {}
    fn on_stage_progress(&self, _stage: PipelineStage, _delta: u64) {}
    fn on_stage_end(&self, _stage: PipelineStage, _report: StageReport) {}
}

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
pub(crate) fn engine_config_with_edges(
    storage_dir: &TempDir,
    max_gap_nights: u8,
) -> EngineConfig {
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

// ---------------------------------------------------------------------------
// RuntimeState factory
// ---------------------------------------------------------------------------

/// Create a fresh, empty [`RuntimeState`] with default stores.
pub(crate) fn new_runtime_state() -> RuntimeState {
    RuntimeState {
        manifest: Manifest::new(0),
        window: None,
        alert_store: AlertStore::new(),
        seed_store: SeedStore::new(),
        graph: AlertLinkageDAG::new(),
        track_hypotheses: HypothesisSet::new(),
        orbit_results: FullOrbitResult::default(),
    }
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
