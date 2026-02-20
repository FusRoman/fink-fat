//! Integration test for the `IngestNights` pipeline stage.
//!
//! This test exercises the full `PipelineRunner::run` path with a single
//! `PipelineStage::IngestNights` stage, loading synthetic alerts from a
//! Parquet file generated via the `synthetic_alerts` module.

use tempfile::TempDir;

use fink_fat_engine::{
    error::EngineError,
    night_id::{NightId, PairingMode},
    persistence::PersistenceManager,
    pipeline::{
        PersistPolicy, PipelineContext, PipelineInputs, PipelinePlan, PipelineRunner,
        stages::{PipelineStage, alert_inputs::input_uri::InputUri},
    },
};

use crate::synthetic_alerts::{AsteroidPopulation, SyntheticDatasetBuilder};
use super::{
    NoopHooks, PipelineTestResult, INGEST_ONLY,
    engine_config_minimal, new_runtime_state, run_pipeline_minimal, test_edge_models,
};

// ---------------------------------------------------------------------------
// Integration tests
// ---------------------------------------------------------------------------

#[test]
fn ingest_nights_stage_loads_alerts_and_populates_runtime_state() {
    // ---- 1) Generate synthetic dataset and write to Parquet ----
    let n_trajectories = 3;
    let n_nights = 3;
    let obs_per_night = 2;
    let start_night_id = 60000_u32;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, n_trajectories)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(42)
        .build();

    let expected_total_alerts = n_trajectories * n_nights * obs_per_night;
    assert_eq!(dataset.n_alerts(), expected_total_alerts);

    let data_dir = TempDir::new().expect("create data temp dir");
    let storage_dir = TempDir::new().expect("create storage temp dir");

    // ---- 2) Run the pipeline ----
    let PipelineTestResult { output, state, .. } =
        run_pipeline_minimal(&dataset, &data_dir, &storage_dir, INGEST_ONLY);

    // ---- 3) Verify pipeline output ----
    assert_eq!(output.reports.len(), 1, "exactly one stage report expected");
    let (stage, report) = &output.reports[0];
    assert_eq!(*stage, PipelineStage::IngestNights);

    let counters: std::collections::HashMap<&str, u64> =
        report.counters.iter().copied().collect();
    assert_eq!(
        counters.get("n_alerts").copied(),
        Some(expected_total_alerts as u64),
        "expected {expected_total_alerts} alerts total"
    );
    assert_eq!(
        counters.get("n_nights").copied(),
        Some(n_nights as u64),
        "expected {n_nights} distinct nights"
    );

    // ---- 4) Verify runtime state: alert store ----
    let store = &state.alert_store;
    assert_eq!(
        store.n_alerts(),
        expected_total_alerts,
        "alert store should hold all alerts"
    );
    assert_eq!(
        store.n_nights(),
        n_nights,
        "alert store should hold the correct number of nights"
    );

    // Each night should have (n_trajectories × obs_per_night) alerts.
    let expected_per_night = n_trajectories * obs_per_night;
    for night_offset in 0..n_nights {
        let nid = NightId(start_night_id + night_offset as u32);
        let night_alerts = store
            .get(&nid)
            .unwrap_or_else(|| panic!("night {:?} should be present", nid));
        assert_eq!(
            night_alerts.len(),
            expected_per_night,
            "night {nid:?} should have {expected_per_night} alerts"
        );

        // ---- 5) Verify alerts are time-ordered within each night ----
        for w in night_alerts.windows(2) {
            assert!(
                w[0].mjd_tt <= w[1].mjd_tt,
                "alerts should be time-ordered within night {nid:?}"
            );
        }

        // ---- 6) Verify alert keys are consistent ----
        for alert in night_alerts {
            assert_eq!(
                alert.key.night_id, nid,
                "alert key night_id must match the night"
            );
        }
    }

    // ---- 7) Verify runtime window was set ----
    assert!(
        state.window.is_some(),
        "runtime window should be set after IngestNights"
    );

    let last_night_id = NightId(start_night_id + (n_nights as u32) - 1);
    match state.window.unwrap() {
        PairingMode::SingleNight { anchor, .. } => {
            assert_eq!(
                anchor, last_night_id,
                "anchor night should be the last ingested night"
            );
        }
        other => panic!("expected SingleNight pairing mode, got: {:?}", other),
    }

    // ---- 8) Verify all dia_source_ids are unique ----
    let all_dia_ids: Vec<u64> = store
        .nights()
        .flat_map(|nid| {
            store
                .get(nid)
                .unwrap()
                .iter()
                .map(|a| a.key.dia_source_id)
        })
        .collect();
    assert_eq!(all_dia_ids.len(), expected_total_alerts);

    let mut sorted_ids = all_dia_ids.clone();
    sorted_ids.sort_unstable();
    sorted_ids.dedup();
    assert_eq!(
        sorted_ids.len(),
        expected_total_alerts,
        "all dia_source_ids must be unique"
    );
}

#[test]
fn ingest_nights_stage_fails_on_missing_parquet_file() {
    let storage_dir = TempDir::new().expect("create storage temp dir");
    let engine_config = engine_config_minimal(&storage_dir);
    let persistence =
        PersistenceManager::open_or_create(engine_config.storage_path_buf())
            .expect("open persistence");

    let edge_models = test_edge_models();
    let solver_manager = fink_fat_engine::solver::solver_manager::SolverManager::default();

    let plan = PipelinePlan {
        window: None,
        stages: vec![PipelineStage::IngestNights],
        persist: PersistPolicy::None,
        inputs: PipelineInputs {
            alerts_uri: InputUri("file:///nonexistent/path/alerts.parquet".to_string()),
        },
    };

    let mut runtime_state = new_runtime_state();

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

    let result = runner.run(&mut ctx, &hooks);

    assert!(result.is_err(), "pipeline should fail for missing file");
    match result.err().unwrap() {
        EngineError::StageFailed { stage, message } => {
            assert_eq!(stage, PipelineStage::IngestNights);
            assert!(
                !message.is_empty(),
                "error message should describe the failure"
            );
        }
        other => panic!("expected StageFailed, got: {other:?}"),
    }
}
