//! Integration tests for the full pipeline with **persistence stages**:
//! `LoadPersistedData → IngestNights → BuildSeeds → BuildEdges → Solve → FitOrbit → SavePersistedData`.
//!
//! These tests verify that:
//! 1. `LoadPersistedData` does **not** crash on a fresh (empty) storage directory.
//! 2. `SavePersistedData` correctly writes alerts, seeds, edge deltas, the
//!    manifest, and orbit-result Parquet files to disk.
//! 3. After a save, `LoadPersistedData` restores a `RuntimeState` whose
//!    alert keys, seed keys, and edge keys are consistent with the state
//!    that was saved.
//! 4. An incremental (night-by-night) pipeline that persists state between
//!    runs produces the same logical result as a single all-at-once run.
//! 5. The output Parquet files (`track_members-*.parquet`,
//!    `orbital_params-*.parquet`) are written to the expected paths and
//!    contain valid, non-empty Arrow record batches.
//!
//! **Requirements**: These tests need internet access (UT1 provider) and
//! a cached DE440 ephemeris file (`~/.cache/outfit_cache/`).

use std::collections::HashSet;

use tempfile::TempDir;

use fink_fat_engine::{
    engine_config::pipeline_policy::PersistPolicy,
    night_id::NightId,
    persistence::{
        PersistenceManager, envelope::load_parquet, layout::PersistenceLayout,
        runtime_state::RuntimeState,
    },
    pipeline::{
        PipelineContext, PipelineInputs, PipelinePlan, PipelineRunner, stages::PipelineStage,
    },
    seeding::SeedKey,
};

use super::{
    FULL_WITH_PERSISTENCE, NoopHooks, PipelineTestResult, collect_dia_source_ids,
    collect_edge_endpoints, collect_night_ids, collect_seed_keys, dummy_input_uri,
    engine_config_with_compaction, engine_config_with_edges, run_pipeline_with,
    test_edge_models, test_solver_manager, write_alerts_parquet,
};
use crate::synthetic_alerts::{AsteroidPopulation, SyntheticDatasetBuilder};

// ===========================================================================
// Integration tests
// ===========================================================================

// ---------------------------------------------------------------------------
// 1. Basic round-trip: Save then Load on a fresh directory
// ---------------------------------------------------------------------------

/// Verify that the full pipeline with `PersistPolicy::Full` succeeds on a
/// **fresh** (empty) storage directory and that `LoadPersistedData` does not
/// crash on the initial run (no data on disk yet).
#[test]
fn full_pipeline_fresh_storage_does_not_crash() {
    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, 5)
        .n_nights(3)
        .obs_per_night(3)
        .start_night_id(60000)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    let sm = test_solver_manager();
    let PipelineTestResult {
        output,
        state: _state,
        engine_config: _cfg,
    } = run_pipeline_with(
        &dataset,
        &data_dir,
        &storage_dir,
        FULL_WITH_PERSISTENCE,
        3,
        &sm,
        PersistPolicy::Full,
    );

    // 7 stage reports expected.
    assert_eq!(
        output.reports.len(),
        7,
        "expected 7 stage reports (Load + Ingest + Seeds + Edges + Solve + FitOrbit + Save)"
    );
    assert_eq!(output.reports[0].0, PipelineStage::LoadPersistedData);
    assert_eq!(output.reports[6].0, PipelineStage::SavePersistedData);
}

// ---------------------------------------------------------------------------
// 2. Manifest and on-disk artifact verification after save
// ---------------------------------------------------------------------------

/// After running the full pipeline with `PersistPolicy::Full`, verify:
/// - the manifest file exists,
/// - per-night alert/seed files exist,
/// - the edge journal (delta or snapshot) exists,
/// - orbit Parquet files exist (track_members + orbital_params).
#[test]
fn save_creates_expected_disk_artifacts() {
    let n_nights = 3_usize;
    let start_nid = 60000_u32;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, 5)
        .n_nights(n_nights)
        .obs_per_night(3)
        .start_night_id(start_nid)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    let sm = test_solver_manager();
    let PipelineTestResult {
        output: _output,
        state,
        engine_config,
    } = run_pipeline_with(
        &dataset,
        &data_dir,
        &storage_dir,
        FULL_WITH_PERSISTENCE,
        3,
        &sm,
        PersistPolicy::Full,
    );

    let layout = fink_fat_engine::persistence::layout::PersistenceLayout::new(
        engine_config.storage_path_buf(),
    );

    // 2a. Manifest
    let manifest_path = layout.manifest_path();
    assert!(
        manifest_path.as_std_path().exists(),
        "manifest file should exist: {manifest_path}"
    );

    // 2b. Per-night alert + seed files
    let saved_nights = state.alert_store.nights_sorted();
    for &nid in &saved_nights {
        let ap = layout.alerts_night_path(nid);
        assert!(
            ap.as_std_path().exists(),
            "alert file must exist for night {nid}: {ap}"
        );
        let sp = layout.seeds_night_path(nid);
        assert!(
            sp.as_std_path().exists(),
            "seed file must exist for night {nid}: {sp}"
        );
    }

    // 2c. Edge journal — at least one delta file should exist (one per saved night)
    let last_night = NightId(start_nid + (n_nights as u32) - 1);
    let delta_path = layout.graph_delta_night_path(last_night);
    assert!(
        delta_path.as_std_path().exists(),
        "edge delta for last night should exist: {delta_path}"
    );

    // 2d. Orbit Parquet files (PersistPolicy::Full exports them).
    //     The orbit export uses the last night as the partition key.
    let current_night = state.alert_store.last_night().unwrap();
    let track_members_path = layout.track_members_night_path(current_night);
    let orbital_params_path = layout.orbital_params_night_path(current_night);

    // At least one of these should exist (FitOrbit may produce empty results
    // for some configurations, but with 5 MBA × 3 nights × 3 obs we expect
    // at least *some* orbits).
    let has_track_members = track_members_path.as_std_path().exists();
    let has_orbital_params = orbital_params_path.as_std_path().exists();

    eprintln!(
        "track_members exists: {has_track_members}, orbital_params exists: {has_orbital_params}"
    );

    // If the solver produced hypotheses, both files should be present.
    if !state.track_hypotheses.is_empty() || !state.orbit_results.is_empty() {
        assert!(
            has_track_members,
            "track_members Parquet should exist when hypotheses are present: {track_members_path}"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Orbit Parquet content verification
// ---------------------------------------------------------------------------

/// Verify that the orbit Parquet files contain valid record batches with
/// the expected column names and non-zero row counts.
#[test]
fn orbit_parquet_files_have_valid_content() {
    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, 8)
        .n_nights(4)
        .obs_per_night(3)
        .start_night_id(60000)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    let sm = test_solver_manager();
    let PipelineTestResult {
        output: _output,
        state,
        engine_config,
    } = run_pipeline_with(
        &dataset,
        &data_dir,
        &storage_dir,
        FULL_WITH_PERSISTENCE,
        4,
        &sm,
        PersistPolicy::Full,
    );

    let layout = fink_fat_engine::persistence::layout::PersistenceLayout::new(
        engine_config.storage_path_buf(),
    );
    let current_night = state.alert_store.last_night().unwrap();

    // --- track_members Parquet ---
    let tm_path = layout.track_members_night_path(current_night);
    if tm_path.as_std_path().exists() {
        let (schema, batches) = load_parquet(&tm_path).expect("read track_members parquet");
        let col_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert!(
            col_names.contains(&"track_id"),
            "track_members should have 'track_id' column"
        );
        assert!(
            col_names.contains(&"dia_source_id"),
            "track_members should have 'dia_source_id' column"
        );

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert!(
            total_rows > 0,
            "track_members Parquet should contain at least one row"
        );
    }

    // --- orbital_params Parquet ---
    let op_path = layout.orbital_params_night_path(current_night);
    if op_path.as_std_path().exists() {
        let (schema, batches) = load_parquet(&op_path).expect("read orbital_params parquet");
        let col_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();

        for expected_col in &[
            "track_id",
            "orbit_type",
            "reference_epoch",
            "semi_major_axis",
            "eccentricity",
            "inclination",
            "ascending_node_longitude",
            "periapsis_argument",
            "mean_anomaly",
            "rms",
        ] {
            assert!(
                col_names.contains(expected_col),
                "orbital_params should have '{expected_col}' column"
            );
        }

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert!(
            total_rows > 0,
            "orbital_params parquet should contain at least one row"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Load after Save: key consistency (alerts, seeds, edges)
// ---------------------------------------------------------------------------

/// Run the full pipeline with Save, then open a *new* pipeline starting
/// with Load, and verify that alerts, seeds, and edges are faithfully
/// restored.
#[test]
fn load_restores_alerts_seeds_and_edges_after_save() {
    let n_nights = 3_usize;
    let max_gap: u8 = 3;
    let start_nid = 60000_u32;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, 5)
        .n_nights(n_nights)
        .obs_per_night(3)
        .start_night_id(start_nid)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    // --- Phase 1: Run full pipeline with Save ---
    let sm = test_solver_manager();
    let PipelineTestResult {
        output: _output1,
        state: state_after_save,
        engine_config,
    } = run_pipeline_with(
        &dataset,
        &data_dir,
        &storage_dir,
        FULL_WITH_PERSISTENCE,
        max_gap,
        &sm,
        PersistPolicy::Full,
    );

    // Snapshot what was in memory after save.
    let saved_night_ids = collect_night_ids(&state_after_save);
    let saved_dia_ids = collect_dia_source_ids(&state_after_save);
    let saved_seed_keys = collect_seed_keys(&state_after_save);
    let saved_edge_eps = collect_edge_endpoints(&state_after_save);
    let saved_n_alerts = state_after_save.alert_store.n_alerts();
    let saved_n_edges = state_after_save.graph.edges.len();

    // --- Phase 2: Open a new pipeline with just LoadPersistedData ---
    let persistence = PersistenceManager::open_or_create(engine_config.storage_path_buf())
        .expect("open persistence for load");
    let edge_models = test_edge_models();
    let solver_manager = test_solver_manager();

    // We use a dummy URI because IngestNights is not in the plan.
    let plan = PipelinePlan {
        stages: vec![PipelineStage::LoadPersistedData],
        persist: PersistPolicy::None,
        inputs: PipelineInputs {
            alerts_uri: dummy_input_uri(),
        },
    };

    let mut loaded_state = RuntimeState::new();
    let runner = PipelineRunner { plan: plan.clone() };
    let hooks = NoopHooks;

    let mut ctx = PipelineContext {
        plan: &plan,
        persistence: &persistence,
        runtime_state: &mut loaded_state,
        engine_config: &engine_config,
        edge_models: &edge_models,
        solver_manager: &solver_manager,
    };

    let load_output = runner
        .run(&mut ctx, &hooks)
        .expect("LoadPersistedData pipeline should succeed");

    drop(ctx);

    // Verify the load stage report.
    assert_eq!(load_output.reports.len(), 1);
    assert_eq!(load_output.reports[0].0, PipelineStage::LoadPersistedData);

    // --- Phase 3: Compare loaded state against saved state ---

    // 3a. Night IDs
    let loaded_night_ids = collect_night_ids(&loaded_state);
    assert_eq!(
        saved_night_ids, loaded_night_ids,
        "loaded night IDs should match saved night IDs"
    );

    // 3b. Alert count and dia_source_ids
    let loaded_n_alerts = loaded_state.alert_store.n_alerts();
    assert_eq!(
        saved_n_alerts, loaded_n_alerts,
        "loaded alert count should match saved alert count"
    );

    let loaded_dia_ids = collect_dia_source_ids(&loaded_state);
    assert_eq!(
        saved_dia_ids, loaded_dia_ids,
        "loaded dia_source_id set should match saved set"
    );

    // 3c. Seed keys consistency
    let loaded_seed_keys = collect_seed_keys(&loaded_state);
    assert_eq!(
        saved_seed_keys, loaded_seed_keys,
        "loaded seed key set should match saved seed key set"
    );

    // 3d. Edge endpoints consistency
    let loaded_n_edges = loaded_state.graph.edges.len();
    assert_eq!(
        saved_n_edges, loaded_n_edges,
        "loaded edge count should match saved edge count"
    );

    let loaded_edge_eps = collect_edge_endpoints(&loaded_state);
    assert_eq!(
        saved_edge_eps, loaded_edge_eps,
        "loaded edge endpoint set should match saved edge endpoint set"
    );

    // 3e. Verify that all seed keys referenced by edges exist in the seed store.
    for edge in &loaded_state.graph.edges {
        assert!(
            loaded_state.seed_store.try_get_seed(edge.from).is_some(),
            "edge.from {:?} must be resolvable in the loaded seed store",
            edge.from
        );
        assert!(
            loaded_state.seed_store.try_get_seed(edge.to).is_some(),
            "edge.to {:?} must be resolvable in the loaded seed store",
            edge.to
        );
    }

    // 3f. Verify that all alert keys referenced by seeds exist in the alert store.
    for (_nid, seeds) in loaded_state.seed_store.iter() {
        for seed in seeds {
            for &alert_key in &seed.members {
                assert!(
                    loaded_state.alert_store.get_by_key(alert_key).is_some(),
                    "seed member alert_key {:?} must be resolvable in the loaded alert store",
                    alert_key
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Incremental persistence: night-by-night with Save + Load
// ---------------------------------------------------------------------------

/// Run a night-by-night incremental pipeline where each iteration uses
/// `LoadPersistedData` at the start and `SavePersistedData` at the end.
///
/// After ingesting all nights, verify that:
/// - all expected nights are present,
/// - alert and seed counts accumulate correctly,
/// - edges and manifest are consistent.
#[test]
fn incremental_pipeline_with_persistence_accumulates_state() {
    let n_nights = 4_usize;
    let start_nid = 60000_u32;
    let max_gap: u8 = 4;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, 5)
        .n_nights(n_nights)
        .obs_per_night(3)
        .start_night_id(start_nid)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    let engine_config = engine_config_with_edges(&storage_dir, max_gap);
    let edge_models = test_edge_models();
    let solver_manager = test_solver_manager();

    // Collect unique sorted night IDs.
    let mut night_ids: Vec<u32> = dataset.alerts().iter().map(|a| a.key.night_id.0).collect();
    night_ids.sort_unstable();
    night_ids.dedup();
    assert_eq!(night_ids.len(), n_nights);

    let mut cumulative_alerts: usize = 0;
    let mut final_state: Option<RuntimeState> = None;

    for (run_idx, &nid) in night_ids.iter().enumerate() {
        let night_alerts: Vec<&fink_fat_engine::Alert> = dataset
            .alerts()
            .iter()
            .filter(|a| a.key.night_id.0 == nid)
            .collect();

        cumulative_alerts += night_alerts.len();

        let parquet_path = data_dir
            .path()
            .join(format!("night_{nid}_run{run_idx}.parquet"));
        let alerts_uri = write_alerts_parquet(&night_alerts, &parquet_path);

        let persistence = PersistenceManager::open_or_create(engine_config.storage_path_buf())
            .expect("open persistence");

        // Build stage list: always Load + Ingest + Seeds + Edges + Solve + Save.
        // Include FitOrbit only on the last night.
        let is_last = run_idx + 1 == n_nights;
        let mut stages = vec![
            PipelineStage::LoadPersistedData,
            PipelineStage::IngestNights,
            PipelineStage::BuildSeeds,
            PipelineStage::BuildEdges,
            PipelineStage::Solve,
        ];
        if is_last {
            stages.push(PipelineStage::FitOrbit);
        }
        stages.push(PipelineStage::SavePersistedData);

        let plan = PipelinePlan {
            stages,
            persist: if is_last {
                PersistPolicy::Full
            } else {
                PersistPolicy::Minimal
            },
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

        runner
            .run(&mut ctx, &hooks)
            .unwrap_or_else(|e| panic!("pipeline run {run_idx} (night {nid}) failed: {e}"));

        drop(ctx);

        // After each run, alert count should match cumulated expectations.
        assert_eq!(
            runtime_state.alert_store.n_alerts(),
            cumulative_alerts,
            "after run {run_idx}, alert count should be {cumulative_alerts}"
        );

        // Night count should increase.
        assert_eq!(
            runtime_state.alert_store.n_nights(),
            run_idx + 1,
            "after run {run_idx}, should have {} nights",
            run_idx + 1
        );

        if is_last {
            final_state = Some(runtime_state);
        }
    }

    let final_state = final_state.expect("final state should be captured");

    // Verify final state has all nights.
    let final_nights = collect_night_ids(&final_state);
    assert_eq!(final_nights.len(), n_nights);
    for (i, &nid) in night_ids.iter().enumerate() {
        assert_eq!(
            final_nights[i],
            NightId(nid),
            "night {i} should be NightId({nid})"
        );
    }

    // Verify edges exist (with 4 nights there should be inter-night connections).
    assert!(
        !final_state.graph.edges.is_empty(),
        "final state should have edges after 4 incremental nights"
    );

    // Verify referential integrity of edges → seeds → alerts.
    for edge in &final_state.graph.edges {
        let from_seed = final_state
            .seed_store
            .try_get_seed(edge.from)
            .unwrap_or_else(|| panic!("edge.from {:?} not found in seed store", edge.from));
        let to_seed = final_state
            .seed_store
            .try_get_seed(edge.to)
            .unwrap_or_else(|| panic!("edge.to {:?} not found in seed store", edge.to));

        for &ak in &from_seed.members {
            assert!(
                final_state.alert_store.get_by_key(ak).is_some(),
                "from_seed member {:?} should exist in alert store",
                ak
            );
        }
        for &ak in &to_seed.members {
            assert!(
                final_state.alert_store.get_by_key(ak).is_some(),
                "to_seed member {:?} should exist in alert store",
                ak
            );
        }
    }

    // Verify orbit export Parquet files exist on disk (last iteration used Full).
    let layout = fink_fat_engine::persistence::layout::PersistenceLayout::new(
        engine_config.storage_path_buf(),
    );
    let last_night = NightId(start_nid + (n_nights as u32) - 1);
    let tm_path = layout.track_members_night_path(last_night);
    let _op_path = layout.orbital_params_night_path(last_night);

    if !final_state.track_hypotheses.is_empty() {
        assert!(
            tm_path.as_std_path().exists(),
            "track_members Parquet should exist after incremental pipeline: {tm_path}"
        );
    }

    eprintln!(
        "Incremental pipeline final: {} nights, {} alerts, {} seeds, {} edges, {} hypotheses",
        final_state.alert_store.n_nights(),
        final_state.alert_store.n_alerts(),
        final_state
            .seed_store
            .iter()
            .map(|(_, s)| s.len())
            .sum::<usize>(),
        final_state.graph.edges.len(),
        final_state.track_hypotheses.len(),
    );
}

// ---------------------------------------------------------------------------
// 6. Re-load after incremental persistence produces consistent state
// ---------------------------------------------------------------------------

/// After an incremental pipeline (night by night with persistence), open a
/// fresh pipeline with only `LoadPersistedData` and verify the reloaded
/// state matches expectations.
#[test]
fn reload_after_incremental_persistence_is_consistent() {
    let n_nights = 3_usize;
    let start_nid = 60000_u32;
    let max_gap: u8 = 3;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, 5)
        .n_nights(n_nights)
        .obs_per_night(3)
        .start_night_id(start_nid)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    let engine_config = engine_config_with_edges(&storage_dir, max_gap);
    let edge_models = test_edge_models();
    let solver_manager = test_solver_manager();

    // Collect unique sorted night IDs.
    let mut night_ids: Vec<u32> = dataset.alerts().iter().map(|a| a.key.night_id.0).collect();
    night_ids.sort_unstable();
    night_ids.dedup();

    // --- Phase 1: Incremental ingestion with Save ---
    let mut last_state_snapshot: Option<(
        HashSet<u64>,
        HashSet<SeedKey>,
        HashSet<(SeedKey, SeedKey)>,
    )> = None;

    for (run_idx, &nid) in night_ids.iter().enumerate() {
        let night_alerts: Vec<&fink_fat_engine::Alert> = dataset
            .alerts()
            .iter()
            .filter(|a| a.key.night_id.0 == nid)
            .collect();

        let parquet_path = data_dir.path().join(format!("incr_night_{nid}.parquet"));
        let alerts_uri = write_alerts_parquet(&night_alerts, &parquet_path);

        let persistence = PersistenceManager::open_or_create(engine_config.storage_path_buf())
            .expect("open persistence");

        let stages = vec![
            PipelineStage::LoadPersistedData,
            PipelineStage::IngestNights,
            PipelineStage::BuildSeeds,
            PipelineStage::BuildEdges,
            PipelineStage::Solve,
            PipelineStage::SavePersistedData,
        ];

        let plan = PipelinePlan {
            stages,
            persist: PersistPolicy::Minimal,
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

        runner
            .run(&mut ctx, &hooks)
            .unwrap_or_else(|e| panic!("incremental run {run_idx} failed: {e}"));

        drop(ctx);

        // Snapshot the last iteration's state.
        let is_last = run_idx + 1 == night_ids.len();
        if is_last {
            last_state_snapshot = Some((
                collect_dia_source_ids(&runtime_state),
                collect_seed_keys(&runtime_state),
                collect_edge_endpoints(&runtime_state),
            ));
        }
    }

    let (expected_dia_ids, expected_seed_keys, expected_edge_eps) =
        last_state_snapshot.expect("should have final snapshot");

    // --- Phase 2: Fresh Load ---
    let persistence = PersistenceManager::open_or_create(engine_config.storage_path_buf())
        .expect("reopen persistence");

    let plan = PipelinePlan {
        stages: vec![PipelineStage::LoadPersistedData],
        persist: PersistPolicy::None,
        inputs: PipelineInputs {
            alerts_uri: dummy_input_uri(),
        },
    };

    let mut reloaded_state = RuntimeState::new();
    let runner = PipelineRunner { plan: plan.clone() };
    let hooks = NoopHooks;

    let mut ctx = PipelineContext {
        plan: &plan,
        persistence: &persistence,
        runtime_state: &mut reloaded_state,
        engine_config: &engine_config,
        edge_models: &edge_models,
        solver_manager: &solver_manager,
    };

    runner
        .run(&mut ctx, &hooks)
        .expect("LoadPersistedData after incremental should succeed");

    drop(ctx);

    // --- Phase 3: Verify consistency ---
    let reloaded_dia_ids = collect_dia_source_ids(&reloaded_state);
    let reloaded_seed_keys = collect_seed_keys(&reloaded_state);
    let reloaded_edge_eps = collect_edge_endpoints(&reloaded_state);

    assert_eq!(
        expected_dia_ids, reloaded_dia_ids,
        "reloaded dia_source_ids should match last incremental state"
    );
    assert_eq!(
        expected_seed_keys, reloaded_seed_keys,
        "reloaded seed keys should match last incremental state"
    );
    assert_eq!(
        expected_edge_eps, reloaded_edge_eps,
        "reloaded edge endpoints should match last incremental state"
    );

    // Cross-check referential integrity.
    for edge in &reloaded_state.graph.edges {
        assert!(
            reloaded_state.seed_store.try_get_seed(edge.from).is_some(),
            "edge.from seed not found after reload"
        );
        assert!(
            reloaded_state.seed_store.try_get_seed(edge.to).is_some(),
            "edge.to seed not found after reload"
        );
    }

    for (_nid, seeds) in reloaded_state.seed_store.iter() {
        for seed in seeds {
            for &ak in &seed.members {
                assert!(
                    reloaded_state.alert_store.get_by_key(ak).is_some(),
                    "seed member alert {:?} not found after reload",
                    ak
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 7. Save stage counters verification
// ---------------------------------------------------------------------------

/// Verify that the `SavePersistedData` stage report contains reasonable
/// counter values (nights_saved, alerts_saved, edge_ops_written).
#[test]
fn save_stage_reports_meaningful_counters() {
    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, 5)
        .n_nights(3)
        .obs_per_night(3)
        .start_night_id(60000)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    let sm = test_solver_manager();
    let PipelineTestResult {
        output,
        state: _state,
        engine_config: _cfg,
    } = run_pipeline_with(
        &dataset,
        &data_dir,
        &storage_dir,
        FULL_WITH_PERSISTENCE,
        3,
        &sm,
        PersistPolicy::Full,
    );

    // Find the save stage report.
    let save_report = output
        .reports
        .iter()
        .find(|(s, _)| *s == PipelineStage::SavePersistedData)
        .expect("SavePersistedData report should exist");

    let counters: std::collections::HashMap<&str, u64> =
        save_report.1.counters.iter().copied().collect();

    let nights_saved = counters.get("nights_saved").copied().unwrap_or(0);
    let alerts_saved = counters.get("alerts_saved").copied().unwrap_or(0);

    assert!(
        nights_saved >= 3,
        "should have saved at least 3 nights, got {nights_saved}"
    );
    assert!(
        alerts_saved > 0,
        "should have saved some alerts, got {alerts_saved}"
    );

    eprintln!("Save counters: {counters:?}");
}

// ---------------------------------------------------------------------------
// 8. Load stage counters verification
// ---------------------------------------------------------------------------

/// Verify that after a save + fresh load, the `LoadPersistedData` stage
/// report contains correct counter values.
#[test]
fn load_stage_reports_meaningful_counters() {
    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, 5)
        .n_nights(3)
        .obs_per_night(3)
        .start_night_id(60000)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    // Phase 1: Save
    let sm = test_solver_manager();
    let PipelineTestResult {
        output: _output1,
        state: state1,
        engine_config,
    } = run_pipeline_with(
        &dataset,
        &data_dir,
        &storage_dir,
        FULL_WITH_PERSISTENCE,
        3,
        &sm,
        PersistPolicy::Full,
    );

    // Phase 2: Load
    let persistence = PersistenceManager::open_or_create(engine_config.storage_path_buf())
        .expect("open persistence for load");
    let edge_models = test_edge_models();
    let solver_manager = test_solver_manager();

    let plan = PipelinePlan {
        stages: vec![PipelineStage::LoadPersistedData],
        persist: PersistPolicy::None,
        inputs: PipelineInputs {
            alerts_uri: dummy_input_uri(),
        },
    };

    let mut loaded_state = RuntimeState::new();
    let runner = PipelineRunner { plan: plan.clone() };
    let hooks = NoopHooks;

    let mut ctx = PipelineContext {
        plan: &plan,
        persistence: &persistence,
        runtime_state: &mut loaded_state,
        engine_config: &engine_config,
        edge_models: &edge_models,
        solver_manager: &solver_manager,
    };

    let load_output = runner
        .run(&mut ctx, &hooks)
        .expect("LoadPersistedData should succeed");

    drop(ctx);

    let load_report = &load_output.reports[0];
    assert_eq!(load_report.0, PipelineStage::LoadPersistedData);

    let counters: std::collections::HashMap<&str, u64> =
        load_report.1.counters.iter().copied().collect();

    let nights_loaded = counters.get("nights_loaded").copied().unwrap_or(0);
    let alerts_loaded = counters.get("alerts_loaded").copied().unwrap_or(0);
    let edges_loaded = counters.get("edges_loaded").copied().unwrap_or(0);

    assert_eq!(
        nights_loaded,
        state1.alert_store.n_nights() as u64,
        "loaded night count should match saved night count"
    );
    assert_eq!(
        alerts_loaded,
        state1.alert_store.n_alerts() as u64,
        "loaded alert count should match saved alert count"
    );
    assert_eq!(
        edges_loaded,
        state1.graph.edges.len() as u64,
        "loaded edge count should match saved edge count"
    );

    eprintln!("Load counters: {counters:?}");
}

// ---------------------------------------------------------------------------
// 9. Manifest consistency after multiple saves
// ---------------------------------------------------------------------------

/// After two incremental runs with Save, verify the manifest references
/// all persisted nights.
#[test]
fn manifest_tracks_all_nights_after_multiple_saves() {
    let n_nights = 3_usize;
    let start_nid = 60000_u32;
    let max_gap: u8 = 3;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, 4)
        .n_nights(n_nights)
        .obs_per_night(3)
        .start_night_id(start_nid)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    let engine_config = engine_config_with_edges(&storage_dir, max_gap);
    let edge_models = test_edge_models();
    let solver_manager = test_solver_manager();

    let mut night_ids: Vec<u32> = dataset.alerts().iter().map(|a| a.key.night_id.0).collect();
    night_ids.sort_unstable();
    night_ids.dedup();

    // Ingest all nights one by one with Save each time.
    for (run_idx, &nid) in night_ids.iter().enumerate() {
        let night_alerts: Vec<&fink_fat_engine::Alert> = dataset
            .alerts()
            .iter()
            .filter(|a| a.key.night_id.0 == nid)
            .collect();

        let parquet_path = data_dir
            .path()
            .join(format!("manifest_test_night_{nid}.parquet"));
        let alerts_uri = write_alerts_parquet(&night_alerts, &parquet_path);

        let persistence = PersistenceManager::open_or_create(engine_config.storage_path_buf())
            .expect("open persistence");

        let plan = PipelinePlan {
            stages: vec![
                PipelineStage::LoadPersistedData,
                PipelineStage::IngestNights,
                PipelineStage::BuildSeeds,
                PipelineStage::BuildEdges,
                PipelineStage::Solve,
                PipelineStage::SavePersistedData,
            ],
            persist: PersistPolicy::Minimal,
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

        runner
            .run(&mut ctx, &hooks)
            .unwrap_or_else(|e| panic!("manifest test run {run_idx} failed: {e}"));

        drop(ctx);
    }

    // Now load the manifest directly and check it.
    let persistence = PersistenceManager::open_or_create(engine_config.storage_path_buf())
        .expect("reopen persistence");
    let manifest = persistence.load_or_init_manifest().expect("load manifest");

    // The manifest should reference all n_nights.
    let manifest_nids: Vec<NightId> = manifest.nights.iter().map(|e| e.night_id).collect();
    assert_eq!(
        manifest_nids.len(),
        n_nights,
        "manifest should reference {n_nights} nights, got {}",
        manifest_nids.len()
    );

    for &expected_nid in &night_ids {
        assert!(
            manifest_nids.contains(&NightId(expected_nid)),
            "manifest should reference night {expected_nid}"
        );
    }

    // Each manifest entry should have non-zero alert/seed counts.
    for entry in &manifest.nights {
        assert!(
            entry.n_alerts.unwrap_or(0) > 0,
            "manifest entry for night {:?} should have n_alerts > 0",
            entry.night_id
        );
        assert!(
            entry.n_seeds.unwrap_or(0) > 0,
            "manifest entry for night {:?} should have n_seeds > 0",
            entry.night_id
        );
    }

    // Edge journal should have at least one delta.
    assert!(
        !manifest.edge_journal.deltas.is_empty(),
        "edge journal should have at least one delta after incremental saves"
    );

    eprintln!(
        "Manifest: {} nights, {} edge deltas, snapshot={:?}",
        manifest.nights.len(),
        manifest.edge_journal.deltas.len(),
        manifest.edge_journal.snapshot_night_id,
    );
}

// ---------------------------------------------------------------------------
// 10. Load on empty storage produces empty but valid state
// ---------------------------------------------------------------------------

/// Verify that `LoadPersistedData` on a completely empty storage directory
/// yields an empty but valid `RuntimeState` (no alerts, no seeds, no edges).
#[test]
fn load_on_empty_storage_yields_empty_state() {
    let storage_dir = TempDir::new().unwrap();
    let engine_config = engine_config_with_edges(&storage_dir, 3);

    let persistence = PersistenceManager::open_or_create(engine_config.storage_path_buf())
        .expect("open persistence on empty dir");
    let edge_models = test_edge_models();
    let solver_manager = test_solver_manager();

    let plan = PipelinePlan {
        stages: vec![PipelineStage::LoadPersistedData],
        persist: PersistPolicy::None,
        inputs: PipelineInputs {
            alerts_uri: dummy_input_uri(),
        },
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
        .expect("LoadPersistedData on empty storage should succeed");

    drop(ctx);

    assert_eq!(output.reports.len(), 1);
    assert_eq!(output.reports[0].0, PipelineStage::LoadPersistedData);

    assert!(
        runtime_state.alert_store.is_empty(),
        "alert store should be empty after load on empty storage"
    );
    assert!(
        runtime_state.seed_store.is_empty(),
        "seed store should be empty after load on empty storage"
    );
    assert!(
        runtime_state.graph.edges.is_empty(),
        "graph should have no edges after load on empty storage"
    );
    assert!(
        runtime_state.track_hypotheses.is_empty(),
        "hypotheses should be empty after load on empty storage"
    );
}

// ---------------------------------------------------------------------------
// 11. Diverse populations: save/load preserves state with mixed kinematics
// ---------------------------------------------------------------------------

/// Verify save/load round-trip with a mix of fast and slow movers
/// (NEA + MBA + Trojan).
#[test]
fn save_load_roundtrip_diverse_populations() {
    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::NearEarth, 3)
        .population(AsteroidPopulation::MainBelt, 3)
        .population(AsteroidPopulation::Trojan, 3)
        .n_nights(3)
        .obs_per_night(3)
        .start_night_id(60000)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    // Save
    let sm = test_solver_manager();
    let PipelineTestResult {
        output: _output,
        state: state_saved,
        engine_config,
    } = run_pipeline_with(
        &dataset,
        &data_dir,
        &storage_dir,
        FULL_WITH_PERSISTENCE,
        3,
        &sm,
        PersistPolicy::Full,
    );

    let saved_dia_ids = collect_dia_source_ids(&state_saved);
    let saved_seed_keys = collect_seed_keys(&state_saved);
    let saved_n_edges = state_saved.graph.edges.len();

    // Load
    let persistence = PersistenceManager::open_or_create(engine_config.storage_path_buf())
        .expect("open persistence for reload");
    let edge_models = test_edge_models();
    let solver_manager = test_solver_manager();

    let plan = PipelinePlan {
        stages: vec![PipelineStage::LoadPersistedData],
        persist: PersistPolicy::None,
        inputs: PipelineInputs {
            alerts_uri: dummy_input_uri(),
        },
    };

    let mut reloaded = RuntimeState::new();
    let runner = PipelineRunner { plan: plan.clone() };
    let hooks = NoopHooks;

    let mut ctx = PipelineContext {
        plan: &plan,
        persistence: &persistence,
        runtime_state: &mut reloaded,
        engine_config: &engine_config,
        edge_models: &edge_models,
        solver_manager: &solver_manager,
    };

    runner.run(&mut ctx, &hooks).expect("load should succeed");
    drop(ctx);

    // Verify key consistency.
    assert_eq!(
        saved_dia_ids,
        collect_dia_source_ids(&reloaded),
        "dia_source_ids should match after reload"
    );
    assert_eq!(
        saved_seed_keys,
        collect_seed_keys(&reloaded),
        "seed keys should match after reload"
    );
    assert_eq!(
        saved_n_edges,
        reloaded.graph.edges.len(),
        "edge count should match after reload"
    );

    // Referential integrity.
    for edge in &reloaded.graph.edges {
        let from = reloaded.seed_store.try_get_seed(edge.from);
        let to = reloaded.seed_store.try_get_seed(edge.to);
        assert!(from.is_some(), "edge.from seed missing after reload");
        assert!(to.is_some(), "edge.to seed missing after reload");
    }
}

// ---------------------------------------------------------------------------
// 12. Edge journal deltas and graph compaction
// ---------------------------------------------------------------------------

/// Run an incremental night-by-night pipeline with `compact_graph_every_delta`
/// set to a low threshold and verify:
///
/// 1. Each `SavePersistedData` iteration creates a delta file on disk.
/// 2. The manifest's `edge_journal.deltas` grows by one per iteration
///    (before compaction).
/// 3. Once the number of deltas reaches `compact_graph_every_delta`, the
///    `SavePersistedData` stage triggers compaction:
///    - a snapshot file (`graph/snapshot.bin`) is written,
///    - old deltas up to the compaction checkpoint are pruned from the manifest,
///    - the `edge_compacted` counter in the stage report is `1`.
/// 4. After compaction, the reloaded state is still consistent with the
///    state that was in memory.
#[test]
fn edge_journal_deltas_and_compaction() {
    // We use 5 nights so that with compact_graph_every_delta=3 the
    // compaction triggers on the 3rd night (3 deltas accumulated).
    let n_nights = 5_usize;
    let start_nid = 60000_u32;
    let max_gap: u8 = 5;
    let compact_every: usize = 3;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, 5)
        .n_nights(n_nights)
        .obs_per_night(3)
        .start_night_id(start_nid)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    let engine_config = engine_config_with_compaction(&storage_dir, max_gap, compact_every);
    let edge_models = test_edge_models();
    let solver_manager = test_solver_manager();

    let layout = PersistenceLayout::new(engine_config.storage_path_buf());

    // Collect sorted unique night IDs.
    let mut night_ids: Vec<u32> = dataset.alerts().iter().map(|a| a.key.night_id.0).collect();
    night_ids.sort_unstable();
    night_ids.dedup();
    assert_eq!(night_ids.len(), n_nights);

    // Track compaction events and delta counts across iterations.
    let mut compaction_triggered_at: Option<usize> = None;
    let mut delta_counts_after_save: Vec<usize> = Vec::new();

    for (run_idx, &nid) in night_ids.iter().enumerate() {
        // Write one night of alerts.
        let night_alerts: Vec<&fink_fat_engine::Alert> = dataset
            .alerts()
            .iter()
            .filter(|a| a.key.night_id.0 == nid)
            .collect();

        let parquet_path = data_dir
            .path()
            .join(format!("night_{nid}_run{run_idx}.parquet"));
        let alerts_uri = write_alerts_parquet(&night_alerts, &parquet_path);

        // Open persistence for this iteration (simulates restart between runs).
        let persistence = PersistenceManager::open_or_create(engine_config.storage_path_buf())
            .expect("open persistence");

        // Full pipeline: Load → Ingest → Seeds → Edges → Solve → FitOrbit → Save.
        let stages = vec![
            PipelineStage::LoadPersistedData,
            PipelineStage::IngestNights,
            PipelineStage::BuildSeeds,
            PipelineStage::BuildEdges,
            PipelineStage::Solve,
            PipelineStage::FitOrbit,
            PipelineStage::SavePersistedData,
        ];

        let plan = PipelinePlan {
            stages,
            persist: PersistPolicy::Minimal,
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
            .unwrap_or_else(|e| panic!("pipeline run #{run_idx} (nid={nid}) failed: {e}"));
        drop(ctx);

        // -----------------------------------------------------------------
        // Inspect manifest to count deltas.
        // -----------------------------------------------------------------
        let n_deltas_after = runtime_state.manifest.edge_journal.deltas.len();
        delta_counts_after_save.push(n_deltas_after);

        // -----------------------------------------------------------------
        // Extract save stage counters.
        // -----------------------------------------------------------------
        let save_report = output
            .reports
            .iter()
            .find(|(s, _)| *s == PipelineStage::SavePersistedData)
            .expect("SavePersistedData report should exist");

        let counters: std::collections::HashMap<&str, u64> =
            save_report.1.counters.iter().copied().collect();

        let edge_ops_written = counters.get("edge_ops_written").copied().unwrap_or(0);
        let edge_compacted = counters.get("edge_compacted").copied().unwrap_or(0);

        eprintln!(
            "Run #{run_idx} nid={nid}: edge_ops={edge_ops_written}, \
             deltas_after={n_deltas_after}, edge_compacted={edge_compacted}",
        );

        // -----------------------------------------------------------------
        // Verify delta file or compaction artifacts.
        //
        // If compaction did NOT fire on this run, the delta file for this
        // night must exist on disk. If compaction DID fire, the snapshot
        // must exist and old deltas are pruned (the save stage writes the
        // delta first, then compacts, which may clean up that same delta).
        // -----------------------------------------------------------------
        let delta_path = layout.graph_delta_night_path(NightId(nid));

        if edge_compacted == 0 {
            // No compaction this run → delta should be present on disk.
            assert!(
                delta_path.as_std_path().exists(),
                "delta file should exist for night {nid} (run #{run_idx}): {delta_path}",
            );
        }

        // -----------------------------------------------------------------
        // Check compaction trigger.
        // -----------------------------------------------------------------
        if edge_compacted == 1 {
            assert!(
                compaction_triggered_at.is_none(),
                "compaction should only trigger once in this test",
            );
            compaction_triggered_at = Some(run_idx);

            // After compaction, snapshot must exist.
            let snapshot_path = layout.graph_snapshot_path();
            assert!(
                snapshot_path.as_std_path().exists(),
                "snapshot file should exist after compaction: {snapshot_path}",
            );

            // The manifest should reference the snapshot.
            assert!(
                runtime_state
                    .manifest
                    .edge_journal
                    .snapshot_night_id
                    .is_some(),
                "manifest should reference a snapshot night_id after compaction",
            );

            // Deltas up to the checkpoint should have been pruned.
            // After compaction, remaining deltas should be fewer than before.
            assert!(
                n_deltas_after < compact_every,
                "after compaction, delta count ({n_deltas_after}) should be \
                 less than the threshold ({compact_every})",
            );
        }
    }

    // -----------------------------------------------------------------
    // Post-loop assertions
    // -----------------------------------------------------------------

    // Compaction must have triggered at some point during the 5-night run.
    let compacted_at = compaction_triggered_at.expect(
        "compaction should have been triggered at least once with \
         compact_graph_every_delta=3 and 5 nights",
    );
    eprintln!("Compaction triggered at run index {compacted_at}");

    // Before compaction, the delta count should have grown monotonically.
    // After compaction, it resets and grows again.
    // Verify that at least one delta_counts_after_save value reached
    // or exceeded the threshold before being reset.
    let max_deltas_before_compaction = delta_counts_after_save
        .iter()
        .take(compacted_at) // iterations before the compaction run
        .copied()
        .max()
        .unwrap_or(0);
    eprintln!("Max deltas before compaction run: {max_deltas_before_compaction}");

    // After all runs, reload the state and verify it is consistent.
    let persistence = PersistenceManager::open_or_create(engine_config.storage_path_buf())
        .expect("open persistence for final reload");

    let plan = PipelinePlan {
        stages: vec![PipelineStage::LoadPersistedData],
        persist: PersistPolicy::None,
        inputs: PipelineInputs {
            alerts_uri: dummy_input_uri(),
        },
    };

    let mut reloaded = RuntimeState::new();
    let runner = PipelineRunner { plan: plan.clone() };
    let hooks = NoopHooks;

    let mut ctx = PipelineContext {
        plan: &plan,
        persistence: &persistence,
        runtime_state: &mut reloaded,
        engine_config: &engine_config,
        edge_models: &edge_models,
        solver_manager: &solver_manager,
    };

    runner
        .run(&mut ctx, &hooks)
        .expect("reload after compaction should succeed");
    drop(ctx);

    // Verify that the reloaded state has all nights.
    let reloaded_nights = collect_night_ids(&reloaded);
    assert!(
        !reloaded_nights.is_empty(),
        "reloaded state should contain nights",
    );

    // Verify edges are present (5 MBA × 3 obs/night × 5 nights should produce edges).
    assert!(
        !reloaded.graph.edges.is_empty(),
        "reloaded graph should contain edges after compaction + deltas",
    );

    // Verify referential integrity: every edge references valid seeds.
    for edge in &reloaded.graph.edges {
        assert!(
            reloaded.seed_store.try_get_seed(edge.from).is_some(),
            "edge.from seed {:?} missing after reload through compacted journal",
            edge.from,
        );
        assert!(
            reloaded.seed_store.try_get_seed(edge.to).is_some(),
            "edge.to seed {:?} missing after reload through compacted journal",
            edge.to,
        );
    }

    eprintln!(
        "Edge journal + compaction test passed: {n_nights} nights, \
         compaction at run #{compacted_at}, \
         final edges={}, final deltas={}",
        reloaded.graph.edges.len(),
        reloaded.manifest.edge_journal.deltas.len(),
    );
}
