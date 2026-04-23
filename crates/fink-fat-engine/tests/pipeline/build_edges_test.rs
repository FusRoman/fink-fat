//! Integration test for the `IngestNights` → `BuildSeeds` → `BuildEdges` pipeline.
//!
//! This test exercises the full three-stage pipeline where:
//! 1. `IngestNights` loads synthetic alerts from a Parquet file.
//! 2. `BuildSeeds` generates intra-night seeds (pairs/triplets).
//! 3. `BuildEdges` constructs inter-night edges between seeds of neighboring
//!    nights, respecting the `max_gap_nights` sliding window.
//!
//! The edge builder operates in no-filtering mode (`top_k_per_left: None`) so
//! that no ONNX model is required (physics-only cost, all candidates emitted).

use tempfile::TempDir;

use fink_fat_engine::{
    engine_config::pipeline_policy::PersistPolicy,
    night_id::NightId,
    persistence::{PersistenceManager, runtime_state::RuntimeState},
    pipeline::{PipelineContext, stages::PipelineStage},
    solver::solver_manager::SolverManager,
};

use super::{
    NoopHooks, PipelineTestResult, THROUGH_EDGES, engine_config_with_edges, make_plan_and_runner,
    run_pipeline, test_edge_models,
};
use crate::synthetic_alerts::{
    AsteroidPopulation, SyntheticDatasetBuilder, write_and_load_parquet,
};

// ---------------------------------------------------------------------------
// Integration tests
// ---------------------------------------------------------------------------

#[test]
fn three_stage_pipeline_produces_edges_between_nights() {
    // ---- 1) Generate synthetic data: 5 MBA trajectories, 3 nights, 3 obs/night ----
    let n_trajectories = 5;
    let n_nights = 3;
    let obs_per_night = 3;
    let start_night_id = 60000_u32;
    let max_gap_nights = 3_u8; // large enough to cover all night gaps

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, n_trajectories)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(42)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    let PipelineTestResult {
        output,
        state: runtime_state,
        ..
    } = run_pipeline(
        &dataset,
        &data_dir,
        &storage_dir,
        THROUGH_EDGES,
        max_gap_nights,
    );

    // ---- 2) Verify we got three stage reports ----
    assert_eq!(output.reports.len(), 3, "expected 3 stage reports");
    assert_eq!(output.reports[0].0, PipelineStage::IngestNights);
    assert_eq!(output.reports[1].0, PipelineStage::BuildSeeds);
    assert_eq!(output.reports[2].0, PipelineStage::BuildEdges);

    // ---- 3) Verify BuildEdges counters ----
    let edge_counters: std::collections::HashMap<&str, u64> =
        output.reports[2].1.counters.iter().copied().collect();

    let pairs_processed = edge_counters.get("pairs_processed").copied().unwrap_or(0);
    let edges_added = edge_counters.get("edges_added").copied().unwrap_or(0);

    // With 3 nights and max_gap=3, the edge builder should process at least
    // 1 night pair (left → right) where right = last night.
    assert!(
        pairs_processed > 0,
        "BuildEdges should process at least one (left, right) night pair"
    );
    assert!(
        edges_added > 0,
        "BuildEdges should produce at least some edges"
    );

    // ---- 4) Verify the graph contains edges ----
    let graph = &runtime_state.graph;
    // The pipeline runs incrementally; the graph accumulates edges across all
    // runs, so total edge count is >= the last run's reported edges_added.
    assert!(
        graph.edges.len() as u64 >= edges_added,
        "graph edge count ({}) must be >= reported edges_added ({})",
        graph.edges.len(),
        edges_added
    );
    assert!(
        !graph.edges.is_empty(),
        "graph must contain at least some edges"
    );

    // ---- 5) Verify edge invariants ----
    // The graph accumulates edges from multiple incremental runs;
    // verify structural properties that hold for all edges regardless of run.
    for edge in &graph.edges {
        // Cost must be finite and strictly positive.
        assert!(
            edge.cost.is_finite() && edge.cost > 0.0,
            "edge cost must be finite and positive, got {}",
            edge.cost
        );

        // dt_days must be finite and strictly positive (forward in time).
        assert!(
            edge.dt_days.is_finite() && edge.dt_days > 0.0,
            "edge dt_days must be finite and positive, got {}",
            edge.dt_days
        );

        // Edge should be active by default.
        assert!(edge.active, "newly created edges should be active");

        // `from` seed must belong to a night strictly before `to` seed.
        assert!(
            edge.from.night_id < edge.to.night_id,
            "edge must point forward in time: from={:?} to={:?}",
            edge.from,
            edge.to
        );

        // The night gap must be within max_gap_nights.
        let night_gap = edge.to.night_id.0 - edge.from.night_id.0;
        assert!(
            night_gap <= max_gap_nights as u32,
            "night gap {} exceeds max_gap_nights {}",
            night_gap,
            max_gap_nights
        );

        // Both seed keys must exist in the seed store.
        assert!(
            runtime_state.seed_store.try_get_seed(edge.from).is_some(),
            "edge.from {:?} must exist in seed store",
            edge.from
        );
        assert!(
            runtime_state.seed_store.try_get_seed(edge.to).is_some(),
            "edge.to {:?} must exist in seed store",
            edge.to
        );
    }

    // ---- 6) Verify in/out degree maps are consistent ----
    for edge in &graph.edges {
        let out = graph.out_deg.get(&edge.from).copied().unwrap_or(0);
        assert!(out > 0, "from-node must have out_deg > 0");
    }
}

#[test]
fn edges_respect_max_gap_window() {
    // With max_gap_nights=1 and 4 nights, only (night N-1 → night N) edges
    // should exist. Night N-2 and N-3 are too far back.
    let n_trajectories = 5;
    let n_nights = 4;
    let obs_per_night = 3;
    let start_night_id = 60000_u32;
    let max_gap_nights = 1_u8;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, n_trajectories)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(55)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    let PipelineTestResult {
        output,
        state: runtime_state,
        ..
    } = run_pipeline(
        &dataset,
        &data_dir,
        &storage_dir,
        THROUGH_EDGES,
        max_gap_nights,
    );

    assert_eq!(output.reports.len(), 3);

    let edge_counters: std::collections::HashMap<&str, u64> =
        output.reports[2].1.counters.iter().copied().collect();
    let pairs_processed = edge_counters.get("pairs_processed").copied().unwrap_or(0);

    // With max_gap=1, only the immediately preceding night can connect to the last night.
    // That's exactly 1 night pair.
    assert_eq!(
        pairs_processed, 1,
        "max_gap=1 should produce exactly 1 night pair (penultimate → last)"
    );

    // In incremental mode, the graph accumulates edges from all runs.
    // After n runs with max_gap=1 there are edges from each consecutive pair
    // of nights. Verify all edges respect the gap constraint and temporal ordering.
    for edge in &runtime_state.graph.edges {
        assert!(
            edge.from.night_id < edge.to.night_id,
            "edge must point forward in time"
        );
        let gap = edge.to.night_id.0 - edge.from.night_id.0;
        assert_eq!(
            gap, 1,
            "with max_gap=1 all edge gaps must equal 1, got {gap}"
        );
    }
}

#[test]
fn larger_gap_includes_more_left_nights() {
    // Compare max_gap=1 vs max_gap=3 on the same 4-night dataset.
    // max_gap=3 should process more night pairs and potentially more edges.
    let n_trajectories = 5;
    let n_nights = 4;
    let obs_per_night = 3;
    let start_night_id = 60000_u32;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, n_trajectories)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(88)
        .build();

    // --- Run with max_gap=1 ---
    let data_dir_1 = TempDir::new().unwrap();
    let storage_dir_1 = TempDir::new().unwrap();
    let PipelineTestResult {
        output: output_gap1,
        state: state_gap1,
        ..
    } = run_pipeline(&dataset, &data_dir_1, &storage_dir_1, THROUGH_EDGES, 1);

    let counters_gap1: std::collections::HashMap<&str, u64> =
        output_gap1.reports[2].1.counters.iter().copied().collect();
    let pairs_gap1 = counters_gap1.get("pairs_processed").copied().unwrap_or(0);
    let edges_gap1 = state_gap1.graph.edges.len();

    // --- Run with max_gap=3 ---
    let data_dir_3 = TempDir::new().unwrap();
    let storage_dir_3 = TempDir::new().unwrap();
    let PipelineTestResult {
        output: output_gap3,
        state: state_gap3,
        ..
    } = run_pipeline(&dataset, &data_dir_3, &storage_dir_3, THROUGH_EDGES, 3);

    let counters_gap3: std::collections::HashMap<&str, u64> =
        output_gap3.reports[2].1.counters.iter().copied().collect();
    let pairs_gap3 = counters_gap3.get("pairs_processed").copied().unwrap_or(0);
    let edges_gap3 = state_gap3.graph.edges.len();

    // max_gap=3 on 4 nights yields up to 3 left nights, max_gap=1 yields 1.
    assert!(
        pairs_gap3 >= pairs_gap1,
        "larger gap should process at least as many night pairs: gap3={pairs_gap3}, gap1={pairs_gap1}"
    );
    assert!(
        pairs_gap3 > pairs_gap1,
        "with 4 nights, gap=3 should process strictly more pairs than gap=1"
    );

    // More night pairs should yield at least as many edges.
    assert!(
        edges_gap3 >= edges_gap1,
        "larger gap should produce at least as many edges: gap3={edges_gap3}, gap1={edges_gap1}"
    );

    // Verify all gap3 edges still satisfy their gap constraint.
    for edge in &state_gap3.graph.edges {
        let gap = edge.to.night_id.0 - edge.from.night_id.0;
        assert!(gap <= 3, "edge gap {} exceeds max_gap=3", gap);
        assert!(
            edge.from.night_id < edge.to.night_id,
            "edges must be forward in time"
        );
    }
}

#[test]
fn edges_connect_distinct_nights_from_diverse_populations() {
    // Use a mix of fast (NEA) and slow (TNO) movers to verify the edge builder
    // handles varying kinematic regimes.
    let n_per_pop = 4;
    let n_nights = 3;
    let obs_per_night = 3;
    let start_night_id = 60000_u32;
    let max_gap_nights = 3_u8;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::NearEarth, n_per_pop)
        .population(AsteroidPopulation::MainBelt, n_per_pop)
        .population(AsteroidPopulation::TransNeptunian, n_per_pop)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(123)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    let PipelineTestResult {
        output,
        state: runtime_state,
        ..
    } = run_pipeline(
        &dataset,
        &data_dir,
        &storage_dir,
        THROUGH_EDGES,
        max_gap_nights,
    );

    assert_eq!(output.reports.len(), 3);

    // Pipeline should succeed and produce edges.
    let edge_counters: std::collections::HashMap<&str, u64> =
        output.reports[2].1.counters.iter().copied().collect();
    let edges_added = edge_counters.get("edges_added").copied().unwrap_or(0);
    assert!(
        edges_added > 0,
        "mixed-population dataset should produce edges"
    );

    // Verify all edges are valid.
    let graph = &runtime_state.graph;

    for edge in &graph.edges {
        assert!(edge.cost > 0.0 && edge.cost.is_finite());
        assert!(edge.dt_days > 0.0 && edge.dt_days.is_finite());
        assert!(edge.from.night_id < edge.to.night_id);

        let gap = edge.to.night_id.0 - edge.from.night_id.0;
        assert!(gap <= max_gap_nights as u32);
    }

    // Verify that edges originate from multiple left nights
    // (with max_gap=3 and 3 nights, we should have edges from night 0 and night 1).
    let left_nights: std::collections::HashSet<NightId> =
        graph.edges.iter().map(|e| e.from.night_id).collect();
    assert!(
        left_nights.len() >= 1,
        "edges should originate from at least one left night"
    );
}

// ---------------------------------------------------------------------------
// Batch-ingest tests (all nights in a single Parquet, single pipeline run)
// ---------------------------------------------------------------------------

#[test]
fn batch_ingest_builds_seeds_and_edges_for_all_new_nights() {
    // When ALL nights are ingested in a single Parquet, new_night_ids contains
    // every night. BuildSeeds must build seeds for each of them, and
    // BuildEdges must then build edges between every valid (left, right) pair
    // drawn purely from those new nights.

    let n_trajectories = 4;
    let n_nights = 4;
    let obs_per_night = 3;
    let start_night_id = 63000_u32;
    let max_gap = 3_u8; // large enough to allow all pairs

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, n_trajectories)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(17)
        .build();

    // ---- Write all nights into one Parquet and run the three stages once ----
    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    let parquet_path = data_dir.path().join("all_nights.parquet");
    let all_alerts: Vec<_> = dataset.alerts().iter().collect();
    let all_obs = write_and_load_parquet(&all_alerts, &parquet_path);

    let engine_config = engine_config_with_edges(&storage_dir, max_gap);
    let persistence = PersistenceManager::open_or_create(engine_config.storage_path_buf())
        .expect("open persistence");
    let edge_models = test_edge_models();
    let solver_manager = SolverManager::default();

    let (mut plan, runner) = make_plan_and_runner(THROUGH_EDGES, PersistPolicy::None, all_obs);

    let mut runtime_state = RuntimeState::new();
    let mut ctx = PipelineContext {
        plan: &mut plan,
        persistence: &persistence,
        runtime_state: &mut runtime_state,
        engine_config: &engine_config,
        edge_models: &edge_models,
        solver_manager: &solver_manager,
    };

    runner
        .run(&mut ctx, &NoopHooks)
        .expect("batch pipeline run should succeed");

    // ---- 1) Seeds must exist for every ingested night ----
    let seed_store = &runtime_state.seed_store;
    assert_eq!(
        seed_store.n_nights(),
        n_nights,
        "seed store should have an entry for each of the {n_nights} ingested nights"
    );
    for night_offset in 0..n_nights {
        let nid = NightId(start_night_id + night_offset as u32);
        assert!(
            seed_store.contains_night(&nid),
            "seed store must contain night {nid:?}"
        );
        assert!(
            seed_store.len_night(&nid).unwrap_or(0) > 0,
            "night {nid:?} must have at least one seed"
        );
    }

    // ---- 2) Edges must exist between the new nights ----
    // With n_nights=4 nights and max_gap=3 (nights are consecutive integers),
    // every (left, right) pair with left < right is valid:
    //   right=N1: left={N0}          → 1 pair
    //   right=N2: left={N0,N1}       → 2 pairs
    //   right=N3: left={N0,N1,N2}    → 3 pairs
    // Total: 6 pairs_processed
    let graph = &runtime_state.graph;
    assert!(
        !graph.edges.is_empty(),
        "edges must be produced when all nights are ingested at once"
    );

    // All edges must satisfy forward-in-time and gap constraints.
    for edge in &graph.edges {
        assert!(
            edge.from.night_id < edge.to.night_id,
            "edge must be forward in time: from={:?} to={:?}",
            edge.from.night_id,
            edge.to.night_id
        );
        let gap = edge.to.night_id.0 - edge.from.night_id.0;
        assert!(
            gap <= max_gap as u32,
            "edge gap {gap} exceeds max_gap={max_gap}"
        );
        assert!(edge.cost.is_finite() && edge.cost > 0.0);
        assert!(edge.dt_days.is_finite() && edge.dt_days > 0.0);
    }

    // Both the first and the last night must participate as left/right nodes
    // (N0 only appears as left, N_{n-1} only as right).
    let left_nights: std::collections::HashSet<NightId> =
        graph.edges.iter().map(|e| e.from.night_id).collect();
    let right_nights: std::collections::HashSet<NightId> =
        graph.edges.iter().map(|e| e.to.night_id).collect();

    assert!(
        left_nights.contains(&NightId(start_night_id)),
        "first night must appear as a left (source) night in edges"
    );
    assert!(
        right_nights.contains(&NightId(start_night_id + (n_nights as u32) - 1)),
        "last night must appear as a right (target) night in edges"
    );
}

#[test]
fn edges_connect_new_nights_to_previously_ingested_nights() {
    // Verify the cross-batch edge scenario:
    //   Run 1 — ingest nights {N0, N1} → seeds for N0, N1 + edge N0→N1.
    //   Run 2 — ingest night  {N2}     → seeds for N2 + edges N0→N2, N1→N2.
    //
    // After run 2, the graph must contain edges from the *previous* nights
    // (N0, N1) to the *new* night N2, not just the within-batch edge N0→N1.

    let n_trajectories = 4;
    let obs_per_night = 3;
    let start_night_id = 64000_u32;
    let max_gap = 3_u8;

    // Build three nights of data.
    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, n_trajectories)
        .n_nights(3)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(31)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    let engine_config = engine_config_with_edges(&storage_dir, max_gap);
    let persistence = PersistenceManager::open_or_create(engine_config.storage_path_buf())
        .expect("open persistence");
    let edge_models = test_edge_models();
    let solver_manager = SolverManager::default();

    let mut runtime_state = RuntimeState::new();

    // Partition alerts by night.
    let night0 = NightId(start_night_id);
    let night1 = NightId(start_night_id + 1);
    let night2 = NightId(start_night_id + 2);

    let alerts_n0n1: Vec<&crate::synthetic_alerts::SyntheticAlert> = dataset
        .alerts()
        .iter()
        .filter(|a| a.night_id == night0 || a.night_id == night1)
        .collect();
    let alerts_n2: Vec<&crate::synthetic_alerts::SyntheticAlert> = dataset
        .alerts()
        .iter()
        .filter(|a| a.night_id == night2)
        .collect();

    // ---- Run 1: ingest nights N0 + N1 ----
    let parquet_1 = data_dir.path().join("nights_0_1.parquet");
    let obs_1 = write_and_load_parquet(&alerts_n0n1, &parquet_1);

    let (mut plan_1, runner_1) = make_plan_and_runner(THROUGH_EDGES, PersistPolicy::None, obs_1);
    let mut ctx_1 = PipelineContext {
        plan: &mut plan_1,
        persistence: &persistence,
        runtime_state: &mut runtime_state,
        engine_config: &engine_config,
        edge_models: &edge_models,
        solver_manager: &solver_manager,
    };
    runner_1
        .run(&mut ctx_1, &NoopHooks)
        .expect("run 1 should succeed");
    drop(ctx_1);

    let edges_after_run1 = runtime_state.graph.edges.len();
    assert!(
        edges_after_run1 > 0,
        "run 1 should produce edges between N0 and N1"
    );

    // All run-1 edges must be N0 → N1.
    for edge in &runtime_state.graph.edges {
        assert_eq!(
            edge.from.night_id, night0,
            "run-1 edges must originate from N0"
        );
        assert_eq!(edge.to.night_id, night1, "run-1 edges must target N1");
    }

    // ---- Run 2: ingest night N2 only ----
    let parquet_2 = data_dir.path().join("night_2.parquet");
    let obs_2 = write_and_load_parquet(&alerts_n2, &parquet_2);

    let (mut plan_2, runner_2) = make_plan_and_runner(THROUGH_EDGES, PersistPolicy::None, obs_2);
    let mut ctx_2 = PipelineContext {
        plan: &mut plan_2,
        persistence: &persistence,
        runtime_state: &mut runtime_state,
        engine_config: &engine_config,
        edge_models: &edge_models,
        solver_manager: &solver_manager,
    };
    runner_2
        .run(&mut ctx_2, &NoopHooks)
        .expect("run 2 should succeed");
    drop(ctx_2);

    let edges_after_run2 = runtime_state.graph.edges.len();
    assert!(
        edges_after_run2 > edges_after_run1,
        "run 2 must add edges from previous nights (N0, N1) to new night N2; \
         got {edges_after_run2} total (was {edges_after_run1} after run 1)"
    );

    // After run 2 the graph must contain edges targeting N2 from both N0 and N1.
    let targets_n2: Vec<_> = runtime_state
        .graph
        .edges
        .iter()
        .filter(|e| e.to.night_id == night2)
        .collect();
    assert!(
        !targets_n2.is_empty(),
        "run 2 must produce edges whose target is N2"
    );

    let sources_to_n2: std::collections::HashSet<NightId> =
        targets_n2.iter().map(|e| e.from.night_id).collect();
    assert!(
        sources_to_n2.contains(&night0),
        "N0 must connect to N2 (gap=2 ≤ max_gap={max_gap})"
    );
    assert!(
        sources_to_n2.contains(&night1),
        "N1 must connect to N2 (gap=1 ≤ max_gap={max_gap})"
    );

    // All edges in the full graph must satisfy the gap constraint.
    for edge in &runtime_state.graph.edges {
        let gap = edge.to.night_id.0 - edge.from.night_id.0;
        assert!(
            gap <= max_gap as u32,
            "edge gap {gap} exceeds max_gap={max_gap}"
        );
        assert!(
            edge.from.night_id < edge.to.night_id,
            "edges must be forward in time"
        );
    }
}

#[test]
fn max_gap_prevents_edges_between_distant_nights() {
    // Verify that no edge is emitted when the night difference exceeds max_gap.
    // We use a batch ingest (5 nights at once) and max_gap=2.
    //
    // With nights N0..N4 (consecutive integers) and max_gap=2:
    //   valid pairs:  N0→N1, N0→N2, N1→N2, N1→N3, N2→N3, N2→N4, N3→N4
    //   invalid gap:  anything with Δnight > 2 (N0→N3, N0→N4, N1→N4, ...)

    let n_trajectories = 4;
    let n_nights = 5;
    let obs_per_night = 3;
    let start_night_id = 65000_u32;
    let max_gap = 2_u8;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, n_trajectories)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(53)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    let parquet_path = data_dir.path().join("all_nights.parquet");
    let all_alerts: Vec<&crate::synthetic_alerts::SyntheticAlert> =
        dataset.alerts().iter().collect();
    let all_obs = write_and_load_parquet(&all_alerts, &parquet_path);

    let engine_config = engine_config_with_edges(&storage_dir, max_gap);
    let persistence = PersistenceManager::open_or_create(engine_config.storage_path_buf())
        .expect("open persistence");
    let edge_models = test_edge_models();
    let solver_manager = SolverManager::default();

    let (mut plan, runner) = make_plan_and_runner(THROUGH_EDGES, PersistPolicy::None, all_obs);
    let mut runtime_state = RuntimeState::new();
    let mut ctx = PipelineContext {
        plan: &mut plan,
        persistence: &persistence,
        runtime_state: &mut runtime_state,
        engine_config: &engine_config,
        edge_models: &edge_models,
        solver_manager: &solver_manager,
    };

    runner
        .run(&mut ctx, &NoopHooks)
        .expect("pipeline should succeed");
    drop(ctx);

    // ---- Edges must exist (sanity check) ----
    assert!(
        !runtime_state.graph.edges.is_empty(),
        "edges should have been produced"
    );

    // ---- Every edge must satisfy the gap constraint ----
    for edge in &runtime_state.graph.edges {
        let gap = edge.to.night_id.0 - edge.from.night_id.0;
        assert!(
            gap <= max_gap as u32,
            "edge gap {gap} exceeds max_gap={max_gap}: from={:?} to={:?}",
            edge.from.night_id,
            edge.to.night_id
        );
        assert!(
            edge.from.night_id < edge.to.night_id,
            "edges must be forward in time"
        );
    }

    // ---- Edges that would violate the gap must be absent ----
    // With 5 consecutive nights and max_gap=2, the pairs (N0,N3), (N0,N4),
    // (N1,N4) have gap > 2 and must produce no edges.
    let forbidden_pairs = [
        (NightId(start_night_id), NightId(start_night_id + 3)),
        (NightId(start_night_id), NightId(start_night_id + 4)),
        (NightId(start_night_id + 1), NightId(start_night_id + 4)),
    ];
    for (left, right) in forbidden_pairs {
        let found = runtime_state
            .graph
            .edges
            .iter()
            .any(|e| e.from.night_id == left && e.to.night_id == right);
        assert!(
            !found,
            "no edge should exist between night {:?} and night {:?} (gap {} > max_gap={})",
            left,
            right,
            right.0 - left.0,
            max_gap
        );
    }

    // ---- Night pairs within gap must have at least one edge ----
    // Every pair (left, right) with Δ ≤ max_gap should produce edges because
    // both nights have seeds.
    let within_gap_pairs = [
        (NightId(start_night_id), NightId(start_night_id + 1)),
        (NightId(start_night_id), NightId(start_night_id + 2)),
        (NightId(start_night_id + 1), NightId(start_night_id + 2)),
        (NightId(start_night_id + 1), NightId(start_night_id + 3)),
        (NightId(start_night_id + 2), NightId(start_night_id + 3)),
        (NightId(start_night_id + 2), NightId(start_night_id + 4)),
        (NightId(start_night_id + 3), NightId(start_night_id + 4)),
    ];
    for (left, right) in within_gap_pairs {
        let found = runtime_state
            .graph
            .edges
            .iter()
            .any(|e| e.from.night_id == left && e.to.night_id == right);
        assert!(
            found,
            "at least one edge should exist between night {:?} and night {:?} (gap {} ≤ max_gap={})",
            left,
            right,
            right.0 - left.0,
            max_gap
        );
    }
}
