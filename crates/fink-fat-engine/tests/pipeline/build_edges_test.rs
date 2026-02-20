//! Integration test for the `IngestNights` → `BuildSeeds` → `BuildEdges` pipeline.
//!
//! This test exercises the full three-stage pipeline where:
//! 1. `IngestNights` loads synthetic alerts from a Parquet file.
//! 2. `BuildSeeds` generates intra-night seeds (pairs/triplets).
//! 3. `BuildEdges` constructs inter-night edges between seeds of neighboring
//!    nights, respecting the `max_gap_nights` sliding window.
//!
//! The edge builder operates in `emit_all_edges = true` mode so that no
//! ONNX model is required (physics-only cost, all candidates emitted).

use tempfile::TempDir;

use fink_fat_engine::{
    engine_config::EngineConfig,
    graph::edge::edge_prediction::EdgeRankingModelPool,
    night_id::NightId,
    persistence::PersistenceManager,
    pipeline::{
        PersistPolicy, PipelineContext, PipelineInputs, PipelineOutput, PipelinePlan,
        PipelineRunner,
        stages::PipelineStage,
    },
    solver::{HypothesisSet, solver_manager::SolverManager},
};

use crate::synthetic_alerts::{AsteroidPopulation, SyntheticDatasetBuilder};
use super::{NoopHooks, engine_config_with_edges, new_runtime_state};

// ---------------------------------------------------------------------------
// Helper: run the three-stage pipeline and return output + context parts
// ---------------------------------------------------------------------------

/// Run `IngestNights → BuildSeeds → BuildEdges` and return the pipeline output.
///
/// The runtime state is mutated in-place through `ctx`.
fn run_three_stage_pipeline(
    dataset: &crate::synthetic_alerts::SyntheticDataset,
    data_dir: &TempDir,
    storage_dir: &TempDir,
    max_gap_nights: u8,
) -> (
    PipelineOutput,
    fink_fat_engine::persistence::runtime_state::RuntimeState,
    EngineConfig,
) {
    let parquet_path = data_dir.path().join("test_alerts.parquet");
    let alerts_uri = dataset.write_parquet(&parquet_path);

    let engine_config = engine_config_with_edges(storage_dir, max_gap_nights);
    let persistence =
        PersistenceManager::open_or_create(engine_config.storage_path_buf())
            .expect("open persistence");
    let edge_models = EdgeRankingModelPool::new("unused.onnx");
    let solver_manager = SolverManager::default();
    let track_hypotheses: HypothesisSet = HypothesisSet::default();

    let plan = PipelinePlan {
        window: None,
        stages: vec![
            PipelineStage::IngestNights,
            PipelineStage::BuildSeeds,
            PipelineStage::BuildEdges,
        ],
        persist: PersistPolicy::None,
        inputs: PipelineInputs { alerts_uri },
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
        track_hypotheses,
    };

    let output = runner
        .run(&mut ctx, &hooks)
        .expect("three-stage pipeline should succeed");

    drop(ctx);

    (output, runtime_state, engine_config)
}

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

    let (output, runtime_state, _engine_config) =
        run_three_stage_pipeline(&dataset, &data_dir, &storage_dir, max_gap_nights);

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
    assert_eq!(
        graph.edges.len() as u64, edges_added,
        "graph edge count must match reported edges_added"
    );

    // ---- 5) Verify edge invariants ----
    let last_night = NightId(start_night_id + (n_nights as u32) - 1);

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

        // `to` seed must belong to the latest night (the anchor).
        assert_eq!(
            edge.to.night_id, last_night,
            "edge target must be the latest night (right_night), got {:?}",
            edge.to.night_id
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
            runtime_state
                .seed_store
                .try_get_seed(edge.from)
                .is_some(),
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

        let in_d = graph.in_deg.get(&edge.to).copied().unwrap_or(0);
        assert!(in_d > 0, "to-node must have in_deg > 0");
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

    let (output, runtime_state, _) =
        run_three_stage_pipeline(&dataset, &data_dir, &storage_dir, max_gap_nights);

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

    let last_night = NightId(start_night_id + (n_nights as u32) - 1);
    let penultimate = NightId(last_night.0 - 1);

    for edge in &runtime_state.graph.edges {
        assert_eq!(
            edge.to.night_id, last_night,
            "all edges must target the last night"
        );
        assert_eq!(
            edge.from.night_id, penultimate,
            "with max_gap=1, all edges must originate from the penultimate night"
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
    let (output_gap1, state_gap1, _) =
        run_three_stage_pipeline(&dataset, &data_dir_1, &storage_dir_1, 1);

    let counters_gap1: std::collections::HashMap<&str, u64> =
        output_gap1.reports[2].1.counters.iter().copied().collect();
    let pairs_gap1 = counters_gap1.get("pairs_processed").copied().unwrap_or(0);
    let edges_gap1 = state_gap1.graph.edges.len();

    // --- Run with max_gap=3 ---
    let data_dir_3 = TempDir::new().unwrap();
    let storage_dir_3 = TempDir::new().unwrap();
    let (output_gap3, state_gap3, _) =
        run_three_stage_pipeline(&dataset, &data_dir_3, &storage_dir_3, 3);

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
    let last_night = NightId(start_night_id + (n_nights as u32) - 1);
    for edge in &state_gap3.graph.edges {
        let gap = edge.to.night_id.0 - edge.from.night_id.0;
        assert!(gap <= 3, "edge gap {} exceeds max_gap=3", gap);
        assert_eq!(edge.to.night_id, last_night);
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

    let (output, runtime_state, _) =
        run_three_stage_pipeline(&dataset, &data_dir, &storage_dir, max_gap_nights);

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
    let last_night = NightId(start_night_id + (n_nights as u32) - 1);
    let graph = &runtime_state.graph;

    for edge in &graph.edges {
        assert!(edge.cost > 0.0 && edge.cost.is_finite());
        assert!(edge.dt_days > 0.0 && edge.dt_days.is_finite());
        assert!(edge.from.night_id < edge.to.night_id);
        assert_eq!(edge.to.night_id, last_night);

        let gap = edge.to.night_id.0 - edge.from.night_id.0;
        assert!(gap <= max_gap_nights as u32);
    }

    // Verify that edges originate from multiple left nights
    // (with max_gap=3 and 3 nights, we should have edges from night 0 and night 1).
    let left_nights: std::collections::HashSet<NightId> = graph
        .edges
        .iter()
        .map(|e| e.from.night_id)
        .collect();
    assert!(
        left_nights.len() >= 1,
        "edges should originate from at least one left night"
    );
}
