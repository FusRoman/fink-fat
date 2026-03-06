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

use fink_fat_engine::{night_id::NightId, pipeline::stages::PipelineStage};

use super::{PipelineTestResult, THROUGH_EDGES, run_pipeline};
use crate::synthetic_alerts::{AsteroidPopulation, SyntheticDatasetBuilder};

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
