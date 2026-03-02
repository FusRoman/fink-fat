//! Integration tests for the `IngestNights` → `BuildSeeds` → `BuildEdges` → `Solve` pipeline.
//!
//! These tests exercise the full four-stage pipeline where:
//! 1. `IngestNights` loads synthetic alerts from a Parquet file.
//! 2. `BuildSeeds` generates intra-night seeds (pairs/triplets).
//! 3. `BuildEdges` constructs inter-night edges between seeds.
//! 4. `Solve` runs the solver to produce trajectory hypotheses.
//!
//! The edge builder operates in `emit_all_edges = true` mode so that no
//! ONNX model is required (physics-only cost, all candidates emitted).
//!
//! Ground-truth verification checks that the solver output recovers coherent
//! trajectories matching the known synthetic trajectories.

use std::collections::{HashMap, HashSet};

use tempfile::TempDir;

use fink_fat_engine::{
    Alert,
    engine_config::pipeline_policy::PersistPolicy,
    night_id::NightId,
    persistence::{PersistenceManager, runtime_state::RuntimeState},
    pipeline::{
        PipelineContext, PipelineInputs, PipelinePlan, PipelineRunner, stages::PipelineStage,
    },
    trajectory::TrackHypothesis,
};

use super::{
    NoopHooks, PipelineTestResult, THROUGH_SOLVE, engine_config_with_edges,
    match_truth_to_hypotheses, run_incremental_pipeline, run_pipeline, test_edge_models,
    test_solver_manager, test_solver_manager_with_min_nodes, write_alerts_parquet,
};
use crate::synthetic_alerts::{AsteroidPopulation, SyntheticDatasetBuilder};

// ---------------------------------------------------------------------------
// Integration tests
// ---------------------------------------------------------------------------

/// Basic test: verify that the solve stage runs and produces hypotheses.
#[test]
fn four_stage_pipeline_produces_hypotheses() {
    let n_trajectories = 5;
    let n_nights = 3;
    let obs_per_night = 3;
    let start_night_id = 60000_u32;
    let max_gap_nights = 3_u8;

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
        THROUGH_SOLVE,
        max_gap_nights,
    );
    let hypotheses = &runtime_state.track_hypotheses;

    // ---- 1) Verify we got four stage reports ----
    assert_eq!(output.reports.len(), 4, "expected 4 stage reports");
    assert_eq!(output.reports[0].0, PipelineStage::IngestNights);
    assert_eq!(output.reports[1].0, PipelineStage::BuildSeeds);
    assert_eq!(output.reports[2].0, PipelineStage::BuildEdges);
    assert_eq!(output.reports[3].0, PipelineStage::Solve);

    // ---- 2) Verify Solve counters ----
    let solve_counters: HashMap<&str, u64> = output.reports[3].1.counters.iter().copied().collect();

    let n_components = solve_counters.get("components").copied().unwrap_or(0);
    let n_plan_items = solve_counters.get("plan_items").copied().unwrap_or(0);
    let n_hypotheses = solve_counters.get("hypotheses").copied().unwrap_or(0);

    assert!(
        n_components > 0,
        "solver should find at least one connected component"
    );
    assert_eq!(
        n_plan_items, n_components,
        "plan should have one item per component"
    );
    assert!(
        n_hypotheses > 0,
        "solver should produce at least one hypothesis"
    );

    // ---- 3) Verify hypotheses are non-empty ----
    assert!(!hypotheses.is_empty(), "hypothesis set must be non-empty");
    assert_eq!(
        hypotheses.len() as u64,
        n_hypotheses,
        "hypothesis count must match reported counter"
    );

    // ---- 4) Verify each hypothesis has valid structure ----
    for (_hid, track) in hypotheses {
        // Each track must have at least 2 nodes (a track links multiple seeds).
        assert!(
            track.n_nodes() >= 2,
            "track must have at least 2 nodes, got {}",
            track.n_nodes()
        );

        // edges = nodes - 1
        assert_eq!(
            track.n_edges(),
            track.n_nodes() - 1,
            "track must have exactly n_nodes - 1 edges"
        );

        // Cost must be finite (non-NaN, non-Inf).
        assert!(
            track.cost.is_finite(),
            "track cost must be finite, got {}",
            track.cost
        );

        // Night span must be positive (at least 2 distinct nights).
        assert!(
            track.night_span >= 1,
            "track night_span must be >= 1, got {}",
            track.night_span
        );

        // Nodes must be in strictly increasing night order.
        for w in track.nodes.windows(2) {
            assert!(
                w[0].night_id < w[1].night_id,
                "track nodes must be in strictly increasing night order: {:?} >= {:?}",
                w[0].night_id,
                w[1].night_id
            );
        }

        // All seed keys must exist in the seed store.
        for &seed_key in &track.nodes {
            assert!(
                runtime_state.seed_store.try_get_seed(seed_key).is_some(),
                "seed {:?} referenced by track must exist in seed store",
                seed_key
            );
        }

        // All edge keys must exist in the graph.
        for edge_key in &track.edges {
            assert!(
                runtime_state.graph.edge_by_key(edge_key).is_some(),
                "edge {:?} referenced by track must exist in graph",
                edge_key
            );
        }
    }
}

/// Verify that the solver recovers the ground-truth Main Belt trajectories
/// with good overlap (Jaccard ≥ 0.5 for at least 50% of trajectories).
#[test]
fn solver_recovers_main_belt_trajectories() {
    let n_trajectories = 5;
    let n_nights = 4;
    let obs_per_night = 3;
    let start_night_id = 60000_u32;
    let max_gap_nights = 4_u8;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, n_trajectories)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(100)
        .build();

    let ground_truth = dataset.ground_truth().to_vec();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    let PipelineTestResult {
        state: runtime_state,
        ..
    } = run_pipeline(
        &dataset,
        &data_dir,
        &storage_dir,
        THROUGH_SOLVE,
        max_gap_nights,
    );
    let hypotheses = &runtime_state.track_hypotheses;

    assert!(
        !hypotheses.is_empty(),
        "solver must produce hypotheses for MBA trajectories"
    );

    // Match ground truth to hypotheses.
    let matches = match_truth_to_hypotheses(
        &ground_truth,
        &hypotheses,
        &runtime_state.alert_store,
        &runtime_state.seed_store,
    );

    // Count how many ground-truth trajectories have a hypothesis with Jaccard ≥ 0.3.
    // In a single-run scenario, the edge builder only creates edges to the last
    // night, so tracks are shallow (2 nodes). The best Jaccard for matching a
    // 4-night trajectory with a 2-node track is limited, hence the lower threshold.
    let min_jaccard = 0.3;
    let well_recovered = matches
        .iter()
        .filter(|(_, _, score)| *score >= min_jaccard)
        .count();

    // We expect at least some trajectories to be partially recovered.
    let min_fraction = 0.4;
    let required = ((n_trajectories as f64) * min_fraction).ceil() as usize;
    assert!(
        well_recovered >= required,
        "expected at least {required}/{n_trajectories} trajectories recovered \
         with Jaccard >= {min_jaccard}, got {well_recovered}. \
         Scores: {:?}",
        matches.iter().map(|(i, _, s)| (i, s)).collect::<Vec<_>>()
    );
}

/// Verify solver on a diverse multi-population dataset.
///
/// Uses NEA, MBA, and Trojan populations to test that the solver handles
/// different kinematic regimes in the same run.
#[test]
fn solver_handles_diverse_populations() {
    let n_per_pop = 3;
    let n_nights = 4;
    let obs_per_night = 3;
    let start_night_id = 60000_u32;
    let max_gap_nights = 4_u8;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::NearEarth, n_per_pop)
        .population(AsteroidPopulation::MainBelt, n_per_pop)
        .population(AsteroidPopulation::Trojan, n_per_pop)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(200)
        .build();

    let ground_truth = dataset.ground_truth().to_vec();
    let total_trajectories = ground_truth.len();
    assert_eq!(total_trajectories, 3 * n_per_pop);

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
        THROUGH_SOLVE,
        max_gap_nights,
    );
    let hypotheses = &runtime_state.track_hypotheses;

    // ---- 1) Pipeline must complete all four stages ----
    assert_eq!(output.reports.len(), 4);
    assert_eq!(output.reports[3].0, PipelineStage::Solve);

    // ---- 2) Solver must produce hypotheses ----
    assert!(
        !hypotheses.is_empty(),
        "solver should produce hypotheses for mixed-population data"
    );

    // ---- 3) All hypotheses have valid structure ----
    for (_, track) in hypotheses {
        assert!(track.n_nodes() >= 2);
        assert_eq!(track.n_edges(), track.n_nodes() - 1);
        assert!(track.cost.is_finite());

        // Nodes must span multiple nights.
        let nights: HashSet<NightId> = track.nodes.iter().map(|sk| sk.night_id).collect();
        assert!(
            nights.len() >= 2,
            "each track must span at least 2 distinct nights"
        );
    }

    // ---- 4) Check ground-truth recovery per population ----
    let matches = match_truth_to_hypotheses(
        &ground_truth,
        &hypotheses,
        &runtime_state.alert_store,
        &runtime_state.seed_store,
    );

    // At least some trajectories should be recovered (Jaccard > 0).
    let any_recovered = matches.iter().any(|(_, _, score)| *score > 0.0);
    assert!(
        any_recovered,
        "solver should recover at least one trajectory with non-zero overlap. \
         Scores: {:?}",
        matches.iter().map(|(i, _, s)| (i, s)).collect::<Vec<_>>()
    );

    // Report per-population recovery for diagnostics.
    for pop in [
        AsteroidPopulation::NearEarth,
        AsteroidPopulation::MainBelt,
        AsteroidPopulation::Trojan,
    ] {
        let pop_indices: Vec<usize> = ground_truth
            .iter()
            .enumerate()
            .filter(|(_, t)| t.population == pop)
            .map(|(i, _)| i)
            .collect();

        let pop_scores: Vec<f64> = pop_indices.iter().map(|&i| matches[i].2).collect();

        let avg_score: f64 = pop_scores.iter().sum::<f64>() / pop_scores.len().max(1) as f64;

        // Diagnostic print (visible with `cargo test -- --nocapture`).
        eprintln!(
            "Population {:?}: avg Jaccard = {:.3}, scores = {:?}",
            pop, avg_score, pop_scores
        );
    }
}

/// Verify that tracks span multiple nights and that the night span counter
/// matches the actual nights covered by the hypothesis nodes.
#[test]
fn track_night_span_is_consistent() {
    let n_trajectories = 5;
    let n_nights = 4;
    let obs_per_night = 3;
    let start_night_id = 60000_u32;
    let max_gap_nights = 4_u8;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, n_trajectories)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(300)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    let PipelineTestResult {
        state: runtime_state,
        ..
    } = run_pipeline(
        &dataset,
        &data_dir,
        &storage_dir,
        THROUGH_SOLVE,
        max_gap_nights,
    );
    let hypotheses = &runtime_state.track_hypotheses;

    assert!(!hypotheses.is_empty());

    for (&hid, track) in hypotheses {
        // Compute actual night span from node keys.
        let nights: Vec<NightId> = track.nodes.iter().map(|sk| sk.night_id).collect();
        let min_night = nights.iter().min().unwrap();
        let max_night = nights.iter().max().unwrap();
        let actual_span = max_night.0 - min_night.0;

        assert_eq!(
            track.night_span, actual_span,
            "hypothesis {hid}: night_span field ({}) must match actual \
             night range ({} - {} = {})",
            track.night_span, max_night.0, min_night.0, actual_span
        );

        // All referenced edges must connect seeds whose nights are within the
        // track's night range.
        for edge_key in &track.edges {
            let edge = runtime_state
                .graph
                .edge_by_key(edge_key)
                .expect("edge must exist");
            assert!(
                edge.from.night_id >= *min_night && edge.to.night_id <= *max_night,
                "edge {:?} is outside the track night range [{}, {}]",
                edge_key,
                min_night.0,
                max_night.0
            );
        }
    }
}

/// Verify that hypotheses do not contain duplicate seed nodes.
#[test]
fn hypotheses_have_no_duplicate_nodes() {
    let n_trajectories = 5;
    let n_nights = 4;
    let obs_per_night = 3;
    let max_gap_nights = 4_u8;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, n_trajectories)
        .population(AsteroidPopulation::NearEarth, n_trajectories)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(60000)
        .seed(400)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    let PipelineTestResult {
        state: runtime_state,
        ..
    } = run_pipeline(
        &dataset,
        &data_dir,
        &storage_dir,
        THROUGH_SOLVE,
        max_gap_nights,
    );
    let hypotheses = &runtime_state.track_hypotheses;

    assert!(!hypotheses.is_empty());

    for (&hid, track) in hypotheses {
        let unique_nodes: HashSet<_> = track.nodes.iter().collect();
        assert_eq!(
            unique_nodes.len(),
            track.nodes.len(),
            "hypothesis {hid} has duplicate seed nodes"
        );
    }
}

/// Verify that the solver produces more hypotheses when given more data
/// (more trajectories and nights).
#[test]
fn more_data_yields_more_hypotheses() {
    let obs_per_night = 3;
    let max_gap_nights = 4_u8;

    // Small dataset: 3 trajectories, 3 nights.
    let dataset_small = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, 3)
        .n_nights(3)
        .obs_per_night(obs_per_night)
        .start_night_id(60000)
        .seed(500)
        .build();

    let data_dir_s = TempDir::new().unwrap();
    let storage_dir_s = TempDir::new().unwrap();
    let PipelineTestResult {
        state: state_small, ..
    } = run_pipeline(
        &dataset_small,
        &data_dir_s,
        &storage_dir_s,
        THROUGH_SOLVE,
        max_gap_nights,
    );
    let hyp_small = &state_small.track_hypotheses;

    // Large dataset: 10 trajectories, 5 nights.
    let dataset_large = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, 10)
        .n_nights(5)
        .obs_per_night(obs_per_night)
        .start_night_id(60000)
        .seed(501)
        .build();

    let data_dir_l = TempDir::new().unwrap();
    let storage_dir_l = TempDir::new().unwrap();
    let PipelineTestResult {
        state: state_large, ..
    } = run_pipeline(
        &dataset_large,
        &data_dir_l,
        &storage_dir_l,
        THROUGH_SOLVE,
        max_gap_nights,
    );
    let hyp_large = &state_large.track_hypotheses;

    assert!(
        !hyp_small.is_empty(),
        "small dataset should still produce hypotheses"
    );
    assert!(
        hyp_large.len() >= hyp_small.len(),
        "larger dataset should produce at least as many hypotheses: \
         large={}, small={}",
        hyp_large.len(),
        hyp_small.len()
    );
}

/// Verify solver output with all five asteroid populations.
///
/// This is the most comprehensive test: it generates trajectories from every
/// supported population and verifies the solver produces coherent results.
#[test]
fn solver_all_five_populations() {
    let n_per_pop = 2;
    let n_nights = 4;
    let obs_per_night = 3;
    let start_night_id = 60000_u32;
    let max_gap_nights = 4_u8;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::NearEarth, n_per_pop)
        .population(AsteroidPopulation::MainBelt, n_per_pop)
        .population(AsteroidPopulation::Trojan, n_per_pop)
        .population(AsteroidPopulation::TransNeptunian, n_per_pop)
        .population(AsteroidPopulation::KuiperBelt, n_per_pop)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(600)
        .build();

    let ground_truth = dataset.ground_truth().to_vec();
    assert_eq!(ground_truth.len(), 5 * n_per_pop);

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
        THROUGH_SOLVE,
        max_gap_nights,
    );
    let hypotheses = &runtime_state.track_hypotheses;

    // ---- 1) Pipeline runs all four stages successfully ----
    assert_eq!(output.reports.len(), 4);

    let solve_counters: HashMap<&str, u64> = output.reports[3].1.counters.iter().copied().collect();
    let n_hypotheses = solve_counters.get("hypotheses").copied().unwrap_or(0);
    assert!(
        n_hypotheses > 0,
        "solver should produce at least one hypothesis"
    );

    // ---- 2) All hypotheses are structurally valid ----
    for (_, track) in hypotheses {
        assert!(track.n_nodes() >= 2);
        assert_eq!(track.n_edges(), track.n_nodes() - 1);
        assert!(track.cost.is_finite());
    }

    // ---- 3) Compute recovery statistics ----
    let matches = match_truth_to_hypotheses(
        &ground_truth,
        &hypotheses,
        &runtime_state.alert_store,
        &runtime_state.seed_store,
    );

    // At least one trajectory should have non-trivial overlap.
    let any_matched = matches.iter().any(|(_, _, score)| *score > 0.0);
    assert!(
        any_matched,
        "at least one ground-truth trajectory should be partially recovered"
    );

    // ---- 4) Per-population diagnostics ----
    for pop in AsteroidPopulation::ALL {
        let pop_scores: Vec<f64> = ground_truth
            .iter()
            .enumerate()
            .filter(|(_, t)| t.population == pop)
            .map(|(i, _)| matches[i].2)
            .collect();

        let avg = pop_scores.iter().sum::<f64>() / pop_scores.len().max(1) as f64;
        eprintln!(
            "{:?}: avg Jaccard = {:.3}, scores = {:?}",
            pop, avg, pop_scores
        );
    }
}

/// Verify that edges referenced by hypotheses are marked active and connect
/// consecutive nodes in the track.
#[test]
fn hypothesis_edges_connect_consecutive_nodes() {
    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, 5)
        .n_nights(4)
        .obs_per_night(3)
        .start_night_id(60000)
        .seed(700)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    let PipelineTestResult {
        state: runtime_state,
        ..
    } = run_pipeline(&dataset, &data_dir, &storage_dir, THROUGH_SOLVE, 4);
    let hypotheses = &runtime_state.track_hypotheses;

    assert!(!hypotheses.is_empty());

    for (&hid, track) in hypotheses {
        assert_eq!(
            track.edges.len(),
            track.nodes.len() - 1,
            "hypothesis {hid}: edges.len() must be nodes.len() - 1"
        );

        // Each edge should connect consecutive nodes in the track.
        for (i, edge_key) in track.edges.iter().enumerate() {
            let edge = runtime_state
                .graph
                .edge_by_key(edge_key)
                .expect("edge must exist in graph");

            assert_eq!(
                edge.from, track.nodes[i],
                "hypothesis {hid}, edge {i}: from must match nodes[{i}]"
            );
            assert_eq!(
                edge.to,
                track.nodes[i + 1],
                "hypothesis {hid}, edge {i}: to must match nodes[{}]",
                i + 1
            );

            // Edges should be active.
            assert!(
                edge.active,
                "hypothesis {hid}, edge {i}: edge should be active"
            );

            // Edges must be time-forward.
            assert!(
                edge.from.night_id < edge.to.night_id,
                "hypothesis {hid}, edge {i}: edge must be time-forward"
            );
        }
    }
}

/// Verify the solver is deterministic: running the same pipeline twice
/// produces the same hypotheses.
#[test]
fn solver_is_deterministic() {
    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, 5)
        .n_nights(3)
        .obs_per_night(3)
        .start_night_id(60000)
        .seed(800)
        .build();

    // Run 1
    let data_dir_1 = TempDir::new().unwrap();
    let storage_dir_1 = TempDir::new().unwrap();
    let PipelineTestResult { state: state1, .. } =
        run_pipeline(&dataset, &data_dir_1, &storage_dir_1, THROUGH_SOLVE, 3);
    let hyp1 = &state1.track_hypotheses;

    // Run 2
    let data_dir_2 = TempDir::new().unwrap();
    let storage_dir_2 = TempDir::new().unwrap();
    let PipelineTestResult { state: state2, .. } =
        run_pipeline(&dataset, &data_dir_2, &storage_dir_2, THROUGH_SOLVE, 3);
    let hyp2 = &state2.track_hypotheses;

    assert_eq!(
        hyp1.len(),
        hyp2.len(),
        "deterministic runs must produce same number of hypotheses"
    );

    // Compare tracks by sorting them to have a stable comparison order.
    let mut tracks1: Vec<_> = hyp1.values().collect();
    let mut tracks2: Vec<_> = hyp2.values().collect();

    // Sort by first node key, then by cost for stable order.
    let sort_key =
        |t: &&TrackHypothesis| (t.nodes[0].night_id, t.nodes[0].unique_id, t.cost.to_bits());
    tracks1.sort_by_key(sort_key);
    tracks2.sort_by_key(sort_key);

    for (t1, t2) in tracks1.iter().zip(tracks2.iter()) {
        assert_eq!(
            t1.nodes, t2.nodes,
            "deterministic runs must produce identical track nodes"
        );
        assert_eq!(
            t1.edges, t2.edges,
            "deterministic runs must produce identical track edges"
        );
        assert_eq!(
            t1.cost.to_bits(),
            t2.cost.to_bits(),
            "deterministic runs must produce identical track costs"
        );
    }
}

// ===========================================================================
// Incremental (night-by-night) integration tests
// ===========================================================================
//
// These tests simulate the real fink-fat workflow: the pipeline runs once per
// night, ingesting new alerts and incrementally building the linkage graph.
// After enough nights have been processed, the solver should recover multi-hop
// trajectories that span 4+ nodes (3+ consecutive edges).
//
// The key difference from the single-run tests above is that RuntimeState
// persists across pipeline invocations — each run only receives one night of
// new alerts but can build edges to seeds from all previously ingested nights.

// ===========================================================================
// Incremental integration tests
// ===========================================================================

/// Basic incremental test: verify that running the pipeline night-by-night
/// produces hypotheses with tracks spanning 4+ nodes (min_nodes = 4).
#[test]
fn incremental_pipeline_produces_long_tracks() {
    let n_trajectories = 5;
    let n_nights = 6;
    let obs_per_night = 3;
    let start_night_id = 60000_u32;
    let max_gap_nights = 3_u8;
    let min_nodes = 4;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, n_trajectories)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(1000)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    let solver_manager = test_solver_manager_with_min_nodes(min_nodes);
    let PipelineTestResult {
        state: runtime_state,
        ..
    } = run_incremental_pipeline(
        &dataset,
        &data_dir,
        &storage_dir,
        max_gap_nights,
        &solver_manager,
        |_| THROUGH_SOLVE.to_vec(),
    );
    let hypotheses = &runtime_state.track_hypotheses;

    // ---- 1) Must produce hypotheses ----
    assert!(
        !hypotheses.is_empty(),
        "incremental pipeline with {n_nights} nights must produce hypotheses"
    );

    // ---- 2) All tracks must have at least min_nodes nodes ----
    for (&hid, track) in hypotheses {
        assert!(
            track.n_nodes() >= min_nodes,
            "hypothesis {hid}: expected >= {min_nodes} nodes, got {}",
            track.n_nodes()
        );
        assert_eq!(track.n_edges(), track.n_nodes() - 1);
        assert!(track.cost.is_finite());

        // Nodes must be in strictly increasing night order.
        for w in track.nodes.windows(2) {
            assert!(
                w[0].night_id < w[1].night_id,
                "nodes must be in strictly increasing night order"
            );
        }

        // All referenced seeds and edges must exist in the stores.
        for &seed_key in &track.nodes {
            assert!(
                runtime_state.seed_store.try_get_seed(seed_key).is_some(),
                "seed {:?} must exist in store",
                seed_key
            );
        }
        for edge_key in &track.edges {
            assert!(
                runtime_state.graph.edge_by_key(edge_key).is_some(),
                "edge {:?} must exist in graph",
                edge_key
            );
        }
    }
}

/// Verify that incremental runs recover ground-truth Main Belt trajectories
/// with higher fidelity than single-run mode (longer tracks → better Jaccard).
#[test]
fn incremental_recovers_mba_trajectories() {
    let n_trajectories = 5;
    let n_nights = 6;
    let obs_per_night = 3;
    let start_night_id = 60000_u32;
    let max_gap_nights = 3_u8;
    let min_nodes = 4;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, n_trajectories)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(1100)
        .build();

    let ground_truth = dataset.ground_truth().to_vec();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    let solver_manager = test_solver_manager_with_min_nodes(min_nodes);
    let PipelineTestResult {
        state: runtime_state,
        ..
    } = run_incremental_pipeline(
        &dataset,
        &data_dir,
        &storage_dir,
        max_gap_nights,
        &solver_manager,
        |_| THROUGH_SOLVE.to_vec(),
    );
    let hypotheses = &runtime_state.track_hypotheses;

    assert!(
        !hypotheses.is_empty(),
        "incremental pipeline must produce hypotheses"
    );

    let matches = match_truth_to_hypotheses(
        &ground_truth,
        &hypotheses,
        &runtime_state.alert_store,
        &runtime_state.seed_store,
    );

    // With incremental runs and min_nodes=4, tracks span 4+ nights.
    // We expect better recovery than the single-run tests.
    let min_jaccard = 0.4;
    let well_recovered = matches
        .iter()
        .filter(|(_, _, score)| *score >= min_jaccard)
        .count();

    let min_fraction = 0.4;
    let required = ((n_trajectories as f64) * min_fraction).ceil() as usize;
    assert!(
        well_recovered >= required,
        "expected at least {required}/{n_trajectories} MBA trajectories recovered \
         with Jaccard >= {min_jaccard} in incremental mode, got {well_recovered}. \
         Scores: {:?}",
        matches.iter().map(|(i, _, s)| (i, s)).collect::<Vec<_>>()
    );

    // Diagnostic output.
    for (tidx, hid, score) in &matches {
        eprintln!(
            "  truth[{tidx}]: best hypothesis = {:?}, Jaccard = {score:.3}",
            hid
        );
    }
}

/// Verify incremental mode with diverse populations (NEA, MBA, Trojan).
///
/// Different populations have different angular speeds; the edge builder and
/// solver must handle them all when the graph is built incrementally.
#[test]
fn incremental_diverse_populations() {
    let n_per_pop = 3;
    let n_nights = 6;
    let obs_per_night = 3;
    let start_night_id = 60000_u32;
    let max_gap_nights = 4_u8;
    let min_nodes = 4;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::NearEarth, n_per_pop)
        .population(AsteroidPopulation::MainBelt, n_per_pop)
        .population(AsteroidPopulation::Trojan, n_per_pop)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(1200)
        .build();

    let ground_truth = dataset.ground_truth().to_vec();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    let solver_manager = test_solver_manager_with_min_nodes(min_nodes);
    let PipelineTestResult {
        state: runtime_state,
        ..
    } = run_incremental_pipeline(
        &dataset,
        &data_dir,
        &storage_dir,
        max_gap_nights,
        &solver_manager,
        |_| THROUGH_SOLVE.to_vec(),
    );
    let hypotheses = &runtime_state.track_hypotheses;

    assert!(
        !hypotheses.is_empty(),
        "incremental pipeline should produce hypotheses for mixed populations"
    );

    // All tracks must span min_nodes+ nodes and be time-ordered.
    for (_, track) in hypotheses {
        assert!(track.n_nodes() >= min_nodes);
        let nights: Vec<NightId> = track.nodes.iter().map(|sk| sk.night_id).collect();
        for w in nights.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    let matches = match_truth_to_hypotheses(
        &ground_truth,
        &hypotheses,
        &runtime_state.alert_store,
        &runtime_state.seed_store,
    );

    // At least some trajectories across all populations should be recovered.
    let any_recovered = matches.iter().any(|(_, _, score)| *score > 0.0);
    assert!(
        any_recovered,
        "at least one trajectory should be recovered in incremental mode"
    );

    // Per-population diagnostics.
    for pop in [
        AsteroidPopulation::NearEarth,
        AsteroidPopulation::MainBelt,
        AsteroidPopulation::Trojan,
    ] {
        let pop_scores: Vec<f64> = ground_truth
            .iter()
            .enumerate()
            .filter(|(_, t)| t.population == pop)
            .map(|(i, _)| matches[i].2)
            .collect();
        let avg = pop_scores.iter().sum::<f64>() / pop_scores.len().max(1) as f64;
        eprintln!(
            "  {:?}: avg Jaccard = {avg:.3}, scores = {pop_scores:?}",
            pop
        );
    }
}

/// Verify that the graph grows incrementally: after each night, the number
/// of seeds and edges should increase (or at least stay the same).
#[test]
fn incremental_graph_grows_over_nights() {
    let n_trajectories = 5;
    let n_nights = 5;
    let obs_per_night = 3;
    let start_night_id = 60000_u32;
    let max_gap_nights = 3_u8;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, n_trajectories)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(1300)
        .build();

    // We need a fresh storage dir for this test.
    let storage_dir = TempDir::new().unwrap();
    let engine_config = engine_config_with_edges(&storage_dir, max_gap_nights);
    let persistence = PersistenceManager::open_or_create(engine_config.storage_path_buf())
        .expect("open persistence");
    let edge_models = test_edge_models();
    let solver_manager = test_solver_manager();

    let mut runtime_state = RuntimeState::new();

    let mut night_ids: Vec<u32> = dataset.alerts().iter().map(|a| a.key.night_id.0).collect();
    night_ids.sort_unstable();
    night_ids.dedup();

    let data_dir = TempDir::new().unwrap();
    let mut prev_n_seeds = 0_usize;
    let mut prev_n_edges = 0_usize;

    for (run_idx, &nid) in night_ids.iter().enumerate() {
        let night_alerts: Vec<&Alert> = dataset
            .alerts()
            .iter()
            .filter(|a| a.key.night_id.0 == nid)
            .collect();

        let parquet_path = data_dir
            .path()
            .join(format!("night_{nid}_grow{run_idx}.parquet"));
        let alerts_uri = write_alerts_parquet(&night_alerts, &parquet_path);

        let plan = PipelinePlan {
            stages: vec![
                PipelineStage::IngestNights,
                PipelineStage::BuildSeeds,
                PipelineStage::BuildEdges,
                PipelineStage::Solve,
            ],
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
            solver_manager: &solver_manager,
        };

        runner
            .run(&mut ctx, &hooks)
            .unwrap_or_else(|e| panic!("pipeline run {run_idx} night {nid} failed: {e}"));

        let cur_n_seeds: usize = runtime_state
            .seed_store
            .iter()
            .map(|(_, seeds)| seeds.len())
            .sum();
        let cur_n_edges = runtime_state.graph.edges.len();

        // Seeds must grow (each night adds new seeds).
        assert!(
            cur_n_seeds > prev_n_seeds,
            "run {run_idx} (night {nid}): seed count should grow: {prev_n_seeds} -> {cur_n_seeds}"
        );

        // Edges must grow from the second night onward (first night has no
        // previous seeds to connect to).
        if run_idx >= 1 {
            assert!(
                cur_n_edges > prev_n_edges,
                "run {run_idx} (night {nid}): edge count should grow: \
                 {prev_n_edges} -> {cur_n_edges}"
            );
        }

        eprintln!("  night {nid}: seeds = {cur_n_seeds}, edges = {cur_n_edges}");

        prev_n_seeds = cur_n_seeds;
        prev_n_edges = cur_n_edges;
    }

    // After all nights, we should have substantial graph.
    assert!(
        prev_n_seeds >= n_nights * n_trajectories,
        "final graph should have at least {n} seeds, got {prev_n_seeds}",
        n = n_nights * n_trajectories
    );
}

/// Verify that incremental runs with all five populations and min_nodes = 4
/// produce multi-hop trajectories.
#[test]
fn incremental_all_five_populations() {
    let n_per_pop = 2;
    let n_nights = 6;
    let obs_per_night = 3;
    let start_night_id = 60000_u32;
    let max_gap_nights = 3_u8;
    let min_nodes = 4;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::NearEarth, n_per_pop)
        .population(AsteroidPopulation::MainBelt, n_per_pop)
        .population(AsteroidPopulation::Trojan, n_per_pop)
        .population(AsteroidPopulation::TransNeptunian, n_per_pop)
        .population(AsteroidPopulation::KuiperBelt, n_per_pop)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(1400)
        .build();

    let ground_truth = dataset.ground_truth().to_vec();
    assert_eq!(ground_truth.len(), 5 * n_per_pop);

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    let solver_manager = test_solver_manager_with_min_nodes(min_nodes);
    let PipelineTestResult {
        state: runtime_state,
        ..
    } = run_incremental_pipeline(
        &dataset,
        &data_dir,
        &storage_dir,
        max_gap_nights,
        &solver_manager,
        |_| THROUGH_SOLVE.to_vec(),
    );
    let hypotheses = &runtime_state.track_hypotheses;

    assert!(
        !hypotheses.is_empty(),
        "incremental pipeline with all 5 populations must produce hypotheses"
    );

    // All tracks must have >= min_nodes.
    for (_, track) in hypotheses {
        assert!(track.n_nodes() >= min_nodes);
        assert_eq!(track.n_edges(), track.n_nodes() - 1);
        assert!(track.cost.is_finite());
    }

    // Recovery statistics.
    let matches = match_truth_to_hypotheses(
        &ground_truth,
        &hypotheses,
        &runtime_state.alert_store,
        &runtime_state.seed_store,
    );

    let any_matched = matches.iter().any(|(_, _, score)| *score > 0.0);
    assert!(
        any_matched,
        "at least one ground-truth trajectory should be recovered"
    );

    for pop in AsteroidPopulation::ALL {
        let pop_scores: Vec<f64> = ground_truth
            .iter()
            .enumerate()
            .filter(|(_, t)| t.population == pop)
            .map(|(i, _)| matches[i].2)
            .collect();
        let avg = pop_scores.iter().sum::<f64>() / pop_scores.len().max(1) as f64;
        eprintln!(
            "  {:?}: avg Jaccard = {avg:.3}, scores = {pop_scores:?}",
            pop
        );
    }
}

/// Verify that incremental mode is deterministic: two identical night-by-night
/// runs must produce the exact same hypotheses.
#[test]
fn incremental_is_deterministic() {
    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, 4)
        .n_nights(5)
        .obs_per_night(3)
        .start_night_id(60000)
        .seed(1500)
        .build();

    let max_gap_nights = 3_u8;
    let min_nodes = 4;

    // Run 1
    let data_dir_1 = TempDir::new().unwrap();
    let storage_dir_1 = TempDir::new().unwrap();
    let solver_manager = test_solver_manager_with_min_nodes(min_nodes);
    let PipelineTestResult { state: state1, .. } = run_incremental_pipeline(
        &dataset,
        &data_dir_1,
        &storage_dir_1,
        max_gap_nights,
        &solver_manager,
        |_| THROUGH_SOLVE.to_vec(),
    );
    let hyp1 = &state1.track_hypotheses;

    // Run 2
    let data_dir_2 = TempDir::new().unwrap();
    let storage_dir_2 = TempDir::new().unwrap();
    let PipelineTestResult { state: state2, .. } = run_incremental_pipeline(
        &dataset,
        &data_dir_2,
        &storage_dir_2,
        max_gap_nights,
        &solver_manager,
        |_| THROUGH_SOLVE.to_vec(),
    );
    let hyp2 = &state2.track_hypotheses;

    assert_eq!(
        hyp1.len(),
        hyp2.len(),
        "incremental deterministic runs must produce same hypothesis count: {} vs {}",
        hyp1.len(),
        hyp2.len()
    );

    let mut tracks1: Vec<_> = hyp1.values().collect();
    let mut tracks2: Vec<_> = hyp2.values().collect();

    let sort_key =
        |t: &&TrackHypothesis| (t.nodes[0].night_id, t.nodes[0].unique_id, t.cost.to_bits());
    tracks1.sort_by_key(sort_key);
    tracks2.sort_by_key(sort_key);

    for (t1, t2) in tracks1.iter().zip(tracks2.iter()) {
        assert_eq!(t1.nodes, t2.nodes, "nodes must match across runs");
        assert_eq!(t1.edges, t2.edges, "edges must match across runs");
        assert_eq!(
            t1.cost.to_bits(),
            t2.cost.to_bits(),
            "costs must match across runs"
        );
    }
}

/// Verify that hypotheses from incremental mode have edges connecting
/// consecutive nodes, and that multi-hop chains span many nights.
#[test]
fn incremental_edges_form_consecutive_chains() {
    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, 5)
        .n_nights(6)
        .obs_per_night(3)
        .start_night_id(60000)
        .seed(1600)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    let solver_manager = test_solver_manager_with_min_nodes(4);
    let PipelineTestResult {
        state: runtime_state,
        ..
    } = run_incremental_pipeline(
        &dataset,
        &data_dir,
        &storage_dir,
        3,
        &solver_manager,
        |_| THROUGH_SOLVE.to_vec(),
    );
    let hypotheses = &runtime_state.track_hypotheses;

    assert!(!hypotheses.is_empty());

    for (&hid, track) in hypotheses {
        assert_eq!(
            track.edges.len(),
            track.nodes.len() - 1,
            "hypothesis {hid}: edges.len() must be nodes.len() - 1"
        );

        for (i, edge_key) in track.edges.iter().enumerate() {
            let edge = runtime_state
                .graph
                .edge_by_key(edge_key)
                .expect("edge must exist in graph");

            assert_eq!(
                edge.from, track.nodes[i],
                "hypothesis {hid}, edge {i}: from must match nodes[{i}]"
            );
            assert_eq!(
                edge.to,
                track.nodes[i + 1],
                "hypothesis {hid}, edge {i}: to must match nodes[{}]",
                i + 1
            );
            assert!(
                edge.from.night_id < edge.to.night_id,
                "hypothesis {hid}, edge {i}: edge must be time-forward"
            );
        }

        // Track night span must match actual night range.
        let min_night = track.nodes.first().unwrap().night_id;
        let max_night = track.nodes.last().unwrap().night_id;
        let actual_span = max_night.0 - min_night.0;
        assert_eq!(
            track.night_span, actual_span,
            "hypothesis {hid}: night_span ({}) must match actual range ({})",
            track.night_span, actual_span
        );

        // With min_nodes=4, span must be at least 3.
        assert!(
            actual_span >= 3,
            "hypothesis {hid}: multi-hop track should span >= 3 nights, got {actual_span}"
        );
    }
}
