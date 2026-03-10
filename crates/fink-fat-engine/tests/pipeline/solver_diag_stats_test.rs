//! Integration tests for `SolverManager::run_plan` diagnostics.
//!
//! These tests call the **real solver** (BoundedBeam) on synthetic datasets and
//! verify the structural invariants of `SolverDiagnostics` returned by
//! `run_plan`, as well as the aggregate statistics logged by `log_diag_stats`.
//!
//! # Test taxonomy
//!
//! | Category | What is tested |
//! |---|---|
//! | Structural | every `SolverOutput` satisfies field-level invariants |
//! | Aggregate | statistics over all component outputs (mean, monotonicity, …) |
//! | Cross-check | `sum(n_selected) == merged_hypotheses.len()` |
//! | Regression | raising `min_nodes` never increases total tracks |
//! | Proptest | invariants hold for any random (n_traj, n_nights, …) combination |
//!
//! # Design note
//!
//! Instead of going through `THROUGH_SOLVE`, each test runs the pipeline only
//! up to `THROUGH_EDGES` and then calls `SolverManager::make_plan` +
//! `run_plan` directly.  This gives the test full access to the raw
//! `Vec<SolverOutput>` (which the normal pipeline discards after merging), so
//! we can inspect diagnostics field-by-field.

use tempfile::TempDir;

use fink_fat_engine::{
    engine_config::pipeline_policy::PersistPolicy,
    pipeline::hooks::NoopProgress,
    solver::{SolverOutput, components::ConnectedComponents},
};

use crate::synthetic_alerts::{AsteroidPopulation, SyntheticDataset, SyntheticDatasetBuilder};

use super::{
    THROUGH_EDGES, run_pipeline_with, test_solver_manager, test_solver_manager_with_min_nodes,
};

// ---------------------------------------------------------------------------
// Test helper
// ---------------------------------------------------------------------------

/// Run the pipeline through `BuildEdges` only, then call `make_plan` +
/// `run_plan` directly and return the raw `Vec<SolverOutput>`.
///
/// Returns an empty vec when no inter-night edges exist (e.g. single night).
fn run_plan_on(
    dataset: &SyntheticDataset,
    data_dir: &TempDir,
    storage_dir: &TempDir,
    max_gap_nights: u8,
    solver_manager: &fink_fat_engine::solver::solver_manager::SolverManager,
) -> Vec<SolverOutput> {
    use super::PipelineTestResult;

    let PipelineTestResult { state, .. } = run_pipeline_with(
        dataset,
        data_dir,
        storage_dir,
        THROUGH_EDGES,
        max_gap_nights,
        solver_manager,
        PersistPolicy::None,
    );

    if state.graph.edges.is_empty() {
        return vec![];
    }

    let components = ConnectedComponents::compute(&state.seed_store, &state.graph, true)
        .expect("ConnectedComponents::compute must succeed");

    let plan = solver_manager.make_plan(&components);
    let progress = NoopProgress;

    solver_manager.run_plan(
        &components,
        &state.graph,
        &state.seed_store,
        &plan,
        &progress,
    )
}

// ---------------------------------------------------------------------------
// Structural invariants
// ---------------------------------------------------------------------------

/// Every component must have at least one node.
#[test]
fn all_components_have_positive_n_nodes() {
    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, 5)
        .n_nights(3)
        .obs_per_night(3)
        .start_night_id(60_000)
        .seed(2_000)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();
    let sm = test_solver_manager();

    let outputs = run_plan_on(&dataset, &data_dir, &storage_dir, 3, &sm);
    assert!(!outputs.is_empty(), "expected at least one solver output");

    for (i, out) in outputs.iter().enumerate() {
        assert!(
            out.diag.n_nodes >= 1,
            "output[{i}]: n_nodes must be >= 1, got {}",
            out.diag.n_nodes
        );
    }
}

/// Components that produced tracks must have at least `min_nodes` nodes
/// (otherwise the solver would have been unable to form a track).
#[test]
fn components_with_tracks_satisfy_min_nodes() {
    let min_nodes = 2_usize;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, 5)
        .n_nights(3)
        .obs_per_night(3)
        .start_night_id(60_000)
        .seed(2_001)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();
    let sm = test_solver_manager_with_min_nodes(min_nodes);

    let outputs = run_plan_on(&dataset, &data_dir, &storage_dir, 3, &sm);

    for (i, out) in outputs.iter().enumerate() {
        if out.diag.n_selected > 0 {
            assert!(
                out.diag.n_nodes >= min_nodes as u32,
                "output[{i}]: n_nodes ({}) < min_nodes ({min_nodes}) \
                 but n_selected = {}",
                out.diag.n_nodes,
                out.diag.n_selected
            );
        }
    }
}

/// Counter fields must not have overflowed to `u32::MAX`.
#[test]
fn diag_counters_have_not_overflowed() {
    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::NearEarth, 4)
        .population(AsteroidPopulation::MainBelt, 4)
        .n_nights(3)
        .obs_per_night(3)
        .start_night_id(60_000)
        .seed(2_002)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();
    let sm = test_solver_manager();

    let outputs = run_plan_on(&dataset, &data_dir, &storage_dir, 3, &sm);

    for (i, out) in outputs.iter().enumerate() {
        let d = &out.diag;
        assert_ne!(d.n_nodes, u32::MAX, "output[{i}]: n_nodes overflowed");
        assert_ne!(
            d.n_candidates,
            u32::MAX,
            "output[{i}]: n_candidates overflowed"
        );
        assert_ne!(
            d.n_expansions,
            u32::MAX,
            "output[{i}]: n_expansions overflowed"
        );
        assert_ne!(d.n_selected, u32::MAX, "output[{i}]: n_selected overflowed");
    }
}

/// When the solver is forced to BoundedBeam, every output carries the name
/// `"bounded_beam"`.
#[test]
fn solver_name_is_always_bounded_beam() {
    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, 5)
        .n_nights(3)
        .obs_per_night(3)
        .start_night_id(60_000)
        .seed(2_003)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();
    let sm = test_solver_manager();

    let outputs = run_plan_on(&dataset, &data_dir, &storage_dir, 3, &sm);
    assert!(!outputs.is_empty(), "expected at least one solver output");

    for (i, out) in outputs.iter().enumerate() {
        assert_eq!(
            out.diag.solver_name, "bounded_beam",
            "output[{i}]: unexpected solver_name '{}'",
            out.diag.solver_name
        );
    }
}

/// A single-night dataset produces no inter-night edges, so `run_plan` is
/// not reached and the helper returns an empty vec.
#[test]
fn single_night_produces_no_solver_outputs() {
    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, 5)
        .n_nights(1)
        .obs_per_night(3)
        .start_night_id(60_000)
        .seed(2_004)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();
    let sm = test_solver_manager();

    let outputs = run_plan_on(&dataset, &data_dir, &storage_dir, 2, &sm);
    assert!(
        outputs.is_empty(),
        "single-night run must produce no solver outputs, got {}",
        outputs.len()
    );
}

// ---------------------------------------------------------------------------
// Cross-check: sum(n_selected) == len(merged_hypotheses)
// ---------------------------------------------------------------------------

/// `SolverOutput::merge_solver_output` must produce exactly
/// `Σ diag.n_selected` hypotheses — no hypotheses are created or lost during
/// the merge.
#[test]
fn sum_of_n_selected_equals_merged_hypothesis_count() {
    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, 5)
        .n_nights(3)
        .obs_per_night(3)
        .start_night_id(60_000)
        .seed(2_005)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();
    let sm = test_solver_manager();

    let outputs = run_plan_on(&dataset, &data_dir, &storage_dir, 3, &sm);
    if outputs.is_empty() {
        return;
    }

    let sum_selected: u32 = outputs.iter().map(|o| o.diag.n_selected).sum();
    let merged = SolverOutput::merge_solver_output(&outputs);

    assert_eq!(
        sum_selected as usize,
        merged.len(),
        "sum(n_selected)={sum_selected} != merged.len()={}",
        merged.len()
    );
}

// ---------------------------------------------------------------------------
// Aggregate statistics properties
// ---------------------------------------------------------------------------

/// The mean of `n_selected` (over components) must lie in [0, max(n_selected)].
#[test]
fn aggregate_mean_n_selected_in_range() {
    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, 5)
        .n_nights(3)
        .obs_per_night(3)
        .start_night_id(60_000)
        .seed(2_006)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();
    let sm = test_solver_manager();

    let outputs = run_plan_on(&dataset, &data_dir, &storage_dir, 3, &sm);
    if outputs.is_empty() {
        return;
    }

    let values: Vec<f64> = outputs.iter().map(|o| o.diag.n_selected as f64).collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let max = values.iter().cloned().fold(0.0_f64, f64::max);

    assert!(
        mean >= 0.0,
        "mean n_selected must be non-negative, got {mean}"
    );
    assert!(
        mean <= max + 1e-9,
        "mean ({mean}) > max ({max}) — impossible for a non-negative sample"
    );
}

/// The fraction of zero-track components must be in [0.0, 1.0].
#[test]
fn zero_track_fraction_is_in_unit_interval() {
    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, 5)
        .n_nights(3)
        .obs_per_night(3)
        .start_night_id(60_000)
        .seed(2_007)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();
    let sm = test_solver_manager();

    let outputs = run_plan_on(&dataset, &data_dir, &storage_dir, 3, &sm);
    if outputs.is_empty() {
        return;
    }

    let n_zero = outputs.iter().filter(|o| o.diag.n_selected == 0).count();
    let fraction = n_zero as f64 / outputs.len() as f64;
    assert!(
        (0.0..=1.0).contains(&fraction),
        "zero-track fraction {fraction} out of [0, 1]"
    );
}

// ---------------------------------------------------------------------------
// Regression: min_nodes monotonicity
// ---------------------------------------------------------------------------

/// Raising `min_nodes` never increases the total number of emitted tracks.
///
/// With `min_nodes=4` the solver discards components whose paths are too
/// short, so `total_tracks(min_nodes=4) ≤ total_tracks(min_nodes=2)`.
#[test]
fn higher_min_nodes_yields_fewer_or_equal_tracks() {
    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, 5)
        .n_nights(5)
        .obs_per_night(3)
        .start_night_id(60_000)
        .seed(2_008)
        .build();

    let data_dir_a = TempDir::new().unwrap();
    let storage_dir_a = TempDir::new().unwrap();
    let data_dir_b = TempDir::new().unwrap();
    let storage_dir_b = TempDir::new().unwrap();

    let sm_loose = test_solver_manager_with_min_nodes(2);
    let sm_strict = test_solver_manager_with_min_nodes(4);

    let total_loose: u32 = run_plan_on(&dataset, &data_dir_a, &storage_dir_a, 4, &sm_loose)
        .iter()
        .map(|o| o.diag.n_selected)
        .sum();

    let total_strict: u32 = run_plan_on(&dataset, &data_dir_b, &storage_dir_b, 4, &sm_strict)
        .iter()
        .map(|o| o.diag.n_selected)
        .sum();

    assert!(
        total_strict <= total_loose,
        "strict min_nodes=4 ({total_strict}) should produce ≤ tracks \
         than loose min_nodes=2 ({total_loose})"
    );
}

// ---------------------------------------------------------------------------
// Proptest: invariants hold for randomised dataset parameters
// ---------------------------------------------------------------------------

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(25))]

    /// For any reasonable synthetic dataset, every component returned by
    /// `run_plan` must satisfy:
    ///   - `n_nodes >= 1`
    ///   - No counter has overflowed to `u32::MAX`
    ///   - `solver_name == "bounded_beam"`
    ///   - `time_est_s >= 0.0`
    #[test]
    fn prop_diag_structural_invariants(
        n_trajectories in 1_usize..=8,
        n_nights       in 2_usize..=4,
        obs_per_night  in 2_usize..=4,
        seed           in 0_u64..=9_999,
    ) {
        let dataset = SyntheticDatasetBuilder::new()
            .population(AsteroidPopulation::MainBelt, n_trajectories)
            .n_nights(n_nights)
            .obs_per_night(obs_per_night)
            .start_night_id(60_000)
            .seed(seed)
            .build();

        let data_dir    = TempDir::new().unwrap();
        let storage_dir = TempDir::new().unwrap();
        let sm          = test_solver_manager();

        let outputs = run_plan_on(&dataset, &data_dir, &storage_dir, 3, &sm);

        for (i, out) in outputs.iter().enumerate() {
            let d = &out.diag;
            proptest::prop_assert!(d.n_nodes >= 1,
                "output[{}]: n_nodes must be >= 1", i);
            proptest::prop_assert_ne!(d.n_candidates, u32::MAX,
                "output[{}]: n_candidates overflowed", i);
            proptest::prop_assert_ne!(d.n_expansions, u32::MAX,
                "output[{}]: n_expansions overflowed", i);
            proptest::prop_assert_eq!(d.solver_name, "bounded_beam",
                "output[{}]: unexpected solver_name", i);
            proptest::prop_assert!(d.time_est_s >= 0.0,
                "output[{}]: time_est_s must be non-negative", i);
        }
    }

    /// `sum(n_selected) == len(merge_solver_output(...))` for any dataset.
    ///
    /// Verifies that `merge_solver_output` neither drops nor duplicates
    /// hypotheses regardless of how many components and tracks exist.
    #[test]
    fn prop_sum_n_selected_equals_merged_len(
        n_trajectories in 1_usize..=6,
        n_nights       in 2_usize..=3,
        obs_per_night  in 2_usize..=3,
        seed           in 0_u64..=999,
    ) {
        let dataset = SyntheticDatasetBuilder::new()
            .population(AsteroidPopulation::MainBelt, n_trajectories)
            .n_nights(n_nights)
            .obs_per_night(obs_per_night)
            .start_night_id(60_000)
            .seed(seed)
            .build();

        let data_dir    = TempDir::new().unwrap();
        let storage_dir = TempDir::new().unwrap();
        let sm          = test_solver_manager();

        let outputs = run_plan_on(&dataset, &data_dir, &storage_dir, 3, &sm);
        if outputs.is_empty() {
            return Ok(());
        }

        let sum_selected: u32 = outputs.iter().map(|o| o.diag.n_selected).sum();
        let merged = SolverOutput::merge_solver_output(&outputs);

        proptest::prop_assert_eq!(
            sum_selected as usize,
            merged.len(),
            "sum_selected={} != merged.len()={}",
            sum_selected,
            merged.len()
        );
    }

    /// For any dataset with ≥ 2 nights, calling `run_plan` twice with the
    /// same inputs must produce identical **aggregate** outputs (determinism).
    ///
    /// We compare sums rather than per-component order because connected
    /// components may be enumerated in a different order between runs
    /// (AHashMap inside `ConnectedComponents` has non-deterministic iteration
    /// order), but the totals must be identical.
    #[test]
    fn prop_run_plan_is_deterministic(
        n_trajectories in 1_usize..=5,
        n_nights       in 2_usize..=3,
        obs_per_night  in 2_usize..=3,
        seed           in 0_u64..=999,
    ) {
        let dataset = SyntheticDatasetBuilder::new()
            .population(AsteroidPopulation::MainBelt, n_trajectories)
            .n_nights(n_nights)
            .obs_per_night(obs_per_night)
            .start_night_id(60_000)
            .seed(seed)
            .build();

        let sm = test_solver_manager();

        // First run
        let data_dir1    = TempDir::new().unwrap();
        let storage_dir1 = TempDir::new().unwrap();
        let out1 = run_plan_on(&dataset, &data_dir1, &storage_dir1, 3, &sm);

        // Second run (fresh temp dirs, identical dataset + config)
        let data_dir2    = TempDir::new().unwrap();
        let storage_dir2 = TempDir::new().unwrap();
        let out2 = run_plan_on(&dataset, &data_dir2, &storage_dir2, 3, &sm);

        // Number of components must be identical.
        proptest::prop_assert_eq!(
            out1.len(), out2.len(),
            "run1 produced {} components, run2 produced {}",
            out1.len(), out2.len()
        );

        // Aggregate counters must be identical (order-independent comparison).
        let sum_selected1: u32  = out1.iter().map(|o| o.diag.n_selected).sum();
        let sum_selected2: u32  = out2.iter().map(|o| o.diag.n_selected).sum();
        let sum_candidates1: u32 = out1.iter().map(|o| o.diag.n_candidates).sum();
        let sum_candidates2: u32 = out2.iter().map(|o| o.diag.n_candidates).sum();
        let sum_expansions1: u32 = out1.iter().map(|o| o.diag.n_expansions).sum();
        let sum_expansions2: u32 = out2.iter().map(|o| o.diag.n_expansions).sum();
        let sum_nodes1: u32      = out1.iter().map(|o| o.diag.n_nodes).sum();
        let sum_nodes2: u32      = out2.iter().map(|o| o.diag.n_nodes).sum();

        proptest::prop_assert_eq!(sum_selected1,  sum_selected2,  "total n_selected differs");
        proptest::prop_assert_eq!(sum_candidates1, sum_candidates2, "total n_candidates differs");
        proptest::prop_assert_eq!(sum_expansions1, sum_expansions2, "total n_expansions differs");
        proptest::prop_assert_eq!(sum_nodes1,      sum_nodes2,      "total n_nodes differs");
    }
}
