//! Integration tests for post-`FitOrbit` edge deactivation.
//!
//! These tests verify the invariant:
//!
//! > Once a trajectory receives a confirmed orbit, every edge that belongs to
//! > it is marked `active = false` in the [`AlertLinkageDAG`], and a
//! > subsequent [`Solve`] pass ignores those edges entirely.
//!
//! ## Mechanism
//!
//! At the end of the [`FitOrbit`] stage, [`AlertLinkageDAG::deactivate_edges`]
//! is called with the edge keys collected from every hypothesis whose orbit
//! fit succeeded. Because the [`Solve`] stage builds its connected components
//! with `active_only = true` (via
//! [`ConnectedComponents::compute`](fink_fat_engine::solver::components::ConnectedComponents::compute)),
//! deactivated edges are excluded from the next solver pass and can no longer
//! contribute to new trajectory hypotheses.
//!
//! ## Tests
//!
//! | Test | What it checks |
//! |------|----------------|
//! | [`fitted_trajectory_edges_are_deactivated`] | All edges of successfully fitted hypotheses have `active = false` after `FitOrbit`. |
//! | [`subsequent_solve_excludes_deactivated_edges`] | Running `Solve` again on the same runtime state never produces a hypothesis that contains a deactivated edge key. |

use std::collections::HashSet;

use outfit::ObjectNumber;
use tempfile::TempDir;

use fink_fat_engine::{
    engine_config::pipeline_policy::PersistPolicy,
    graph::edge::EdgeKey,
    persistence::{PersistenceManager, runtime_state::RuntimeState},
    pipeline::{
        PipelineContext, PipelineInputs, PipelinePlan, PipelineRunner, stages::PipelineStage,
    },
};

use super::{
    NoopHooks, PipelineTestResult, THROUGH_ORBIT, dummy_input_uri, run_pipeline, test_edge_models,
    test_solver_manager,
};
use crate::synthetic_alerts::{AsteroidPopulation, SyntheticDatasetBuilder};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Collect every [`EdgeKey`] that belongs to a successfully fitted trajectory.
///
/// Iterates `orbit_results` for `Ok` entries, looks up the corresponding
/// [`TrackHypothesis`](fink_fat_engine::trajectory::TrackHypothesis) in
/// `track_hypotheses`, and returns the union of all their edge-key sets.
fn collect_fitted_edge_keys(state: &RuntimeState) -> HashSet<EdgeKey> {
    let mut keys = HashSet::new();
    for (obj, result) in &state.orbit_results {
        if result.is_err() {
            continue;
        }
        let ObjectNumber::Int(hyp_id) = obj else {
            continue;
        };
        if let Some(track) = state.track_hypotheses.get(hyp_id) {
            keys.extend(track.edges.iter().copied());
        }
    }
    keys
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Verify that every edge belonging to a successfully fitted trajectory has
/// `active = false` in the graph after the `FitOrbit` stage completes.
///
/// The test also asserts that at least one successful orbit fit occurred;
/// if no fit succeeded the deactivation logic is never exercised and the
/// test would be vacuous.
#[test]
fn fitted_trajectory_edges_are_deactivated() {
    let n_trajectories = 6;
    let n_nights = 4;
    let obs_per_night = 3;
    let start_night_id = 60000_u32;
    let max_gap_nights = 4_u8;

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
        state: runtime_state,
        ..
    } = run_pipeline(
        &dataset,
        &data_dir,
        &storage_dir,
        THROUGH_ORBIT,
        max_gap_nights,
    );

    // ---- Precondition: at least one orbit fit must have succeeded. ----
    let n_successful = runtime_state
        .orbit_results
        .values()
        .filter(|r| r.is_ok())
        .count();
    assert!(
        n_successful > 0,
        "precondition: at least one orbit fit should succeed; got 0 / {}",
        runtime_state.orbit_results.len(),
    );

    eprintln!(
        "[fitted_trajectory_edges_are_deactivated] {n_successful} / {} orbits succeeded",
        runtime_state.orbit_results.len(),
    );

    // ---- For each successful fit, every edge in that hypothesis must be
    //      inactive in the graph. ----
    let mut n_checked = 0_usize;
    for (obj, result) in &runtime_state.orbit_results {
        if result.is_err() {
            continue;
        }
        let ObjectNumber::Int(hyp_id) = obj else {
            continue;
        };
        let track = runtime_state
            .track_hypotheses
            .get(hyp_id)
            .unwrap_or_else(|| panic!("hypothesis {hyp_id} not found in track_hypotheses"));

        for edge_key in &track.edges {
            let edge = runtime_state
                .graph
                .edge_by_key(edge_key)
                .unwrap_or_else(|| {
                    panic!("edge {edge_key:?} referenced by hypothesis {hyp_id} not in graph")
                });
            assert!(
                !edge.active,
                "edge {:?} → {:?} (hypothesis {hyp_id}) should be inactive after FitOrbit \
                 but has active=true",
                edge_key.from, edge_key.to,
            );
            n_checked += 1;
        }
    }

    eprintln!(
        "[fitted_trajectory_edges_are_deactivated] verified active=false on {n_checked} edges",
    );
    assert!(
        n_checked > 0,
        "at least one edge should have been checked; \
         successful fits exist but hypotheses have no edges?"
    );
}

/// Verify that edges belonging to trajectories with failed orbit fits are
/// **not** deactivated — they must remain active for potential retry on
/// a subsequent night.
#[test]
fn failed_fit_edges_remain_active() {
    let n_trajectories = 8;
    let n_nights = 4;
    let obs_per_night = 3;
    let start_night_id = 60000_u32;
    let max_gap_nights = 4_u8;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, n_trajectories)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(7)
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
        THROUGH_ORBIT,
        max_gap_nights,
    );

    let n_failed = runtime_state
        .orbit_results
        .values()
        .filter(|r| r.is_err())
        .count();

    if n_failed == 0 {
        eprintln!("[failed_fit_edges_remain_active] all fits succeeded — skipping check");
        return;
    }

    eprintln!("[failed_fit_edges_remain_active] {n_failed} failed fits to check",);

    // Collect edge keys of failed-fit hypotheses.
    let mut n_checked = 0_usize;
    for (obj, result) in &runtime_state.orbit_results {
        if result.is_ok() {
            continue;
        }
        let ObjectNumber::Int(hyp_id) = obj else {
            continue;
        };
        let Some(track) = runtime_state.track_hypotheses.get(hyp_id) else {
            continue;
        };
        for edge_key in &track.edges {
            if let Some(edge) = runtime_state.graph.edge_by_key(edge_key) {
                assert!(
                    edge.active,
                    "edge {:?} → {:?} (failed-fit hypothesis {hyp_id}) should remain active",
                    edge_key.from, edge_key.to,
                );
                n_checked += 1;
            }
        }
    }

    eprintln!(
        "[failed_fit_edges_remain_active] verified active=true on {n_checked} edges of failed fits",
    );
}

/// Verify that a `Solve` pass executed immediately after `FitOrbit` never
/// produces a hypothesis whose edge set intersects the deactivated edge keys.
///
/// Because the solver builds connected components with
/// `active_only = true`, deactivated edges should be invisible to it.
/// This test makes that invariant observable at the integration level.
#[test]
fn subsequent_solve_excludes_deactivated_edges() {
    let n_trajectories = 6;
    let n_nights = 4;
    let obs_per_night = 3;
    let start_night_id = 60000_u32;
    let max_gap_nights = 4_u8;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, n_trajectories)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(42)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    // ---- Run the full five-stage pipeline. ----
    let PipelineTestResult {
        mut state,
        engine_config,
        ..
    } = run_pipeline(
        &dataset,
        &data_dir,
        &storage_dir,
        THROUGH_ORBIT,
        max_gap_nights,
    );

    // ---- Precondition: at least one successful fit (otherwise deactivated
    //      set is empty and the test is vacuous). ----
    let deactivated_keys = collect_fitted_edge_keys(&state);
    if deactivated_keys.is_empty() {
        eprintln!(
            "[subsequent_solve_excludes_deactivated_edges] no successful orbit fits; \
             deactivated set is empty — test vacuous, skipping assertion"
        );
        return;
    }

    eprintln!(
        "[subsequent_solve_excludes_deactivated_edges] \
         {} deactivated edge keys from {} successful fits",
        deactivated_keys.len(),
        state.orbit_results.values().filter(|r| r.is_ok()).count(),
    );

    // ---- Re-run the Solve stage on the same runtime state. ----
    //
    // No new alerts are ingested: the stage reads the already-accumulated
    // graph (where fitted edges are inactive) and re-runs the solver.
    // The hypotheses produced must not contain any deactivated edge key.
    let persistence = PersistenceManager::open_or_create(engine_config.storage_path_buf())
        .expect("open persistence");
    let edge_models = test_edge_models();
    let solver_manager = test_solver_manager();

    let solve_plan = PipelinePlan {
        stages: vec![PipelineStage::Solve],
        persist: PersistPolicy::None,
        inputs: PipelineInputs {
            alerts_uri: dummy_input_uri(),
        },
    };

    let runner = PipelineRunner {
        plan: solve_plan.clone(),
    };
    let hooks = NoopHooks;

    {
        let mut ctx = PipelineContext {
            plan: &solve_plan,
            persistence: &persistence,
            runtime_state: &mut state,
            engine_config: &engine_config,
            edge_models: &edge_models,
            solver_manager: &solver_manager,
        };

        runner
            .run(&mut ctx, &hooks)
            .expect("second Solve pass should not fail");
    }

    // ---- Verify: no new hypothesis uses a deactivated edge key. ----
    let new_hypotheses = &state.track_hypotheses;

    eprintln!(
        "[subsequent_solve_excludes_deactivated_edges] \
         second Solve produced {} hypotheses",
        new_hypotheses.len(),
    );

    for (&hyp_id, track) in new_hypotheses {
        for edge_key in &track.edges {
            assert!(
                !deactivated_keys.contains(edge_key),
                "hypothesis {hyp_id} references deactivated edge \
                 {:?} → {:?} after a re-Solve pass",
                edge_key.from,
                edge_key.to,
            );
        }
    }
}
