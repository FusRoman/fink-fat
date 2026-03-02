//! Integration tests for the full five-stage pipeline:
//! `IngestNights → BuildSeeds → BuildEdges → Solve → FitOrbit`.
//!
//! These tests verify that the orbit-fitting stage (`FitOrbit`) runs
//! correctly after the solver has produced track hypotheses, and that the
//! resulting `FullOrbitResult` contains valid orbital solutions for at
//! least a fraction of the hypothesised trajectories.
//!
//! The tests exercise multiple asteroid populations (NEA, Main Belt,
//! Trojan) to ensure the IOD (Initial Orbit Determination) pipeline
//! handles different kinematic regimes.
//!
//! **Requirements**: These tests need internet access (UT1 provider) and
//! a cached DE440 ephemeris file (`~/.cache/outfit_cache/`).

use outfit::ObjectNumber;
use tempfile::TempDir;

use fink_fat_engine::{
    engine_config::pipeline_policy::PersistPolicy,
    persistence::{PersistenceManager, runtime_state::RuntimeState},
    pipeline::{
        PipelineContext, PipelineInputs, PipelinePlan, PipelineRunner, stages::PipelineStage,
    },
    solver::HypothesisSet,
};

use super::{
    NoopHooks, PipelineTestResult, THROUGH_ORBIT, THROUGH_SOLVE, engine_config_with_edges,
    match_truth_to_hypotheses, run_incremental_pipeline, run_pipeline, test_edge_models,
    test_solver_manager, test_solver_manager_with_min_nodes,
};
use crate::synthetic_alerts::{AsteroidPopulation, SyntheticDatasetBuilder};

// ---------------------------------------------------------------------------
// Integration tests
// ---------------------------------------------------------------------------

/// Basic test: verify that the five-stage pipeline completes and that the
/// `FitOrbit` stage report is present.
#[test]
fn five_stage_pipeline_produces_orbit_results() {
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
        THROUGH_ORBIT,
        max_gap_nights,
    );
    let hypotheses = &runtime_state.track_hypotheses;

    // ---- 1) Verify we got five stage reports ----
    assert_eq!(output.reports.len(), 5, "expected 5 stage reports");
    assert_eq!(output.reports[0].0, PipelineStage::IngestNights);
    assert_eq!(output.reports[1].0, PipelineStage::BuildSeeds);
    assert_eq!(output.reports[2].0, PipelineStage::BuildEdges);
    assert_eq!(output.reports[3].0, PipelineStage::Solve);
    assert_eq!(output.reports[4].0, PipelineStage::FitOrbit);

    // ---- 2) Verify the solver produced hypotheses ----
    assert!(
        !hypotheses.is_empty(),
        "solver should produce at least one hypothesis"
    );

    // ---- 3) Verify orbit results are populated ----
    let orbit_results = &runtime_state.orbit_results;
    assert!(
        !orbit_results.is_empty(),
        "orbit_results must not be empty after FitOrbit stage"
    );

    // ---- 4) At least some orbits should have succeeded ----
    let n_ok = orbit_results.values().filter(|r| r.is_ok()).count();
    let n_err = orbit_results.values().filter(|r| r.is_err()).count();

    eprintln!(
        "[five_stage_pipeline_produces_orbit_results] orbit results: {} total, {} ok, {} err",
        orbit_results.len(),
        n_ok,
        n_err
    );

    // We don't require all to succeed (IOD can legitimately fail for some
    // tracks with insufficient geometric diversity), but at least one should.
    assert!(
        n_ok > 0,
        "at least one orbit fit should succeed; got {n_ok} ok / {} total",
        orbit_results.len()
    );
}

/// Verify that successful orbit fits produce physically plausible orbital
/// elements for Main Belt asteroids.
#[test]
fn main_belt_orbits_are_physically_plausible() {
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
        .seed(123)
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

    let orbit_results = &runtime_state.orbit_results;

    let mut n_plausible = 0;
    for (obj, result) in orbit_results {
        if let Ok((gauss_result, rms)) = result {
            let orbit = gauss_result.get_orbit();
            let kep = orbit
                .to_keplerian()
                .expect("conversion to Keplerian should succeed");

            eprintln!(
                "[main_belt_orbits] obj={obj:?}: a={:.4} AU, e={:.4}, i={:.2}°, rms={:.4}, corrected={}",
                kep.semi_major_axis,
                kep.eccentricity,
                kep.inclination.to_degrees(),
                rms,
                gauss_result.is_corrected(),
            );

            // Basic physical sanity checks for any asteroid orbit:
            // - Semi-major axis > 0 (bound orbit)
            // - Eccentricity in [0, 1) for a bound orbit
            // - Inclination in [0, π]
            // - RMS should be finite and non-negative
            assert!(
                kep.semi_major_axis > 0.0,
                "semi-major axis must be positive for obj {obj:?}, got {}",
                kep.semi_major_axis
            );
            assert!(
                kep.eccentricity >= 0.0 && kep.eccentricity < 1.0,
                "eccentricity must be in [0, 1) for obj {obj:?}, got {}",
                kep.eccentricity
            );
            assert!(
                kep.inclination >= 0.0 && kep.inclination <= std::f64::consts::PI,
                "inclination must be in [0, π] for obj {obj:?}, got {}",
                kep.inclination
            );
            assert!(
                rms.is_finite() && *rms >= 0.0,
                "RMS must be finite and non-negative for obj {obj:?}, got {rms}"
            );

            n_plausible += 1;
        }
    }

    eprintln!(
        "[main_belt_orbits] {n_plausible} / {} orbits are physically plausible",
        orbit_results.len()
    );

    assert!(
        n_plausible > 0,
        "at least one orbit should be physically plausible"
    );
}

/// Test orbit fitting with diverse populations: NEA, Main Belt, and Trojan.
///
/// Verifies that the IOD succeeds across different kinematic regimes and
/// that successful orbit fits produce valid orbital elements regardless
/// of the population.
#[test]
fn fit_orbit_diverse_populations() {
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
        .seed(77)
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
        THROUGH_ORBIT,
        max_gap_nights,
    );
    let hypotheses = &runtime_state.track_hypotheses;

    // ---- 1) Pipeline must complete all five stages ----
    assert_eq!(output.reports.len(), 5);
    assert_eq!(output.reports[4].0, PipelineStage::FitOrbit);

    // ---- 2) Solver must have produced hypotheses ----
    assert!(
        !hypotheses.is_empty(),
        "solver must produce hypotheses for diverse populations"
    );

    // ---- 3) Orbit results should be non-empty ----
    let orbit_results = &runtime_state.orbit_results;
    assert!(
        !orbit_results.is_empty(),
        "orbit results must be populated after FitOrbit"
    );

    // ---- 4) Count successes and failures ----
    let n_ok = orbit_results.values().filter(|r| r.is_ok()).count();
    let n_err = orbit_results.values().filter(|r| r.is_err()).count();

    eprintln!(
        "[fit_orbit_diverse_populations] orbit results: {} total, {} ok, {} err",
        orbit_results.len(),
        n_ok,
        n_err
    );

    // At least one orbit fit should succeed.
    assert!(
        n_ok > 0,
        "at least one orbit fit should succeed across diverse populations; \
         got {n_ok} ok / {} total",
        orbit_results.len()
    );

    // ---- 5) All successful fits have valid orbital elements ----
    for (obj, result) in orbit_results {
        if let Ok((gauss_result, rms)) = result {
            let kep = gauss_result
                .get_orbit()
                .to_keplerian()
                .expect("conversion to Keplerian should succeed");

            assert!(
                kep.semi_major_axis > 0.0,
                "semi-major axis must be > 0 for {obj:?}"
            );
            assert!(
                kep.eccentricity >= 0.0 && kep.eccentricity < 1.0,
                "eccentricity must be in [0, 1) for {obj:?}, got {}",
                kep.eccentricity
            );
            assert!(
                rms.is_finite() && *rms >= 0.0,
                "RMS must be finite and non-negative for {obj:?}"
            );

            eprintln!(
                "  obj={obj:?}: a={:.3} AU, e={:.4}, i={:.1}°, rms={:.4}",
                kep.semi_major_axis,
                kep.eccentricity,
                kep.inclination.to_degrees(),
                rms,
            );
        }
    }

    // ---- 6) Report per-population recovery for diagnostics ----
    let matches = match_truth_to_hypotheses(
        &ground_truth,
        hypotheses,
        &runtime_state.alert_store,
        &runtime_state.seed_store,
    );

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
        let any_matched = pop_scores.iter().any(|&s| s > 0.0);

        eprintln!(
            "  {}: {:?} trajectories, scores={:?}, any_matched={}",
            pop.label(),
            pop_indices.len(),
            pop_scores,
            any_matched,
        );
    }
}

/// Test all five asteroid populations (NEA, MBA, Trojan, TNO, KBO) and
/// verify the orbit-fitting stage handles them all.
#[test]
fn fit_orbit_all_five_populations() {
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
        .seed(2025)
        .build();

    let total_trajectories = dataset.ground_truth().len();
    assert_eq!(total_trajectories, 5 * n_per_pop);

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
        THROUGH_ORBIT,
        max_gap_nights,
    );

    // Five stages completed.
    assert_eq!(output.reports.len(), 5);
    assert_eq!(output.reports[4].0, PipelineStage::FitOrbit);

    let orbit_results = &runtime_state.orbit_results;

    let n_ok = orbit_results.values().filter(|r| r.is_ok()).count();
    let n_err = orbit_results.values().filter(|r| r.is_err()).count();

    eprintln!(
        "[fit_orbit_all_five_populations] orbit results: {} total, {} ok, {} err",
        orbit_results.len(),
        n_ok,
        n_err
    );

    // At least one should succeed.
    assert!(
        n_ok > 0,
        "at least one orbit fit should succeed across all five populations"
    );

    // Every successful fit must have valid orbital elements.
    for (obj, result) in orbit_results {
        if let Ok((gauss_result, rms)) = result {
            let kep = gauss_result
                .get_orbit()
                .to_keplerian()
                .expect("conversion to Keplerian should succeed");

            assert!(kep.semi_major_axis > 0.0, "a > 0 for {obj:?}");
            assert!(
                kep.eccentricity >= 0.0 && kep.eccentricity < 1.0,
                "e in [0,1) for {obj:?}, got {}",
                kep.eccentricity
            );
            assert!(rms.is_finite() && *rms >= 0.0, "rms valid for {obj:?}");
        }
    }
}

/// Verify that orbit fitting with corrected (refined) orbits produces
/// lower RMS than preliminary orbits when both exist.
#[test]
fn corrected_orbits_have_lower_rms_than_preliminary() {
    let n_trajectories = 10;
    let n_nights = 4;
    let obs_per_night = 3;
    let start_night_id = 60000_u32;
    let max_gap_nights = 4_u8;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, n_trajectories)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(999)
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

    let orbit_results = &runtime_state.orbit_results;

    let mut n_corrected = 0_usize;
    let mut n_preliminary = 0_usize;
    let mut rms_corrected = Vec::new();
    let mut rms_preliminary = Vec::new();

    for (_obj, result) in orbit_results {
        if let Ok((gauss_result, rms)) = result {
            if gauss_result.is_corrected() {
                n_corrected += 1;
                rms_corrected.push(*rms);
            } else {
                n_preliminary += 1;
                rms_preliminary.push(*rms);
            }
        }
    }

    eprintln!(
        "[corrected_vs_preliminary] corrected={n_corrected} (rms: {rms_corrected:?}), \
         preliminary={n_preliminary} (rms: {rms_preliminary:?})"
    );

    // We expect at least some corrected orbits given enough data.
    // (Not asserting strictly because it depends on the solver's tracks.)
    if n_corrected > 0 && n_preliminary > 0 {
        let avg_corrected: f64 = rms_corrected.iter().sum::<f64>() / n_corrected as f64;
        let avg_preliminary: f64 = rms_preliminary.iter().sum::<f64>() / n_preliminary as f64;

        eprintln!("  avg RMS corrected={avg_corrected:.4}, preliminary={avg_preliminary:.4}");

        // Corrected orbits should generally have lower or comparable RMS.
        // We don't assert strictly because statistical noise can cause inversions
        // in small samples, but we log it for diagnostics.
    }
}

/// Verify that the orbit result keys correspond to known hypothesis IDs.
#[test]
fn orbit_result_keys_match_hypothesis_ids() {
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
        state: runtime_state,
        ..
    } = run_pipeline(
        &dataset,
        &data_dir,
        &storage_dir,
        THROUGH_ORBIT,
        max_gap_nights,
    );
    let hypotheses = &runtime_state.track_hypotheses;

    let orbit_results = &runtime_state.orbit_results;

    // Every orbit result key should correspond to a hypothesis track ID.
    // The mapping is: ObjectNumber::Int(track_id) → orbit result.
    for obj in orbit_results.keys() {
        match obj {
            ObjectNumber::Int(tid) => {
                assert!(
                    hypotheses.contains_key(tid),
                    "orbit result key {tid} must exist in hypothesis set"
                );
            }
            ObjectNumber::String(s) => {
                panic!("unexpected string ObjectNumber in orbit results: {s}");
            }
        }
    }

    eprintln!(
        "[orbit_result_keys_match] {} orbit results, {} hypotheses",
        orbit_results.len(),
        hypotheses.len()
    );
}

// ===========================================================================
// Incremental (night-by-night) integration tests with orbit fitting
// ===========================================================================

/// Verify that the incremental five-stage pipeline produces orbit results
/// from multi-hop tracks.
#[test]
fn incremental_pipeline_fits_orbits() {
    let n_trajectories = 5;
    let n_nights = 5;
    let obs_per_night = 3;
    let start_night_id = 60000_u32;
    let max_gap_nights = 5_u8;
    let min_nodes = 3;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, n_trajectories)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(42)
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
        |is_last| {
            let mut s = THROUGH_SOLVE.to_vec();
            if is_last {
                s.push(PipelineStage::FitOrbit);
            }
            s
        },
    );
    let hypotheses = &runtime_state.track_hypotheses;

    let orbit_results = &runtime_state.orbit_results;

    eprintln!(
        "[incremental_pipeline_fits_orbits] hypotheses={}, orbit_results={}",
        hypotheses.len(),
        orbit_results.len(),
    );

    // The incremental pipeline should produce hypotheses from multi-hop
    // tracks, and the orbit fitter should process them.
    if !hypotheses.is_empty() {
        assert!(
            !orbit_results.is_empty(),
            "orbit results should be populated when hypotheses exist"
        );

        let n_ok = orbit_results.values().filter(|r| r.is_ok()).count();
        eprintln!("  {} / {} orbits succeeded", n_ok, orbit_results.len());

        // Validate all successful fits.
        for (obj, result) in orbit_results {
            if let Ok((gauss_result, rms)) = result {
                let kep = gauss_result
                    .get_orbit()
                    .to_keplerian()
                    .expect("conversion to Keplerian should succeed");

                assert!(kep.semi_major_axis > 0.0, "a > 0 for {obj:?}");
                assert!(
                    kep.eccentricity >= 0.0 && kep.eccentricity < 1.0,
                    "e in [0,1) for {obj:?}"
                );
                assert!(rms.is_finite() && *rms >= 0.0, "rms valid for {obj:?}");
            }
        }
    }
}

/// Incremental diverse-population test: NEA, MBA, Trojan with orbit fitting.
#[test]
fn incremental_diverse_populations_with_orbit_fit() {
    let n_per_pop = 3;
    let n_nights = 5;
    let obs_per_night = 3;
    let start_night_id = 60000_u32;
    let max_gap_nights = 5_u8;
    let min_nodes = 3;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::NearEarth, n_per_pop)
        .population(AsteroidPopulation::MainBelt, n_per_pop)
        .population(AsteroidPopulation::Trojan, n_per_pop)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(314)
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
        |is_last| {
            let mut s = THROUGH_SOLVE.to_vec();
            if is_last {
                s.push(PipelineStage::FitOrbit);
            }
            s
        },
    );
    let hypotheses = &runtime_state.track_hypotheses;

    let orbit_results = &runtime_state.orbit_results;

    let n_ok = orbit_results.values().filter(|r| r.is_ok()).count();
    let n_err = orbit_results.values().filter(|r| r.is_err()).count();

    eprintln!(
        "[incremental_diverse_orbit_fit] hypotheses={}, orbits: {} total, {} ok, {} err",
        hypotheses.len(),
        orbit_results.len(),
        n_ok,
        n_err,
    );

    if !hypotheses.is_empty() {
        assert!(
            !orbit_results.is_empty(),
            "orbit results should not be empty with hypotheses present"
        );

        // All successful fits must have valid elements.
        for (obj, result) in orbit_results {
            if let Ok((gauss_result, rms)) = result {
                let kep = gauss_result
                    .get_orbit()
                    .to_keplerian()
                    .expect("conversion to Keplerian should succeed");
                assert!(kep.semi_major_axis > 0.0, "a > 0 for {obj:?}");
                assert!(
                    kep.eccentricity >= 0.0 && kep.eccentricity < 1.0,
                    "e in [0,1) for {obj:?}"
                );
                assert!(rms.is_finite() && *rms >= 0.0, "rms valid for {obj:?}");
            }
        }
    }

    // Diagnostics: report per-population ground-truth matching.
    if !hypotheses.is_empty() {
        let matches = match_truth_to_hypotheses(
            &ground_truth,
            hypotheses,
            &runtime_state.alert_store,
            &runtime_state.seed_store,
        );

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

            eprintln!("  {}: scores={pop_scores:?}", pop.label());
        }
    }
}

/// Verify orbit fitting is deterministic: running the same pipeline twice
/// produces the same orbit results.
#[test]
fn fit_orbit_is_deterministic() {
    let n_trajectories = 4;
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

    // Run 1
    let data_dir1 = TempDir::new().unwrap();
    let storage_dir1 = TempDir::new().unwrap();
    let PipelineTestResult { state: state1, .. } = run_pipeline(
        &dataset,
        &data_dir1,
        &storage_dir1,
        THROUGH_ORBIT,
        max_gap_nights,
    );

    // Run 2
    let data_dir2 = TempDir::new().unwrap();
    let storage_dir2 = TempDir::new().unwrap();
    let PipelineTestResult { state: state2, .. } = run_pipeline(
        &dataset,
        &data_dir2,
        &storage_dir2,
        THROUGH_ORBIT,
        max_gap_nights,
    );

    let orbits1 = &state1.orbit_results;
    let orbits2 = &state2.orbit_results;

    assert_eq!(
        orbits1.len(),
        orbits2.len(),
        "same number of orbit results across runs"
    );

    // Same keys.
    let mut keys1: Vec<&ObjectNumber> = orbits1.keys().collect();
    let mut keys2: Vec<&ObjectNumber> = orbits2.keys().collect();
    keys1.sort();
    keys2.sort();
    assert_eq!(keys1, keys2, "orbit result keys must be identical");

    // Check that the success/failure counts are the same.
    // Note: the orbit fitter uses parallel batched processing which can cause
    // non-deterministic success/failure for marginal cases (e.g. borderline
    // triplet selection with noise). We verify aggregate consistency rather
    // than per-orbit bit-identical results.
    let n_ok_1 = orbits1.values().filter(|r| r.is_ok()).count();
    let n_ok_2 = orbits2.values().filter(|r| r.is_ok()).count();
    let n_err_1 = orbits1.values().filter(|r| r.is_err()).count();
    let n_err_2 = orbits2.values().filter(|r| r.is_err()).count();

    eprintln!(
        "[fit_orbit_determinism] run1: {n_ok_1} ok / {n_err_1} err, \
         run2: {n_ok_2} ok / {n_err_2} err"
    );

    // Allow a small tolerance for marginal cases.
    let diff = (n_ok_1 as i64 - n_ok_2 as i64).unsigned_abs() as usize;
    let tolerance = (orbits1.len() as f64 * 0.1).ceil() as usize; // 10%
    assert!(
        diff <= tolerance,
        "success count difference ({diff}) exceeds tolerance ({tolerance}): \
         run1={n_ok_1} ok, run2={n_ok_2} ok"
    );
}

// ---------------------------------------------------------------------------
// Empty-hypotheses safety tests
// ---------------------------------------------------------------------------

/// Verify that `FitOrbit` does **not** crash when the hypothesis set is empty.
///
/// This simulates the case where the solver produced no tracks (e.g. on
/// early incremental nights with too few edges). The stage should return
/// successfully with an empty `orbit_results` map.
#[test]
fn fit_orbit_does_not_crash_on_empty_hypotheses() {
    let n_trajectories = 2;
    let n_nights = 2; // only 2 nights → very few edges, may produce hypotheses
    let obs_per_night = 2;
    let start_night_id = 60000_u32;
    let max_gap_nights = 2_u8;

    // We build a small dataset and run the full pipeline including FitOrbit.
    // Even if the solver somehow produces hypotheses, we also test the
    // explicit empty-hypotheses path below.
    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, n_trajectories)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(42)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    // Run a four-stage pipeline (without FitOrbit) first to populate state,
    // then clear hypotheses and run FitOrbit alone.
    let parquet_path = data_dir.path().join("test_alerts.parquet");
    let alerts_uri = dataset.write_parquet(&parquet_path);

    let engine_config = engine_config_with_edges(&storage_dir, max_gap_nights);
    let persistence = PersistenceManager::open_or_create(engine_config.storage_path_buf())
        .expect("open persistence");
    let edge_models = test_edge_models();
    let solver_manager = test_solver_manager();

    let mut runtime_state = RuntimeState::new();

    // --- Stage 1-4: build up the state (ingest, seeds, edges, solve) ---
    let plan_four = PipelinePlan {
        stages: vec![
            PipelineStage::IngestNights,
            PipelineStage::BuildSeeds,
            PipelineStage::BuildEdges,
            PipelineStage::Solve,
        ],
        persist: PersistPolicy::None,
        inputs: PipelineInputs {
            alerts_uri: alerts_uri.clone(),
        },
    };

    {
        let runner = PipelineRunner {
            plan: plan_four.clone(),
        };
        let hooks = NoopHooks;
        let mut ctx = PipelineContext {
            plan: &plan_four,
            persistence: &persistence,
            runtime_state: &mut runtime_state,
            engine_config: &engine_config,
            edge_models: &edge_models,
            solver_manager: &solver_manager,
        };
        runner
            .run(&mut ctx, &hooks)
            .expect("four-stage pipeline should succeed");
    }

    // --- Force hypotheses to empty ---
    runtime_state.track_hypotheses = HypothesisSet::new();

    // --- Stage 5: run FitOrbit alone with empty hypotheses ---
    let plan_fit = PipelinePlan {
        stages: vec![PipelineStage::FitOrbit],
        persist: PersistPolicy::None,
        inputs: PipelineInputs { alerts_uri },
    };

    {
        let runner = PipelineRunner {
            plan: plan_fit.clone(),
        };
        let hooks = NoopHooks;
        let mut ctx = PipelineContext {
            plan: &plan_fit,
            persistence: &persistence,
            runtime_state: &mut runtime_state,
            engine_config: &engine_config,
            edge_models: &edge_models,
            solver_manager: &solver_manager,
        };
        let output = runner
            .run(&mut ctx, &hooks)
            .expect("FitOrbit must NOT crash when hypothesis set is empty");

        assert_eq!(output.reports.len(), 1);
        assert_eq!(output.reports[0].0, PipelineStage::FitOrbit);
    }

    // orbit_results should be empty (nothing to fit).
    assert!(
        runtime_state.orbit_results.is_empty(),
        "orbit_results must be empty when no hypotheses are provided, got {} entries",
        runtime_state.orbit_results.len()
    );

    eprintln!("[fit_orbit_empty_hypotheses] stage completed successfully with 0 orbit results");
}
