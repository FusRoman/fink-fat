//! Integration test for the `IngestNights` → `BuildSeeds` pipeline stages.
//!
//! This test exercises the full `PipelineRunner::run` path with two stages
//! executed sequentially: first `IngestNights` loads synthetic alerts from a
//! Parquet file, then `BuildSeeds` generates intra-night seeds (pairs → triplets)
//! from those alerts.

use tempfile::TempDir;

use fink_fat_engine::{night_id::NightId, pipeline::stages::PipelineStage};

use super::{PipelineTestResult, THROUGH_SEEDS, run_pipeline_minimal};
use crate::synthetic_alerts::{AsteroidPopulation, SyntheticDatasetBuilder};

// ---------------------------------------------------------------------------
// Integration tests
// ---------------------------------------------------------------------------

#[test]
fn ingest_then_build_seeds_produces_seeds_for_each_night() {
    // ---- 1) Generate synthetic dataset ----
    //
    // We use MainBelt asteroids with 3 observations per night so that the
    // pair/triplet generators can form full triplets (need ≥ 3 obs).
    let n_trajectories = 5;
    let n_nights = 3;
    let obs_per_night = 3;
    let start_night_id = 60000_u32;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, n_trajectories)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(99)
        .build();

    let expected_total_alerts = n_trajectories * n_nights * obs_per_night;
    assert_eq!(dataset.n_alerts(), expected_total_alerts);

    // ---- 2) Run the pipeline (IngestNights + BuildSeeds) ----
    let data_dir = TempDir::new().expect("create data temp dir");
    let storage_dir = TempDir::new().expect("create storage temp dir");

    let PipelineTestResult { output, state, .. } =
        run_pipeline_minimal(&dataset, &data_dir, &storage_dir, THROUGH_SEEDS);

    // ---- 3) Verify we got two stage reports ----
    assert_eq!(output.reports.len(), 2, "expected 2 stage reports");

    let (stage_0, report_0) = &output.reports[0];
    assert_eq!(*stage_0, PipelineStage::IngestNights);

    let (stage_1, report_1) = &output.reports[1];
    assert_eq!(*stage_1, PipelineStage::BuildSeeds);

    // ---- 9) Verify IngestNights counters ----
    let ingest_counters: std::collections::HashMap<&str, u64> =
        report_0.counters.iter().copied().collect();
    assert_eq!(
        ingest_counters.get("n_alerts").copied(),
        Some(expected_total_alerts as u64),
    );
    assert_eq!(
        ingest_counters.get("n_nights").copied(),
        Some(n_nights as u64),
    );

    // ---- 10) Verify BuildSeeds counters ----
    let seed_counters: std::collections::HashMap<&str, u64> =
        report_1.counters.iter().copied().collect();

    let reported_nights = seed_counters.get("nights").copied().unwrap_or(0);
    let reported_alerts = seed_counters.get("alerts").copied().unwrap_or(0);
    let reported_pairs = seed_counters.get("pairs").copied().unwrap_or(0);
    let reported_seeds = seed_counters.get("seeds").copied().unwrap_or(0);

    assert_eq!(
        reported_nights, n_nights as u64,
        "BuildSeeds should process all {n_nights} nights"
    );
    assert_eq!(
        reported_alerts, expected_total_alerts as u64,
        "BuildSeeds should see all {expected_total_alerts} alerts"
    );
    assert!(
        reported_pairs > 0,
        "BuildSeeds should produce at least some pairs"
    );
    assert!(
        reported_seeds > 0,
        "BuildSeeds should produce at least some seeds"
    );

    // ---- 11) Verify seed store is populated ----
    let seed_store = &state.seed_store;
    assert_eq!(
        seed_store.n_nights(),
        n_nights,
        "seed store should contain seeds for all {n_nights} nights"
    );

    let mut total_seeds_in_store: usize = 0;
    for night_offset in 0..n_nights {
        let nid = NightId(start_night_id + night_offset as u32);
        assert!(
            seed_store.contains_night(&nid),
            "seed store should contain night {nid:?}"
        );

        let night_seeds = seed_store
            .get(&nid)
            .expect("seed store should have seeds for this night");
        assert!(
            !night_seeds.is_empty(),
            "night {nid:?} should have at least one seed"
        );
        total_seeds_in_store += night_seeds.len();

        // ---- 12) Verify seed members reference valid alerts ----
        let alert_store = &state.alert_store;
        let night_alerts = alert_store
            .get(&nid)
            .expect("alert store should have this night");

        for seed in night_seeds {
            assert_eq!(
                seed.night_id(),
                nid,
                "seed night_id must match the night it was stored under"
            );
            // Each seed member should reference an alert in the same night.
            assert!(
                seed.members.len() >= 2,
                "seed should have at least 2 members (pair or triplet)"
            );
            for member_key in &seed.members {
                assert_eq!(
                    member_key.night_id, nid,
                    "seed member alert must belong to the same night"
                );
                // Verify the dia_source_id exists in the night's alerts.
                let found = night_alerts
                    .iter()
                    .any(|a| a.key.dia_source_id == member_key.dia_source_id);
                assert!(
                    found,
                    "seed member dia_source_id {} should exist in the alert store for night {nid:?}",
                    member_key.dia_source_id
                );
            }
        }

        // ---- 13) Verify seeds are sorted (required by edge builder) ----
        for w in night_seeds.windows(2) {
            assert!(w[0] <= w[1], "seeds within a night must be sorted");
        }
    }

    // ---- 14) Verify total seed count matches reported counter ----
    assert_eq!(
        total_seeds_in_store as u64, reported_seeds,
        "total seeds in store must match the reported counter"
    );
}

#[test]
fn build_seeds_with_mixed_populations() {
    // Test with a mix of fast (NEA) and slow (TNO) movers to verify
    // the seeding pipeline handles varying angular speeds correctly.
    let n_per_pop = 3;
    let n_nights = 3;
    let obs_per_night = 3;
    let start_night_id = 60000_u32;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::NearEarth, n_per_pop)
        .population(AsteroidPopulation::TransNeptunian, n_per_pop)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(77)
        .build();

    let total_trajectories = 2 * n_per_pop;
    let expected_total_alerts = total_trajectories * n_nights * obs_per_night;

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();

    let PipelineTestResult { output, state, .. } =
        run_pipeline_minimal(&dataset, &data_dir, &storage_dir, THROUGH_SEEDS);

    // Both stages should succeed.
    assert_eq!(output.reports.len(), 2);
    assert_eq!(output.reports[0].0, PipelineStage::IngestNights);
    assert_eq!(output.reports[1].0, PipelineStage::BuildSeeds);

    // Alert store should have all alerts.
    assert_eq!(state.alert_store.n_alerts(), expected_total_alerts);

    // Seed store should be populated for all nights.
    let seed_store = &state.seed_store;
    assert_eq!(seed_store.n_nights(), n_nights);

    // Count total seeds — we expect at least some (the exact count depends
    // on pair/triplet config defaults and the synthetic data geometry).
    let total_seeds: usize = (0..n_nights)
        .map(|i| {
            let nid = NightId(start_night_id + i as u32);
            seed_store.len_night(&nid).unwrap_or(0)
        })
        .sum();
    assert!(
        total_seeds > 0,
        "mixed-population dataset should produce seeds"
    );

    // Verify the BuildSeeds counter is consistent.
    let seed_counters: std::collections::HashMap<&str, u64> =
        output.reports[1].1.counters.iter().copied().collect();
    assert_eq!(
        seed_counters.get("seeds").copied().unwrap_or(0),
        total_seeds as u64,
    );
}
