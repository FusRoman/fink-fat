//! Integration tests for BuildSeeds with Hough seeding method.

use tempfile::TempDir;

use fink_fat_engine::{
    Alert,
    engine_config::{EngineConfig, pipeline_policy::PersistPolicy},
    night_id::NightId,
    persistence::{PersistenceManager, runtime_state::RuntimeState},
    pipeline::{
        PipelineContext, PipelineInputs, PipelinePlan, PipelineRunner, stages::PipelineStage,
    },
};

use super::{
    NoopHooks, THROUGH_SEEDS, test_edge_models, test_solver_manager, write_alerts_parquet,
};
use crate::synthetic_alerts::{AsteroidPopulation, SyntheticDatasetBuilder};

fn engine_config_hough(storage_dir: &TempDir, triplet_only: bool) -> EngineConfig {
    let storage_path = storage_dir.path().to_str().unwrap();
    let yaml = format!(
        r#"
version: 1
storage_path: "{storage_path}"
seeding:
  method: hough
  triplet_only: {triplet_only}
  hough:
    min_angular_speed: "0 arcsec/hour"
    max_angular_speed: "4000 arcsec/hour"
    velocity_grid_steps: 41
    spatial_bin_size: "7200 arcsec"
    min_alerts_per_peak: 2
    max_peaks_per_night: 2048
    photometric_filter: false
    photometric_max_mag_diff: 0.7
    photometric_sigma_multiplier: 3.0
    weight_by_photometric_error: true
    max_seeds_per_alert: 0
"#,
    );
    serde_yaml::from_str(&yaml).expect("deserialize hough engine config")
}

fn run_through_seeds_with_config(
    dataset: &crate::synthetic_alerts::SyntheticDataset,
    data_dir: &TempDir,
    engine_config: &EngineConfig,
) -> (fink_fat_engine::pipeline::PipelineOutput, RuntimeState) {
    let persistence = PersistenceManager::open_or_create(engine_config.storage_path_buf())
        .expect("open persistence");
    let edge_models = test_edge_models();
    let solver_manager = test_solver_manager();

    let mut runtime_state = RuntimeState::new();

    let mut night_ids: Vec<u32> = dataset.alerts().iter().map(|a| a.key.night_id.0).collect();
    night_ids.sort_unstable();
    night_ids.dedup();

    let mut last_output = None;

    for (run_idx, &nid) in night_ids.iter().enumerate() {
        let night_alerts: Vec<&Alert> = dataset
            .alerts()
            .iter()
            .filter(|a| a.key.night_id.0 == nid)
            .collect();

        let parquet_path = data_dir
            .path()
            .join(format!("hough_night_{nid}_{run_idx}.parquet"));
        let alerts_uri = write_alerts_parquet(&night_alerts, &parquet_path);

        let plan = PipelinePlan {
            stages: THROUGH_SEEDS.to_vec(),
            persist: PersistPolicy::None,
            inputs: PipelineInputs { alerts_uri },
        };

        let runner = PipelineRunner { plan: plan.clone() };
        let hooks = NoopHooks;

        let mut ctx = PipelineContext {
            plan: &plan,
            persistence: &persistence,
            runtime_state: &mut runtime_state,
            engine_config,
            edge_models: &edge_models,
            solver_manager: &solver_manager,
        };

        let output = runner
            .run(&mut ctx, &hooks)
            .unwrap_or_else(|e| panic!("hough run for night {nid} failed: {e}"));
        let _ = ctx;
        last_output = Some(output);
    }

    (last_output.expect("at least one run"), runtime_state)
}

#[test]
fn hough_build_seeds_produces_seeds_across_nights() {
    let n_trajectories = 6;
    let n_nights = 3;
    let obs_per_night = 3;
    let start_night_id = 64000_u32;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, n_trajectories)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(123)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();
    let cfg = engine_config_hough(&storage_dir, false);

    let (last_output, state) = run_through_seeds_with_config(&dataset, &data_dir, &cfg);

    assert_eq!(last_output.reports.len(), 2);
    assert_eq!(last_output.reports[1].0, PipelineStage::BuildSeeds);

    let counters: std::collections::HashMap<&str, u64> =
        last_output.reports[1].1.counters.iter().copied().collect();
    assert_eq!(counters.get("nights").copied(), Some(1));
    assert!(
        counters.get("seeds").copied().unwrap_or(0) > 0,
        "hough build seeds should produce seeds on the anchor night"
    );

    for i in 0..n_nights {
        let nid = NightId(start_night_id + i as u32);
        let n = state.seed_store.len_night(&nid).unwrap_or(0);
        assert!(n > 0, "night {nid} should have at least one hough seed");
    }
}

#[test]
fn hough_triplet_only_rejects_two_obs_nights() {
    let n_trajectories = 4;
    let n_nights = 2;
    let obs_per_night = 2;
    let start_night_id = 65000_u32;

    let dataset = SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::MainBelt, n_trajectories)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .start_night_id(start_night_id)
        .seed(456)
        .build();

    let data_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();
    let cfg = engine_config_hough(&storage_dir, true);

    let (last_output, state) = run_through_seeds_with_config(&dataset, &data_dir, &cfg);

    let counters: std::collections::HashMap<&str, u64> =
        last_output.reports[1].1.counters.iter().copied().collect();
    assert_eq!(counters.get("seeds").copied(), Some(0));

    for i in 0..n_nights {
        let nid = NightId(start_night_id + i as u32);
        let n = state.seed_store.len_night(&nid).unwrap_or(0);
        assert_eq!(
            n, 0,
            "triplet_only hough with 2 obs/night should emit no seeds"
        );
    }
}
