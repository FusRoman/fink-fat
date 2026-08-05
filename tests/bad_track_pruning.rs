//! Integration test for LLR-gated stale-lineage pruning
//! ([`purge_stale_lineages`]). Drives the real engine library
//! ([`BranchCollection::advance_one_night`]) over a slice of the `tests/data`
//! fixture three ways — pruning disabled, mild (safe) pruning, and aggressive
//! pruning — and asserts the two behaviours that matter:
//!
//! 1. **Safety**: mild pruning (`stale_llr_floor` well below any real object's
//!    LLR) preserves reconstruction of the well-observed known objects — the
//!    guarantee that sparsely-observed-but-real lineages are not culled.
//! 2. **Teeth**: aggressive pruning (`stale_llr_floor` above every LLR, short
//!    lifetime) reconstructs strictly fewer objects — the mechanism actually
//!    removes lineages, so the config is a real lever.
//!
//! The fine-grained NaN/-inf/boundary decision logic is covered exhaustively
//! by the unit + property tests on `lineage_is_stale` in
//! `crates/fink-fat-engine/src/topocentric_kf/branching/pruning.rs`.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use fink_fat_engine::{
    engine_config::EngineConfig, engine_config::kalman_context::KalmanContext,
    topocentric_kf::branching::BranchCollection,
};
use fink_fat_eval::{cli::load_data, seed_bank_report::ground_truth::ObsTrajLookup};
use photom::{
    NightId, TrajId, observation_dataset::ObsDataset, observation_dataset::observation::Observation,
};
use polars::prelude::*;
use tempfile::TempDir;

const CONFIG_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/config.yaml");
const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");
/// Nights to run — enough to build up lineages (and let some coast) without
/// paying for the whole 177-night fixture on every `cargo test`.
const N_NIGHTS: usize = 20;
/// Reconstruction is "solid" for an object when its best branch covers at
/// least this fraction of its observations.
const SOLID_RECALL: f64 = 0.9;
/// After mild pruning, a previously-solid object must still be reconstructed
/// at least this well (a little slack for benign branch churn).
const SAFE_RECALL_FLOOR: f64 = 0.8;

fn sorted_night_paths() -> Vec<PathBuf> {
    let mut nights: Vec<(u32, PathBuf)> = std::fs::read_dir(DATA_DIR)
        .expect("read tests/data dir")
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("parquet") {
                return None;
            }
            let n: u32 = path
                .file_stem()?
                .to_str()?
                .strip_prefix("night_id_")?
                .parse()
                .ok()?;
            Some((n, path))
        })
        .collect();
    nights.sort_by_key(|(n, _)| *n);
    nights.into_iter().map(|(_, p)| p).collect()
}

fn write_combined_parquet(files: &[PathBuf], out_path: &Path) {
    let lfs: Vec<LazyFrame> = files
        .iter()
        .map(|p| {
            LazyFrame::scan_parquet(p.to_str().expect("utf8 path").into(), Default::default())
                .expect("scan_parquet")
        })
        .collect();
    let mut df = concat(&lfs, UnionArgs::default())
        .expect("concat")
        .collect()
        .expect("collect");
    let file = File::create(out_path).expect("create combined parquet");
    ParquetWriter::new(file)
        .finish(&mut df)
        .expect("write parquet");
}

/// Owned summary of a run: recall per ground-truth trajectory and the total
/// number of distinct lineages — enough to compare runs after the borrowing
/// `BranchCollection` is dropped.
struct RunSummary {
    recall_per_traj: HashMap<TrajId, f64>,
    n_lineages: usize,
}

impl RunSummary {
    fn n_solid(&self, floor: f64) -> usize {
        self.recall_per_traj
            .values()
            .filter(|&&r| r >= floor)
            .count()
    }
}

/// Run the night-by-night pipeline with `config` and reduce the final
/// collection to an owned [`RunSummary`]. The `BranchCollection` (which borrows
/// `config`/`ctx`) is dropped before returning, so the caller may then mutate
/// `config` for the next run.
fn run_and_summarize(
    config: &EngineConfig,
    ctx: &KalmanContext,
    obs_dataset: &ObsDataset,
    ground_truth: &ObsTrajLookup,
    night_ids: &[NightId],
) -> RunSummary {
    let mut collection = BranchCollection::empty();
    for (step, night_id) in night_ids.iter().enumerate() {
        let night_obs: Vec<&Observation> = obs_dataset
            .iter_night_observations(night_id)
            .into_iter()
            .flatten()
            .collect();
        collection = collection
            .advance_one_night(&night_obs, obs_dataset, config, ctx, step)
            .expect("advance_one_night should not fail on the fixture");
    }

    // Best branch overlap per ground-truth trajectory → recall.
    let mut obs_total: HashMap<TrajId, usize> = HashMap::new();
    for (traj, count) in ground_truth.obs_counts_by_traj() {
        obs_total.insert(traj, count);
    }
    let mut best_overlap: HashMap<TrajId, usize> = HashMap::new();
    let mut lineages = std::collections::HashSet::new();
    for branch in &collection.branches {
        lineages.insert(branch.lineage_id);
        let mut per_traj: HashMap<TrajId, usize> = HashMap::new();
        for &obs_id in branch.track_ids() {
            if let Some(traj) = ground_truth.traj_of(obs_id) {
                *per_traj.entry(traj.clone()).or_insert(0) += 1;
            }
        }
        for (traj, overlap) in per_traj {
            let e = best_overlap.entry(traj).or_insert(0);
            *e = (*e).max(overlap);
        }
    }
    let recall_per_traj = best_overlap
        .into_iter()
        .map(|(traj, overlap)| {
            let total = obs_total.get(&traj).copied().unwrap_or(0).max(1);
            (traj, overlap as f64 / total as f64)
        })
        .collect();

    RunSummary {
        recall_per_traj,
        n_lineages: lineages.len(),
    }
}

#[test]
fn stale_pruning_preserves_real_objects_and_removes_lineages() {
    let paths: Vec<PathBuf> = sorted_night_paths().into_iter().take(N_NIGHTS).collect();
    assert!(!paths.is_empty(), "tests/data must contain night fixtures");

    let tmp = TempDir::new().expect("tempdir");
    let combined = tmp.path().join("combined.parquet");
    write_combined_parquet(&paths, &combined);
    let obs_dataset = load_data(combined.to_str().expect("utf8"), None);
    let ground_truth = ObsTrajLookup::build(&obs_dataset);
    assert!(
        ground_truth.has_ground_truth(),
        "fixture must carry traj_id"
    );

    let mut night_ids: Vec<NightId> = obs_dataset
        .iter_night_id()
        .expect("dataset must have a night index")
        .copied()
        .collect();
    night_ids.sort_unstable();

    let mut config =
        EngineConfig::load_engine_config_validated(CONFIG_PATH).expect("load engine config");
    let ctx = config.build_context();

    // ── Run 1: pruning disabled ──────────────────────────────────────────
    config.advance_params.max_lineage_lifetime_nights = 0;
    let disabled = run_and_summarize(&config, &ctx, &obs_dataset, &ground_truth, &night_ids);
    let solid_disabled: Vec<TrajId> = disabled
        .recall_per_traj
        .iter()
        .filter(|(_, r)| **r >= SOLID_RECALL)
        .map(|(t, _)| t.clone())
        .collect();
    assert!(
        !solid_disabled.is_empty(),
        "the fixture should reconstruct some objects with pruning off"
    );

    // ── Run 2: mild, safe pruning — floor well below any real evidence ─────
    config.advance_params.max_lineage_lifetime_nights = 5;
    config.advance_params.stale_llr_floor = -100.0;
    let mild = run_and_summarize(&config, &ctx, &obs_dataset, &ground_truth, &night_ids);
    // Safety: every object solidly reconstructed with pruning off is still
    // reconstructed well under mild pruning — sparse/real lineages survive.
    for traj in &solid_disabled {
        let r = mild.recall_per_traj.get(traj).copied().unwrap_or(0.0);
        assert!(
            r >= SAFE_RECALL_FLOOR,
            "mild pruning destroyed a real object {traj}: recall {r:.2} < {SAFE_RECALL_FLOOR}"
        );
    }

    // ── Run 3: aggressive pruning — floor above every LLR, short lifetime ──
    config.advance_params.max_lineage_lifetime_nights = 1;
    config.advance_params.stale_llr_floor = 1e12;
    let aggressive = run_and_summarize(&config, &ctx, &obs_dataset, &ground_truth, &night_ids);
    // Teeth: culling every coasting lineage reconstructs strictly fewer
    // objects than leaving pruning off — the lever measurably removes lineages.
    assert!(
        aggressive.n_solid(SOLID_RECALL) < disabled.n_solid(SOLID_RECALL),
        "aggressive pruning should reconstruct fewer objects than disabled \
         (disabled solid={}, aggressive solid={})",
        disabled.n_solid(SOLID_RECALL),
        aggressive.n_solid(SOLID_RECALL),
    );

    println!(
        "lineages: disabled={} mild={} aggressive={}; solid objects: disabled={} mild={} aggressive={}",
        disabled.n_lineages,
        mild.n_lineages,
        aggressive.n_lineages,
        disabled.n_solid(SOLID_RECALL),
        mild.n_solid(SOLID_RECALL),
        aggressive.n_solid(SOLID_RECALL),
    );
}
