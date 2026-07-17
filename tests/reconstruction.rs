//! Black-box, non-regression integration test: run the real `fink-fat`
//! binary over the full 177-night `tests/data/` fixture (20 known synthetic
//! asteroids tagged via `traj_id`), alternating single-night and
//! batch-night CLI invocations against the same on-disk snapshot, then
//! grade the resulting `BranchCollection` against ground truth.
//!
//! There is no ground-truth orbital-elements file for this fixture (only
//! the `traj_id` observation labeling), so orbit checks are limited to
//! physical plausibility (bound, finite Keplerian elements) rather than
//! comparison to a known truth.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use assert_cmd::Command;
use fink_fat_engine::{engine_config::EngineConfig, topocentric_kf::branching::BranchCollection};
use fink_fat_eval::{cli::load_data, seed_bank_report::ground_truth::ObsTrajLookup};
use outfit::OrbitalElements;
use photom::TrajId;
use polars::prelude::*;
use tempfile::TempDir;

const CONFIG_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/config.yaml");
const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");

/// A source parquet file for one night, plus its parsed `night_id`.
struct NightFile {
    night_id: u32,
    path: PathBuf,
}

/// List every `tests/data/night_id_<N>.parquet` file, sorted by `night_id`.
fn list_nights_sorted() -> Vec<NightFile> {
    let mut nights: Vec<NightFile> = std::fs::read_dir(DATA_DIR)
        .expect("read tests/data dir")
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("parquet") {
                return None;
            }
            let stem = path.file_stem()?.to_str()?.to_string();
            let night_id: u32 = stem.strip_prefix("night_id_")?.parse().ok()?;
            Some(NightFile { night_id, path })
        })
        .collect();
    nights.sort_by_key(|n| n.night_id);
    nights
}

/// Concatenate the given per-night parquet files into a single parquet file
/// at `out_path` (used for batch-mode invocations and for loading combined
/// ground truth).
fn write_combined_parquet(files: &[PathBuf], out_path: &Path) {
    let lfs: Vec<LazyFrame> = files
        .iter()
        .map(|p| {
            LazyFrame::scan_parquet(p.to_str().expect("utf8 path").into(), Default::default())
                .expect("scan_parquet")
        })
        .collect();
    let mut df = concat(&lfs, UnionArgs::default())
        .expect("concat lazy frames")
        .collect()
        .expect("collect combined dataframe");
    let file = File::create(out_path).expect("create combined parquet file");
    ParquetWriter::new(file)
        .finish(&mut df)
        .expect("write combined parquet");
}

/// Run one `fink-fat` invocation against `alerts_path`, asserting success.
/// Inherits `FINK_FAT__STORAGE_PATH` from the test process's own
/// environment (set once at the top of the test).
fn run_fink_fat(alerts_path: &str) {
    Command::cargo_bin(assert_cmd::pkg_name!())
        .expect("binary should build")
        .args(["--alerts", alerts_path, "--config", CONFIG_PATH])
        .assert()
        .success();
}

#[test]
fn alternating_single_and_batch_modes_reconstruct_known_trajectories() {
    let nights = list_nights_sorted();
    assert!(
        !nights.is_empty(),
        "tests/data should contain per-night parquet fixtures"
    );

    let storage_dir = TempDir::new().expect("create storage tempdir");
    // SAFETY: this test is the only one in this binary and does not run
    // concurrently with other tests mutating this env var.
    unsafe {
        std::env::set_var("FINK_FAT__STORAGE_PATH", storage_dir.path());
    }

    let batch_dir = TempDir::new().expect("create batch-fixture tempdir");

    // Alternating chunks: [single][batch][single][batch][single][batch],
    // resuming from the same on-disk snapshot throughout.
    let n = nights.len();
    let boundaries: Vec<usize> = [3usize, 40, 45, 150, 160, n]
        .into_iter()
        .map(|b| b.min(n))
        .collect();

    let mut start = 0;
    for (chunk_idx, &end) in boundaries.iter().enumerate() {
        if end <= start {
            continue;
        }
        let chunk = &nights[start..end];
        let is_batch = chunk_idx % 2 == 1;
        if is_batch {
            let combined_path = batch_dir.path().join(format!("batch_{chunk_idx}.parquet"));
            let paths: Vec<PathBuf> = chunk.iter().map(|nf| nf.path.clone()).collect();
            write_combined_parquet(&paths, &combined_path);
            run_fink_fat(combined_path.to_str().expect("utf8 path"));
        } else {
            for night in chunk {
                run_fink_fat(night.path.to_str().expect("utf8 path"));
            }
        }
        start = end;
    }

    // ---- Reload final state ----
    let engine_config =
        EngineConfig::load_engine_config_validated(CONFIG_PATH).expect("reload engine config");
    assert_eq!(
        engine_config.storage_path().as_str(),
        storage_dir.path().to_str().expect("utf8 tempdir path"),
        "FINK_FAT__STORAGE_PATH override should have taken effect"
    );
    let kalman_context = engine_config.build_context();
    let snapshot_path = engine_config.snapshot_path();
    let collection =
        BranchCollection::load_snapshot_from_disk(&snapshot_path, &kalman_context, &engine_config)
            .expect("load final snapshot");

    assert_eq!(
        collection.current_step,
        nights.len(),
        "every night should be counted exactly once regardless of single/batch mode"
    );

    // ---- Ground truth ----
    let all_files: Vec<PathBuf> = nights.iter().map(|nf| nf.path.clone()).collect();
    let combined_gt_path = batch_dir.path().join("combined_ground_truth.parquet");
    write_combined_parquet(&all_files, &combined_gt_path);
    let (_df, obs_dataset) = load_data(combined_gt_path.to_str().expect("utf8 path"));
    let ground_truth = ObsTrajLookup::build(&obs_dataset);
    assert!(
        ground_truth.has_ground_truth(),
        "test fixtures must carry traj_id ground truth"
    );
    let obs_counts = ground_truth.obs_counts_by_traj();

    // ---- Best-overlapping branch per ground-truth trajectory ----
    // traj_id -> (overlap_count, branch_index)
    let mut best_overlap: HashMap<TrajId, (usize, usize)> = HashMap::new();
    for (branch_idx, branch) in collection.branches.iter().enumerate() {
        let mut counts: HashMap<TrajId, usize> = HashMap::new();
        for &obs_id in branch.track_ids() {
            if let Some(traj) = ground_truth.traj_of(obs_id) {
                *counts.entry(traj.clone()).or_insert(0) += 1;
            }
        }
        for (traj, count) in counts {
            let entry = best_overlap.entry(traj).or_insert((0, branch_idx));
            if count > entry.0 {
                *entry = (count, branch_idx);
            }
        }
    }

    let mut traj_ids: Vec<TrajId> = obs_counts.keys().cloned().collect();
    traj_ids.sort_by_key(|t| match t {
        TrajId::Int(n) => *n,
        TrajId::Str(_) => u32::MAX,
    });

    // Non-regression baseline: (traj_id, n_obs, best_overlap) as observed on
    // this fixture today. Every best-matching branch is 100% pure (no
    // cross-object contamination) — recall is the only axis that varies.
    // Trajectories 10/11/12/15/21 are only partially recovered (fragmented
    // across branches); this is a known, reported limitation, not something
    // this test tries to fix — see the conversation report.
    let baseline: HashMap<u32, (usize, usize)> = HashMap::from([
        (1, (247, 217)),
        (2, (237, 224)),
        (3, (236, 224)),
        (4, (233, 208)),
        (5, (232, 190)),
        (6, (227, 218)),
        (7, (225, 224)),
        (8, (223, 223)),
        (9, (218, 217)),
        (10, (217, 37)),
        (11, (215, 20)),
        (12, (215, 60)),
        (13, (214, 172)),
        (14, (212, 212)),
        (15, (212, 61)),
        (16, (211, 185)),
        (17, (210, 178)),
        (19, (210, 205)),
        (20, (210, 204)),
        (21, (210, 136)),
    ]);

    // Collected rather than asserted inline, so the full table always prints
    // before the test fails on any regression — panicking mid-loop (the
    // previous behavior) hid every trajectory after the first mismatch,
    // which is exactly the information needed to judge whether a change
    // helped, hurt, or was neutral across the whole fixture.
    let mut failures: Vec<String> = Vec::new();

    println!(
        "\n{:<10} {:>8} {:>10} {:>10}  orbit (a [AU], e, i [deg])",
        "traj_id", "n_obs", "recall", "precision"
    );
    for traj in &traj_ids {
        let n_obs = obs_counts[traj];
        let TrajId::Int(traj_num) = traj else {
            panic!("test fixtures use UInt32 traj_id columns, got {traj:?}");
        };
        let &(expected_n_obs, expected_overlap) = baseline
            .get(traj_num)
            .unwrap_or_else(|| panic!("no non-regression baseline recorded for traj {traj}"));
        if n_obs != expected_n_obs {
            failures.push(format!(
                "traj {traj}: fixture observation count changed (was {expected_n_obs}, now {n_obs}) \
                 — update the baseline if tests/data was intentionally regenerated"
            ));
        }

        match best_overlap.get(traj) {
            Some(&(overlap, branch_idx)) => {
                if overlap != expected_overlap {
                    failures.push(format!(
                        "traj {traj}: reconstruction regressed — best-branch overlap was \
                         {expected_overlap}/{n_obs}, now {overlap}/{n_obs}"
                    ));
                }

                let branch = &collection.branches[branch_idx];
                let recall = overlap as f64 / n_obs as f64;
                let precision = overlap as f64 / branch.track_ids().len() as f64;
                if precision != 1.0 {
                    failures.push(format!(
                        "traj {traj}: best-matching branch is no longer pure ({:.1}% precision) \
                         — a cross-object contamination regression",
                        precision * 100.0
                    ));
                }

                let orbit_str = if recall >= 0.8 {
                    match branch.bank.best().map(|h| h.kf.to_orbit()) {
                        Some(OrbitalElements::Keplerian { elements, .. }) => {
                            if !(0.0..1.0).contains(&elements.eccentricity) {
                                failures.push(format!(
                                    "traj {traj}: eccentricity {} is not a bound orbit",
                                    elements.eccentricity
                                ));
                            }
                            if !(elements.semi_major_axis.is_finite()
                                && elements.semi_major_axis > 0.0)
                            {
                                failures.push(format!(
                                    "traj {traj}: implausible semi-major axis {}",
                                    elements.semi_major_axis
                                ));
                            }
                            if !elements.inclination.is_finite() {
                                failures.push(format!("traj {traj}: non-finite inclination"));
                            }
                            format!(
                                "a={:.3} e={:.3} i={:.2}",
                                elements.semi_major_axis,
                                elements.eccentricity,
                                elements.inclination.to_degrees()
                            )
                        }
                        Some(_) => "non-Keplerian orbit".to_string(),
                        None => "no surviving hypothesis".to_string(),
                    }
                } else {
                    "-".to_string()
                };

                println!(
                    "{:<10} {:>8} overlap={:<6} {:>9.1}% {:>9.1}%  {}",
                    traj.to_string(),
                    n_obs,
                    overlap,
                    recall * 100.0,
                    precision * 100.0,
                    orbit_str
                );
            }
            None => {
                if expected_overlap != 0 {
                    failures.push(format!(
                        "traj {traj}: reconstruction regressed — completely missed \
                         (baseline expected overlap {expected_overlap}/{n_obs})"
                    ));
                }
                println!(
                    "{:<10} {:>8} {:>10} {:>10}  -",
                    traj.to_string(),
                    n_obs,
                    "MISSED",
                    "-"
                );
            }
        }
    }
    println!();

    assert!(
        failures.is_empty(),
        "{} non-regression check(s) failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
