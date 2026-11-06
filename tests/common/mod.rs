//! Fixture and fingerprint helpers shared by the root integration tests
//! (`night_session_regression.rs`, `stream_resume.rs`): the `tests/data`
//! night fixtures, the id-agnostic collection fingerprint and the golden
//! file it is compared to (see `night_session_regression.rs`'s module doc).

// Each test crate includes this module and uses a different subset of it.
#![allow(dead_code)]

pub mod stream;

use std::fs::File;
use std::path::{Path, PathBuf};

use fink_fat_engine::topocentric_kf::branching::{
    ArchivedTrajectory, BranchCollection, BranchSnapshot,
};
use fink_fat_eval::cli::load_data;
use photom::{
    NightId,
    observation_dataset::{
        ObsDataset,
        observation::{Observation, ObservationInput},
    },
};
use polars::prelude::*;
use tempfile::TempDir;

pub const CONFIG_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/config.yaml");
pub const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");
pub const GOLDEN_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/golden/night_session_fingerprint.txt"
);
/// Environment variable forcing the golden file to be (re)written.
pub const BLESS_ENV: &str = "FINK_FAT_BLESS_GOLDEN";
/// Nights to run — enough for lineages to be seeded, extended, coasted,
/// N-scan pruned and archived, without paying for the whole fixture.
pub const N_NIGHTS: usize = 20;
/// Nights advanced before the resume test persists and reloads the snapshot.
pub const RESUME_AFTER: usize = N_NIGHTS / 2;

/// Night fixture files of `tests/data`, sorted by night id.
pub fn sorted_night_paths() -> Vec<PathBuf> {
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

/// Concatenate the night fixtures into one parquet so a single
/// [`ObsDataset`](photom::observation_dataset::ObsDataset) carries a night
/// index over all of them.
pub fn write_combined_parquet(files: &[PathBuf], out_path: &Path) {
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

/// 64-bit FNV-1a: a tiny, dependency-free hash whose output is stable across
/// Rust versions (unlike `DefaultHasher`), as a committed golden requires.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Order-independent combination of per-item hashes: sort, then hash the
/// concatenated little-endian words.
pub fn combine_unordered(mut hashes: Vec<u64>) -> u64 {
    hashes.sort_unstable();
    let bytes: Vec<u8> = hashes.iter().flat_map(|h| h.to_le_bytes()).collect();
    fnv1a(&bytes)
}

/// Hash of a branch snapshot with every run-dependent numeric id zeroed.
pub fn branch_hash(mut snapshot: BranchSnapshot) -> u64 {
    snapshot.lineage_id = 0;
    snapshot.parent_branch_id = 0;
    snapshot.branch_id = 0;
    snapshot.ancestor_at_scan_horizon = 0;
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&snapshot).expect("serialize branch");
    fnv1a(&bytes)
}

/// Hash of an archived trajectory with its `lineage_id` zeroed.
pub fn archived_hash(archived: &ArchivedTrajectory) -> u64 {
    let mut archived = archived.clone();
    archived.lineage_id = 0;
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&archived).expect("serialize archived");
    fnv1a(&bytes)
}

/// One golden line describing the collection right after `night_id` closed.
pub fn night_fingerprint(night_id: NightId, collection: &BranchCollection) -> String {
    let branches = combine_unordered(
        collection
            .branches
            .iter()
            .map(|b| branch_hash(b.to_snapshot()))
            .collect(),
    );
    let archived = combine_unordered(collection.archived.iter().map(archived_hash).collect());
    let consumed_then_pruned = combine_unordered(
        collection
            .last_night_consumed_then_pruned_ids
            .iter()
            .map(|id| fnv1a(&id.to_le_bytes()))
            .collect(),
    );
    format!(
        "night={} step={} branches={} archived={} ctp={} h_branches={branches:016x} \
         h_archived={archived:016x} h_ctp={consumed_then_pruned:016x}",
        night_id.0,
        collection.current_step,
        collection.branches.len(),
        collection.archived.len(),
        collection.last_night_consumed_then_pruned_ids.len(),
    )
}

/// The fixture's first [`N_NIGHTS`] nights as one dataset, with their ids in
/// chronological order. The [`TempDir`] holds the combined parquet and must
/// outlive the dataset.
pub fn load_fixture() -> (TempDir, ObsDataset, Vec<NightId>) {
    let paths: Vec<PathBuf> = sorted_night_paths().into_iter().take(N_NIGHTS).collect();
    assert!(!paths.is_empty(), "tests/data must contain night fixtures");

    let tmp = TempDir::new().expect("tempdir");
    let combined = tmp.path().join("combined.parquet");
    write_combined_parquet(&paths, &combined);
    let obs_dataset = load_data(combined.to_str().expect("utf8"), None);

    let mut night_ids: Vec<NightId> = obs_dataset
        .iter_night_id()
        .expect("dataset must have a night index")
        .copied()
        .collect();
    night_ids.sort_unstable();
    (tmp, obs_dataset, night_ids)
}

/// Observations of `night_id`, in the dataset's night order.
pub fn night_observations(obs_dataset: &ObsDataset, night_id: NightId) -> Vec<&Observation> {
    obs_dataset
        .iter_night_observations(&night_id)
        .into_iter()
        .flatten()
        .collect()
}

/// Compare fingerprint lines to the golden; (re)write it when blessing.
pub fn check_against_golden(lines: &[String]) {
    let actual = lines.join("\n") + "\n";

    let golden = Path::new(GOLDEN_PATH);
    if std::env::var_os(BLESS_ENV).is_some() || !golden.exists() {
        std::fs::create_dir_all(golden.parent().expect("golden has a parent dir"))
            .expect("create golden dir");
        std::fs::write(golden, &actual).expect("write golden");
        eprintln!("golden fingerprint written to {GOLDEN_PATH}; rerun to check determinism");
        return;
    }

    let expected = std::fs::read_to_string(golden).expect("read golden");
    if let Some((i, (exp, act))) = expected
        .lines()
        .zip(actual.lines())
        .enumerate()
        .find(|(_, (e, a))| e != a)
    {
        panic!("fingerprint diverges at night #{i}:\n  expected: {exp}\n  actual:   {act}");
    }
    assert_eq!(
        expected.lines().count(),
        actual.lines().count(),
        "fingerprint night count differs from the golden"
    );
}

/// The [`ObservationInput`] re-creating `observation` in another dataset
/// (a streaming night's own dataset).
pub fn observation_input(observation: &Observation) -> ObservationInput {
    ObservationInput::new(
        *observation.id(),
        *observation.equ_coord(),
        observation.photometry().clone(),
        observation.mjd_tt(),
        observation.observer_id().copied(),
    )
}

/// Connection string of the throwaway Postgres the SQL tests write into
/// (`sql_convert.rs`, `stream_sql.rs`): `$DATABASE_URL`, defaulting to the
/// `db` service of the repo's `docker-compose.yml` reached from the host
/// (`docker compose up -d db` exposes it on `localhost:5432`).
pub fn database_url() -> String {
    std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://finkfat:finkfat@localhost:5432/finkfat_test".to_string())
}
