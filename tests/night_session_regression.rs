//! Non-regression guard for the per-night tracking pipeline
//! ([`BranchCollection::advance_one_night`]).
//!
//! Runs the engine night by night over the first [`N_NIGHTS`] nights of the
//! `tests/data` fixture and reduces the collection after every night to an
//! **id-agnostic fingerprint** line: branch / archive / consumed-then-pruned
//! counts plus order-independent hashes of their content. The lines are
//! compared against the committed golden file [`GOLDEN_PATH`].
//!
//! # Why id-agnostic
//!
//! `branch_id`s are drawn from an `AtomicU64` shared by rayon workers, so
//! their values are not reproducible run to run, and `lineage_id`s depend on
//! the id-counter policy (which a refactor may legitimately change). Every
//! numeric id (`lineage_id`, `branch_id`, `parent_branch_id`,
//! `ancestor_at_scan_horizon`) is therefore zeroed before hashing, and
//! per-item hashes are sorted before being combined, so neither id values nor
//! branch order influence the fingerprint. Everything else — Kalman states,
//! hypothesis weights, LLRs, track ids, step counters, designations — is
//! hashed bit for bit.
//!
//! # Blessing the golden
//!
//! When the golden file is missing, or when `FINK_FAT_BLESS_GOLDEN=1` is set,
//! the test writes the current fingerprint and passes. Bless it on a commit
//! whose behaviour is the reference, run the test a second time to confirm
//! the pipeline is deterministic, then commit the file.
//!
//! # Resume
//!
//! [`resume_from_snapshot_matches_golden_fingerprint`] runs the same nights
//! with a save/reload of the snapshot in the middle and checks it lands on
//! the same golden, that already-closed nights are refused, and that no
//! branch or lineage id is ever handed out twice.

use std::collections::HashSet;
use std::fs::File;
use std::path::{Path, PathBuf};

use camino::Utf8PathBuf;
use fink_fat_engine::{
    engine_config::EngineConfig,
    error::EngineError,
    topocentric_kf::branching::{
        ArchivedTrajectory, BranchCollection, BranchSnapshot, IdCounters, NightSession,
        SNAPSHOT_FILENAME,
    },
};
use fink_fat_eval::cli::load_data;
use photom::{
    NightId,
    observation_dataset::{ObsDataset, observation::Observation},
};
use polars::prelude::*;
use tempfile::TempDir;

const CONFIG_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/config.yaml");
const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");
const GOLDEN_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/golden/night_session_fingerprint.txt"
);
/// Environment variable forcing the golden file to be (re)written.
const BLESS_ENV: &str = "FINK_FAT_BLESS_GOLDEN";
/// Nights to run — enough for lineages to be seeded, extended, coasted,
/// N-scan pruned and archived, without paying for the whole fixture.
const N_NIGHTS: usize = 20;
/// Nights advanced before the resume test persists and reloads the snapshot.
const RESUME_AFTER: usize = N_NIGHTS / 2;

/// Night fixture files of `tests/data`, sorted by night id.
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

/// Concatenate the night fixtures into one parquet so a single
/// [`ObsDataset`](photom::observation_dataset::ObsDataset) carries a night
/// index over all of them.
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

/// 64-bit FNV-1a: a tiny, dependency-free hash whose output is stable across
/// Rust versions (unlike `DefaultHasher`), as a committed golden requires.
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Order-independent combination of per-item hashes: sort, then hash the
/// concatenated little-endian words.
fn combine_unordered(mut hashes: Vec<u64>) -> u64 {
    hashes.sort_unstable();
    let bytes: Vec<u8> = hashes.iter().flat_map(|h| h.to_le_bytes()).collect();
    fnv1a(&bytes)
}

/// Hash of a branch snapshot with every run-dependent numeric id zeroed.
fn branch_hash(mut snapshot: BranchSnapshot) -> u64 {
    snapshot.lineage_id = 0;
    snapshot.parent_branch_id = 0;
    snapshot.branch_id = 0;
    snapshot.ancestor_at_scan_horizon = 0;
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&snapshot).expect("serialize branch");
    fnv1a(&bytes)
}

/// Hash of an archived trajectory with its `lineage_id` zeroed.
fn archived_hash(archived: &ArchivedTrajectory) -> u64 {
    let mut archived = archived.clone();
    archived.lineage_id = 0;
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&archived).expect("serialize archived");
    fnv1a(&bytes)
}

/// One golden line describing the collection right after `night_id` closed.
fn night_fingerprint(night_id: NightId, collection: &BranchCollection) -> String {
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
fn load_fixture() -> (TempDir, ObsDataset, Vec<NightId>) {
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
fn night_observations(obs_dataset: &ObsDataset, night_id: NightId) -> Vec<&Observation> {
    obs_dataset
        .iter_night_observations(&night_id)
        .into_iter()
        .flatten()
        .collect()
}

/// Advance `collection` through `night_id`, as `fink-fat track` does.
fn advance<'a>(
    collection: BranchCollection<'a, 'a>,
    obs_dataset: &ObsDataset,
    night_id: NightId,
    config: &'a EngineConfig,
    ctx: &'a fink_fat_engine::engine_config::kalman_context::KalmanContext,
) -> Result<BranchCollection<'a, 'a>, EngineError> {
    let step = collection.current_step;
    collection.advance_one_night(
        &night_observations(obs_dataset, night_id),
        obs_dataset,
        config,
        ctx,
        step,
        Some(night_id),
    )
}

/// Compare fingerprint lines to the golden; (re)write it when blessing.
fn check_against_golden(lines: &[String]) {
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

/// The ids a collection carries before a night is advanced — all that
/// [`assert_fresh_ids_only`] needs, since advancing consumes the collection.
struct IdsBefore {
    branch_ids: HashSet<u64>,
    lineage_ids: HashSet<u64>,
    counters: IdCounters,
}

impl IdsBefore {
    fn of(collection: &BranchCollection) -> Self {
        Self {
            branch_ids: collection.branches.iter().map(|b| b.branch_id).collect(),
            lineage_ids: collection.branches.iter().map(|b| b.lineage_id).collect(),
            counters: collection.ids,
        }
    }
}

/// Every id of `next` absent before the night must have been drawn after the
/// counters it started from — i.e. no id is ever handed out twice — and the
/// counters themselves must never go back.
fn assert_fresh_ids_only(before: &IdsBefore, next: &BranchCollection, night_id: NightId) {
    let (prev, now) = (before.counters, next.ids);
    for branch in &next.branches {
        if !before.branch_ids.contains(&branch.branch_id) {
            assert!(
                (prev.next_branch_id..now.next_branch_id).contains(&branch.branch_id),
                "night {night_id}: branch_id {} outside the ids drawn tonight {prev:?}..{now:?}",
                branch.branch_id,
            );
        }
        if !before.lineage_ids.contains(&branch.lineage_id) {
            assert!(
                (prev.next_lineage_id..now.next_lineage_id).contains(&branch.lineage_id),
                "night {night_id}: lineage_id {} reused (counters {prev:?}..{now:?})",
                branch.lineage_id,
            );
        }
    }
    assert!(now.next_branch_id >= prev.next_branch_id);
    assert!(now.next_lineage_id >= prev.next_lineage_id);
}

#[test]
fn night_pipeline_matches_golden_fingerprint() {
    let (_tmp, obs_dataset, night_ids) = load_fixture();
    let config =
        EngineConfig::load_engine_config_validated(CONFIG_PATH).expect("load engine config");
    let ctx = config.build_context();

    let mut lines = Vec::with_capacity(night_ids.len());
    let mut collection = BranchCollection::empty();
    for &night_id in &night_ids {
        collection = advance(collection, &obs_dataset, night_id, &config, &ctx)
            .expect("advance_one_night should not fail on the fixture");
        lines.push(night_fingerprint(night_id, &collection));
    }
    check_against_golden(&lines);
}

#[test]
fn resume_from_snapshot_matches_golden_fingerprint() {
    let (tmp, obs_dataset, night_ids) = load_fixture();
    let config =
        EngineConfig::load_engine_config_validated(CONFIG_PATH).expect("load engine config");
    let ctx = config.build_context();
    let snapshot_path =
        Utf8PathBuf::from_path_buf(tmp.path().join(SNAPSHOT_FILENAME)).expect("utf8 tempdir");
    let (first, rest) = night_ids.split_at(RESUME_AFTER.min(night_ids.len()));

    let mut lines = Vec::with_capacity(night_ids.len());
    let mut collection = BranchCollection::empty();
    for &night_id in first {
        let before = IdsBefore::of(&collection);
        let next = advance(collection, &obs_dataset, night_id, &config, &ctx)
            .expect("advance_one_night should not fail on the fixture");
        assert_fresh_ids_only(&before, &next, night_id);
        assert_eq!(next.last_closed_night_id, Some(night_id));
        lines.push(night_fingerprint(night_id, &next));
        collection = next;
    }

    // Persist, then resume from disk exactly as a restarted process would.
    collection
        .save_snapshot_to_disk(&snapshot_path)
        .expect("save snapshot");
    let saved_ids: IdCounters = collection.ids;
    let saved_last_closed = collection.last_closed_night_id;
    let saved_step = collection.current_step;
    let saved_branches = collection.branches.len();
    drop(collection);
    let mut collection = BranchCollection::load_snapshot_from_disk(&snapshot_path, &ctx, &config)
        .expect("load snapshot");
    assert_eq!(collection.ids, saved_ids);
    assert_eq!(collection.last_closed_night_id, saved_last_closed);
    assert_eq!(collection.current_step, saved_step);
    assert_eq!(collection.branches.len(), saved_branches);

    // Every night already folded in is refused, so a resumed run cannot
    // double-count one — and the refused collection comes back intact.
    for &night_id in first {
        assert!(collection.is_night_closed(night_id));
        let step = collection.current_step;
        let Err(refused) = NightSession::begin(collection, Some(night_id), step) else {
            panic!("night {night_id} was already closed and must be refused");
        };
        assert!(matches!(
            refused.error,
            EngineError::NightAlreadyClosed { .. }
        ));
        collection = *refused.collection;
        assert_eq!(collection.branches.len(), saved_branches);
    }

    for &night_id in rest {
        assert!(!collection.is_night_closed(night_id));
        let before = IdsBefore::of(&collection);
        let next = advance(collection, &obs_dataset, night_id, &config, &ctx)
            .expect("advance_one_night should not fail on the fixture");
        assert_fresh_ids_only(&before, &next, night_id);
        lines.push(night_fingerprint(night_id, &next));
        collection = next;
    }

    if Path::new(GOLDEN_PATH).exists() && std::env::var_os(BLESS_ENV).is_none() {
        check_against_golden(&lines);
    }
}
