//! Non-regression guard for the per-night tracking pipeline
//! ([`BranchCollection::advance_one_night`]).
//!
//! Runs the engine night by night over the first `N_NIGHTS` nights of the
//! `tests/data` fixture and reduces the collection after every night to an
//! **id-agnostic fingerprint** line: branch / archive / consumed-then-pruned
//! counts plus order-independent hashes of their content. The lines are
//! compared against the committed golden file `GOLDEN_PATH`.
//! Fixture and fingerprint helpers live in `common/`.
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
//!
//! # Streaming
//!
//! [`streamed_visits_match_golden_fingerprint`] feeds the same nights visit
//! by visit through `fink-fat-stream`'s [`LiveNight`], each night with its
//! **own** dataset holding only that night's observations, and checks it
//! lands on the same golden: incremental insertion plus a per-night dataset
//! is the batch pipeline.

mod common;

use std::collections::HashSet;
use std::path::Path;

use camino::Utf8PathBuf;
use common::{
    BLESS_ENV, CONFIG_PATH, GOLDEN_PATH, RESUME_AFTER, check_against_golden, load_fixture,
    night_fingerprint, night_observations, observation_input,
};
use fink_fat_engine::{
    engine_config::EngineConfig,
    error::EngineError,
    topocentric_kf::branching::{
        BranchCollection, IdCounters, NightSession, SNAPSHOT_FILENAME,
        visit::group_observations_into_visits,
    },
};
use fink_fat_stream::LiveNight;
use photom::{NightId, observation_dataset::ObsDataset};

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

#[test]
fn streamed_visits_match_golden_fingerprint() {
    let (_tmp, obs_dataset, night_ids) = load_fixture();
    let config =
        EngineConfig::load_engine_config_validated(CONFIG_PATH).expect("load engine config");
    let ctx = config.build_context();
    let error_model = *obs_dataset
        .get_error_model()
        .expect("the fixture dataset carries an error model");
    let tolerance = config.advance_params.visit_epoch_tolerance_days;

    let mut lines = Vec::with_capacity(night_ids.len());
    let mut collection = BranchCollection::empty();
    for &night_id in &night_ids {
        let night_obs = night_observations(&obs_dataset, night_id);
        let mut live = LiveNight::begin(collection, night_id, error_model, &ctx, &config)
            .expect("each fixture night is new");
        for visit in group_observations_into_visits(&night_obs, tolerance) {
            let inputs = visit
                .observations
                .iter()
                .map(|o| observation_input(o))
                .collect();
            let outcome = live.advance_visit(inputs).expect("advance visit");
            assert!(
                outcome.observer_resolved,
                "night {night_id}: the night's own dataset must resolve the observer"
            );
        }
        assert_eq!(live.n_observations(), night_obs.len());
        collection = live.close().expect("close night");
        lines.push(night_fingerprint(night_id, &collection));
    }

    // Never blesses: the golden is the batch pipeline's.
    if Path::new(GOLDEN_PATH).exists() && std::env::var_os(BLESS_ENV).is_none() {
        check_against_golden(&lines);
    }
}
