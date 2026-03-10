//! Regression tests: five ground-truth trajectories that `check_reconstruction.py`
//! identified as not detected by the pipeline.
//!
//! # Test data
//!
//! Fixture parquet files in `tests/data_tests/` contain the observations
//! of exactly five trajectories extracted from the real ZTF-like survey data
//! stored in `test_exp/test_night/`.  Each file covers one night; 18 nights
//! are present (union of all nights where at least one of the five trajectories
//! was observed).
//!
//! # Fragment model
//!
//! The bounded-beam solver emits a track as soon as it reaches `min_nodes`
//! nodes, without extending it further.  A long ground-truth trajectory is
//! therefore expected to produce **multiple short fragments** rather than a
//! single continuous track.  Fragmentation is **expected and correct behavior**.
//!
//! Each test therefore verifies two properties:
//!
//! 1. **Purity** — every fragment that overlaps the GT trajectory must contain
//!    observations exclusively from that trajectory (`purity ≥ 90 %`).  An
//!    impure fragment indicates that the solver is mixing distinct objects.
//!
//! 2. **Union coverage** — the fragments together must cover at least 50 % of
//!    the GT trajectory's observations.  Low coverage indicates that either
//!    seeds were not formed on enough nights, or edges connecting those seeds
//!    are missing or filtered out.
//!
//! # Methodology
//!
//! All five tests share a single pipeline run (18 `fink-fat night-run`
//! invocations through the compiled CLI binary) via a `OnceLock`-protected
//! temp directory.  The storage directory is leaked at process level rather
//! than dropped per-test, which is safe because the OS reclaims temp files when
//! the test process exits.
//!
//! # Expected outcome
//!
//! These tests are deliberately written as *regression* checks: they are
//! expected to fail on a version of the pipeline that does not reconstruct the
//! trajectories, and to pass once the underlying issue has been fixed.  The
//! diagnostic messages embedded in each `assert!` indicate the most likely
//! cause of failure for each trajectory.

mod helpers;

use std::{
    path::{Path, PathBuf},
    sync::OnceLock,
};

use tempfile::TempDir;

use helpers::{
    config::write_test_config, fixture_night_files, fixture_nights_dir, night_run::run_night,
    reconstruction::check_reconstruction,
};

// ---------------------------------------------------------------------------
// Ground-truth trajectory IDs
// ---------------------------------------------------------------------------

/// Trajectory spanning nights 2925–2956 with 7 seedable nights.
const TRAJ_1732: i32 = 1732;
/// Trajectory spanning nights 2925–2956 with 8 seedable nights.
const TRAJ_1789: i32 = 1789;
/// Trajectory spanning nights 2925–2955 with 4 seedable nights.
const TRAJ_1882: i32 = 1882;
/// Trajectory spanning nights 2925–2955 with 9 seedable nights.
const TRAJ_2154: i32 = 2154;
/// Trajectory spanning nights 2925–2945 with 4 seedable nights.
const TRAJ_2169: i32 = 2169;

/// Minimum purity threshold for each fragment: ≥ 90 % of a fragment's
/// observations must belong to the GT trajectory under test.  A lower value
/// indicates that the solver is combining observations from distinct objects.
const MIN_PURITY: f64 = 0.9;

/// Minimum union coverage threshold: the fragments together must cover at
/// least 50 % of the GT trajectory's observations.
const MIN_UNION_COVERAGE: f64 = 0.5;

// ---------------------------------------------------------------------------
// Shared pipeline storage (one run, five assertions)
// ---------------------------------------------------------------------------

/// Path to the shared temp storage directory, initialized exactly once.
///
/// The `TempDir` is leaked (via `Box::leak`) so it outlives all tests in
/// this binary.  The OS reclaims the files when the process exits.
static SHARED_STORAGE_ROOT: OnceLock<PathBuf> = OnceLock::new();

/// Return a reference to the `storage_root` produced by the shared pipeline run.
///
/// The first call builds the temp directory, writes the config, and executes
/// `fink-fat night-run` for all fixture nights.  Subsequent calls return the
/// cached path immediately.
fn shared_storage_root() -> &'static Path {
    SHARED_STORAGE_ROOT.get_or_init(|| {
        // Leak the TempDir: it will be cleaned up when the test process exits.
        let dir: &'static TempDir =
            Box::leak(Box::new(TempDir::new().expect("create temp storage dir")));

        let config_path = dir.path().join("config.yml");
        write_test_config(dir.path(), &config_path);

        let nights = fixture_night_files();
        assert!(
            !nights.is_empty(),
            "no fixture night files found in {}",
            fixture_nights_dir().display()
        );

        for night in &nights {
            run_night(night, &config_path);
        }

        dir.path().to_path_buf()
    })
}

// ---------------------------------------------------------------------------
// Individual regression tests
// ---------------------------------------------------------------------------

/// Trajectory 1732 — 11 nights (2925–2956), 7 seedable nights.
///
/// Seedable nights: 2925 (2 obs), 2926 (2), 2930 (2), 2934 (2), 2953 (17),
/// 2954 (12), 2955 (10). Single-obs nights (not seedable): 2932, 2941, 2951,
/// 2956.
///
/// The dense cluster on nights 2953–2955 (39 obs) should produce multiple
/// fragments.  The early nights (2925–2934) should produce additional fragments
/// linked to the dense cluster via low-cost singer_cwna edges (cost ≤ 0.06).
#[test]
fn trajectory_1732_is_detected() {
    let r = check_reconstruction(shared_storage_root(), &fixture_nights_dir(), TRAJ_1732);

    assert!(
        r.has_any_fragment(),
        "Trajectory 1732 produced NO fragments at all (regression).\n\
         GT: {} obs, {} total tracks in storage.\n\
         Expected: the dense cluster (nights 2953–2955, 39 obs) should produce\n\
         at least one fragment.\n\n\
         {r}",
        r.n_gt_alerts,
        r.total_tracks_in_storage,
    );

    assert!(
        r.all_pure(MIN_PURITY),
        "Trajectory 1732: some fragments are IMPURE (regression).\n\
         Expected: every fragment belongs exclusively to this GT trajectory.\n\n\
         {r}"
    );

    assert!(
        r.union_coverage() >= MIN_UNION_COVERAGE,
        "Trajectory 1732: union coverage {:.1}% < {:.0}% (regression).\n\
         GT: {} obs across 7 seedable nights, {} fragments.\n\
         Expected: fragments together cover the dense cluster + early nights.\n\n\
         {r}",
        r.union_coverage() * 100.0,
        MIN_UNION_COVERAGE * 100.0,
        r.n_gt_alerts,
        r.n_fragments(),
    );
}

/// Trajectory 1789 — 11 nights (2925–2956), 8 seedable nights.
///
/// Seedable nights: 2925 (2 obs), 2926 (2), 2930 (2), 2934 (2), 2939 (2),
/// 2953 (16), 2954 (18), 2955 (15). Single-obs nights: 2932, 2951, 2956.
///
/// With 8 seedable nights the bounded-beam solver has more starting points
/// than for traj 1732, so union coverage should be at least as high.
#[test]
fn trajectory_1789_is_detected() {
    let r = check_reconstruction(shared_storage_root(), &fixture_nights_dir(), TRAJ_1789);

    assert!(
        r.has_any_fragment(),
        "Trajectory 1789 produced NO fragments at all (regression).\n\
         GT: {} obs, {} total tracks in storage.\n\n\
         {r}",
        r.n_gt_alerts,
        r.total_tracks_in_storage,
    );

    assert!(
        r.all_pure(MIN_PURITY),
        "Trajectory 1789: some fragments are IMPURE (regression).\n\n{r}"
    );

    assert!(
        r.union_coverage() >= MIN_UNION_COVERAGE,
        "Trajectory 1789: union coverage {:.1}% < {:.0}% (regression).\n\
         GT: {} obs across 8 seedable nights, {} fragments.\n\n\
         {r}",
        r.union_coverage() * 100.0,
        MIN_UNION_COVERAGE * 100.0,
        r.n_gt_alerts,
        r.n_fragments(),
    );
}

/// Trajectory 1882 — 8 nights (2925–2955), 4 seedable nights.
///
/// Seedable nights: 2926 (2 obs), 2953 (8), 2954 (10), 2955 (19).
/// Single-obs nights (not seedable): 2925, 2930, 2932, 2934.
///
/// Edge costs on the dense cluster:
/// - 2953→2954: Δt=1.0d, **cost 0.00** (per-fect alignment)
/// - 2953→2955: Δt=2.0d, **cost 17.23** (photometric outlier — HIGH)
/// - 2954→2955: Δt=1.0d, cost presumably low
///
/// The solver must route 2953→2954→2955 (two-hop) instead of the direct
/// 2953→2955 edge which has HIGH cost.  If it only explores the direct high-cost
/// edge, it may reject the component entirely before reaching the two-hop path.
///
/// This is a **particularly diagnostic test**: zero coverage here means the
/// pipeline fails to produce ANY track for this object, most likely because the
/// solver is being misled by the 17.23 cost edge before discovering the
/// low-cost two-hop route.
#[test]
fn trajectory_1882_is_detected() {
    let r = check_reconstruction(shared_storage_root(), &fixture_nights_dir(), TRAJ_1882);

    assert!(
        r.has_any_fragment(),
        "Trajectory 1882 produced NO fragments at all (regression).\n\
         GT: {} obs across 4 seedable nights (2926, 2953, 2954, 2955).\n\
         Total tracks in storage: {}.\n\n\
         Key diagnostic: the direct edge 2953→2955 has cost 17.23 (photometric\n\
         outlier).  The solver should find the two-hop path 2953→2954→2955\n\
         (both edges have low cost) before the direct edge causes the component\n\
         to be skipped or truncated.\n\n\
         If total_tracks_in_storage > 0 but no fragments exist, the solver is\n\
         processing other trajectories but failing to generate seeds or edges\n\
         for this one.\n\
         If total_tracks_in_storage == 0, no night produced any output at all.\n\n\
         {r}",
        r.n_gt_alerts,
        r.total_tracks_in_storage,
    );

    assert!(
        r.all_pure(MIN_PURITY),
        "Trajectory 1882: some fragments are IMPURE (regression).\n\n{r}"
    );

    assert!(
        r.union_coverage() >= MIN_UNION_COVERAGE,
        "Trajectory 1882: union coverage {:.1}% < {:.0}% (regression).\n\
         GT: {} obs across 4 seedable nights, {} fragments.\n\
         The four seedable nights form a clear 3-node minimum chain.\n\n\
         {r}",
        r.union_coverage() * 100.0,
        MIN_UNION_COVERAGE * 100.0,
        r.n_gt_alerts,
        r.n_fragments(),
    );
}

/// Trajectory 2154 — 12 nights (2925–2955), 9 seedable nights.
///
/// Seedable nights: 2925 (2 obs), 2930 (2), 2932 (2), 2934 (2), 2939 (2),
/// 2941 (2), 2953 (17), 2954 (19), 2955 (10).
/// Note: nights 2926 and 2927 have only 1 observation each → no seed.
///
/// At 9 seedable nights and 11+ valid inter-night edges this is the richest
/// trajectory in the fixture set and should fragment the most thoroughly.
#[test]
fn trajectory_2154_is_detected() {
    let r = check_reconstruction(shared_storage_root(), &fixture_nights_dir(), TRAJ_2154);

    assert!(
        r.has_any_fragment(),
        "Trajectory 2154 produced NO fragments at all (regression).\n\
         GT: {} obs, {} total tracks in storage.\n\n\
         {r}",
        r.n_gt_alerts,
        r.total_tracks_in_storage,
    );

    assert!(
        r.all_pure(MIN_PURITY),
        "Trajectory 2154: some fragments are IMPURE (regression).\n\n{r}"
    );

    assert!(
        r.union_coverage() >= MIN_UNION_COVERAGE,
        "Trajectory 2154: union coverage {:.1}% < {:.0}% (regression).\n\
         GT: {} obs across 9 seedable nights, {} fragments.\n\
         Expected: this is the richest trajectory — high coverage is expected.\n\n\
         {r}",
        r.union_coverage() * 100.0,
        MIN_UNION_COVERAGE * 100.0,
        r.n_gt_alerts,
        r.n_fragments(),
    );
}

/// Trajectory 2169 — 8 nights (2925–2945), 4 seedable nights.
///
/// Seedable nights: 2925 (2 obs), 2936 (2), 2938 (2), 2940 (2).
/// Single-obs nights: 2927, 2934, 2943, 2945.
///
/// The minimum valid chain is exactly 3 nodes: 2936→2938→2940 (all consecutive
/// by 2 nights, within max_gap_nights=10).  The night-2925 seed is isolated:
/// the gap to 2936 is 11 nights which exceeds max_gap_nights=10.
///
/// The solver must find this 3-node chain in isolation with no anchor from
/// earlier nights.
#[test]
fn trajectory_2169_is_detected() {
    let r = check_reconstruction(shared_storage_root(), &fixture_nights_dir(), TRAJ_2169);

    assert!(
        r.has_any_fragment(),
        "Trajectory 2169 produced NO fragments at all (regression).\n\
         GT: {} obs, {} total tracks in storage.\n\
         Expected: the 3-node chain 2936→2938→2940 satisfies min_nodes=3.\n\n\
         {r}",
        r.n_gt_alerts,
        r.total_tracks_in_storage,
    );

    assert!(
        r.all_pure(MIN_PURITY),
        "Trajectory 2169: some fragments are IMPURE (regression).\n\n{r}"
    );

    assert!(
        r.union_coverage() >= MIN_UNION_COVERAGE,
        "Trajectory 2169: union coverage {:.1}% < {:.0}% (regression).\n\
         GT: {} obs across 4 seedable nights, {} fragments.\n\
         The chain 2936→2938→2940 covers 6 observations (3 seeds from 3 nights).\n\n\
         {r}",
        r.union_coverage() * 100.0,
        MIN_UNION_COVERAGE * 100.0,
        r.n_gt_alerts,
        r.n_fragments(),
    );
}
