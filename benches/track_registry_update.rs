//! # Benchmark — `TrackRegistry::update_from_link`
//!
//! ## Goal
//! Measure the cost of the **hot loop** inside `update_from_link`:
//! iterating over `Assignment`s (left→right), **merging trajectories** with DSU,
//! and **assigning all member detections** for both seeds under a given
//! `DetectConflictPolicy`.
//!
//! In short, this benchmark approximates the per-night **link-and-grow**
//! trajectory step at LSST scale (millions of detections, ~10⁵ seeds per night).
//!
//! ## What is measured
//! - DSU operations (`find`, `union_keep_min`) when merging `TrajectoryId`s.
//! - Detection assignments in the hot path (hash map/set insertions).
//! - Policy-dependent behavior:
//!   - `KeepFirst`: branch that avoids extra inserts when a key is already set.
//!   - `Overwrite`: replace-by-singleton fast path.
//!   - `Combinatorial`: multi-id accumulation with dedupe.
//!
//! > Note: This benchmark **does not** read from `AlertStore.alerts` in the loop;
//! > membership comes from `SeedNode.members` only (synthetic).
//!
//! ## Scaling knobs (environment variables)
//! - `N_SEEDS_LEFT`      — number of seeds on the left night (default: `50_000`)
//! - `N_SEEDS_RIGHT`     — number of seeds on the right night (default: `50_000`)
//! - `MEMBERS_PER_SEED`  — number of detections per seed (default: `3`)
//! - `N_LINKS`           — number of 1-to-1 assignments (default: `min(left,right)`)
//! - `POLICY`            — `KeepFirst | Overwrite | Combinatorial` (default: `KeepFirst`)
//!
//! Examples:
//!
//! ```bash
//! cargo bench --bench track_registry_update
//!
//! N_SEEDS_LEFT=200000 N_SEEDS_RIGHT=200000 MEMBERS_PER_SEED=4 N_LINKS=180000 POLICY=Overwrite \
//!   cargo bench --bench track_registry_update
//! ```
//!
//! ## Synthetic data model
//! `make_snapshot(night_id, n_seeds, members_per_seed)` builds a `NightSnapshot` where:
//! - each `SeedNode` has `members_per_seed` **contiguous** `AlertId`s,
//! - kinematics/photometry are plausible but irrelevant to the loop,
//! - seeds are index-aligned so `Assignment { from: i, to: i }` hits valid pairs.
//!
//! `make_assignments(n_links, ...)` generates `from i → to i` for `i < n_links`.
//!
//! ## Reproducible runs (Criterion)
//! Criterion uses warm-up and statistical sampling. For stable results:
//! - **Disable logs** inside the code under test (see *Noisy I/O* below).
//! - Pin to a CPU / performance governor:
//!   ```bash
//!   # Linux examples
//!   sudo cpupower frequency-set -g performance
//!   taskset -c 2 cargo bench --bench track_registry_update
//!   ```
//! - Run multiple times and compare medians.
//!
//! ## Profiling recipes
//! ### Linux `perf` (cycles, branches, cache)
//! ```bash
//! RUSTFLAGS='-C debuginfo=2 -C target-cpu=native' \
//!   perf record -g --call-graph dwarf -- \
//!   cargo bench --bench track_registry_update
//!
//! perf report            # TUI
//! perf script > out.perf # Raw stack dump
//! ```
//!
//! ### Firefox Profiler (interactive flamegraph from `perf`)
//! ```bash
//! perf script > linux-perf.txt
//! # Ouvrir https://profiler.firefox.com/ et importer linux-perf.txt
//! ```
//!
//! ### Flamegraph (static SVG)
//! ```bash
//! cargo install flamegraph
//! RUSTFLAGS='-C debuginfo=2 -C target-cpu=native' \
//!   cargo flamegraph --bench track_registry_update
//! ```
//!
//! ## Interpreting results
//! - If **DSU** dominates: check `dsu_find` path compression efficiency and
//!   union-by-size balance. Hot paths should show shallow trees.
//! - If **assignments** dominate: expect hotspots in `assign_seed_members` /
//!   `assign_detection` (hash lookup/insert, policy branches).
//!   - `KeepFirst` should hit the **early-return** path frequently once filled.
//!   - `Overwrite` should show **replace-by-singleton** writes (no `clear()`).
//!   - `Combinatorial` will scale with the **distinct #traj/DetectKey**.
//!
//! **Throughput heuristics** (ballpark):
//! - Total detection assignments ≈ `N_LINKS × (members_left + members_right)`.
//! - Memory grows with the number of distinct `(DetectKey → TrajectoryId)` pairs
//!   plus DSU maps (one entry per trajectory ever created).
//!
//! ## Noisy I/O (disable prints)
//! `println!` inside `update_from_link` will distort timings. Prefer a feature flag:
//!
//! ```rust
//! // in the library code
//! #[cfg(feature = "fat-logs")]
//! println!("Updating from link: ...");
//! ```
//!
//! puis bench sans logs :
//! ```bash
//! cargo bench --bench track_registry_update --no-default-features
//! ```
//!
//! ## Advanced: system counters
//! ```bash
//! perf stat -d -d -d \
//!   env N_SEEDS_LEFT=200000 N_SEEDS_RIGHT=200000 MEMBERS_PER_SEED=6 POLICY=Combinatorial \
//!   cargo bench --bench track_registry_update
//! # Look at: instructions, cycles, IPC, branches-misses, LLC-load-misses
//! ```
//!
//! ## Common pitfalls
//! - Mixed builds (debug symbols off) make stacks unusable → set `-C debuginfo=2`.
//! - Frequency scaling / thermal throttling skews results → pin CPU & governor.
//! - Changing `POLICY` changes algorithmic shape; compare policies at the same scale.
//!
//! ---

use std::{cmp::min, env, sync::Arc};

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

use fink_fat::{
    alerts::{Alert, AlertStore},
    propagation::{
        engine::LinkResult,
        features::{SeedId, SeedNode},
        linking::NightSnapshot,
        solver::Assignment,
    },
    track_registry::{DetectConflictPolicy, TrackRegistry},
    AlertId, NightId,
};

fn env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_policy() -> DetectConflictPolicy {
    match env::var("POLICY")
        .unwrap_or_else(|_| "KeepFirst".to_string())
        .as_str()
    {
        "KeepFirst" => DetectConflictPolicy::KeepFirst,
        "Overwrite" => DetectConflictPolicy::Overwrite,
        "Combinatorial" => DetectConflictPolicy::Combinatorial,
        _ => DetectConflictPolicy::KeepFirst,
    }
}

fn make_snapshot(night_id: NightId, n_seeds: usize, members_per_seed: usize) -> NightSnapshot {
    let mut seeds = Vec::with_capacity(n_seeds);
    let mut next_alert_idx: usize = 0;

    for sid in 0..n_seeds {
        let mut members = Vec::with_capacity(members_per_seed);
        for _ in 0..members_per_seed {
            members.push(AlertId::from(next_alert_idx));
            next_alert_idx += 1;
        }

        let seed = SeedNode::new(
            sid as u64,
            night_id,
            60_000.0,      // MJD TT
            [0.0, 0.0],    // rad
            [1e-4, -1e-4], // rad/day
            [[(1e-6_f64).powi(2), 0.0], [0.0, (1e-6_f64).powi(2)]],
            [[(1e-6_f64).powi(2), 0.0], [0.0, (1e-6_f64).powi(2)]],
            None,
            1000.0,
            50.0,
            2,
            members_per_seed as u16,
            members,
            1.0, // rad
            0.1, // rad
            1.0, // rad
            0.1, // rad
        );
        seeds.push(seed);
    }

    NightSnapshot {
        night_id,
        pairs: Vec::new(),
        triplets: Vec::new(),
        seeds,
    }
}

fn make_assignments(n_links: usize, max_left: usize, max_right: usize) -> Vec<Assignment> {
    let n = min(n_links, min(max_left, max_right));
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let from: SeedId = i as SeedId;
        let to: SeedId = i as SeedId;
        v.push(Assignment {
            from,
            to,
            cost: 0.0,
        });
    }
    v
}

fn make_dummy_store() -> AlertStore {
    AlertStore {
        start_mjd: 60_000.0,
        alerts: Vec::<Alert>::new(),
    }
}

fn make_link_result(
    night_left: NightId,
    night_right: NightId,
    matches: Vec<Assignment>,
) -> LinkResult {
    LinkResult {
        night_left,
        night_right,
        matches,
        edges_kept: 0,
    }
}

pub fn bench_update_from_link(c: &mut Criterion) {
    let n_left = env_usize("N_SEEDS_LEFT", 200_000);
    let n_right = env_usize("N_SEEDS_RIGHT", 200_000);
    let members_per_seed = env_usize("MEMBERS_PER_SEED", 6);
    let policy = env_policy();

    let n_links_req = env_usize("N_LINKS", usize::MAX);
    let n_links = min(n_links_req, min(n_left, n_right));

    let left_id: NightId = 3156;
    let right_id: NightId = 3157;

    let left_snap = make_snapshot(left_id, n_left, members_per_seed);
    let right_snap = make_snapshot(right_id, n_right, members_per_seed);
    let link = make_link_result(
        left_id,
        right_id,
        make_assignments(n_links, n_left, n_right),
    );

    let left_store = Arc::new(make_dummy_store());
    let right_store = Arc::new(make_dummy_store());

    let group_name = format!(
        "update_from_link/{:?}/links={}/mps={}",
        policy, n_links, members_per_seed
    );
    let mut group = c.benchmark_group(group_name);

    group.bench_function("call", |b| {
        b.iter_batched(
            || {
                let reg = TrackRegistry::new(policy);
                (
                    reg,
                    &left_snap,
                    &right_snap,
                    &link,
                    &*left_store,
                    &*right_store,
                )
            },
            |(mut reg, l, r, link, ls, rs)| {
                let l = black_box(l);
                let r = black_box(r);
                let link = black_box(link);

                reg.update_from_link(l, r, link, ls, rs);
                black_box(reg);
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(benches, bench_update_from_link);
criterion_main!(benches);
