//! # TrackRegistry — Trajectory Reconstruction & Union-Find (DSU)
//!
//! ## Overview
//! `TrackRegistry` is the central in-memory index that reconstructs asteroid
//! trajectories from inter-night **seed linking** results. It stores:
//! - a *seed → trajectory* mapping (`seed_to_traj`),
//! - a *detection → {trajectories}* mapping (`detect_to_traj`),
//! - and a **Disjoint-Set Union** (a.k.a. **Union-Find**, DSU) over trajectory ids,
//!   to merge them efficiently when cross-night linking discovers continuity.
//!
//! The registry supports **conflict policies** for re-assigning detections (keep-first,
//! overwrite, or combinatorial), guarantees **deterministic export ordering**, and
//! avoids global rewrites when merging trajectories by relying on **lazy canonicalization**.
//!
//! ## Why a DSU?
//! Traditional "merge by relabeling" would scan and rewrite every occurrence of a
//! trajectory id across all seeds and detections, which is **O(total_assignments)** per merge
//! and quickly becomes prohibitive at LSST alert scale.
//!
//! Instead, we maintain a DSU over `TrajectoryId`s. A merge becomes:
//!
//! ```text
//! union_keep_min(a, b) → keep = min(a, b)
//! ```
//!
//! Subsequent reads **canonicalize** via `find()`: whenever we *store* or *read* a
//! trajectory id, we map it to its **representative** (the DSU root). This makes
//! merging **near O(α(N))** (inverse Ackermann) and eliminates full-table scans.
//!
//! ## Data Model
//!
//! - `SeedKey = (night_id, seed_id)` identifies a *seed* (e.g., pair/triplet features).
//! - `DetectKey = (night_id, alert_id)` identifies a *detection* (single alert).
//! - `TrajectoryId(u64)` labels a *trajectory* (monotonic, starts at 1).
//!
//! Mappings:
//!
//! ```text
//! seed_to_traj:    BTreeMap<SeedKey, TrajectoryId>          // ordered, stable
//! detect_to_traj:  HashMap<DetectKey, HashSet<TrajectoryId>> // hot path: fast inserts
//! DSU:             parent: HashMap<TrajectoryId, TrajectoryId>, size: HashMap<...>
//! ```
//!
//! We **store canonical ids** (DSU representatives) whenever possible
//! (on writes), and **canonicalize on reads** (with a non-mutating `find`) to avoid
//! stale ids leaking to outputs.
//!
//! ## Conflict Policies
//!
//! - `KeepFirst`: once a detection is assigned, future assignments are ignored.
//! - `Overwrite`: last writer wins; the detection set becomes a singleton.
//! - `Combinatorial`: the detection belongs to multiple trajectories (deduped by set).
//!
//! These policies apply **per detection key** when calling `assign_detection` or via
//! `assign_seed_members`.
//!
//! ## DSU — Disjoint-Set Union (Union-Find)
//!
//! A DSU maintains a forest of rooted trees over `TrajectoryId`s where each tree
//! represents one equivalence class (merged trajectory). Operations:
//!
//! - **find(x)**: follow parent pointers to the root (representative). We use:
//!   - `dsu_find(&mut self, x)` with **path compression** in hot write paths,
//!   - `dsu_find_ro(&self, x)` (read-only) in getters to avoid mutability.
//! - **union(a, b)**: attach one root under another, using **union by size** (or rank)
//!   to keep trees shallow. Here we **always keep the *minimum id*** as the *root*
//!   for deterministic outputs.
//!
//! ### ASCII sketch
//!
//! ```text
//!   Before merges:
//!
//!     t5       t7       t9
//!     |        |        |
//!    (root)   (root)   (root)
//!
//!   union_keep_min(t7, t9):
//!     - representative is min(7, 9) = 7
//!     - parent[9] = 7, size[7] += size[9]
//!
//!   After:
//!
//!     t5       t7
//!              |
//!             t9
//!
//!   find(t9) → 7
//! ```
//!
//! **Key property:** we never scan the registry to "relabel" from `t9` to `t7`. All
//! reads see `t9` as `7` through `find()`, and all **new writes store canonical ids**.
//!
//! ## Building Trajectories (Algorithmic Pipeline)
//!
//! **Seeding a night (optional)**: for a first night with no links, we call
//! `seed_all_of_snapshot`, which creates one trajectory per seed and assigns all
//! member detections to it.
//!
//! **Linking nights**: `update_from_link(left, right, link)` applies the inter-night
//! matching `(from, to)` over seed indices:
//!
//! 1. `ensure_seed_traj((left.night_id, left.seed_id))  → lt`
//! 2. `ensure_seed_traj((right.night_id, right.seed_id)) → rt`
//! 3. `keep = merge_traj(lt, rt)` (DSU union, **keep min id**)
//! 4. `assign_seed_members(left.night_id, left_seed, keep, ...)`
//! 5. `assign_seed_members(right.night_id, right_seed, keep, ...)`
//!
//! Result: all detections of both seeds now point (via `find`) to the **same trajectory
//! representative**. Repeating across consecutive nights progressively grows
//! trajectories over time.
//!
//! ## Complexity & Performance
//!
//! - `merge_traj`: near **O(α(N))** per merge (**no global scans**).
//! - `assign_seed_members`: amortized **O(1)** insertions on the hot map/set
//!   (thanks to `AHashMap/AHashSet`), plus `find` cost.
//! - `collect_detection_columns` / iteration: deterministic ordering is enforced
//!   by sorting keys and ids **at export time**, which is off the hot write path.
//!
//! **Memory** scales with the number of distinct `(DetectKey → TrajectoryId)` pairs,
//! plus DSU overhead (one parent/size per distinct trajectory id ever created).
//!
//! ## Deterministic Export
//! Although `detect_to_traj` is hash-backed for speed, we preserve **stable output**
//! by sorting keys `(night_id, alert_id)` and then sorting canonical trajectory ids
//! for each key. This makes outputs reproducible across runs given the same inputs.
//!
//! ## Invariants
//!
//! - DSU representative is always the **minimum id** of its set (deterministic).
//! - Public getters (`get_trajs_for_detection`, `get_traj_for_seed`, `iter_detection_assignments`)
//!   return **canonical** ids (post-`find`), sorted & deduped where relevant.
//! - Under `KeepFirst`, the detection set cardinality is ≤ 1; under `Overwrite` it is 1;
//!   under `Combinatorial` it can be ≥ 1 (deduped).
//!
//! ## Failure Modes / Error Handling
//!
//! - `collect_detection_columns` requires a `store` for each `night_id` present
//!   in the detection keys. Missing stores trigger `FinkFatError::TrackExportError`.
//! - `update_from_link` asserts (in debug) that snapshot ids match the link header
//!   and that seed indices are consistent.
//!
//! ## Example (pseudo-code)
//!
//! ```rust, ignore
//!
//! // 1) Create a registry with a conflict policy.
//! let mut reg = TrackRegistry::new(DetectConflictPolicy::Combinatorial);
//!
//! // 2) Seed night N (optional bootstrap).
//! reg.seed_all_of_snapshot(&nightN, &storeN);
//!
//! // 3) Link N → N+1 using a solver result:
//! reg.update_from_link(&nightN, &nightNp1, &link, &storeN, &storeNp1);
//!
////! // 4) Export to Python (DataFrame):
//! let cols = reg.collect_detection_columns(&stores_by_night)?;
//! // or
//! let py_dict = reg.export_detection_dict_py(py, &stores_by_night)?;
//! ```
//!
//! ## Visual Summary
//!
//! ```text
//! Seeds (night N)              Seeds (night N+1)
//!  sN0  sN1  sN2   --link-->    sN1' sN2' sN3'
//!   |    |    |                 |     |     |
//!  members per seed           members per seed
//!
//! ensure_seed_traj(sN1) → t17
//! ensure_seed_traj(sN1') → t29
//! merge_traj(t17, t29) → keep = min(17, 29) = 17
//! assign_seed_members(N,   sN1,  t17, storeN)
//! assign_seed_members(N+1, sN1', t17, storeN+1)
//!
//! DSU parent:
//!   parent[29] = 17 (representative)
//!
//! Reads/exports:
//!   Any occurrence of 29 canonicalizes to 17 via dsu_find/_ro.
//!   Export sorts keys and tids → deterministic rows.
//! ```
//!
//! ## Practical Tips
//!
//! - For extremely large batches, call `reserve_for_link` before `update_from_link`
//!   to reduce hash reallocation overhead.
//! - If you export **massive** tables repeatedly, you may consider (optionally)
//!   “canonicalizing all” once (rewrite stored ids to representatives) to make
//!   subsequent `find` calls trivial; but this is not necessary for correctness.
//!
//! ## See Also
//! - [`DetectConflictPolicy`] — conflict resolution semantics.
//! - [`TrackRegistry::merge_traj`] — DSU union with min representative.
//! - [`TrackRegistry::iter_detection_assignments`] — stable flattened rows.
//! - [`TrackRegistry::collect_detection_columns`] — typed export buffers.
//! - [`TrackRegistry::export_detection_dict_py`] — Python dict for DataFrame.

use std::{collections::BTreeMap, sync::Arc};

use ahash::{AHashMap, AHashSet};

use pyo3::{
    pyclass,
    types::{PyDict, PyDictMethods},
    Py,
};

use crate::{
    alerts::AlertStore,
    errors::FinkFatError,
    propagation::{
        engine::LinkResult,
        features::{SeedId, SeedNode},
        linking::NightSnapshot,
        solver::Assignment,
    },
    AlertId, NightId,
};

/// Unique trajectory identifier.
///
/// Overview
/// --------
/// A simple newtype wrapper around `u64` used to label trajectories in a
/// monotonically increasing and human-friendly way (starts at `1`).
///
/// Rationale
/// ---------
/// Using a newtype instead of a raw integer improves type-safety and makes
/// function signatures self-documenting.
///
/// Methods
/// -------
/// - [`TrajectoryId::raw`] — Access to the underlying `u64`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TrajectoryId(pub u64);

impl TrajectoryId {
    /// Return the underlying numeric identifier (monotonic `u64`).
    ///
    /// Returns
    /// -------
    /// `u64`
    ///     The raw trajectory id.
    #[inline]
    pub fn raw(self) -> u64 {
        self.0
    }
}

/// Global key for a detection across nights.
///
/// Overview
/// --------
/// A stable composite key that identifies a **single detection** by its
/// `(night_id, alert_id)` pair. This is used as the key in the
/// detection→trajectory registry.
///
/// Notes
/// -----
/// - Keys are `Ord` to allow deterministic order when needed (e.g., export).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DetectKey {
    pub night_id: NightId,
    pub alert_id: AlertId,
}

/// Global key for a seed across nights.
///
/// Overview
/// --------
/// Uniquely identifies a **seed** during inter-night linking by the pair
/// `(night_id, seed_id)`. `seed_id` matches `SeedNode.seed_id` within an
/// extraction batch.
///
/// Notes
/// -----
/// - `SeedKey` is `Ord`, which makes it suitable for deterministic maps.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SeedKey {
    pub night_id: NightId,
    /// Matches `SeedNode.seed_id` (0..N-1 for that extraction batch).
    pub seed_id: SeedId,
}

/// Conflict policy for assigning the *same* detection to trajectories.
///
/// Variants
/// --------
/// - `KeepFirst` — Keep the first trajectory that claimed this detection
///   (idempotent, stable).
/// - `Overwrite` — Always replace any previous trajectory with the most recent
///   one (last writer wins).
/// - `Combinatorial` — Allow a detection to belong to **several** trajectories
///   simultaneously (useful for exploratory or ambiguous linking). The pair
///   `(DetectKey, TrajectoryId)` remains unique.
///
/// Python
/// ------
/// Exposed as an enum in the `fink_fat` module.
#[pyclass(eq, eq_int, module = "fink_fat")]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DetectConflictPolicy {
    KeepFirst = 0,
    Overwrite = 1,
    Combinatorial = 2,
}

/* ------------------------ TrackRegistry ------------------------ */

/// Registry mapping seeds and detections to trajectory ids.
///
/// Overview
/// --------
/// `TrackRegistry` is the central structure that:
/// - creates/stores per-seed trajectory ids,
/// - assigns detections to trajectories under a configurable **conflict policy**,
/// - merges (unifies) trajectories efficiently via a **Union-Find (DSU)**,
/// - exposes query/iteration utilities and export helpers.
///
/// Storage layout
/// --------------
/// - `seed_to_traj` — forward map (ordered) from `SeedKey` to `TrajectoryId`.
/// - `detect_to_traj` — forward map from `DetectKey` to a **set** of
///   `TrajectoryId` (hash-backed for performance).
/// - `parent/size` — Union-Find (DSU) structure storing **representatives** of
///   trajectory sets and ranks/sizes for near-constant merges.
///
/// Design choices
/// --------------
/// - **Canonicalization on write**: when writing any trajectory id, the DSU
///   representative is used. This reduces stale ids in storage.
/// - **Canonicalization on read**: all public getters return **canonical**
///   ids (`find`), without mutating storage (uses a read-only find).
/// - **Deterministic export**: even though the hot-path uses hash maps,
///   external iteration for export is made deterministic by sorting keys/ids.
///
/// Conflict policies
/// -----------------
/// See [`DetectConflictPolicy`] for the behavior on repeated assignments.
///
/// Performance
/// -----------
/// - DSU `merge_traj` runs in near O(α(N)) and never scans all detections.
/// - `detect_to_traj` is hash-based to reduce per-insert cost during
///   intra-night member assignment.
///
/// Python
/// ------
/// This registry is not directly exposed to Python, but its exports are used to
/// build `pandas.DataFrame`s on the Python side.
///
/// See also
/// --------
/// - [`TrackRegistry::collect_detection_columns`] — Build typed export columns.
/// - [`TrackRegistry::export_detection_dict_py`] — Return a `dict` ready for `pandas.DataFrame`.
#[derive(Clone, Debug)]
pub struct TrackRegistry {
    next_id: TrajectoryId,

    // Forward maps:
    seed_to_traj: BTreeMap<SeedKey, TrajectoryId>,
    // Hot path: hash map / set to reduce per-insert cost.
    detect_to_traj: AHashMap<DetectKey, AHashSet<TrajectoryId>>,

    conflict: DetectConflictPolicy,

    // DSU / Union-Find:
    parent: AHashMap<TrajectoryId, TrajectoryId>,
    size: AHashMap<TrajectoryId, u32>,
}

impl Default for TrackRegistry {
    /// Construct a registry with sane defaults.
    ///
    /// Defaults
    /// --------
    /// - `next_id = TrajectoryId(1)` (human-readable ids),
    /// - empty maps,
    /// - `conflict = DetectConflictPolicy::KeepFirst`.
    fn default() -> Self {
        Self {
            next_id: TrajectoryId(1),
            seed_to_traj: BTreeMap::new(),
            detect_to_traj: AHashMap::new(),
            conflict: DetectConflictPolicy::KeepFirst,

            parent: AHashMap::new(),
            size: AHashMap::new(),
        }
    }
}

impl TrackRegistry {
    /// Create a registry with the desired conflict policy.
    ///
    /// Arguments
    /// ---------
    /// `conflict` : DetectConflictPolicy
    ///     Conflict resolution mode for detection assignments.
    ///
    /// Returns
    /// -------
    /// `Self`
    ///     A new registry instance.
    pub fn new(conflict: DetectConflictPolicy) -> Self {
        Self {
            conflict,
            ..Default::default()
        }
    }

    /* ------------------------ DSU helpers ------------------------ */

    /// Create a new DSU singleton set if missing.
    ///
    /// Arguments
    /// ---------
    /// `t` : TrajectoryId
    ///     The trajectory id to initialize in the parent/size maps.
    ///
    /// Notes
    /// -----
    /// Idempotent: calling this several times for the same id is cheap.
    #[inline]
    fn dsu_make_set(&mut self, t: TrajectoryId) {
        // Create singleton set only if not present.
        self.parent.entry(t).or_insert(t);
        self.size.entry(t).or_insert(1);
    }

    /// Find with path compression (mutable).
    ///
    /// Overview
    /// --------
    /// Returns the DSU **representative** of `t` and performs path compression
    /// to speed up future queries.
    ///
    /// Arguments
    /// ---------
    /// `t` : TrajectoryId
    ///     Any trajectory id in the union-find forest.
    ///
    /// Returns
    /// -------
    /// `TrajectoryId`
    ///     The canonical (representative) id.
    #[inline]
    fn dsu_find(&mut self, t: TrajectoryId) -> TrajectoryId {
        let p = *self.parent.get(&t).unwrap_or(&t);
        if p == t {
            return t;
        }
        let root = self.dsu_find(p);
        self.parent.insert(t, root);
        root
    }

    /// Read-only find (no path compression).
    ///
    /// Overview
    /// --------
    /// Follows parent links until a root is found, without mutating the DSU.
    /// This is used in read-only methods to keep their signature as `&self`.
    ///
    /// Arguments
    /// ---------
    /// `t` : TrajectoryId
    ///     Any trajectory id.
    ///
    /// Returns
    /// -------
    /// `TrajectoryId`
    ///     The canonical (representative) id.
    #[inline]
    fn dsu_find_ro(&self, mut t: TrajectoryId) -> TrajectoryId {
        // Follow parents until a root (parent[t] == t) or missing entry.
        while let Some(&p) = self.parent.get(&t) {
            if p == t {
                return t;
            }
            t = p;
        }
        t
    }

    /// Union by size; keep the **minimum id** as the representative.
    ///
    /// Overview
    /// --------
    /// Merges the sets of `a` and `b`, choosing the **numerically smallest**
    /// id as the **deterministic** representative (stable outputs).
    ///
    /// Arguments
    /// ---------
    /// `a`, `b` : TrajectoryId
    ///     Trajectory ids to merge.
    ///
    /// Returns
    /// -------
    /// `TrajectoryId`
    ///     The kept representative (minimum id).
    ///
    /// Notes
    /// -----
    /// - Uses *union-by-size* to keep trees shallow.
    /// - Near O(α(N)) amortized complexity.
    #[inline]
    fn dsu_union_keep_min(&mut self, a: TrajectoryId, b: TrajectoryId) -> TrajectoryId {
        let mut ra = self.dsu_find(a);
        let mut rb = self.dsu_find(b);
        if ra == rb {
            return ra;
        }

        // Deterministic representative: min id
        let keep = ra.min(rb);
        let drop = ra.max(rb);

        // Union by size to keep trees shallow.
        let sa = *self.size.get(&ra).unwrap_or(&1);
        let sb = *self.size.get(&rb).unwrap_or(&1);

        // Ensure (ra, sa) refer to `keep`
        if keep != ra {
            std::mem::swap(&mut ra, &mut rb);
        }

        let keep_size = self.size.entry(keep).or_insert(sa);
        *keep_size = sa + sb;
        self.parent.insert(drop, keep);

        keep
    }

    /* ------------------------ Id generation ------------------------ */

    /// Generate a fresh trajectory id and initialize it in the DSU.
    ///
    /// Returns
    /// -------
    /// `TrajectoryId`
    ///     A new, unique id (monotonic).
    #[inline]
    fn fresh_id(&mut self) -> TrajectoryId {
        let id = self.next_id;
        self.next_id.0 += 1;
        self.dsu_make_set(id);
        id
    }

    /* ------------------------ Seed / detection writes ------------------------ */

    /// Ensure a seed has a trajectory id and return it (create if missing).
    ///
    /// Overview
    /// --------
    /// If `key` is new, allocates a fresh id. Then, **canonicalizes** the id
    /// using DSU `find` and stores it back to avoid stale ids.
    ///
    /// Arguments
    /// ---------
    /// `key` : SeedKey
    ///     The global key for this seed.
    ///
    /// Returns
    /// -------
    /// `TrajectoryId`
    ///     The canonical trajectory id for this seed.
    pub fn ensure_seed_traj(&mut self, key: SeedKey) -> TrajectoryId {
        let fresh_id = self.fresh_id();
        let t = *self.seed_to_traj.entry(key).or_insert(fresh_id);
        let r = self.dsu_find(t);
        if r != t {
            self.seed_to_traj.insert(key, r);
        }
        r
    }

    /// Assign a single detection to a trajectory under the conflict policy.
    ///
    /// Overview
    /// --------
    /// Inserts a `(DetectKey → TrajectoryId)` association respecting the current
    /// [`DetectConflictPolicy`]. Trajectory ids are **canonicalized** (DSU `find`)
    /// before insertion, which prevents stale ids from accumulating.
    ///
    /// Arguments
    /// ---------
    /// `dk` : DetectKey
    ///     Target detection key `(night_id, alert_id)`.
    /// `traj` : TrajectoryId
    ///     Trajectory id to assign (can be a non-representative; canonicalized inside).
    ///
    /// Policy details
    /// --------------
    /// - `KeepFirst` — If an entry already exists and is non-empty, do nothing.
    /// - `Overwrite` — Replace the set by a singleton containing the new id.
    /// - `Combinatorial` — Insert and deduplicate by hash set.
    #[inline]
    fn assign_detection(&mut self, dk: DetectKey, traj: TrajectoryId) {
        use DetectConflictPolicy::*;
        let r = self.dsu_find(traj);

        match self.conflict {
            KeepFirst => {
                // Fast path: if an entry exists and is not empty, keep it as-is.
                if let Some(existing) = self.detect_to_traj.get(&dk) {
                    if !existing.is_empty() {
                        return;
                    }
                }
                self.detect_to_traj.entry(dk).or_default().insert(r);
            }
            Overwrite => {
                // Replace by a singleton set (avoid clear + reinserts).
                let mut s = AHashSet::with_capacity(1);
                s.insert(r);
                self.detect_to_traj.insert(dk, s);
            }
            Combinatorial => {
                // HashSet dedups; we insert the canonical id.
                self.detect_to_traj.entry(dk).or_default().insert(r);
            }
        }
    }

    /// Assign all *member detections* of a seed to a trajectory id.
    ///
    /// Arguments
    /// ---------
    /// `night_id` : NightId
    ///     Night identifier for the detection keys.
    /// `seed` : &SeedNode
    ///     Seed whose members are `(AlertId)` indices into `AlertStore.alerts`.
    /// `traj` : TrajectoryId
    ///     Target trajectory id (canonicalized internally).
    /// `_store` : &AlertStore
    ///     Alert store for the night; only used for context/validation upstream.
    ///
    /// Notes
    /// -----
    /// - No-op if `seed.members` is empty.
    #[inline]
    pub fn assign_seed_members(
        &mut self,
        night_id: NightId,
        seed: &SeedNode,
        traj: TrajectoryId,
        _store: &AlertStore,
    ) {
        if seed.members.is_empty() {
            return;
        }
        for &aid in &seed.members {
            let dk = DetectKey {
                night_id,
                alert_id: aid,
            };
            self.assign_detection(dk, traj);
        }
    }

    /* ------------------------ Merging ------------------------ */

    /// Merge two trajectories, keeping the **minimum id** as representative.
    ///
    /// Overview
    /// --------
    /// DSU union with deterministic representative (min); runs in near O(α(N)).
    /// No global scans are performed.
    ///
    /// Arguments
    /// ---------
    /// `a`, `b` : TrajectoryId
    ///     Trajectory ids to merge.
    ///
    /// Returns
    /// -------
    /// `TrajectoryId`
    ///     The kept representative (minimum id).
    #[inline]
    pub fn merge_traj(&mut self, a: TrajectoryId, b: TrajectoryId) -> TrajectoryId {
        self.dsu_union_keep_min(a, b)
    }

    /* ------------------------ Capacity hint ------------------------ */

    /// Reserve capacity in the detection map before a large link update (heuristic).
    ///
    /// Overview
    /// --------
    /// Attempts to reduce hash-map rehashing during `update_from_link` by
    /// reserving an approximate number of entries based on:
    /// `(mean members per seed on left + right) × |matches|`.
    ///
    /// Arguments
    /// ---------
    /// `left`, `right` : &NightSnapshot
    ///     Snapshots for consecutive nights.
    /// `link` : &LinkResult
    ///     Matching between `left.seeds` and `right.seeds`.
    pub fn reserve_for_link(
        &mut self,
        left: &NightSnapshot,
        right: &NightSnapshot,
        link: &LinkResult,
    ) {
        let m_left = left.seeds.first().map(|s| s.members.len()).unwrap_or(0);
        let m_right = right.seeds.first().map(|s| s.members.len()).unwrap_or(0);
        let approx = link
            .matches
            .len()
            .saturating_mul(m_left.saturating_add(m_right))
            .max(1);
        self.detect_to_traj.reserve(approx);
    }

    /* ------------------------ Update from link ------------------------ */

    /// Update the registry from a left→right linking result between two snapshots.
    ///
    /// Overview
    /// --------
    /// For each match `(from, to)` in `link.matches`:
    /// 1. Ensure both seeds have trajectory ids,
    /// 2. Merge the two trajectories (DSU, min representative),
    /// 3. Assign **all member detections** of both seeds to the kept trajectory.
    ///
    /// Arguments
    /// ---------
    /// `left`, `right` : &NightSnapshot
    ///     Consecutive nightly snapshots (left then right).
    /// `link` : &LinkResult
    ///     Solution containing `(from, to)` assignments over seed indices.
    /// `left_store`, `right_store` : &AlertStore
    ///     Per-night alert stores for membership resolution and optional checks.
    ///
    /// Panics
    /// ------
    /// In debug builds, asserts that `left.night_id == link.night_left` and
    /// `right.night_id == link.night_right`, and that the seed indices match.
    pub fn update_from_link(
        &mut self,
        left: &NightSnapshot,
        right: &NightSnapshot,
        link: &LinkResult,
        left_store: &AlertStore,
        right_store: &AlertStore,
    ) {
        println!(
            "Updating from link: {} seeds left, {} seeds right, {} matches",
            left.seeds.len(),
            right.seeds.len(),
            link.matches.len()
        );

        debug_assert_eq!(left.night_id, link.night_left);
        debug_assert_eq!(right.night_id, link.night_right);

        // Capacity hint to reduce rehashing on large updates.
        self.reserve_for_link(left, right, link);

        for Assignment { from, to, .. } in &link.matches {
            let lseed = &left.seeds[*from as usize];
            let rseed = &right.seeds[*to as usize];
            debug_assert_eq!(lseed.seed_id as usize, *from as usize);
            debug_assert_eq!(rseed.seed_id as usize, *to as usize);

            let lt = self.ensure_seed_traj(SeedKey {
                night_id: left.night_id,
                seed_id: lseed.seed_id,
            });
            let rt = self.ensure_seed_traj(SeedKey {
                night_id: right.night_id,
                seed_id: rseed.seed_id,
            });
            let keep = self.merge_traj(lt, rt);

            self.assign_seed_members(left.night_id, lseed, keep, left_store);
            self.assign_seed_members(right.night_id, rseed, keep, right_store);
        }

        println!(
            "  → now {} seeds, {} detections, {} trajectories",
            self.seed_to_traj.len(),
            self.detect_to_traj.len(),
            self.parent.len()
        );
    }

    /// Seed (initialize) all seeds of a snapshot into distinct trajectories.
    ///
    /// Overview
    /// --------
    /// For nights without prior links (e.g., the first night), assign a fresh
    /// trajectory to each seed and assign all their member detections.
    ///
    /// Arguments
    /// ---------
    /// `snap` : &NightSnapshot
    ///     The nightly snapshot to seed.
    /// `store` : &AlertStore
    ///     The associated alert store for membership resolution.
    pub fn seed_all_of_snapshot(&mut self, snap: &NightSnapshot, store: &AlertStore) {
        for s in &snap.seeds {
            let tid = self.ensure_seed_traj(SeedKey {
                night_id: snap.night_id,
                seed_id: s.seed_id,
            });
            self.assign_seed_members(snap.night_id, s, tid, store);
        }
    }

    /* ------------------------ Reads (canonicalized) ------------------------ */

    /// Iterate over `(DetectKey, TrajectoryId)` rows with canonical ids, deterministically ordered.
    ///
    /// Overview
    /// --------
    /// - Keys are sorted by `(night_id, alert_id)`.
    /// - For each key, trajectory ids are canonicalized, sorted, and deduplicated.
    /// - One row per `(DetectKey, TrajectoryId)` pair is yielded.
    ///
    /// Returns
    /// -------
    /// `impl Iterator<Item = (DetectKey, TrajectoryId)>`
    ///     A stable iterator suitable for export.
    pub fn iter_detection_assignments(&self) -> impl Iterator<Item = (DetectKey, TrajectoryId)> {
        // Sort keys for deterministic traversal.
        let mut keys: Vec<_> = self.detect_to_traj.keys().copied().collect();
        keys.sort_unstable(); // DetectKey has Ord (night_id, alert_id)

        // Upper bound hint; real number of rows may be larger (combinatorial).
        let mut rows: Vec<(DetectKey, TrajectoryId)> = Vec::with_capacity(keys.len());

        for dk in keys {
            if let Some(set) = self.detect_to_traj.get(&dk) {
                if set.is_empty() {
                    continue;
                }
                // Canonicalize, sort, dedup for deterministic output.
                let mut tids: Vec<TrajectoryId> =
                    set.iter().map(|&t| self.dsu_find_ro(t)).collect();
                tids.sort_unstable();
                tids.dedup();
                for tid in tids {
                    rows.push((dk, tid));
                }
            }
        }
        rows.into_iter()
    }

    /// Return all canonical trajectory ids for a detection, sorted and deduped.
    ///
    /// Arguments
    /// ---------
    /// `k` : DetectKey
    ///     Target detection key.
    ///
    /// Returns
    /// -------
    /// `impl Iterator<Item = TrajectoryId>`
    ///     Sorted, deduplicated canonical ids (may be empty).
    pub fn get_trajs_for_detection(&self, k: DetectKey) -> impl Iterator<Item = TrajectoryId> {
        let mut out = Vec::new();
        if let Some(s) = self.detect_to_traj.get(&k) {
            let mut tids: Vec<_> = s.iter().map(|&t| self.dsu_find_ro(t)).collect();
            tids.sort_unstable();
            tids.dedup();
            out = tids;
        }
        out.into_iter()
    }

    /// Primary trajectory (smallest canonical id) for a detection, if any.
    ///
    /// Arguments
    /// ---------
    /// `k` : DetectKey
    ///     Detection key.
    ///
    /// Returns
    /// -------
    /// `Option<TrajectoryId>`
    ///     `Some(min_id)` if assigned, else `None`.
    pub fn get_primary_traj_for_detection(&self, k: DetectKey) -> Option<TrajectoryId> {
        self.detect_to_traj.get(&k).and_then(|s| {
            s.iter().map(|&t| self.dsu_find_ro(t)).min() // smallest canonical id
        })
    }

    /// Trajectory id for a seed (canonicalized), if any.
    ///
    /// Arguments
    /// ---------
    /// `k` : SeedKey
    ///     Seed global key.
    ///
    /// Returns
    /// -------
    /// `Option<TrajectoryId>`
    ///     Canonical id if present, else `None`.
    #[inline]
    pub fn get_traj_for_seed(&self, k: SeedKey) -> Option<TrajectoryId> {
        self.seed_to_traj
            .get(&k)
            .copied()
            .map(|t| self.dsu_find_ro(t))
    }

    /// Number of distinct canonical trajectories touching a detection.
    ///
    /// Arguments
    /// ---------
    /// `k` : DetectKey
    ///     Detection key.
    ///
    /// Returns
    /// -------
    /// `usize`
    ///     Cardinality of the set of canonical ids.
    pub fn multiplicity(&self, k: DetectKey) -> usize {
        self.detect_to_traj
            .get(&k)
            .map(|s| {
                let mut tids: Vec<_> = s.iter().map(|&t| self.dsu_find_ro(t)).collect();
                tids.sort_unstable();
                tids.dedup();
                tids.len()
            })
            .unwrap_or(0)
    }

    /* ------------------------ Export ------------------------ */

    /// Build export columns from the registered stores.
    ///
    /// Overview
    /// --------
    /// Produces column buffers suitable for constructing a `pandas.DataFrame`
    /// on the Python side. One row is emitted for each `(DetectKey, TrajectoryId)`
    /// pair yielded by [`TrackRegistry::iter_detection_assignments`].
    ///
    /// Arguments
    /// ---------
    /// `stores_by_night` : Resolver from `night_id` to the corresponding `AlertStore`.
    ///
    /// Returns
    /// -------
    /// Ok(out) : Typed column buffers (on success).
    ///
    /// Errors
    /// ------
    /// - `FinkFatError::TrackExportError(night_id)` if a `night_id` is missing
    ///   in `stores_by_night`.
    ///
    /// Notes
    /// -----
    /// - RA/Dec are converted to degrees (`to_degrees()`).
    /// - JD is computed as `mjd_tt + 2_400_000.5` (TT days).
    pub fn collect_detection_columns(
        &self,
        stores_by_night: &BTreeMap<NightId, Arc<AlertStore>>,
    ) -> Result<ExportColumns, FinkFatError> {
        let mut out = ExportColumns::default();
        for (dk, tid) in self.iter_detection_assignments() {
            let store = stores_by_night
                .get(&dk.night_id)
                .ok_or(FinkFatError::TrackExportError(dk.night_id))?;
            let a = &store.alerts[dk.alert_id.idx()];

            out.candid.push(a.dia_source_id);
            out.ra.push(a.ra.to_degrees());
            out.dec.push(a.dec.to_degrees());
            out.jd.push(a.mjd_tt + 2_400_000.5);
            out.mjd_tt.push(a.mjd_tt);
            out.trajectory_id.push(tid.raw());
        }
        Ok(out)
    }

    /// Return a Python `dict` of lists ready for `pandas.DataFrame(...)`.
    ///
    /// Overview
    /// --------
    /// Convenience wrapper around [`TrackRegistry::collect_detection_columns`] for the
    /// Python bindings. Keys are the column names:
    /// `["candid", "ra", "dec", "jd", "mjd_tt", "trajectory_id"]`.
    ///
    /// Arguments
    /// ---------
    /// `py` : GIL-bound Python token.
    /// `stores_by_night` : Resolver from `night_id` to per-night stores.
    ///
    /// Returns
    /// -------
    /// d: A Python dictionary with list-valued columns.
    ///
    /// Errors
    /// ------
    /// - Raises `PyRuntimeError` if export fails internally.
    pub fn export_detection_dict_py(
        &self,
        py: pyo3::Python<'_>,
        stores_by_night: &BTreeMap<NightId, Arc<AlertStore>>,
    ) -> pyo3::PyResult<Py<PyDict>> {
        let cols = self
            .collect_detection_columns(stores_by_night)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let d = PyDict::new(py);
        d.set_item("candid", cols.candid)?;
        d.set_item("ra", cols.ra)?;
        d.set_item("dec", cols.dec)?;
        d.set_item("jd", cols.jd)?;
        d.set_item("mjd_tt", cols.mjd_tt)?;
        d.set_item("trajectory_id", cols.trajectory_id)?;

        Ok(d.into())
    }
}

/// Export columns ready for `pandas.DataFrame(cols_dict)`.
///
/// Overview
/// --------
/// Column buffers aligned row-wise with the flattened `(DetectKey, TrajectoryId)`
/// iterator provided by [`TrackRegistry::iter_detection_assignments`].
///
/// Ordering
/// --------
/// Deterministic ordering is guaranteed by sorting keys and trajectory ids at
/// iteration time in the registry.
///
/// Fields
/// ------
/// - `candid` — LSST `diaSourceId` (u64).
/// - `ra`, `dec` — ICRS coordinates in **degrees** (f64).
/// - `jd` — Julian date (**TT days**).
/// - `mjd_tt` — Modified Julian date (**TT days**).
/// - `trajectory_id` — Canonical trajectory id (u64).
#[derive(Default, Debug)]
pub struct ExportColumns {
    pub candid: Vec<u64>,
    pub ra: Vec<f64>,     // deg
    pub dec: Vec<f64>,    // deg
    pub jd: Vec<f64>,     // TT days
    pub mjd_tt: Vec<f64>, // TT days
    pub trajectory_id: Vec<u64>,
}

#[cfg(test)]
mod track_registry_test {
    use crate::alerts::Alert;

    use super::*;

    use std::collections::BTreeMap;
    use std::sync::Arc;

    use proptest::prelude::*;

    /// Minimal helper to create a SeedNode with given `seed_id`, `night_id` and `members`.
    ///
    /// Only the fields needed by TrackRegistry::assign_seed_members are relevant here.
    /// The remaining fields receive plausible dummy values.
    fn make_seed(seed_id: u64, night_id: NightId, members: Vec<AlertId>) -> SeedNode {
        SeedNode {
            seed_id,
            night_id,
            epoch_mid: 60_000.0,
            pos_xy: [0.0, 0.0],
            vel_xy: [1e-4, -1e-4],
            cov_pos: [[(1e-6_f64).powi(2), 0.0], [0.0, (1e-6_f64).powi(2)]],
            cov_vel: [[(1e-6_f64).powi(2), 0.0], [0.0, (1e-6_f64).powi(2)]],
            acc_xy: None,
            flux_mean: 1000.0,
            flux_std: 50.0,
            band: 2,
            n_obs: members.len() as u16,
            members,
            center_ra: 1.0,
            center_dec: 0.1,
            ra_mid: 1.0,
            dec_mid: 0.1,
        }
    }

    /// Minimal helper to create an AlertStore containing at least the alert at `aid.idx()`.
    fn make_store_with_alert(aid: AlertId) -> AlertStore {
        // We only need the alert at position `aid.idx()`. Build a vector of that length and place one alert.
        let idx = aid.idx();
        let mut alerts = Vec::with_capacity(idx + 1);
        // Fill with a dummy alert if idx > 0
        for i in 0..idx {
            alerts.push(Alert {
                id: AlertId(i as u32),
                dia_source_id: 10_000 + i as u64,
                ra: 1.0,
                ra_err: 0.0,
                dec: 0.1,
                dec_err: 0.0,
                mjd_tt: 60_000.0,
                flux: 0.0,
                flux_err: 0.0,
                band: 1,
            });
        }
        // Real one at `idx`
        alerts.push(Alert {
            id: aid,
            dia_source_id: 42_4242,
            ra: 1.234, // rad
            ra_err: 0.0,
            dec: 0.567, // rad
            dec_err: 0.0,
            mjd_tt: 60_001.5,
            flux: 100.0,
            flux_err: 3.0,
            band: 2,
        });

        AlertStore {
            start_mjd: 60_000.0,
            alerts,
        }
    }

    /* ------------------------- Unit tests ------------------------- */

    #[test]
    fn keepfirst_policy_keeps_first_assignment() {
        let mut reg = TrackRegistry::new(DetectConflictPolicy::KeepFirst);

        let night: NightId = 1234;
        let dk = DetectKey {
            night_id: night,
            alert_id: AlertId(0),
        };

        // Two distinct trajectories created from two seeds
        let t1 = reg.ensure_seed_traj(SeedKey {
            night_id: night,
            seed_id: 0,
        });
        let t2 = reg.ensure_seed_traj(SeedKey {
            night_id: night,
            seed_id: 1,
        });

        // Single-member seed; assign twice with different traj ids.
        let seed = make_seed(0, night, vec![AlertId(0)]);
        let dummy_store = AlertStore {
            start_mjd: 0.0,
            alerts: vec![],
        };

        reg.assign_seed_members(night, &seed, t1, &dummy_store);
        reg.assign_seed_members(night, &seed, t2, &dummy_store);

        // Under KeepFirst, the assignment should remain the first one (t1).
        let v: Vec<_> = reg.get_trajs_for_detection(dk).collect();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0], t1);

        assert_eq!(reg.multiplicity(dk), 1);
        assert_eq!(reg.get_primary_traj_for_detection(dk), Some(t1));
    }

    #[test]
    fn overwrite_policy_overwrites_last_assignment() {
        let mut reg = TrackRegistry::new(DetectConflictPolicy::Overwrite);

        let night: NightId = 7;
        let dk = DetectKey {
            night_id: night,
            alert_id: AlertId(0),
        };

        let t1 = reg.ensure_seed_traj(SeedKey {
            night_id: night,
            seed_id: 11,
        });
        let t2 = reg.ensure_seed_traj(SeedKey {
            night_id: night,
            seed_id: 22,
        });

        let seed = make_seed(0, night, vec![AlertId(0)]);
        let dummy_store = AlertStore {
            start_mjd: 0.0,
            alerts: vec![],
        };

        reg.assign_seed_members(night, &seed, t1, &dummy_store);
        reg.assign_seed_members(night, &seed, t2, &dummy_store);

        // Under Overwrite, the last wins.
        let v: Vec<_> = reg.get_trajs_for_detection(dk).collect();
        assert_eq!(v, vec![t2]);
        assert_eq!(reg.multiplicity(dk), 1);
        assert_eq!(reg.get_primary_traj_for_detection(dk), Some(t2));
    }

    #[test]
    fn combinatorial_policy_accumulates_all_assignments() {
        let mut reg = TrackRegistry::new(DetectConflictPolicy::Combinatorial);

        let night: NightId = 999;
        let dk = DetectKey {
            night_id: night,
            alert_id: AlertId(0),
        };

        let t1 = reg.ensure_seed_traj(SeedKey {
            night_id: night,
            seed_id: 1,
        });
        let t2 = reg.ensure_seed_traj(SeedKey {
            night_id: night,
            seed_id: 2,
        });

        let seed = make_seed(0, night, vec![AlertId(0)]);
        let dummy_store = AlertStore {
            start_mjd: 0.0,
            alerts: vec![],
        };

        reg.assign_seed_members(night, &seed, t1, &dummy_store);
        reg.assign_seed_members(night, &seed, t2, &dummy_store);

        // Under Combinatorial, both ids are present (sorted).
        let mut v: Vec<_> = reg.get_trajs_for_detection(dk).collect();
        v.sort_unstable();
        assert_eq!(v, vec![t1, t2]);
        assert_eq!(reg.multiplicity(dk), 2);
    }

    #[test]
    fn merge_traj_canonicalizes_ids_in_reads() {
        let mut reg = TrackRegistry::new(DetectConflictPolicy::Combinatorial);

        let night: NightId = 42;
        let dk = DetectKey {
            night_id: night,
            alert_id: AlertId(0),
        };

        let t1 = reg.ensure_seed_traj(SeedKey {
            night_id: night,
            seed_id: 1,
        });
        let t2 = reg.ensure_seed_traj(SeedKey {
            night_id: night,
            seed_id: 2,
        });

        let seed = make_seed(0, night, vec![AlertId(0)]);
        let dummy_store = AlertStore {
            start_mjd: 0.0,
            alerts: vec![],
        };

        // Assign detection to both trajectories (combinatorial).
        reg.assign_seed_members(night, &seed, t1, &dummy_store);
        reg.assign_seed_members(night, &seed, t2, &dummy_store);

        // Merge the two ids — representative is the minimum id.
        let keep = reg.merge_traj(t1, t2);

        // Now the multiplicity should be 1 and the only id is `keep`.
        let v: Vec<_> = reg.get_trajs_for_detection(dk).collect();
        assert_eq!(v, vec![keep]);
        assert_eq!(reg.multiplicity(dk), 1);
        assert_eq!(reg.get_primary_traj_for_detection(dk), Some(keep));
    }

    #[test]
    fn iter_detection_assignments_is_deterministic() {
        let mut reg = TrackRegistry::new(DetectConflictPolicy::Combinatorial);

        // Two nights, interleaved keys.
        let n1: NightId = 1;
        let n2: NightId = 2;

        // Prepare trajectories
        let t1 = reg.ensure_seed_traj(SeedKey {
            night_id: n1,
            seed_id: 10,
        });
        let t2 = reg.ensure_seed_traj(SeedKey {
            night_id: n1,
            seed_id: 11,
        });
        let t3 = reg.ensure_seed_traj(SeedKey {
            night_id: n2,
            seed_id: 12,
        });

        // Seed with different members
        let s_a = make_seed(0, n1, vec![AlertId(3), AlertId(1)]);
        let s_b = make_seed(0, n2, vec![AlertId(2)]);

        let dummy_store = AlertStore {
            start_mjd: 0.0,
            alerts: vec![],
        };

        // Assign:
        // (n1,1)->t1, (n1,3)->t2, (n2,2)->t3
        reg.assign_seed_members(n1, &s_a, t1, &dummy_store); // adds (n1,1) and (n1,3) to t1
        reg.assign_seed_members(n2, &s_b, t3, &dummy_store); // adds (n2,2) to t3
                                                             // also (n1,3) to t2 to get a combinatorial multi-id on that key
        reg.assign_seed_members(n1, &make_seed(1, n1, vec![AlertId(3)]), t2, &dummy_store);

        // Collect rows; they should be sorted by (night_id, alert_id) then by traj id
        let rows: Vec<_> = reg.iter_detection_assignments().collect();

        // Expected sorted order:
        // (n1,1)->{t1}
        // (n1,3)->{min(t1,t2), max(t1,t2)}
        // (n2,2)->{t3}
        let mut expect: Vec<(DetectKey, TrajectoryId)> = Vec::new();
        let mut ids_13 = [t1, t2];
        ids_13.sort_unstable();
        expect.push((
            DetectKey {
                night_id: n1,
                alert_id: AlertId(1),
            },
            ids_13[0], /* actually t1 */
        ));
        // note: (n1,1) only had t1
        // for (n1,3) two rows with both ids, sorted
        expect.push((
            DetectKey {
                night_id: n1,
                alert_id: AlertId(3),
            },
            ids_13[0],
        ));
        expect.push((
            DetectKey {
                night_id: n1,
                alert_id: AlertId(3),
            },
            ids_13[1],
        ));
        expect.push((
            DetectKey {
                night_id: n2,
                alert_id: AlertId(2),
            },
            t3,
        ));

        assert_eq!(rows, expect);
    }

    #[test]
    fn collect_detection_columns_exports_expected_values() {
        let mut reg = TrackRegistry::new(DetectConflictPolicy::Overwrite);

        let night: NightId = 77;
        let aid = AlertId(0);

        let t = reg.ensure_seed_traj(SeedKey {
            night_id: night,
            seed_id: 0,
        });
        let seed = make_seed(0, night, vec![aid]);

        // Alert store with the needed alert
        let store = Arc::new(make_store_with_alert(aid));
        let mut stores = BTreeMap::new();
        stores.insert(night, store.clone());

        // Assign and export
        reg.assign_seed_members(night, &seed, t, &store);

        let cols = reg.collect_detection_columns(&stores).expect("export ok");
        assert_eq!(cols.candid.len(), 1);
        assert_eq!(cols.trajectory_id.len(), 1);
        assert_eq!(cols.trajectory_id[0], t.raw());

        // Verify RA/DEC conversion is applied (degrees), and JD = MJD(TT) + 2_400_000.5
        // Using values set in make_store_with_alert(aid):
        let ra_deg = 1.234f64.to_degrees();
        let dec_deg = 0.567f64.to_degrees();
        assert!((cols.ra[0] - ra_deg).abs() < 1e-12);
        assert!((cols.dec[0] - dec_deg).abs() < 1e-12);
        assert!((cols.jd[0] - (60_001.5 + 2_400_000.5)).abs() < 1e-9);
    }

    #[test]
    fn empty_members_is_noop() {
        let mut reg = TrackRegistry::new(DetectConflictPolicy::KeepFirst);
        let night: NightId = 1;

        let t = reg.ensure_seed_traj(SeedKey {
            night_id: night,
            seed_id: 0,
        });
        let seed = make_seed(0, night, vec![]);

        let store = AlertStore {
            start_mjd: 0.0,
            alerts: vec![],
        };
        reg.assign_seed_members(night, &seed, t, &store);

        // No assignment took place
        let dk = DetectKey {
            night_id: night,
            alert_id: AlertId(0),
        };
        assert_eq!(reg.multiplicity(dk), 0);
    }

    /* ------------------------- Proptests ------------------------- */

    proptest! {
        /// For KeepFirst: the final id for a given detection is the **first** assigned.
        #[test]
        fn prop_keepfirst_first_wins(
            // number of assignments, pool size for trajectory ids
            (n, k) in (1usize..20, 1usize..5),
            // sequence of indices into that pool
            seq in prop::collection::vec(0usize..5, 1..20)
        ) {
            let mut reg = TrackRegistry::new(DetectConflictPolicy::KeepFirst);
            let night: NightId = 10;
            let dk = DetectKey { night_id: night, alert_id: AlertId(0) };

            // Prepare K trajectories
            let tids: Vec<_> = (0..k)
                .map(|i| reg.ensure_seed_traj(SeedKey { night_id: night, seed_id: i as u64 }))
                .collect();

            let store = AlertStore { start_mjd: 0.0, alerts: vec![] };
            let seed = make_seed(0, night, vec![AlertId(0)]);

            // Apply the sequence (truncate to n)
            let mut first: Option<TrajectoryId> = None;
            for &ix in seq.iter().take(n) {
                let tid = tids[ix % k];
                if first.is_none() { first = Some(tid); }
                reg.assign_seed_members(night, &seed, tid, &store);
            }

            let v: Vec<_> = reg.get_trajs_for_detection(dk).collect();
            prop_assert_eq!(v.len(), 1);
            prop_assert_eq!(v[0], first.unwrap());
        }

        /// For Overwrite: the final id for a given detection is the **last** assigned.
        #[test]
        fn prop_overwrite_last_wins(
            (n, k) in (1usize..20, 1usize..5),
            seq in prop::collection::vec(0usize..5, 1..20)
        ) {
            let mut reg = TrackRegistry::new(DetectConflictPolicy::Overwrite);
            let night: NightId = 11;
            let dk = DetectKey { night_id: night, alert_id: AlertId(0) };

            let tids: Vec<_> = (0..k)
                .map(|i| reg.ensure_seed_traj(SeedKey { night_id: night, seed_id: i as u64 }))
                .collect();

            let store = AlertStore { start_mjd: 0.0, alerts: vec![] };
            let seed = make_seed(0, night, vec![AlertId(0)]);

            let mut last: Option<TrajectoryId> = None;
            for &ix in seq.iter().take(n) {
                let tid = tids[ix % k];
                last = Some(tid);
                reg.assign_seed_members(night, &seed, tid, &store);
            }

            let v: Vec<_> = reg.get_trajs_for_detection(dk).collect();
            prop_assert_eq!(v, vec![last.unwrap()]);
        }

        /// For Combinatorial: multiplicity equals the number of distinct (canonical) ids assigned.
        #[test]
        fn prop_combinatorial_counts_distinct_ids(
            (n, k) in (1usize..20, 1usize..5),
            seq in prop::collection::vec(0usize..5, 1..20)
        ) {
            let mut reg = TrackRegistry::new(DetectConflictPolicy::Combinatorial);
            let night: NightId = 12;
            let dk = DetectKey { night_id: night, alert_id: AlertId(0) };

            let tids: Vec<_> = (0..k)
                .map(|i| reg.ensure_seed_traj(SeedKey { night_id: night, seed_id: i as u64 }))
                .collect();

            let store = AlertStore { start_mjd: 0.0, alerts: vec![] };
            let seed = make_seed(0, night, vec![AlertId(0)]);

            // Track distinct ids assigned
            let mut used = std::collections::BTreeSet::new();

            for &ix in seq.iter().take(n) {
                let tid = tids[ix % k];
                used.insert(tid);
                reg.assign_seed_members(night, &seed, tid, &store);
            }

            let m = reg.multiplicity(dk);
            prop_assert_eq!(m, used.len());
        }

        /// Merging two trajectories should canonicalize to the min id under Combinatorial, reducing multiplicity.
        #[test]
        fn prop_merge_reduces_mult_after_union(
            i in 0usize..10, j in 0usize..10 // two different seeds → tids
        ) {
            prop_assume!(i != j);

            let mut reg = TrackRegistry::new(DetectConflictPolicy::Combinatorial);
            let night: NightId = 13;
            let dk = DetectKey { night_id: night, alert_id: AlertId(0) };

            let t1 = reg.ensure_seed_traj(SeedKey { night_id: night, seed_id: i as u64 });
            let t2 = reg.ensure_seed_traj(SeedKey { night_id: night, seed_id: j as u64 });

            let store = AlertStore { start_mjd: 0.0, alerts: vec![] };
            let seed = make_seed(0, night, vec![AlertId(0)]);

            // Assign both
            reg.assign_seed_members(night, &seed, t1, &store);
            reg.assign_seed_members(night, &seed, t2, &store);
            prop_assert_eq!(reg.multiplicity(dk), 2);

            // Merge and check multiplicity == 1 and id == min(t1, t2)
            let keep = reg.merge_traj(t1, t2);

            let v: Vec<_> = reg.get_trajs_for_detection(dk).collect();
            prop_assert_eq!(v, vec![keep]);
            prop_assert_eq!(reg.multiplicity(dk), 1);
        }
    }
}
