//! Seed storage (`SeedStore`) grouped by `NightId`, with borrowed and owned forms.
//!
//! Overview
//! --------
//! `SeedStore<'alert_lf>` is the in-memory representation of **intra-night seeds**
//! (typically pairs/triplets of alerts turned into [`SeedNode`]) indexed by [`NightId`].
//! Each night maps to a contiguous `Vec<SeedNode<'alert_lf>>`, allowing:
//!
//! - fast sequential scans per-night,
//! - stable per-night indexing (`SeedKey { night_id, idx_in_night }`),
//! - cheap borrowing of alert-backed nodes during graph construction and solving.
//!
//! The store is keyed by [`NightId`] and does **not** require that nights are
//! contiguous or complete: missing nights simply do not appear in the map.
//!
//! Borrowed vs owned
//! -----------------
//! `SeedStore<'alert_lf>` holds [`SeedNode<'alert_lf>`] values that typically borrow
//! alert data (`'alert_lf`) from an upstream alert store or batch.
//!
//! For persistence, the store can be converted into an owned serializable form
//! [`SeedStoreOwned`] using [`SeedStore::to_owned`]. The owned store contains
//! [`SeedNodeOwned`] values that no longer borrow alert slices.
//!
//! Determinism
//! -----------
//! Hash map iteration order is not deterministic. Any operation that needs stable
//! ordering (for reproducible output, deterministic truncation, stable journaling,
//! etc.) must sort keys explicitly. This module exposes helpers such as
//! [`SeedStore::night_ids_sorted`] and the deterministic night-pair enumeration
//! helpers (`night_pairs_in_window_*`) that sort night ids internally.
//!
//! Inter-night scheduling helper
//! -----------------------------
//! Inter-night stages (edge construction, linking, scoring) often need to iterate
//! over **night pairs** constrained by:
//!
//! - a processing window [`NightWindow`] (inclusive bounds),
//! - a maximum temporal gap `max_gap_nights`.
//!
//! This module provides two APIs:
//!
//! - [`SeedStore::night_pairs_in_window_iter`]: lazy iterator yielding `(left, right)` pairs,
//! - [`SeedStore::night_pairs_in_window`]: eager `Vec` builder implemented as `collect()`
//!   over the iterator.
//!
//! The iterator uses a **two-pointer sliding window** over sorted nights to achieve
//! `O(N + P)` complexity, where `N` is the number of nights present in the store
//! inside the window, and `P` is the number of pairs emitted.
//!
//! Implementation notes for the iterator
//! -------------------------------------
//! Returning `impl Iterator` from a function that may yield an empty iterator
//! introduces a common Rust constraint: all branches must return the **same**
//! concrete iterator type.
//!
//! `night_pairs_in_window_iter` solves this without boxing by:
//! - creating an `Option<(Arc<[NightId]>, usize)>` state,
//! - converting it to an iterator via `Option::into_iter()` (0 or 1 element),
//! - `flat_map`-ing into the actual pair generator.
//!
//! `Arc<[NightId]>` is used instead of `Vec<NightId>` because the iterator returns
//! nested closures (`flat_map` + `map`) that must capture the night list multiple
//! times. `Arc` cloning is cheap (pointer + refcount) and avoids moving the original
//! vector into a closure that would be consumed multiple times.
//!
//! Typical usage
//! -------------
//! ```rust,ignore
//! // Enumerate all eligible night pairs in a window (deterministic order):
//! for (left, right) in seed_store.night_pairs_in_window_iter(window, max_gap) {
//!     // build inter-night edges between `left` and `right`
//! }
//!
//! // Or materialize them:
//! let pairs = seed_store.night_pairs_in_window(window, max_gap);
//! ```
//!
//! Notes
//! -----
//! - Night IDs are treated as logical integers; the gap constraint is enforced as
//!   a difference on the underlying `u32` values of `NightId`.
//! - `max_gap_nights == 0` yields an empty iterator and an empty vector.

use ahash::AHashMap;

use crate::{
    night_id::{NightId, PairingMode},
    persistence::{
        seed_node::{SeedKey, SeedNodeOwned},
        seed_store::SeedStoreOwned,
    },
    seeding::seed_node::SeedNode,
    solver::components::seed_index::SeedGlobalIndex,
};

/// In-memory store of seed nodes grouped by observation night.
///
/// Data model
/// ----------
/// The store is a map:
///
/// - key: [`NightId`]
/// - value: `Vec<SeedNode<'alert_lf>>`
///
/// Each per-night vector is indexed by `idx_in_night`, which is used as part of
/// [`SeedKey`]. This enables stable addressing of seeds within a given night
/// without requiring global contiguous indexing.
///
/// Lifetimes
/// ---------
/// `'alert_lf` is the lifetime of borrowed alert references inside [`SeedNode`].
/// The store itself does not own alerts; it only owns the seed node structures.
///
/// Persistence
/// -----------
/// Use [`SeedStore::to_owned`] to convert to [`SeedStoreOwned`] and persist the
/// result through the persistence layer.
///
/// Determinism
/// -----------
/// `AHashMap` iteration order is not deterministic. Functions in this impl that
/// need deterministic ordering explicitly sort night IDs before producing output.
pub struct SeedStore<'alert_lf>(AHashMap<NightId, Vec<SeedNode<'alert_lf>>>);

impl<'alert_lf> SeedStore<'alert_lf> {
    /// Create a new empty `SeedStore`.
    #[inline]
    pub fn new() -> Self {
        Self(AHashMap::new())
    }

    /// Create a `SeedStore` from a pre-constructed map of night IDs to seed nodes.
    ///
    /// Notes
    /// -----
    /// - The map is stored as-is; no sorting or validation is performed.
    /// - Deterministic operations should use methods that sort keys explicitly.
    #[inline]
    pub fn from_map(map: AHashMap<NightId, Vec<SeedNode<'alert_lf>>>) -> Self {
        Self(map)
    }

    /// Convert this `SeedStore` into an owned version that can be serialized.
    ///
    /// Behavior
    /// --------
    /// - For each `(night_id, seeds)` entry, converts all borrowed [`SeedNode`]
    ///   into [`SeedNodeOwned`] via `SeedNode::to_owned()`.
    /// - Produces a [`SeedStoreOwned`] containing a map of `NightId -> Vec<SeedNodeOwned>`.
    ///
    /// Notes
    /// -----
    /// - This operation allocates new vectors and copies owned fields required for
    ///   persistence. It is typically used at pipeline boundaries where data must
    ///   survive beyond the current borrow scope.
    pub fn to_owned(&self) -> SeedStoreOwned {
        let mut map = AHashMap::with_capacity(self.0.len());

        for (night_id, seeds) in self.0.iter() {
            let owned_seeds: Vec<SeedNodeOwned> = seeds.iter().map(|s| s.to_owned()).collect();
            map.insert(*night_id, owned_seeds);
        }

        SeedStoreOwned(map)
    }

    /// Insert a vector of seed nodes for a given night.
    ///
    /// Semantics
    /// ---------
    /// - Replaces any existing vector for `night_id`.
    #[inline]
    pub fn insert(&mut self, night_id: NightId, seeds: Vec<SeedNode<'alert_lf>>) {
        self.0.insert(night_id, seeds);
    }

    /// Get the vector of seed nodes for a given night, if it exists.
    #[inline]
    pub fn get(&self, night_id: &NightId) -> Option<&Vec<SeedNode<'alert_lf>>> {
        self.0.get(night_id)
    }

    /// Get a mutable reference to the vector of seed nodes for a given night, if it exists.
    #[inline]
    pub fn get_mut(&mut self, night_id: &NightId) -> Option<&mut Vec<SeedNode<'alert_lf>>> {
        self.0.get_mut(night_id)
    }

    /// Get a specific seed node by its key (night ID + index in night).
    ///
    /// Semantics
    /// ---------
    /// - Returns `None` if:
    ///   - the night is not present,
    ///   - or the index is out of bounds for that night's vector.
    #[inline]
    pub fn get_by_key(&self, key: SeedKey) -> Option<&SeedNode<'alert_lf>> {
        let vec = self.0.get(&key.night_id)?;
        vec.get(key.idx_in_night as usize)
    }

    /// Return the number of seeds stored for a given night, if present.
    #[inline]
    pub fn len_for_night(&self, night_id: &NightId) -> Option<usize> {
        self.0.get(night_id).map(|v| v.len())
    }

    /// Return all night IDs present in the store, sorted increasingly.
    ///
    /// Determinism
    /// -----------
    /// This function provides a stable ordering across runs.
    pub fn night_ids_sorted(&self) -> Vec<NightId> {
        let mut night_ids: Vec<NightId> = self.0.keys().copied().collect();
        night_ids.sort();
        night_ids
    }

    /// Build a global seed index over the entire store.
    ///
    /// The global index provides a stable mapping between a "global seed id"
    /// and per-night addresses (`SeedKey`-like addressing), used by some solver
    /// components.
    #[inline]
    pub fn build_global_index(&self) -> SeedGlobalIndex {
        SeedGlobalIndex::build(self)
    }

    /// Iterate over `(night_id, seeds)` entries (hash-map iteration order).
    ///
    /// Notes
    /// -----
    /// - The iteration order is not deterministic. Prefer [`night_ids_sorted`]
    ///   if stable ordering is required.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&NightId, &Vec<SeedNode<'alert_lf>>)> {
        self.0.iter()
    }

    /// Whether the store contains `n`.
    #[inline]
    pub fn contains_night(&self, n: NightId) -> bool {
        self.0.contains_key(&n)
    }

    /// Iterate over night IDs present in the store (hash-map iteration order).
    ///
    /// Return
    /// ------
    /// An iterator yielding `&NightId` values corresponding to the keys in the internal map.
    ///
    /// Notes
    /// -----
    /// - The iteration order is not deterministic. Prefer [`nights_sorted`] if stable ordering is required.
    #[inline]
    pub fn nights(&self) -> impl Iterator<Item = &NightId> {
        self.0.keys()
    }

    /// Get a sorted list of night IDs currently present in the store.
    ///
    /// Return
    /// ------
    /// A `Vec<NightId>` containing all night IDs in the store, sorted in increasing order.
    ///
    /// Notes
    /// -----
    /// - The sorting is done at call time; the internal map does not maintain order.
    #[inline]
    pub fn nights_sorted(&self) -> Vec<NightId> {
        let mut night_ids: Vec<NightId> = self.nights().copied().collect();
        night_ids.sort();
        night_ids
    }

    /// Remove and return the seeds for night `n`, if present.
    #[inline]
    pub fn remove(&mut self, n: NightId) -> Option<Vec<SeedNode<'alert_lf>>> {
        self.0.remove(&n)
    }

    /// Enumerate eligible inter-night `(left_night, right_night)` pairs anchored to
    /// the **latest night** present in the store within a requested [`NightWindow`].
    ///
    /// This iterator is a building block for *sliding-window* inter-night stages,
    /// where the pipeline treats a "current" night as the right-hand side and
    /// connects it to a bounded set of earlier nights on the left-hand side.
    ///
    /// Overview
    /// --------
    /// The function proceeds in two conceptual steps:
    ///
    /// 1. **Select `right_night`**:
    ///    - Collect nights that are both:
    ///      - present in the store (`self`), and
    ///      - inside `night_window` (inclusive bounds),
    ///    - choose `right_night` as the maximum (latest) among them.
    ///
    /// 2. **Select eligible `left_night`**:
    ///    - Enforce that `left_night < right_night` (strictly earlier),
    ///    - enforce a maximum temporal gap:
    ///      `right_night.value() - left_night.value() <= max_gap_nights`,
    ///      implemented as the lower bound:
    ///      `left_night.value() >= right_night.value() - max_gap_nights`.
    ///
    /// The iterator emits one pair per eligible `left_night`:
    ///
    /// ```text
    /// (left_0, right), (left_1, right), ..., (left_k, right)
    /// ```
    ///
    /// Behavior
    /// --------
    /// This function has two modes depending on whether `night_window` is a single night.
    ///
    /// ### Multi-night windows (`night_window.is_single() == false`)
    ///
    /// - `right_night` is the latest night present in the store inside `night_window`.
    /// - `left_night` candidates are restricted to:
    ///   - nights present in the store,
    ///   - inside `night_window`,
    ///   - earlier than `right_night`,
    ///   - within `max_gap_nights` of `right_night`.
    ///
    /// This corresponds to the interpretation:
    /// **the window bounds both the anchor (`right`) and the candidates (`left`)**.
    ///
    /// ### Single-night windows (`night_window.is_single() == true`)
    ///
    /// Single-night runs are common in orchestration: the pipeline is asked to process
    /// “the current night” only, but still needs to link it with earlier nights.
    ///
    /// In this special case:
    /// - `right_night` is that single night *if it exists in the store*,
    /// - `left_night` candidates are searched in the store using only:
    ///   - `left_night < right_night`,
    ///   - the gap constraint (`right - left <= max_gap_nights`),
    ///   - **without** requiring `left_night` to be inside `night_window`.
    ///
    /// In other words: a single-night window anchors `right_night`, but does not
    /// constrain the search space for `left_night` beyond the gap rule.
    ///
    /// This matches the typical pipeline pattern:
    ///
    /// ```text
    /// current = night_window.single_night()
    /// for prev in store.nights_in_gap_before(current):
    ///     build_edges(prev, current)
    /// ```
    ///
    /// Semantics
    /// ---------
    /// - `night_window` bounds are inclusive (`start..=end`).
    /// - Only nights present in the store are considered.
    /// - Only pairs with `left_night < right_night` are emitted.
    /// - The gap constraint is applied on underlying `u32` values:
    ///   `right.value() - left.value() <= max_gap_nights`.
    /// - If `max_gap_nights == 0`, the iterator yields no pairs.
    /// - If there is no night present in the store within `night_window`,
    ///   the iterator yields no pairs (no `right_night` anchor).
    ///
    /// Ordering
    /// --------
    /// The output ordering is deterministic:
    /// - `right_night` is constant (the selected latest night),
    /// - `left_night` values are emitted in strictly increasing order.
    ///
    /// Determinism
    /// -----------
    /// - The selection of `right_night` is deterministic because it is derived from
    ///   a sorted list of nights present in the window.
    /// - The `left_night` set is collected and sorted before emission.
    ///
    /// Complexity
    /// ----------
    /// Let:
    /// - `Nw` be the number of nights present in the store within `night_window`,
    /// - `Ns` be the total number of nights present in the store,
    /// - `K` be the number of emitted pairs.
    ///
    /// The cost is:
    /// - selecting `right_night`: `O(Nw log Nw)` due to sorting within the window,
    /// - collecting `left_night` candidates:
    ///   - multi-night window: filters over `Ns` keys, then sorts `K` elements,
    ///   - single-night window: same, but without the window containment test,
    /// - sorting `left_night`: `O(K log K)`,
    /// - emitting: `O(K)`.
    ///
    /// In practice, `Ns` is typically small compared to alert counts, and this
    /// iterator is intended to be used at the night granularity.
    ///
    /// Memory
    /// ------
    /// - Allocates intermediate vectors:
    ///   - `nights_in_window` (nights present inside the window),
    ///   - `lefts` (eligible left nights).
    /// - No additional allocations during iteration once the state is built.
    ///
    /// Arguments
    /// ---------
    /// * `night_window` – Inclusive bounds used to select the anchor `right_night`.
    /// * `max_gap_nights` – Maximum allowed difference between `right_night` and
    ///   `left_night` in units of night IDs.
    ///
    /// Return
    /// ------
    /// Returns an iterator yielding `(left_night, right_night)` pairs.
    /// Each pair is suitable as input to inter-night edge construction
    /// (connect seeds from `left_night` to seeds from `right_night`).
    ///
    /// Notes
    /// -----
    /// - This helper does **not** enumerate all pairs within `night_window`.
    ///   It enumerates only pairs anchored to the selected latest night.
    /// - If the store contains sparse nights (missing IDs), the gap constraint is
    ///   evaluated on the numeric difference of IDs, not on “count of available nights”.
    #[inline]
    pub fn night_pairs_iter(
        &self,
        night_window: PairingMode,
    ) -> impl Iterator<Item = (NightId, NightId)> {
        let mut nights: Vec<NightId> = self.0.keys().copied().collect();
        nights.sort();
        night_window.night_pairs_iter(nights)
    }
}

#[cfg(test)]
mod night_pairs_iter_tests {
    use super::*;
    use proptest::prelude::*;

    fn nid(v: u32) -> NightId {
        NightId(v)
    }

    fn make_store(nights: &[u32]) -> SeedStore<'static> {
        let mut map: AHashMap<NightId, Vec<SeedNode<'static>>> = AHashMap::new();
        for &n in nights {
            // Empty seed vectors are sufficient for night-pair enumeration tests
            map.insert(NightId(n), Vec::new());
        }
        SeedStore::from_map(map)
    }

    // -------------------------------------------------------------------------
    // Single-night mode tests
    // -------------------------------------------------------------------------

    #[test]
    fn single_night_empty_store() {
        let store = make_store(&[]);
        let mode = PairingMode::single_night(nid(100), 10).unwrap();
        let pairs: Vec<_> = store.night_pairs_iter(mode).collect();
        assert_eq!(pairs, vec![]);
    }

    #[test]
    fn single_night_anchor_not_in_store() {
        let store = make_store(&[10, 20, 30]);
        let mode = PairingMode::single_night(nid(100), 10).unwrap();
        let pairs: Vec<_> = store.night_pairs_iter(mode).collect();
        assert_eq!(pairs, vec![]);
    }

    #[test]
    fn single_night_no_eligible_lefts() {
        let store = make_store(&[100]);
        let mode = PairingMode::single_night(nid(100), 10).unwrap();
        let pairs: Vec<_> = store.night_pairs_iter(mode).collect();
        assert_eq!(pairs, vec![]);
    }

    #[test]
    fn single_night_basic() {
        let store = make_store(&[85, 92, 100]);
        let mode = PairingMode::single_night(nid(100), 10).unwrap();
        let pairs: Vec<_> = store.night_pairs_iter(mode).collect();

        // 85 is outside gap (100 - 85 = 15 > 10)
        // 92 is within gap
        assert_eq!(pairs, vec![(nid(92), nid(100))]);
    }

    #[test]
    fn single_night_multiple_lefts() {
        let store = make_store(&[90, 92, 95, 100]);
        let mode = PairingMode::single_night(nid(100), 10).unwrap();
        let pairs: Vec<_> = store.night_pairs_iter(mode).collect();

        // All nights within gap
        assert_eq!(
            pairs,
            vec![
                (nid(90), nid(100)),
                (nid(92), nid(100)),
                (nid(95), nid(100)),
            ]
        );
    }

    #[test]
    fn single_night_saturating_sub() {
        let store = make_store(&[0, 1, 2]);
        let mode = PairingMode::single_night(nid(2), 250).unwrap();
        let pairs: Vec<_> = store.night_pairs_iter(mode).collect();

        // min_left = 2 - 250 saturates to 0
        // eligible: 0, 1
        assert_eq!(pairs, vec![(nid(0), nid(2)), (nid(1), nid(2))]);
    }

    // -------------------------------------------------------------------------
    // Multi-night batch mode tests
    // -------------------------------------------------------------------------

    #[test]
    fn batch_range_empty_store() {
        let store = make_store(&[]);
        let mode = PairingMode::batch_range(nid(50), nid(100)).unwrap();
        let pairs: Vec<_> = store.night_pairs_iter(mode).collect();
        assert_eq!(pairs, vec![]);
    }

    #[test]
    fn batch_range_no_nights_in_range() {
        let store = make_store(&[10, 20, 30]);
        let mode = PairingMode::batch_range(nid(50), nid(100)).unwrap();
        let pairs: Vec<_> = store.night_pairs_iter(mode).collect();
        assert_eq!(pairs, vec![]);
    }

    #[test]
    fn batch_range_only_right_in_range() {
        let store = make_store(&[100]);
        let mode = PairingMode::batch_range(nid(50), nid(100)).unwrap();
        let pairs: Vec<_> = store.night_pairs_iter(mode).collect();
        assert_eq!(pairs, vec![]);
    }

    #[test]
    fn batch_range_basic() {
        let store = make_store(&[40, 60, 80, 100]);
        let mode = PairingMode::batch_range(nid(50), nid(100)).unwrap();
        let pairs: Vec<_> = store.night_pairs_iter(mode).collect();

        // right = 100
        // left candidates: 60, 80 (40 is outside range, 100 is right)
        assert_eq!(pairs, vec![(nid(60), nid(100)), (nid(80), nid(100))]);
    }

    #[test]
    fn batch_range_all_in_range() {
        let store = make_store(&[50, 60, 70, 80, 90, 100]);
        let mode = PairingMode::batch_range(nid(50), nid(100)).unwrap();
        let pairs: Vec<_> = store.night_pairs_iter(mode).collect();

        // right = 100
        // All others are eligible lefts
        assert_eq!(
            pairs,
            vec![
                (nid(50), nid(100)),
                (nid(60), nid(100)),
                (nid(70), nid(100)),
                (nid(80), nid(100)),
                (nid(90), nid(100)),
            ]
        );
    }

    #[test]
    fn batch_range_partial_overlap() {
        let store = make_store(&[10, 20, 30, 40, 50, 60]);
        let mode = PairingMode::batch_range(nid(25), nid(55)).unwrap();
        let pairs: Vec<_> = store.night_pairs_iter(mode).collect();

        // right = 50 (latest in [25, 55])
        // left candidates: 30, 40 (in range and < 50)
        assert_eq!(pairs, vec![(nid(30), nid(50)), (nid(40), nid(50))]);
    }

    #[test]
    fn batch_range_equal_bounds() {
        let store = make_store(&[50]);
        let mode = PairingMode::batch_range(nid(50), nid(50)).unwrap();
        let pairs: Vec<_> = store.night_pairs_iter(mode).collect();

        // right = 50, no lefts possible
        assert_eq!(pairs, vec![]);
    }

    // -------------------------------------------------------------------------
    // Determinism tests
    // -------------------------------------------------------------------------

    #[test]
    fn deterministic_ordering_single_night() {
        let store = make_store(&[17, 15, 16, 20]);
        let mode = PairingMode::single_night(nid(20), 5).unwrap();

        let pairs: Vec<_> = store.night_pairs_iter(mode).collect();
        // Should be sorted by left
        assert_eq!(
            pairs,
            vec![(nid(15), nid(20)), (nid(16), nid(20)), (nid(17), nid(20)),]
        );
    }

    #[test]
    fn deterministic_ordering_batch_range() {
        let store = make_store(&[25, 15, 20, 10, 30]);
        let mode = PairingMode::batch_range(nid(10), nid(30)).unwrap();

        let pairs: Vec<_> = store.night_pairs_iter(mode).collect();
        // right = 30, lefts sorted
        assert_eq!(
            pairs,
            vec![
                (nid(10), nid(30)),
                (nid(15), nid(30)),
                (nid(20), nid(30)),
                (nid(25), nid(30)),
            ]
        );
    }

    // -------------------------------------------------------------------------
    // Property-based tests
    // -------------------------------------------------------------------------

    prop_compose! {
        fn unique_nights_vec()(v in prop::collection::vec(0u32..1000, 0..50)) -> Vec<u32> {
            let mut sorted = v;
            sorted.sort_unstable();
            sorted.dedup();
            sorted
        }
    }

    proptest! {
        /// All pairs must have left < right
        #[test]
        fn prop_left_less_than_right(
            nights in unique_nights_vec(),
            anchor in 10u32..1000,
            gap in 1u8..100,
        ) {
            let store = make_store(&nights);
            let mode = PairingMode::single_night(nid(anchor), gap).unwrap();
            let pairs = store.night_pairs_iter(mode);

            for (left, right) in pairs {
                prop_assert!(left < right);
            }
        }

        /// Single-night: all lefts must respect gap constraint
        #[test]
        fn prop_single_night_gap_constraint(
            nights in unique_nights_vec(),
            anchor in 10u32..1000,
            gap in 1u8..100,
        ) {
            let store = make_store(&nights);
            let mode = PairingMode::single_night(nid(anchor), gap).unwrap();
            let pairs = store.night_pairs_iter(mode);

            for (left, right) in pairs {
                let actual_gap = right.0 - left.0;
                prop_assert!(actual_gap <= gap as u32);
            }
        }

        /// Single-night: right must be the anchor (if present)
        #[test]
        fn prop_single_night_right_is_anchor(
            nights in unique_nights_vec(),
            anchor in 10u32..1000,
            gap in 1u8..100,
        ) {
            let store = make_store(&nights);
            let mode = PairingMode::single_night(nid(anchor), gap).unwrap();
            let pairs = store.night_pairs_iter(mode).collect::<Vec<_>>();

            if !pairs.is_empty() {
                let expected_anchor = mode.anchor().unwrap();
                for (_, right) in pairs {
                    prop_assert_eq!(right, expected_anchor);
                }
            }
        }

        /// Batch range: all lefts must be in range
        #[test]
        fn prop_batch_range_lefts_in_range(
            nights in unique_nights_vec(),
            start in 0u32..400,
            end in 400u32..500,
        ) {
            let store = make_store(&nights);
            let mode = PairingMode::batch_range(nid(start), nid(end)).unwrap();
            let pairs = store.night_pairs_iter(mode);

            for (left, _) in pairs {
                prop_assert!(left.0 >= start);
                prop_assert!(left.0 <= end);
            }
        }

        /// Batch range: right must be latest in range
        #[test]
        fn prop_batch_range_right_is_latest(
            nights in unique_nights_vec(),
            start in 0u32..400,
            end in 400u32..500,
        ) {
            let store = make_store(&nights);
            let mode = PairingMode::batch_range(nid(start), nid(end)).unwrap();
            let pairs = store.night_pairs_iter(mode).collect::<Vec<_>>();

            if !pairs.is_empty() {
                let (_, right) = pairs[0];

                // right should be the max night in [start, end] present in store
                let expected_right = nights
                    .iter()
                    .map(|&n| nid(n))
                    .filter(|&n| n.0 >= start && n.0 <= end)
                    .max();

                if let Some(expected) = expected_right {
                    prop_assert_eq!(right, expected);
                }

                // All pairs should have same right
                for (_, r) in &pairs {
                    prop_assert_eq!(*r, right);
                }
            }
        }

        /// All left nights must be present in the store
        #[test]
        fn prop_all_lefts_in_store(
            nights in unique_nights_vec(),
            anchor in 10u32..1000,
            gap in 1u8..100,
        ) {
            let store = make_store(&nights);
            let mode = PairingMode::single_night(nid(anchor), gap).unwrap();
            let pairs = store.night_pairs_iter(mode);

            let store_nights: std::collections::HashSet<u32> = nights.into_iter().collect();

            for (left, _) in pairs {
                prop_assert!(store_nights.contains(&left.0));
            }
        }

        /// Determinism: repeated calls yield same result
        #[test]
        fn prop_deterministic(
            nights in unique_nights_vec(),
            anchor in 10u32..1000,
            gap in 1u8..100,
        ) {
            let store = make_store(&nights);
            let mode = PairingMode::single_night(nid(anchor), gap).unwrap();

            let pairs1 = store.night_pairs_iter(mode).collect::<Vec<_>>();
            let pairs2 = store.night_pairs_iter(mode).collect::<Vec<_>>();

            prop_assert_eq!(pairs1, pairs2);
        }

        /// Output is sorted
        #[test]
        fn prop_output_sorted(
            nights in unique_nights_vec(),
            start in 0u32..400,
            end in 400u32..500,
        ) {
            let store = make_store(&nights);
            let mode = PairingMode::batch_range(nid(start), nid(end)).unwrap();
            let pairs = store.night_pairs_iter(mode).collect::<Vec<_>>();

            let lefts: Vec<NightId> = pairs.iter().map(|(l, _)| *l).collect();
            let mut sorted_lefts = lefts.clone();
            sorted_lefts.sort();

            prop_assert_eq!(lefts, sorted_lefts);
        }
    }
}
