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
    night_id::{NightId, NightWindow},
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
    /// Notes
    /// -----
    /// - The iteration order is not deterministic.
    #[inline]
    pub fn nights(&self) -> impl Iterator<Item = &NightId> {
        self.0.keys()
    }

    /// Remove and return the seeds for night `n`, if present.
    #[inline]
    pub fn remove(&mut self, n: NightId) -> Option<Vec<SeedNode<'alert_lf>>> {
        self.0.remove(&n)
    }

    /// Collect nights present in the store and inside `night_window`,
    /// sorted in increasing order.
    ///
    /// Semantics
    /// ---------
    /// - Only nights present in the store are returned.
    /// - `night_window` bounds are inclusive.
    ///
    /// Determinism
    /// -----------
    /// Returned vector is sorted increasingly, stable across runs.
    #[inline]
    fn sorted_nights_in_window(&self, night_window: NightWindow) -> Vec<NightId> {
        let mut nights: Vec<NightId> = self
            .0
            .keys()
            .copied()
            .filter(|&n| night_window.contains(n))
            .collect();
        nights.sort();
        nights
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
    pub fn night_pairs_to_latest_in_window_iter(
        &self,
        night_window: NightWindow,
        max_gap_nights: u8,
    ) -> impl Iterator<Item = (NightId, NightId)> {
        // Early-out: a gap of 0 forbids any strictly earlier left night.
        // (Even if the same night existed, we require `left < right`.)
        let state = (max_gap_nights != 0).then(|| {
            // 1) Determine `right`:
            // collect nights present in the store within `night_window` and
            // select the maximum one (latest). If none exist, nothing to emit.
            let mut nights_in_window = self.sorted_nights_in_window(night_window);
            let right = match nights_in_window.pop() {
                Some(r) => r,
                None => return None,
            };

            // Compute the lower bound for eligible `left` nights:
            // we require `right - left <= max_gap_nights`
            // <=> `left >= right - max_gap_nights`.
            //
            // Use saturating arithmetic to avoid underflow when `right` is small.
            let right_u32 = right.0;
            let min_left = right_u32.saturating_sub(max_gap_nights as u32);

            // 2) Collect eligible `left` nights.
            //
            // Multi-night windows preserve the "window restricts left candidates"
            // interpretation. Single-night windows are a pipeline convenience:
            // the window picks the current `right` only, and `left` candidates are
            // searched outside the window but inside the gap constraint.
            let constrain_left_to_window = !night_window.is_single();

            let mut lefts: Vec<NightId> = self
                .0
                .keys()
                .copied()
                .filter(|&n| {
                    // Must be strictly earlier than `right`.
                    if n >= right {
                        return false;
                    }

                    // Must satisfy the gap lower bound: n >= min_left.
                    if n.0 < min_left {
                        return false;
                    }

                    // Optional window constraint:
                    // - enabled for multi-night windows,
                    // - disabled for single-night windows (current-night mode).
                    if constrain_left_to_window && !night_window.contains(n) {
                        return false;
                    }

                    true
                })
                .collect();

            // If there are no eligible `left`s, keep the anchor `right` but emit nothing.
            // Returning an empty vector avoids special-casing downstream iteration.
            if lefts.is_empty() {
                return Some((Vec::new(), right));
            }

            // Deterministic emission: sort eligible left nights increasingly.
            lefts.sort();

            Some((lefts, right))
        });

        // Flatten the optional state into an iterator, then emit one pair per left.
        state
            .flatten()
            .into_iter()
            .flat_map(|(lefts, right)| lefts.into_iter().map(move |l| (l, right)))
    }

    /// Eager version returning all `(left, latest)` pairs as a `Vec`.
    ///
    /// Overview
    /// --------
    /// This is a thin wrapper over
    /// [`SeedStore::night_pairs_to_latest_in_window_iter`]
    /// collecting the iterator into a vector.
    ///
    /// When to use
    /// -----------
    /// - Use the iterator version when streaming pairs directly into
    ///   edge construction logic.
    /// - Use this version when:
    ///   - the full pair set must be materialized,
    ///   - debugging or logging requires inspection,
    ///   - deterministic truncation or sorting is needed downstream.
    ///
    /// Complexity
    /// ----------
    /// Same as iterator version, plus `O(K)` allocation for the resulting vector.
    pub fn night_pairs_to_latest_in_window(
        &self,
        night_window: NightWindow,
        max_gap_nights: u8,
    ) -> Vec<(NightId, NightId)> {
        self.night_pairs_to_latest_in_window_iter(night_window, max_gap_nights)
            .collect()
    }
}

#[cfg(test)]
mod night_pairs_to_latest_in_window_tests {
    use super::*;
    use ahash::AHashMap;
    use proptest::prelude::*;

    fn make_store(nights: &[u32]) -> SeedStore<'static> {
        let mut map: AHashMap<NightId, Vec<SeedNode<'static>>> = AHashMap::new();
        for &n in nights {
            // We never inspect seeds for night-pair enumeration, so empty vectors are enough.
            map.insert(NightId(n), Vec::new());
        }
        SeedStore::from_map(map)
    }

    fn win(start: u32, end: u32) -> NightWindow {
        // Assumes NightWindow::new(start, end) validates (start <= end) and is infallible or returns Result.
        // If your API differs, adapt this helper only; the rest of the tests stay the same.
        #[allow(clippy::unwrap_used)]
        {
            NightWindow::new(NightId(start), NightId(end))
        }
    }

    /// Brute-force reference implementation matching the doc/spec of
    /// `night_pairs_to_latest_in_window_iter`.
    fn reference_pairs_to_latest(
        nights: &[u32],
        window: NightWindow,
        max_gap: u8,
    ) -> Vec<(NightId, NightId)> {
        if max_gap == 0 {
            return vec![];
        }

        // right = latest night present inside window
        let mut in_window: Vec<u32> = nights
            .iter()
            .copied()
            .filter(|&n| window.contains(NightId(n)))
            .collect();
        in_window.sort_unstable();
        let Some(&right_u32) = in_window.last() else {
            return vec![];
        };

        let right = NightId(right_u32);
        let min_left = right_u32.saturating_sub(max_gap as u32);
        let constrain_left_to_window = !window.is_single();

        let mut lefts: Vec<u32> = nights
            .iter()
            .copied()
            .filter(|&n| {
                if n >= right_u32 {
                    return false;
                }
                if n < min_left {
                    return false;
                }
                if constrain_left_to_window && !window.contains(NightId(n)) {
                    return false;
                }
                true
            })
            .collect();

        lefts.sort_unstable();
        lefts.into_iter().map(|l| (NightId(l), right)).collect()
    }

    #[test]
    fn max_gap_zero_is_always_empty() {
        let store = make_store(&[10, 11, 12]);
        let w = win(10, 12);

        let got: Vec<_> = store.night_pairs_to_latest_in_window_iter(w, 0).collect();
        assert!(got.is_empty());

        let got_vec = store.night_pairs_to_latest_in_window(w, 0);
        assert!(got_vec.is_empty());
    }

    #[test]
    fn empty_store_is_empty() {
        let store = make_store(&[]);
        let w = win(100, 200);

        let got: Vec<_> = store.night_pairs_to_latest_in_window_iter(w, 5).collect();
        assert!(got.is_empty());
    }

    #[test]
    fn no_right_night_in_window_is_empty_even_if_store_has_nights() {
        let store = make_store(&[1, 2, 3, 4]);
        let w = win(10, 12);

        let got: Vec<_> = store.night_pairs_to_latest_in_window_iter(w, 5).collect();
        assert!(got.is_empty());
    }

    #[test]
    fn picks_latest_right_in_window_and_emits_lefts_sorted_increasing() {
        let store = make_store(&[10, 12, 13, 20, 21]);
        let w = win(10, 21);
        let max_gap = 3;

        // right = 21
        // eligible lefts: 18..=20 present -> 20 only
        let got: Vec<_> = store
            .night_pairs_to_latest_in_window_iter(w, max_gap)
            .collect();
        assert_eq!(got, vec![(NightId(20), NightId(21))]);
    }

    #[test]
    fn multi_night_window_constrains_lefts_to_window() {
        // Store has a left candidate outside the window but within gap.
        let store = make_store(&[90, 95, 100]);
        let w = win(95, 100); // multi-night (95..=100)
        let max_gap = 20;

        // right = 100; left candidates within gap: 80..=99 => {90,95}
        // BUT multi-night window constrains lefts to window => {95} only.
        let got: Vec<_> = store
            .night_pairs_to_latest_in_window_iter(w, max_gap)
            .collect();
        assert_eq!(got, vec![(NightId(95), NightId(100))]);
    }

    #[test]
    fn single_night_window_does_not_constrain_lefts_to_window() {
        let store = make_store(&[90, 95, 100]);
        let w = win(100, 100); // single-night
        let max_gap = 20;

        // right = 100; left candidates within gap: {90,95} (both allowed, window does not constrain)
        let got: Vec<_> = store
            .night_pairs_to_latest_in_window_iter(w, max_gap)
            .collect();
        assert_eq!(
            got,
            vec![(NightId(90), NightId(100)), (NightId(95), NightId(100))]
        );
    }

    #[test]
    fn saturating_sub_prevents_underflow_for_small_right() {
        let store = make_store(&[0, 1, 2]);
        let w = win(2, 2); // right = 2
        let max_gap = 250; // huge; min_left must saturate to 0

        let got: Vec<_> = store
            .night_pairs_to_latest_in_window_iter(w, max_gap)
            .collect();
        assert_eq!(
            got,
            vec![(NightId(0), NightId(2)), (NightId(1), NightId(2))]
        );
    }

    #[test]
    fn iterator_and_vec_api_are_equivalent() {
        let store = make_store(&[10, 11, 12, 20]);
        let w = win(10, 20);
        let max_gap = 15;

        let it: Vec<_> = store
            .night_pairs_to_latest_in_window_iter(w, max_gap)
            .collect();
        let vec_api = store.night_pairs_to_latest_in_window(w, max_gap);
        assert_eq!(it, vec_api);
    }

    #[test]
    fn deterministic_output_independent_of_hashmap_insertion_order() {
        let nights_a = vec![10, 12, 13, 20, 21];
        let nights_b = vec![21, 20, 13, 12, 10]; // reverse insertion order

        let store_a = make_store(&nights_a);
        let store_b = make_store(&nights_b);

        let w = win(10, 21);
        let max_gap = 15;

        let a: Vec<_> = store_a
            .night_pairs_to_latest_in_window_iter(w, max_gap)
            .collect();
        let b: Vec<_> = store_b
            .night_pairs_to_latest_in_window_iter(w, max_gap)
            .collect();
        assert_eq!(a, b);
    }

    #[test]
    fn all_emitted_pairs_respect_invariants() {
        let store = make_store(&[1, 3, 4, 10, 12, 13]);
        let w = win(3, 13);
        let max_gap = 9;

        let pairs: Vec<_> = store
            .night_pairs_to_latest_in_window_iter(w, max_gap)
            .collect();
        for (l, r) in pairs {
            assert!(l < r, "must have left < right");
            let gap = r.0 - l.0;
            assert!(gap <= max_gap as u32, "gap must be <= max_gap_nights");
            // right must be inside window by construction
            assert!(w.contains(r));
            // For this window (multi-night), left must also be inside.
            assert!(w.contains(l));
        }
    }

    // -----------------------
    // Property-based tests
    // -----------------------

    prop_compose! {
        fn unique_nights_vec()
            (mut v in proptest::collection::vec(0u32..500u32, 0..60))
            -> Vec<u32>
        {
            v.sort_unstable();
            v.dedup();
            v
        }
    }

    fn window_from_bounds(a: u32, b: u32) -> NightWindow {
        let (start, end) = if a <= b { (a, b) } else { (b, a) };
        win(start, end)
    }

    proptest! {
        #[test]
        fn prop_matches_reference_spec(
            nights in unique_nights_vec(),
            a in 0u32..500u32,
            b in 0u32..500u32,
            max_gap in any::<u8>(),
        ) {
            let store = make_store(&nights);
            let w = window_from_bounds(a, b);

            let got: Vec<(NightId, NightId)> = store.night_pairs_to_latest_in_window_iter(w, max_gap).collect();
            let expect = reference_pairs_to_latest(&nights, w, max_gap);

            prop_assert_eq!(got, expect);
        }

        #[test]
        fn prop_vec_api_equals_iter_api(
            nights in unique_nights_vec(),
            a in 0u32..500u32,
            b in 0u32..500u32,
            max_gap in any::<u8>(),
        ) {
            let store = make_store(&nights);
            let w = window_from_bounds(a, b);

            let it: Vec<_> = store.night_pairs_to_latest_in_window_iter(w, max_gap).collect();
            let vec_api = store.night_pairs_to_latest_in_window(w, max_gap);

            prop_assert_eq!(it, vec_api);
        }

        #[test]
        fn prop_output_is_sorted_by_left_for_nonempty_output(
            nights in unique_nights_vec(),
            a in 0u32..500u32,
            b in 0u32..500u32,
            max_gap in 1u8..=255u8,
        ) {
            let store = make_store(&nights);
            let w = window_from_bounds(a, b);

            let got: Vec<_> = store.night_pairs_to_latest_in_window_iter(w, max_gap).collect();
            if got.len() >= 2 {
                for i in 1..got.len() {
                    prop_assert!(got[i-1].0 < got[i].0, "left nights must be strictly increasing");
                    prop_assert_eq!(got[i-1].1, got[i].1, "right night must be constant");
                }
            }
        }

        #[test]
        fn prop_all_pairs_obey_constraints(
            nights in unique_nights_vec(),
            a in 0u32..500u32,
            b in 0u32..500u32,
            max_gap in any::<u8>(),
        ) {
            let store = make_store(&nights);
            let w = window_from_bounds(a, b);

            let got: Vec<_> = store.night_pairs_to_latest_in_window_iter(w, max_gap).collect();

            // Either empty (common) or every pair respects the rules.
            for (l, r) in got {
                prop_assert!(l < r);

                // right must be in window (anchor selected from window)
                prop_assert!(w.contains(r));

                let gap = r.0 - l.0;
                prop_assert!(gap <= max_gap as u32);

                // For multi-night windows, left must also be in window.
                if !w.is_single() {
                    prop_assert!(w.contains(l));
                }
            }
        }
    }
}
