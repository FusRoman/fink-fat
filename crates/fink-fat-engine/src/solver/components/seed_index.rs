//! Dense global indexing for seeds.
//!
//! Overview
//! --------
//! Many algorithms operating on large graphs (e.g., Union-Find for connected
//! components) benefit from representing nodes as **dense integer ids**
//! in `[0..N_total)`.
//!
//! This module provides a lightweight mapping from the stable seed identifier
//! `SeedKey = (night_id, idx_in_night)` to a **global dense index**.
//!
//! The mapping is computed by assigning each night a global base offset:
//!
//! ```text
//! global_idx(seed) = base[night_id] + idx_in_night
//! ```
//!
//! Requirements
//! ------------
//! This indexing scheme assumes that within each night:
//!
//! - `SeedKey.idx_in_night` is a dense index starting at 0,
//! - and `idx_in_night < number_of_seeds_in_that_night`.
//!
//! If these conditions are violated, `idx_of_key()` may produce invalid indices
//! (including out-of-range values) or panic due to missing base entries.
//!
//! Determinism
//! -----------
//! The base offsets are built in a deterministic order by iterating nights in
//! sorted `NightId` order (`SeedStore::night_ids_sorted()`).
//!
//! This ensures that global indices are stable across runs as long as:
//! - the set of nights and per-night seed counts are stable,
//! - and `night_ids_sorted()` is deterministic.
//!
//! Complexity
//! ----------
//! Let `K` be the number of nights.
//!
//! - Building the index is `O(K)` plus the cost of obtaining the sorted night list.
//! - `idx_of_key()` is `O(1)` expected time (hash map lookup).
//!
//! Usage
//! -----
//! The main consumer is component computation (Union-Find), where seeds must be
//! addressed by dense ids.

use ahash::AHashMap;

use crate::{night_id::NightId, persistence::seed_node::SeedKey, pipeline::seed_store::SeedStore};

/// Dense identifier used for seeds in global index space.
///
/// This is typically used as the node id type in algorithms that require
/// contiguous indexing (e.g., Union-Find).
pub type SeedId = u32;

/// Map `(night_id, idx_in_night)` -> global dense index `[0..N_total)`.
///
/// The global index is computed by assigning each night a base offset and
/// interpreting `idx_in_night` as an offset within that night:
///
/// ```text
/// global_idx = base[night_id] + idx_in_night
/// ```
///
/// Stored data
/// -----------
/// - `base[night_id]` is the starting global index for seeds of that night.
/// - `n_total` is the total number of seeds across all nights.
///
/// Requirements
/// ------------
/// - `SeedKey.idx_in_night` must be dense and 0-based within each night.
/// - The `SeedStore` must contain all nights referenced by queried `SeedKey`s.
///
/// Notes
/// -----
/// - This index does not validate that `idx_in_night` is within bounds for the
///   corresponding night. It is the caller’s responsibility to ensure validity.
/// - `idx_of_key()` uses `self.base[&key.night_id]` and will panic if the night id
///   is unknown (not present in the index).
#[derive(Debug, Clone)]
pub struct SeedGlobalIndex {
    /// Global base offset per night:
    /// `base[night_id]` is the first global id assigned to that night.
    base: AHashMap<NightId, SeedId>,

    /// Total number of seeds across all nights.
    n_total: u32,
}

impl SeedGlobalIndex {
    /// Build global night base offsets in deterministic night order.
    ///
    /// The index is built by iterating nights in sorted order and assigning
    /// contiguous ranges:
    ///
    /// ```text
    /// night N0: base = 0
    /// night N1: base = len(N0)
    /// night N2: base = len(N0) + len(N1)
    /// ...
    /// ```
    ///
    /// Arguments
    /// ---------
    /// * `seed_store` – Seed storage grouped by night; provides:
    ///   - the sorted night list,
    ///   - the seed count per night.
    ///
    /// Return
    /// ------
    /// `SeedGlobalIndex` containing:
    /// - base offsets for each night,
    /// - total number of seeds across all nights.
    ///
    /// Notes
    /// -----
    /// - Determinism relies on `seed_store.night_ids_sorted()` being deterministic.
    /// - If a night is present in the returned night list but `len_for_night` returns
    ///   `None`, the night is treated as having length 0.
    pub fn build(seed_store: &SeedStore) -> Self {
        // Nights are processed in sorted order to make the mapping deterministic.
        let nights: Vec<NightId> = seed_store.night_ids_sorted();

        let mut base: AHashMap<NightId, SeedId> = AHashMap::default();
        let mut cursor: u32 = 0;

        for nid in nights {
            // Assign the base offset for this night.
            base.insert(nid, cursor);

            // Advance cursor by the number of seeds in this night.
            let len = seed_store.len_for_night(&nid).unwrap_or(0) as u32;
            cursor += len;
        }

        Self {
            base,
            n_total: cursor,
        }
    }

    /// Return the total number of seeds in the global index space.
    ///
    /// Return
    /// ------
    /// Total number of seeds across all nights.
    #[inline]
    pub fn n_total(&self) -> usize {
        self.n_total as usize
    }

    /// Return the global dense index of a seed identified by its `SeedKey`.
    ///
    /// The returned value is:
    ///
    /// ```text
    /// base[key.night_id] + key.idx_in_night
    /// ```
    ///
    /// Arguments
    /// ---------
    /// * `key` – Stable seed identifier `(night_id, idx_in_night)`.
    ///
    /// Return
    /// ------
    /// Global dense index in `[0..N_total)`.
    ///
    /// Panics
    /// ------
    /// - Panics if `key.night_id` is not present in the index (missing base offset).
    ///
    /// Notes
    /// -----
    /// - This function does not bounds-check `idx_in_night` against the number of
    ///   seeds in that night.
    /// - Callers are expected to query only valid `SeedKey`s derived from the same
    ///   `SeedStore` used to build the index.
    #[inline]
    pub fn idx_of_key(&self, key: SeedKey) -> usize {
        // Indexing with `[]` intentionally panics if the night id is unknown.
        (self.base[&key.night_id] + key.idx_in_night) as usize
    }
}
