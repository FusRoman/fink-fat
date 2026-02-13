//! Alert store: in-memory grouping of alerts by night, plus persistence helpers.
//!
//! Overview
//! --------
//! This module defines [`AlertStore`], a lightweight container that groups
//! [`Alert`](crate::persistence::alert::Alert) values by [`NightId`](crate::night_id::NightId)
//! using an [`AHashMap`](ahash::AHashMap).
//!
//! The structure is optimized for the Fink-FAT pipeline common access patterns:
//! - **batch processing per night** (seeding is typically intra-night),
//! - **contiguous iteration** within a night (`Vec<Alert>`),
//! - **fast key lookup** by `(night_id, idx_in_night)` via [`AlertKey`](crate::persistence::alert::AlertKey),
//! - **cheap merging** of partial stores without cloning via `Vec::append`.
//!
//! Data model
//! ----------
//! - The store maps each `night_id` to a `Vec<Alert>`.
//! - The index within the vector is the *in-night* alert index.
//! - An [`AlertKey`](crate::persistence::alert::AlertKey) can be used to retrieve
//!   a specific alert: `(night_id, idx_in_night)`.
//!
//! Invariants and conventions
//! --------------------------
//! This type is intentionally small and does not enforce strong invariants on its own,
//! but the pipeline typically relies on the following conventions:
//!
//! - **Stable indexing per night:** `idx_in_night` refers to the index in the stored vector.
//!   Any reordering of `Vec<Alert>` will change the meaning of existing keys.
//! - **Immutability by convention:** alerts are treated as immutable once inserted, which
//!   simplifies sharing across threads and avoids hard-to-debug aliasing issues.
//! - **Night completeness is contextual:** a night may contain a subset of all alerts
//!   (e.g., after filtering) depending on pipeline stage.
//!
//! Performance notes
//! -----------------
//! - `AHashMap` provides fast hashing for integer-like keys (such as `NightId`).
//! - Within a night, `Vec<Alert>` offers cache-friendly iteration.
//! - Prefer [`AlertStore::merge_in_place`] to combine stores:
//!   it **moves** alerts without cloning and uses `Vec::append`.
//! - If you need deterministic iteration order across nights, sort nights at call site;
//!   hash map iteration order is not stable.
//!
//! Persistence integration
//! -----------------------
//! The store includes a convenience helper [`AlertStore::save_alert_night`] to persist
//! the alerts for a given night using the project persistence layer:
//! - `AlertSlice::save_alerts_night(layout, manifest, night_id)` is used to write data
//!   and update the [`Manifest`](crate::persistence::manifest::Manifest).
//!
//! This helper is intentionally conservative:
//! - it only writes the requested night,
//! - it updates/creates the manifest entry for that night,
//! - it does **not** remove stale manifest entries for nights not present in the store.
//!
//! Limitations
//! -----------
//! - This module does not expose deletion APIs or compaction logic.
//! - Thread-safety is handled at higher layers (e.g., by partitioning nights or using
//!   immutable sharing patterns).
//! - The store does not validate that `AlertKey.idx_in_night` matches `Alert` internal
//!   fields (if any). Keys are treated as external indices into vectors.

use std::collections::hash_map::Entry;

use ahash::AHashMap;
use camino::Utf8PathBuf;

use crate::{
    MJDTT,
    night_id::{NightId, NightWindow},
    persistence::{
        alert::{Alert, AlertKey, AlertSlice},
        error::PersistenceIoError,
        layout::PersistenceLayout,
        manifest::Manifest,
    },
};

/// In-memory store of alerts grouped by night.
///
/// This is a thin wrapper around an `AHashMap<NightId, Vec<Alert>>`
/// providing common operations used by the ingestion, seeding, and persistence
/// layers of the pipeline.
///
/// Key properties
/// --------------
/// - **Grouping by night:** each key is a [`NightId`](crate::night_id::NightId).
/// - **Contiguous storage:** alerts for a night are in a `Vec<Alert>` for
///   cache-friendly iteration.
/// - **Index-based addressing:** [`AlertKey`](crate::persistence::alert::AlertKey)
///   locates an alert by `(night_id, idx_in_night)`.
///
/// See the module-level documentation for invariants and performance notes.
#[derive(Debug, Clone)]
pub struct AlertStore(AHashMap<NightId, Vec<Alert>>);

impl AlertStore {
    /// Create a new empty `AlertStore`.
    ///
    /// Behavior
    /// --------
    /// - Allocates an empty internal [`AHashMap`](ahash::AHashMap).
    /// - No nights are present initially.
    ///
    /// Complexity
    /// ----------
    /// - Time: `O(1)`
    /// - Space: `O(1)` (empty map)
    pub fn new() -> Self {
        Self(AHashMap::new())
    }

    /// Create an `AlertStore` from a pre-existing map of night IDs to alert vectors.
    ///
    /// Arguments
    /// ---------
    /// * `map` – Map from [`NightId`](crate::night_id::NightId) to `Vec<Alert>`.
    ///
    /// Notes
    /// -----
    /// - No validation is performed on alert ordering or key consistency.
    /// - The caller is responsible for ensuring that `idx_in_night` conventions
    ///   match vector indexing if keys are used later.
    pub fn from_map(map: AHashMap<NightId, Vec<Alert>>) -> Self {
        Self(map)
    }

    /// Get a sorted list of night IDs currently present in the store.
    ///
    /// Return
    /// ------
    /// A `Vec<NightId>` containing all night IDs in the store, sorted in ascending order.
    ///
    /// Notes
    /// -----
    /// - If the store is empty, returns an empty vector.
    /// - The sorting is done at call time; the internal map does not maintain order.
    pub fn nights(&self) -> Vec<NightId> {
        let mut v: Vec<NightId> = self.0.keys().copied().collect();
        v.sort();
        v
    }

    /// Get the night window covering all nights currently present in the store.
    ///
    /// Return
    /// ------
    /// - `Some(NightWindow)` if the store contains at least one night, where
    ///   `start` is the minimum night ID and `end` is the maximum night ID.
    /// - `None` if the store is empty (no nights).
    ///
    /// Notes
    /// -----
    /// - This is a convenience function that derives a window from the night IDs.
    pub fn get_night_window(&self) -> Option<NightWindow> {
        let night_ids = self.nights();
        if night_ids.is_empty() {
            None
        } else {
            Some(NightWindow {
                start: night_ids.first().unwrap().to_owned(),
                end: night_ids.last().unwrap().to_owned(),
            })
        }
    }

    /// Get the MJD TT of the first alert in a given night, if it exists.
    ///
    /// Arguments
    /// ---------
    /// * `night_id` – Night identifier to query.
    ///
    /// Return
    /// ------
    /// - `Some(MJDTT)` if the night exists and contains at least one alert,
    ///   where the returned value is the `mjd_tt` of the first alert in the vector.
    /// - `None` if the night does not exist or contains no alerts.
    ///
    /// Notes
    /// -----
    /// - Return the first alert's `mjd_tt` of the night only if the alerts are sorted by time 
    ///     (e.g., after calling `sort_each_night_and_rekey`).
    /// - Should be the case if the alerts have been 
    ///     ingested using the [`crate::pipeline::stages::PipelineStage::IngestNights`] stage, 
    ///     which calls `sort_each_night_and_rekey` after loading.
    pub fn night_t0(&self, night_id: &NightId) -> Option<MJDTT> {
        self.0
            .get(night_id)
            .and_then(|alerts| alerts.first().map(|a| a.mjd_tt))
    }

    /// Insert a vector of alerts for a given night.
    ///
    /// Behavior
    /// --------
    /// - Replaces any existing vector stored under `night_id`.
    /// - Ownership of `alerts` is moved into the store.
    ///
    /// Arguments
    /// ---------
    /// * `night_id` – Night identifier for the inserted alerts.
    /// * `alerts` – Contiguous alerts for that night.
    ///
    /// Notes
    /// -----
    /// - This is an overwrite operation. If you want to extend an existing night,
    ///   use [`get_or_init`](Self::get_or_init) / [`get_or_init_with_capacity`](Self::get_or_init_with_capacity)
    ///   or [`merge_in_place`](Self::merge_in_place).
    pub fn insert(&mut self, night_id: NightId, alerts: Vec<Alert>) {
        self.0.insert(night_id, alerts);
    }

    /// Get the vector of alerts for a given night, if it exists.
    ///
    /// Return
    /// ------
    /// - `Some(&Vec<Alert>)` if the night exists.
    /// - `None` otherwise.
    ///
    /// Notes
    /// -----
    /// - The returned vector should be treated as immutable by convention to keep
    ///   key stability (`AlertKey.idx_in_night`).
    pub fn get(&self, night_id: &NightId) -> Option<&Vec<Alert>> {
        self.0.get(night_id)
    }

    /// Get a mutable reference to the vector of alerts for a given night,
    /// creating an empty vector if it does not exist.
    ///
    /// Behavior
    /// --------
    /// - If `night_id` is present, returns a mutable reference to its vector.
    /// - Otherwise, inserts `Vec::new()` and returns a mutable reference to it.
    ///
    /// Notes
    /// -----
    /// - Mutating the vector (especially reordering/removals) may invalidate any
    ///   previously issued [`AlertKey`](crate::persistence::alert::AlertKey) that
    ///   expects stable indices.
    pub fn get_or_init(&mut self, night_id: NightId) -> &mut Vec<Alert> {
        self.0.entry(night_id).or_default()
    }

    /// Get a mutable reference to the vector of alerts for a given night,
    /// creating a vector with the specified capacity if it does not exist.
    ///
    /// This is a capacity-optimized variant of [`get_or_init`](Self::get_or_init),
    /// useful when the expected number of alerts for a night is known in advance.
    ///
    /// Arguments
    /// ---------
    /// * `night_id` – Night identifier.
    /// * `capacity` – Initial capacity used when creating the vector.
    ///
    /// Notes
    /// -----
    /// - The capacity hint can reduce reallocations when pushing alerts.
    pub fn get_or_init_with_capacity(
        &mut self,
        night_id: NightId,
        capacity: usize,
    ) -> &mut Vec<Alert> {
        self.0
            .entry(night_id)
            .or_insert_with(|| Vec::with_capacity(capacity))
    }

    /// Get an iterator over all alerts in the store, across all nights.
    ///
    /// Return
    /// ------
    /// An iterator yielding `&Alert` in the hash map’s internal iteration order.
    ///
    /// Notes
    /// -----
    /// - The iteration order over nights is **not deterministic** because it depends
    ///   on hash map internal state.
    /// - Within each night, order matches the `Vec<Alert>` order.
    pub fn iter(&self) -> impl Iterator<Item = &Alert> {
        self.0.values().flatten()
    }

    /// Get an iterator over all alerts for a specific night, if it exists.
    ///
    /// Return
    /// ------
    /// - `Some(iterator)` if the night exists.
    /// - `None` otherwise.
    ///
    /// Notes
    /// -----
    /// - The iterator yields alerts in the stored vector order.
    pub fn iter_night(&self, night_id: &NightId) -> Option<impl Iterator<Item = &Alert>> {
        self.0.get(night_id).map(|alerts| alerts.iter())
    }

    /// Get an alert by its key (night ID + index within night).
    ///
    /// This is the canonical constant-time lookup for pipeline components that
    /// carry compact references via [`AlertKey`](crate::persistence::alert::AlertKey).
    ///
    /// Arguments
    /// ---------
    /// * `key` – Alert location: `(night_id, idx_in_night)`.
    ///
    /// Return
    /// ------
    /// - `Some(&Alert)` if the night exists and the index is in bounds.
    /// - `None` otherwise.
    ///
    /// Notes
    /// -----
    /// - If alerts are reordered or removed from the per-night vector, previously
    ///   created keys may no longer refer to the intended alert.
    pub fn get_by_key(&self, key: AlertKey) -> Option<&Alert> {
        let vec = self.0.get(&key.night_id)?;
        vec.get(key.idx_in_night as usize)
    }

    /// Iterate over `(night_id, alerts)` pairs.
    ///
    /// Return
    /// ------
    /// An iterator over references to the internal map entries.
    ///
    /// Notes
    /// -----
    /// - The order is hash-dependent and not deterministic.
    /// - This is useful for bulk operations such as persistence or aggregation.
    pub fn as_map_iter(&self) -> impl Iterator<Item = (&NightId, &Vec<Alert>)> {
        self.0.iter()
    }

    /// Iterate over `(night_id, alerts)` pairs for nights within a specified window.
    ///
    /// Arguments
    /// ---------
    /// * `night_window` – Window to filter nights.
    ///
    /// Return
    /// ------
    /// An iterator over `(night_id, alerts)` pairs where `night_id` is within the window.
    ///
    /// Notes
    /// -----
    /// - The order is hash-dependent and not deterministic.
    /// - This is a convenient filter for operations that only need a subset of nights.
    pub fn night_window_iter(
        &self,
        night_window: NightWindow,
    ) -> impl Iterator<Item = (NightId, &[Alert])> {
        self.0
            .iter()
            .filter(move |(night_id, _)| night_window.contains(**night_id))
            .map(|(night_id, alerts)| (*night_id, alerts.as_slice()))
    }

    /// Get the internal map size (number of nights present).
    ///
    /// Return
    /// ------
    /// Number of distinct `NightId` keys currently stored.
    pub fn n_nights(&self) -> usize {
        self.0.len()
    }

    /// Get the total number of alerts across all nights.
    ///
    /// Return
    /// ------
    /// Total count of `Alert` values stored across all nights.
    pub fn n_alerts(&self) -> usize {
        self.0.values().map(|v| v.len()).sum()
    }

    /// Merge `other` into `self` (moves alerts, no cloning).
    ///
    /// This is the preferred way to combine partial alert stores (e.g., when loading
    /// multiple shards or assembling data from parallel workers).
    ///
    /// Behavior
    /// --------
    /// For each `(night_id, alerts_vec)` in `other`:
    /// - If `night_id` is **absent** in `self`, the vector is inserted as-is (moved).
    /// - If `night_id` is **present**, the incoming vector is appended to the existing
    ///   vector using `Vec::append` (moves elements, does not clone).
    ///
    /// Complexity
    /// ----------
    /// Let `M` be the number of nights in `other`, and `K` the total number of alerts moved.
    /// - Time: `O(M + K)`
    /// - Space: amortized `O(1)` extra (may reallocate existing night vectors as they grow)
    ///
    /// Notes
    /// -----
    /// - Appending preserves the relative order within each appended chunk but does not
    ///   enforce any global ordering across chunks (e.g., by observation time).
    /// - If `AlertKey.idx_in_night` stability matters across merges, ensure callers only
    ///   generate keys *after* all merges are complete (or use a stable keying scheme).
    pub fn merge_in_place(&mut self, mut other: AlertStore) {
        for (night_id, mut alerts) in other.0.drain() {
            match self.0.entry(night_id) {
                Entry::Occupied(mut e) => {
                    e.get_mut().append(&mut alerts); // move, no clone
                }
                Entry::Vacant(e) => {
                    e.insert(alerts); // move vec
                }
            }
        }
    }

    /// Sort alerts within each night by their `mjd_tt` and rekey them according to their new index.
    ///
    /// This is a utility function that can be used after inserting or merging alerts to ensure that
    /// the alerts within each night are ordered by observation time (`mjd_tt`) and that their
    /// corresponding `AlertKey.idx_in_night` values reflect this new order.
    ///
    /// Behavior
    /// --------
    /// For each night in the store:
    /// - The vector of alerts is sorted in-place by `mjd_tt` using `sort_unstable`.
    /// - After sorting, each alert's `AlertKey.idx_in_night` is updated to match its new index in the vector.
    ///
    /// - This operation modifies the internal state of the store and may invalidate any previously
    ///   issued `AlertKey` values that expect stable indices. It should be used with caution
    ///   and typically only once after all insertions/merges are complete.
    ///
    /// Complexity
    /// ----------
    /// Let `N` be the number of nights and `K` the total number of alerts across all nights.
    /// - Time: `O(K log K)` in the worst case (if all alerts are in one night),
    ///     but typically `O(N * M log M)` where   `M` is the average number of alerts per night.
    /// - Space: `O(1)` extra (sort is in-place, rekeying is done in-place).
    ///
    /// Notes
    /// -----
    /// - Sorting is done by `mjd_tt` to ensure temporal ordering within each night,
    ///     which is a common convention for alert processing.
    /// - After this operation, the `idx_in_night` field of each alert's key will
    ///     match its index in the sorted vector, which can be important for
    ///     downstream components that rely on key-based access.
    pub fn sort_each_night_and_rekey(&mut self) {
        for (night_id, v) in self.0.iter_mut() {
            v.sort_unstable(); // utilise Alert::cmp => mjd_tt d'abord
            for (idx, a) in v.iter_mut().enumerate() {
                a.key = AlertKey {
                    night_id: *night_id,
                    idx_in_night: idx as u32,
                };
            }
        }
    }

    /// Persist the alerts of a single night currently present in the store.
    ///
    /// This is a convenience function that:
    /// - retrieves the alerts for `night_id`,
    /// - writes the night payload using [`AlertSlice::save_alerts_night`],
    /// - updates the manifest entry via the persistence layer.
    ///
    /// Arguments
    /// ---------
    /// * `night_id` – Night to persist.
    /// * `layout` – Persistence layout describing the root paths and naming scheme.
    /// * `manifest` – Manifest to update with the written night payload metadata.
    ///
    /// Return
    /// ------
    /// * `Ok(Utf8PathBuf)` – Path of the written night file (as returned by the persistence layer).
    /// * `Err(PersistenceIoError)` – If:
    ///   - `night_id` is not present in the store,
    ///   - or the underlying I/O / serialization fails.
    ///
    /// Notes
    /// -----
    /// - This does **not** remove old manifest entries. If a "sync" semantics is required
    ///   (drop nights not present in the store), implement it at the call site.
    /// - This persists **only one night**. Bulk persistence can be implemented by iterating
    ///   over [`as_map_iter`](Self::as_map_iter) and calling this function per night.
    pub fn save_alert_night(
        &self,
        night_id: NightId,
        layout: &PersistenceLayout,
        manifest: &mut Manifest,
    ) -> Result<Utf8PathBuf, PersistenceIoError> {
        let alerts = self.0.get(&night_id).ok_or_else(|| {
            PersistenceIoError::Other(format!("No alerts for night_id {}", night_id))
        })?;

        let path = alerts
            .as_slice()
            .save_alerts_night(layout, manifest, night_id)?;

        Ok(path)
    }
}
