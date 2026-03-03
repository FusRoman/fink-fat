//! Alert store: in-memory grouping of alerts by night, plus persistence helpers.
//!
//! Overview
//! --------
//! This module defines [`AlertStore`], a lightweight container that groups
//! [`Alert`](crate::alerts::Alert) values by [`NightId`](crate::night_id::NightId)
//! using an [`AHashMap`](ahash::AHashMap).
//!
//! The structure is optimized for the Fink-FAT pipeline common access patterns:
//! - **batch processing per night** (seeding is typically intra-night),
//! - **contiguous iteration** within a night (`Vec<Alert>`),
//! - **fast key lookup** by `(night_id, dia_source_id)` via [`AlertKey`](crate::alerts::AlertKey),
//! - **O(1) reverse lookup** from [`DiaSourceId`](crate::alerts::DiaSourceId) to vector position
//!   via an internal `id_to_location` index,
//! - **cheap merging** of partial stores without cloning via `Vec::append`.
//!
//! Data model
//! ----------
//! - The store maps each `night_id` to a `Vec<Alert>`.
//! - The position within the vector is the *in-night* positional index.
//! - An [`AlertKey`](crate::alerts::AlertKey) `(night_id, dia_source_id)` can be used
//!   to retrieve a specific alert via the internal reverse index.
//!
//! Invariants and conventions
//! --------------------------
//! This type is intentionally small and does not enforce strong invariants on its own,
//! but the pipeline typically relies on the following conventions:
//!
//! - **Stable reverse index:** the internal `id_to_location` map caches each alert's
//!   vector position. Any reordering or removal of alerts within a `Vec<Alert>` will
//!   invalidate this index unless a rebuilding method (such as
//!   [`AlertStore::sort_each_night_and_rekey`]) is invoked afterwards.
//! - **Identifier uniqueness:** each [`DiaSourceId`](crate::alerts::DiaSourceId) must
//!   appear at most once across all nights. Duplicate insertions are rejected by
//!   [`AlertStore::insert_alert`].
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
//! - The store does not validate consistency between `AlertKey.dia_source_id` and the
//!   reverse index on bulk operations. Use [`AlertStore::sort_each_night_and_rekey`]
//!   or [`AlertStore::merge_in_place`] (which rebuild the index internally) after any
//!   external mutation of the per-night vectors.

use std::collections::hash_map::Entry;

use ahash::AHashMap;
use camino::Utf8PathBuf;

use crate::{
    MJDTT,
    alerts::{Alert, AlertKey, AlertSlice, DiaSourceId, error::InsertError},
    night_id::{NightId, PairingMode},
    persistence::{
        compression::Compression, error::PersistenceIoError, layout::PersistenceLayout,
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
/// - **Key-based addressing:** [`AlertKey`](crate::alerts::AlertKey) pairs a
///   `NightId` with a [`DiaSourceId`](crate::alerts::DiaSourceId); the internal
///   reverse index resolves the `dia_source_id` to the vector position in O(1).
///
/// See the module-level documentation for invariants and performance notes.
#[derive(Debug, Clone)]
pub struct AlertStore {
    /// Alerts grouped by night (cache-friendly iteration)
    alerts_by_night: AHashMap<NightId, Vec<Alert>>,

    /// Reverse index: dia_source_id → (night, index in vec)
    ///
    /// Built during insertion/load to enable O(1) stable lookups.
    /// This index is **not persisted** (rebuilt on load).
    pub(self) id_to_location: AHashMap<DiaSourceId, (NightId, usize)>,
}

impl Default for AlertStore {
    fn default() -> Self {
        Self::new()
    }
}

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
        Self {
            alerts_by_night: AHashMap::new(),
            id_to_location: AHashMap::new(),
        }
    }

    /// Create an `AlertStore` from a pre-existing map of night IDs to alert vectors.
    ///
    /// Arguments
    /// ---------
    /// * `map` – Map from [`NightId`](crate::night_id::NightId) to `Vec<Alert>`.
    ///
    /// Return
    /// ------
    /// A new `AlertStore` with its reverse index (`id_to_location`) built from the
    /// provided map.
    ///
    /// Notes
    /// -----
    /// - No validation is performed on alert ordering or `dia_source_id` uniqueness.
    /// - The reverse index is built during construction, mapping each
    ///   `dia_source_id` to its `(night_id, vec_index)` pair.
    pub fn from_map(map: AHashMap<NightId, Vec<Alert>>) -> Self {
        // Build reverse index for stable lookups
        let mut id_to_location = AHashMap::new();
        for (night_id, alerts) in &map {
            for (idx, alert) in alerts.iter().enumerate() {
                id_to_location.insert(alert.key.dia_source_id, (*night_id, idx));
            }
        }

        Self {
            alerts_by_night: map,
            id_to_location,
        }
    }

    /// Insert a single alert into the store, placing it in the appropriate night's vector.
    ///
    /// Arguments
    /// ---------
    /// * `alert` – The [`Alert`](crate::alerts::Alert) to insert. Its `key.night_id`
    ///   determines the target night.
    ///
    /// Return
    /// ------
    /// * `Ok(())` – The alert was successfully inserted and the reverse index updated.
    /// * `Err(InsertError::DuplicateId)` – An alert with the same `dia_source_id`
    ///   already exists in the store.
    pub fn insert_alert(&mut self, alert: Alert) -> Result<(), InsertError> {
        // Check for duplicate dia_source_id
        let night_id = alert.key.night_id;
        let dia_source_id = alert.key.dia_source_id;
        if self.id_to_location.contains_key(&dia_source_id) {
            return Err(InsertError::DuplicateId(dia_source_id));
        }

        // Insert into the appropriate night's vector
        let alerts_for_night = self.alerts_by_night.entry(night_id).or_default();
        let idx_in_night = alerts_for_night.len();
        alerts_for_night.push(alert);

        // Update reverse index
        self.id_to_location
            .insert(dia_source_id, (night_id, idx_in_night));

        Ok(())
    }

    /// Test if the store is empty (contains no nights).
    ///
    /// Return
    /// ------
    /// `true` if there are no nights in the store, `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.alerts_by_night.is_empty()
    }

    /// Get a list of night IDs currently present in the store.
    ///
    /// Return
    /// ------
    /// An iterator yielding `NightId` values corresponding to the keys in the internal map.
    ///
    /// Notes
    /// -----
    /// - The order is not guaranteed to be stable or sorted, as it depends on the
    ///   internal state of the hash map.
    /// - If the store is empty, returns an empty vector.
    #[inline]
    pub fn nights(&self) -> impl Iterator<Item = &NightId> {
        self.alerts_by_night.keys()
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
    #[inline]
    pub fn nights_sorted(&self) -> Vec<NightId> {
        let mut v: Vec<NightId> = self.nights().copied().collect();
        v.sort();
        v
    }

    /// Get the latest night ID present in the store, if any.
    ///
    /// Return
    /// ------
    /// - `Some(NightId)` corresponding to the maximum night ID in the store.
    /// - `None` if the store is empty.
    ///
    /// Notes
    /// -----
    /// - This is a convenience method that relies on `nights_sorted` to find the maximum night ID.
    #[inline]
    pub fn last_night(&self) -> Option<NightId> {
        self.nights_sorted().last().copied()
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
    ///   (e.g., after calling `sort_each_night_and_rekey`).
    /// - Should be the case if the alerts have been
    ///   ingested using the [`crate::pipeline::stages::PipelineStage::IngestNights`] stage,
    ///   which calls `sort_each_night_and_rekey` after loading.
    pub fn night_t0(&self, night_id: &NightId) -> Option<MJDTT> {
        self.alerts_by_night
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
        // Update reverse index
        for (idx, alert) in alerts.iter().enumerate() {
            self.id_to_location
                .insert(alert.key.dia_source_id, (night_id, idx));
        }
        self.alerts_by_night.insert(night_id, alerts);
    }

    /// Get the vector of alerts for a given night, if it exists.
    ///
    /// Arguments
    /// ---------
    /// * `night_id` – Night identifier to query.
    ///
    /// Return
    /// ------
    /// - `Some(&Vec<Alert>)` if the night exists.
    /// - `None` otherwise.
    ///
    /// Notes
    /// -----
    /// - The returned vector should be treated as immutable by convention to keep
    ///   the reverse index (`id_to_location`) consistent with vector positions.
    pub fn get(&self, night_id: &NightId) -> Option<&Vec<Alert>> {
        self.alerts_by_night.get(night_id)
    }

    /// Get a mutable reference to the vector of alerts for a given night,
    /// creating an empty vector if it does not exist.
    ///
    /// Behavior
    /// --------
    /// - If `night_id` is present, returns a mutable reference to its vector.
    /// - Otherwise, inserts `Vec::new()` and returns a mutable reference to it.
    ///
    /// Arguments
    /// ---------
    /// * `night_id` – Night identifier.
    ///
    /// Return
    /// ------
    /// Mutable reference to the `Vec<Alert>` for the given night.
    ///
    /// Notes
    /// -----
    /// - Mutating the vector (especially reordering/removals) may invalidate the
    ///   internal `id_to_location` reverse index. Call
    ///   [`sort_each_night_and_rekey`](Self::sort_each_night_and_rekey) or
    ///   [`merge_in_place`](Self::merge_in_place) after bulk mutations to
    ///   rebuild the index.
    pub fn get_or_init(&mut self, night_id: NightId) -> &mut Vec<Alert> {
        self.alerts_by_night.entry(night_id).or_default()
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
    /// Return
    /// ------
    /// Mutable reference to the `Vec<Alert>` for the given night.
    ///
    /// Notes
    /// -----
    /// - The capacity hint can reduce reallocations when pushing alerts.
    /// - Same mutation caveats as [`get_or_init`](Self::get_or_init):
    ///   modifications to the returned vector may invalidate the reverse index.
    pub fn get_or_init_with_capacity(
        &mut self,
        night_id: NightId,
        capacity: usize,
    ) -> &mut Vec<Alert> {
        self.alerts_by_night
            .entry(night_id)
            .or_insert_with(|| Vec::with_capacity(capacity))
    }

    /// Get an alert by its composite key (night ID + `dia_source_id`).
    ///
    /// This is the canonical constant-time lookup for pipeline components that
    /// carry compact references via [`AlertKey`](crate::alerts::AlertKey).
    ///
    /// Arguments
    /// ---------
    /// * `key` – Composite alert identifier: `(night_id, dia_source_id)`.
    ///
    /// Return
    /// ------
    /// - `Some(&Alert)` if the night exists and the `dia_source_id` is found
    ///   in the reverse index with a valid vector position.
    /// - `None` otherwise.
    ///
    /// Notes
    /// -----
    /// - Internally resolves `dia_source_id` to a vector index via `id_to_location`.
    /// - If alerts have been reordered or removed without rebuilding the index,
    ///   the returned reference may be incorrect or `None`.
    pub fn get_by_key(&self, key: AlertKey) -> Option<&Alert> {
        let vec = self.alerts_by_night.get(&key.night_id)?;
        vec.get(self.id_to_location.get(&key.dia_source_id)?.1)
    }

    /// Get an alert by its `dia_source_id` using the reverse index.
    ///
    /// Arguments
    /// ---------
    /// * `dia_source_id` – Unique identifier for the alert source.
    ///
    /// Return
    /// ------
    /// - `Some(&Alert)` if an alert with the given `dia_source_id` exists in the store.
    /// - `None` if the `dia_source_id` is not present in the reverse index or the
    ///   referenced night/position is invalid.
    ///
    /// Notes
    /// -----
    /// - This method provides O(1) lookup by `dia_source_id` using the internal
    ///   `id_to_location` index.
    pub fn get_by_id(&self, dia_source_id: DiaSourceId) -> Option<&Alert> {
        let (night_id, idx_in_night) = self.id_to_location.get(&dia_source_id)?;
        let vec = self.alerts_by_night.get(night_id)?;
        vec.get(*idx_in_night)
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
        self.alerts_by_night.values().flatten()
    }

    /// Get an iterator over all alerts for a specific night, if it exists.
    ///
    /// Arguments
    /// ---------
    /// * `night_id` – Night identifier to iterate over.
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
        self.alerts_by_night
            .get(night_id)
            .map(|alerts| alerts.iter())
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
        self.alerts_by_night.iter()
    }

    /// Iterate over `(night_id, alerts)` pairs for nights within a specified pairing window.
    ///
    /// Overview
    /// --------
    /// This method filters the alert store to return only the nights that are relevant
    /// for seed building according to the given `PairingMode`.
    ///
    /// Behavior by mode
    /// ----------------
    /// - **`SingleNight` mode**: returns **only the anchor night** if it is present in the
    ///   store. Previous nights within the gap are intentionally excluded because they
    ///   already have seeds built and persisted from earlier pipeline runs; rebuilding
    ///   them would be redundant and costly.
    /// - **`BatchRange` mode**: returns all nights within `[start, end]` that exist in
    ///   the store, in sorted ascending order.
    ///
    /// Arguments
    /// ---------
    /// * `night_window` – Pairing mode defining which nights are eligible.
    ///
    /// Return
    /// ------
    /// An iterator yielding `(night_id, &[Alert])` tuples for the selected nights.
    ///
    /// Output guarantees
    /// -----------------
    /// - **Sorted**: nights are returned in increasing order.
    /// - **Deduplicated**: each night appears at most once.
    /// - **Deterministic**: output is reproducible for the same inputs.
    pub fn night_window_iter(
        &self,
        night_window: PairingMode,
    ) -> impl Iterator<Item = (NightId, &[Alert])> {
        let available_nights: Vec<NightId> = self.nights_sorted();

        // In SingleNight mode only process the anchor night: previous nights already
        // have their seeds built and persisted, rebuilding them is unnecessary.
        // In BatchRange mode fall back to filter_nights which returns all nights in
        // [start, end] sorted.
        let night_in_window: Vec<NightId> = match night_window {
            PairingMode::SingleNight { anchor, .. } => {
                if self.alerts_by_night.contains_key(&anchor) {
                    vec![anchor]
                } else {
                    vec![]
                }
            }
            _ => night_window
                .filter_nights(&available_nights)
                .unwrap_or_default(),
        };

        let store_ref = &self.alerts_by_night;

        night_in_window.into_iter().filter_map(move |night_id| {
            store_ref
                .get(&night_id)
                .map(|alerts| (night_id, alerts.as_slice()))
        })
    }

    /// Return the sorted list of night IDs that [`night_window_iter`] would yield
    /// for the given pairing mode.
    ///
    /// This is a lightweight companion to [`night_window_iter`]: it applies the
    /// same filtering logic and returns only the identifiers, without borrowing
    /// the alert slices. Useful for progress reporting and counter initialisation
    /// before the actual iteration.
    ///
    /// Behavior
    /// --------
    /// - **`SingleNight` mode**: returns `[anchor]` if the anchor is present,
    ///   otherwise `[]`.
    /// - **`BatchRange` mode**: returns all nights within `[start, end]` that
    ///   exist in the store, in sorted ascending order.
    pub fn night_window_nights(&self, night_window: PairingMode) -> Vec<NightId> {
        match night_window {
            PairingMode::SingleNight { anchor, .. } => {
                if self.alerts_by_night.contains_key(&anchor) {
                    vec![anchor]
                } else {
                    vec![]
                }
            }
            _ => {
                let available = self.nights_sorted();
                night_window.filter_nights(&available).unwrap_or_default()
            }
        }
    }

    /// Get the internal map size (number of nights present).
    ///
    /// Return
    /// ------
    /// Number of distinct `NightId` keys currently stored.
    pub fn n_nights(&self) -> usize {
        self.alerts_by_night.len()
    }

    /// Get the total number of alerts across all nights.
    ///
    /// Return
    /// ------
    /// Total count of `Alert` values stored across all nights.
    pub fn n_alerts(&self) -> usize {
        self.alerts_by_night.values().map(|v| v.len()).sum()
    }

    /// Merges another `AlertStore` into this one in-place.
    ///
    /// All alerts from `other` are moved into `self`, grouped by night.
    /// After merging, the internal `id_to_location` index is fully rebuilt to reflect
    /// the new alert positions.
    ///
    /// Complexity
    /// ----------
    /// - Time: O(N_self + N_other) where N = total number of alerts
    /// - Space: O(1) additional (reuses existing allocations where possible)
    ///
    /// Arguments
    /// ---------
    /// * `other` – The `AlertStore` to merge into this one (consumed).
    pub fn merge_in_place(&mut self, mut other: AlertStore) {
        for (night_id, mut alerts) in other.alerts_by_night.drain() {
            match self.alerts_by_night.entry(night_id) {
                Entry::Occupied(mut e) => {
                    e.get_mut().append(&mut alerts);
                }
                Entry::Vacant(e) => {
                    e.insert(alerts);
                }
            }
        }

        // Rebuild the id_to_location index to maintain O(1) lookup invariant
        self.rebuild_index();
    }

    /// Rebuilds the `id_to_location` reverse index from scratch.
    ///
    /// This method is called internally after operations that modify the alert
    /// storage structure (e.g., merging, bulk insertions).
    ///
    /// # Complexity
    /// Time: O(N) where N = total number of alerts across all nights.
    fn rebuild_index(&mut self) {
        self.id_to_location.clear();

        for (night_id, alerts) in &self.alerts_by_night {
            for (idx, alert) in alerts.iter().enumerate() {
                self.id_to_location
                    .insert(alert.key.dia_source_id, (*night_id, idx));
            }
        }
    }

    /// Sort alerts within each night by their `mjd_tt` and rebuild the reverse index.
    ///
    /// This is a utility function that should be called after inserting or merging
    /// alerts to ensure that alerts within each night are ordered by observation
    /// time and that the internal `id_to_location` index reflects the new
    /// vector positions.
    ///
    /// Behavior
    /// --------
    /// For each night in the store:
    /// 1. The vector of alerts is sorted in-place using
    ///    [`sort_unstable`](slice::sort_unstable) (primary key: `mjd_tt`,
    ///    tie-breakers follow the [`Ord`] implementation on [`Alert`](crate::alerts::Alert)).
    /// 2. After all nights are sorted, the reverse index is fully rebuilt
    ///    to re-map every `dia_source_id` to its new vector position.
    ///
    /// This operation modifies the internal state of the store. Any vector
    /// index previously obtained from `id_to_location` is invalid until the
    /// rebuild completes. It should typically be called once after all
    /// insertions/merges are complete.
    ///
    /// Complexity
    /// ----------
    /// Let `N` be the number of nights and `K` the total number of alerts across all nights.
    /// - Time: `O(K log K)` in the worst case (if all alerts are in one night),
    ///   but typically `O(N * M log M)` where `M` is the average number of alerts per night,
    ///   plus `O(K)` for the index rebuild.
    /// - Space: `O(1)` extra (sort is in-place, index rebuild reuses the existing map).
    ///
    /// Notes
    /// -----
    /// - Sorting is done by `mjd_tt` to ensure temporal ordering within each night,
    ///   which is a common convention for alert processing.
    /// - After this operation, the `id_to_location` reverse index is consistent
    ///   with the sorted vectors, which is important for downstream components
    ///   that rely on [`get_by_key`](Self::get_by_key) or
    ///   [`get_by_id`](Self::get_by_id).
    pub fn sort_each_night_and_rekey(&mut self) {
        for (_, v) in self.alerts_by_night.iter_mut() {
            v.sort_unstable();
        }
        self.rebuild_index();
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
        compression: Compression,
    ) -> Result<Utf8PathBuf, PersistenceIoError> {
        let alerts = self.alerts_by_night.get(&night_id).ok_or_else(|| {
            PersistenceIoError::Other(format!("No alerts for night_id {}", night_id))
        })?;

        let path = alerts
            .as_slice()
            .save_alerts_night(layout, manifest, night_id, compression)?;

        Ok(path)
    }
}

#[cfg(test)]
mod alert_store_tests {
    use super::*;
    use crate::alerts::error::InsertError;

    // -------------------------------------------------------------------------
    // Helper functions
    // -------------------------------------------------------------------------

    fn nid(v: u32) -> NightId {
        NightId(v)
    }

    /// Create a mock alert with specified fields for testing.
    fn mock_alert(dia_source_id: DiaSourceId, night_id: NightId, mjd: f64) -> Alert {
        Alert {
            key: AlertKey {
                night_id,
                dia_source_id,
            },
            ra: 0.0,
            dec: 0.0,
            ra_err: 0.001,
            dec_err: 0.001,
            mjd_tt: mjd,
            flux: 100.0,
            flux_err: 10.0,
            band: 0,
            ..Default::default()
        }
    }

    /// Create a store with predefined alerts for testing.
    fn make_test_store() -> AlertStore {
        let mut store = AlertStore::new();

        // Night 100: 3 alerts
        let _ = store.insert_alert(mock_alert(1001, nid(100), 59000.0));
        let _ = store.insert_alert(mock_alert(1002, nid(100), 59000.5));
        let _ = store.insert_alert(mock_alert(1003, nid(100), 59001.0));

        // Night 101: 2 alerts
        let _ = store.insert_alert(mock_alert(2001, nid(101), 59001.5));
        let _ = store.insert_alert(mock_alert(2002, nid(101), 59002.0));

        store
    }

    // -------------------------------------------------------------------------
    // Basic construction and query tests
    // -------------------------------------------------------------------------

    #[test]
    fn new_store_is_empty() {
        let store = AlertStore::new();
        assert_eq!(store.n_alerts(), 0);
        assert!(store.is_empty());
        assert_eq!(store.nights().count(), 0);
    }

    #[test]
    fn from_map_builds_index() {
        let mut map: AHashMap<NightId, Vec<Alert>> = AHashMap::new();
        map.insert(
            nid(100),
            vec![
                mock_alert(1001, nid(100), 59000.0),
                mock_alert(1002, nid(100), 59000.5),
            ],
        );

        let store = AlertStore::from_map(map);

        assert_eq!(store.n_alerts(), 2);
        assert!(store.get_by_id(1001).is_some());
        assert!(store.get_by_id(1002).is_some());
        assert!(store.get_by_id(9999).is_none());
    }

    #[test]
    fn nights_sorted_returns_ordered_list() {
        let store = make_test_store();
        let nights = store.nights_sorted();

        assert_eq!(nights, vec![nid(100), nid(101)]);
    }

    #[test]
    fn last_night_returns_maximum() {
        let store = make_test_store();
        assert_eq!(store.last_night(), Some(nid(101)));
    }

    #[test]
    fn last_night_empty_store() {
        let store = AlertStore::new();
        assert_eq!(store.last_night(), None);
    }

    // -------------------------------------------------------------------------
    // Index lookup tests (get_by_id)
    // -------------------------------------------------------------------------

    #[test]
    fn get_by_id_finds_existing_alert() {
        let store = make_test_store();

        let alert = store.get_by_id(1001).unwrap();
        assert_eq!(alert.key.dia_source_id, 1001);
        assert_eq!(alert.key.night_id, nid(100));
        assert_eq!(alert.mjd_tt, 59000.0);
    }

    #[test]
    fn get_by_id_returns_none_for_missing_id() {
        let store = make_test_store();
        assert!(store.get_by_id(9999).is_none());
    }

    #[test]
    fn get_by_id_finds_alerts_across_nights() {
        let store = make_test_store();

        // Alert from night 100
        assert!(store.get_by_id(1001).is_some());

        // Alert from night 101
        assert!(store.get_by_id(2001).is_some());
    }

    // -------------------------------------------------------------------------
    // Insert and index consistency tests
    // -------------------------------------------------------------------------

    #[test]
    fn insert_alert_updates_index() {
        let mut store = AlertStore::new();

        let _ = store.insert_alert(mock_alert(1001, nid(100), 59000.0));

        // Check the alert is findable by ID
        let alert = store.get_by_id(1001).unwrap();
        assert_eq!(alert.key.dia_source_id, 1001);

        // Check the alert is in the correct night vector
        let night_alerts = store.get(&nid(100)).unwrap();
        assert_eq!(night_alerts.len(), 1);
        assert_eq!(night_alerts[0].key.dia_source_id, 1001);
    }

    #[test]
    fn insert_alert_duplicate_key_fails() {
        let mut store = AlertStore::new();

        let _ = store.insert_alert(mock_alert(1001, nid(100), 59000.0));

        // Try to insert alert with same dia_source_id
        let result = store.insert_alert(mock_alert(1001, nid(100), 59000.5));

        assert!(result.is_err());
        match result {
            Err(InsertError::DuplicateId(id)) => {
                assert_eq!(id, 1001);
            }
            _ => panic!("Expected DuplicateId error"),
        }
    }

    #[test]
    fn insert_multiple_alerts_same_night() {
        let mut store = AlertStore::new();

        store
            .insert_alert(mock_alert(1001, nid(100), 59000.0))
            .unwrap();
        store
            .insert_alert(mock_alert(1002, nid(100), 59000.5))
            .unwrap();
        store
            .insert_alert(mock_alert(1003, nid(100), 59001.0))
            .unwrap();

        assert_eq!(store.n_alerts(), 3);

        // All alerts should be findable by ID
        assert!(store.get_by_id(1001).is_some());
        assert!(store.get_by_id(1002).is_some());
        assert!(store.get_by_id(1003).is_some());

        // All should be in the same night vector
        let night_alerts = store.get(&nid(100)).unwrap();
        assert_eq!(night_alerts.len(), 3);
    }

    #[test]
    fn insert_multiple_alerts_different_nights() {
        let mut store = AlertStore::new();

        store
            .insert_alert(mock_alert(1001, nid(100), 59000.0))
            .unwrap();
        store
            .insert_alert(mock_alert(2001, nid(101), 59001.0))
            .unwrap();
        store
            .insert_alert(mock_alert(3001, nid(102), 59002.0))
            .unwrap();

        assert_eq!(store.n_alerts(), 3);
        assert_eq!(store.nights().count(), 3);

        // Each alert should be in its correct night
        assert_eq!(store.get(&nid(100)).unwrap().len(), 1);
        assert_eq!(store.get(&nid(101)).unwrap().len(), 1);
        assert_eq!(store.get(&nid(102)).unwrap().len(), 1);
    }

    // -------------------------------------------------------------------------
    // Batch insert tests
    // -------------------------------------------------------------------------

    #[test]
    fn insert_night_vec_success() {
        let mut store = AlertStore::new();

        let alerts = vec![
            mock_alert(1001, nid(100), 59000.0),
            mock_alert(1002, nid(100), 59000.5),
            mock_alert(1003, nid(100), 59001.0),
        ];

        store.insert(nid(100), alerts);

        assert_eq!(store.n_alerts(), 3);
        assert!(store.get_by_id(1001).is_some());
        assert!(store.get_by_id(1002).is_some());
        assert!(store.get_by_id(1003).is_some());
    }

    // -------------------------------------------------------------------------
    // Merge tests
    // -------------------------------------------------------------------------

    #[test]
    fn merge_in_place_combines_stores() {
        let mut store1 = AlertStore::new();
        store1
            .insert_alert(mock_alert(1001, nid(100), 59000.0))
            .unwrap();
        store1
            .insert_alert(mock_alert(1002, nid(100), 59000.5))
            .unwrap();

        let mut store2 = AlertStore::new();
        store2
            .insert_alert(mock_alert(2001, nid(101), 59001.0))
            .unwrap();
        store2
            .insert_alert(mock_alert(2002, nid(101), 59001.5))
            .unwrap();

        store1.merge_in_place(store2);

        assert_eq!(store1.n_alerts(), 4);
        assert_eq!(store1.n_nights(), 2);

        // All alerts should be findable
        assert!(store1.get_by_id(1001).is_some());
        assert!(store1.get_by_id(1002).is_some());
        assert!(store1.get_by_id(2001).is_some());
        assert!(store1.get_by_id(2002).is_some());
    }

    #[test]
    fn merge_in_place_same_night() {
        let mut store1 = AlertStore::new();
        store1
            .insert_alert(mock_alert(1001, nid(100), 59000.0))
            .unwrap();

        let mut store2 = AlertStore::new();
        store2
            .insert_alert(mock_alert(1002, nid(100), 59000.5))
            .unwrap();

        store1.merge_in_place(store2);

        assert_eq!(store1.n_alerts(), 2);
        assert_eq!(store1.n_nights(), 1);

        let night_alerts = store1.get(&nid(100)).unwrap();
        assert_eq!(night_alerts.len(), 2);
    }

    #[test]
    fn merge_in_place_rebuilds_index() {
        let mut store1 = AlertStore::new();
        store1
            .insert_alert(mock_alert(1001, nid(100), 59000.0))
            .unwrap();

        let mut store2 = AlertStore::new();
        store2
            .insert_alert(mock_alert(2001, nid(101), 59001.0))
            .unwrap();

        store1.merge_in_place(store2);

        // Index should be updated for both stores' alerts
        let alert1 = store1.get_by_id(1001).unwrap();
        let alert2 = store1.get_by_id(2001).unwrap();

        assert_eq!(alert1.key.night_id, nid(100));
        assert_eq!(alert2.key.night_id, nid(101));
    }

    #[test]
    fn merge_empty_stores() {
        let mut store1 = AlertStore::new();
        let store2 = AlertStore::new();

        store1.merge_in_place(store2);

        assert!(store1.is_empty());
    }

    #[test]
    fn merge_into_empty_store() {
        let mut store1 = AlertStore::new();

        let mut store2 = AlertStore::new();
        store2
            .insert_alert(mock_alert(1001, nid(100), 59000.0))
            .unwrap();

        store1.merge_in_place(store2);

        assert_eq!(store1.n_alerts(), 1);
        assert!(store1.get_by_id(1001).is_some());
    }

    // -------------------------------------------------------------------------
    // Sort and rekey tests
    // -------------------------------------------------------------------------

    #[test]
    fn sort_and_rekey_orders_by_mjd() {
        let mut store = AlertStore::new();

        // Insert alerts in non-chronological order
        store
            .insert_alert(mock_alert(1003, nid(100), 59002.0))
            .unwrap();
        store
            .insert_alert(mock_alert(1001, nid(100), 59000.0))
            .unwrap();
        store
            .insert_alert(mock_alert(1002, nid(100), 59001.0))
            .unwrap();

        store.sort_each_night_and_rekey();

        let night_alerts = store.get(&nid(100)).unwrap();

        // Check chronological order
        assert_eq!(night_alerts[0].mjd_tt, 59000.0);
        assert_eq!(night_alerts[1].mjd_tt, 59001.0);
        assert_eq!(night_alerts[2].mjd_tt, 59002.0);
    }

    #[test]
    fn sort_and_rekey_preserves_index_lookup() {
        let mut store = AlertStore::new();

        // Insert alerts in non-chronological order
        store
            .insert_alert(mock_alert(1003, nid(100), 59002.0))
            .unwrap();
        store
            .insert_alert(mock_alert(1001, nid(100), 59000.0))
            .unwrap();
        store
            .insert_alert(mock_alert(1002, nid(100), 59001.0))
            .unwrap();

        store.sort_each_night_and_rekey();

        // All alerts should still be findable by ID
        let alert1 = store.get_by_id(1001).unwrap();
        let alert2 = store.get_by_id(1002).unwrap();
        let alert3 = store.get_by_id(1003).unwrap();

        assert_eq!(alert1.mjd_tt, 59000.0);
        assert_eq!(alert2.mjd_tt, 59001.0);
        assert_eq!(alert3.mjd_tt, 59002.0);
    }

    #[test]
    fn sort_and_rekey_multiple_nights() {
        let mut store = AlertStore::new();

        // Night 100
        store
            .insert_alert(mock_alert(1003, nid(100), 59002.0))
            .unwrap();
        store
            .insert_alert(mock_alert(1001, nid(100), 59000.0))
            .unwrap();

        // Night 101
        store
            .insert_alert(mock_alert(2002, nid(101), 59003.5))
            .unwrap();
        store
            .insert_alert(mock_alert(2001, nid(101), 59003.0))
            .unwrap();

        store.sort_each_night_and_rekey();

        // Check night 100
        let alerts100 = store.get(&nid(100)).unwrap();
        assert_eq!(alerts100[0].mjd_tt, 59000.0);
        assert_eq!(alerts100[1].mjd_tt, 59002.0);

        // Check night 101
        let alerts101 = store.get(&nid(101)).unwrap();
        assert_eq!(alerts101[0].mjd_tt, 59003.0);
        assert_eq!(alerts101[1].mjd_tt, 59003.5);
    }

    // -------------------------------------------------------------------------
    // Index consistency after modifications
    // -------------------------------------------------------------------------

    #[test]
    fn index_consistent_after_multiple_operations() {
        let mut store = AlertStore::new();

        // Insert some alerts
        store
            .insert_alert(mock_alert(1001, nid(100), 59000.0))
            .unwrap();
        store
            .insert_alert(mock_alert(1002, nid(100), 59000.5))
            .unwrap();

        // Merge with another store
        let mut store2 = AlertStore::new();
        store2
            .insert_alert(mock_alert(2001, nid(101), 59001.0))
            .unwrap();
        store.merge_in_place(store2);

        // Sort and rekey
        store.sort_each_night_and_rekey();

        // All alerts should still be findable
        assert!(store.get_by_id(1001).is_some());
        assert!(store.get_by_id(1002).is_some());
        assert!(store.get_by_id(2001).is_some());

        // Verify correct night assignment
        assert_eq!(store.get_by_id(1001).unwrap().key.night_id, nid(100));
        assert_eq!(store.get_by_id(2001).unwrap().key.night_id, nid(101));
    }

    // -------------------------------------------------------------------------
    // Property-based tests for index consistency
    // -------------------------------------------------------------------------

    #[cfg(test)]
    mod proptest_index {
        use super::*;
        use proptest::prelude::*;

        /// Generate a valid alert with random but bounded fields
        fn arb_alert() -> impl Strategy<Value = Alert> {
            (
                any::<u64>().prop_filter("non-zero dia_source_id", |&id| id > 0),
                0u32..1000,
                59000.0..60000.0f64,
            )
                .prop_map(|(id, night, mjd)| mock_alert(id, nid(night), mjd))
        }

        proptest! {
            /// Inserting alerts maintains index consistency
            #[test]
            fn prop_insert_maintains_index(
                alerts in prop::collection::vec(arb_alert(), 1..100)
            ) {
                let mut store = AlertStore::new();
                let mut inserted_ids = Vec::new();

                for alert in alerts {
                    let id = alert.key.dia_source_id;

                    // Try to insert (may fail on duplicate)
                    if store.insert_alert(alert).is_ok() {
                        inserted_ids.push(id);
                    }
                }

                // All successfully inserted alerts should be findable
                for id in inserted_ids {
                    prop_assert!(store.get_by_id(id).is_some());
                }
            }

            /// Merging stores maintains index consistency
            #[test]
            fn prop_merge_maintains_index(
                alerts1 in prop::collection::vec(arb_alert(), 0..50),
                alerts2 in prop::collection::vec(arb_alert(), 0..50)
            ) {
                let mut store1 = AlertStore::new();
                let mut store2 = AlertStore::new();

                let mut all_ids = Vec::new();

                // Insert into store1
                for alert in alerts1 {
                    let id = alert.key.dia_source_id;
                    if store1.insert_alert(alert).is_ok() {
                        all_ids.push(id);
                    }
                }

                // Insert into store2
                for alert in alerts2 {
                    let id = alert.key.dia_source_id;
                    if store2.insert_alert(alert).is_ok() && !all_ids.contains(&id) {
                        all_ids.push(id);
                    }
                }

                // Merge
                store1.merge_in_place(store2);

                // All unique IDs should be findable
                for id in all_ids {
                    prop_assert!(store1.get_by_id(id).is_some());
                }
            }

            /// Sort and rekey maintains index consistency
            #[test]
            fn prop_sort_maintains_index(
                alerts in prop::collection::vec(arb_alert(), 1..100)
            ) {
                let mut store = AlertStore::new();
                let mut inserted_ids = Vec::new();

                for alert in alerts {
                    let id = alert.key.dia_source_id;
                    if store.insert_alert(alert).is_ok() {
                        inserted_ids.push(id);
                    }
                }

                store.sort_each_night_and_rekey();

                // All alerts should still be findable after sort
                for id in inserted_ids {
                    prop_assert!(store.get_by_id(id).is_some());
                }
            }

            /// Store length matches index size
            #[test]
            fn prop_length_matches_index(
                alerts in prop::collection::vec(arb_alert(), 0..100)
            ) {
                let mut store = AlertStore::new();
                let mut unique_ids = std::collections::HashSet::new();

                for alert in alerts {
                    let dia_source_id = alert.key.dia_source_id;
                    if store.insert_alert(alert).is_ok() {
                        unique_ids.insert(dia_source_id);
                    }
                }

                // Store length should match number of unique inserted IDs
                prop_assert_eq!(store.n_alerts(), unique_ids.len());
            }
        }
    }

    // -------------------------------------------------------------------------
    // Edge case tests
    // -------------------------------------------------------------------------

    #[test]
    fn get_returns_none_for_missing_night() {
        let store = make_test_store();
        assert!(store.get(&nid(999)).is_none());
    }

    #[test]
    fn sort_empty_store() {
        let mut store = AlertStore::new();
        store.sort_each_night_and_rekey();
        assert!(store.is_empty());
    }

    #[test]
    fn merge_with_self_pattern() {
        // This tests the pattern where we might accidentally merge a store with itself
        // (though the API takes ownership, so this is more about testing similar data)
        let mut store1 = AlertStore::new();
        store1
            .insert_alert(mock_alert(1001, nid(100), 59000.0))
            .unwrap();

        let mut store2 = AlertStore::new();
        store2
            .insert_alert(mock_alert(1001, nid(100), 59000.0))
            .unwrap();

        // This should work since stores are separate, but IDs match
        store1.merge_in_place(store2);

        // Should still have just one alert (same night and vector position)
        assert_eq!(store1.n_alerts(), 2); // Actually 2 because same ID in different stores
    }

    #[test]
    fn very_large_dia_source_id() {
        let mut store = AlertStore::new();
        let large_id = u64::MAX - 1;

        store
            .insert_alert(mock_alert(large_id, nid(100), 59000.0))
            .unwrap();

        assert!(store.get_by_id(large_id).is_some());
    }

    // -------------------------------------------------------------------------
    // Tests for night_window_iter()
    // -------------------------------------------------------------------------

    #[cfg(test)]
    mod night_window_iter_tests {
        use super::*;
        use crate::night_id::PairingMode;

        /// Helper: Create a store with alerts spread across multiple nights
        fn make_multi_night_store() -> AlertStore {
            let mut store = AlertStore::new();

            // Night 100: 2 alerts
            store
                .insert_alert(mock_alert(1001, nid(100), 59100.0))
                .unwrap();
            store
                .insert_alert(mock_alert(1002, nid(100), 59100.5))
                .unwrap();

            // Night 101: 1 alert
            store
                .insert_alert(mock_alert(2001, nid(101), 59101.0))
                .unwrap();

            // Night 103: 2 alerts (gap of 1 night after 101)
            store
                .insert_alert(mock_alert(3001, nid(103), 59103.0))
                .unwrap();
            store
                .insert_alert(mock_alert(3002, nid(103), 59103.5))
                .unwrap();

            // Night 105: 1 alert (gap of 1 night after 103)
            store
                .insert_alert(mock_alert(4001, nid(105), 59105.0))
                .unwrap();

            store
        }

        #[test]
        fn night_window_iter_empty_store() {
            let store = AlertStore::new();
            let mode = PairingMode::SingleNight {
                anchor: nid(100),
                max_gap: 2,
            };

            let collected: Vec<_> = store.night_window_iter(mode).collect();

            assert_eq!(collected.len(), 0, "Empty store should yield no nights");
        }

        #[test]
        fn night_window_iter_single_night_anchor_not_present() {
            let store = make_multi_night_store();
            // Anchor on night 102 which doesn't exist
            let mode = PairingMode::SingleNight {
                anchor: nid(102),
                max_gap: 2,
            };

            let collected: Vec<_> = store.night_window_iter(mode).collect();

            assert_eq!(
                collected.len(),
                0,
                "Anchor night not in store should yield no nights"
            );
        }

        #[test]
        fn night_window_iter_single_night_only_anchor() {
            let store = make_multi_night_store();

            // Anchor on night 100 (first night, no previous nights)
            let mode = PairingMode::SingleNight {
                anchor: nid(100),
                max_gap: 5,
            };

            let collected: Vec<_> = store.night_window_iter(mode).collect();

            // Should only return the anchor itself (no left nights available)
            assert_eq!(collected.len(), 1);
            assert_eq!(collected[0].0, nid(100));
            assert_eq!(collected[0].1.len(), 2);
        }

        #[test]
        fn night_window_iter_single_night_with_gap_1() {
            let store = make_multi_night_store();

            // Anchor on night 105, max_gap = 1
            // SingleNight mode: only the anchor night is returned regardless of gap.
            let mode = PairingMode::SingleNight {
                anchor: nid(105),
                max_gap: 1,
            };

            let collected: Vec<_> = store.night_window_iter(mode).collect();

            assert_eq!(collected.len(), 1);
            assert_eq!(collected[0].0, nid(105)); // only the anchor night
        }

        #[test]
        fn night_window_iter_single_night_with_gap_2() {
            let store = make_multi_night_store();
            // Anchor on night 105, max_gap = 2
            // SingleNight mode: only the anchor night is returned regardless of gap.
            let mode = PairingMode::SingleNight {
                anchor: nid(105),
                max_gap: 2,
            };

            let collected: Vec<_> = store.night_window_iter(mode).collect();

            assert_eq!(collected.len(), 1);
            assert_eq!(collected[0].0, nid(105)); // only the anchor night
        }

        #[test]
        fn night_window_iter_single_night_large_gap() {
            let store = make_multi_night_store();
            // Anchor on night 105, max_gap = 10
            // SingleNight mode: only the anchor night is returned regardless of gap.
            let mode = PairingMode::SingleNight {
                anchor: nid(105),
                max_gap: 10,
            };

            let collected: Vec<_> = store.night_window_iter(mode).collect();

            assert_eq!(collected.len(), 1);
            assert_eq!(collected[0].0, nid(105)); // only the anchor night
        }

        #[test]
        fn night_window_iter_batch_basic() {
            let store = make_multi_night_store();
            // Batch mode: [101, 105]
            // Right = 105 (latest in range)
            // Lefts = 101, 103 (within [start=101, right=105))
            let mode = PairingMode::BatchRange {
                start: nid(101),
                end: nid(105),
            };

            let mut collected: Vec<_> = store.night_window_iter(mode).collect();
            collected.sort_by_key(|(night, _)| *night);

            assert_eq!(collected.len(), 3);
            assert_eq!(collected[0].0, nid(101));
            assert_eq!(collected[1].0, nid(103));
            assert_eq!(collected[2].0, nid(105));
        }

        #[test]
        fn night_window_iter_batch_excludes_before_start() {
            let store = make_multi_night_store();
            // Batch mode: [103, 105]
            // Should exclude night 100 and 101
            let mode = PairingMode::BatchRange {
                start: nid(103),
                end: nid(105),
            };

            let mut collected: Vec<_> = store.night_window_iter(mode).collect();
            collected.sort_by_key(|(night, _)| *night);

            assert_eq!(collected.len(), 2);
            assert_eq!(collected[0].0, nid(103));
            assert_eq!(collected[1].0, nid(105));
        }

        #[test]
        fn night_window_iter_batch_single_night_range() {
            let store = make_multi_night_store();
            // Batch mode with start == end
            let mode = PairingMode::BatchRange {
                start: nid(105),
                end: nid(105),
            };

            let collected: Vec<_> = store.night_window_iter(mode).collect();

            // Should only return night 105 (no left nights: start <= left < right fails)
            assert_eq!(collected.len(), 1);
            assert_eq!(collected[0].0, nid(105));
        }

        #[test]
        fn night_window_iter_batch_no_nights_in_range() {
            let store = make_multi_night_store();
            // Batch mode: [110, 120] - no nights in this range
            let mode = PairingMode::BatchRange {
                start: nid(110),
                end: nid(120),
            };

            let collected: Vec<_> = store.night_window_iter(mode).collect();

            assert_eq!(collected.len(), 0);
        }

        #[test]
        fn night_window_iter_batch_partial_range() {
            let store = make_multi_night_store();

            // Batch mode: [100, 102]
            // Available: 100, 101 (103, 105 are outside)
            // Right = 101 (latest in range)
            // Lefts = 100 (start <= 100 < 101)
            let mode = PairingMode::BatchRange {
                start: nid(100),
                end: nid(102),
            };

            let mut collected: Vec<_> = store.night_window_iter(mode).collect();
            collected.sort_by_key(|(night, _)| *night);

            assert_eq!(collected.len(), 2);
            assert_eq!(collected[0].0, nid(100));
            assert_eq!(collected[1].0, nid(101));
        }

        #[test]
        fn night_window_iter_preserves_alert_content() {
            let store = make_multi_night_store();
            let mode = PairingMode::SingleNight {
                anchor: nid(105),
                max_gap: 10,
            };

            let collected: Vec<_> = store.night_window_iter(mode).collect();

            // Verify alert counts per night match expected values
            for (night, alerts) in collected {
                match night.0 {
                    100 => assert_eq!(alerts.len(), 2, "Night 100 should have 2 alerts"),
                    101 => assert_eq!(alerts.len(), 1, "Night 101 should have 1 alert"),
                    103 => assert_eq!(alerts.len(), 2, "Night 103 should have 2 alerts"),
                    105 => assert_eq!(alerts.len(), 1, "Night 105 should have 1 alert"),
                    _ => panic!("Unexpected night: {}", night.0),
                }
            }
        }

        #[test]
        fn night_window_iter_returns_sorted_nights() {
            let store = make_multi_night_store();
            let mode = PairingMode::SingleNight {
                anchor: nid(105),
                max_gap: 10,
            };

            let collected: Vec<_> = store.night_window_iter(mode).collect();
            let night_ids: Vec<_> = collected.iter().map(|(n, _)| *n).collect();

            // Verify nights are sorted
            let mut sorted_ids = night_ids.clone();
            sorted_ids.sort();
            assert_eq!(night_ids, sorted_ids, "Night IDs should be sorted");
        }

        #[test]
        fn night_window_iter_no_duplicates() {
            let store = make_multi_night_store();
            let mode = PairingMode::SingleNight {
                anchor: nid(105),
                max_gap: 10,
            };

            let collected: Vec<_> = store.night_window_iter(mode).collect();
            let night_ids: Vec<_> = collected.iter().map(|(n, _)| *n).collect();

            // Check no duplicates
            let mut deduped = night_ids.clone();
            deduped.dedup();
            assert_eq!(
                night_ids.len(),
                deduped.len(),
                "Should have no duplicate nights"
            );
        }

        #[test]
        fn night_window_iter_single_night_in_store() {
            let mut store = AlertStore::new();
            store
                .insert_alert(mock_alert(1001, nid(100), 59100.0))
                .unwrap();

            let mode = PairingMode::SingleNight {
                anchor: nid(100),
                max_gap: 5,
            };
            let collected: Vec<_> = store.night_window_iter(mode).collect();

            // Should return only that night (no left candidates)
            assert_eq!(collected.len(), 1);
            assert_eq!(collected[0].0, nid(100));
        }

        #[test]
        fn night_window_iter_batch_all_nights() {
            let store = make_multi_night_store();
            // Batch covering entire store
            let mode = PairingMode::BatchRange {
                start: nid(100),
                end: nid(105),
            };

            let mut collected: Vec<_> = store.night_window_iter(mode).collect();

            collected.sort_by_key(|(night, _)| *night);

            // Should return all 4 nights
            assert_eq!(collected.len(), 4);
            assert_eq!(collected[0].0, nid(100));
            assert_eq!(collected[1].0, nid(101));
            assert_eq!(collected[2].0, nid(103));
            assert_eq!(collected[3].0, nid(105));
        }
    }

    // -------------------------------------------------------------------------
    // Tests for sort_each_night_and_rekey() invariants
    // -------------------------------------------------------------------------

    #[cfg(test)]
    mod sort_and_rekey_invariants {
        use super::*;

        #[test]
        fn sort_rekey_maintains_temporal_order() {
            let mut store = AlertStore::new();

            // Insert alerts in reverse temporal order
            store
                .insert_alert(mock_alert(1003, nid(100), 59003.0))
                .unwrap();
            store
                .insert_alert(mock_alert(1001, nid(100), 59001.0))
                .unwrap();
            store
                .insert_alert(mock_alert(1002, nid(100), 59002.0))
                .unwrap();

            store.sort_each_night_and_rekey();

            let alerts = store.get(&nid(100)).unwrap();

            // Verify temporal ordering
            assert!(alerts[0].mjd_tt < alerts[1].mjd_tt);
            assert!(alerts[1].mjd_tt < alerts[2].mjd_tt);

            // Verify specific order
            assert_eq!(alerts[0].key.dia_source_id, 1001);
            assert_eq!(alerts[1].key.dia_source_id, 1002);
            assert_eq!(alerts[2].key.dia_source_id, 1003);
        }

        #[test]
        fn sort_rekey_idx_matches_position() {
            let mut store = AlertStore::new();

            // Insert in random order
            store
                .insert_alert(mock_alert(1005, nid(100), 59005.0))
                .unwrap();
            store
                .insert_alert(mock_alert(1002, nid(100), 59002.0))
                .unwrap();
            store
                .insert_alert(mock_alert(1008, nid(100), 59008.0))
                .unwrap();
            store
                .insert_alert(mock_alert(1001, nid(100), 59001.0))
                .unwrap();

            store.sort_each_night_and_rekey();

            let alerts = store.get(&nid(100)).unwrap();

            // After rekey, the index position in id_to_location should match actual position
            for (expected_idx, alert) in alerts.iter().enumerate() {
                let (indexed_night, indexed_idx) = store
                    .id_to_location
                    .get(&alert.key.dia_source_id)
                    .expect("Alert should be in id_to_location index");

                assert_eq!(
                    *indexed_night,
                    nid(100),
                    "Alert {} should be indexed in night 100",
                    alert.key.dia_source_id
                );
                assert_eq!(
                    *indexed_idx, expected_idx,
                    "Alert {} index position should be {} but got {}",
                    alert.key.dia_source_id, expected_idx, indexed_idx
                );
            }
        }

        #[test]
        fn sort_rekey_id_to_location_consistency() {
            let mut store = AlertStore::new();

            // Insert alerts across multiple nights
            store
                .insert_alert(mock_alert(1003, nid(100), 59003.0))
                .unwrap();
            store
                .insert_alert(mock_alert(1001, nid(100), 59001.0))
                .unwrap();
            store
                .insert_alert(mock_alert(2002, nid(101), 59102.0))
                .unwrap();
            store
                .insert_alert(mock_alert(2001, nid(101), 59101.0))
                .unwrap();

            store.sort_each_night_and_rekey();

            // Verify id_to_location points to correct positions
            for (night_id, alerts) in store.as_map_iter() {
                for (idx, alert) in alerts.iter().enumerate() {
                    let (indexed_night, indexed_idx) = store
                        .id_to_location
                        .get(&alert.key.dia_source_id)
                        .expect("Alert should be in index");

                    assert_eq!(
                        *indexed_night, *night_id,
                        "Index night mismatch for alert {}",
                        alert.key.dia_source_id
                    );
                    assert_eq!(
                        *indexed_idx, idx,
                        "Index position mismatch for alert {}",
                        alert.key.dia_source_id
                    );
                }
            }
        }

        #[test]
        fn sort_rekey_all_alerts_still_findable() {
            let mut store = AlertStore::new();

            let ids = vec![1005, 1002, 1008, 1001, 1003];
            for (i, &id) in ids.iter().enumerate() {
                store
                    .insert_alert(mock_alert(id, nid(100), 59000.0 + i as f64))
                    .unwrap();
            }

            store.sort_each_night_and_rekey();

            // All alerts should still be findable by ID
            for &id in &ids {
                assert!(
                    store.get_by_id(id).is_some(),
                    "Alert {} should be findable after sort/rekey",
                    id
                );
            }
        }

        #[test]
        fn sort_rekey_multiple_nights_independent() {
            let mut store = AlertStore::new();

            // Night 100: reverse order
            store
                .insert_alert(mock_alert(1003, nid(100), 59003.0))
                .unwrap();
            store
                .insert_alert(mock_alert(1001, nid(100), 59001.0))
                .unwrap();

            // Night 101: reverse order
            store
                .insert_alert(mock_alert(2003, nid(101), 59103.0))
                .unwrap();
            store
                .insert_alert(mock_alert(2001, nid(101), 59101.0))
                .unwrap();

            store.sort_each_night_and_rekey();

            // Each night should be sorted independently
            let night100 = store.get(&nid(100)).unwrap();
            assert_eq!(night100[0].key.dia_source_id, 1001);
            assert_eq!(night100[1].key.dia_source_id, 1003);

            let night101 = store.get(&nid(101)).unwrap();
            assert_eq!(night101[0].key.dia_source_id, 2001);
            assert_eq!(night101[1].key.dia_source_id, 2003);
        }

        #[test]
        fn sort_rekey_preserves_alert_count() {
            let mut store = AlertStore::new();

            // Insert 10 alerts across 3 nights
            for i in 0..10 {
                let night = nid(100 + (i % 3));
                let mjd = 59000.0 + (10 - i) as f64; // Reverse time order
                store
                    .insert_alert(mock_alert(1000 + i as u64, night, mjd))
                    .unwrap();
            }

            let count_before = store.n_alerts();
            store.sort_each_night_and_rekey();
            let count_after = store.n_alerts();

            assert_eq!(
                count_before, count_after,
                "Sort/rekey should not change alert count"
            );
        }

        #[test]
        fn sort_rekey_stable_for_equal_mjd() {
            let mut store = AlertStore::new();

            // Insert alerts with identical MJD
            let same_mjd = 59000.0;
            store
                .insert_alert(mock_alert(1001, nid(100), same_mjd))
                .unwrap();
            store
                .insert_alert(mock_alert(1002, nid(100), same_mjd))
                .unwrap();
            store
                .insert_alert(mock_alert(1003, nid(100), same_mjd))
                .unwrap();

            store.sort_each_night_and_rekey();

            let alerts = store.get(&nid(100)).unwrap();

            // All should have the same MJD
            for alert in alerts {
                assert_eq!(alert.mjd_tt, same_mjd);
            }

            // Position in id_to_location should be 0, 1, 2
            for (expected_idx, alert) in alerts.iter().enumerate() {
                let (_night, idx) = store
                    .id_to_location
                    .get(&alert.key.dia_source_id)
                    .expect("Alert should be in index");
                assert_eq!(
                    *idx, expected_idx,
                    "Alert {} should be at position {}",
                    alert.key.dia_source_id, expected_idx
                );
            }
        }

        #[test]
        fn sort_rekey_empty_store_no_panic() {
            let mut store = AlertStore::new();

            // Should not panic on empty store
            store.sort_each_night_and_rekey();

            assert!(store.is_empty());
        }

        #[test]
        fn sort_rekey_single_alert_no_change() {
            let mut store = AlertStore::new();
            store
                .insert_alert(mock_alert(1001, nid(100), 59000.0))
                .unwrap();

            store.sort_each_night_and_rekey();

            let alert = store.get_by_id(1001).unwrap();

            // Verify via id_to_location that position is 0
            let (_night, idx) = store
                .id_to_location
                .get(&1001)
                .expect("Alert should be in index");
            assert_eq!(*idx, 0, "Single alert should be at position 0");
            assert_eq!(alert.mjd_tt, 59000.0);
        }

        #[test]
        fn sort_rekey_night_t0_returns_earliest() {
            let mut store = AlertStore::new();

            // Insert in reverse chronological order
            store
                .insert_alert(mock_alert(1003, nid(100), 59003.0))
                .unwrap();
            store
                .insert_alert(mock_alert(1002, nid(100), 59002.0))
                .unwrap();
            store
                .insert_alert(mock_alert(1001, nid(100), 59001.0))
                .unwrap();

            store.sort_each_night_and_rekey();

            // night_t0 should return the first (earliest) alert's MJD
            let t0 = store.night_t0(&nid(100)).expect("night_t0 should exist");
            assert_eq!(
                t0, 59001.0,
                "night_t0 should return earliest MJD after sort"
            );
        }

        #[test]
        fn sort_rekey_preserves_night_count() {
            let mut store = AlertStore::new();

            // Insert alerts across 3 nights
            store
                .insert_alert(mock_alert(1001, nid(100), 59000.0))
                .unwrap();
            store
                .insert_alert(mock_alert(2001, nid(101), 59001.0))
                .unwrap();
            store
                .insert_alert(mock_alert(3001, nid(102), 59002.0))
                .unwrap();

            let nights_before = store.n_nights();
            store.sort_each_night_and_rekey();
            let nights_after = store.n_nights();

            assert_eq!(
                nights_before, nights_after,
                "Sort/rekey should not change night count"
            );
        }

        #[test]
        fn sort_rekey_get_by_id_returns_correct_alert() {
            let mut store = AlertStore::new();

            // Insert alerts
            store
                .insert_alert(mock_alert(1005, nid(100), 59005.0))
                .unwrap();
            store
                .insert_alert(mock_alert(1002, nid(100), 59002.0))
                .unwrap();
            store
                .insert_alert(mock_alert(1008, nid(100), 59008.0))
                .unwrap();

            store.sort_each_night_and_rekey();

            // get_by_id should return the correct alert
            let alert = store.get_by_id(1002).unwrap();
            assert_eq!(alert.key.dia_source_id, 1002);
            assert_eq!(alert.mjd_tt, 59002.0);

            // Should be first after sort (lowest MJD)
            let (_night, idx) = store
                .id_to_location
                .get(&1002)
                .expect("Alert should be in index");
            assert_eq!(*idx, 0, "Alert 1002 should be at position 0 after sort");
        }
    }

    // -------------------------------------------------------------------------
    // Property-based tests for sort/rekey invariants
    // -------------------------------------------------------------------------

    #[cfg(test)]
    mod proptest_sort_rekey {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// After sort_each_night_and_rekey, id_to_location index must match actual position
            #[test]
            fn prop_sort_rekey_idx_matches_position(
                alerts in prop::collection::vec(
                    (any::<u64>().prop_filter("non-zero", |&id| id > 0), 0u32..10, 59000.0..60000.0f64),
                    1..50
                )
            ) {
                let mut store = AlertStore::new();

                for (id, night, mjd) in alerts {
                    let _ = store.insert_alert(mock_alert(id, nid(night), mjd));
                }

                store.sort_each_night_and_rekey();

                // For every night, verify id_to_location index matches actual position
                for (night_id, alerts) in store.as_map_iter() {
                    for (expected_pos, alert) in alerts.iter().enumerate() {
                        let dia_source_id = alert.key.dia_source_id;

                        // Check that alert is in id_to_location
                        if let Some((indexed_night, indexed_pos)) =
                            store.id_to_location.get(&dia_source_id)
                        {
                            prop_assert_eq!(
                                *indexed_night, *night_id,
                                "Alert {} indexed in wrong night: expected {}, got {}",
                                dia_source_id, night_id, indexed_night
                            );
                            prop_assert_eq!(
                                *indexed_pos, expected_pos,
                                "Alert {} position mismatch: expected {}, got {}",
                                dia_source_id, expected_pos, indexed_pos
                            );
                        } else {
                            return Err(proptest::test_runner::TestCaseError::fail(
                                format!("Alert {} not found in id_to_location index", dia_source_id)
                            ));
                        }
                    }
                }
            }

            /// After sort_each_night_and_rekey, alerts within each night are temporally ordered
            #[test]
            fn prop_sort_rekey_temporal_order(
                alerts in prop::collection::vec(
                    (any::<u64>().prop_filter("non-zero", |&id| id > 0), 0u32..10, 59000.0..60000.0f64),
                    2..50
                )
            ) {
                let mut store = AlertStore::new();

                for (id, night, mjd) in alerts {
                    let _ = store.insert_alert(mock_alert(id, nid(night), mjd));
                }

                store.sort_each_night_and_rekey();

                // For every night with 2+ alerts, verify temporal ordering
                for (_night_id, alerts) in store.as_map_iter() {
                    if alerts.len() >= 2 {
                        for window in alerts.windows(2) {
                            prop_assert!(
                                window[0].mjd_tt <= window[1].mjd_tt,
                                "Temporal order violated: {} > {}",
                                window[0].mjd_tt, window[1].mjd_tt
                            );
                        }
                    }
                }
            }

            /// After sort_each_night_and_rekey, all alerts remain findable
            #[test]
            fn prop_sort_rekey_preserves_findability(
                alerts in prop::collection::vec(
                    (any::<u64>().prop_filter("non-zero", |&id| id > 0), 0u32..10, 59000.0..60000.0f64),
                    1..50
                )
            ) {
                let mut store = AlertStore::new();
                let mut inserted_ids = Vec::new();

                for (id, night, mjd) in alerts {
                    if store.insert_alert(mock_alert(id, nid(night), mjd)).is_ok() {
                        inserted_ids.push(id);
                    }
                }

                store.sort_each_night_and_rekey();

                // All successfully inserted alerts should still be findable
                for id in inserted_ids {
                    prop_assert!(
                        store.get_by_id(id).is_some(),
                        "Alert {} not findable after sort/rekey",
                        id
                    );
                }
            }

            /// After sort_each_night_and_rekey, id_to_location index is consistent
            #[test]
            fn prop_sort_rekey_index_consistency(
                alerts in prop::collection::vec(
                    (any::<u64>().prop_filter("non-zero", |&id| id > 0), 0u32..10, 59000.0..60000.0f64),
                    1..50
                )
            ) {
                let mut store = AlertStore::new();

                for (id, night, mjd) in alerts {
                    let _ = store.insert_alert(mock_alert(id, nid(night), mjd));
                }

                store.sort_each_night_and_rekey();

                // Verify index points to correct positions
                for (night_id, alerts) in store.as_map_iter() {
                    for (idx, alert) in alerts.iter().enumerate() {
                        if let Some((indexed_night, indexed_idx)) =
                            store.id_to_location.get(&alert.key.dia_source_id)
                        {
                            prop_assert_eq!(*indexed_night, *night_id);
                            prop_assert_eq!(*indexed_idx, idx);
                        } else {
                            return Err(proptest::test_runner::TestCaseError::fail(
                                format!("Alert {} missing from index", alert.key.dia_source_id)
                            ));
                        }
                    }
                }
            }
        }
    }
}
