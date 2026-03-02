use ahash::AHashMap;
use serde::{Deserialize, Serialize};

use crate::{
    night_id::{NightId, PairingMode},
    persistence::{
        compression::Compression, error::PersistenceIoError, layout::PersistenceLayout,
        manifest::Manifest,
    },
    seeding::{SeedKey, SeedNode, SeedNodeSlice},
    solver::components::{error::ComponentError, seed_index::SeedGlobalIndex},
};

pub type SeedId = u64;

/// Persistent seed storage with built-in unique key generation.
///
/// Architecture
/// ------------
/// The store manages:
/// - A map from `NightId` to seed collections.
/// - A global monotonic counter ensuring unique seed keys across runs.
/// - Automatic persistence of both seeds and counter state.
///
/// Key Uniqueness
/// --------------
/// Each seed is assigned a globally unique `global_id` (stored in
/// `SeedKey.idx_in_night`) via an internal counter that:
/// - Increments on every seed insertion.
/// - Persists to disk with the seed data.
/// - Resumes from the saved value on next load.
///
/// Notes
/// -----
/// The field name `idx_in_night` is now a misnomer—it actually holds a
/// **global** ID. Consider renaming to `global_id` in `SeedKey` for clarity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SeedStore {
    /// Seeds organized by night
    seeds: AHashMap<NightId, Vec<SeedNode>>,

    /// Reverse index: SeedId → (night, index in vec)
    ///
    /// Built during insertion/load to enable O(1) stable lookups.
    /// This index is **not persisted** (rebuilt on load).
    id_to_location: AHashMap<SeedKey, (NightId, usize)>,

    /// Next available global seed ID (monotonically increasing)
    next_global_id: SeedId,

    /// Pool of retired IDs available for reuse
    ///
    /// Stored as a `Vec` (LIFO) for cache locality.
    /// Could use `VecDeque` for FIFO if temporal separation is desired.
    free_ids: Vec<SeedId>,
}

impl SeedStore {
    /// Create a new empty seed store.
    ///
    /// Return
    /// ------
    /// A fresh store with counter initialized to 0.
    pub fn new() -> Self {
        Self {
            seeds: AHashMap::new(),
            id_to_location: AHashMap::new(),
            next_global_id: 0,
            free_ids: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.seeds.is_empty()
    }

    pub fn from_map(map: AHashMap<NightId, Vec<SeedNode>>) -> Self {
        let mut store = Self {
            seeds: map,
            id_to_location: AHashMap::new(),
            next_global_id: 0,
            free_ids: Vec::new(),
        };

        // Build reverse index and determine next_global_id
        for (night_id, seeds) in &store.seeds {
            for (idx, seed) in seeds.iter().enumerate() {
                store.id_to_location.insert(seed.key, (*night_id, idx));
            }
        }

        store
    }

    /// Allocate a new seed ID, reusing freed IDs when available.
    ///
    /// Return
    /// ------
    /// A unique seed ID (either recycled or fresh).
    ///
    /// Strategy
    /// --------
    /// 1. Pop from free list if non-empty (LIFO).
    /// 2. Otherwise, increment `next_global_id`.
    fn allocate_id(&mut self) -> SeedId {
        self.free_ids.pop().unwrap_or_else(|| {
            let id = self.next_global_id;
            self.next_global_id += 1;
            id
        })
    }

    /// Get the next key based on the current global ID counter and with the given night.
    ///
    /// Arguments
    /// ----------
    /// * `night_id` - The night ID to associate with the new key.
    pub fn next_key(&mut self, night_id: NightId) -> SeedKey {
        SeedKey {
            night_id,
            unique_id: self.allocate_id(),
        }
    }

    /// Add a seed to the store, automatically assigning a unique key.
    ///
    /// Arguments
    /// ---------
    /// * `night_id` – Night identifier for the seed.
    /// * `seed` – Seed data (without a valid key).
    ///
    /// Return
    /// ------
    /// The assigned `SeedKey`.
    ///
    /// Notes
    /// -----
    /// - The input seed's `key` field is **overwritten**.
    /// - Seeds are appended to the night's collection in insertion order.
    /// - The global counter is incremented atomically.
    pub fn insert_seed(&mut self, night_id: NightId, mut seed: SeedNode) -> SeedKey {
        let unique_id = self.allocate_id();
        let key = SeedKey {
            night_id,
            unique_id,
        };

        seed.key = key;

        let night_seeds = self.seeds.entry(night_id).or_insert_with(Vec::new);
        let idx = night_seeds.len();
        night_seeds.push(seed);

        self.id_to_location.insert(key, (night_id, idx));

        key
    }

    /// Remove a seed, returning its ID to the free pool.
    ///
    /// Arguments
    /// ---------
    /// * `key` – Seed identifier to remove.
    ///
    /// Return
    /// ------
    /// * `Some(SeedNode)` – The removed seed.
    /// * `None` – If no seed exists with that key.
    ///
    /// Notes
    /// -----
    /// The removed ID becomes available for immediate reuse.
    pub fn remove_seed(&mut self, key: SeedKey) -> Option<SeedNode> {
        let (night_id, idx) = self.id_to_location.remove(&key)?;
        let night_seeds = self.seeds.get_mut(&night_id)?;

        let removed_seed = night_seeds.swap_remove(idx);

        // Update index for swapped seed
        if idx < night_seeds.len() {
            let swapped_key = night_seeds[idx].key;
            self.id_to_location.insert(swapped_key, (night_id, idx));
        }

        // Clean up empty nights
        if night_seeds.is_empty() {
            self.seeds.remove(&night_id);
        }

        // Return ID to free pool
        self.free_ids.push(key.unique_id);

        Some(removed_seed)
    }

    /// Compact the free list to prevent unbounded growth.
    ///
    /// Strategy
    /// --------
    /// Sorts and deduplicates the free list. Call periodically if delete
    /// rate is high.
    pub fn compact_free_list(&mut self) {
        if self.free_ids.len() > 1000 {
            self.free_ids.sort_unstable();
            self.free_ids.dedup();
        }
    }

    /// Get free list statistics.
    ///
    /// Return
    /// ------
    /// * `free_count` – Number of IDs available for reuse.
    /// * `total_allocated` – Highest ID ever assigned.
    /// * `active_count` – Number of currently existing seeds.
    pub fn id_stats(&self) -> (usize, SeedId, usize) {
        (
            self.free_ids.len(),
            self.next_global_id,
            self.id_to_location.len(),
        )
    }

    pub fn try_get_seed(&self, seed_key: SeedKey) -> Option<&SeedNode> {
        self.id_to_location
            .get(&seed_key)
            .and_then(|(night_id, idx)| self.seeds.get(night_id).and_then(|seeds| seeds.get(*idx)))
    }

    pub fn rebuild_index_at_night(&mut self, night_id: NightId) {
        if let Some(seeds) = self.seeds.get(&night_id) {
            for (idx, seed) in seeds.iter().enumerate() {
                self.id_to_location.insert(seed.key, (night_id, idx));
            }
        }
    }

    pub fn insert_vec_seed(&mut self, night_id: NightId, seeds: Vec<SeedNode>) {
        self.seeds
            .entry(night_id)
            .or_insert_with(Vec::new)
            .extend(seeds);
        self.rebuild_index_at_night(night_id);
    }

    pub fn save_seeds_night(
        &self,
        layout: &PersistenceLayout,
        manifest: &mut Manifest,
        night_id: NightId,
        compression: Compression,
    ) -> Result<(), PersistenceIoError> {
        if let Some(seeds) = self.seeds.get(&night_id) {
            seeds
                .as_slice()
                .save_seeds_night(layout, manifest, night_id, compression)?;
        }
        Ok(())
    }

    pub fn sort_night(&mut self, night_id: NightId) {
        if let Some(seeds) = self.seeds.get_mut(&night_id) {
            seeds.sort();
            self.rebuild_index_at_night(night_id);
        }
    }

    pub(crate) fn get_reverse_index(&self, key: SeedKey) -> Option<(NightId, usize)> {
        self.id_to_location.get(&key).cloned()
    }

    pub fn nights(&self) -> impl Iterator<Item = &NightId> {
        self.seeds.keys()
    }

    pub fn len_night(&self, night_id: &NightId) -> Option<usize> {
        self.seeds.get(night_id).map(|seeds| seeds.len())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&NightId, &Vec<SeedNode>)> {
        self.seeds.iter()
    }

    pub fn n_nights(&self) -> usize {
        self.seeds.len()
    }

    pub fn contains_night(&self, night_id: &NightId) -> bool {
        self.seeds.contains_key(&night_id)
    }

    pub fn get(&self, night_id: &NightId) -> Option<&[SeedNode]> {
        self.seeds.get(night_id).map(|seeds| seeds.as_slice())
    }

    /// Build a global seed index over the entire store.
    ///
    /// The global index provides a stable mapping between a "global seed id"
    /// and per-night addresses (`SeedKey`-like addressing), used by some solver
    /// components.
    #[inline]
    pub fn build_global_index(&self) -> Result<SeedGlobalIndex, ComponentError> {
        SeedGlobalIndex::build(self)
    }

    #[inline]
    pub fn night_pairs_iter(
        &self,
        pairing_mode: PairingMode,
    ) -> impl Iterator<Item = (NightId, NightId)> {
        let mut nights: Vec<NightId> = self.seeds.keys().copied().collect();
        nights.sort();
        pairing_mode.night_pairs_iter(nights)
    }
}

#[cfg(test)]
mod seed_store_tests {
    use super::*;
    use crate::{alerts::AlertKey, night_id::NightId};

    // -------------------------------------------------------------------------
    // Test helpers
    // -------------------------------------------------------------------------

    fn nid(v: u32) -> NightId {
        NightId(v)
    }

    /// Create a minimal `SeedNodeOwned` for testing.
    fn make_seed_owned(night_id: NightId, uniq_id: u64, alert_keys: Vec<AlertKey>) -> SeedNode {
        SeedNode {
            key: SeedKey {
                night_id,
                unique_id: uniq_id,
            },
            members: alert_keys,
            ..Default::default()
        }
    }

    // -------------------------------------------------------------------------
    // Construction and insertion
    // -------------------------------------------------------------------------

    #[test]
    fn new_creates_empty_store() {
        let store = SeedStore::new();
        assert!(store.is_empty());
    }

    #[test]
    fn insert_adds_seeds_for_night() {
        let mut store = SeedStore::new();
        let night_id = nid(100);

        let seeds = vec![
            make_seed_owned(
                night_id,
                0,
                vec![
                    AlertKey {
                        night_id,
                        dia_source_id: 0,
                    },
                    AlertKey {
                        night_id,
                        dia_source_id: 1,
                    },
                ],
            ),
            make_seed_owned(
                night_id,
                1,
                vec![
                    AlertKey {
                        night_id,
                        dia_source_id: 2,
                    },
                    AlertKey {
                        night_id,
                        dia_source_id: 3,
                    },
                ],
            ),
        ];

        store.insert_vec_seed(night_id, seeds);

        assert_eq!(store.n_nights(), 1);
        assert!(store.contains_night(&night_id));
        assert_eq!(store.len_night(&night_id), Some(2));
    }

    #[test]
    fn insert_overwrites_existing_night() {
        let mut store = SeedStore::new();
        let night_id = nid(100);

        let seed1 = make_seed_owned(
            night_id,
            0,
            vec![AlertKey {
                night_id,
                dia_source_id: 0,
            }],
        );
        store.insert_vec_seed(night_id, vec![seed1]);
        assert_eq!(store.len_night(&night_id), Some(1));

        let seeds2 = vec![
            make_seed_owned(
                night_id,
                1,
                vec![AlertKey {
                    night_id,
                    dia_source_id: 0,
                }],
            ),
            make_seed_owned(
                night_id,
                2,
                vec![AlertKey {
                    night_id,
                    dia_source_id: 1,
                }],
            ),
        ];
        store.insert_vec_seed(night_id, seeds2);
        assert_eq!(store.len_night(&night_id), Some(3));
    }

    #[test]
    fn insert_multiple_nights() {
        let mut store = SeedStore::new();

        for night_val in [100, 101, 102] {
            let night_id = nid(night_val);
            let seed = make_seed_owned(
                night_id,
                0,
                vec![AlertKey {
                    night_id,
                    dia_source_id: 0,
                }],
            );
            store.insert_vec_seed(night_id, vec![seed]);
        }

        assert_eq!(store.n_nights(), 3);
        assert!(store.contains_night(&nid(100)));
        assert!(store.contains_night(&nid(101)));
        assert!(store.contains_night(&nid(102)));
    }

    #[cfg(test)]
    mod sort_night_tests {
        use super::*;
        use proptest::prelude::*;

        // ── helpers ──────────────────────────────────────────────────────────────

        /// Realistic MJD (TT) range covering ZTF and early Rubin operations.
        ///
        /// ZTF first light: ~MJD 58119 (2018-01-01)
        /// Rubin early ops:  ~MJD 61000 (2026)
        const MJD_MIN: f64 = 58_000.0;
        const MJD_MAX: f64 = 62_000.0;

        /// Build a minimal `SeedNode` for ordering and reverse-index tests.
        ///
        /// Only `key` and `plane.epoch_mid` (MJD TT) are meaningful here;
        /// all other fields are left at their `Default` values.
        fn make_seed(night_id: NightId, unique_id: SeedId, epoch_mid_mjd: f64) -> SeedNode {
            let key = SeedKey {
                night_id,
                unique_id,
            };
            let mut seed = SeedNode::default();
            seed.key = key;
            seed.plane.epoch_mid = epoch_mid_mjd;
            seed
        }

        fn night(n: u32) -> NightId {
            NightId::from(n)
        }

        // ── unit tests ────────────────────────────────────────────────────────────

        /// After `sort_night`, seeds within the target night are in ascending
        /// `epoch_mid` order and the reverse index resolves every key to the
        /// correct position.
        ///
        /// The five `epoch_mid` values span a single realistic ZTF night
        /// (sub-second cadence within ~MJD 59000), inserted deliberately out of
        /// order to exercise the sort.
        #[test]
        fn sort_night_seeds_are_ordered_and_index_is_coherent() {
            let mut store = SeedStore::new();
            let nid = night(0);

            // Five observations within a single ZTF night, out of order.
            // Offsets are in days (a few minutes apart).
            let base_mjd = 59_000.0_f64;
            let offsets = [0.003, 0.001, 0.004, 0.0015, 0.002];

            let keys: Vec<SeedKey> = offsets
                .iter()
                .map(|&dt| store.insert_seed(nid, make_seed(nid, 0, base_mjd + dt)))
                .collect();

            store.sort_night(nid);

            // Seeds must be in ascending epoch_mid order.
            let seeds = store.get(&nid).expect("night must exist");
            for w in seeds.windows(2) {
                assert!(
                    w[0].plane.epoch_mid <= w[1].plane.epoch_mid,
                    "seeds out of epoch_mid order after sort_night: {} > {}",
                    w[0].plane.epoch_mid,
                    w[1].plane.epoch_mid,
                );
            }

            // Every inserted key must still resolve to the slot that holds it.
            for key in &keys {
                let (resolved_night, idx) = store
                    .get_reverse_index(*key)
                    .expect("key must be present in reverse index after sort_night");

                assert_eq!(resolved_night, nid, "night mismatch in reverse index");

                let seed_at_idx = store
                    .get(&nid)
                    .and_then(|s| s.get(idx))
                    .expect("index must point to a valid slot");

                assert_eq!(
                    seed_at_idx.key, *key,
                    "reverse index points to wrong seed after sort_night"
                );
            }
        }

        /// `sort_night` must not disturb seeds or the reverse index of nights
        /// other than the one being sorted.
        ///
        /// Night A (sorted) and night B (untouched) each span a different
        /// realistic MJD window so their observations cannot be confused.
        #[test]
        fn sort_night_does_not_affect_other_nights() {
            let mut store = SeedStore::new();
            let nid_a = night(0); // e.g. a ZTF night
            let nid_b = night(1); // e.g. an earlier ZTF night

            // Night B: three observations on MJD ~58500, inserted out of order.
            let base_b = 58_500.0_f64;
            let keys_b: Vec<SeedKey> = [0.009, 0.002, 0.007]
                .iter()
                .map(|&dt| store.insert_seed(nid_b, make_seed(nid_b, 0, base_b + dt)))
                .collect();

            // Night A: two observations on MJD ~59000, out of order.
            let base_a = 59_000.0_f64;
            store.insert_seed(nid_a, make_seed(nid_a, 0, base_a + 0.005));
            store.insert_seed(nid_a, make_seed(nid_a, 0, base_a + 0.001));

            // Snapshot night B's key order before sorting night A.
            let snapshot_b: Vec<SeedKey> =
                store.get(&nid_b).unwrap().iter().map(|s| s.key).collect();

            store.sort_night(nid_a);

            // Night B's order must be unchanged.
            let after_b: Vec<SeedKey> = store.get(&nid_b).unwrap().iter().map(|s| s.key).collect();

            assert_eq!(
                snapshot_b, after_b,
                "sort_night must not reorder seeds from other nights"
            );

            // Night B's reverse index must still be fully coherent.
            for key in &keys_b {
                let (resolved_night, idx) = store
                    .get_reverse_index(*key)
                    .expect("key from night B must still be in reverse index");

                assert_eq!(resolved_night, nid_b, "night mismatch for night B key");
                assert_eq!(
                    store.get(&nid_b).unwrap()[idx].key,
                    *key,
                    "reverse index points to wrong slot in night B"
                );
            }
        }

        // ── proptest ──────────────────────────────────────────────────────────────

        proptest! {
            /// For any non-empty sequence of valid MJD (TT) values in
            /// `[MJD_MIN, MJD_MAX]`, `sort_night` must:
            ///
            /// 1. Produce a slice sorted in ascending `epoch_mid` order.
            /// 2. Leave the reverse index fully coherent: every key resolves to
            ///    the slot that actually holds it.
            ///
            /// The range `[58 000, 62 000]` covers ZTF operations and early
            /// Rubin commissioning, and excludes non-finite values that have no
            /// physical meaning for MJD timestamps.
            #[test]
            fn sort_night_index_coherent_for_arbitrary_mjd(
                epoch_mids in prop::collection::vec(MJD_MIN..=MJD_MAX, 1..=64)
            ) {
                let mut store = SeedStore::new();
                let nid = night(0);

                let keys: Vec<SeedKey> = epoch_mids
                    .iter()
                    .map(|&mjd| store.insert_seed(nid, make_seed(nid, 0, mjd)))
                    .collect();

                store.sort_night(nid);

                let seeds = store.get(&nid).expect("night must exist after inserts");

                // 1. Sorted order.
                for w in seeds.windows(2) {
                    prop_assert!(
                        w[0].plane.epoch_mid <= w[1].plane.epoch_mid,
                        "epoch_mid out of order: {} > {}",
                        w[0].plane.epoch_mid,
                        w[1].plane.epoch_mid,
                    );
                }

                // 2. Full reverse-index coherence.
                for key in &keys {
                    let (resolved_night, idx) = store
                        .get_reverse_index(*key)
                        .ok_or_else(|| TestCaseError::fail("key missing from reverse index"))?;

                    prop_assert_eq!(resolved_night, nid);
                    prop_assert_eq!(store.get(&nid).unwrap()[idx].key, *key);
                }
            }
        }
    }
}

#[cfg(test)]
mod night_pairs_iter_tests {
    use super::*;
    use proptest::prelude::*;

    fn nid(v: u32) -> NightId {
        NightId(v)
    }

    fn make_store(nights: &[u32]) -> SeedStore {
        let mut map: AHashMap<NightId, Vec<SeedNode>> = AHashMap::new();
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
