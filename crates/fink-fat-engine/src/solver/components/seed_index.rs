use ahash::AHashMap;

use crate::{
    night_id::NightId,
    seeding::{SeedKey, store::SeedStore},
    solver::components::error::{ComponentError, SeedOrigin},
};

/// Dense identifier used for seeds in global index space.
pub type DenseSeedId = u32;

/// Map `SeedKey` -> dense global index `[0..N_total)`.
///
/// This implementation leverages the internal reverse index of
/// `SeedStore` (`id_to_location`) to retrieve the intra-night
/// position of each seed.
///
/// The dense index is computed as:
///
/// ```text
/// global_idx = base[night_id] + index_in_vec
/// ```
///
/// Unlike the previous design, `SeedKey.unique_id` is no longer
/// assumed to be dense nor structured per-night.
#[derive(Debug, Clone)]
pub struct SeedGlobalIndex {
    /// Base offset per night.
    base: AHashMap<NightId, DenseSeedId>,

    /// Total number of seeds.
    n_total: u32,
}

impl SeedGlobalIndex {
    /// Build global dense indexing using per-night contiguous layout.
    ///
    /// Nights are processed in sorted deterministic order.
    /// Within each night, the natural `Vec` order is used.
    ///
    /// Notes
    /// -----
    /// - The mapping is **not stable** across calls: rebuilding after inserting
    ///   new nights may reassign all indices. Callers must rebuild and not cache
    ///   individual indices across store mutations.
    ///
    /// Complexity
    /// ----------
    /// $O(K)$ where $K$ is the number of nights.
    pub fn build(seed_store: &SeedStore) -> Result<Self, ComponentError> {
        let mut base = AHashMap::default();
        let mut cursor: u32 = 0;

        for night_id in seed_store.nights() {
            base.insert(*night_id, cursor);

            let len = seed_store
                .len_night(night_id)
                .ok_or(ComponentError::NightNotFound(*night_id))?;
            cursor += len as u32;
        }

        Ok(Self {
            base,
            n_total: cursor,
        })
    }

    /// Total number of indexed seeds.
    #[inline]
    pub fn n_total(&self) -> usize {
        self.n_total as usize
    }

    /// Return dense index for a seed key.
    ///
    /// Arguments
    /// ---------
    /// * `seed_store` – Seed storage to query for reverse indexing.
    /// * `key` – Seed key to index.
    ///
    /// Return
    /// ------
    /// * `Ok(usize)` – Dense index of the seed key.
    /// * `Err(ComponentError::SeedKeyNotFound)` – If the seed key is not found in the seed store's reverse index.
    #[inline]
    pub fn idx_of_key(
        &self,
        seed_store: &SeedStore,
        key: SeedKey,
    ) -> Result<usize, ComponentError> {
        let (night_id, index_in_vec) =
            seed_store
                .get_reverse_index(key)
                .ok_or(ComponentError::SeedKeyNotFound {
                    key,
                    origin: SeedOrigin::Index,
                })?;

        Ok((self.base[&night_id] + (index_in_vec as u32)) as usize)
    }
}

#[cfg(test)]
mod seed_global_index_tests {
    use super::*;
    use proptest::prelude::*;

    use crate::{
        Alert, AlertKey,
        astro_math::arcsec_to_rad,
        night_id::NightId,
        seeding::{SeedKey, SeedNode, store::SeedStore},
        solver::components::error::ComponentError,
    };

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    fn nid(v: u32) -> NightId {
        NightId::from(v)
    }

    /// Build a minimal `Alert` with a unique `dia_source_id`.
    /// Position is fixed at (ra=1.0, dec=0.1) rad; only timing varies.
    fn mk_alert(source_id: u64, night_id: NightId, mjd_tt: f64) -> Alert {
        Alert {
            key: AlertKey {
                night_id,
                dia_source_id: source_id,
            },
            ra: 1.0,
            ra_err: arcsec_to_rad(0.5),
            dec: 0.1,
            dec_err: arcsec_to_rad(0.5),
            mjd_tt,
            flux: 1000.0,
            flux_err: 10.0,
            band: 1,
        }
    }

    /// Insert `count` seeds built from pairs of consecutive alerts into `store`
    /// for `night_id`. Returns the `SeedKey`s in insertion order.
    ///
    /// Each pair is built from alerts separated by 30 minutes (~0.02 days),
    /// which is well within typical intra-night constraints.
    fn insert_seeds(
        store: &mut SeedStore,
        night_id: NightId,
        count: usize,
        source_id_offset: u64,
    ) -> Vec<SeedKey> {
        let mut keys = Vec::with_capacity(count);
        let t0 = 60000.0;

        for i in 0..count {
            let sid_a = source_id_offset + (2 * i) as u64;
            let sid_b = source_id_offset + (2 * i + 1) as u64;

            // 30-minute separation between the two alerts of each pair.
            let dt = 30.0 / 1440.0;
            let alert_a = mk_alert(sid_a, night_id, t0 + i as f64);
            let alert_b = mk_alert(sid_b, night_id, t0 + i as f64 + dt);

            if let Some(seed) = SeedNode::from_pair(store, night_id, &alert_a, &alert_b, None) {
                keys.push(seed.key());
                store.insert_vec_seed(night_id, vec![seed]);
            }
        }
        keys
    }

    /// Build a store from a list of `(night_id_u32, seed_count)` pairs.
    /// `source_id` offsets are managed to avoid collisions across nights.
    fn build_store(spec: &[(u32, usize)]) -> (SeedStore, Vec<(NightId, Vec<SeedKey>)>) {
        let mut store = SeedStore::new();
        let mut record = Vec::new();
        let mut source_id_offset: u64 = 0;

        for (n, count) in spec {
            let night = nid(*n);
            // Each pair uses 2 source ids, so the offset grows by 2 * count.
            let keys = insert_seeds(&mut store, night, *count, source_id_offset);
            source_id_offset += (2 * count) as u64;
            record.push((night, keys));
        }
        (store, record)
    }

    // -------------------------------------------------------------------------
    // Unit tests
    // -------------------------------------------------------------------------

    #[test]
    fn empty_store_gives_zero_total() {
        let store = SeedStore::new();
        let idx = SeedGlobalIndex::build(&store).unwrap();
        assert_eq!(idx.n_total(), 0);
    }

    #[test]
    fn single_night_total_matches_count() {
        let (store, record) = build_store(&[(0, 5)]);
        let idx = SeedGlobalIndex::build(&store).unwrap();
        // n_total must match how many seeds were actually built and inserted.
        let actual_count = record[0].1.len();
        assert_eq!(idx.n_total(), actual_count);
    }

    #[test]
    fn multiple_nights_total_is_sum() {
        let (store, record) = build_store(&[(0, 3), (1, 4), (2, 2)]);
        let idx = SeedGlobalIndex::build(&store).unwrap();
        let expected: usize = record.iter().map(|(_, k)| k.len()).sum();
        assert_eq!(idx.n_total(), expected);
    }

    #[test]
    fn single_seed_has_index_zero() {
        let (store, record) = build_store(&[(0, 1)]);
        let idx = SeedGlobalIndex::build(&store).unwrap();
        let key = record[0].1[0];
        assert_eq!(idx.idx_of_key(&store, key).unwrap(), 0);
    }

    #[test]
    fn indices_within_single_night_are_contiguous_from_zero() {
        let (store, record) = build_store(&[(0, 5)]);
        let idx = SeedGlobalIndex::build(&store).unwrap();

        let mut dense_ids: Vec<usize> = record[0]
            .1
            .iter()
            .map(|k| idx.idx_of_key(&store, *k).unwrap())
            .collect();
        dense_ids.sort_unstable();

        let n = dense_ids.len();
        let expected: Vec<usize> = (0..n).collect();
        assert_eq!(dense_ids, expected);
    }

    #[test]
    fn all_indices_are_unique_across_nights() {
        let (store, record) = build_store(&[(0, 3), (1, 4), (2, 2)]);
        let idx = SeedGlobalIndex::build(&store).unwrap();

        let mut all_ids: Vec<usize> = record
            .iter()
            .flat_map(|(_, keys)| keys.iter().map(|k| idx.idx_of_key(&store, *k).unwrap()))
            .collect();

        let total = all_ids.len();
        all_ids.sort_unstable();
        all_ids.dedup();

        assert_eq!(
            all_ids.len(),
            total,
            "duplicate dense indices detected across nights"
        );
    }

    #[test]
    fn all_indices_cover_range_zero_to_n_total() {
        let (store, record) = build_store(&[(10, 2), (20, 3), (30, 4)]);
        let idx = SeedGlobalIndex::build(&store).unwrap();

        let mut all_ids: Vec<usize> = record
            .iter()
            .flat_map(|(_, keys)| keys.iter().map(|k| idx.idx_of_key(&store, *k).unwrap()))
            .collect();
        all_ids.sort_unstable();

        let expected: Vec<usize> = (0..idx.n_total()).collect();
        assert_eq!(
            all_ids, expected,
            "dense indices must cover [0, n_total) exactly"
        );
    }

    #[test]
    fn unknown_key_returns_seed_key_not_found_error() {
        let (store, _) = build_store(&[(0, 3)]);
        let idx = SeedGlobalIndex::build(&store).unwrap();

        // Craft a SeedKey that was never inserted via from_pair.
        // NightId 99 does not exist in the store.
        let ghost_key = SeedKey {
            night_id: nid(99),
            unique_id: Default::default(),
        };

        let result = idx.idx_of_key(&store, ghost_key);
        assert!(
            matches!(result, Err(ComponentError::SeedKeyNotFound { .. })),
            "expected SeedKeyNotFound, got {result:?}"
        );
    }

    #[test]
    fn n_total_grows_monotonically_as_nights_are_added() {
        let mut store = SeedStore::new();
        let mut prev_total = 0usize;
        let mut source_id_offset: u64 = 0;

        for (i, count) in [3usize, 5, 1, 4, 2].iter().enumerate() {
            insert_seeds(&mut store, nid(i as u32), *count, source_id_offset);
            source_id_offset += (2 * count) as u64;

            let idx = SeedGlobalIndex::build(&store).unwrap();
            assert!(
                idx.n_total() >= prev_total,
                "n_total decreased: {} -> {}",
                prev_total,
                idx.n_total()
            );
            prev_total = idx.n_total();
        }
    }

    // -------------------------------------------------------------------------
    // Property-based tests
    // -------------------------------------------------------------------------

    proptest! {
        /// For any store layout, `n_total` must equal the sum of per-night seed counts.
        #[test]
        fn prop_n_total_equals_sum_of_counts(
            // Generate unique night ids in [0, 100) with counts in [1, 8).
            raw_spec in prop::collection::vec((0u32..100u32, 1usize..8usize), 0..12)
        ) {
            // Deduplicate night ids (keep first occurrence).
            let mut seen = std::collections::HashSet::new();
            let spec: Vec<(u32, usize)> = raw_spec
                .into_iter()
                .filter(|(n, _)| seen.insert(*n))
                .collect();

            let (store, record) = build_store(&spec);
            let idx = SeedGlobalIndex::build(&store).unwrap();

            let expected: usize = record.iter().map(|(_, k)| k.len()).sum();
            prop_assert_eq!(idx.n_total(), expected);
        }

        /// Every inserted key must resolve to a valid dense index without error.
        #[test]
        fn prop_all_inserted_keys_resolve(
            raw_spec in prop::collection::vec((0u32..50u32, 1usize..6usize), 1..10)
        ) {
            let mut seen = std::collections::HashSet::new();
            let spec: Vec<(u32, usize)> = raw_spec
                .into_iter()
                .filter(|(n, _)| seen.insert(*n))
                .collect();

            let (store, record) = build_store(&spec);
            let idx = SeedGlobalIndex::build(&store).unwrap();

            for (_, keys) in &record {
                for key in keys {
                    prop_assert!(
                        idx.idx_of_key(&store, *key).is_ok(),
                        "key {:?} failed to resolve",
                        key
                    );
                }
            }
        }

        /// Dense indices must be globally unique.
        #[test]
        fn prop_dense_indices_are_unique(
            raw_spec in prop::collection::vec((0u32..50u32, 1usize..6usize), 1..10)
        ) {
            let mut seen = std::collections::HashSet::new();
            let spec: Vec<(u32, usize)> = raw_spec
                .into_iter()
                .filter(|(n, _)| seen.insert(*n))
                .collect();

            let (store, record) = build_store(&spec);
            let idx = SeedGlobalIndex::build(&store).unwrap();

            let mut ids: Vec<usize> = record
                .iter()
                .flat_map(|(_, keys)| {
                    keys.iter().map(|k| idx.idx_of_key(&store, *k).unwrap())
                })
                .collect();

            let total = ids.len();
            ids.sort_unstable();
            ids.dedup();
            prop_assert_eq!(ids.len(), total, "duplicate dense indices found");
        }

        /// Dense indices must cover exactly $[0, \text{n\_total})$.
        #[test]
        fn prop_dense_indices_cover_full_range(
            raw_spec in prop::collection::vec((0u32..50u32, 1usize..6usize), 1..10)
        ) {
            let mut seen = std::collections::HashSet::new();
            let spec: Vec<(u32, usize)> = raw_spec
                .into_iter()
                .filter(|(n, _)| seen.insert(*n))
                .collect();

            let (store, record) = build_store(&spec);
            let idx = SeedGlobalIndex::build(&store).unwrap();

            let mut ids: Vec<usize> = record
                .iter()
                .flat_map(|(_, keys)| {
                    keys.iter().map(|k| idx.idx_of_key(&store, *k).unwrap())
                })
                .collect();
            ids.sort_unstable();

            let expected: Vec<usize> = (0..idx.n_total()).collect();
            prop_assert_eq!(ids, expected, "indices do not cover [0, n_total) exactly");
        }
    }
}
