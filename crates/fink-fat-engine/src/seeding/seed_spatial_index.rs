// src/seeding/seed_spatial_index.rs

//! Per-night spatial index for [`SeedNode`] built on top of generic buckets.
//!
//! This module provides a thin wrapper, [`SeedSpatialIndex`], around a
//! [`BucketIndex`] that is specialised for **single-night** operations:
//!
//! - all seeds are indexed using their tangent-plane mid-position
//!   `(plane.ra_mid, plane.dec_mid)`,
//! - the temporal dimension is collapsed to a single [`TimeBin`] value
//!   (`TimeBin(0)`) because all seeds are assumed to belong to the same night,
//! - queries are expressed as *approximate cone searches* on the sky using
//!   the [`SpatialBinner`] API.
//!
//! Unlike earlier versions that stored only identifiers, this index stores
//! **borrowed references** to seeds (`&SeedNode`). This allows callers to
//! retrieve candidate seeds directly without an additional `SeedId -> index`
//! lookup, at the cost of tying the index lifetime to the underlying seed slice.
//!
//! Typical usage
//! -------------
//! ```ignore
//! let index = SeedSpatialIndex::build(&seeds, &binner);
//!
//! // Collect borrowed candidate seeds (no extra indirection).
//! let candidates: Vec<&SeedNode> = index
//!     .cone_query(&binner, ra, dec, search_radius)
//!     .collect();
//! ```
//!
//! The returned candidates are **cell-level approximate** and are typically
//! passed to more accurate geometric filters (exact angular separation) or to
//! higher-level linkage/scoring logic.

use ahash::AHashMap;

use crate::{
    Radians,
    seeding::{seed_id::SeedId, seed_node::SeedNode},
    spacetime_bucket::{
        bucket::{Bucket, BucketIndex, BucketKey},
        spatial_binner::{SpatialBinner, SpatialKey},
        time_binner::TimeBin,
    },
};

/// Per-night spatial index for seeds.
///
/// This is a thin, domain-specific wrapper around [`BucketIndex<&SeedNode>`]
/// that:
///
/// - stores **borrowed references** to [`SeedNode`] (no cloning, no payload copy),
/// - groups seeds by their spatial cell,
/// - uses a single `TimeBin(0)` for the entire night.
///
/// # Lifetimes
///
/// The index borrows the seed slice passed to [`SeedSpatialIndex::build`].
/// Therefore, the underlying `seeds: &[SeedNode]` must outlive the index.
#[derive(Clone, Debug)]
pub struct SeedSpatialIndex<'a> {
    /// Underlying bucket index storing the mapping
    /// `(space_key, TimeBin(0)) → Vec<&SeedNode>`.
    inner: BucketIndex<&'a SeedNode>,
}

impl<'a> SeedSpatialIndex<'a> {
    /// Build a per-night spatial index from a slice of [`SeedNode`].
    ///
    /// Each seed is:
    ///
    /// 1. Mapped to a spatial cell via its tangent-plane mid-position
    ///    `(plane.ra_mid, plane.dec_mid)`.
    /// 2. Inserted into the bucket associated with that cell and the unique
    ///    nightly time bin `TimeBin(0)`.
    ///
    /// Arguments
    /// ---------
    /// * `seeds` – Collection of seeds to index (typically all seeds of one night).
    /// * `binner` – Spatial binner that maps sky coordinates to discrete cells.
    ///
    /// Return
    /// ------
    /// * `SeedSpatialIndex` – A new index ready for cone searches.
    ///
    /// Notes
    /// -----
    /// - This method does **not** deduplicate seeds. If the same `SeedNode`
    ///   reference (or effectively the same physical seed) appears multiple times
    ///   in `seeds`, it will appear multiple times in the corresponding bucket.
    /// - The temporal axis is deliberately collapsed. If you need time-resolved
    ///   indexing, build multiple indices (e.g. per time bin) or use the generic
    ///   [`BucketIndex`] directly with meaningful [`TimeBin`] values.
    pub fn build<Bs: SpatialBinner>(seeds: &'a [SeedNode], binner: &Bs) -> Self {
        // Buckets are keyed by (spatial cell, time bin). Here the time bin is
        // always `TimeBin(0)` because the index is scoped to a single night.
        let mut buckets: AHashMap<BucketKey, Bucket<&'a SeedNode>> = AHashMap::new();

        for s in seeds {
            let space_key = binner.key_for(s.plane.ra_mid, s.plane.dec_mid);

            let key = BucketKey {
                space_key,
                time_bin: TimeBin(0),
            };

            // Lazily create the bucket for this spatial cell, then push
            // the borrowed seed reference into its member list.
            buckets
                .entry(key)
                .or_insert_with(|| Bucket {
                    members: Vec::new(),
                })
                .members
                .push(s);
        }

        SeedSpatialIndex {
            inner: BucketIndex { buckets },
        }
    }

    /// Read-only access to the underlying bucket index.
    ///
    /// This can be useful when generic bucket operations are required or when
    /// debugging the internal layout of the index.
    pub fn inner(&self) -> &BucketIndex<&'a SeedNode> {
        &self.inner
    }

    /// Perform an approximate cone search in the per-night seed index.
    ///
    /// The query is evaluated by:
    ///
    /// 1. Projecting the cone centre `(ra, dec)` to a [`SpatialKey`].
    /// 2. Asking the [`SpatialBinner`] for all neighbour cells that intersect
    ///    the cone of angular radius `radius`.
    /// 3. Collecting all [`SeedNode`] references found in the buckets
    ///    `(space_key, TimeBin(0))` for those cells.
    ///
    /// Arguments
    /// ---------
    /// * `binner` – The same spatial binner that was used to build the index.
    /// * `ra` – Right ascension of the cone centre (radians).
    /// * `dec` – Declination of the cone centre (radians).
    /// * `radius` – Angular radius of the search cone (radians).
    ///
    /// Return
    /// ------
    /// * An iterator over borrowed [`SeedNode`] references in the spatial cells
    ///   that cover the requested cone.
    ///
    /// Notes
    /// -----
    /// - The search is **cell-level approximate**:
    ///   - some returned seeds might lie just outside the exact cone;
    ///   - seeds may be missed if the `SpatialBinner` only provides an
    ///     approximate cover.
    /// - Downstream code should apply a precise angular separation filter if
    ///   strict cone membership is required.
    pub fn cone_query<Bs: SpatialBinner>(
        &'a self,
        binner: &'a Bs,
        ra: Radians,
        dec: Radians,
        radius: Radians,
    ) -> impl Iterator<Item = &'a SeedNode> + 'a {
        let center_key: SpatialKey = binner.key_for(ra, dec);
        let cover_keys: Vec<SpatialKey> = binner.neighbors(center_key, radius);

        cover_keys
            .into_iter()
            .filter_map(|space_key| {
                // We always query the single nightly time bin `TimeBin(0)`.
                let key = BucketKey {
                    space_key,
                    time_bin: TimeBin(0),
                };
                self.inner.buckets.get(&key)
            })
            .flat_map(|bucket| bucket.members.iter().copied())
    }

    /// Low-allocation cone query that outputs `SeedId`s.
    ///
    /// This avoids returning `&SeedNode` references, which would otherwise
    /// tie the output lifetime to the temporary per-bin `SeedSpatialIndex`
    /// (and cause borrow-checker issues when reusing buffers across bins).
    #[inline]
    pub fn cone_query_ids_into<Bs: SpatialBinner>(
        &self,
        binner: &Bs,
        ra_center: Radians,
        dec_center: Radians,
        radius: Radians,
        cover_keys_buf: &mut Vec<SpatialKey>,
        out_ids: &mut Vec<SeedId>,
    ) {
        out_ids.clear();

        let center_key = binner.key_for(ra_center, dec_center);

        // Reuse caller buffer: no allocation here.
        binner.neighbors_into(center_key, radius, cover_keys_buf);

        let time_bin = TimeBin(0);

        for &space_key in cover_keys_buf.iter() {
            let bucket_key = BucketKey {
                space_key,
                time_bin,
            };

            if let Some(bucket) = self.inner.buckets.get(&bucket_key) {
                // bucket.members: Vec<&'a SeedNode>
                out_ids.extend(bucket.members.iter().map(|sn| sn.seed_id));
            }
        }
    }
}

#[cfg(test)]
mod seed_spatial_index_tests {
    use super::*;
    use proptest::prelude::*;

    use std::ptr;

    use crate::{
        astro_math::arcsec_to_rad,
        night_id::NightId,
        seeding::seed_id::SeedId,
        seeding::seed_node::SeedNode,
        seeding::tangent_plane::{TangentCenter, TangentPlaneModel},
        spacetime_bucket::{healpix_binner::HealpixBinner, time_binner::TimeBin},
    };

    /* ------------------------- helpers ------------------------- */

    // Build a minimal SeedNode with given (ra_mid, dec_mid). Plane fields are simple constants.
    fn mk_seed(seed_id: u64, ra_mid: f64, dec_mid: f64) -> SeedNode {
        let center = TangentCenter::new(ra_mid, dec_mid);
        let plane = TangentPlaneModel::new(
            center,
            60000.0,                  // epoch_mid
            [0.0, 0.0],               // pos_xy
            [0.0, 0.0],               // vel_xy
            None,                     // acc_xy
            [[0.0, 0.0], [0.0, 0.0]], // cov_pos
            [[0.0, 0.0], [0.0, 0.0]], // cov_vel
            ra_mid,
            dec_mid,
        );
        SeedNode {
            seed_id: SeedId::new(seed_id),
            night_id: NightId::new(1),
            plane,
            photom: crate::seeding::photometry::Photometry::from_pair(1.0, 0.1, 1, 2),
            n_obs: 2,
            members: vec![],
        }
    }

    /// True if `items` contains the exact borrowed reference `needle`.
    fn contains_ref<'a>(items: &[&'a SeedNode], needle: &'a SeedNode) -> bool {
        items.iter().any(|x| ptr::eq(*x, needle))
    }

    /// True if `bucket.members` contains the exact borrowed reference `needle`.
    fn bucket_contains_ref<'a>(
        bucket: &crate::spacetime_bucket::bucket::Bucket<&'a SeedNode>,
        needle: &'a SeedNode,
    ) -> bool {
        bucket.members.iter().any(|x| ptr::eq(*x, needle))
    }

    /* ------------------------- unit tests ------------------------- */

    #[test]
    fn build_creates_buckets_with_timebin_zero() {
        let binner = HealpixBinner::new(8);

        // Two seeds in the same cell, one in a neighboring cell.
        let s1 = mk_seed(0, 1.0, 0.2);
        let s2 = mk_seed(1, 1.0 + arcsec_to_rad(1.0) / 0.2f64.cos(), 0.2);
        let s3 = mk_seed(2, 2.0, -0.1);

        // IMPORTANT: keep seeds in a stable Vec so the index can borrow them.
        let seeds = vec![s1, s2, s3];

        let index = SeedSpatialIndex::build(&seeds, &binner);
        let inner = index.inner();

        // Compute bucket keys (all must use TimeBin(0)).
        let key1 = BucketKey {
            space_key: binner.key_for(seeds[0].plane.ra_mid, seeds[0].plane.dec_mid),
            time_bin: TimeBin(0),
        };
        let key2 = BucketKey {
            space_key: binner.key_for(seeds[1].plane.ra_mid, seeds[1].plane.dec_mid),
            time_bin: TimeBin(0),
        };
        let key3 = BucketKey {
            space_key: binner.key_for(seeds[2].plane.ra_mid, seeds[2].plane.dec_mid),
            time_bin: TimeBin(0),
        };

        let b1 = inner.buckets.get(&key1).expect("bucket for s1");
        assert!(bucket_contains_ref(b1, &seeds[0]));

        let b2 = inner.buckets.get(&key2).expect("bucket for s2");
        assert!(bucket_contains_ref(b2, &seeds[1]));

        let b3 = inner.buckets.get(&key3).expect("bucket for s3");
        assert!(bucket_contains_ref(b3, &seeds[2]));

        // s1 and s2 might be same cell depending on NSIDE; if so, both members appear together.
        if key1.space_key == key2.space_key {
            let b = inner.buckets.get(&key1).unwrap();
            assert!(bucket_contains_ref(b, &seeds[0]) && bucket_contains_ref(b, &seeds[1]));
        }
    }

    #[test]
    fn cone_query_returns_seeds_covering_cone() {
        let binner = HealpixBinner::new(8);

        // Place seeds close to each other; small radius should cover both if same cell,
        // otherwise a slightly larger radius should get neighbor cells too.
        let ra = 1.5;
        let dec = 0.3;

        let s_primary = mk_seed(10, ra, dec);
        let s_neighbor = mk_seed(11, ra + arcsec_to_rad(30.0) / dec.cos(), dec);

        let seeds = vec![s_primary, s_neighbor];
        let index = SeedSpatialIndex::build(&seeds, &binner);

        // Query with radius including at least cell_radius to ensure coverage of the containing cell.
        let radius = binner.cell_radius().max(arcsec_to_rad(45.0)); // 45" cone
        let found: Vec<&SeedNode> = index.cone_query(&binner, ra, dec, radius).collect();

        // At least the primary seed must be found (by reference identity).
        assert!(contains_ref(&found, &seeds[0]));

        // Depending on binner neighbors, the neighbor seed may also be found.
        // If not, increase radius and check again.
        if !contains_ref(&found, &seeds[1]) {
            let radius2 = radius * 2.0;
            let found2: Vec<&SeedNode> = index.cone_query(&binner, ra, dec, radius2).collect();
            assert!(contains_ref(&found2, &seeds[1]));
        }
    }

    #[test]
    fn inner_exposes_consistent_mapping() {
        let binner = HealpixBinner::new(7);
        let s1 = mk_seed(0, 0.5, 0.1);
        let s2 = mk_seed(1, 0.501, 0.101);

        let seeds = vec![s1, s2];
        let index = SeedSpatialIndex::build(&seeds, &binner);
        let inner = index.inner();

        // Compute keys and verify buckets exist with matching members (by reference identity).
        for s in [&seeds[0], &seeds[1]] {
            let key = BucketKey {
                space_key: binner.key_for(s.plane.ra_mid, s.plane.dec_mid),
                time_bin: TimeBin(0),
            };
            let bucket = inner.buckets.get(&key).expect("bucket must exist");
            assert!(bucket_contains_ref(bucket, s));
        }
    }

    /* ------------------------- property-based tests ------------------------- */

    fn ra_strategy() -> impl Strategy<Value = f64> {
        0.0f64..(2.0 * std::f64::consts::PI)
    }
    fn dec_strategy() -> impl Strategy<Value = f64> {
        (-(std::f64::consts::PI / 2.0 - 1e-6))..(std::f64::consts::PI / 2.0 - 1e-6)
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 32,
            .. ProptestConfig::default()
        })]

        /// For random seed positions, querying a cone centered on a seed with
        /// radius ≥ cell_radius must return that seed (by reference identity).
        #[test]
        fn prop_cone_query_includes_center_seed(
            pts in proptest::collection::vec((ra_strategy(), dec_strategy()), 1..50)
        ) {
            let binner = HealpixBinner::new(8);

            // Keep seeds in a Vec so the index can borrow them.
            let seeds: Vec<SeedNode> = pts.iter().enumerate().map(|(i, (ra, dec))| {
                mk_seed(i as u64, *ra, *dec)
            }).collect();

            let index = SeedSpatialIndex::build(&seeds, &binner);

            // For each seed, query cone at its mid-position and ensure it is included.
            for s in &seeds {
                let radius = binner.cell_radius().max(arcsec_to_rad(1.0));
                let found: Vec<&SeedNode> = index.cone_query(&binner, s.plane.ra_mid, s.plane.dec_mid, radius).collect();
                prop_assert!(contains_ref(&found, s));
            }
        }

        /// Buckets should only be associated with TimeBin(0) in this index.
        #[test]
        fn prop_all_buckets_use_timebin_zero(
            pts in proptest::collection::vec((ra_strategy(), dec_strategy()), 0..40)
        ) {
            let binner = HealpixBinner::new(7);

            let seeds: Vec<SeedNode> = pts.iter().enumerate().map(|(i, (ra, dec))| {
                mk_seed(i as u64, *ra, *dec)
            }).collect();

            let index = SeedSpatialIndex::build(&seeds, &binner);

            for (key, _) in &index.inner().buckets {
                prop_assert_eq!(key.time_bin, TimeBin(0));
            }
        }
    }

    mod cone_query_id_into_test {
        use super::super::*;
        use super::*;

        /* ------------------------- tests: cone_query_ids_into ------------------------- */

        #[test]
        fn cone_query_ids_into_includes_center_seed_id() {
            let binner = HealpixBinner::new(8);

            let ra = 1.2;
            let dec = -0.4;

            let s0 = mk_seed(123, ra, dec);
            let s1 = mk_seed(124, ra + arcsec_to_rad(10.0) / dec.cos(), dec);

            let seeds = vec![s0, s1];
            let index = SeedSpatialIndex::build(&seeds, &binner);

            let mut cover = Vec::<SpatialKey>::new();
            let mut out = Vec::<SeedId>::new();

            let radius = binner.cell_radius().max(arcsec_to_rad(1.0));

            index.cone_query_ids_into(&binner, ra, dec, radius, &mut cover, &mut out);

            assert!(out.contains(&seeds[0].seed_id));
        }

        #[test]
        fn cone_query_ids_into_clears_output_and_does_not_shrink_buffers() {
            let binner = HealpixBinner::new(8);

            let ra = 0.7;
            let dec = 0.2;

            let seeds = vec![mk_seed(1, ra, dec), mk_seed(2, ra, dec)];
            let index = SeedSpatialIndex::build(&seeds, &binner);

            let mut cover = Vec::<SpatialKey>::with_capacity(64);
            let mut out = Vec::<SeedId>::with_capacity(64);

            // Pre-fill buffers to ensure function clears `out_ids`.
            cover.extend(std::iter::repeat(binner.key_for(ra, dec)).take(10));
            out.extend([SeedId::new(999), SeedId::new(1000)]);

            let cap_cover_before = cover.capacity();
            let cap_out_before = out.capacity();

            let radius = binner.cell_radius().max(arcsec_to_rad(1.0));
            index.cone_query_ids_into(&binner, ra, dec, radius, &mut cover, &mut out);

            // Must have cleared the output (no stale IDs).
            assert!(!out.contains(&SeedId::new(999)));
            assert!(!out.contains(&SeedId::new(1000)));

            // “Low-allocation”: should not shrink caller buffers.
            assert_eq!(cover.capacity(), cap_cover_before);
            assert_eq!(out.capacity(), cap_out_before);
        }

        #[test]
        fn cone_query_ids_into_matches_cone_query_as_set() {
            let binner = HealpixBinner::new(8);

            let ra = 2.1;
            let dec = 0.35;

            let s0 = mk_seed(10, ra, dec);
            let s1 = mk_seed(11, ra + arcsec_to_rad(40.0) / dec.cos(), dec);
            let s2 = mk_seed(12, ra - arcsec_to_rad(40.0) / dec.cos(), dec);

            let seeds = vec![s0, s1, s2];
            let index = SeedSpatialIndex::build(&seeds, &binner);

            let radius = binner.cell_radius().max(arcsec_to_rad(90.0));

            let found_refs: Vec<&SeedNode> = index.cone_query(&binner, ra, dec, radius).collect();
            let mut ids_from_refs: Vec<SeedId> = found_refs.iter().map(|s| s.seed_id).collect();
            ids_from_refs.sort_unstable();
            ids_from_refs.dedup();

            let mut cover = Vec::<SpatialKey>::new();
            let mut out = Vec::<SeedId>::new();

            index.cone_query_ids_into(&binner, ra, dec, radius, &mut cover, &mut out);

            let mut out_dedup = out.clone();
            out_dedup.sort_unstable();
            out_dedup.dedup();

            assert_eq!(out_dedup, ids_from_refs);
        }

        #[test]
        fn cone_query_ids_into_keeps_duplicates_when_seeds_are_duplicated_in_input() {
            let binner = HealpixBinner::new(8);

            let ra = 1.0;
            let dec = 0.1;

            // Same seed_id, same coordinates, duplicated twice in the seed slice.
            let s = mk_seed(42, ra, dec);
            let seeds = vec![s.clone(), s];

            let index = SeedSpatialIndex::build(&seeds, &binner);

            let radius = binner.cell_radius().max(arcsec_to_rad(1.0));

            let mut cover = Vec::<SpatialKey>::new();
            let mut out = Vec::<SeedId>::new();

            index.cone_query_ids_into(&binner, ra, dec, radius, &mut cover, &mut out);

            let count = out.iter().filter(|&&id| id == SeedId::new(42)).count();
            assert_eq!(count, 2, "expected duplicate SeedId to appear twice");
        }

        proptest! {
            #![proptest_config(ProptestConfig {
                cases: 32,
                .. ProptestConfig::default()
            })]

            /// For random seed positions, querying a cone centered on a seed with
            /// radius ≥ cell_radius must return that seed_id.
            #[test]
            fn prop_cone_query_ids_into_includes_center_seed_id(
                pts in proptest::collection::vec((ra_strategy(), dec_strategy()), 1..50)
            ) {
                let binner = HealpixBinner::new(8);

                let seeds: Vec<SeedNode> = pts.iter().enumerate().map(|(i, (ra, dec))| {
                    mk_seed(i as u64, *ra, *dec)
                }).collect();

                let index = SeedSpatialIndex::build(&seeds, &binner);

                let mut cover = Vec::<SpatialKey>::new();
                let mut out = Vec::<SeedId>::new();

                let radius = binner.cell_radius().max(arcsec_to_rad(1.0));

                for s in &seeds {
                    index.cone_query_ids_into(
                        &binner,
                        s.plane.ra_mid,
                        s.plane.dec_mid,
                        radius,
                        &mut cover,
                        &mut out
                    );

                    prop_assert!(out.contains(&s.seed_id));
                }
            }

            /// cone_query_ids_into should produce the same candidate-id set as cone_query.
            #[test]
            fn prop_cone_query_ids_into_matches_cone_query_as_set(
                pts in proptest::collection::vec((ra_strategy(), dec_strategy()), 1..80),
                q in (ra_strategy(), dec_strategy())
            ) {
                let binner = HealpixBinner::new(8);

                let seeds: Vec<SeedNode> = pts.iter().enumerate().map(|(i, (ra, dec))| {
                    mk_seed(i as u64, *ra, *dec)
                }).collect();

                let index = SeedSpatialIndex::build(&seeds, &binner);

                let radius = binner.cell_radius().max(arcsec_to_rad(1.0));

                let found_refs: Vec<&SeedNode> = index.cone_query(&binner, q.0, q.1, radius).collect();
                let mut ids_from_refs: Vec<SeedId> = found_refs.iter().map(|s| s.seed_id).collect();
                ids_from_refs.sort_unstable();
                ids_from_refs.dedup();

                let mut cover = Vec::<SpatialKey>::new();
                let mut out = Vec::<SeedId>::new();

                index.cone_query_ids_into(&binner, q.0, q.1, radius, &mut cover, &mut out);

                out.sort_unstable();
                out.dedup();

                prop_assert_eq!(out, ids_from_refs);
            }
        }
    }
}
