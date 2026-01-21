// src/seeding/seed_spatial_index.rs

//! Per-night spatial index for `SeedNode`s built on top of generic buckets.
//!
//! This module provides a thin wrapper, [`SeedSpatialIndex`], around
//! [`BucketIndex<SeedId>`] that is specialised for **intra-night** operations:
//!
//! - all seeds are indexed using their tangent-plane mid-position
//!   `(plane.ra_mid, plane.dec_mid)`,
//! - the temporal dimension is collapsed to a single [`TimeBin`] value
//!   (`TimeBin(0)`) because all seeds are assumed to belong to the same night,
//! - queries are expressed as *approximate cone searches* on the sky using
//!   the [`SpatialBinner`] API.
//!
//! Typical usage
//! -------------
//! ```ignore
//! let index = SeedSpatialIndex::build(&seeds, &binner);
//!
//! let candidates: Vec<SeedId> = index
//!     .cone_query(&binner, ra, dec, search_radius)
//!     .collect();
//! ```
//!
//! The resulting `SeedId`s can then be passed to more accurate geometric
//! filters (e.g. exact angular separation) or to higher-level linkage logic.

use std::collections::HashMap;

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
/// This is a thin, domain-specific wrapper around [`BucketIndex<SeedId>`]
/// that:
///
/// - stores only `SeedId` values (no full `SeedNode` payload),
/// - groups seeds by their spatial cell,
/// - uses a single `TimeBin(0)` for the entire night.
#[derive(Clone, Debug, Default)]
pub struct SeedSpatialIndex {
    /// Underlying bucket index storing the mapping
    /// `(space_key, TimeBin(0)) → Vec<SeedId>`.
    inner: BucketIndex<SeedId>,
}

impl SeedSpatialIndex {
    /// Build a per-night spatial index from a slice of `SeedNode`.
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
    /// - This method does **not** deduplicate seeds: if the same `SeedId`
    ///   occurs multiple times in `seeds`, it will appear multiple times in
    ///   the corresponding bucket.
    /// - The temporal axis is deliberately collapsed: if you need multi-night
    ///   or time-resolved indexing, use the generic [`BucketIndex`] directly
    ///   or build a higher-level structure.
    pub fn build<Bs: SpatialBinner>(seeds: &[SeedNode], binner: &Bs) -> Self {
        // Buckets are keyed by (spatial cell, time bin). Here the time bin is
        // always `TimeBin(0)` because the index is scoped to a single night.
        let mut buckets: HashMap<BucketKey, Bucket<SeedId>> = HashMap::new();

        for s in seeds {
            let space_key = binner.key_for(s.plane.ra_mid, s.plane.dec_mid);

            let key = BucketKey {
                space_key,
                time_bin: TimeBin(0),
            };

            // Lazily create the bucket for this spatial cell, then push
            // the seed identifier into its member list.
            buckets
                .entry(key)
                .or_insert_with(|| Bucket {
                    members: Vec::new(),
                })
                .members
                .push(s.seed_id);
        }

        SeedSpatialIndex {
            inner: BucketIndex { buckets },
        }
    }

    /// Read-only access to the underlying bucket index.
    ///
    /// This can be useful when generic bucket operations are required or when
    /// debugging the internal layout of the index.
    pub fn inner(&self) -> &BucketIndex<SeedId> {
        &self.inner
    }

    /// Perform an approximate cone search in the seed index.
    ///
    /// The query is evaluated by:
    ///
    /// 1. Projecting the cone centre `(ra, dec)` to a [`SpatialKey`].
    /// 2. Asking the [`SpatialBinner`] for all neighbour cells that intersect
    ///    the cone of angular radius `radius`.
    /// 3. Collecting all `SeedId`s found in the buckets
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
    /// * An iterator over all `SeedId`s in the spatial cells that cover
    ///   the requested cone.
    ///
    /// Notes
    /// -----
    /// - The search is **cell-level approximate**:
    ///   - some returned seeds might lie just outside the exact cone;
    ///   - seeds may be missed if the `SpatialBinner` only provides an
    ///     approximate cover.
    /// - Downstream code should apply a precise angular separation filter if
    ///   strict cone membership is required.
    pub fn cone_query<'a, Bs: SpatialBinner>(
        &'a self,
        binner: &'a Bs,
        ra: Radians,
        dec: Radians,
        radius: Radians,
    ) -> impl Iterator<Item = SeedId> + 'a {
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
}

#[cfg(test)]
mod seed_spatial_index_tests {
    use super::*;
    use proptest::prelude::*;

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

    /* ------------------------- unit tests ------------------------- */

    #[test]
    fn build_creates_buckets_with_timebin_zero() {
        let binner = HealpixBinner::new(8);

        // Two seeds in the same cell, one in a neighboring cell.
        let s1 = mk_seed(0, 1.0, 0.2);
        let s2 = mk_seed(1, 1.0 + arcsec_to_rad(1.0) / 0.2f64.cos(), 0.2);
        let s3 = mk_seed(2, 2.0, -0.1);

        let index = SeedSpatialIndex::build(&[s1.clone(), s2.clone(), s3.clone()], &binner);
        let inner = index.inner();

        // Check that all buckets use TimeBin(0) and contain the expected SeedId members.
        let key1 = BucketKey {
            space_key: binner.key_for(s1.plane.ra_mid, s1.plane.dec_mid),
            time_bin: TimeBin(0),
        };
        let key2 = BucketKey {
            space_key: binner.key_for(s2.plane.ra_mid, s2.plane.dec_mid),
            time_bin: TimeBin(0),
        };
        let key3 = BucketKey {
            space_key: binner.key_for(s3.plane.ra_mid, s3.plane.dec_mid),
            time_bin: TimeBin(0),
        };

        let b1 = inner.buckets.get(&key1).expect("bucket for s1");
        assert!(b1.members.contains(&s1.seed_id));

        let b2 = inner.buckets.get(&key2).expect("bucket for s2");
        assert!(b2.members.contains(&s2.seed_id));

        let b3 = inner.buckets.get(&key3).expect("bucket for s3");
        assert!(b3.members.contains(&s3.seed_id));

        // s1 and s2 might be same cell depending on NSIDE; if so, both members appear together.
        if key1.space_key == key2.space_key {
            let b = inner.buckets.get(&key1).unwrap();
            assert!(b.members.contains(&s1.seed_id) && b.members.contains(&s2.seed_id));
        }
    }

    #[test]
    fn cone_query_returns_ids_covering_cone() {
        let binner = HealpixBinner::new(8);

        // Place seeds close to each other; small radius should cover both if same cell,
        // otherwise a slightly larger radius should get neighbor cells too.
        let ra = 1.5;
        let dec = 0.3;
        let s_primary = mk_seed(10, ra, dec);
        let s_neighbor = mk_seed(11, ra + arcsec_to_rad(30.0) / dec.cos(), dec);

        let index = SeedSpatialIndex::build(&[s_primary.clone(), s_neighbor.clone()], &binner);

        // Query with radius including at least cell_radius to ensure coverage of the containing cell.
        let radius = binner.cell_radius().max(arcsec_to_rad(45.0)); // 45" cone
        let found: Vec<SeedId> = index.cone_query(&binner, ra, dec, radius).collect();

        // At least the primary seed must be found.
        assert!(found.contains(&s_primary.seed_id));

        // Depending on binner neighbors, the neighbor seed may also be found.
        // If not, increase radius and check again.
        if !found.contains(&s_neighbor.seed_id) {
            let radius2 = radius * 2.0;
            let found2: Vec<SeedId> = index.cone_query(&binner, ra, dec, radius2).collect();
            assert!(found2.contains(&s_neighbor.seed_id));
        }
    }

    #[test]
    fn inner_exposes_consistent_mapping() {
        let binner = HealpixBinner::new(7);
        let s1 = mk_seed(0, 0.5, 0.1);
        let s2 = mk_seed(1, 0.501, 0.101);

        let index = SeedSpatialIndex::build(&[s1.clone(), s2.clone()], &binner);
        let inner = index.inner();

        // Compute keys and verify buckets exist with matching members.
        for s in [&s1, &s2] {
            let key = BucketKey {
                space_key: binner.key_for(s.plane.ra_mid, s.plane.dec_mid),
                time_bin: TimeBin(0),
            };
            let bucket = inner.buckets.get(&key).expect("bucket must exist");
            assert!(bucket.members.contains(&s.seed_id));
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
        /// radius ≥ cell_radius must return that seed.
        #[test]
        fn prop_cone_query_includes_center_seed(
            pts in proptest::collection::vec((ra_strategy(), dec_strategy()), 1..50)
        ) {
            let binner = HealpixBinner::new(8);

            // Build seeds from points.
            let seeds: Vec<SeedNode> = pts.iter().enumerate().map(|(i, (ra, dec))| {
                mk_seed(i as u64, *ra, *dec)
            }).collect();

            let index = SeedSpatialIndex::build(&seeds, &binner);

            // For each seed, query cone at its mid-position and ensure it is included.
            for s in &seeds {
                let radius = binner.cell_radius().max(arcsec_to_rad(1.0));
                let found: Vec<SeedId> = index.cone_query(&binner, s.plane.ra_mid, s.plane.dec_mid, radius).collect();
                prop_assert!(found.contains(&s.seed_id));
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
}
