// src/seeding/seed_spatial_index.rs

//! Spatio-temporal bucket index for per-night (or per-slice) seed queries.
//!
//! This module defines [`SeedSpatialIndex`], a thin, domain-specific wrapper
//! around a generic [`BucketIndex`] that stores **borrowed references** to
//! [`SeedNode`] objects.
//!
//! Purpose
//! -------
//! Inter-night linking requires repeatedly querying “right-hand” seeds near a
//! predicted sky position and epoch. A naive scan over all right seeds is
//! prohibitively expensive. [`SeedSpatialIndex`] accelerates this by:
//! - discretizing the sky into spatial cells (`SpatialBinner`),
//! - discretizing time into bins (`TimeBinner`),
//! - storing seeds into buckets keyed by `(SpatialKey, TimeBin)`,
//! - supporting fast *approximate* cone queries that return candidate seeds.
//!
//! Key idea
//! --------
//! Rather than returning integer IDs, this index stores `&SeedNode` references.
//! This removes a `SeedId -> SeedNode` indirection in hot loops and is
//! cache-friendly, but ties the index lifetime to the underlying seed slice.
//!
//! Typical usage
//! -------------
//! ```ignore
//! // Build once per right-hand slice (night / night-pair / time window).
//! let index = SeedSpatialIndex::build(&right_seeds, &spatial_binner, &time_binner);
//!
//! // Query many times: returns borrowed seeds.
//! let center = EquCoord::new(ra, 0.0, dec, 0.0);
//! let candidates: Vec<&SeedNode> = index
//!     .cone_query(&center, radius, t_target)
//!     .collect();
//! ```
//!
//! Approximate nature of queries
//! ----------------------------
//! Cone queries operate at the **cell cover** level: they return all seeds in
//! spatial cells reported by [`SpatialBinner::neighbors`] for the requested
//! radius (and for the relevant time bin). As a result:
//! - some returned seeds can lie slightly outside the strict geometric cone,
//! - some strict cone members could be missed if the cover is approximate.
//!
//! Downstream scoring code should apply exact geometry / kinematic checks.
//!
//! See also
//! --------
//! - [`SeedNode::predict_cone`] – builds conservative search cones.
//! - [`SeedNode::seed_edge_candidates`] – iterates per time-bin queries.
//! - [`BucketIndex`] – generic bucket storage underlying this wrapper.

use ahash::{AHashMap, AHashSet};
use photom::{MJDTT, Radians, coordinates::equatorial::EquCoord};

use crate::{
    seeding::SeedNode,
    spacetime_bucket::{
        bucket::{Bucket, BucketIndex, BucketKey},
        spatial_binner::{SpatialBinner, SpatialKey},
        time_binner::{TimeBin, TimeBinner},
    },
};

/// Spatio-temporal bucket index storing references to [`SeedNode`] values.
///
/// Internally, this wraps a [`BucketIndex<&SeedNode>`] keyed by:
/// - `SpatialKey` from [`SpatialBinner::key_for`], using [`SeedNode::tangent_seed_center`],
/// - `TimeBin` from [`TimeBinner::bin_for`], using `seed.plane_model.epoch_mid`.
///
/// The index is typically built for all seeds of a “right-hand” night, but can
/// also be used for any pre-filtered slice of seeds as long as the caller
/// provides a compatible `TimeBinner`.
///
/// Lifetimes
/// ---------
/// - `'seed_lf`: lifetime of the seed slice passed to [`SeedSpatialIndex::build`].
/// - `'alert_lf`: lifetime of alerts borrowed by each seed.
/// - `'binner_lf`: lifetime of the binner references stored in the index.
///
/// Because the index stores `&SeedNode`, the underlying slice must outlive the
/// index.
#[derive(Clone)]
pub struct SeedSpatialIndex<'seed_lf, 'binner_lf> {
    /// Underlying bucket index mapping `(space_key, time_bin) -> members`.
    inner: BucketIndex<&'seed_lf SeedNode>,

    /// Spatial binner used for key computation and neighbor cover queries.
    pub spatial_binner: &'binner_lf dyn SpatialBinner,

    /// Time binner used to map epochs to discrete bins.
    pub time_binner: &'binner_lf dyn TimeBinner,

    /// Set of time bins that are present in `inner`.
    ///
    /// This is useful for iterating only existing bins (e.g. in
    /// `SeedNode::seed_edge_candidates`) rather than scanning an arbitrary time range.
    pub time_bins: AHashSet<TimeBin>,
}

impl<'seed_lf, 'binner_lf> SeedSpatialIndex<'seed_lf, 'binner_lf> {
    /// Build a spatio-temporal seed index from a slice of [`SeedNode`].
    ///
    /// Each seed is inserted into exactly one bucket:
    /// - `space_key = spatial_binner.key_for(&seed.tangent_seed_center())`
    /// - `time_bin  = time_binner.bin_for(seed.plane_model.epoch_mid)`
    ///
    /// Parameters
    /// ----------
    /// seeds : &[SeedNode]
    ///     Seeds to index (commonly all seeds from one night, already in memory).
    /// spatial_binner : &impl SpatialBinner
    ///     Spatial discretization backend (e.g. HEALPix).
    /// time_binner : &impl TimeBinner
    ///     Time discretization backend (e.g. uniform bins).
    ///
    /// Returns
    /// -------
    /// SeedSpatialIndex
    ///     A new index storing borrowed references to the input seeds.
    ///
    /// Notes
    /// -----
    /// - No deduplication is performed. If the same `SeedNode` reference is present
    ///   multiple times in `seeds`, it will be inserted multiple times.
    /// - The choice of `time_binner` (origin + bin width) impacts candidate fan-out.
    ///   Keep it consistent across indexing and queries.
    pub fn build<Bs: SpatialBinner, Ts: TimeBinner>(
        seeds: &'seed_lf [SeedNode],
        spatial_binner: &'binner_lf Bs,
        time_binner: &'binner_lf Ts,
    ) -> Self {
        let mut buckets: AHashMap<BucketKey, Bucket<&'seed_lf SeedNode>> = AHashMap::new();
        let mut time_bins: AHashSet<TimeBin> = AHashSet::new();

        for s in seeds {
            let space_key = spatial_binner.key_for(&s.tangent_seed_center());
            let time_key = time_binner.bin_for(s.plane_model.epoch_mid);
            time_bins.insert(time_key);

            let key = BucketKey {
                space_key,
                time_bin: time_key,
            };

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
            spatial_binner,
            time_binner,
            time_bins,
        }
    }

    /// Borrow the underlying [`BucketIndex`].
    ///
    /// This is useful for debugging or when you need generic bucket-level
    /// inspection not exposed by this wrapper.
    #[inline]
    pub fn inner(&self) -> &BucketIndex<&'seed_lf SeedNode> {
        &self.inner
    }

    /// Iterate over every seed stored in this index.
    ///
    /// Seeds are yielded in bucket-insertion order (not sorted). Use this
    /// when you need to inspect all right-hand seeds without a cone query
    /// — for example to build a key-to-reference lookup map.
    #[inline]
    pub fn iter_seeds(&self) -> impl Iterator<Item = &SeedNode> + '_ {
        self.inner
            .buckets
            .values()
            .flat_map(|b| b.members.iter().copied())
    }

    /// Perform an approximate cone query at a given epoch.
    ///
    /// The query proceeds as:
    /// 1. Convert `e` to a central spatial cell key via [`SpatialBinner::key_for`].
    /// 2. Ask the spatial binner for a set of neighboring spatial keys whose
    ///    cells cover (approximately) the cone of radius `radius`.
    /// 3. Convert `time` to `time_bin = time_binner.bin_for(time)`.
    /// 4. For each covered spatial key, lookup the bucket `(space_key, time_bin)`
    ///    and yield all member seeds found in that bucket.
    ///
    /// Parameters
    /// ----------
    /// e : &EquCoord
    ///     Equatorial coordinates of the cone center.
    /// radius : Radians
    ///     Angular cone radius (radians).
    /// time : MJDTT
    ///     Target epoch (MJD TT). Determines which `TimeBin` is queried.
    ///
    /// Returns
    /// -------
    /// impl Iterator<Item = &SeedNode>
    ///     Borrowed candidate seeds from the covered spatial cells in the
    ///     relevant time bin.
    ///
    /// Notes
    /// -----
    /// - This is **cell-cover approximate**. It is intended as a fast prefilter.
    /// - Downstream code should apply exact geometry / kinematic scoring.
    /// - If `time` maps to a bin that contains no seeds (not in `time_bins`),
    ///   the iterator will be empty.
    pub fn cone_query(
        &self,
        e: &EquCoord,
        radius: Radians,
        time: MJDTT,
    ) -> impl Iterator<Item = &'seed_lf SeedNode> + '_ {
        let center_key: SpatialKey = self.spatial_binner.key_for(e);
        let time_key = self.time_binner.bin_for(time);

        // Cell cover for the requested radius.
        // For HealpixBinner this is expected to be unique; other SpatialBinner
        // implementations may or may not guarantee uniqueness.
        let cover_keys: Vec<SpatialKey> = self.spatial_binner.neighbors(center_key, radius);

        cover_keys
            .into_iter()
            .filter_map(move |space_key| {
                let key = BucketKey {
                    space_key,
                    time_bin: time_key,
                };
                self.inner.buckets.get(&key)
            })
            .flat_map(|bucket| bucket.members.iter().copied())
    }
}

#[cfg(test)]
mod seed_spatial_index_tests {
    use super::*;
    use photom::{NightId, photometry::Filter};
    use proptest::prelude::*;

    use std::ptr;

    use crate::{
        astro_math::arcsec_to_rad,
        seeding::{
            SeedKey, SeedNode,
            photometry::SeedPhotometry,
            tangent_plane::{PosWithCov, TangentPlaneModel, VelWithCov},
        },
        spacetime_bucket::{
            healpix_binner::HealpixBinner, time_binner::TimeBin,
            uniform_time_binner::UniformTimeBinner,
        },
    };
    use photom::coordinates::{
        cov2::Cov2,
        gnomonic_projection::{TangentPlane, TangentPoint, TangentVec},
    };

    /* ------------------------- helpers ------------------------- */

    // Build a minimal SeedNode positioned at the given sky coordinates.
    fn mk_seed(uniq_id: u64, ra: f64, dec: f64) -> SeedNode {
        let equ = EquCoord::new(ra, 0.0, dec, 0.0);
        let plane = TangentPlane::new(equ);
        let tangent_point = TangentPoint::new(plane, 0.0, 0.0);
        let cov_zero = Cov2::zero();
        let plane_model = TangentPlaneModel {
            epoch_mid: 60000.0,
            pos: PosWithCov {
                tangent_point,
                cov: cov_zero,
            },
            vel: VelWithCov {
                v: TangentVec { dx: 0.0, dy: 0.0 },
                cov: cov_zero,
            },
            acc: None,
        };
        SeedNode {
            key: SeedKey {
                night_id: NightId::new(1),
                unique_id: uniq_id,
            },
            plane_model,
            photom: SeedPhotometry::from_pair(1.0, 0.1, Filter::Int(1), Filter::Int(2)),
            n_obs: 2,
            members: vec![],
        }
    }

    /// True if `items` contains the exact borrowed reference `needle`.
    fn contains_ref(items: &[&SeedNode], needle: &SeedNode) -> bool {
        items.iter().any(|x| ptr::eq(*x, needle))
    }

    /// True if `bucket.members` contains the exact borrowed reference `needle`.
    fn bucket_contains_ref(bucket: &Bucket<&SeedNode>, needle: &SeedNode) -> bool {
        bucket.members.iter().any(|x| ptr::eq(*x, needle))
    }

    /* ------------------------- unit tests ------------------------- */

    #[test]
    fn build_creates_buckets_with_timebin_zero() {
        let spatial_binner = HealpixBinner::new(8);
        let time_binner = UniformTimeBinner::new(60000.0, 1.0);

        // Two seeds in the same cell, one in a neighboring cell.
        let s1 = mk_seed(1, 1.0, 0.2);
        let s2 = mk_seed(2, 1.0 + arcsec_to_rad(1.0) / 0.2f64.cos(), 0.2);
        let s3 = mk_seed(3, 2.0, -0.1);

        // IMPORTANT: keep seeds in a stable Vec so the index can borrow them.
        let seeds = vec![s1, s2, s3];

        let index = SeedSpatialIndex::build(&seeds, &spatial_binner, &time_binner);
        let inner = index.inner();

        // Compute bucket keys (all must use TimeBin(0)).
        let key1 = BucketKey {
            space_key: spatial_binner.key_for(&seeds[0].tangent_seed_center()),
            time_bin: TimeBin(0),
        };
        let key2 = BucketKey {
            space_key: spatial_binner.key_for(&seeds[1].tangent_seed_center()),
            time_bin: TimeBin(0),
        };
        let key3 = BucketKey {
            space_key: spatial_binner.key_for(&seeds[2].tangent_seed_center()),
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
        let spatial_binner = HealpixBinner::new(8);
        let time_binner = UniformTimeBinner::new(60000.0, 1.0);

        // Place seeds close to each other; small radius should cover both if same cell,
        // otherwise a slightly larger radius should get neighbor cells too.
        let ra = 1.5;
        let dec = 0.3;

        let s_primary = mk_seed(1, ra, dec);
        let s_neighbor = mk_seed(2, ra + arcsec_to_rad(30.0) / dec.cos(), dec);
        let t_target = s_primary.plane_model.epoch_mid;

        let seeds = vec![s_primary, s_neighbor];
        let index = SeedSpatialIndex::build(&seeds, &spatial_binner, &time_binner);

        // Query with radius including at least cell_radius to ensure coverage of the containing cell.
        let radius = spatial_binner.cell_radius().max(arcsec_to_rad(45.0)); // 45" cone
        let center = EquCoord::new(ra, 0.0, dec, 0.0);
        let found: Vec<&SeedNode> = index.cone_query(&center, radius, t_target).collect();
        // At least the primary seed must be found (by reference identity).
        assert!(contains_ref(&found, &seeds[0]));

        // Depending on binner neighbors, the neighbor seed may also be found.
        // If not, increase radius and check again.
        if !contains_ref(&found, &seeds[1]) {
            let radius2 = radius * 2.0;
            let found2: Vec<&SeedNode> = index.cone_query(&center, radius2, t_target).collect();
            assert!(contains_ref(&found2, &seeds[1]));
        }
    }

    #[test]
    fn inner_exposes_consistent_mapping() {
        let spatial_binner = HealpixBinner::new(7);
        let time_binner = UniformTimeBinner::new(60000.0, 1.0);
        let s1 = mk_seed(1, 0.5, 0.1);
        let s2 = mk_seed(2, 0.501, 0.101);

        let seeds = vec![s1, s2];
        let index = SeedSpatialIndex::build(&seeds, &spatial_binner, &time_binner);
        let inner = index.inner();

        // Compute keys and verify buckets exist with matching members (by reference identity).
        for s in [&seeds[0], &seeds[1]] {
            let key = BucketKey {
                space_key: spatial_binner.key_for(&s.tangent_seed_center()),
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
            let spatial_binner = HealpixBinner::new(8);
            let time_binner = UniformTimeBinner::new(60000.0, 1.0);

            // Keep seeds in a Vec so the index can borrow them.
            let seeds: Vec<SeedNode> = pts.iter().enumerate().map(|(i, (ra, dec))| {
                mk_seed(i as u64 + 1, *ra, *dec)
            }).collect();

            let index = SeedSpatialIndex::build(&seeds, &spatial_binner, &time_binner);

            // For each seed, query cone at its mid-position and ensure it is included.
            for s in &seeds {
                let radius = spatial_binner.cell_radius().max(arcsec_to_rad(1.0));
                let center = s.tangent_seed_center();
                let found: Vec<&SeedNode> = index.cone_query(&center, radius, s.plane_model.epoch_mid).collect();
                prop_assert!(contains_ref(&found, s));
            }
        }

        /// Buckets should only be associated with TimeBin(0) in this index.
        #[test]
        fn prop_all_buckets_use_timebin_zero(
            pts in proptest::collection::vec((ra_strategy(), dec_strategy()), 0..40)
        ) {
            let spatial_binner = HealpixBinner::new(7);
            let time_binner = UniformTimeBinner::new(60000.0, 1.0);

            let seeds: Vec<SeedNode> = pts.iter().enumerate().map(|(i, (ra, dec))| {
                mk_seed(i as u64 + 1, *ra, *dec)
            }).collect();

            let index = SeedSpatialIndex::build(&seeds, &spatial_binner, &time_binner);

            for (key, _) in &index.inner().buckets {
                prop_assert_eq!(key.time_bin, TimeBin(0));
            }
        }
    }
}
