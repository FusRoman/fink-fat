//! # Keys & Buckets
//!
//! Spatio–temporal bucketing utilities for LSST/Fink alerts.
//!
//! This module provides:
//! - **Compact keys** for spatial cells and time bins (`SpatialKey`, `TimeBin`),
//! - A **joint key** (`BucketKey`) and a `Bucket` container of alert memberships,
//! - A memory-efficient `BucketIndex` mapping keys → members (with precomputed sizes),
//! - Two **binners** traits (`SpatialBinner`, `TimeBinner`) to plug different schemes
//!   (e.g., HEALPix for space, uniform bins for time),
//! - Fast builders that take a slice of `Alert` and produce a ready-to-query index,
//!   with an optional progress bar for long runs.
//!
//! ## Units
//! - Right ascension / declination: **radians** (`Radians`),
//! - Time stamps: **MJD (TT)** days (`MjdTt`).
//!
//! ## Invariants
//! - For each `Bucket`, `members` are **sorted by increasing MJD(TT)**,
//!   with `AlertId` as **tie-break** on exact time equality. This makes later
//!   time-ordered scans and neighbor searches deterministic.
//!
//! ## Performance notes
//! - The builder performs **two passes** over input alerts:
//!   1) size precount (to reserve exact capacity per bucket),
//!   2) contiguous push + final in-bucket sort.
//! - Hash maps use `AHashMap` for throughput on large nightly volumes (10M+ alerts).
//! - Sorting uses `sort_unstable_by`, which is non-stable but deterministic with the
//!   explicit `(time, id)` comparator.
//!
//! ## Example
//! ```rust
//! # use fink_fat::keys_buckets::*;
//! # use fink_fat::alerts::Alert;
//! // Given concrete binners (e.g., HealpixBinner, UniformTimeBinner) implementing the traits:
//! let space = HealpixBinner::new(10);
//! let time  = UniformTimeBinner::new(0.01); // 0.01 day ≈ 14.4 minutes
//!
//! // `alerts` is a contiguous slice of nightly detections (radians / MJD(TT)):
//! let index = build_index_from_alerts_precise(&alerts, &space, &time);
//!
//! // Access a specific bucket and iterate members in chronological order:
//! let key = BucketKey {
//!     space_key: space.key_for(alerts[0].ra, alerts[0].dec),
//!     time_bin:  time.bin_for(alerts[0].mjd_tt),
//! };
//! if let Some(bucket) = index.buckets.get(&key) {
//!     for alert_id in &bucket.members {
//!         // Use alert_id to access your external store
//!     }
//! }
//! ```
//!
//! ## See also
//! - `generate_pairs` / `generate_triplets` consumers that rely on the sorting invariant,
//! - Spatial binners such as **HEALPix**/**HTM** adapters,
//! - Time binners such as fixed-width or cadence-aware schemes.

use ahash::AHashMap;
use indicatif::ProgressBar;

use crate::{
    alerts::{Alert, AlertId},
    progress::throttled_inc,
    MjdTt, Radians,
};

/// Compact spatial cell identifier.
///
/// Typically the output of a sky partitioner (e.g., HEALPix, HTM, or a lon/lat grid).
/// Stores the cell id as an unsigned 64-bit integer to accommodate deep tessellations.
///
/// ### Notes
/// - The specific **encoding** depends on the `SpatialBinner` implementation.
/// - Comparable and hashable to serve as a map key.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct SpatialKey(pub u64);

/// Compact time bin identifier.
///
/// Usually an integer index of uniform-width bins on the MJD(TT) axis.
/// Signed 64-bit to support long spans and negative offsets if needed.
///
/// ### Notes
/// - The exact mapping `MJD → TimeBin` depends on the `TimeBinner`.
/// - Comparable and hashable to serve as a map key.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct TimeBin(pub i64);

/// Joint spatio-temporal key = (spatial cell, time bin).
///
/// This is the hash-map key into bucketed memberships. Compared/orderable and hashable
/// to support fast indexing and deterministic iteration.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct BucketKey {
    /// Spatial cell id.
    pub space_key: SpatialKey,
    /// Time bin id.
    pub time_bin: TimeBin,
}

/// Container of alert memberships for a given `(SpatialKey, TimeBin)`.
///
/// ### Invariant
/// `members` are sorted by **increasing MJD(TT)** with `AlertId` as **tie-break**.
/// This is enforced by the builders provided in this module.
#[derive(Clone, Debug)]
pub struct Bucket {
    /// The joint spatio-temporal key for this bucket.
    pub key: BucketKey,
    /// List of alert identifiers belonging to that bucket (sorted; see invariant).
    pub members: Vec<AlertId>,
}

/// Global index of all buckets for a given night or time span.
///
/// Provides two maps:
/// - `buckets`: the actual memberships,
/// - `bucket_sizes`: the pre-counted sizes (useful for diagnostics and pre-allocation).
#[derive(Default, Debug)]
pub struct BucketIndex {
    /// Memberships keyed by `(SpatialKey, TimeBin)`.
    pub buckets: AHashMap<BucketKey, Bucket>,
    /// Precounted bucket sizes (same keys as `buckets`).
    pub bucket_sizes: AHashMap<BucketKey, usize>,
}

/* ---------------------------- Binners ------------------------------- */

/// Spatial binning interface.
///
/// Implement this for your sky partitioner (HEALPix, HTM, lon/lat grid…).
pub trait SpatialBinner {
    /// Return the spatial cell for a given sky position.
    ///
    /// Parameters
    /// ----------
    /// - `ra`: Right ascension (radians).
    /// - `dec`: Declination (radians).
    ///
    /// Return
    /// ------
    /// `SpatialKey` – the spatial cell id covering `(ra, dec)`.
    fn key_for(&self, ra: Radians, dec: Radians) -> SpatialKey;

    /// Enumerate neighbor cells needed to cover an **angular radius** around `key`.
    ///
    /// The radius is in **radians** and typically chosen as a small multiple of the
    /// cell's characteristic scale (see [`cell_radius`](crate::seeding::space_time_bucket::SpatialBinner::cell_radius)).
    ///
    /// Notes
    /// -----
    /// Implementations usually **include `key` itself** in the returned list,
    /// but callers should not rely on this unless documented by the concrete type.
    fn neighbors(&self, key: SpatialKey, ang_radius: Radians) -> Vec<SpatialKey>;

    /// Characteristic angular **radius** for a single cell (radians).
    ///
    /// This can drive the choice of neighbor coverage (e.g., `k × cell_radius()`).
    fn cell_radius(&self) -> Radians;
}

/// Time binning interface.
///
/// Implement this for your time partitioner (uniform bins, cadence-aware bins…).
pub trait TimeBinner {
    /// Return the **time bin** covering `mjd_tt`.
    ///
    /// Parameters
    /// ----------
    /// - `mjd_tt`: Time stamp in MJD(TT) days.
    fn bin_for(&self, mjd_tt: MjdTt) -> TimeBin;

    /// Enumerate all bins **overlapping** the closed interval `[t0, t1]`.
    ///
    /// Parameters
    /// ----------
    /// - `t0`, `t1`: Start/end in MJD(TT) days (no ordering required; implementations may swap).
    fn bins_in_range(&self, t0: MjdTt, t1: MjdTt) -> Vec<TimeBin>;

    /// The **bin width** in days.
    fn bin_width(&self) -> MjdTt;
}

/// Compute the joint `BucketKey` for a single alert sample.
///
/// Thin helper around `SpatialBinner::key_for` and `TimeBinner::bin_for`.
#[inline]
fn bucket_key_for<Bs: SpatialBinner, Bt: TimeBinner>(
    ra: Radians,
    dec: Radians,
    mjd_tt: MjdTt,
    sb: &Bs,
    tb: &Bt,
) -> BucketKey {
    BucketKey {
        space_key: sb.key_for(ra, dec),
        time_bin: tb.bin_for(mjd_tt),
    }
}

/// Precount the number of members per bucket for capacity reservation.
///
/// Parameters
/// ----------
/// - `alerts`: Iterable of `&Alert`.
/// - `sb`: Spatial binner.
/// - `tb`: Time binner.
///
/// Return
/// ------
/// `AHashMap<BucketKey, usize>` – Exact per-bucket counts to reserve capacities.
///
/// Complexity
/// ----------
/// `O(N)` over the number of input alerts.
fn precount_bucket_sizes<'a, I, Bs, Bt>(alerts: I, sb: &Bs, tb: &Bt) -> AHashMap<BucketKey, usize>
where
    I: IntoIterator<Item = &'a Alert>,
    Bs: SpatialBinner,
    Bt: TimeBinner,
{
    let mut sizes: AHashMap<BucketKey, usize> = AHashMap::new();
    for a in alerts {
        let key = bucket_key_for(a.ra, a.dec, a.mjd_tt, sb, tb);
        *sizes.entry(key).or_insert(0) += 1;
    }
    sizes
}

/// Build a `AlertId → MJD(TT)` lookup used for in-bucket sorting.
///
/// Complexity: `O(N)` time and `O(N)` extra memory.
#[inline]
fn build_time_lookup(alerts: &[Alert]) -> AHashMap<AlertId, MjdTt> {
    let mut map = AHashMap::with_capacity(alerts.len());
    for a in alerts {
        map.insert(a.id, a.mjd_tt);
    }
    map
}

/// Build a `BucketIndex` from a slice of alerts (two-pass, precise capacities).
///
/// The function performs:
/// 1. **Precount** per-bucket sizes to allocate exact `Vec` capacities,
/// 2. **Populate** members by a single push per alert,
/// 3. **Sort** each bucket by `(MJD(TT), AlertId)` to enforce the invariant.
///
/// Parameters
/// ----------
/// - `alerts`: Slice of input detections (fields must be filled in radians / MJD(TT)).
/// - `space_binner`: Spatial binner implementation (e.g., HEALPix).
/// - `time_binner`: Time binner implementation (e.g., uniform width).
///
/// Return
/// ------
/// `BucketIndex` – Ready-to-use index with sorted memberships.
///
/// Complexity
/// ----------
/// - Time: `O(N + B log B)` where `B` is the sum of per-bucket member counts,
///   i.e. the cost of sorting each bucket.
/// - Memory: `O(N)` for memberships + maps.
///
/// Panics
/// ------
/// - If internal logic encounters a missing bucket when inserting (should not happen).
///
/// See also
/// --------
/// - [`build_index_from_alerts_precise_with_progress`] – same, but with progress feedback.
pub fn build_index_from_alerts_precise<Bs, Bt>(
    alerts: &[Alert],
    space_binner: &Bs,
    time_binner: &Bt,
) -> BucketIndex
where
    Bs: SpatialBinner,
    Bt: TimeBinner,
{
    let sizes = precount_bucket_sizes(alerts.iter(), space_binner, time_binner);

    let mut index = BucketIndex {
        buckets: AHashMap::with_capacity(sizes.len()),
        bucket_sizes: sizes.clone(),
    };

    for (key, &cap) in &sizes {
        index.buckets.insert(
            *key,
            Bucket {
                key: *key,
                members: Vec::with_capacity(cap),
            },
        );
    }

    // Pass 2: populate (raw push, no sorting yet).
    for a in alerts {
        let key = bucket_key_for(a.ra, a.dec, a.mjd_tt, space_binner, time_binner);
        let bucket = index.buckets.get_mut(&key).expect("bucket must exist");
        bucket.members.push(a.id);
    }

    // Final sort by time (and AlertId as tie-breaker) — enforces the invariant.
    let time_of = build_time_lookup(alerts);
    for bucket in index.buckets.values_mut() {
        bucket.members.sort_unstable_by(|&id1, &id2| {
            match time_of[&id1]
                .partial_cmp(&time_of[&id2])
                .expect("valid times")
            {
                std::cmp::Ordering::Equal => id1.cmp(&id2),
                ord => ord,
            }
        });
    }

    index
}

/// Build a `BucketIndex` with progress reporting (useful for 10M+ alerts).
///
/// Same algorithm as [`build_index_from_alerts_precise`], but regularly updates
/// an `indicatif::ProgressBar` during:
/// - the **precount** pass,
/// - the **populate** pass,
/// - the **per-bucket sort** phase.
///
/// The progress bar message is set to `"buckets"`, and the final status ends
/// with `"buckets ✓"`.
///
/// Parameters
/// ----------
/// - `alerts`: Slice of input detections (radians / MJD(TT)).
/// - `space_binner`: Spatial binner implementation.
/// - `time_binner`: Time binner implementation.
/// - `pb`: External progress bar instance to update.
///
/// Return
/// ------
/// `BucketIndex` – Ready-to-use index with sorted memberships.
///
/// Notes
/// -----
/// - Updates are throttled via `throttled_inc` to avoid excessive redraws.
/// - The bar length is set to `2 * alerts.len() + index.buckets.len()` to account
///   for both passes and the bucket-sorting loop.
///
/// Example
/// -------
/// ```rust
/// # use indicatif::ProgressBar;
/// # use fink_fat::keys_buckets::*;
/// let pb = ProgressBar::new(0);
/// let index = build_index_from_alerts_precise_with_progress(&alerts, &space, &time, &pb);
/// pb.finish_and_clear();
/// ```
pub fn build_index_from_alerts_precise_with_progress<Bs, Bt>(
    alerts: &[Alert],
    space_binner: &Bs,
    time_binner: &Bt,
    pb: &ProgressBar,
) -> BucketIndex
where
    Bs: SpatialBinner,
    Bt: TimeBinner,
{
    let mut processed = 0u64;
    let mut last_drawn = 0u64;
    let tick = 10_000u64;
    pb.set_message("buckets");

    // Pass 1: precount sizes.
    let mut sizes: AHashMap<BucketKey, usize> = AHashMap::new();
    for a in alerts {
        let key = BucketKey {
            space_key: space_binner.key_for(a.ra, a.dec),
            time_bin: time_binner.bin_for(a.mjd_tt),
        };
        *sizes.entry(key).or_insert(0) += 1;
        processed += 1;
        throttled_inc(pb, processed, &mut last_drawn, tick);
    }

    let mut index = BucketIndex {
        buckets: AHashMap::with_capacity(sizes.len()),
        bucket_sizes: sizes.clone(),
    };

    for (key, &cap) in &sizes {
        index.buckets.insert(
            *key,
            Bucket {
                key: *key,
                members: Vec::with_capacity(cap),
            },
        );
    }

    // Pass 2: populate.
    for a in alerts {
        let key = BucketKey {
            space_key: space_binner.key_for(a.ra, a.dec),
            time_bin: time_binner.bin_for(a.mjd_tt),
        };
        let bucket = index.buckets.get_mut(&key).expect("bucket must exist");
        bucket.members.push(a.id);
        processed += 1;
        throttled_inc(pb, processed, &mut last_drawn, tick);
    }

    // Final per-bucket sort (by time then id).
    let mut time_of: AHashMap<AlertId, MjdTt> = AHashMap::with_capacity(alerts.len());
    for a in alerts {
        time_of.insert(a.id, a.mjd_tt);
    }

    // Extend the total length to cover the sorting phase.
    pb.set_length(2 * alerts.len() as u64 + index.buckets.len() as u64);

    for b in index.buckets.values_mut() {
        b.members.sort_unstable_by(|&i, &j| {
            match time_of[&i].partial_cmp(&time_of[&j]).expect("valid times") {
                std::cmp::Ordering::Equal => i.cmp(&j),
                ord => ord,
            }
        });
        processed += 1;
        throttled_inc(pb, processed, &mut last_drawn, 1_000);
    }

    pb.set_position(2 * alerts.len() as u64 + index.buckets.len() as u64);
    pb.finish_with_message("buckets ✓");
    index
}

#[cfg(test)]
mod bucket_tests {
    use super::*;
    use std::f64::consts::PI;

    // Ton type Alert/AlertId
    use crate::{
        alerts::{Alert, AlertId},
        seeding::{healpix_binners::HealpixBinner, uniform_time_binner::UniformTimeBinner},
    };

    /* ----------------------- helpers ----------------------- */

    fn mk_alert(id: AlertId, ra: f64, dec: f64, mjd_tt: f64, band: u8) -> Alert {
        Alert {
            id,
            dia_source_id: id as u64,
            ra,
            dec,
            mjd_tt,
            flux: 0.0,
            flux_err: 0.0,
            band,
        }
    }

    fn two_pi_wrap(x: f64) -> f64 {
        let mut y = x % (2.0 * PI);
        if y < 0.0 {
            y += 2.0 * PI;
        }
        y
    }

    /* ----------------------- unit tests ----------------------- */

    #[test]
    fn buckets_empty_input() {
        let sb = HealpixBinner::new(7);
        let tb = UniformTimeBinner::new(59000.0, 1.0 / 48.0); // 30 min

        let alerts: Vec<Alert> = vec![];
        let idx = build_index_from_alerts_precise(&alerts, &sb, &tb);

        assert!(idx.buckets.is_empty());
        assert!(idx.bucket_sizes.is_empty());
    }

    #[test]
    fn one_alert_one_bucket() {
        let sb = HealpixBinner::new(7);
        let tb = UniformTimeBinner::new(59000.0, 1.0);

        let a = mk_alert(1, 1.0, 0.1, 59000.25, 1);
        let idx = build_index_from_alerts_precise(std::slice::from_ref(&a), &sb, &tb);

        assert_eq!(idx.buckets.len(), 1);
        assert_eq!(idx.bucket_sizes.len(), 1);

        let key = BucketKey {
            space_key: sb.key_for(a.ra, a.dec),
            time_bin: tb.bin_for(a.mjd_tt),
        };
        let b = idx.buckets.get(&key).expect("expected bucket missing");
        assert_eq!(b.members, vec![a.id]);
        assert_eq!(*idx.bucket_sizes.get(&key).unwrap(), 1);
    }

    #[test]
    fn many_alerts_same_bucket() {
        let sb = HealpixBinner::new(8);
        let tb = UniformTimeBinner::new(59000.0, 0.5); // 12h

        // reste dans le même pixel et même bin temporel
        let base_ra = 1.2;
        let base_dec = 0.2;
        let base_t = 59000.1;
        let mut alerts = Vec::new();
        for i in 0..20u32 {
            alerts.push(mk_alert(
                i + 1,
                base_ra + 1e-6 * (i as f64),
                base_dec - 1e-6 * (i as f64),
                base_t + 1e-7 * (i as f64),
                1,
            ));
        }

        let idx = build_index_from_alerts_precise(&alerts, &sb, &tb);

        assert_eq!(idx.buckets.len(), 1);
        let key = BucketKey {
            space_key: sb.key_for(base_ra, base_dec),
            time_bin: tb.bin_for(base_t),
        };
        let b = idx.buckets.get(&key).unwrap();
        assert_eq!(b.members.len(), alerts.len());
        assert_eq!(*idx.bucket_sizes.get(&key).unwrap(), alerts.len());
    }

    #[test]
    fn split_across_time_bins() {
        let sb = HealpixBinner::new(7);
        let dt = 1.0 / 24.0; // 1h
        let tb = UniformTimeBinner::new(59000.0, dt);

        // Même pixel, mais on force 3 bins temporels successifs
        let ra = 2.0;
        let dec = 0.0;
        let t0 = 59000.2;
        let alerts = vec![
            mk_alert(1, ra, dec, t0 + 0.1 * dt, 1), // bin k
            mk_alert(2, ra, dec, t0 + 1.1 * dt, 1), // bin k+1
            mk_alert(3, ra, dec, t0 + 2.1 * dt, 1), // bin k+2
        ];

        let idx = build_index_from_alerts_precise(&alerts, &sb, &tb);
        assert_eq!(idx.buckets.len(), 3);

        for a in &alerts {
            let key = BucketKey {
                space_key: sb.key_for(a.ra, a.dec),
                time_bin: tb.bin_for(a.mjd_tt),
            };
            let b = idx.buckets.get(&key).unwrap();
            assert_eq!(b.members, vec![a.id]);
            assert_eq!(*idx.bucket_sizes.get(&key).unwrap(), 1);
        }
    }

    #[test]
    fn split_across_spatial_pixels() {
        let sb = HealpixBinner::new(6);
        let tb = UniformTimeBinner::new(59000.0, 1.0);

        let r_cell = sb.cell_radius();
        let ra0 = 1.0;
        let dec = 0.0;
        // écarte suffisamment en longitude pour changer de pixel
        let ra1 = two_pi_wrap(ra0 + 3.0 * r_cell);

        let t = 59000.25;
        let a0 = mk_alert(1, ra0, dec, t, 1);
        let a1 = mk_alert(2, ra1, dec, t, 1);

        // s'assure qu'on est bien sur deux pixels différents
        let s0 = sb.key_for(ra0, dec);
        let s1 = sb.key_for(ra1, dec);
        assert_ne!(s0, s1, "expected two different spatial pixels");

        let idx = build_index_from_alerts_precise(&[a0.clone(), a1.clone()], &sb, &tb);
        assert_eq!(idx.buckets.len(), 2);

        let k0 = BucketKey {
            space_key: s0,
            time_bin: tb.bin_for(t),
        };
        let k1 = BucketKey {
            space_key: s1,
            time_bin: tb.bin_for(t),
        };

        assert_eq!(idx.buckets.get(&k0).unwrap().members, vec![a0.id]);
        assert_eq!(idx.buckets.get(&k1).unwrap().members, vec![a1.id]);
        assert_eq!(*idx.bucket_sizes.get(&k0).unwrap(), 1);
        assert_eq!(*idx.bucket_sizes.get(&k1).unwrap(), 1);
    }

    #[test]
    fn bucket_sizes_match_membership_counts() {
        let sb = HealpixBinner::new(7);
        let tb = UniformTimeBinner::new(59000.0, 1.0 / 48.0);

        // mélange d'alertes (quelques dizaines suffisent)
        let mut alerts = Vec::new();
        for i in 0..100u32 {
            let ra = two_pi_wrap(0.1 + (i as f64) * 0.05);
            let dec = 0.3 - 0.002 * (i as f64 % 10.0);
            let t = 59000.0 + (i as f64 % 20.0) * (1.0 / 48.0);
            alerts.push(mk_alert(i + 1, ra, dec, t, 1));
        }

        let idx = build_index_from_alerts_precise(&alerts, &sb, &tb);

        // reconstruit un histogramme attendu
        let mut expected: AHashMap<BucketKey, usize> = AHashMap::new();
        for a in &alerts {
            let key = BucketKey {
                space_key: sb.key_for(a.ra, a.dec),
                time_bin: tb.bin_for(a.mjd_tt),
            };
            *expected.entry(key).or_insert(0) += 1;
        }

        assert_eq!(
            idx.bucket_sizes, expected,
            "bucket_sizes must equal expected histogram"
        );

        // vérifie len(members) == bucket_sizes
        for (key, bucket) in &idx.buckets {
            let sz = *idx.bucket_sizes.get(key).unwrap();
            assert_eq!(bucket.members.len(), sz);
        }
    }

    /* ----------------------- property test ----------------------- */

    mod prop {
        use super::*;
        use proptest::prelude::*;
        use std::f64::consts::PI;

        // RA in [0, 2π), DEC in [-π/2+ε, π/2-ε], T ~ [58999.5, 59002.5]
        const LAT_EPS: f64 = 1e-6;

        fn ra_strategy() -> impl Strategy<Value = f64> {
            0.0f64..(2.0 * PI)
        }
        fn dec_strategy() -> impl Strategy<Value = f64> {
            (-(PI / 2.0 - LAT_EPS))..(PI / 2.0 - LAT_EPS)
        }
        fn t_strategy() -> impl Strategy<Value = f64> {
            58999.5f64..59002.5f64
        }

        proptest! {
            #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]

            /// Histogramme attendu == bucket_sizes, total des membres == nb d'alertes,
            /// et chaque id est bien dans le bucket correspondant à (ra, dec, t).
            #[test]
            fn prop_bucket_histogram_consistency(
                triples in proptest::collection::vec((ra_strategy(), dec_strategy(), t_strategy()), 0..200)
            ) {
                let sb = HealpixBinner::new(7);
                let tb = UniformTimeBinner::new(59000.0, 1.0/48.0); // 30 min

                // Construire la liste d'alertes à partir des triplets
                let alerts: Vec<Alert> = triples.iter().enumerate().map(|(i, (ra, dec, t))| {
                    mk_alert((i + 1) as u32, *ra, *dec, *t, 1)
                }).collect();

                let idx = build_index_from_alerts_precise(&alerts, &sb, &tb);

                // Histogramme attendu
                let mut expected: AHashMap<BucketKey, usize> = AHashMap::new();
                for a in &alerts {
                    let key = BucketKey {
                        space_key: sb.key_for(a.ra, a.dec),
                        time_bin: tb.bin_for(a.mjd_tt),
                    };
                    *expected.entry(key).or_insert(0) += 1;
                }

                // 1) bucket_sizes == histogramme attendu
                prop_assert_eq!(idx.bucket_sizes, expected);

                // 2) somme des membres == nb d'alertes
                let total_members: usize = idx.buckets.values().map(|b| b.members.len()).sum();
                prop_assert_eq!(total_members, alerts.len());

                // 3) chaque id dans son bon bucket
                for (key, bucket) in &idx.buckets {
                    for &id in &bucket.members {
                        let a = alerts.iter().find(|x| x.id == id).expect("missing alert in test");
                        let recomputed = BucketKey {
                            space_key: sb.key_for(a.ra, a.dec),
                            time_bin: tb.bin_for(a.mjd_tt),
                        };
                        prop_assert_eq!(&recomputed, key);
                    }
                }
            }
        }
    }
}
