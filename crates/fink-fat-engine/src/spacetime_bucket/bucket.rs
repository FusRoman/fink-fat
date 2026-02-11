//! Generic spatio-temporal bucket index for alerts or alert-like objects.
//!
//! This module provides a minimal and generic infrastructure to group objects
//! (typically alerts or seeds) into **spatio-temporal buckets** keyed by:
//!
//! - a discrete spatial cell ([`SpatialKey`]),
//! - a discrete time bin ([`TimeBin`]).
//!
//! The resulting [`BucketIndex`] is used as a fast pre-filter in higher-level
//! algorithms such as:
//! - intra-night pair generation,
//! - seed construction,
//! - inter-night candidate searches.
//!
//! Design goals
//! ------------
//! - **Generic**: the bucket container is parameterized over the stored object
//!   type (`Object`), and does not depend on `AlertId` or any specific identifier.
//! - **Deterministic**: bucket keys are totally ordered and members inside each
//!   bucket are sorted, enabling reproducible iteration and testing.
//! - **Zero-copy**: when built from alerts, the index stores `&Alert` references
//!   rather than duplicating payloads.
//!
//! Conceptually, this module implements a sparse 2D grid over
//! `(spatial cell × time bin)` with variable occupancy.
//!
//! See also
//! --------
//! - [`SpatialBinner`] – mapping from sky coordinates to spatial cells.
//! - [`TimeBinner`] – mapping from epochs to discrete time bins.
//! - `seeding::pairs` – uses `BucketIndex<&Alert>` for pair generation.
//! - `seeding::seed_spatial_index` – wraps `BucketIndex<&SeedNode>`.

use ahash::AHashMap;

use crate::{
    Alert, MJDTT, Radians,
    spacetime_bucket::{
        spatial_binner::{SpatialBinner, SpatialKey},
        time_binner::{TimeBin, TimeBinner},
    },
};

/// Joint spatio-temporal bucket key.
///
/// A `BucketKey` uniquely identifies one bucket in the index by combining:
/// - a spatial cell (`SpatialKey`),
/// - a time bin (`TimeBin`).
///
/// This type is:
/// - hashable (for use as a `HashMap` key),
/// - totally ordered (useful for deterministic iteration and sorting).
///
/// Ordering
/// --------
/// Ordering is lexicographic:
/// 1. `space_key`,
/// 2. then `time_bin`.
///
/// This ordering has **no physical meaning**; it is only intended for
/// determinism and reproducibility.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct BucketKey {
    /// Discrete spatial cell identifier.
    pub space_key: SpatialKey,
    /// Discrete time-bin identifier.
    pub time_bin: TimeBin,
}

/// Container of objects belonging to a given `(SpatialKey, TimeBin)` bucket.
///
/// The bucket itself is intentionally simple: it is just a sorted list of
/// members. All geometric or temporal logic is handled outside.
///
/// Invariant
/// ---------
/// `members` are sorted by **increasing observation time (`mjd_tt`)**.
///
/// When `Object = &Alert`, this ordering relies on the total ordering
/// implemented for [`Alert`] (primary key: `mjd_tt`, then deterministic
/// tie-breakers).
#[derive(Clone, Debug)]
pub struct Bucket<Object> {
    /// Members of this bucket, sorted by time.
    pub members: Vec<Object>,
}

/// Global index of spatio-temporal buckets.
///
/// A `BucketIndex` is a sparse mapping:
///
/// ```text
/// (SpatialKey, TimeBin) → Bucket<Object>
/// ```
///
/// It does **not** impose any interpretation on the stored objects; it merely
/// groups them according to externally defined spatial and temporal binners.
///
/// Typical usage patterns include:
/// - building once from a slice of alerts,
/// - then iterating over selected neighboring buckets in downstream algorithms.
///
/// Notes
/// -----
/// - Empty buckets are not stored.
/// - The index makes no assumptions about uniqueness of objects across buckets.
#[derive(Clone, Debug, Default)]
pub struct BucketIndex<Object> {
    /// Mapping from `(SpatialKey, TimeBin)` to bucketed members.
    pub buckets: AHashMap<BucketKey, Bucket<Object>>,
}

/// Compute the spatio-temporal [`BucketKey`] for a single sample.
///
/// This is a small convenience wrapper combining:
/// - [`SpatialBinner::key_for`] for spatial discretization,
/// - [`TimeBinner::bin_for`] for temporal discretization.
///
/// Parameters
/// ----------
/// ra : Radians
///     Right ascension (radians).
/// dec : Radians
///     Declination (radians).
/// mjd_tt : MJDTT
///     Observation epoch (MJD TT).
/// sb : &impl SpatialBinner
///     Spatial discretization backend.
/// tb : &impl TimeBinner
///     Temporal discretization backend.
///
/// Returns
/// -------
/// BucketKey
///     Combined spatial + temporal bucket key.
#[inline]
fn bucket_key_for<Bs: SpatialBinner, Bt: TimeBinner>(
    ra: Radians,
    dec: Radians,
    mjd_tt: MJDTT,
    sb: &Bs,
    tb: &Bt,
) -> BucketKey {
    BucketKey {
        space_key: sb.key_for(ra, dec),
        time_bin: tb.bin_for(mjd_tt),
    }
}

/// Build a [`BucketIndex`] from a slice of alerts, storing **borrowed references**.
///
/// This is the canonical constructor used in the early stages of the pipeline
/// (pair generation, seeding).
///
/// The function performs two steps:
/// 1. Group alerts into `(SpatialKey, TimeBin)` buckets.
/// 2. Sort alerts **within each bucket** by increasing `mjd_tt`.
///
/// Parameters
/// ----------
/// alerts : &[Alert]
///     Input alerts to index. They must outlive the returned index.
/// space_binner : &impl SpatialBinner
///     Spatial discretization backend.
/// time_binner : &impl TimeBinner
///     Temporal discretization backend.
///
/// Returns
/// -------
/// BucketIndex<&Alert>
///     A bucket index borrowing the input alerts.
///
/// Notes
/// -----
/// - The index stores `&Alert` references: no alert data is copied.
/// - Buckets are sorted using the total ordering implemented for [`Alert`].
/// - If multiple alerts fall into the same `(SpatialKey, TimeBin)`, their
///   relative ordering is deterministic.
///
/// Invariants guaranteed on return
/// -------------------------------
/// - Every alert appears in exactly one bucket.
/// - Every bucket’s `members` slice is sorted by increasing time.
pub fn build_alert_bucket_index<'alert_lf, Bs, Bt>(
    alerts: &'alert_lf [Alert],
    space_binner: &Bs,
    time_binner: &Bt,
) -> BucketIndex<&'alert_lf Alert>
where
    Bs: SpatialBinner,
    Bt: TimeBinner,
{
    let mut buckets: AHashMap<BucketKey, Bucket<&'alert_lf Alert>> = AHashMap::new();

    // 1) Group alerts into spatio-temporal buckets.
    for alert in alerts {
        let key = bucket_key_for(alert.ra, alert.dec, alert.mjd_tt, space_binner, time_binner);

        buckets
            .entry(key)
            .or_insert_with(|| Bucket {
                members: Vec::new(),
            })
            .members
            .push(alert);
    }

    // 2) Enforce the per-bucket time-ordering invariant.
    //
    // This relies on `Ord for Alert`, so `&Alert` is also orderable.
    for bucket in buckets.values_mut() {
        bucket.members.sort_unstable();
    }

    BucketIndex { buckets }
}

#[cfg(test)]
mod bucket_tests {
    use super::*;

    use std::f64::consts::PI;

    /// Dummy spatial binner: everything goes into the same spatial cell.
    struct DummySpatialBinner;

    impl SpatialBinner for DummySpatialBinner {
        fn key_for(&self, _ra: Radians, _dec: Radians) -> SpatialKey {
            SpatialKey(0)
        }

        fn neighbors(&self, key: SpatialKey, _ang_radius: Radians) -> Vec<SpatialKey> {
            // Trivial implementation: only the cell itself.
            vec![key]
        }

        fn cell_radius(&self) -> Radians {
            // Arbitrary positive value; not used in these tests.
            1.0
        }

        fn neighbors_into(
            &self,
            _key: SpatialKey,
            _ang_radius: Radians,
            _out: &mut Vec<SpatialKey>,
        ) {
            todo!()
        }
    }

    /// Dummy time binner: bins by floor(MJD).
    struct DummyTimeBinner;

    impl TimeBinner for DummyTimeBinner {
        fn bin_for(&self, mjd_tt: MJDTT) -> TimeBin {
            TimeBin(mjd_tt.floor() as i64)
        }

        fn bins_in_range(&self, t0: MJDTT, t1: MJDTT) -> Vec<TimeBin> {
            // Inclusive range of integer days between min(t0, t1) and max(t0, t1).
            let (t_min, t_max) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
            let start = t_min.floor() as i64;
            let end = t_max.floor() as i64;
            (start..=end).map(TimeBin).collect()
        }

        fn bin_width(&self) -> MJDTT {
            // 1 day bins.
            1.0
        }

        fn bin_start(&self, k: i64) -> MJDTT {
            k as f64
        }
    }

    /// Helper to build a minimal `Alert` for tests.
    fn mk_alert(dia_source_id: u64, ra: Radians, dec: Radians, mjd_tt: MJDTT) -> Alert {
        Alert {
            dia_source_id,
            ra,
            dec,
            mjd_tt,
            ..Default::default()
        }
    }

    #[test]
    fn empty_input_yields_empty_index() {
        let sb = DummySpatialBinner;
        let tb = DummyTimeBinner;

        let alerts: Vec<Alert> = Vec::new();
        let index = build_alert_bucket_index(&alerts, &sb, &tb);

        assert!(index.buckets.is_empty());
    }

    #[test]
    fn one_alert_creates_one_bucket_with_single_member() {
        let sb = DummySpatialBinner;
        let tb = DummyTimeBinner;

        let a0 = mk_alert(0, 1.0, 0.1, 59000.25);
        let a0_id = a0.dia_source_id;
        let alerts = vec![a0];

        let index = build_alert_bucket_index(&alerts, &sb, &tb);
        assert_eq!(index.buckets.len(), 1);

        let key = BucketKey {
            space_key: sb.key_for(alerts[0].ra, alerts[0].dec),
            time_bin: tb.bin_for(alerts[0].mjd_tt),
        };

        let bucket = index
            .buckets
            .get(&key)
            .expect("missing bucket for single alert");
        assert_eq!(bucket.members.len(), 1);
        assert_eq!(bucket.members[0].dia_source_id, a0_id);
    }

    #[test]
    fn alerts_in_same_cell_and_bin_end_up_in_same_bucket() {
        let sb = DummySpatialBinner;
        // 1-day bins => both in same bin
        let tb = DummyTimeBinner;

        let t0 = 59000.1;
        let a0 = mk_alert(0, 1.0, 0.1, t0);
        let a1 = mk_alert(1, 1.000_001, 0.100_001, t0 + 0.3); // same floor(MJD)

        let alerts = vec![a0, a1];
        let index = build_alert_bucket_index(&alerts, &sb, &tb);

        assert_eq!(index.buckets.len(), 1);

        let key = BucketKey {
            space_key: sb.key_for(alerts[0].ra, alerts[0].dec),
            time_bin: tb.bin_for(alerts[0].mjd_tt),
        };

        let bucket = index.buckets.get(&key).unwrap();
        assert_eq!(bucket.members.len(), 2);
        // Members should be sorted by time, but here t0 < t1 and ids are 0,1
        assert_eq!(
            bucket
                .members
                .iter()
                .map(|a| a.dia_source_id)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn alerts_split_into_different_time_bins() {
        let sb = DummySpatialBinner;
        let tb = DummyTimeBinner;

        let a0 = mk_alert(0, 1.0, 0.0, 59000.4); // bin 59000
        let a1 = mk_alert(1, 1.0, 0.0, 59001.2); // bin 59001
        let a2 = mk_alert(2, 1.0, 0.0, 59002.9); // bin 59002

        let alerts = vec![a0, a1, a2];
        let index = build_alert_bucket_index(&alerts, &sb, &tb);

        // 3 different time bins, same spatial cell => 3 buckets
        assert_eq!(index.buckets.len(), 3);

        for alert in &alerts {
            let key = BucketKey {
                space_key: sb.key_for(alert.ra, alert.dec),
                time_bin: tb.bin_for(alert.mjd_tt),
            };
            let bucket = index.buckets.get(&key).expect("missing bucket");
            assert_eq!(bucket.members.len(), 1);
            assert_eq!(bucket.members[0].dia_source_id, alert.dia_source_id);
        }
    }

    #[test]
    fn alerts_split_into_different_spatial_cells() {
        // Spatial binner that splits by RA at π
        struct SplitSpatialBinner;
        impl SpatialBinner for SplitSpatialBinner {
            fn key_for(&self, ra: Radians, _dec: Radians) -> SpatialKey {
                if ra < PI {
                    SpatialKey(0)
                } else {
                    SpatialKey(1)
                }
            }

            fn neighbors(&self, key: SpatialKey, _ang_radius: Radians) -> Vec<SpatialKey> {
                // Minimal implementation for tests: just return the cell itself.
                vec![key]
            }

            fn cell_radius(&self) -> Radians {
                // Arbitrary positive radius.
                1.0
            }

            fn neighbors_into(
                &self,
                _key: SpatialKey,
                _ang_radius: Radians,
                _out: &mut Vec<SpatialKey>,
            ) {
                todo!()
            }
        }

        let sb = SplitSpatialBinner;
        let tb = DummyTimeBinner;

        let t = 59000.5;
        let a0 = mk_alert(0, 1.0, 0.0, t); // RA < π -> cell 0
        let a1 = mk_alert(1, 3.5, 0.0, t); // RA > π -> cell 1

        // Capture ids and keys before moving alerts into the vector.
        let a0_id = a0.dia_source_id;
        let a1_id = a1.dia_source_id;
        let key0 = BucketKey {
            space_key: sb.key_for(a0.ra, a0.dec),
            time_bin: tb.bin_for(a0.mjd_tt),
        };
        let key1 = BucketKey {
            space_key: sb.key_for(a1.ra, a1.dec),
            time_bin: tb.bin_for(a1.mjd_tt),
        };

        let alerts = vec![a0, a1];
        let index = build_alert_bucket_index(&alerts, &sb, &tb);

        assert_eq!(index.buckets.len(), 2);

        let b0 = index.buckets.get(&key0).unwrap();
        let b1 = index.buckets.get(&key1).unwrap();

        assert_eq!(
            b0.members
                .iter()
                .map(|a| a.dia_source_id)
                .collect::<Vec<_>>(),
            vec![a0_id]
        );
        assert_eq!(
            b1.members
                .iter()
                .map(|a| a.dia_source_id)
                .collect::<Vec<_>>(),
            vec![a1_id]
        );
    }

    #[test]
    fn members_are_sorted_by_time_then_id() {
        let sb = DummySpatialBinner;
        let tb = DummyTimeBinner;

        // All alerts in same spatial cell & same time bin (floor 59000)
        let t_base = 59000.2;
        // We will deliberately give them out-of-order ids / times
        let a0 = mk_alert(0, 1.0, 0.0, t_base + 0.3);
        let a1 = mk_alert(1, 1.0, 0.0, t_base); // earliest
        let a2 = mk_alert(2, 1.0, 0.0, t_base + 0.3); // same time as a0, higher id

        // Note: order in the slice is [a0, a1, a2]
        let alerts = vec![a0, a1, a2];
        let index = build_alert_bucket_index(&alerts, &sb, &tb);

        assert_eq!(index.buckets.len(), 1);
        let key = BucketKey {
            space_key: sb.key_for(alerts[0].ra, alerts[0].dec),
            time_bin: tb.bin_for(alerts[0].mjd_tt),
        };
        let bucket = index.buckets.get(&key).unwrap();

        // Check sorted by (time, id):
        //  - a1 first (earliest time),
        //  - then a0 and a2 (same time), sorted by id → 0 then 2.
        assert_eq!(
            bucket
                .members
                .iter()
                .map(|a| a.dia_source_id)
                .collect::<Vec<_>>(),
            vec![1, 0, 2]
        );
    }

    #[test]
    fn bucket_key_for_uses_both_spatial_and_time_binners() {
        struct TestSpatialBinner;
        impl SpatialBinner for TestSpatialBinner {
            fn key_for(&self, ra: Radians, _dec: Radians) -> SpatialKey {
                // encode ra bucket roughly as integer
                SpatialKey((ra / 0.5).floor() as u64)
            }

            fn neighbors(&self, key: SpatialKey, _ang_radius: Radians) -> Vec<SpatialKey> {
                // Minimal impl: only the given key.
                vec![key]
            }

            fn cell_radius(&self) -> Radians {
                // Cell size consistent with the 0.5 rad "bucket" we used.
                0.5
            }

            fn neighbors_into(
                &self,
                _key: SpatialKey,
                _ang_radius: Radians,
                _out: &mut Vec<SpatialKey>,
            ) {
                todo!()
            }
        }

        struct TestTimeBinner;
        impl TimeBinner for TestTimeBinner {
            fn bin_for(&self, mjd_tt: MJDTT) -> TimeBin {
                TimeBin((mjd_tt * 10.0).floor() as i64) // 0.1 day bins
            }

            fn bins_in_range(&self, t0: MJDTT, t1: MJDTT) -> Vec<TimeBin> {
                let (t_min, t_max) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
                let start = (t_min * 10.0).floor() as i64;
                let end = (t_max * 10.0).floor() as i64;
                (start..=end).map(TimeBin).collect()
            }

            fn bin_width(&self) -> MJDTT {
                // 0.1 day bins.
                0.1
            }

            fn bin_start(&self, k: i64) -> MJDTT {
                k as f64 * 0.1
            }
        }

        let sb = TestSpatialBinner;
        let tb = TestTimeBinner;

        let ra = 1.2; // 1.2 / 0.5 = 2.4 -> 2
        let dec = 0.0;
        let t = 59000.34; // *10 = 590003.4 -> 590003

        let key = bucket_key_for(ra, dec, t, &sb, &tb);
        assert_eq!(key.space_key, SpatialKey(2));
        assert_eq!(key.time_bin, TimeBin(590_003));
    }
}
