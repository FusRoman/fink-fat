use std::collections::HashMap;

use crate::{
    AlertId, MjdTt, Radians,
    alerts::Alert,
    spacetime_bucket::{
        spatial_binner::{SpatialBinner, SpatialKey},
        time_binner::{TimeBin, TimeBinner},
    },
};

/// Joint spatio-temporal key = (spatial cell, time bin).
///
/// This is the hash-map key into bucketed memberships. Comparable and hashable
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
/// Invariant
/// --------
/// `members` are sorted by **increasing MJD(TT)** with [`AlertId`] as
/// **tie-break** on exact time equality.
#[derive(Clone, Debug)]
pub struct Bucket<Id> {
    /// List of alert identifiers belonging to that bucket (sorted; see invariant).
    pub members: Vec<Id>,
}

/// Global index of all buckets for a given night or time span.
///
/// This is a thin wrapper around a `HashMap` keyed by [`BucketKey`].
#[derive(Clone, Debug, Default)]
pub struct BucketIndex<Id> {
    /// Memberships keyed by `(SpatialKey, TimeBin)`.
    pub buckets: HashMap<BucketKey, Bucket<Id>>,
}

/// Compute the joint `BucketKey` for a single alert sample.
///
/// Thin helper around [`SpatialBinner::key_for`] and [`TimeBinner::bin_for`].
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

/// Build a `BucketIndex` from a slice of alerts.
///
/// This simple builder:
///
/// 1. Computes the `(SpatialKey, TimeBin)` for each alert,
/// 2. Pushes the alert id into the corresponding bucket,
/// 3. Sorts memberships in each bucket by `(MJD(TT), AlertId)`.
///
/// It avoids any progress reporting or pre-count pass and is intended as a
/// straightforward, easy-to-read implementation.
///
/// Parameters
/// ----------
/// * `alerts` – Contiguous slice of alerts for one night (or time span).
/// * `space_binner` – Spatial binner mapping `(ra, dec)` to `SpatialKey`.
/// * `time_binner` – Time binner mapping `mjd_tt` to `TimeBin`.
///
/// Return
/// ------
/// A [`BucketIndex`] with all non-empty buckets populated and sorted.
pub fn build_bucket_index<Bs, Bt>(
    alerts: &[Alert],
    space_binner: &Bs,
    time_binner: &Bt,
) -> BucketIndex<AlertId>
where
    Bs: SpatialBinner,
    Bt: TimeBinner,
{
    let mut buckets: HashMap<BucketKey, Bucket<AlertId>> = HashMap::new();

    // 1) Group alerts into buckets.
    for alert in alerts {
        let key = bucket_key_for(alert.ra, alert.dec, alert.mjd_tt, space_binner, time_binner);

        let bucket = buckets.entry(key).or_insert_with(|| Bucket {
            members: Vec::new(),
        });

        bucket.members.push(alert.id);
    }

    // 2) Sort members inside each bucket by (time, id).
    //
    // This implementation assumes that `AlertId` is a dense index into `alerts`,
    // i.e. `alert.id.idx() == index` for the given slice. If you ever relax this
    // invariant, you will need a small `AlertId -> MjdTt` lookup map here.
    for bucket in buckets.values_mut() {
        bucket.members.sort_unstable_by(|&id1, &id2| {
            let t1 = alerts[id1.idx()].mjd_tt;
            let t2 = alerts[id2.idx()].mjd_tt;

            match t1.total_cmp(&t2) {
                std::cmp::Ordering::Equal => id1.cmp(&id2),
                ord => ord,
            }
        });
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
    }

    /// Dummy time binner: bins by floor(MJD).
    struct DummyTimeBinner;

    impl TimeBinner for DummyTimeBinner {
        fn bin_for(&self, mjd_tt: MjdTt) -> TimeBin {
            TimeBin(mjd_tt.floor() as i64)
        }

        fn bins_in_range(&self, t0: MjdTt, t1: MjdTt) -> Vec<TimeBin> {
            // Inclusive range of integer days between min(t0, t1) and max(t0, t1).
            let (t_min, t_max) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
            let start = t_min.floor() as i64;
            let end = t_max.floor() as i64;
            (start..=end).map(TimeBin).collect()
        }

        fn bin_width(&self) -> MjdTt {
            // 1 day bins.
            1.0
        }
    }

    /// Helper to build a minimal `Alert` for tests.
    fn mk_alert(id: AlertId, ra: Radians, dec: Radians, mjd_tt: MjdTt) -> Alert {
        Alert {
            id,
            ra,
            dec,
            mjd_tt,
            ..Default::default()
        }
    }

    fn aid(i: u32) -> AlertId {
        AlertId::new(i)
    }

    #[test]
    fn empty_input_yields_empty_index() {
        let sb = DummySpatialBinner;
        let tb = DummyTimeBinner;

        let alerts: Vec<Alert> = Vec::new();
        let index = build_bucket_index(&alerts, &sb, &tb);

        assert!(index.buckets.is_empty());
    }

    #[test]
    fn one_alert_creates_one_bucket_with_single_member() {
        let sb = DummySpatialBinner;
        let tb = DummyTimeBinner;

        let a0 = mk_alert(aid(0), 1.0, 0.1, 59000.25);
        let a0_id = a0.id;
        let alerts = vec![a0];

        let index = build_bucket_index(&alerts, &sb, &tb);
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
        assert_eq!(bucket.members[0], a0_id);
    }

    #[test]
    fn alerts_in_same_cell_and_bin_end_up_in_same_bucket() {
        let sb = DummySpatialBinner;
        // 1-day bins => both in same bin
        let tb = DummyTimeBinner;

        let t0 = 59000.1;
        let a0 = mk_alert(aid(0), 1.0, 0.1, t0);
        let a1 = mk_alert(aid(1), 1.000_001, 0.100_001, t0 + 0.3); // same floor(MJD)

        let alerts = vec![a0, a1];
        let index = build_bucket_index(&alerts, &sb, &tb);

        assert_eq!(index.buckets.len(), 1);

        let key = BucketKey {
            space_key: sb.key_for(alerts[0].ra, alerts[0].dec),
            time_bin: tb.bin_for(alerts[0].mjd_tt),
        };

        let bucket = index.buckets.get(&key).unwrap();
        assert_eq!(bucket.members.len(), 2);
        // Members should be sorted by time, but here t0 < t1 and ids are 0,1
        assert_eq!(bucket.members, vec![aid(0), aid(1)]);
    }

    #[test]
    fn alerts_split_into_different_time_bins() {
        let sb = DummySpatialBinner;
        let tb = DummyTimeBinner;

        let a0 = mk_alert(aid(0), 1.0, 0.0, 59000.4); // bin 59000
        let a1 = mk_alert(aid(1), 1.0, 0.0, 59001.2); // bin 59001
        let a2 = mk_alert(aid(2), 1.0, 0.0, 59002.9); // bin 59002

        let alerts = vec![a0, a1, a2];
        let index = build_bucket_index(&alerts, &sb, &tb);

        // 3 different time bins, same spatial cell => 3 buckets
        assert_eq!(index.buckets.len(), 3);

        for alert in &alerts {
            let key = BucketKey {
                space_key: sb.key_for(alert.ra, alert.dec),
                time_bin: tb.bin_for(alert.mjd_tt),
            };
            let bucket = index.buckets.get(&key).expect("missing bucket");
            assert_eq!(bucket.members.len(), 1);
            assert_eq!(bucket.members[0], alert.id);
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
        }

        let sb = SplitSpatialBinner;
        let tb = DummyTimeBinner;

        let t = 59000.5;
        let a0 = mk_alert(aid(0), 1.0, 0.0, t); // RA < π -> cell 0
        let a1 = mk_alert(aid(1), 3.5, 0.0, t); // RA > π -> cell 1

        // Capture ids and keys before moving alerts into the vector.
        let a0_id = a0.id;
        let a1_id = a1.id;
        let key0 = BucketKey {
            space_key: sb.key_for(a0.ra, a0.dec),
            time_bin: tb.bin_for(a0.mjd_tt),
        };
        let key1 = BucketKey {
            space_key: sb.key_for(a1.ra, a1.dec),
            time_bin: tb.bin_for(a1.mjd_tt),
        };

        let alerts = vec![a0, a1];
        let index = build_bucket_index(&alerts, &sb, &tb);

        assert_eq!(index.buckets.len(), 2);

        let b0 = index.buckets.get(&key0).unwrap();
        let b1 = index.buckets.get(&key1).unwrap();

        assert_eq!(b0.members, vec![a0_id]);
        assert_eq!(b1.members, vec![a1_id]);
    }

    #[test]
    fn members_are_sorted_by_time_then_id() {
        let sb = DummySpatialBinner;
        let tb = DummyTimeBinner;

        // All alerts in same spatial cell & same time bin (floor 59000)
        let t_base = 59000.2;
        // We will deliberately give them out-of-order ids / times
        let a0 = mk_alert(aid(0), 1.0, 0.0, t_base + 0.3);
        let a1 = mk_alert(aid(1), 1.0, 0.0, t_base); // earliest
        let a2 = mk_alert(aid(2), 1.0, 0.0, t_base + 0.3); // same time as a0, higher id

        // Note: order in the slice is [a0, a1, a2]
        let alerts = vec![a0, a1, a2];
        let index = build_bucket_index(&alerts, &sb, &tb);

        assert_eq!(index.buckets.len(), 1);
        let key = BucketKey {
            space_key: sb.key_for(alerts[0].ra, alerts[0].dec),
            time_bin: tb.bin_for(alerts[0].mjd_tt),
        };
        let bucket = index.buckets.get(&key).unwrap();

        // Check sorted by (time, id):
        //  - a1 first (earliest time),
        //  - then a0 and a2 (same time), sorted by id → 0 then 2.
        assert_eq!(bucket.members, vec![aid(1), aid(0), aid(2)]);
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
        }

        struct TestTimeBinner;
        impl TimeBinner for TestTimeBinner {
            fn bin_for(&self, mjd_tt: MjdTt) -> TimeBin {
                TimeBin((mjd_tt * 10.0).floor() as i64) // 0.1 day bins
            }

            fn bins_in_range(&self, t0: MjdTt, t1: MjdTt) -> Vec<TimeBin> {
                let (t_min, t_max) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
                let start = (t_min * 10.0).floor() as i64;
                let end = (t_max * 10.0).floor() as i64;
                (start..=end).map(TimeBin).collect()
            }

            fn bin_width(&self) -> MjdTt {
                // 0.1 day bins.
                0.1
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
