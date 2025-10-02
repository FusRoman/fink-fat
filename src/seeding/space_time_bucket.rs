/* --------------------------- Keys & Buckets ------------------------- */

use ahash::AHashMap;
use indicatif::ProgressBar;

use crate::{
    alerts::{Alert, AlertId},
    progress::throttled_inc,
    MjdTt, Radians,
};

/// Spatial key (e.g. HEALPix/HTM cell id, or simple lon/lat grid index)
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct SpatialKey(pub u64);

/// Time bin key (e.g. uniform bins of Δt)
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct TimeBin(pub i64);

/// Joint spatio-temporal bucket key = (spatial cell, time bin)
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct BucketKey {
    pub space_key: SpatialKey,
    pub time_bin: TimeBin,
}

/// A bucket stores the list of Detection ids that fall into (SpatialKey, TimeBin)
#[derive(Clone, Debug)]
pub struct Bucket {
    pub key: BucketKey,
    pub members: Vec<AlertId>,
}

/// Index of all buckets for a given night / time span
#[derive(Default, Debug)]
pub struct BucketIndex {
    pub buckets: AHashMap<BucketKey, Bucket>,
    pub bucket_sizes: AHashMap<BucketKey, usize>,
}

/* ---------------------------- Binners ------------------------------- */

/// Converts (ra, dec) into a spatial key; can also enumerate neighbor keys.
pub trait SpatialBinner {
    /// Return the spatial cell for given sky position.
    fn key_for(&self, ra: Radians, dec: Radians) -> SpatialKey;

    /// Return neighbor cells needed to cover an angular radius (radians).
    /// (Inclut typiquement `key` lui-même.)
    fn neighbors(&self, key: SpatialKey, ang_radius: Radians) -> Vec<SpatialKey>;

    /// Characteristic cell angular size (radians), utile pour régler le rayon de voisinage.
    fn cell_radius(&self) -> Radians;
}

/// Converts mjd into a time bin; can also enumerate bins in a time window.
pub trait TimeBinner {
    /// Return the time bin for a given MJD(TT).
    fn bin_for(&self, mjd_tt: MjdTt) -> TimeBin;

    /// Return all bins overlapping [t0, t1].
    fn bins_in_range(&self, t0: MjdTt, t1: MjdTt) -> Vec<TimeBin>;

    /// Bin width (days).
    fn bin_width(&self) -> MjdTt;
}

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

// invariants: Bucket.members est trié par MJD(TT) croissant.
// tie-break: AlertId croissant si MJD égaux.

#[inline]
fn build_time_lookup(alerts: &[Alert]) -> AHashMap<AlertId, MjdTt> {
    let mut map = AHashMap::with_capacity(alerts.len());
    for a in alerts {
        map.insert(a.id, a.mjd_tt);
    }
    map
}

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

    // Pass 2: remplissage (push « brut »)
    for a in alerts {
        let key = bucket_key_for(a.ra, a.dec, a.mjd_tt, space_binner, time_binner);
        let bucket = index.buckets.get_mut(&key).unwrap();
        bucket.members.push(a.id);
    }

    // Tri final par temps (et AlertId en tie-break) — garantit l’invariant.
    let time_of = build_time_lookup(alerts);
    for bucket in index.buckets.values_mut() {
        bucket.members.sort_unstable_by(|&id1, &id2| {
            match time_of[&id1].partial_cmp(&time_of[&id2]).unwrap() {
                std::cmp::Ordering::Equal => id1.cmp(&id2),
                ord => ord,
            }
        });
    }

    index
}

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

    // Pass 1: precount
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

    // Pass 2: remplissage
    for a in alerts {
        let key = BucketKey {
            space_key: space_binner.key_for(a.ra, a.dec),
            time_bin: time_binner.bin_for(a.mjd_tt),
        };
        let bucket = index.buckets.get_mut(&key).unwrap();
        bucket.members.push(a.id);
        processed += 1;
        throttled_inc(pb, processed, &mut last_drawn, tick);
    }

    // Tri final des membres par temps (et id en tie-break)
    let mut time_of: AHashMap<AlertId, MjdTt> = AHashMap::with_capacity(alerts.len());
    for a in alerts {
        time_of.insert(a.id, a.mjd_tt);
    }
    // on étend la longueur totale pour couvrir aussi le tri bucket-par-bucket
    pb.set_length(2 * alerts.len() as u64 + index.buckets.len() as u64);

    for b in index.buckets.values_mut() {
        b.members.sort_unstable_by(
            |&i, &j| match time_of[&i].partial_cmp(&time_of[&j]).unwrap() {
                std::cmp::Ordering::Equal => i.cmp(&j),
                ord => ord,
            },
        );
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
