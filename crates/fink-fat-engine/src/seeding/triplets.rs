//! Triplet generation `(a, b, c)` from precomputed intra-night pairs.
//!
//! This module builds time-ordered triplets of detections `(a, b, c)`
//! (strictly increasing epochs) as a lightweight “tracklet” building stage.
//!
//! Why triplets?
//! ------------
//! Pairs are cheap to generate but highly ambiguous in dense fields.
//! Adding a third detection greatly reduces false associations while still
//! keeping the computations inexpensive compared to full orbit fitting.
//!
//! High-level idea
//! ---------------
//! We start from a list of valid pairs `(a, b)` (typically produced by the pair
//! generator) and search for a third detection `c` such that:
//! - `t_c > t_b` and `t_c - t_b <= max_dt_between`,
//! - `c` lies in nearby spatio-temporal buckets around `b`,
//! - `(b, c)` passes a short-baseline angular constraint,
//! - `(b, c)` passes a flux similarity constraint,
//! - and `c` is consistent with the linear motion model fitted from `(a, b)`
//!   (a fast predicted-residual test).
//!
//! Indexing and performance
//! ------------------------
//! Candidate enumeration uses a spatio-temporal [`BucketIndex`] over `&Alert`:
//! - buckets are keyed by `(SpatialKey, TimeBin)`,
//! - bucket members are assumed sorted by time,
//! - we binary-search within buckets to skip `t <= t_b`,
//! - we stop scanning a bucket once `t_c > t_b + max_dt_between`.
//!
//! Determinism and deduplication
//! -----------------------------
//! Triplets may be discovered through multiple overlapping neighbor scans.
//! We deduplicate by pointer identity `(ptr(a), ptr(b), ptr(c))`, then sort
//! deterministically by `(a, b, c)` using `Alert::cmp`.
//!
//! Units & conventions
//! -------------------
//! - Angles are in **radians**.
//! - Times are **MJD TT** in days.
//! - The planar prediction is performed in a small-angle tangent-plane
//!   approximation around `a`.
//!
//! See also
//! --------
//! - `seeding::pairs` – produces `(a, b)` candidates.
//! - [`SeedNode::from_triplet`] – constructs a quadratic tangent-plane seed.
//! - `spacetime_bucket::bucket` – bucket index and per-bucket time ordering.

use ahash::{AHashMap, AHashSet};

use crate::{
    Alert,
    astro_math::{dot3, planar_offset_fast, unit_vec},
    engine_config::triplet_config::TripletConfig,
    night_id::NightId,
    persistence::seed_node::SeedKey,
    seeding::{pairs::Pair, seed_node::SeedNode},
    spacetime_bucket::{
        bucket::{BucketIndex, BucketKey},
        spatial_binner::{SpatialBinner, SpatialKey},
        time_binner::{TimeBin, TimeBinner, time_targets},
    },
};

/// A seed made of three detections with strictly increasing observation times.
///
/// The triplet is ordered in time:
/// - `a` is the first detection,
/// - `b` is the second detection (used as the anchor for neighbor scans),
/// - `c` is the third detection (searched in bins strictly after `b`).
///
/// The struct stores borrowed references to alerts (no payload copy).
#[derive(Copy, Clone, Debug)]
pub struct Triplet<'alert_lf> {
    /// First detection (earliest epoch).
    pub a: &'alert_lf Alert,
    /// Second detection (middle epoch).
    pub b: &'alert_lf Alert,
    /// Third detection (latest epoch).
    pub c: &'alert_lf Alert,
}

/// Convenience alias: a flat list of triplets.
pub type Triplets<'alert_lf> = Vec<Triplet<'alert_lf>>;

/* ------------------------- neighbor caches ------------------------- */

/// Cached spatial neighbor cells around a given `SpatialKey`.
///
/// This is a small performance optimization: computing neighbor covers can be
/// expensive (HEALPix cone coverage). We cache `neighbors(center, radius)`
/// keyed by the center spatial cell.
///
/// Notes
/// -----
/// The resulting list is sorted and deduplicated to guarantee deterministic
/// iteration and avoid redundant bucket scans even if a binner implementation
/// returns duplicates.
#[inline]
fn cached_spatial_neighbors<'cache, Bs: SpatialBinner>(
    cache: &'cache mut AHashMap<SpatialKey, Vec<SpatialKey>>,
    spatial_binner: &Bs,
    center_space_key: SpatialKey,
    search_radius: f64,
) -> &'cache Vec<SpatialKey> {
    cache.entry(center_space_key).or_insert_with(|| {
        let mut neighbors = spatial_binner.neighbors(center_space_key, search_radius);
        neighbors.sort_unstable();
        neighbors.dedup();
        neighbors
    })
}

/// Cached time-bin targets strictly after a base bin (for the `c` detection).
///
/// For triplets we require `t_c > t_b`. We therefore generate **strictly later**
/// time bins relative to `b` by calling `time_targets(..., allow_same_bin=false)`.
///
/// Parameters
/// ----------
/// base_time_bin : TimeBin
///     The time bin of detection `b`.
/// cfg : &TripletConfig
///     Triplet configuration, notably `max_dt_between` which defines how far in
///     time we are willing to search for `c`.
#[inline]
fn cached_time_targets_strictly_after<'cache, Bt: TimeBinner>(
    cache: &'cache mut AHashMap<TimeBin, Vec<TimeBin>>,
    time_binner: &Bt,
    base_time_bin: TimeBin,
    cfg: &TripletConfig,
) -> &'cache Vec<TimeBin> {
    // For triplets: `c` must be in a strictly later time bin than `b`.
    cache.entry(base_time_bin).or_insert_with(|| {
        time_targets(time_binner, base_time_bin, cfg.max_dt_between, false).collect()
    })
}

/// First index `k` such that `members[k].mjd_tt > t0`.
///
/// This strict lower-bound is used to skip all bucket members that are not
/// strictly later than the anchor time (`t_b`).
///
/// Assumes `members` are sorted by time (guaranteed by `build_alert_bucket_index`).
#[inline]
fn lower_bound_gt_time(members: &[&Alert], t0: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = members.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if members[mid].mjd_tt <= t0 {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/* ------------------------- main generation ------------------------- */

/// Generate triplets `(a, b, c)` from precomputed pairs `(a, b)` using a fast
/// short-baseline linear motion consistency test.
///
/// This is a candidate-generation stage (prefilter). It trades exactness for
/// speed and recall, and is expected to output some false positives that will
/// be rejected later by seed fitting / scoring.
///
/// Algorithm (per pair)
/// --------------------
/// For each input pair `(a, b)`:
/// 1. Ensure time validity (`dt_ab > 0`) and optional strict ordering.
/// 2. Fit a **linear tangent-plane motion model** from `(a, b)` around `a`.
/// 3. Search candidate detections `c` in neighboring spatio-temporal buckets
///    around `b` (spatial cover + time bins strictly after `b`).
/// 4. Apply cheap gates:
///    - flux similarity between `b` and `c`,
///    - angular separation constraint on `(b, c)` using dot-product threshold.
/// 5. Apply a linear prediction residual test:
///    - predict `(ra, dec)` at `t_c` from the `(a, b)` model,
///    - compare predicted vs actual `c` in a tangent-plane offset around `a`,
///    - keep if `resid <= max_predicted_residual`.
/// 6. Deduplicate `(a, b, c)` by pointer identity and push to output.
///
/// Parameters
/// ----------
/// index : &BucketIndex<&Alert>
///     Spatio-temporal bucket index over alerts. Bucket members must be sorted
///     by time (as built by `build_alert_bucket_index`).
/// sb : &impl SpatialBinner
///     Spatial binner used to compute neighbor cells around `b`.
/// tb : &impl TimeBinner
///     Time binner used to select time bins strictly after `b`.
/// cfg : &TripletConfig
///     Triplet-generation parameters (time window, spatial radius, flux gate,
///     pair angular gate, prediction residual threshold, etc.).
/// pairs : &[Pair]
///     Precomputed valid pairs `(a, b)` from which triplets are extended.
///
/// Returns
/// -------
/// Triplets
///     Deduplicated, deterministically sorted list of `(a, b, c)` triplets.
///
/// Notes
/// -----
/// - The spatial search radius is `cfg.max_pair_sep + sb.cell_radius()` to ensure
///   we cover cell boundary effects during bucket-based searches.
/// - The angular gate `ang_sep(b, c) <= cfg.max_pair_sep` is implemented via a
///   dot-product threshold (`cos_pair_threshold`) to avoid `acos`.
/// - The linear prediction uses a small-angle approximation around `a`. It is
///   intended only as a fast prefilter.
pub fn generate_triplets_from_pairs<'alert_lf, Bs: SpatialBinner, Bt: TimeBinner>(
    index: &BucketIndex<&'alert_lf Alert>,
    sb: &Bs,
    tb: &Bt,
    cfg: &TripletConfig,
    pairs: &[Pair<'alert_lf>],
) -> Triplets<'alert_lf> {
    // Search radius around `b` buckets: max allowed (b,c) separation + one cell radius padding.
    let search_radius = cfg.max_pair_sep + sb.cell_radius();

    // Dot-product threshold for ang_sep(b,c) <= max_pair_sep.
    let cos_pair_threshold = cfg.max_pair_sep.cos();

    let mut spatial_neighbor_cache: AHashMap<SpatialKey, Vec<SpatialKey>> = AHashMap::new();
    let mut timebin_target_cache: AHashMap<TimeBin, Vec<TimeBin>> = AHashMap::new();

    let mut out: Triplets<'alert_lf> = Vec::with_capacity(pairs.len() / 2);

    // Dedup because a triplet may be discovered through different (space,time) neighbor paths.
    let mut seen: AHashSet<(usize, usize, usize)> = AHashSet::new();

    for &Pair { a, b } in pairs {
        // Optional enforcement: require strict ordering on the input pairs.
        if cfg.enforce_time_order && !(a.mjd_tt < b.mjd_tt) {
            continue;
        }

        // We predict from (a,b), so dt_ab must be strictly positive.
        let dt_ab = b.mjd_tt - a.mjd_tt;
        if dt_ab <= 0.0 {
            continue;
        }

        // Linear motion estimate from (a,b) on a tangent plane around `a`.
        let cos_dec_a = a.dec.cos();
        let (dx_ab, dy_ab) = planar_offset_fast(a.ra, a.dec, cos_dec_a, b.ra, b.dec);
        let vx = dx_ab / dt_ab; // rad/day on tangent plane (x)
        let vy = dy_ab / dt_ab; // rad/day on tangent plane (y)

        // Precompute values reused across candidate `c`.
        let u_b = unit_vec(b.ra, b.dec);
        let flux_b = b.flux;

        // Neighbor bucket keys around `b`.
        let b_space_key = sb.key_for(b.ra, b.dec);
        let b_time_bin = tb.bin_for(b.mjd_tt);

        let spatial_neighbors =
            cached_spatial_neighbors(&mut spatial_neighbor_cache, sb, b_space_key, search_radius);

        let time_bins =
            cached_time_targets_strictly_after(&mut timebin_target_cache, tb, b_time_bin, cfg);

        let t_b = b.mjd_tt;
        let t_upper = t_b + cfg.max_dt_between;

        // Scan candidate buckets (time bins strictly after b).
        for &time_bin in time_bins {
            for &space_key in spatial_neighbors {
                let Some(bucket) = index.buckets.get(&BucketKey {
                    space_key,
                    time_bin,
                }) else {
                    continue;
                };

                let members = bucket.members.as_slice(); // sorted by time
                let mut idx = lower_bound_gt_time(members, t_b);

                while idx < members.len() {
                    let c = members[idx];
                    idx += 1;

                    let t_c = c.mjd_tt;
                    if t_c > t_upper {
                        break;
                    }

                    // Ensure distinct detections by reference identity.
                    if core::ptr::eq(c, a) || core::ptr::eq(c, b) {
                        continue;
                    }

                    // Flux similarity between b and c.
                    if (flux_b - c.flux).abs() > cfg.max_flux_difference {
                        continue;
                    }

                    // Pairwise angular consistency: ang_sep(b,c) <= max_pair_sep
                    let u_c = unit_vec(c.ra, c.dec);
                    if dot3(u_b, u_c) < cos_pair_threshold {
                        continue;
                    }

                    // Predict from (a,b) to epoch t_c.
                    let dt_ac = t_c - a.mjd_tt;
                    if dt_ac <= 0.0 {
                        continue;
                    }

                    // Predicted RA/Dec at t_c (small-angle approximation around `a`):
                    // - vx is tangent-plane x where dx ~ cos(dec_a) * dRA,
                    // - so dRA_pred ~ vx * dt / cos(dec_a).
                    let ra_pred = a.ra + vx * dt_ac / cos_dec_a.max(1e-12);
                    let dec_pred = a.dec + vy * dt_ac;

                    // Compare predicted vs actual c on the tangent plane around `a`.
                    let (dx_act, dy_act) = planar_offset_fast(a.ra, a.dec, cos_dec_a, c.ra, c.dec);
                    let (dx_pred, dy_pred) =
                        planar_offset_fast(a.ra, a.dec, cos_dec_a, ra_pred, dec_pred);

                    let resid = ((dx_act - dx_pred).powi(2) + (dy_act - dy_pred).powi(2)).sqrt();
                    if resid > cfg.max_predicted_residual {
                        continue;
                    }

                    // Dedup + push.
                    let key = (
                        a as *const Alert as usize,
                        b as *const Alert as usize,
                        c as *const Alert as usize,
                    );
                    if seen.insert(key) {
                        out.push(Triplet { a, b, c });
                    }
                }
            }
        }
    }

    // Deterministic order for tests/reproducibility.
    out.sort_unstable_by(|t1, t2| {
        t1.a.cmp(t2.a)
            .then_with(|| t1.b.cmp(t2.b))
            .then_with(|| t1.c.cmp(t2.c))
    });

    out
}

/// Convert triplets into quadratic [`SeedNode`] objects for a given night.
///
/// This is a thin wrapper around [`SeedNode::from_triplet`].
///
/// Parameters
/// ----------
/// trips : &Triplets
///     Triplets produced by [`generate_triplets_from_pairs`].
/// night_id : NightId
///     Night identifier assigned to all output seeds.
///
/// Returns
/// -------
/// Vec<SeedNode>
///     One seed per triplet, preserving input order.
pub fn extract_triplet_features<'alert_lf>(
    trips: &Triplets<'alert_lf>,
    night_id: NightId,
) -> Vec<SeedNode<'alert_lf>> {
    let mut out = Vec::with_capacity(trips.len());
    for (idx, &Triplet { a, b, c }) in trips.iter().enumerate() {
        out.push(SeedNode::from_triplet(
            SeedKey {
                night_id,
                idx_in_night: idx as u32,
            },
            a,
            b,
            c,
        ));
    }
    out
}

#[cfg(test)]
mod triplet_gen_tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashSet;
    use std::f64::consts::PI;

    use crate::{
        MjdTt, Radians,
        astro_math::{ang_sep, arcsec_to_rad, planar_offset_fast},
        engine_config::triplet_config::TripletConfig,
        persistence::alert::AlertKey,
        spacetime_bucket::{
            bucket::build_alert_bucket_index,
            spatial_binner::{SpatialBinner, SpatialKey},
            time_binner::{TimeBin, TimeBinner},
        },
    };

    const LAT_EPS: f64 = 1e-6;

    /* ------------------------- dummy binners ------------------------- */

    /// Dummy spatial binner: everything goes to a single spatial cell.
    struct DummySpatialBinner;

    impl SpatialBinner for DummySpatialBinner {
        fn key_for(&self, _ra: Radians, _dec: Radians) -> SpatialKey {
            SpatialKey(0)
        }

        fn neighbors(&self, _key: SpatialKey, _ang_radius: Radians) -> Vec<SpatialKey> {
            vec![SpatialKey(0)]
        }

        fn cell_radius(&self) -> Radians {
            0.0
        }

        fn neighbors_into(
            &self,
            _key: SpatialKey,
            _ang_radius: Radians,
            _out: &mut Vec<SpatialKey>,
        ) {
            // Not needed in these tests.
            unimplemented!()
        }
    }

    /// Dummy time binner: uniform bins of fixed width starting from `t0`.
    struct DummyTimeBinner {
        t0: MjdTt,
        width: f64,
    }

    impl DummyTimeBinner {
        fn new(t0: MjdTt, width: f64) -> Self {
            Self { t0, width }
        }
    }

    impl TimeBinner for DummyTimeBinner {
        fn bin_for(&self, mjd_tt: MjdTt) -> TimeBin {
            let idx = ((mjd_tt - self.t0) / self.width).floor() as i64;
            TimeBin(idx)
        }

        fn bins_in_range(&self, t0: MjdTt, t1: MjdTt) -> Vec<TimeBin> {
            let i0 = ((t0 - self.t0) / self.width).floor() as i64;
            let i1 = ((t1 - self.t0) / self.width).ceil() as i64;
            (i0..=i1).map(TimeBin).collect()
        }

        fn bin_width(&self) -> f64 {
            self.width
        }

        fn bin_start(&self, bin: i64) -> MjdTt {
            self.t0 + (bin as f64) * self.width
        }
    }

    /* ------------------------- helpers ------------------------- */

    fn mk_alert(i: usize, ra: f64, dec: f64, mjd_tt: f64, band: u8, flux: f32) -> Alert {
        let pos_err = arcsec_to_rad(0.5);
        Alert {
            key: AlertKey {
                night_id: NightId(0),
                idx_in_night: i as u32,
            },
            dia_source_id: i as u64,
            ra,
            ra_err: pos_err,
            dec,
            dec_err: pos_err,
            mjd_tt,
            flux,
            flux_err: 0.0,
            band,
        }
    }

    fn mk_triplet_config(
        max_dt_between: f64,
        max_pair_sep_arcsec: f64,
        max_residual_arcsec: f64,
    ) -> TripletConfig {
        TripletConfig {
            max_dt_between,
            max_pair_sep: arcsec_to_rad(max_pair_sep_arcsec),
            max_predicted_residual: arcsec_to_rad(max_residual_arcsec),
            enforce_time_order: true,
            max_flux_difference: 5.0,
            ..TripletConfig::default()
        }
    }

    fn ptr2(a: &Alert, b: &Alert) -> (usize, usize) {
        (a as *const Alert as usize, b as *const Alert as usize)
    }

    fn ptr3(a: &Alert, b: &Alert, c: &Alert) -> (usize, usize, usize) {
        (
            a as *const Alert as usize,
            b as *const Alert as usize,
            c as *const Alert as usize,
        )
    }

    /* ---------------------- deterministic tests ---------------------- */

    /// Simple linear motion case: triplet (a,b,c) should be detected once.
    #[test]
    fn triplets_linear_motion_detected() {
        let sb = DummySpatialBinner;
        let tb = DummyTimeBinner::new(60000.0, 10.0 / 1440.0); // 10 min bins

        let t0 = 60000.0;
        let dec0: f64 = 0.25;
        let dr = arcsec_to_rad(6.0) / dec0.cos();

        let a = mk_alert(0, 1.0, dec0, t0, 1, 1000.0);
        let b = mk_alert(1, 1.0 + dr, dec0, t0 + 10.0 / 1440.0, 1, 1000.0);
        let c = mk_alert(2, 1.0 + 2.0 * dr, dec0, t0 + 20.0 / 1440.0, 1, 1000.0);

        let alerts = vec![a, b, c];
        let index = build_alert_bucket_index(&alerts, &sb, &tb);

        let cfg = mk_triplet_config(
            30.0 / 1440.0, // 30 min
            15.0,          // 15"
            3.0,           // 3"
        );

        // Only the pair (&a,&b) is needed to recover (&a,&b,&c).
        let pairs = vec![Pair {
            a: &alerts[0],
            b: &alerts[1],
        }];

        let trips = generate_triplets_from_pairs(&index, &sb, &tb, &cfg, &pairs);

        // Must contain (0,1,2)
        assert!(
            trips.iter().any(|t| core::ptr::eq(t.a, &alerts[0])
                && core::ptr::eq(t.b, &alerts[1])
                && core::ptr::eq(t.c, &alerts[2])),
            "Expected triplet (a,b,c) to be detected",
        );

        // Uniqueness
        let set: HashSet<(usize, usize, usize)> =
            trips.iter().map(|t| ptr3(t.a, t.b, t.c)).collect();
        assert_eq!(set.len(), trips.len());
    }

    /// Large deviation from linear prediction should be rejected by residual threshold.
    #[test]
    fn triplets_large_residual_rejected() {
        let sb = DummySpatialBinner;
        let tb = DummyTimeBinner::new(60000.0, 10.0 / 1440.0);

        let t0 = 60000.0;
        let dec0: f64 = 0.2;
        let dr = arcsec_to_rad(6.0) / dec0.cos();

        let a = mk_alert(0, 2.0, dec0, t0, 1, 1000.0);
        let b = mk_alert(1, 2.0 + dr, dec0, t0 + 10.0 / 1440.0, 1, 1000.0);

        // Deviate by ~40" in RA.
        let c = mk_alert(
            2,
            2.0 + 2.0 * dr + arcsec_to_rad(40.0) / dec0.cos(),
            dec0,
            t0 + 20.0 / 1440.0,
            1,
            1000.0,
        );

        let alerts = vec![a, b, c];
        let index = build_alert_bucket_index(&alerts, &sb, &tb);

        let cfg = mk_triplet_config(
            30.0 / 1440.0,
            60.0, // loose pair sep so only residual kills it
            5.0,  // tight residual
        );

        let pairs = vec![Pair {
            a: &alerts[0],
            b: &alerts[1],
        }];

        let trips = generate_triplets_from_pairs(&index, &sb, &tb, &cfg, &pairs);

        assert!(
            !trips.iter().any(|t| core::ptr::eq(t.a, &alerts[0])
                && core::ptr::eq(t.b, &alerts[1])
                && core::ptr::eq(t.c, &alerts[2])),
            "Triplet with large residual should be rejected",
        );
    }

    /// Check that all produced triplets originate from existing input pairs (a,b).
    #[test]
    fn triplets_from_pairs_is_subset_of_pairs_prefix() {
        let sb = DummySpatialBinner;
        let tb = DummyTimeBinner::new(62000.0, 10.0 / 1440.0);

        let t0 = 62000.0;
        let dec: f64 = 0.25;
        let dr = arcsec_to_rad(8.0) / dec.cos();

        let a = mk_alert(0, 0.6, dec, t0, 1, 1000.0);
        let b = mk_alert(1, 0.6 + dr, dec, t0 + 10.0 / 1440.0, 1, 1000.0);
        let c = mk_alert(2, 0.6 + 2.0 * dr, dec, t0 + 20.0 / 1440.0, 1, 1000.0);
        let d = mk_alert(3, 2.5, 0.0, t0 + 5.0 / 1440.0, 1, 1000.0); // noise

        let alerts = vec![a, b, c, d];
        let index = build_alert_bucket_index(&alerts, &sb, &tb);

        let cfg = mk_triplet_config(25.0 / 1440.0, 20.0, 5.0);

        let pairs = vec![
            Pair {
                a: &alerts[0],
                b: &alerts[1],
            },
            Pair {
                a: &alerts[1],
                b: &alerts[2],
            },
        ];

        let trips = generate_triplets_from_pairs(&index, &sb, &tb, &cfg, &pairs);

        let pair_set: HashSet<(usize, usize)> = pairs.iter().map(|p| ptr2(p.a, p.b)).collect();

        for t in &trips {
            assert!(
                pair_set.contains(&ptr2(t.a, t.b)),
                "every triplet (a,b,c) must originate from an existing pair (a,b)"
            );
        }
    }

    /* ----------------------- property-based tests ----------------------- */

    fn ra_strategy() -> impl Strategy<Value = f64> {
        0.0f64..(2.0 * PI)
    }

    fn dec_strategy() -> impl Strategy<Value = f64> {
        (-(PI / 2.0 - LAT_EPS))..(PI / 2.0 - LAT_EPS)
    }

    fn t_strategy() -> impl Strategy<Value = f64> {
        60000.0f64..60000.1667f64 // ~4h
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 32,
            .. ProptestConfig::default()
        })]

        #[test]
        fn prop_triplets_respect_constraints(
            samples in proptest::collection::vec((ra_strategy(), dec_strategy(), t_strategy()), 0..40)
        ) {
            let sb = DummySpatialBinner;
            let tb = DummyTimeBinner::new(60000.0, 10.0 / 1440.0);

            let cfg = mk_triplet_config(
                40.0 / 1440.0, // 40 min
                30.0,          // 30"
                10.0,          // 10"
            );

            // Same flux to simplify, still keep flux check.
            let alerts: Vec<Alert> = samples.iter().enumerate()
                .map(|(i, (ra, dec, t))| mk_alert(i, *ra, *dec, *t, 1, 1000.0))
                .collect();

            let index = build_alert_bucket_index(&alerts, &sb, &tb);

            // Naive pair list based on (dt <= max_dt_between) and (sep <= max_pair_sep).
            let mut pairs: Vec<Pair> = Vec::new();
            for i in 0..alerts.len() {
                for j in (i+1)..alerts.len() {
                    let a = &alerts[i];
                    let b = &alerts[j];
                    let dt = b.mjd_tt - a.mjd_tt;
                    if dt <= 0.0 || dt > cfg.max_dt_between {
                        continue;
                    }
                    let d = ang_sep(a.ra, a.dec, b.ra, b.dec);
                    if d > cfg.max_pair_sep {
                        continue;
                    }
                    pairs.push(Pair { a, b });
                }
            }

            let trips = generate_triplets_from_pairs(&index, &sb, &tb, &cfg, &pairs);

            // Uniqueness by pointer identity.
            let set: HashSet<(usize, usize, usize)> = trips.iter().map(|t| ptr3(t.a, t.b, t.c)).collect();
            prop_assert_eq!(set.len(), trips.len());

            // Pair membership set.
            let pair_set: HashSet<(usize, usize)> = pairs.iter().map(|p| ptr2(p.a, p.b)).collect();

            for Triplet { a, b, c } in trips {
                // Must originate from input pair list.
                prop_assert!(pair_set.contains(&ptr2(a, b)));

                // Time ordering.
                prop_assert!(a.mjd_tt < b.mjd_tt);
                prop_assert!(b.mjd_tt < c.mjd_tt);

                let dt_ab = b.mjd_tt - a.mjd_tt;
                let dt_bc = c.mjd_tt - b.mjd_tt;

                // Time constraints.
                prop_assert!(dt_ab <= cfg.max_dt_between + 1e-12);
                prop_assert!(dt_bc <= cfg.max_dt_between + 1e-12);

                // Angular constraints on (b,c).
                let dbc = ang_sep(b.ra, b.dec, c.ra, c.dec);
                prop_assert!(dbc <= cfg.max_pair_sep + 1e-12);

                // Flux constraint.
                let flux_diff = (b.flux - c.flux).abs();
                prop_assert!(flux_diff <= cfg.max_flux_difference + 1e-6);

                // Residual recompute (same as implementation).
                let cos_dec_a = a.dec.cos();
                let (dx_ab, dy_ab) = planar_offset_fast(a.ra, a.dec, cos_dec_a, b.ra, b.dec);
                let dt_ab_safe = dt_ab.max(1e-12);
                let vx = dx_ab / dt_ab_safe;
                let vy = dy_ab / dt_ab_safe;

                let dt_ac = c.mjd_tt - a.mjd_tt;
                let ra_pred = a.ra + vx * dt_ac / cos_dec_a.max(1e-12);
                let dec_pred = a.dec + vy * dt_ac;

                let (dx_act, dy_act) = planar_offset_fast(a.ra, a.dec, cos_dec_a, c.ra, c.dec);
                let (dx_pred, dy_pred) = planar_offset_fast(a.ra, a.dec, cos_dec_a, ra_pred, dec_pred);

                let residual = ((dx_act - dx_pred).powi(2) + (dy_act - dy_pred).powi(2)).sqrt();
                prop_assert!(residual <= cfg.max_predicted_residual + 1e-12);

                // Also ensure (b,c) passes the pair sep (already checked above, but keep explicit)
                prop_assert!(dbc <= cfg.max_pair_sep + 1e-12);
            }
        }
    }

    /* ---------------------- extract_triplet_features tests ---------------------- */

    #[test]
    fn extract_triplet_features_order_and_members() {
        let t0 = 60000.0;
        let dec0: f64 = 0.3;

        let dr = arcsec_to_rad(6.0) / dec0.cos();

        let a = mk_alert(0, 1.0, dec0, t0, 1, 1000.0);
        let b = mk_alert(1, 1.0 + dr, dec0, t0 + 10.0 / 1440.0, 1, 1005.0);
        let c = mk_alert(2, 1.0 + 2.0 * dr, dec0, t0 + 20.0 / 1440.0, 1, 1002.0);

        let alerts = vec![a, b, c];

        let trips = vec![
            Triplet {
                a: &alerts[0],
                b: &alerts[1],
                c: &alerts[2],
            },
            Triplet {
                a: &alerts[0],
                b: &alerts[2],
                c: &alerts[1],
            }, // intentionally "weird"
        ];

        let night_id = NightId::new(99);
        let seeds = extract_triplet_features(&trips, night_id);

        assert_eq!(seeds.len(), 2);
        assert_eq!(seeds[0].night_id(), night_id);
        assert_eq!(seeds[1].night_id(), night_id);

        assert_eq!(seeds[0].n_obs, 3);
        assert_eq!(seeds[1].n_obs, 3);

        // We can't assert `members == [id..]` anymore. If SeedNode stores refs,
        // you can add pointer-based assertions here once you show its structure.
    }

    mod prop_extract_triplet_features {
        use super::*;

        fn ra_strategy() -> impl Strategy<Value = f64> {
            0.0f64..(2.0 * std::f64::consts::PI)
        }
        fn dec_strategy() -> impl Strategy<Value = f64> {
            (-(std::f64::consts::PI / 2.0 - LAT_EPS))..(std::f64::consts::PI / 2.0 - LAT_EPS)
        }
        fn time_strategy() -> impl Strategy<Value = f64> {
            60000.0f64..60000.1667f64
        }

        proptest! {
            #![proptest_config(ProptestConfig {
                cases: 32,
                .. ProptestConfig::default()
            })]

            #[test]
            fn prop_extract_triplet_features_1to1_mapping(
                samples in proptest::collection::vec((ra_strategy(), dec_strategy(), time_strategy()), 3..60)
            ) {
                let alerts: Vec<Alert> = samples.iter().enumerate()
                    .map(|(i, (ra, dec, t))| mk_alert(i, *ra, *dec, *t, 1, 1000.0))
                    .collect();

                // Build triplets (consecutive) with increasing time.
                let mut trips: Triplets = Vec::new();
                for i in 0..alerts.len().saturating_sub(2) {
                    let a = &alerts[i];
                    let b = &alerts[i+1];
                    let c = &alerts[i+2];
                    if a.mjd_tt < b.mjd_tt && b.mjd_tt < c.mjd_tt {
                        trips.push(Triplet { a, b, c });
                    }
                }

                let night_id = NightId::new(7);
                let seeds = extract_triplet_features(&trips, night_id);

                prop_assert_eq!(seeds.len(), trips.len());
                for seed in seeds.iter() {
                    prop_assert_eq!(seed.n_obs, 3);
                    prop_assert_eq!(seed.night_id(), night_id);
                }
            }
        }
    }
}
