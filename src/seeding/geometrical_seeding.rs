//! src/seeding/seeds.rs
//!
//! # Seed generation (pairs & triplets) from spatio-temporal buckets
//!
//! This module builds **intra-night seeds** for moving-object linking using
//! precomputed spatio-temporal buckets. It provides two stages:
//! - **Pair generation**: find (a, b) such that `Δt ≤ max_dt` and angular
//!   separation `Δθ ≤ max_sep`.
//! - **Triplet generation**: extend pairs into (a, b, c) using a **linear
//!   motion** test on the local tangent plane around `a` with a configurable
//!   **prediction residual** threshold.
//!
//! ## Data & units
//! - Right ascension/declination are in **radians**.
//! - Times are **MJD(TT)** in **days**.
//! - Flux / magnitude proxy used here is the alert **PSF flux** (e.g. nJy),
//!   compared via an absolute difference threshold.
//!
//! ## Performance notes
//! The hot paths avoid `HashMap` lookups inside inner loops by using
//! **id-indexed tables** (`id == index`). Neighbor lists (spatial and temporal)
//! are **cached per bucket**. Pair and triplet outputs are **deduplicated**.
//!
//! ## Guarantees
//! - All returned pairs satisfy `t_b > t_a` and `ang_sep(a, b) ≤ max_sep`.
//! - All returned triplets satisfy `t_a < t_b < t_c`, pairwise separation
//!   constraints for (a, b) and (b, c), and `prediction_residual ≤ max_pred`.
//!
//! ## See also
//! - [`crate::seeding::space_time_bucket`] for bucket construction.
//! - [`crate::params::FinkFatParams`] for seeding constraints.
//! - Pair stage: [`generate_pairs`], [`generate_pairs_with_progress`].
//! - Triplet stage: [`generate_triplets_from_pairs`],
//!   [`generate_triplets_from_pairs_with_progress`], [`generate_triplets`].

use ahash::AHashMap;
use indicatif::ProgressBar;

use crate::alerts::Alert;
use crate::params::FinkFatParams;
use crate::progress::{maybe_progress_finish, maybe_progress_start, maybe_progress_throttled_set};
use crate::seeding::space_time_bucket::{
    BucketIndex, BucketKey, SpatialBinner, SpatialKey, TimeBin, TimeBinner,
};
use crate::seeding::{AlertId, Pair, Pairs, Triplets};

/* --------------------------- Types --------------------------- */

/* --------------------------- ID → Alert lookup ------------------------ */

/// O(1) lookup table from `AlertId` to `&Alert`.
///
/// This utility is handy when the input `alerts` are not guaranteed to be
/// contiguous by `id == index`. In the hot paths below we assume contiguity
/// and therefore rely on direct `Vec` indexing instead.
pub struct AlertLookup<'a> {
    by_id: AHashMap<AlertId, &'a Alert>,
}
impl<'a> AlertLookup<'a> {
    /// Build an `AlertLookup` by scanning once.
    ///
    /// Complexity is linear in the number of alerts.
    pub fn new(alerts: &'a [Alert]) -> Self {
        let mut by_id = AHashMap::with_capacity(alerts.len());
        for a in alerts {
            by_id.insert(a.id, a);
        }
        Self { by_id }
    }
    /// Return a reference to the alert for `id` or **panic** if missing.
    #[inline]
    pub fn get(&self, id: AlertId) -> &'a Alert {
        self.by_id.get(&id).expect("unknown AlertId in buckets")
    }
}

/// Enumerate **target time bins** starting from `k0`, bounded by `max_dt`.
///
/// If `include_same == false`, the enumeration starts at `k0 + 1`.
///
/// Arguments
/// ---------
/// - `tb`: time binner.
/// - `k0`: base time bin.
/// - `max_dt` (days): inclusive time horizon after `k0`.
/// - `include_same`: whether to include the same bin `k0`.
///
/// Return
/// ------
/// Iterator over `TimeBin` values: `k0 (+0|+1) .. k0 + ceil(max_dt / bin_width)`.
fn time_targets<Bt: TimeBinner>(
    tb: &Bt,
    k0: TimeBin,
    max_dt: f64,
    include_same: bool,
) -> impl Iterator<Item = TimeBin> {
    let w = tb.bin_width().max(1e-12);
    let max_steps = (max_dt / w).ceil().max(0.0) as i64;
    let start = if include_same { 0 } else { 1 };
    (start..=max_steps).map(move |dk| TimeBin(k0.0 + dk))
}

/* --------------------------- Pair generation --------------------- */

#[inline]
fn unit_vec(ra: f64, dec: f64) -> [f64; 3] {
    let c = dec.cos();
    [c * ra.cos(), c * ra.sin(), dec.sin()]
}
#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Return the **first index** `i` such that `times_by_id[ids[i]] > key_time`.
///
/// This is a standard lower-bound search on a **time-sorted** `ids` slice,
/// implemented to avoid allocations and keep the inner loop branch-light.
///
/// Arguments
/// ---------
/// - `ids`: slice of `AlertId` **sorted by time** (ascending).
/// - `key_time` (days): threshold time.
/// - `times_by_id`: dense table `AlertId → mjd_tt`.
#[inline]
fn lower_bound_gt_ids(ids: &[AlertId], key_time: f64, times_by_id: &[f64]) -> usize {
    let (mut lo, mut hi) = (0usize, ids.len());
    while lo < hi {
        let mid = (lo + hi) / 2;
        let t = times_by_id[ids[mid].idx()];
        if t > key_time {
            hi = mid
        } else {
            lo = mid + 1
        }
    }
    lo
}

/// Core pair generation with optional progress reporting.
///
/// This function contains the shared hot path used by both
/// [`generate_pairs`] and [`generate_pairs_with_progress`].
///
/// Arguments
/// ---------
/// - `index`: bucket index (spatial × temporal).
/// - `alerts`: contiguous array where **`alert.id == index`**.
/// - `sb`, `tb`: spatial and time binners used to build `index`.
/// - `params`: seeding constraints (Δt, Δθ, flux difference, …).
/// - `pb_opt`: optional `indicatif::ProgressBar` for throttled updates.
///
/// Return
/// ------
/// `Vec<(AlertId, AlertId)>` where `t_b > t_a` and angular separation
/// `≤ params.pairs.max_sep`.
///
/// Notes
/// -----
/// - Uses id-indexed dense tables for times, flux proxy, and unit vectors.
/// - Caches spatial neighbors per space key and time targets per time bin.
/// - Deduplicates the output at the end (multiple discovery paths may exist).
fn generate_pairs_core<Bs: SpatialBinner, Bt: TimeBinner>(
    index: &BucketIndex,
    alerts: &[Alert],
    sb: &Bs,
    tb: &Bt,
    params: &FinkFatParams,
    pb_opt: Option<&ProgressBar>,
) -> Pairs {
    // Require contiguous ids for O(1) tables.
    debug_assert!(
        alerts.iter().enumerate().all(|(i, a)| a.id.idx() == i),
        "generate_pairs expects contiguous AlertId (id == index)"
    );

    let n = alerts.len();
    let times_by_id: Vec<f64> = alerts.iter().map(|a| a.mjd_tt).collect();
    let mags_by_id: Vec<f32> = alerts.iter().map(|a| a.flux).collect();
    let vecs_by_id: Vec<[f64; 3]> = alerts.iter().map(|a| unit_vec(a.ra, a.dec)).collect();

    // Light caches
    let r_search = params.pairs.max_sep + sb.cell_radius();
    let cos_thresh = params.pairs.max_sep.cos();
    let mut neigh_cache: AHashMap<SpatialKey, Vec<SpatialKey>> = AHashMap::new();
    let mut ttargets_cache: AHashMap<TimeBin, Vec<TimeBin>> = AHashMap::new();

    // Progress
    maybe_progress_start(pb_opt, n as u64, "pairs");
    let mut processed = 0u64;
    let mut last_tick = 0u64;
    const THROTTLE_STEP: u64 = 10_000;

    // Output
    let mut out: Pairs = Vec::with_capacity(n / 8);

    for (key0, bucket0) in &index.buckets {
        // Spatial neighbors (cached + dedup)
        let s_neighs = neigh_cache.entry(key0.space_key).or_insert_with(|| {
            let mut v = sb.neighbors(key0.space_key, r_search);
            v.sort_unstable();
            v.dedup();
            v
        });
        // Time-bin targets (cached)
        let ttargets = ttargets_cache.entry(key0.time_bin).or_insert_with(|| {
            time_targets(
                tb,
                key0.time_bin,
                params.pairs.max_dt,
                params.pairs.allow_same_timebin,
            )
            .collect()
        });

        // Members are time-sorted inside each bucket
        for &a_id in &bucket0.members {
            let t_a = times_by_id[a_id.idx()];
            let m_a = mags_by_id[a_id.idx()];
            let t_max = t_a + params.pairs.max_dt;
            let va = vecs_by_id[a_id.idx()];

            for &tbin in ttargets.iter() {
                for &s_key in s_neighs.iter() {
                    let k = BucketKey {
                        space_key: s_key,
                        time_bin: tbin,
                    };
                    let Some(btgt) = index.buckets.get(&k) else {
                        continue;
                    };
                    let ids = btgt.members.as_slice();

                    // First b with t_b > t_a
                    let mut i = lower_bound_gt_ids(ids, t_a, &times_by_id);
                    // Scan until t_b > t_max
                    while i < ids.len() {
                        let b_id = ids[i];
                        let t_b = times_by_id[b_id.idx()];
                        let m_b = mags_by_id[b_id.idx()];
                        if t_b > t_max {
                            break;
                        }
                        if b_id != a_id {
                            let vb = vecs_by_id[b_id.idx()];
                            // Angular + magnitude cuts
                            if dot3(va, vb) >= cos_thresh
                                && (m_a - m_b).abs() < params.pairs.max_flux_difference
                            {
                                // Time order guaranteed (t_b > t_a)
                                out.push((a_id, b_id).into());
                            }
                        }
                        i += 1;
                    }
                }
            }

            processed += 1;
            maybe_progress_throttled_set(pb_opt, processed, &mut last_tick, THROTTLE_STEP);
        }
    }

    maybe_progress_finish(pb_opt, n as u64, "pairs ✓");

    // Remove duplicates if the same (a, b) came from multiple paths
    out.sort_unstable();
    out.dedup();
    out
}

/* ---------------- Public APIs: thin wrappers ---------------- */

/// Generate **pairs** without progress reporting.
///
/// Arguments
/// ---------
/// - `index`: bucket index (spatial × temporal).
/// - `alerts`: contiguous array where **`alert.id == index`**.
/// - `sb`, `tb`: spatial and time binners used to build `index`.
/// - `params`: seeding constraints (Δt, Δθ, flux difference, …).
/// - `pb_opt`: optional `indicatif::ProgressBar` for throttled updates.
///
/// Return
/// ------
/// `Vec<(AlertId, AlertId)>` where `t_b > t_a` and angular separation
/// `≤ params.pairs.max_sep`.
///
/// Return
/// ------
/// List of `(a, b)` pairs with `t_b > t_a` and `ang_sep(a, b) ≤ max_sep`.
pub fn generate_pairs<Bs: SpatialBinner, Bt: TimeBinner>(
    index: &BucketIndex,
    alerts: &[Alert],
    sb: &Bs,
    tb: &Bt,
    params: &FinkFatParams,
) -> Pairs {
    generate_pairs_core(index, alerts, sb, tb, params, None)
}

/// Generate **pairs** with a throttled `indicatif::ProgressBar`.
///
/// Same as [`generate_pairs`], but emits progress updates suitable for large-N runs.
pub fn generate_pairs_with_progress<Bs: SpatialBinner, Bt: TimeBinner>(
    index: &BucketIndex,
    alerts: &[Alert],
    sb: &Bs,
    tb: &Bt,
    params: &FinkFatParams,
    pb: &ProgressBar,
) -> Pairs {
    generate_pairs_core(index, alerts, sb, tb, params, Some(pb))
}

/* --------------------------- Triplet generation ------------------ */

#[inline]
fn wrap_pm_pi(x: f64) -> f64 {
    let two_pi = std::f64::consts::PI * 2.0;
    let mut y = (x + std::f64::consts::PI) % two_pi;
    if y < 0.0 {
        y += two_pi;
    }
    y - std::f64::consts::PI
}

/// Compute tangent-plane offsets around `(ra0, dec0)` using a precomputed `cos(dec0)`.
///
/// This avoids repeated `cos(dec0)` calls in the hot path.
///
/// Return
/// ------
/// `(dx, dy)` in **radians** on the local tangent plane where:
/// - `dx = wrap_pm_pi(ra − ra0) * cos(dec0)`
/// - `dy = dec − dec0`
#[inline]
fn planar_offset_fast(ra0: f64, dec0: f64, cos_dec0: f64, ra: f64, dec: f64) -> (f64, f64) {
    let dx = wrap_pm_pi(ra - ra0) * cos_dec0;
    let dy = dec - dec0;
    (dx, dy)
}

// ===================== CORE (shared by both public APIs) =====================

/// Core **triplet** generation from pairs with optional progress reporting.
///
/// Builds `(a, b, c)` by scanning candidate `c` in spatio-temporal neighbors
/// of `b`, then applying:
/// 1) pairwise (b, c) angular consistency, and
/// 2) a **linear prediction** check on the tangent plane around `a`
///    based on the motion measured from (a, b), with residual threshold.
///
/// Arguments
/// ---------
/// - `index`, `alerts`, `sb`, `tb`: bucket index and binners.
/// - `params`: triplet constraints (Δt_between, max_pair_sep, max_predicted_residual,
///   enforce_time_order, flux difference).
/// - `pairs`: a previously generated pair list (typically from [`generate_pairs`]).
/// - `pb_opt`: optional progress bar.
///
/// Return
/// ------
/// `Vec<(AlertId, AlertId, AlertId)>` containing triplets `(a, b, c)` with
/// `t_a < t_b < t_c` and residual ≤ `params.triplets.max_predicted_residual`.
fn generate_triplets_from_pairs_core<Bs: SpatialBinner, Bt: TimeBinner>(
    index: &BucketIndex,
    alerts: &[Alert],
    sb: &Bs,
    tb: &Bt,
    params: &FinkFatParams,
    pairs: &[Pair],
    pb_opt: Option<&ProgressBar>,
) -> Triplets {
    // contiguity assumption: id == index
    debug_assert!(
        alerts.iter().enumerate().all(|(i, a)| a.id.idx() == i),
        "generate_triplets_from_pairs expects contiguous AlertId (id == index)"
    );

    // direct tables (id-indexed)
    let times_by_id: Vec<f64> = alerts.iter().map(|a| a.mjd_tt).collect();
    let ra_by_id: Vec<f64> = alerts.iter().map(|a| a.ra).collect();
    let dec_by_id: Vec<f64> = alerts.iter().map(|a| a.dec).collect();
    let cosdec_by_id: Vec<f64> = alerts.iter().map(|a| a.dec.cos()).collect();
    let vecs_by_id: Vec<[f64; 3]> = alerts.iter().map(|a| unit_vec(a.ra, a.dec)).collect();
    let mags_by_id: Vec<f32> = alerts.iter().map(|a| a.flux).collect();

    // precompute keys (pays off when pairs is large)
    let spacekey_by_id: Vec<SpatialKey> = alerts.iter().map(|a| sb.key_for(a.ra, a.dec)).collect();
    let timebin_by_id: Vec<TimeBin> = alerts.iter().map(|a| tb.bin_for(a.mjd_tt)).collect();

    // light caches
    let r_search = params.triplets.max_pair_sep + sb.cell_radius();
    let cos_pair = params.triplets.max_pair_sep.cos();
    let mut neigh_cache: AHashMap<SpatialKey, Vec<SpatialKey>> = AHashMap::new();
    let mut ttargets_cache: AHashMap<TimeBin, Vec<TimeBin>> = AHashMap::new();

    // progress
    maybe_progress_start(pb_opt, pairs.len() as u64, "triplets");
    let mut processed = 0u64;
    let mut last_tick = 0u64;
    const THROTTLE_STEP: u64 = 20_000;

    // output
    let mut out: Triplets = Vec::with_capacity(pairs.len() / 2);

    for &Pair { a: a_id, b: b_id } in pairs {
        let t_a = times_by_id[a_id.idx()];
        let t_b = times_by_id[b_id.idx()];

        let m_b = mags_by_id[b_id.idx()];

        // Enforce time order if requested; otherwise require t_b > t_a for stability
        if t_b.partial_cmp(&t_a) == Some(std::cmp::Ordering::Less)
            || (params.triplets.enforce_time_order
                && t_a.partial_cmp(&t_b) != Some(std::cmp::Ordering::Less))
        {
            processed += 1;
            maybe_progress_throttled_set(pb_opt, processed, &mut last_tick, THROTTLE_STEP);
            continue;
        }

        // Center on b (spatial/time keys)
        let key_b = BucketKey {
            space_key: spacekey_by_id[b_id.idx()],
            time_bin: timebin_by_id[b_id.idx()],
        };

        // Spatial neighbors (cached + dedup)
        let s_neighs = neigh_cache.entry(key_b.space_key).or_insert_with(|| {
            let mut v = sb.neighbors(key_b.space_key, r_search);
            v.sort_unstable();
            v.dedup();
            v
        });

        // Temporal targets after b (cached). We never allow same-bin for c.
        let ttargets = ttargets_cache.entry(key_b.time_bin).or_insert_with(|| {
            time_targets(tb, key_b.time_bin, params.triplets.max_dt_between, false).collect()
        });

        // Linear motion a->b estimated on tangent plane around a
        let ra_a = ra_by_id[a_id.idx()];
        let dec_a = dec_by_id[a_id.idx()];
        let cos_a = cosdec_by_id[a_id.idx()];
        let ra_b = ra_by_id[b_id.idx()];
        let dec_b = dec_by_id[b_id.idx()];
        let (dx_ab, dy_ab) = planar_offset_fast(ra_a, dec_a, cos_a, ra_b, dec_b);
        let dt_ab = (t_b - t_a).max(1e-12);
        let vx = dx_ab / dt_ab;
        let vy = dy_ab / dt_ab;

        // Pairwise b->c angular test
        let vb = vecs_by_id[b_id.idx()];
        let t_bmax = t_b + params.triplets.max_dt_between;

        for &tbin in ttargets.iter() {
            for &s_key in s_neighs.iter() {
                let k = BucketKey {
                    space_key: s_key,
                    time_bin: tbin,
                };
                let Some(bucket_c) = index.buckets.get(&k) else {
                    continue;
                };
                let ids = bucket_c.members.as_slice(); // time-sorted

                // First c with t_c > t_b
                let mut i = lower_bound_gt_ids(ids, t_b, &times_by_id);

                // Scan while t_c ≤ t_b + Δt_between
                while i < ids.len() {
                    let c_id = ids[i];
                    i += 1;

                    if c_id == a_id || c_id == b_id {
                        continue;
                    }
                    let t_c = times_by_id[c_id.idx()];
                    if t_c > t_bmax {
                        break;
                    }

                    // Fast pairwise b<->c angular consistency
                    let vc = vecs_by_id[c_id.idx()];
                    if dot3(vb, vc) < cos_pair {
                        continue;
                    }

                    // Linear prediction at t_c from motion a->b, compare to c
                    let dt_ac = t_c - t_a;
                    if dt_ac <= 0.0 {
                        continue;
                    }

                    let m_c = mags_by_id[c_id.idx()];
                    if (m_b - m_c).abs() > params.triplets.max_flux_difference {
                        continue;
                    }

                    let ra_pred = ra_a + vx * dt_ac / cos_a.max(1e-12);
                    let dec_pred = dec_a + vy * dt_ac;

                    let (dx_pc, dy_pc) = planar_offset_fast(
                        ra_a,
                        dec_a,
                        cos_a,
                        ra_by_id[c_id.idx()],
                        dec_by_id[c_id.idx()],
                    );
                    let (dx_pp, dy_pp) = planar_offset_fast(ra_a, dec_a, cos_a, ra_pred, dec_pred);

                    let resid = ((dx_pc - dx_pp).powi(2) + (dy_pc - dy_pp).powi(2)).sqrt();
                    if resid <= params.triplets.max_predicted_residual {
                        // Temporal order guaranteed: a < b < c
                        out.push((a_id, b_id, c_id).into());
                    }
                }
            }
        }

        processed += 1;
        maybe_progress_throttled_set(pb_opt, processed, &mut last_tick, THROTTLE_STEP);
    }

    maybe_progress_finish(pb_opt, pairs.len() as u64, "triplets ✓");

    out.sort_unstable();
    out.dedup();
    out
}

// ============================ Public thin wrappers ============================

/// Generate **triplets** from a given list of **pairs** (no progress reporting).
///
/// Typical usage is to call [`generate_pairs`] first and pass the result here.
///
/// Arguments
/// ---------
/// - `index`, `alerts`, `sb`, `tb`, `params`: as above.
/// - `pairs`: `(a, b)` candidates to be extended into `(a, b, c)`.
///
/// Return
/// ------
/// Triplet list `(a, b, c)` with `t_a < t_b < t_c`.
pub fn generate_triplets_from_pairs<Bs: SpatialBinner, Bt: TimeBinner>(
    index: &BucketIndex,
    alerts: &[Alert],
    sb: &Bs,
    tb: &Bt,
    params: &FinkFatParams,
    pairs: &[Pair],
) -> Triplets {
    generate_triplets_from_pairs_core(index, alerts, sb, tb, params, pairs, None)
}

/// Generate **triplets** from a given list of **pairs** with a progress bar.
///
/// Same as [`generate_triplets_from_pairs`], but reports progress via
/// `indicatif::ProgressBar`.
pub fn generate_triplets_from_pairs_with_progress<Bs: SpatialBinner, Bt: TimeBinner>(
    index: &BucketIndex,
    alerts: &[Alert],
    sb: &Bs,
    tb: &Bt,
    params: &FinkFatParams,
    pairs: &[Pair],
    pb: &ProgressBar,
) -> Triplets {
    generate_triplets_from_pairs_core(index, alerts, sb, tb, params, pairs, Some(pb))
}

/// Convenience: generate **triplets** directly by internally generating pairs first.
///
/// This is equivalent to:
/// ```ignore
/// let pairs = generate_pairs(index, alerts, sb, tb, params);
/// let trips = generate_triplets_from_pairs(index, alerts, sb, tb, params, &pairs);
/// ```
///
/// Arguments
/// ---------
/// - `index`, `alerts`, `sb`, `tb`, `params`: as above.
///
/// Return
/// ------
/// Triplet list `(a, b, c)` with time-ordered detections.
pub fn generate_triplets<Bs: SpatialBinner, Bt: TimeBinner>(
    index: &BucketIndex,
    alerts: &[Alert],
    sb: &Bs,
    tb: &Bt,
    params: &FinkFatParams,
) -> Triplets {
    let pairs = generate_pairs(index, alerts, sb, tb, params);
    generate_triplets_from_pairs(index, alerts, sb, tb, params, &pairs)
}

#[cfg(test)]
mod geom_seeds_tests {
    use super::*;
    use std::collections::HashSet;
    use std::f64::consts::PI;

    use crate::alerts::Alert;
    use crate::params;
    use crate::seeding::healpix_binners::HealpixBinner;
    use crate::seeding::space_time_bucket::{build_index_from_alerts_precise, BucketKey};
    use crate::seeding::uniform_time_binner::UniformTimeBinner;
    use crate::seeding::Triplet;

    /* ------------------------- helpers ------------------------- */

    fn mk_alert(id: AlertId, ra: f64, dec: f64, mjd_tt: f64, band: u8) -> Alert {
        Alert {
            id,
            dia_source_id: id.idx() as u64,
            ra,
            ra_err: 2.42406840554768e-06, // ~0.5 arcsec in radians
            dec,
            dec_err: 2.42406840554768e-06, // ~0.5 arcsec in radians
            mjd_tt,
            flux: 0.0,
            flux_err: 0.0,
            band,
        }
    }

    #[inline]
    fn arcsec_to_rad(x: f64) -> f64 {
        x * PI / (180.0 * 3600.0)
    }

    #[inline]
    fn wrap_pm_pi(x: f64) -> f64 {
        let mut y = (x + PI) % (2.0 * PI);
        if y < 0.0 {
            y += 2.0 * PI;
        }
        y - PI
    }

    #[inline]
    fn ang_sep(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
        let s1 = dec1.sin();
        let c1 = dec1.cos();
        let s2 = dec2.sin();
        let c2 = dec2.cos();
        let dlon = wrap_pm_pi(ra2 - ra1);
        let cos_d = s1 * s2 + c1 * c2 * dlon.cos();
        (1.0 - cos_d.clamp(-1.0, 1.0)).max(0.0).sqrt().asin() * 2.0
    }

    fn find_alert(alerts: &[Alert], id: AlertId) -> &Alert {
        alerts
            .iter()
            .find(|a| a.id == id)
            .expect("alert id not found")
    }

    /* --------------------- unit tests (deterministic) --------------------- */

    #[test]
    fn pairs_basic_one_pair() {
        // Binning
        let sb = HealpixBinner::new(10); // NSIDE=1024
        let tb = UniformTimeBinner::new(60000.0, 10.0 / 1440.0); // 10 min

        // Two alerts ~5 arcsec apart and 8 min apart -> should match (Δt<=10 min, sep<=10")
        let t0 = 60000.10;
        let a1 = mk_alert(0_u32.into(), 1.0, 0.2, t0, 1);
        let a2 = mk_alert(
            1_u32.into(),
            1.0 + arcsec_to_rad(5.0) / 0.2_f64.cos(),
            0.2,
            t0 + 8.0 / 1440.0,
            1,
        );

        // A distant outlier (must not match)
        let a3 = mk_alert(2_u32.into(), 2.0, -0.3, t0 + 5.0 / 1440.0, 1);

        let alerts = vec![a1.clone(), a2.clone(), a3.clone()];
        let index = build_index_from_alerts_precise(&alerts, &sb, &tb);

        let params = FinkFatParams::builder()
            .pairs(|p| {
                p.max_dt(10.0 / 1440.0) // 10 min
                    .max_sep(arcsec_to_rad(10.0)) // 10"
                    .allow_same_timebin(false)
                    .max_flux_difference(5.0) // large
            })
            .build()
            .unwrap();

        let pairs = generate_pairs(&index, &alerts, &sb, &tb, &params);

        assert!(
            pairs.contains(&(0_u32.into(), 1_u32.into()).into())
                || pairs.contains(&(1_u32.into(), 2_u32.into()).into())
        );
        assert!(!pairs
            .iter()
            .any(|&Pair { a, b }| (a == 2_u32.into() || a == 1_u32.into()) && b == 3_u32.into()));
        // Uniqueness
        let set: HashSet<_> = pairs.iter().collect();
        assert_eq!(set.len(), pairs.len());
    }

    #[test]
    fn pairs_same_timebin_behavior() {
        let sb = HealpixBinner::new(9);
        let tb = UniformTimeBinner::new(60000.0, 20.0 / 1440.0); // 20 min bins

        let t0 = 60000.25;
        // Two alerts in the same time bin (Δt = 5 min < 20 min)
        let a1 = mk_alert(0_u32.into(), 1.5, 0.1, t0, 1);
        let a2 = mk_alert(
            1_u32.into(),
            1.5 + arcsec_to_rad(4.0) / 0.1_f64.cos(),
            0.1,
            t0 + 5.0 / 1440.0,
            1,
        );

        let alerts = vec![a1.clone(), a2.clone()];
        let index = build_index_from_alerts_precise(&alerts, &sb, &tb);

        let params = FinkFatParams::builder()
            .pairs(|p| {
                p.max_dt(30.0 / 1440.0) // 30 min
                    .max_sep(arcsec_to_rad(8.0)) // 8"
                    .max_flux_difference(5.0) // large
                    .allow_same_timebin(false)
            })
            .build()
            .unwrap();

        // Not allowed to match within the same time bin -> expect no pairs
        let pairs_no_same = generate_pairs(&index, &alerts, &sb, &tb, &params);

        assert!(pairs_no_same.is_empty());

        let params = FinkFatParams::builder()
            .pairs(|p| {
                p.max_dt(30.0 / 1440.0) // 30 min
                    .max_sep(arcsec_to_rad(8.0)) // 8"
                    .max_flux_difference(5.0) // large
                    .allow_same_timebin(true)
            })
            .build()
            .unwrap();

        // Allowed -> the pair should appear
        let pairs_same = generate_pairs(&index, &alerts, &sb, &tb, &params);

        assert_eq!(pairs_same.len(), 1);
        let (i, j) = pairs_same[0].into();
        assert!(
            (i == 0_u32.into() && j == 1_u32.into()) || (i == 1_u32.into() && j == 0_u32.into())
        );
    }

    #[test]
    fn triplets_linear_motion_detected() {
        let sb = HealpixBinner::new(10);
        let tb = UniformTimeBinner::new(60000.0, 10.0 / 1440.0); // 10 min

        let t0 = 60000.0;
        // Linear motion: ~6" every 10 min along RA (tangent plane)
        let dec0: f64 = 0.25;
        let dr = arcsec_to_rad(6.0) / dec0.cos();

        let a = mk_alert(0_u32.into(), 1.0, dec0, t0, 1);
        let b = mk_alert(1_u32.into(), 1.0 + dr, dec0, t0 + 10.0 / 1440.0, 1);
        let c = mk_alert(2_u32.into(), 1.0 + 2.0 * dr, dec0, t0 + 20.0 / 1440.0, 1);

        let alerts = vec![a.clone(), b.clone(), c.clone()];
        let index = build_index_from_alerts_precise(&alerts, &sb, &tb);

        let params = FinkFatParams::builder()
            .triplets(|p| {
                p.max_dt_between(30.0 / 1440.0) // 30 min
                    .max_pair_sep(arcsec_to_rad(15.0)) // 15"
                    .max_predicted_residual(arcsec_to_rad(3.0)) // 3"
                    .enforce_time_order(true)
                    .max_flux_difference(5.0) // large
            })
            .build()
            .unwrap();

        let triplets = generate_triplets(&index, &alerts, &sb, &tb, &params);

        // Expect (0,1,2)
        assert!(triplets.contains(&(0_u32.into(), 1_u32.into(), 2_u32.into()).into()));
        // Uniqueness
        let set: HashSet<_> = triplets.iter().collect();
        assert_eq!(set.len(), triplets.len());
    }

    #[test]
    fn triplets_large_residual_rejected() {
        let sb = HealpixBinner::new(10);
        let tb = UniformTimeBinner::new(60000.0, 10.0 / 1440.0);

        let t0 = 60000.0;
        let dec0: f64 = 0.1;
        let dr = arcsec_to_rad(6.0) / dec0.cos();

        let a = mk_alert(0_u32.into(), 2.0, dec0, t0, 1);
        let b = mk_alert(1_u32.into(), 2.0 + dr, dec0, t0 + 10.0 / 1440.0, 1);
        // Third point deviates by ~40" -> residual should exceed 5"
        let c = mk_alert(
            2_u32.into(),
            2.0 + 2.0 * dr + arcsec_to_rad(40.0) / dec0.cos(),
            dec0,
            t0 + 20.0 / 1440.0,
            1,
        );

        let alerts = vec![a.clone(), b.clone(), c.clone()];
        let index = build_index_from_alerts_precise(&alerts, &sb, &tb);

        let params = params::FinkFatParams::builder()
            .triplets(|p| {
                p.max_dt_between(30.0 / 1440.0) // 30 min
                    .max_pair_sep(arcsec_to_rad(60.0)) // 60"
                    .max_predicted_residual(arcsec_to_rad(5.0)) // 5"
                    .enforce_time_order(true)
                    .max_flux_difference(5.0) // large
            })
            .build()
            .unwrap();

        let triplets = generate_triplets(&index, &alerts, &sb, &tb, &params);

        assert!(!triplets.contains(&(0_u32.into(), 1_u32.into(), 2_u32.into()).into()));
    }

    #[test]
    fn pairs_are_kept_when_no_triplet_found() {
        let sb = HealpixBinner::new(9);
        let tb = UniformTimeBinner::new(61000.0, 10.0 / 1440.0); // 10 min

        // Two points compatible in Δt/Δθ, but no third point in the window -> no triplet.
        let t0 = 61000.20;
        let dec = 0.2;
        let a = mk_alert(0_u32.into(), 1.0, dec, t0, 1);
        let b = mk_alert(
            1_u32.into(),
            1.0 + arcsec_to_rad(6.0) / dec.cos(),
            dec,
            t0 + 8.0 / 1440.0,
            1,
        );

        let alerts = vec![a.clone(), b.clone()];
        let index = build_index_from_alerts_precise(&alerts, &sb, &tb);

        let params = FinkFatParams::builder()
            .pairs(|p| {
                p.max_dt(30.0 / 1440.0) // 30 min
                    .max_sep(arcsec_to_rad(20.0)) // 20"
                    .allow_same_timebin(false)
                    .max_flux_difference(5.0) // large
            })
            .triplets(|p| {
                p.max_dt_between(25.0 / 1440.0) // 25 min
                    .max_pair_sep(arcsec_to_rad(20.0)) // 20"
                    .max_predicted_residual(arcsec_to_rad(5.0)) // 5"
                    .enforce_time_order(true)
                    .max_flux_difference(5.0) // large
            })
            .build()
            .unwrap();

        let pairs = generate_pairs(&index, &alerts, &sb, &tb, &params);

        let triplets = generate_triplets_from_pairs(&index, &alerts, &sb, &tb, &params, &pairs);

        // Keep the pair even when no triplet is found
        assert_eq!(triplets.len(), 0);
        assert_eq!(pairs.len(), 1);
        let (i, j) = pairs[0].into();
        assert!(
            (i == 0_u32.into() && j == 1_u32.into()) || (i == 1_u32.into() && j == 0_u32.into())
        );
    }

    #[test]
    fn both_pairs_and_triplet_returned() {
        let sb = HealpixBinner::new(10);
        let tb = UniformTimeBinner::new(61000.0, 10.0 / 1440.0); // 10 min

        let t0 = 61000.0;
        let dec: f64 = 0.15;
        let dr = arcsec_to_rad(6.0) / dec.cos();

        let a = mk_alert(0_u32.into(), 2.0, dec, t0, 1);
        let b = mk_alert(1_u32.into(), 2.0 + dr, dec, t0 + 10.0 / 1440.0, 1);
        let c = mk_alert(2_u32.into(), 2.0 + 2.0 * dr, dec, t0 + 20.0 / 1440.0, 1);

        let alerts = vec![a.clone(), b.clone(), c.clone()];
        let index = build_index_from_alerts_precise(&alerts, &sb, &tb);

        let params = FinkFatParams::builder()
            .pairs(|p| {
                p.max_dt(30.0 / 1440.0) // 30 min
                    .max_sep(arcsec_to_rad(20.0)) // 20"
                    .allow_same_timebin(false)
                    .max_flux_difference(5.0) // large
            })
            .triplets(|p| {
                p.max_dt_between(25.0 / 1440.0) // 25 min
                    .max_pair_sep(arcsec_to_rad(20.0)) // 20"
                    .max_predicted_residual(arcsec_to_rad(5.0)) // 5"
                    .enforce_time_order(true)
                    .max_flux_difference(5.0) // large
            })
            .build()
            .unwrap();

        let pairs = generate_pairs(&index, &alerts, &sb, &tb, &params);

        let triplets = generate_triplets_from_pairs(&index, &alerts, &sb, &tb, &params, &pairs);

        // Triplet found
        assert!(triplets.contains(&(0_u32.into(), 1_u32.into(), 2_u32.into()).into()));
        // Pairs include at least (a,b) and (b,c) (possibly also (a,c) depending on max_dt)
        let mut pair_set = std::collections::HashSet::new();
        for &Pair { a: i, b: j } in &pairs {
            pair_set.insert(if i < j { (i, j) } else { (j, i) });
        }
        assert!(pair_set.contains(&(0_u32.into(), 1_u32.into())));
        assert!(pair_set.contains(&(1_u32.into(), 2_u32.into())));
    }

    #[test]
    fn triplets_from_pairs_is_subset_of_pairs_prefix() {
        // Verify that every returned (a,b,c) originates from a pair (a,b) present in `pairs`.
        let sb = HealpixBinner::new(9);
        let tb = UniformTimeBinner::new(62000.0, 10.0 / 1440.0);

        let t0 = 62000.0;
        let dec: f64 = 0.25;
        let dr = arcsec_to_rad(8.0) / dec.cos();

        let a = mk_alert(0_u32.into(), 0.6, dec, t0, 1);
        let b = mk_alert(1_u32.into(), 0.6 + dr, dec, t0 + 10.0 / 1440.0, 1);
        let c = mk_alert(2_u32.into(), 0.6 + 2.0 * dr, dec, t0 + 20.0 / 1440.0, 1);
        let d = mk_alert(3_u32.into(), 2.5, 0.0, t0 + 5.0 / 1440.0, 1); // noise

        let alerts = vec![a, b, c, d];
        let index = build_index_from_alerts_precise(&alerts, &sb, &tb);

        let params = FinkFatParams::builder()
            .pairs(|p| {
                p.max_dt(30.0 / 1440.0) // 30 min
                    .max_sep(arcsec_to_rad(20.0)) // 20"
                    .allow_same_timebin(false)
                    .max_flux_difference(5.0) // large
            })
            .triplets(|p| {
                p.max_dt_between(25.0 / 1440.0) // 25 min
                    .max_pair_sep(arcsec_to_rad(20.0)) // 20"
                    .max_predicted_residual(arcsec_to_rad(5.0)) // 5"
                    .enforce_time_order(true)
                    .max_flux_difference(5.0) // large
            })
            .build()
            .unwrap();

        let pairs = generate_pairs(&index, &alerts, &sb, &tb, &params);

        let triplets = generate_triplets_from_pairs(&index, &alerts, &sb, &tb, &params, &pairs);

        // Build a set of pairs (canonical order i<j)
        let mut pair_set = std::collections::HashSet::new();
        for &Pair { a: i, b: j } in &pairs {
            pair_set.insert(if i < j { (i, j) } else { (j, i) });
        }

        for &Triplet { a: i, b: j, c: _ } in &triplets {
            // (i,j) must belong to the `pairs` set (by construction)
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            assert!(
                pair_set.contains(&(a, b)),
                "triplet must originate from an existing pair"
            );
        }
    }

    /* --------------------- property tests --------------------- */

    mod geom_seeds_prop {
        use super::*;
        use proptest::prelude::*;

        const LAT_EPS: f64 = 1e-6;

        fn ra_strategy() -> impl Strategy<Value = f64> {
            0.0f64..(2.0 * PI)
        }
        fn dec_strategy() -> impl Strategy<Value = f64> {
            (-(PI / 2.0 - LAT_EPS))..(PI / 2.0 - LAT_EPS)
        }
        fn t_strategy() -> impl Strategy<Value = f64> {
            // ~4h window
            60000.0f64..60000.1667f64 // 0.1667 ~ 4h
        }

        proptest! {
            #![proptest_config(ProptestConfig { cases: 32, .. ProptestConfig::default() })]

            /// All returned pairs respect the Δt and Δθ constraints,
            /// and belong to compatible buckets (spatio-temporal neighborhood).
            #[test]
            fn prop_pairs_respect_constraints_and_buckets(
                triples in proptest::collection::vec((ra_strategy(), dec_strategy(), t_strategy()), 0..120)
            ) {
                let sb = HealpixBinner::new(8);
                let tb = UniformTimeBinner::new(60000.0, 10.0 / 1440.0); // 10 min

                let params = FinkFatParams::builder()
                    .pairs(|p| {
                        p.max_dt(30.0 / 1440.0) // 30 min
                            .max_sep(arcsec_to_rad(20.0)) // 20"
                            .allow_same_timebin(false)
                            .max_flux_difference(5.0) // large
                    })
                    .build()
                    .unwrap();

                let search_radius = params.pairs.max_sep + sb.cell_radius();

                // Build alerts
                let alerts: Vec<Alert> = triples.iter().enumerate().map(|(i, (ra, dec, t))| {
                    mk_alert(i.into(), *ra, *dec, *t, 1)
                }).collect();
                let index = build_index_from_alerts_precise(&alerts, &sb, &tb);

                let pairs = generate_pairs(&index, &alerts, &sb, &tb, &params);

                // Uniqueness
                let set: HashSet<_> = pairs.iter().collect();
                prop_assert_eq!(set.len(), pairs.len());

                // Constraints
                for Pair {a: i, b: j} in pairs {
                    let a = find_alert(&alerts, i);
                    let b = find_alert(&alerts, j);
                    // Time order in the implementation
                    prop_assert!(b.mjd_tt > a.mjd_tt);
                    prop_assert!((b.mjd_tt - a.mjd_tt) <= params.pairs.max_dt);
                    let d = ang_sep(a.ra, a.dec, b.ra, b.dec);
                    prop_assert!(d <= params.pairs.max_sep);

                    // Compatible buckets:
                    let key_a = BucketKey { space_key: sb.key_for(a.ra, a.dec), time_bin: tb.bin_for(a.mjd_tt) };
                    let neighs = sb.neighbors(key_a.space_key, search_radius);
                    let allowed_bins: HashSet<i64> = {
                        let w = tb.bin_width().max(1e-12);
                        let max_steps = (params.pairs.max_dt / w).ceil().max(0.0) as i64;
                        // allow_same_timebin=false → start at +1
                        (1..=max_steps).map(|dk| key_a.time_bin.0 + dk).collect()
                    };
                    let key_b = BucketKey { space_key: sb.key_for(b.ra, b.dec), time_bin: tb.bin_for(b.mjd_tt) };
                    prop_assert!(neighs.into_iter().any(|k| k == key_b.space_key));
                    prop_assert!(allowed_bins.contains(&key_b.time_bin.0));
                }
            }

            /// All triplets respect (a,b) and (b,c) Δt/Δθ constraints,
            /// the linear prediction residual,
            /// and belong to compatible buckets (spatio-temporal neighborhoods).
            #[test]
            fn prop_triplets_respect_constraints_and_buckets(
                triples in proptest::collection::vec((ra_strategy(), dec_strategy(), t_strategy()), 0..100)
            ) {
                let sb = HealpixBinner::new(8);
                let tb = UniformTimeBinner::new(60000.0, 10.0 / 1440.0); // 10 min

                let params = FinkFatParams::builder()
                    .triplets(|p| {
                        p.max_dt_between(40.0 / 1440.0) // 40 min
                            .max_pair_sep(arcsec_to_rad(30.0)) // 30"
                            .max_predicted_residual(arcsec_to_rad(10.0)) // 10"
                            .enforce_time_order(true)
                            .max_flux_difference(5.0) // large
                    })
                    .build()
                    .unwrap();

                let pair_search_radius = params.triplets.max_pair_sep + sb.cell_radius();

                // Build alerts
                let alerts: Vec<Alert> = triples.iter().enumerate().map(|(i, (ra, dec, t))| {
                    mk_alert(i.into(), *ra, *dec, *t, 1)
                }).collect();
                let index = build_index_from_alerts_precise(&alerts, &sb, &tb);

                let triplets = generate_triplets(&index, &alerts, &sb, &tb, &params);
                // Uniqueness
                let set: HashSet<_> = triplets.iter().collect();
                prop_assert_eq!(set.len(), triplets.len());

                for Triplet { a: i, b: j, c: k } in triplets {
                    let a = find_alert(&alerts, i);
                    let b = find_alert(&alerts, j);
                    let c = find_alert(&alerts, k);

                    // Time ordering
                    prop_assert!(a.mjd_tt < b.mjd_tt && b.mjd_tt < c.mjd_tt);

                    // Pairwise Δt/Δθ constraints
                    let dt_ab = b.mjd_tt - a.mjd_tt;
                    let dt_bc = c.mjd_tt - b.mjd_tt;
                    prop_assert!(dt_ab <= params.triplets.max_dt_between && dt_bc <= params.triplets.max_dt_between);

                    let dab = ang_sep(a.ra, a.dec, b.ra, b.dec);
                    let dbc = ang_sep(b.ra, b.dec, c.ra, c.dec);
                    prop_assert!(dab <= params.triplets.max_pair_sep && dbc <= params.triplets.max_pair_sep);

                    // Linear prediction residual (recomputed as in implementation)
                    let (dx_ab, dy_ab) = {
                        let dx = wrap_pm_pi(b.ra - a.ra) * a.dec.cos();
                        let dy = b.dec - a.dec;
                        (dx, dy)
                    };
                    let vx = dx_ab / dt_ab.max(1e-12);
                    let vy = dy_ab / dt_ab.max(1e-12);
                    let dt_ac = c.mjd_tt - a.mjd_tt;
                    let ra_pred  = a.ra + vx * dt_ac / a.dec.cos().max(1e-12);
                    let dec_pred = a.dec + vy * dt_ac;
                    let (dx_pc, dy_pc) = {
                        let dx = wrap_pm_pi(c.ra - a.ra) * a.dec.cos();
                        let dy = c.dec - a.dec;
                        (dx, dy)
                    };
                    let (dx_pp, dy_pp) = {
                        let dx = wrap_pm_pi(ra_pred - a.ra) * a.dec.cos();
                        let dy = dec_pred - a.dec;
                        (dx, dy)
                    };
                    let resid = ((dx_pc - dx_pp).powi(2) + (dy_pc - dy_pp).powi(2)).sqrt();
                    prop_assert!(resid <= params.triplets.max_predicted_residual);

                    // Compatible buckets:
                    // (a,b): b must be in the neighbors and allowed bins of a
                    let key_a = BucketKey { space_key: sb.key_for(a.ra, a.dec), time_bin: tb.bin_for(a.mjd_tt) };
                    let key_b = BucketKey { space_key: sb.key_for(b.ra, b.dec), time_bin: tb.bin_for(b.mjd_tt) };
                    let neighs_ab = sb.neighbors(key_a.space_key, pair_search_radius);
                    prop_assert!(neighs_ab.into_iter().any(|k| k == key_b.space_key));
                    {
                        let w = tb.bin_width().max(1e-12);
                        let max_steps = (params.triplets.max_dt_between / w).ceil().max(0.0) as i64;
                        // allow_same_timebin=false in generate_pairs upstream
                        let allowed_ab: HashSet<i64> = (1..=max_steps).map(|dk| key_a.time_bin.0 + dk).collect();
                        prop_assert!(allowed_ab.contains(&key_b.time_bin.0));
                    }

                    // (b,c): c must be in the neighbors and allowed bins of b
                    let key_b2 = key_b;
                    let key_c = BucketKey { space_key: sb.key_for(c.ra, c.dec), time_bin: tb.bin_for(c.mjd_tt) };
                    let neighs_bc = sb.neighbors(key_b2.space_key, pair_search_radius);
                    prop_assert!(neighs_bc.into_iter().any(|k| k == key_c.space_key));
                    {
                        let w = tb.bin_width().max(1e-12);
                        let max_steps = (params.triplets.max_dt_between / w).ceil().max(0.0) as i64;
                        let allowed_bc: HashSet<i64> = (1..=max_steps).map(|dk| key_b2.time_bin.0 + dk).collect();
                        prop_assert!(allowed_bc.contains(&key_c.time_bin.0));
                    }
                }
            }
        }
    }
}
