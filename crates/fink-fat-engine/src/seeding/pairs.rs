//! Pair generation `(a, b)` for short-baseline intra-night motion filtering.
//!
//! Overview
//! --------
//! This module constructs ordered pairs `(a, b)` of detections that satisfy
//! simple **temporal**, **kinematic**, and **photometric** constraints:
//!
//! - strictly increasing times: `t_b > t_a`,
//! - time separation within `max_dt`,
//! - **angular speed constraint**: `ang_sep(a, b) / (t_b - t_a) ≤ max_angular_speed`,
//! - flux similarity (`|flux(a) − flux(b)| ≤ max_flux_difference`).
//!
//! Compared to a fixed separation cut (`max_sep`), the angular-speed constraint
//! better matches asteroid-like motion: allowed separation scales linearly with
//! Δt (a "wedge" in (Δt, Δθ) space).
//!
//! Algorithmic structure
//! ---------------------
//! - Alerts are indexed in a **spatio-temporal bucket index** (HEALPix × time).
//! - For each anchor alert `a`, only a small neighborhood in space + time is
//!   searched for candidate `b` alerts.
//! - Fast **id-indexed lookup tables** (`time`, `flux`, `unit-vector`) avoid
//!   repeated per-alert work.
//! - Inside each bucket, **binary search** skips all candidates with
//!   `t ≤ t_a`, then a sequential scan tests all constraints.
//!
//! Performance
//! -----------
//! - Tight inner loops with no allocations.
//! - Lookup table hits are O(1).
//! - Buckets + caches drastically reduce the number of candidate comparisons.
//!
//! Invariants
//! -----------
//! - Alerts must satisfy the contiguity constraint:
//!   `alert.id.idx() == index_in_slice`.
//! - Bucket members are sorted by time (guaranteed by bucket construction).

use ahash::AHashMap;

use crate::{
    Alert, AlertId,
    alerts::{AlertStore, lower_bound_gt_ids},
    astro_math::{dot3, unit_vec},
    engine_config::pair_config::PairConfig,
    night_id::NightId,
    seeding::seed_node::SeedNode,
    spacetime_bucket::{
        bucket::{BucketIndex, BucketKey},
        spatial_binner::{SpatialBinner, SpatialKey},
        time_binner::{TimeBin, TimeBinner, time_targets},
    },
};

/// Ordered pair `(a, b)` of alerts forming a minimal intra-night seed.
///
/// Overview
/// --------
/// A pair captures a short-baseline displacement consistent with asteroid-like
/// motion. Only pairs satisfying time-ordering and kinematic/photometric
/// constraints are produced.
///
/// Fields
/// ------
/// * `a` – First detection in time (anchor).
/// * `b` – Second detection in time (candidate, must satisfy `t_b > t_a`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Pair {
    /// Anchor detection `a`.
    pub a: AlertId,
    /// Candidate detection `b` (`t_b > t_a`).
    pub b: AlertId,
}

/// Convenient wrapper: collection of pairs.
pub type Pairs = Vec<Pair>;

/* -------------------------------------------------------------------------- */
/*  Tuple interoperability                                                    */
/* -------------------------------------------------------------------------- */

impl From<(AlertId, AlertId)> for Pair {
    #[inline]
    fn from(t: (AlertId, AlertId)) -> Self {
        Self { a: t.0, b: t.1 }
    }
}
impl From<Pair> for (AlertId, AlertId) {
    #[inline]
    fn from(p: Pair) -> Self {
        (p.a, p.b)
    }
}

/* -------------------------------------------------------------------------- */
/*  Convenience resolvers                                                     */
/* -------------------------------------------------------------------------- */

impl Pair {
    /// Resolve pair ids to borrowed alerts with bounds checking.
    ///
    /// Arguments
    /// ---------
    /// * `store` – Alert store containing a contiguous `alerts` slice.
    ///
    /// Return
    /// ------
    /// * `Some((&Alert, &Alert))` if both ids are valid,  
    /// * `None` otherwise.
    #[inline]
    pub fn resolve(self, store: &AlertStore) -> Option<(&Alert, &Alert)> {
        let a = store.alerts.get(self.a.idx())?;
        let b = store.alerts.get(self.b.idx())?;
        Some((a, b))
    }

    /// Resolve ids using debug assertions followed by unchecked indexing.
    ///
    /// Panics
    /// ------
    /// Only in debug builds if ids are out of bounds.
    ///
    /// Notes
    /// -----
    /// Prefer in hot loops where contiguity `id == index` is guaranteed.
    #[inline]
    pub fn resolve_fast(self, store: &AlertStore) -> (&Alert, &Alert) {
        debug_assert!(self.a.idx() < store.alerts.len());
        debug_assert!(self.b.idx() < store.alerts.len());
        unsafe {
            (
                store.alerts.get_unchecked(self.a.idx()),
                store.alerts.get_unchecked(self.b.idx()),
            )
        }
    }
}

/* ========================================================================== */
/*  Pair generation: lookup tables                                            */
/* ========================================================================== */

/// Dense id-indexed lookup tables for fast per-alert feature access.
///
/// Overview
/// --------
/// Because alert ids are contiguous, table lookups such as:
///
/// ```text
/// time = times_by_id[id]
/// vector = unit_vectors_by_id[id]
/// ```
///
/// are O(1) and extremely cache-friendly.
///
/// Stored quantities
/// -----------------
/// * `mjd_tt` – observation time  
/// * `flux` – photometric flux proxy  
/// * `unit vector` – sky position as `(x, y, z)` for fast dot-product tests
///
/// Invariant
/// ---------
/// The contiguity condition `alert.id.idx() == index_in_slice` must hold.
struct PairLookupTables {
    times_by_id: Vec<f64>,
    fluxes_by_id: Vec<f32>,
    unit_vectors_by_id: Vec<[f64; 3]>,
}

impl PairLookupTables {
    /// Build all lookup tables in a single linear pass.
    ///
    /// Arguments
    /// ---------
    /// * `alerts` – Contiguous slice of alerts sorted by id-index.
    ///
    /// Return
    /// ------
    /// Lookup table bundle (`PairLookupTables`).
    fn build(alerts: &[Alert]) -> Self {
        Self {
            times_by_id: alerts.iter().map(|a| a.mjd_tt).collect(),
            fluxes_by_id: alerts.iter().map(|a| a.flux).collect(),
            unit_vectors_by_id: alerts.iter().map(|a| unit_vec(a.ra, a.dec)).collect(),
        }
    }

    #[inline]
    fn time(&self, id: AlertId) -> f64 {
        self.times_by_id[id.idx()]
    }

    #[inline]
    fn flux(&self, id: AlertId) -> f32 {
        self.fluxes_by_id[id.idx()]
    }

    #[inline]
    fn unit_vector(&self, id: AlertId) -> [f64; 3] {
        self.unit_vectors_by_id[id.idx()]
    }
}

/* ========================================================================== */
/*  Pair generation: anchor context                                           */
/* ========================================================================== */

/// Per-anchor derived values needed when scanning for `(a, b)` pairs.
///
/// Precomputes:
/// - anchor time,
/// - flux,
/// - upper time bound,
/// - 3D unit vector.
///
/// This avoids recomputing these for each neighboring bucket.
struct AnchorContext {
    anchor_id: AlertId,
    anchor_time: f64,
    anchor_flux: f32,
    anchor_time_upper_bound: f64,
    anchor_unit_vector: [f64; 3],
}

impl AnchorContext {
    /// Construct a context for the anchor alert `a`.
    ///
    /// Arguments
    /// ---------
    /// * `anchor_id` – Identifier of `a`.
    /// * `config` – Pair generation configuration.
    /// * `tables` – Lookup tables used to populate derived quantities.
    ///
    /// Return
    /// ------
    /// Populated `AnchorContext`.
    #[inline]
    fn new(anchor_id: AlertId, config: &PairConfig, tables: &PairLookupTables) -> Self {
        let anchor_time = tables.time(anchor_id);
        let anchor_flux = tables.flux(anchor_id);
        let anchor_unit_vector = tables.unit_vector(anchor_id);

        Self {
            anchor_id,
            anchor_time,
            anchor_flux,
            anchor_time_upper_bound: anchor_time + config.max_dt,
            anchor_unit_vector,
        }
    }
}

/* ========================================================================== */
/*  Pair generation: neighbor caches                                          */
/* ========================================================================== */

/// Retrieve and cache the list of spatial neighbor cells.
///
/// Arguments
/// ---------
/// * `cache` – Map `SpatialKey → Vec<SpatialKey>`.
/// * `spatial_binner` – Binner defining spatial tiling and neighbor expansion.
/// * `target_space_key` – Spatial key of the anchor.
/// * `search_radius` – Angular radius (max_sep + cell_radius).
///
/// Return
/// ------
/// Borrowed neighbor list (sorted & deduped).
#[inline]
fn cached_spatial_neighbors<'a, Bs: SpatialBinner>(
    cache: &'a mut AHashMap<SpatialKey, Vec<SpatialKey>>,
    spatial_binner: &'a Bs,
    target_space_key: SpatialKey,
    search_radius: f64,
) -> &'a Vec<SpatialKey> {
    cache.entry(target_space_key).or_insert_with(|| {
        let mut neighbors = spatial_binner.neighbors(target_space_key, search_radius);
        neighbors.sort_unstable();
        neighbors.dedup();
        neighbors
    })
}

/// Retrieve and cache valid time-bin targets for a given anchor.
///
/// Arguments
/// ---------
/// * `cache` – Map `TimeBin → Vec<TimeBin>`.
/// * `time_binner` – Discrete time partitioning.
/// * `base_bin` – Anchor time bin.
/// * `config` – Pair constraints.
///
/// Return
/// ------
/// Borrowed list of time bins reachable within `max_dt`.
#[inline]
fn cached_time_targets<'a, Bt: TimeBinner>(
    cache: &'a mut AHashMap<TimeBin, Vec<TimeBin>>,
    time_binner: &'a Bt,
    base_bin: TimeBin,
    config: &'a PairConfig,
) -> &'a Vec<TimeBin> {
    cache.entry(base_bin).or_insert_with(|| {
        time_targets(
            time_binner,
            base_bin,
            config.max_dt,
            config.allow_same_timebin,
        )
        .collect()
    })
}

/* ========================================================================== */
/*  Pair generation: bucket scanning                                          */
/* ========================================================================== */

/// Scan a single candidate bucket for alert pairs `(a, b)` starting from anchor `a`.
///
/// Kinematic constraint
/// --------------------
/// Let `dt = t_b - t_a` (days, > 0) and `sep = ang_sep(a, b)` (radians).
/// A candidate passes iff:
///
/// `sep / dt ≤ max_angular_speed`  ⇔  `sep ≤ max_angular_speed * dt`
///
/// We implement this via dot products:
/// `cos(sep) = dot(u_a, u_b)` and require:
/// `dot ≥ cos(max_angular_speed * dt)`.
#[inline]
fn scan_bucket_for_anchor(
    bucket_index: &BucketIndex<AlertId>,
    tables: &PairLookupTables,
    anchor: &AnchorContext,
    config: &PairConfig,
    space_key: SpatialKey,
    time_bin: TimeBin,
    pairs_out: &mut Pairs,
) {
    // Locate bucket.
    let Some(bucket) = bucket_index.buckets.get(&BucketKey {
        space_key,
        time_bin,
    }) else {
        return;
    };

    let candidate_ids = bucket.members.as_slice(); // Sorted by time.

    // Direct tables for branch-free access
    let times_by_id = &tables.times_by_id;
    let fluxes_by_id = &tables.fluxes_by_id;
    let vectors_by_id = &tables.unit_vectors_by_id;

    // Binary search: first index with t_candidate > t_anchor.
    let mut index = lower_bound_gt_ids(candidate_ids, anchor.anchor_time, times_by_id);

    // Sequential scan with early stop.
    while index < candidate_ids.len() {
        let candidate_id = candidate_ids[index];
        index += 1;

        // Time constraint upper bound.
        let t = times_by_id[candidate_id.idx()];
        if t > anchor.anchor_time_upper_bound {
            break;
        }

        // Skip self-matching (rare in practice).
        if candidate_id == anchor.anchor_id {
            continue;
        }

        // Flux test.
        let flux_ok = (anchor.anchor_flux - fluxes_by_id[candidate_id.idx()]).abs()
            <= config.max_flux_difference;

        // Angular-speed test.
        let dt = t - anchor.anchor_time; // dt > 0 because of lower_bound_gt_ids
        // Allowed separation = omega * dt (radians). Clamp to π to avoid "always true"
        // behavior for absurdly large omega.
        let max_sep_dt = (config.max_angular_speed * dt).min(std::f64::consts::PI);
        let cos_thresh = max_sep_dt.cos();

        let angular_ok =
            dot3(anchor.anchor_unit_vector, vectors_by_id[candidate_id.idx()]) >= cos_thresh;

        if flux_ok && angular_ok {
            pairs_out.push((anchor.anchor_id, candidate_id).into());
        }
    }
}

/* ========================================================================== */
/*  Main pair generator                                                       */
/* ========================================================================== */

/// Generate all valid `(a, b)` pairs according to the pair-generation rules.
///
/// Constraints
/// -----------
/// Each candidate detection `b` must satisfy:
///
/// 1. `t_b > t_a`  
/// 2. `t_b − t_a ≤ max_dt`  
/// 3. `ang_sep(a, b) / (t_b − t_a) ≤ max_angular_speed`  
/// 4. `|flux(a) − flux(b)| ≤ max_flux_difference`  
///
/// Arguments
/// ---------
/// * `bucket_index` – Global bucket index mapping `(space_key, time_bin)` → members.
/// * `alerts` – Slice of alerts, **must** satisfy `alert.id.idx() == index`.
/// * `spatial_binner` – HEALPix-like spatial partitioning.
/// * `time_binner` – Time discretization strategy.
/// * `config` – Pair-configuration parameters.
///
/// Return
/// ------
/// `Pairs` – Sorted, deduplicated vector of valid `(AlertId, AlertId)`.
///
/// Notes
/// -----
/// Spatial neighbor expansion uses a derived cap on angular separation:
/// `sep_cap = max_angular_speed * max_dt`, so we only visit buckets that could
/// possibly contain valid matches.
pub fn generate_pairs<Bs: SpatialBinner, Bt: TimeBinner>(
    bucket_index: &BucketIndex<AlertId>,
    alerts: &[Alert],
    spatial_binner: &Bs,
    time_binner: &Bt,
    config: &PairConfig,
) -> Pairs {
    // Invariant: id-indexed contiguity.
    debug_assert!(
        alerts.iter().enumerate().all(|(i, a)| a.id.idx() == i),
        "generate_pairs expects contiguous AlertId (id == index)"
    );

    let lookup_tables = PairLookupTables::build(alerts);

    // Spatial search radius: cap + cell radius.
    let sep_cap = (config.max_angular_speed * config.max_dt).max(0.0);
    let spatial_search_radius = sep_cap + spatial_binner.cell_radius();

    let mut spatial_neighbor_cache = AHashMap::<SpatialKey, Vec<SpatialKey>>::new();
    let mut timebin_target_cache = AHashMap::<TimeBin, Vec<TimeBin>>::new();

    let mut pairs_out: Pairs = Vec::with_capacity(alerts.len() / 8);

    // ----------------------------------------------------------------------
    // Main loop: iterate through all buckets and all anchors inside them.
    // ----------------------------------------------------------------------
    for (bucket_key, bucket) in &bucket_index.buckets {
        // Cached neighbors for spatial and temporal axes.
        let spatial_neighbors = cached_spatial_neighbors(
            &mut spatial_neighbor_cache,
            spatial_binner,
            bucket_key.space_key,
            spatial_search_radius,
        );

        let time_targets = cached_time_targets(
            &mut timebin_target_cache,
            time_binner,
            bucket_key.time_bin,
            config,
        );

        // For each anchor alert inside this bucket:
        for &anchor_id in &bucket.members {
            let anchor_ctx = AnchorContext::new(anchor_id, config, &lookup_tables);

            // Explore (space × time) neighbor buckets.
            for &target_time_bin in time_targets {
                for &target_space_key in spatial_neighbors {
                    scan_bucket_for_anchor(
                        bucket_index,
                        &lookup_tables,
                        &anchor_ctx,
                        config,
                        target_space_key,
                        target_time_bin,
                        &mut pairs_out,
                    );
                }
            }
        }
    }

    // Remove duplicates due to spatial/time neighborhood overlaps.
    pairs_out.sort_unstable();
    pairs_out.dedup();
    pairs_out
}

/* ========================================================================== */
/* Feature extraction for seeds                                               */
/* ========================================================================== */

/// Convert pairs `(a, b)` into `SeedNode` objects for a given night.
///
/// Arguments
/// ---------
/// * `store` – AlertStore used for resolving alert ids.
/// * `pairs` – Pair list (already sorted + deduped).
/// * `night_id` – Night identifier attached to all seeds.
/// * `max_speed_rad_per_day` – Optional maximum angular speed (filter).
///
/// Return
/// ------
/// Vector of `SeedNode` created from the input pairs.
///
/// Notes
/// -----
/// - Output vector preserves the order of `pairs`.
/// - If `max_speed_rad_per_day` is provided, seeds faster than that are dropped.
pub fn extract_pair_features(
    store: &AlertStore,
    pairs: &Pairs,
    night_id: NightId,
    max_speed_rad_per_day: Option<f64>,
) -> Vec<SeedNode> {
    let mut out = Vec::with_capacity(pairs.len());
    for &Pair { a: ia, b: ib } in pairs.iter() {
        let alert_a = &store.alerts[ia.idx()];
        let alert_b = &store.alerts[ib.idx()];

        if let Some(seed) = SeedNode::from_pair(night_id, alert_a, alert_b, max_speed_rad_per_day) {
            out.push(seed);
        }
    }
    out
}

#[cfg(test)]
mod pair_gen_tests {
    use super::*;
    use std::collections::HashSet;
    use std::f64::consts::PI;

    use crate::astro_math::{ang_sep, arcsec_to_rad};
    use crate::engine_config::pair_config::PairConfig;
    use crate::spacetime_bucket::bucket::{BucketKey, build_bucket_index};
    use crate::spacetime_bucket::healpix_binner::HealpixBinner;
    use crate::spacetime_bucket::uniform_time_binner::UniformTimeBinner;

    /* ------------------------- helpers ------------------------- */

    /// Construct a minimal `Alert` for testing, with the fields required by
    /// seeding & bucket building.
    fn mk_alert(id: AlertId, ra: f64, dec: f64, mjd_tt: f64, band: u8, flux: f32) -> Alert {
        Alert {
            id,
            dia_source_id: id.idx() as u64,
            ra,
            ra_err: 0.5 * PI / (180.0 * 3600.0), // ~0.5 arcsec in radians
            dec,
            dec_err: 0.5 * PI / (180.0 * 3600.0), // ~0.5 arcsec in radians
            mjd_tt,
            flux,
            flux_err: 0.0,
            band,
        }
    }

    fn find_alert<'a>(alerts: &'a [Alert], id: AlertId) -> &'a Alert {
        alerts
            .iter()
            .find(|a| a.id == id)
            .expect("alert id not found")
    }

    /* ------------------------- unit tests ------------------------- */

    /// Basic sanity: one valid pair in time/angle, plus a distant outlier.
    #[test]
    fn pairs_basic_one_pair() {
        let spatial_binner = HealpixBinner::new(10); // NSIDE=1024
        let time_binner = UniformTimeBinner::new(60000.0, 10.0 / 1440.0); // 10 min bins

        // Two alerts ~5" apart and 8 min apart.
        let t0 = 60000.10;
        let dec0 = 0.2;
        let a1 = mk_alert(0_u32.into(), 1.0, dec0, t0, 1, 1000.0);
        let a2 = mk_alert(
            1_u32.into(),
            1.0 + arcsec_to_rad(5.0) / dec0.cos(),
            dec0,
            t0 + 8.0 / 1440.0,
            1,
            1002.0,
        );

        // A distant outlier (must not match).
        let a3 = mk_alert(2_u32.into(), 2.0, -0.3, t0 + 5.0 / 1440.0, 1, 900.0);

        let alerts = vec![a1.clone(), a2.clone(), a3.clone()];
        let bucket_index = build_bucket_index(&alerts, &spatial_binner, &time_binner);

        // Allow up to ~10" over 10 minutes.
        let max_dt = 10.0 / 1440.0;
        let max_sep = arcsec_to_rad(10.0);
        let omega = max_sep / max_dt;

        let config = PairConfig {
            max_dt,
            max_angular_speed: omega,
            allow_same_timebin: false,
            max_flux_difference: 10.0,
        };

        let pairs = generate_pairs(
            &bucket_index,
            &alerts,
            &spatial_binner,
            &time_binner,
            &config,
        );

        // Exactly one pair: (0,1) in time order.
        assert_eq!(pairs.len(), 1);
        let (i, j) = pairs[0].into();
        assert_eq!(i, 0_u32.into());
        assert_eq!(j, 1_u32.into());
    }

    /// Check behavior of `allow_same_timebin`.
    #[test]
    fn pairs_same_timebin_behavior() {
        let spatial_binner = HealpixBinner::new(9);
        let time_binner = UniformTimeBinner::new(60000.0, 20.0 / 1440.0); // 20 min bins

        let t0 = 60000.25;
        let dec0 = 0.1;

        // Two alerts in the same time bin (Δt = 5 min < 20 min).
        let a1 = mk_alert(0_u32.into(), 1.5, dec0, t0, 1, 1000.0);
        let a2 = mk_alert(
            1_u32.into(),
            1.5 + arcsec_to_rad(4.0) / dec0.cos(),
            dec0,
            t0 + 5.0 / 1440.0,
            1,
            1001.0,
        );

        let alerts = vec![a1.clone(), a2.clone()];
        let bucket_index = build_bucket_index(&alerts, &spatial_binner, &time_binner);

        let max_dt = 5.0 / 1440.0;
        let max_sep = arcsec_to_rad(4.0);
        let omega = max_sep / max_dt * 1.1; // Slightly generous.

        let config_no_same = PairConfig {
            max_dt,
            max_angular_speed: omega,
            allow_same_timebin: false,
            max_flux_difference: 10.0,
        };

        // Not allowed to match within the same time bin -> expect no pairs.
        let pairs_no_same = generate_pairs(
            &bucket_index,
            &alerts,
            &spatial_binner,
            &time_binner,
            &config_no_same,
        );
        assert!(pairs_no_same.is_empty());

        let config_same = PairConfig {
            max_dt,
            max_angular_speed: omega,
            allow_same_timebin: true,
            max_flux_difference: 10.0,
        };

        let pairs_same = generate_pairs(
            &bucket_index,
            &alerts,
            &spatial_binner,
            &time_binner,
            &config_same,
        );

        assert_eq!(pairs_same.len(), 1);
        let (i, j) = pairs_same[0].into();
        assert_eq!(i, 0_u32.into());
        assert_eq!(j, 1_u32.into());
    }

    /// Ensure all pairs respect time ordering, distinct ids,
    /// and the angular-speed constraint.
    #[test]
    fn pairs_time_order_and_distinct_ids() {
        let spatial_binner = HealpixBinner::new(8);
        let time_binner = UniformTimeBinner::new(60000.0, 5.0 / 1440.0); // 5 min bins

        let t0 = 60000.0;
        let dec0 = 0.3;

        let a0 = mk_alert(0_u32.into(), 1.0, dec0, t0, 1, 1000.0);
        let a1 = mk_alert(
            1_u32.into(),
            1.0 + arcsec_to_rad(5.0) / dec0.cos(),
            dec0,
            t0 + 5.0 / 1440.0,
            1,
            1000.0,
        );
        let a2 = mk_alert(
            2_u32.into(),
            1.0 + arcsec_to_rad(9.0) / dec0.cos(),
            dec0,
            t0 + 10.0 / 1440.0,
            1,
            1000.0,
        );

        let alerts = vec![a0, a1, a2];
        let bucket_index = build_bucket_index(&alerts, &spatial_binner, &time_binner);

        // ------------------------------------------------------------------
        // Kinematic limits:
        // - a0 → a1 : 5" in 5 min
        // - a0 → a2 : 9" in 10 min  (worst-case sep/dt)
        // ------------------------------------------------------------------
        let dt01 = 5.0 / 1440.0;
        let dt02 = 10.0 / 1440.0;

        let sep01 = arcsec_to_rad(5.0);
        let sep02 = arcsec_to_rad(9.0);

        let omega_max = (sep01 / dt01).max(sep02 / dt02);

        // Add a small safety margin to avoid floating-point edge failures.
        let omega = 1.1 * omega_max;

        let config = PairConfig {
            max_dt: 15.0 / 1440.0, // 15 min
            max_angular_speed: omega,
            allow_same_timebin: true,
            max_flux_difference: 1e6,
        };

        let pairs = generate_pairs(
            &bucket_index,
            &alerts,
            &spatial_binner,
            &time_binner,
            &config,
        );

        for Pair { a, b } in &pairs {
            let alert_a = find_alert(&alerts, *a);
            let alert_b = find_alert(&alerts, *b);

            // Time ordering and distinct ids.
            assert!(alert_b.mjd_tt > alert_a.mjd_tt, "t_b must be > t_a");
            assert_ne!(a, b, "Pairs must not contain identical ids");

            // Kinematic constraint: Δθ ≤ ω_max · Δt
            let dt = alert_b.mjd_tt - alert_a.mjd_tt;
            let d = ang_sep(alert_a.ra, alert_a.dec, alert_b.ra, alert_b.dec);

            assert!(
                d <= config.max_angular_speed * dt + 1e-15,
                "angular speed violation: Δθ={} rad, Δt={} d, vmax={} rad/d",
                d,
                dt,
                config.max_angular_speed
            );
        }
    }

    /// Check that duplicates are removed when the same pair can be discovered
    /// via different bucket paths.
    #[test]
    fn pairs_duplicates_are_deduplicated() {
        let spatial_binner = HealpixBinner::new(6);
        let time_binner = UniformTimeBinner::new(60000.0, 2.0 / 1440.0); // 2 min bins

        let t0 = 60000.0;
        let dec0 = 0.4;

        // Three detections close in time and space; depending on binning,
        // (0,1) may be reachable through multiple neighbor combinations.
        let a0 = mk_alert(0_u32.into(), 0.5, dec0, t0, 1, 1000.0);
        let a1 = mk_alert(
            1_u32.into(),
            0.5 + arcsec_to_rad(4.0) / dec0.cos(),
            dec0,
            t0 + 1.0 / 1440.0,
            1,
            1005.0,
        );
        let a2 = mk_alert(
            2_u32.into(),
            0.5 + arcsec_to_rad(7.0) / dec0.cos(),
            dec0,
            t0 + 2.0 / 1440.0,
            1,
            1002.0,
        );

        let alerts = vec![a0, a1, a2];
        let bucket_index = build_bucket_index(&alerts, &spatial_binner, &time_binner);

        // Allow ~15" over 10 minutes (generous here).
        let max_dt = 10.0 / 1440.0;
        let max_sep = arcsec_to_rad(15.0);
        let omega = max_sep / max_dt;

        let config = PairConfig {
            max_dt,
            max_angular_speed: omega,
            allow_same_timebin: true,
            max_flux_difference: 10.0,
        };

        let pairs = generate_pairs(
            &bucket_index,
            &alerts,
            &spatial_binner,
            &time_binner,
            &config,
        );

        // All pairs must be unique.
        let set: HashSet<_> = pairs.iter().collect();
        assert_eq!(set.len(), pairs.len());
    }

    /* ---------------- integration-style test ---------------- */

    /// Integration-like test: construct a small field with a "track" of three
    /// detections and a noisy background; check that all expected pairs are present.
    #[test]
    fn pairs_integration_small_track_with_noise() {
        let spatial_binner = HealpixBinner::new(8);
        let time_binner = UniformTimeBinner::new(61000.0, 5.0 / 1440.0); // 5 min bins

        let t0 = 61000.0;
        let dec0: f64 = 0.25;
        let dr = arcsec_to_rad(6.0) / dec0.cos();

        // A simple linear "track" sampled at 5 min.
        let a = mk_alert(0_u32.into(), 2.0, dec0, t0, 1, 1000.0);
        let b = mk_alert(1_u32.into(), 2.0 + dr, dec0, t0 + 5.0 / 1440.0, 1, 1002.0);
        let c = mk_alert(
            2_u32.into(),
            2.0 + 2.0 * dr,
            dec0,
            t0 + 10.0 / 1440.0,
            1,
            1004.0,
        );

        // Some noise around in space and time.
        let n1 = mk_alert(3_u32.into(), 3.0, -0.1, t0 + 3.0 / 1440.0, 1, 500.0);
        let n2 = mk_alert(4_u32.into(), 1.0, 0.8, t0 + 6.0 / 1440.0, 1, 800.0);

        let alerts = vec![a, b, c, n1, n2];
        let bucket_index = build_bucket_index(&alerts, &spatial_binner, &time_binner);

        // We want to allow up to ~20" over 15 minutes.
        let max_dt = 15.0 / 1440.0;
        let max_sep = arcsec_to_rad(20.0);
        let omega = max_sep / max_dt;

        let config = PairConfig {
            max_dt,
            max_angular_speed: omega,
            allow_same_timebin: true,
            max_flux_difference: 100.0,
        };

        let pairs = generate_pairs(
            &bucket_index,
            &alerts,
            &spatial_binner,
            &time_binner,
            &config,
        );

        // Build canonical set (i<j) for easier checking.
        let mut pair_set = HashSet::new();
        for &Pair { a: i, b: j } in &pairs {
            let (x, y) = if i < j { (i, j) } else { (j, i) };
            pair_set.insert((x, y));
        }

        // Expected track pairs.
        assert!(pair_set.contains(&(0_u32.into(), 1_u32.into())));
        assert!(pair_set.contains(&(1_u32.into(), 2_u32.into())));
        assert!(pair_set.contains(&(0_u32.into(), 2_u32.into())));

        // Every pair respects the speed cut.
        for &Pair { a: i, b: j } in &pairs {
            let aa = find_alert(&alerts, i);
            let bb = find_alert(&alerts, j);
            let dt = bb.mjd_tt - aa.mjd_tt;
            let d = ang_sep(aa.ra, aa.dec, bb.ra, bb.dec);
            assert!(d <= config.max_angular_speed * dt + 1e-12);
        }
    }

    /* --------------------- property-based tests --------------------- */

    mod prop_pairs {
        use super::*;
        use proptest::prelude::*;

        const LAT_EPS: f64 = 1e-6;

        fn ra_strategy() -> impl Strategy<Value = f64> {
            0.0f64..(2.0 * PI)
        }

        fn dec_strategy() -> impl Strategy<Value = f64> {
            (-(PI / 2.0 - LAT_EPS))..(PI / 2.0 - LAT_EPS)
        }

        fn time_strategy() -> impl Strategy<Value = f64> {
            60000.0f64..60000.1667f64 // ~4h window
        }

        proptest! {
            #![proptest_config(ProptestConfig {
                cases: 32,
                .. ProptestConfig::default()
            })]

            /// Property-based test:
            ///
            /// For randomly distributed alerts, all returned pairs must:
            /// - respect the Δt and Δθ constraints from `PairConfig`,
            /// - have `t_b > t_a`,
            /// - be compatible with the underlying bucket structure
            ///   (spatial & temporal neighbors).
            #[test]
            fn prop_pairs_respect_constraints_and_buckets(
                triples in proptest::collection::vec((ra_strategy(), dec_strategy(), time_strategy()), 0..120)
            ) {
                let spatial_binner = HealpixBinner::new(8);
                let time_binner = UniformTimeBinner::new(60000.0, 10.0 / 1440.0); // 10 min bins

                // Allow ~20" over 30 minutes.
                let max_dt = 30.0 / 1440.0;
                let max_sep = arcsec_to_rad(20.0);
                let omega = max_sep / max_dt;

                let config = PairConfig {
                    max_dt,
                    max_angular_speed: omega,
                    allow_same_timebin: false,
                    max_flux_difference: 1e6,
                };

                let sep_cap = config.max_angular_speed * config.max_dt;
                let search_radius = sep_cap + spatial_binner.cell_radius();

                // Build alerts with dummy band & flux.
                let alerts: Vec<Alert> = triples.iter().enumerate()
                    .map(|(i, (ra, dec, t))| {
                        mk_alert(i.into(), *ra, *dec, *t, 1, 1000.0)
                    })
                    .collect();

                let bucket_index = build_bucket_index(&alerts, &spatial_binner, &time_binner);

                let pairs = generate_pairs(
                    &bucket_index,
                    &alerts,
                    &spatial_binner,
                    &time_binner,
                    &config,
                );

                // Pairs must be unique.
                let set: HashSet<_> = pairs.iter().collect();
                prop_assert_eq!(set.len(), pairs.len());

                for Pair { a: i, b: j } in pairs {
                    let alert_a = find_alert(&alerts, i);
                    let alert_b = find_alert(&alerts, j);

                    // Time order & Δt constraint.
                    prop_assert!(alert_b.mjd_tt > alert_a.mjd_tt);
                    prop_assert!((alert_b.mjd_tt - alert_a.mjd_tt) <= config.max_dt);

                    let dt = alert_b.mjd_tt - alert_a.mjd_tt;
                    let d = ang_sep(alert_a.ra, alert_a.dec, alert_b.ra, alert_b.dec);
                    prop_assert!(d <= config.max_angular_speed * dt + 1e-12);
                    prop_assert!(d <= sep_cap + 1e-12);

                    // Bucket compatibility: b must lie in a spatial neighbor cell within search_radius
                    // and in a time bin within the allowed range.
                    let key_a = BucketKey {
                        space_key: spatial_binner.key_for(alert_a.ra, alert_a.dec),
                        time_bin: time_binner.bin_for(alert_a.mjd_tt),
                    };
                    let key_b = BucketKey {
                        space_key: spatial_binner.key_for(alert_b.ra, alert_b.dec),
                        time_bin: time_binner.bin_for(alert_b.mjd_tt),
                    };

                    let spatial_neighbors = spatial_binner.neighbors(key_a.space_key, search_radius);
                    prop_assert!(spatial_neighbors.into_iter().any(|k| k == key_b.space_key));

                    // Compute allowed time-bin offsets consistent with max_dt.
                    let bin_width = time_binner.bin_width().max(1e-12);
                    let max_steps = (config.max_dt / bin_width).ceil().max(0.0) as i64;
                    // allow_same_timebin=false → start at +1.
                    let allowed_bins: HashSet<i64> =
                        (1..=max_steps).map(|dk| key_a.time_bin.0 + dk).collect();

                    prop_assert!(allowed_bins.contains(&key_b.time_bin.0));
                }
            }
        }
    }

    /* ---------------------- extract_pair_features tests ---------------------- */

    /// Unit test for `extract_pair_features`:
    /// - preserves input order,
    /// - assigns incremental SeedId starting at 0,
    /// - filters fast seeds when `max_speed_rad_per_day` is set.
    #[test]
    fn extract_pair_features_order_and_speed_filter() {
        use crate::astro_math::arcsec_to_rad;

        // Build a minimal alert set forming two pairs: a→b (slow), b→c (fast).
        let t0 = 60000.0;
        let dec0: f64 = 0.25;

        let slow_sep = arcsec_to_rad(5.0) / dec0.cos(); // ~5" in 5 min
        let fast_sep = arcsec_to_rad(100.0) / dec0.cos(); // ~100" in 5 min

        let a = Alert {
            id: 0_u32.into(),
            dia_source_id: 0,
            ra: 1.0,
            ra_err: arcsec_to_rad(0.5),
            dec: dec0,
            dec_err: arcsec_to_rad(0.5),
            mjd_tt: t0,
            flux: 1000.0,
            flux_err: 0.0,
            band: 1,
        };
        let b = Alert {
            id: 1_u32.into(),
            dia_source_id: 1,
            ra: 1.0 + slow_sep,
            ra_err: arcsec_to_rad(0.5),
            dec: dec0,
            dec_err: arcsec_to_rad(0.5),
            mjd_tt: t0 + 5.0 / 1440.0,
            flux: 1001.0,
            flux_err: 0.0,
            band: 1,
        };
        let c = Alert {
            id: 2_u32.into(),
            dia_source_id: 2,
            ra: 1.0 + slow_sep + fast_sep,
            ra_err: arcsec_to_rad(0.5),
            dec: dec0,
            dec_err: arcsec_to_rad(0.5),
            mjd_tt: t0 + 10.0 / 1440.0,
            flux: 1002.0,
            flux_err: 0.0,
            band: 1,
        };

        let alerts = vec![a.clone(), b.clone(), c.clone()];
        let store = AlertStore::new(t0.floor(), alerts.clone());

        // Build pairs explicitly in order: (a,b), (b,c)
        let pairs = vec![Pair { a: a.id, b: b.id }, Pair { a: b.id, b: c.id }];

        // Extract features with no speed filter → both seeds should be present.
        let seeds_all = extract_pair_features(&store, &pairs, NightId::new(42), None);
        assert_eq!(seeds_all.len(), 2);
        // Members match input pairs.
        assert_eq!(seeds_all[0].members, vec![a.id, b.id]);
        assert_eq!(seeds_all[1].members, vec![b.id, c.id]);

        // Compute speed threshold to keep first pair and drop second.
        // First pair angular displacement over 5 min:
        let dt_day = 5.0 / 1440.0;
        let speed_slow = slow_sep / dt_day;
        let speed_fast = fast_sep / dt_day;
        assert!(speed_fast > speed_slow);

        // Set threshold between slow and fast speeds.
        let vmax = (speed_slow + speed_fast) * 0.5;
        let seeds_filtered = extract_pair_features(&store, &pairs, NightId::new(42), Some(vmax));

        assert_eq!(seeds_filtered.len(), 1);
        assert_eq!(seeds_filtered[0].members, vec![a.id, b.id]);
    }

    mod prop_extract_features {
        use super::*;
        use proptest::prelude::*;

        const LAT_EPS: f64 = 1e-6;

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
            fn prop_extract_pair_features_1to1_mapping(
                triples in proptest::collection::vec((ra_strategy(), dec_strategy(), time_strategy()), 2..40)
            ) {
                // Build alerts.
                let alerts: Vec<Alert> = triples.iter().enumerate().map(|(i, (ra, dec, t))| Alert {
                    id: (i as u32).into(),
                    dia_source_id: i as u64,
                    ra: *ra,
                    ra_err: 1e-6,
                    dec: *dec,
                    dec_err: 1e-6,
                    mjd_tt: *t,
                    flux: 1000.0,
                    flux_err: 0.0,
                    band: 1,
                }).collect();

                let store = AlertStore::new(60000.0, alerts.clone());

                // Build a trivial ordered pair list: consecutive ids with increasing times.
                // Filter to ensure t_b > t_a.
                let mut pairs: Vec<Pair> = Vec::new();
                for i in 0..alerts.len() {
                    for j in (i+1)..alerts.len() {
                        if alerts[j].mjd_tt > alerts[i].mjd_tt {
                            pairs.push(Pair { a: alerts[i].id, b: alerts[j].id });
                        }
                    }
                }

                // Use a very large max_speed to avoid filtering.
                let seeds = extract_pair_features(&store, &pairs, NightId::new(7), Some(f64::INFINITY));

                // 1:1 mapping: each pair produces exactly one seed.
                prop_assert_eq!(seeds.len(), pairs.len());

                // Basic invariants per seed.
                for (k, seed) in seeds.iter().enumerate() {
                    // Members are exactly the pair ids and ordered.
                    let Pair { a, b } = pairs[k];
                    prop_assert_eq!(seed.members.clone(), vec![a, b]);
                    // n_obs is 2 for pairs.
                    prop_assert_eq!(seed.n_obs, 2);
                }
            }
        }
    }
}
