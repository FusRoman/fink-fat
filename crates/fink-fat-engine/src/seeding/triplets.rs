//! Triplet generation `(a, b, c)` with a short-baseline linear motion test.
//!
//! Overview
//! --------
//! This module builds **seed triplets** from previously generated time-ordered
//! pairs. Each triplet `(a, b, c)` corresponds to three detections from the
//! same night with strictly increasing times `t_a < t_b < t_c` and an
//! approximate linear motion on the tangent plane.
//!
//! Starting from a list of pairs `(a, b)` (anchor `a`, middle `b`), we search
//! for a third detection `c` that:
//!
//! - occurs after `b` and within `max_dt_between`,
//! - lies within `max_pair_sep` of `b` (pairwise angular consistency),
//! - has flux similar to `b` (difference ≤ `max_flux_difference`),
//! - is consistent with a linear motion model extrapolated from `(a, b)` on the
//!   tangent plane around `a`, with residual ≤ `max_predicted_residual`.
//!
//! The resulting triplets are later converted into `SeedNode`s which aggregate
//! motion and photometric features for the inter-night linking stage.
//!
//! Geometry and motion model
//! -------------------------
//! - Positions are represented by `(ra, dec)` on the unit sphere.
//! - For each pair `(a, b)`, we compute a **local velocity** on the tangent
//!   plane centered at `a`. The displacement from `a → b` is expressed as
//!   `(dx_ab, dy_ab)` using `planar_offset_fast`.
//! - The tangent-plane velocity `(vx, vy)` is then `dx_ab / dt_ab`, `dy_ab / dt_ab`.
//! - For a candidate `c`, we predict its position at `t_c` from the pair
//!   `(a, b)`, project both predicted and actual positions to the tangent plane
//!   around `a`, and compute the Euclidean residual.
//!
//! Implementation notes
//! --------------------
//! - Uses dense id-indexed lookup tables for constant-time access to per-alert
//!   quantities (time, position, cos(dec), unit vector, flux).
//! - Searches spatio-temporal neighbors of `b` using bucket indices and caches
//!   spatial/time neighbors per key to avoid repeated work across pairs.
//! - Buckets are time-sorted; we binary-search to start at `t_b` and early-exit
//!   when the time window is exceeded.
//! - Output is sorted and deduplicated as a triplet may be discovered via
//!   multiple neighbor buckets (different space/time bucket paths).
//!
//! Error handling and invariants
//! -----------------------------
//! - No panics on normal paths; time ordering and safe divisions are enforced.
//! - The contiguity invariant `alert.id.idx() == position_in_slice` is assumed
//!   (checked via `debug_assert!`) for all id-indexed tables.
//! - Any violation of this invariant is considered a programmer error and
//!   should be caught in debug builds.
//!
//! Performance
//! -----------
//! - Single linear passes to build lookup tables and per-alert bucket keys.
//! - Hot inner loop avoids allocations; residual computation uses tangent-plane
//!   projection for speed.
//! - Spatial/time neighbor caches grow with the number of distinct keys
//!   actually used by the pair list, which is typically much smaller than the
//!   full index.
//! - Clippy-clean, rustfmt-managed formatting; the module is designed to be
//!   amenable to profiling and further optimization if needed.

use ahash::AHashMap;

use crate::{
    Alert, AlertId,
    alerts::{AlertStore, lower_bound_gt_ids},
    astro_math::{dot3, planar_offset_fast, unit_vec},
    engine_config::triplet_config::TripletConfig,
    night_id::NightId,
    seeding::{pairs::Pair, seed_id::SeedId, seed_node::SeedNode},
    spacetime_bucket::{
        bucket::{BucketIndex, BucketKey},
        spatial_binner::{SpatialBinner, SpatialKey},
        time_binner::{TimeBin, TimeBinner, time_targets},
    },
};

/// A seed made of three detections with strictly increasing observation times.
///
/// Triplets are used to estimate local linear motion on a short intra-night
/// baseline and to reject pairs incompatible with a simple trajectory. They
/// form the bridge between the **pair generation** stage and the construction
/// of higher-level `SeedNode`s.
///
/// Fields
/// ------
/// * `a` – First detection in time (anchor, usually the earliest).
/// * `b` – Second detection in time (middle, search center).
/// * `c` – Third detection in time (candidate).
///
/// See also
/// --------
/// * [`SeedNode::from_triplet`] – Converts a triplet into a feature-rich seed.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Triplet {
    /// First detection in time (anchor) — typically the earliest.
    pub a: AlertId,
    /// Second detection in time (middle) — used as the local search center.
    pub b: AlertId,
    /// Third detection in time (candidate) — extrapolated from (a, b).
    pub c: AlertId,
}

/// Collection of triplets returned by the generator.
///
/// Notes
/// -----
/// The vector is sorted and deduplicated before being returned by the
/// generator, so it can be used directly for seeding or serialization.
pub type Triplets = Vec<Triplet>;

/* ---------- Backward compatibility (tuples <-> wrappers) ---------- */

impl From<(AlertId, AlertId, AlertId)> for Triplet {
    /// Convert a raw tuple of three `AlertId`s into a [`Triplet`].
    ///
    /// This is convenient when interfacing with legacy code that represents
    /// triplets as `(a, b, c)` tuples.
    #[inline]
    fn from(t: (AlertId, AlertId, AlertId)) -> Self {
        Self {
            a: t.0,
            b: t.1,
            c: t.2,
        }
    }
}

impl From<Triplet> for (AlertId, AlertId, AlertId) {
    /// Convert a [`Triplet`] back into a `(a, b, c)` tuple.
    ///
    /// Useful when passing triplets to generic utilities or serialization
    /// code that expects tuples instead of dedicated structs.
    #[inline]
    fn from(t: Triplet) -> Self {
        (t.a, t.b, t.c)
    }
}

/* ---------------------- Convenience resolvers --------------------- */

impl Triplet {
    /// Resolve triplet ids to borrowed alerts with bounds checking.
    ///
    /// Arguments
    /// ---------
    /// * `store` – Alert store containing contiguous `alerts`.
    ///
    /// Return
    /// ------
    /// * `Some((&Alert, &Alert, &Alert))` if all ids are valid,
    /// * `None` otherwise.
    ///
    /// Notes
    /// -----
    /// This is the safe, bounds-checked variant. It should be preferred in
    /// tests, debug tooling, or code paths where performance is not critical.
    #[inline]
    pub fn resolve(self, store: &AlertStore) -> Option<(&Alert, &Alert, &Alert)> {
        let a = store.alerts.get(self.a.idx())?;
        let b = store.alerts.get(self.b.idx())?;
        let c = store.alerts.get(self.c.idx())?;
        Some((a, b, c))
    }

    /// Resolve with a debug bound check and then unchecked indexing (fast path).
    ///
    /// Panics
    /// ------
    /// Does not panic in release builds; debug assertions ensure id bounds.
    ///
    /// Notes
    /// -----
    /// Prefer this in tight loops immediately after triplet generation where
    /// bounds are guaranteed by construction (e.g. `generate_triplets_from_pairs`).
    #[inline]
    pub fn resolve_fast(self, store: &AlertStore) -> (&Alert, &Alert, &Alert) {
        debug_assert!(self.a.idx() < store.alerts.len());
        debug_assert!(self.b.idx() < store.alerts.len());
        debug_assert!(self.c.idx() < store.alerts.len());
        unsafe {
            (
                store.alerts.get_unchecked(self.a.idx()),
                store.alerts.get_unchecked(self.b.idx()),
                store.alerts.get_unchecked(self.c.idx()),
            )
        }
    }
}

/* ------------------------------------------------------------------------- */
/* Triplet generation                                                        */
/* ------------------------------------------------------------------------- */

/// Dense id-indexed lookup tables used for triplet generation.
///
/// Overview
/// --------
/// These tables allow constant-time access to per-alert quantities:
///
/// - observation time `mjd_tt`,
/// - sky position `(ra, dec)`,
/// - precomputed `cos(dec)` (used in tangent-plane projections),
/// - unit vector on the unit sphere,
/// - PSF flux (used as a flux/magnitude proxy).
///
/// Invariant
/// ---------
/// Assumes contiguous IDs: `alert.id.idx() == index_in_slice`.
/// This is enforced by a `debug_assert!` in the main generation routine.
struct TripletLookupTables {
    /// `AlertId → mjd_tt` (observation time in days, TT).
    times_by_id: Vec<f64>,

    /// `AlertId → right ascension` (radians).
    ra_by_id: Vec<f64>,

    /// `AlertId → declination` (radians).
    dec_by_id: Vec<f64>,

    /// `AlertId → cos(declination)` for faster tangent-plane projections.
    cos_dec_by_id: Vec<f64>,

    /// `AlertId → unit vector (x, y, z)` on the unit sphere.
    unit_vectors_by_id: Vec<[f64; 3]>,

    /// `AlertId → PSF flux` (used as flux/magnitude proxy).
    fluxes_by_id: Vec<f32>,
}

impl TripletLookupTables {
    /// Build all lookup tables in a single linear pass over the alerts.
    ///
    /// Arguments
    /// ---------
    /// * `alerts` – Slice of alerts, contiguous by `AlertId` index.
    ///
    /// Return
    /// ------
    /// A populated table bundle for triplet generation.
    ///
    /// Notes
    /// -----
    /// The function does no bounds or contiguity checks; these are performed
    /// by the caller. It simply mirrors the `alerts` slice into column-wise
    /// arrays to improve cache locality in the hot loop.
    fn build(alerts: &[Alert]) -> Self {
        // Single-pass map collecting all columns; avoids repeated derived work.
        let times_by_id = alerts.iter().map(|a| a.mjd_tt).collect();
        let ra_by_id = alerts.iter().map(|a| a.ra).collect();
        let dec_by_id = alerts.iter().map(|a| a.dec).collect();
        let cos_dec_by_id = alerts.iter().map(|a| a.dec.cos()).collect();
        let unit_vectors_by_id = alerts.iter().map(|a| unit_vec(a.ra, a.dec)).collect();
        let fluxes_by_id = alerts.iter().map(|a| a.flux).collect();

        Self {
            times_by_id,
            ra_by_id,
            dec_by_id,
            cos_dec_by_id,
            unit_vectors_by_id,
            fluxes_by_id,
        }
    }

    // Small getters: constant-time id-indexed accessors.
    #[inline]
    fn time(&self, id: AlertId) -> f64 {
        self.times_by_id[id.idx()]
    }

    #[inline]
    fn ra(&self, id: AlertId) -> f64 {
        self.ra_by_id[id.idx()]
    }

    #[inline]
    fn dec(&self, id: AlertId) -> f64 {
        self.dec_by_id[id.idx()]
    }

    #[inline]
    fn cos_dec(&self, id: AlertId) -> f64 {
        self.cos_dec_by_id[id.idx()]
    }

    #[inline]
    fn unit_vector(&self, id: AlertId) -> [f64; 3] {
        self.unit_vectors_by_id[id.idx()]
    }

    #[inline]
    fn flux(&self, id: AlertId) -> f32 {
        self.fluxes_by_id[id.idx()]
    }
}

/// Context describing a pair `(anchor, middle)` used to search for candidates.
///
/// Precomputes:
///
/// - tangent-plane velocity `(vx, vy)` from `(a, b)` around `a`,
/// - upper time bound for candidates (`t_b + max_dt_between`),
/// - unit vector of `b` for fast angular checks.
///
/// The context is rebuilt once per input pair and reused while scanning all
/// neighbor buckets for candidate `c`.
struct TripletPairContext {
    /// First detection in the triplet `(a)`.
    anchor_id: AlertId,

    /// Second detection in the triplet `(b)`, also used as search center.
    middle_id: AlertId,

    /// Time of the anchor detection `a` (days).
    anchor_time: f64,

    /// Time of the middle detection `b` (days).
    middle_time: f64,

    /// Flux of the middle detection `b`.
    middle_flux: f32,

    /// Upper bound on valid candidate times: `middle_time + max_dt_between`.
    middle_time_upper_bound: f64,

    /// Right ascension of the anchor detection `a` (radians).
    anchor_ra: f64,

    /// Declination of the anchor detection `a` (radians).
    anchor_dec: f64,

    /// cos(declination) at the anchor detection `a`.
    anchor_cos_dec: f64,

    /// Unit vector of the middle detection `b` (for fast angular tests b–c).
    middle_unit_vector: [f64; 3],

    /// Tangent-plane velocity component along x (radians/day),
    /// estimated from (a, b) on the tangent plane around `a`.
    velocity_x: f64,

    /// Tangent-plane velocity component along y (radians/day),
    /// estimated from (a, b) on the tangent plane around `a`.
    velocity_y: f64,
}

impl TripletPairContext {
    /// Build the context associated with a given pair `(anchor_id, middle_id)`.
    ///
    /// Arguments
    /// ---------
    /// * `anchor_id` – First detection `(a)` of the pair.
    /// * `middle_id` – Second detection `(b)` of the pair.
    /// * `triplet_config` – Triplet search configuration (time window).
    /// * `tables` – Precomputed id-indexed lookup tables.
    ///
    /// Return
    /// ------
    /// Precomputed pair context for scanning candidate `c`.
    ///
    /// Notes
    /// -----
    /// A small floor `1e-12` is used when computing the time difference
    /// `dt_ab` to avoid division by zero in degenerate cases where `t_a` and
    /// `t_b` are extremely close. This has negligible impact on realistic data.
    fn new(
        anchor_id: AlertId,
        middle_id: AlertId,
        triplet_config: &TripletConfig,
        tables: &TripletLookupTables,
    ) -> Self {
        // Gather minimal scalar columns needed for motion and window bounds.
        let anchor_time = tables.time(anchor_id);
        let middle_time = tables.time(middle_id);
        let middle_flux = tables.flux(middle_id);

        let anchor_ra = tables.ra(anchor_id);
        let anchor_dec = tables.dec(anchor_id);
        let anchor_cos_dec = tables.cos_dec(anchor_id);

        let middle_ra = tables.ra(middle_id);
        let middle_dec = tables.dec(middle_id);

        // Displacement from a → b on tangent plane (centered at a).
        let (dx_ab, dy_ab) =
            planar_offset_fast(anchor_ra, anchor_dec, anchor_cos_dec, middle_ra, middle_dec);

        // Robust time difference to avoid division by zero when computing velocity.
        let dt_ab = (middle_time - anchor_time).max(1e-12);
        let velocity_x = dx_ab / dt_ab;
        let velocity_y = dy_ab / dt_ab;

        // Precompute unit vector of b for fast b–c angular checks.
        let middle_unit_vector = tables.unit_vector(middle_id);
        let middle_time_upper_bound = middle_time + triplet_config.max_dt_between;

        Self {
            anchor_id,
            middle_id,
            anchor_time,
            middle_time,
            middle_flux,
            middle_time_upper_bound,
            anchor_ra,
            anchor_dec,
            anchor_cos_dec,
            middle_unit_vector,
            velocity_x,
            velocity_y,
        }
    }
}

/// Retrieve and cache spatial neighbors for triplet generation.
///
/// Arguments
/// ---------
/// * `cache` – Spatial neighbor cache keyed by `SpatialKey`.
/// * `spatial_binner` – Spatial partitioning strategy.
/// * `center_space_key` – Key around which neighbors are requested.
/// * `search_radius` – Angular search radius in radians.
///
/// Return
/// ------
/// Borrowed reference to the cached vector of neighbor `SpatialKey`s.
///
/// Notes
/// -----
/// - Neighbor lists are sorted and deduplicated exactly once per key.
/// - The cache grows monotonically with the number of distinct spatial keys
///   encountered while scanning the pair list.
#[inline]
fn cached_spatial_neighbors_for_triplets<'a, Bs: SpatialBinner>(
    cache: &'a mut AHashMap<SpatialKey, Vec<SpatialKey>>,
    spatial_binner: &'a Bs,
    center_space_key: SpatialKey,
    search_radius: f64,
) -> &'a Vec<SpatialKey> {
    // Insert-once pattern: compute, sort, dedup, then reuse across many pairs.
    cache.entry(center_space_key).or_insert_with(|| {
        let mut neighbors = spatial_binner.neighbors(center_space_key, search_radius);
        neighbors.sort_unstable();
        neighbors.dedup();
        neighbors
    })
}

/// Retrieve and cache temporal neighbors (time bins) for triplet generation.
///
/// In the triplet stage, only strictly later bins are allowed relative to `b`.
///
/// Arguments
/// ---------
/// * `cache` – Time-bin neighbor cache keyed by `TimeBin`.
/// * `time_binner` – Time discretization strategy.
/// * `base_time_bin` – Time bin of the middle detection `b`.
/// * `triplet_config` – Triplet configuration (provides `max_dt_between`).
///
/// Return
/// ------
/// Borrowed reference to the cached vector of neighbor `TimeBin`s.
///
/// Notes
/// -----
/// The `allow_same_bin` flag in [`time_targets`] is hard-coded to `false`:
/// the candidate `c` is enforced to belong to a strictly later time bin than `b`.
#[inline]
fn cached_time_targets_for_triplets<'a, Bt: TimeBinner>(
    cache: &'a mut AHashMap<TimeBin, Vec<TimeBin>>,
    time_binner: &'a Bt,
    base_time_bin: TimeBin,
    triplet_config: &'a TripletConfig,
) -> &'a Vec<TimeBin> {
    // Cache-only-later strategy: c cannot be in the same bin as b.
    cache.entry(base_time_bin).or_insert_with(|| {
        time_targets(
            time_binner,
            base_time_bin,
            triplet_config.max_dt_between,
            false, // never allow same time-bin for c
        )
        .collect()
    })
}

/// Scan a single candidate bucket near the middle detection `b` for valid `c`.
///
/// Hot inner loop
/// --------------
/// - Restrict candidate times: `t_b < t_c ≤ t_b + max_dt_between`.
/// - Enforce angular consistency `(b, c)`: `ang_sep(b, c) ≤ max_pair_sep`.
/// - Enforce flux similarity between `b` and `c`.
/// - Evaluate tangent-plane residual by extrapolating from `(a, b)` to `t_c`.
///
/// Arguments
/// ---------
/// * `bucket_index` – Global bucket index (spatial × temporal).
/// * `tables` – Precomputed id-indexed lookup tables.
/// * `pair_context` – Precomputed context for `(a, b)`.
/// * `triplet_config` – Triplet search configuration.
/// * `spatial_key` – Spatial key of the target bucket.
/// * `time_bin` – Time bin of the target bucket.
/// * `cos_pair_threshold` – `cos(max_pair_sep)`; dot-product threshold for `(b, c)`.
/// * `triplets_out` – Output vector collecting valid triplets.
///
/// Notes
/// -----
/// Buckets are time-sorted by construction; we binary-search the starting index
/// (first candidate with `t_c > t_b`) and early-exit as soon as
/// `t_c > t_b + max_dt_between`.
#[inline]
fn scan_bucket_for_triplets(
    bucket_index: &BucketIndex<AlertId>,
    tables: &TripletLookupTables,
    pair_context: &TripletPairContext,
    triplet_config: &TripletConfig,
    spatial_key: SpatialKey,
    time_bin: TimeBin,
    cos_pair_threshold: f64,
    triplets_out: &mut Triplets,
) {
    // Resolve bucket; skip if absent (no candidates in this space-time cell).
    let bucket_key = BucketKey {
        space_key: spatial_key,
        time_bin,
    };
    let Some(bucket) = bucket_index.buckets.get(&bucket_key) else {
        return;
    };

    // Members sorted by time; use binary search to find first c after b.
    let candidate_ids = bucket.members.as_slice();
    let times_by_id = &tables.times_by_id;

    // Lower-bound search: first index with t > t_b.
    let mut index = lower_bound_gt_ids(candidate_ids, pair_context.middle_time, times_by_id);

    while index < candidate_ids.len() {
        let candidate_id = candidate_ids[index];
        index += 1;

        // Skip if candidate equals either anchor or middle (no duplicates in triplet).
        if candidate_id == pair_context.anchor_id || candidate_id == pair_context.middle_id {
            continue;
        }

        let candidate_time = tables.time(candidate_id);
        if candidate_time > pair_context.middle_time_upper_bound {
            // Time window exceeded: no subsequent candidates in this bucket can match.
            break;
        }

        // Fast angular consistency check between b and c via dot-product threshold.
        let candidate_unit_vector = tables.unit_vector(candidate_id);
        let angular_pair_ok =
            dot3(pair_context.middle_unit_vector, candidate_unit_vector) >= cos_pair_threshold;
        if !angular_pair_ok {
            continue;
        }

        // Time difference between a and c for linear prediction; must be positive.
        let dt_ac = candidate_time - pair_context.anchor_time;
        if dt_ac <= 0.0 {
            // Guard against numeric issues / non-ordered times.
            continue;
        }

        // Flux similarity between b and c.
        let candidate_flux = tables.flux(candidate_id);
        let flux_ok =
            (pair_context.middle_flux - candidate_flux).abs() <= triplet_config.max_flux_difference;
        if !flux_ok {
            continue;
        }

        // Linear motion extrapolation from (a, b) to the time of c on tangent plane around a.
        // RA component scaled by cos(dec_a) to keep units consistent in tangent plane.
        let ra_predicted = pair_context.anchor_ra
            + pair_context.velocity_x * dt_ac / pair_context.anchor_cos_dec.max(1e-12);
        let dec_predicted = pair_context.anchor_dec + pair_context.velocity_y * dt_ac;

        // Project actual c and predicted position onto the tangent plane around a.
        let (dx_actual, dy_actual) = planar_offset_fast(
            pair_context.anchor_ra,
            pair_context.anchor_dec,
            pair_context.anchor_cos_dec,
            tables.ra(candidate_id),
            tables.dec(candidate_id),
        );
        let (dx_pred, dy_pred) = planar_offset_fast(
            pair_context.anchor_ra,
            pair_context.anchor_dec,
            pair_context.anchor_cos_dec,
            ra_predicted,
            dec_predicted,
        );

        // Tangent-plane residual of predicted vs actual c.
        let residual = ((dx_actual - dx_pred).powi(2) + (dy_actual - dy_pred).powi(2)).sqrt();

        if residual <= triplet_config.max_predicted_residual {
            // All constraints satisfied; temporal order is guaranteed by bucket search and checks.
            triplets_out
                .push((pair_context.anchor_id, pair_context.middle_id, candidate_id).into());
        }
    }
}

/// Core triplet generation from a list of previously built pairs.
///
/// Builds `(a, b, c)` triplets by scanning spatio-temporal neighbors of `b`,
/// checking pairwise time/angle/flux constraints, then enforcing a linear
/// prediction residual on the tangent plane around `a`.
///
/// Arguments
/// ---------
/// * `index` – Spatio-temporal bucket index (same as used for the pair stage).
/// * `alerts` – Contiguous array of alerts (`alert.id.idx() == index` invariant).
/// * `sb` – Spatial binner (e.g. HEALPix).
/// * `tb` – Time binner.
/// * `triplet_config` – Triplet constraints and thresholds:
///   - `max_dt_between` – maximum allowed time gap for b–c,
///   - `max_pair_sep` – maximum separation for `(b, c)`,
///   - `max_predicted_residual` – tangent-plane residual threshold,
///   - `enforce_time_order` – enforce `t_a < t_b` when consuming pairs,
///   - `max_flux_difference` – flux similarity constraint (|flux_b − flux_c|).
/// * `pairs` – List of `(a, b)` pairs previously produced by `generate_pairs`.
///
/// Return
/// ------
/// `Triplets` – vector of `(AlertId, AlertId, AlertId)` triplets satisfying all constraints.
///
/// Notes
/// -----
/// - Output is sorted and deduplicated.
/// - Internally caches spatial/time neighbors per key for efficiency.
/// - The function is deterministic given fixed inputs and configuration.
pub fn generate_triplets_from_pairs<Bs: SpatialBinner, Bt: TimeBinner>(
    index: &BucketIndex<AlertId>,
    alerts: &[Alert],
    sb: &Bs,
    tb: &Bt,
    triplet_config: &TripletConfig,
    pairs: &[Pair],
) -> Triplets {
    // Contiguity assumption: id == index; required for id-indexed tables.
    debug_assert!(
        alerts.iter().enumerate().all(|(i, a)| a.id.idx() == i),
        "generate_triplets_from_pairs expects contiguous AlertId (id == index)"
    );

    // Direct tables (id-indexed) built once; reused across all pairs.
    let lookup_tables = TripletLookupTables::build(alerts);

    // Precompute bucket keys (spatial + temporal) per alert; improves locality and avoids recomputation.
    let spatial_key_by_id: Vec<SpatialKey> =
        alerts.iter().map(|a| sb.key_for(a.ra, a.dec)).collect();
    let time_bin_by_id: Vec<TimeBin> = alerts.iter().map(|a| tb.bin_for(a.mjd_tt)).collect();

    // Light caches for neighbor buckets reused across many pairs.
    let search_radius = triplet_config.max_pair_sep + sb.cell_radius();
    let cos_pair_threshold = triplet_config.max_pair_sep.cos();
    let mut spatial_neighbor_cache: AHashMap<SpatialKey, Vec<SpatialKey>> = AHashMap::new();
    let mut timebin_target_cache: AHashMap<TimeBin, Vec<TimeBin>> = AHashMap::new();

    // Preallocate output; triplet count typically ≤ pair count.
    let mut triplets_out: Triplets = Vec::with_capacity(pairs.len() / 2);

    // ------------------------------------------------------------------
    // Main loop over all input pairs (a, b).
    // ------------------------------------------------------------------
    for &Pair {
        a: anchor_id,
        b: middle_id,
    } in pairs
    {
        // Enforce time order (configurable) and basic stability guard (t_b ≥ t_a).
        let anchor_time = lookup_tables.time(anchor_id);
        let middle_time = lookup_tables.time(middle_id);

        if middle_time.partial_cmp(&anchor_time) == Some(std::cmp::Ordering::Less)
            || (triplet_config.enforce_time_order
                && anchor_time.partial_cmp(&middle_time) != Some(std::cmp::Ordering::Less))
        {
            // Either t_b < t_a, or t_a !< t_b when strict ordering is required.
            continue;
        }

        // Center search around the middle detection `b` in the bucket index.
        let middle_space_key = spatial_key_by_id[middle_id.idx()];
        let middle_time_bin = time_bin_by_id[middle_id.idx()];

        // Spatial neighbors (cached, deduplicated).
        let spatial_neighbors = cached_spatial_neighbors_for_triplets(
            &mut spatial_neighbor_cache,
            sb,
            middle_space_key,
            search_radius,
        );

        // Temporal neighbor bins strictly after `b` (cached).
        let time_bins_for_candidates = cached_time_targets_for_triplets(
            &mut timebin_target_cache,
            tb,
            middle_time_bin,
            triplet_config,
        );

        // Precompute everything needed for this pair `(a, b)` once.
        let pair_context =
            TripletPairContext::new(anchor_id, middle_id, triplet_config, &lookup_tables);

        // Explore all spatio-temporal neighbor buckets of b for candidate c.
        for &candidate_time_bin in time_bins_for_candidates {
            for &candidate_space_key in spatial_neighbors {
                scan_bucket_for_triplets(
                    index,
                    &lookup_tables,
                    &pair_context,
                    triplet_config,
                    candidate_space_key,
                    candidate_time_bin,
                    cos_pair_threshold,
                    &mut triplets_out,
                );
            }
        }
    }

    // Sort + dedup, as the same (a, b, c) can be discovered via multiple bucket paths.
    triplets_out.sort_unstable();
    triplets_out.dedup();

    triplets_out
}

/// Extract [`SeedNode`] features from triplets for a given night.
///
/// Arguments
/// ---------
/// * `store` – Alert store used to resolve ids to [`Alert`] instances.
/// * `trips` – Triplet collection (typically from [`generate_triplets_from_pairs`]).
/// * `night_id` – Identifier of the night for produced seeds.
///
/// Return
/// ------
/// Vector of [`SeedNode`] built from triplets `(a, b, c)` with contiguous seed ids.
///
/// Notes
/// -----
/// - The function preserves the order of `trips`: the `i`-th triplet is
///   assigned `SeedId(i)`.
/// - Feature extraction is delegated to [`SeedNode::from_triplet`].
pub fn extract_triplet_features(
    store: &AlertStore,
    trips: &Triplets,
    night_id: NightId,
) -> Vec<SeedNode> {
    // Preallocate; one seed per triplet.
    let mut out = Vec::with_capacity(trips.len());
    for (
        i,
        &Triplet {
            a: alert_id_a,
            b: alert_id_b,
            c: alert_id_c,
        },
    ) in trips.iter().enumerate()
    {
        // Resolve ids to alerts; indexing is safe here by construction of store.
        let alert_a = &store.alerts[alert_id_a.idx()];
        let alert_b = &store.alerts[alert_id_b.idx()];
        let alert_c = &store.alerts[alert_id_c.idx()];
        let seed_id = SeedId::new(i as u64);

        out.push(SeedNode::from_triplet(
            seed_id, night_id, alert_a, alert_b, alert_c,
        ));
    }
    out
}

#[cfg(test)]
mod triplet_gen_tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashSet;

    use crate::{
        AlertId, MjdTt, Radians,
        alerts::Alert,
        astro_math::{ang_sep, arcsec_to_rad},
        spacetime_bucket::{
            bucket::build_bucket_index,
            spatial_binner::{SpatialBinner, SpatialKey},
            time_binner::{TimeBin, TimeBinner},
        },
    };

    const LAT_EPS: f64 = 1e-6;

    /* ------------------------- dummy binners ------------------------- */

    /// Dummy spatial binner: everything goes to a single spatial cell.
    ///
    /// Keeps tests focused on temporal and geometric logic while still
    /// exercising the bucketing interface.
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

        fn neighbors_into(&self, key: SpatialKey, ang_radius: Radians, out: &mut Vec<SpatialKey>) {
            todo!()
        }
    }

    /// Dummy time binner: uniform bins of fixed width starting from `t0`.
    ///
    /// Used to create predictable time-bin behavior in tests without relying
    /// on the production time-binning configuration.
    struct DummyTimeBinner {
        t0: MjdTt,
        width: f64,
    }

    impl DummyTimeBinner {
        /// Construct a new dummy time binner.
        ///
        /// Arguments
        /// ---------
        /// * `t0` – Reference epoch (MJD) for bin index 0.
        /// * `width` – Bin width in days.
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

    /// Construct a synthetic [`Alert`] with reasonable defaults for tests.
    fn mk_alert(id: AlertId, ra: f64, dec: f64, mjd_tt: f64, band: u8, flux: f32) -> Alert {
        let pos_err = arcsec_to_rad(0.5); // ~0.5" in radians
        Alert {
            id,
            dia_source_id: id.idx() as u64,
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

    /// Convenience builder for a [`TripletConfig`] tuned for tests.
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

    /* ---------------------- deterministic tests ---------------------- */

    /// Simple linear motion case: triplet (0,1,2) should be detected once.
    #[test]
    fn triplets_linear_motion_detected() {
        let sb = DummySpatialBinner;
        let tb = DummyTimeBinner::new(60000.0, 10.0 / 1440.0); // 10 min

        let t0 = 60000.0;
        // Linear motion: ~6" every 10 min along RA (tangent plane).
        let dec0: f64 = 0.25;
        let dr = arcsec_to_rad(6.0) / dec0.cos();

        let a = mk_alert(AlertId::new(0), 1.0, dec0, t0, 1, 1000.0);
        let b = mk_alert(
            AlertId::new(1),
            1.0 + dr,
            dec0,
            t0 + 10.0 / 1440.0,
            1,
            1000.0,
        );
        let c = mk_alert(
            AlertId::new(2),
            1.0 + 2.0 * dr,
            dec0,
            t0 + 20.0 / 1440.0,
            1,
            1000.0,
        );

        let alerts = vec![a, b, c];
        let index = build_bucket_index(&alerts, &sb, &tb);

        let triplet_config = mk_triplet_config(
            30.0 / 1440.0, // 30 min
            15.0,          // 15"
            3.0,           // 3"
        );

        // Only the pair (0,1) is needed to recover (0,1,2).
        let pairs = vec![Pair {
            a: AlertId::new(0),
            b: AlertId::new(1),
        }];

        let triplets =
            generate_triplets_from_pairs(&index, &alerts, &sb, &tb, &triplet_config, &pairs);

        assert!(
            triplets.contains(&Triplet {
                a: AlertId::new(0),
                b: AlertId::new(1),
                c: AlertId::new(2),
            }),
            "Expected triplet (0,1,2) to be detected",
        );

        // Uniqueness.
        let set: HashSet<_> = triplets.iter().collect();
        assert_eq!(set.len(), triplets.len());
    }

    /// A candidate `c` with a large deviation from the linear prediction
    /// should be rejected by the residual threshold.
    #[test]
    fn triplets_large_residual_rejected() {
        let sb = DummySpatialBinner;
        let tb = DummyTimeBinner::new(60000.0, 10.0 / 1440.0); // 10 min

        let t0 = 60000.0;
        let dec0: f64 = 0.2;
        let dr = arcsec_to_rad(6.0) / dec0.cos();

        let a = mk_alert(AlertId::new(0), 2.0, dec0, t0, 1, 1000.0);
        let b = mk_alert(
            AlertId::new(1),
            2.0 + dr,
            dec0,
            t0 + 10.0 / 1440.0,
            1,
            1000.0,
        );
        // Third point deviates by ~40" -> residual should exceed 5"
        let c = mk_alert(
            AlertId::new(2),
            2.0 + 2.0 * dr + arcsec_to_rad(40.0) / dec0.cos(),
            dec0,
            t0 + 20.0 / 1440.0,
            1,
            1000.0,
        );

        let alerts = vec![a, b, c];
        let index = build_bucket_index(&alerts, &sb, &tb);

        let triplet_config = mk_triplet_config(
            30.0 / 1440.0, // 30 min
            60.0,          // 60"
            5.0,           // 5"
        );

        let pairs = vec![Pair {
            a: AlertId::new(0),
            b: AlertId::new(1),
        }];

        let triplets =
            generate_triplets_from_pairs(&index, &alerts, &sb, &tb, &triplet_config, &pairs);

        assert!(
            !triplets.contains(&Triplet {
                a: AlertId::new(0),
                b: AlertId::new(1),
                c: AlertId::new(2),
            }),
            "Triplet with large residual should be rejected",
        );
    }

    /// Check that all produced triplets originate from existing pairs.
    #[test]
    fn triplets_from_pairs_is_subset_of_pairs_prefix() {
        let sb = DummySpatialBinner;
        let tb = DummyTimeBinner::new(62000.0, 10.0 / 1440.0);

        let t0 = 62000.0;
        let dec: f64 = 0.25;
        let dr = arcsec_to_rad(8.0) / dec.cos();

        let a = mk_alert(AlertId::new(0), 0.6, dec, t0, 1, 1000.0);
        let b = mk_alert(
            AlertId::new(1),
            0.6 + dr,
            dec,
            t0 + 10.0 / 1440.0,
            1,
            1000.0,
        );
        let c = mk_alert(
            AlertId::new(2),
            0.6 + 2.0 * dr,
            dec,
            t0 + 20.0 / 1440.0,
            1,
            1000.0,
        );
        let d = mk_alert(AlertId::new(3), 2.5, 0.0, t0 + 5.0 / 1440.0, 1, 1000.0); // noise

        let alerts = vec![a, b, c, d];
        let index = build_bucket_index(&alerts, &sb, &tb);

        let triplet_config = mk_triplet_config(
            25.0 / 1440.0, // 25 min
            20.0,          // 20"
            5.0,           // 5"
        );

        // Build a small set of pairs including (0,1) and (1,2).
        let pairs = vec![
            Pair {
                a: AlertId::new(0),
                b: AlertId::new(1),
            },
            Pair {
                a: AlertId::new(1),
                b: AlertId::new(2),
            },
        ];

        let triplets =
            generate_triplets_from_pairs(&index, &alerts, &sb, &tb, &triplet_config, &pairs);

        // Build a set of pairs (canonical order i<j).
        let mut pair_set = HashSet::new();
        for &Pair { a, b } in &pairs {
            let (i, j) = if a < b { (a, b) } else { (b, a) };
            pair_set.insert((i, j));
        }

        for &Triplet { a, b, c: _ } in &triplets {
            let (i, j) = if a < b { (a, b) } else { (b, a) };
            assert!(
                pair_set.contains(&(i, j)),
                "every triplet (a,b,c) must originate from an existing pair (a,b)"
            );
        }
    }

    /* ----------------------- property-based tests ----------------------- */

    // Strategies, generators, and checks documenting constraints and invariants.

    /// RA generator covering the full [0, 2π) range.
    fn ra_strategy() -> impl Strategy<Value = f64> {
        0.0f64..(2.0 * std::f64::consts::PI)
    }

    /// Dec generator avoiding the exact poles to limit numeric pathologies.
    fn dec_strategy() -> impl Strategy<Value = f64> {
        // Avoid exactly ±π/2 to limit numeric pathologies at the poles.
        (-(std::f64::consts::PI / 2.0 - LAT_EPS))..(std::f64::consts::PI / 2.0 - LAT_EPS)
    }

    /// Time generator spanning a ~4h window.
    fn t_strategy() -> impl Strategy<Value = f64> {
        // ~4h window.
        60000.0f64..60000.1667f64 // 0.1667 ≈ 4h
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 32,
            .. ProptestConfig::default()
        })]

        /// All returned triplets must satisfy:
        /// - time ordering t_a < t_b < t_c,
        /// - |t_b - t_a|, |t_c - t_b| ≤ max_dt_between,
        /// - angular constraints on (b, c),
        /// - linear prediction residual ≤ max_predicted_residual,
        /// - flux similarity constraints between b and c,
        /// - (a, b) pair membership in the input pair list.
        #[test]
        fn prop_triplets_respect_constraints(
            samples in proptest::collection::vec(
                (ra_strategy(), dec_strategy(), t_strategy()),
                0..40
            )
        ) {
            let sb = DummySpatialBinner;
            let tb = DummyTimeBinner::new(60000.0, 10.0 / 1440.0); // 10 min bins

            let triplet_config = mk_triplet_config(
                40.0 / 1440.0, // 40 min
                30.0,          // 30"
                10.0,          // 10"
            );

            // Build alerts with identical flux to simplify flux tests.
            let alerts: Vec<Alert> = samples
                .iter()
                .enumerate()
                .map(|(i, (ra, dec, t))| {
                    mk_alert(AlertId::new(i as u32), *ra, *dec, *t, 1, 1000.0)
                })
                .collect();

            let index = build_bucket_index(&alerts, &sb, &tb);

            // Build a naive pair list (a,b) respecting time & angular constraints.
            let mut pairs: Vec<Pair> = Vec::new();
            for i in 0..alerts.len() {
                for j in (i + 1)..alerts.len() {
                    let a = &alerts[i];
                    let b = &alerts[j];
                    let dt = b.mjd_tt - a.mjd_tt;
                    if dt <= 0.0 || dt > triplet_config.max_dt_between {
                        continue;
                    }
                    let d = ang_sep(a.ra, a.dec, b.ra, b.dec);
                    if d > triplet_config.max_pair_sep {
                        continue;
                    }
                    pairs.push(Pair {
                        a: AlertId::new(i as u32),
                        b: AlertId::new(j as u32),
                    });
                }
            }

            let triplets = generate_triplets_from_pairs(&index, &alerts, &sb, &tb, &triplet_config, &pairs);

            // Uniqueness.
            let set: HashSet<_> = triplets.iter().collect();
            prop_assert_eq!(set.len(), triplets.len());

            // Convert pair list into a set for membership checks.
            let mut pair_set = HashSet::new();
            for &Pair { a, b } in &pairs {
                let (i, j) = if a < b { (a, b) } else { (b, a) };
                pair_set.insert((i, j));
            }

            for Triplet { a, b, c } in triplets {
                let obs_a = &alerts[a.idx()];
                let obs_b = &alerts[b.idx()];
                let obs_c = &alerts[c.idx()];

                // Time ordering.
                prop_assert!(obs_a.mjd_tt < obs_b.mjd_tt && obs_b.mjd_tt < obs_c.mjd_tt);

                let dt_ab = obs_b.mjd_tt - obs_a.mjd_tt;
                let dt_bc = obs_c.mjd_tt - obs_b.mjd_tt;

                // Time constraints.
                prop_assert!(dt_ab <= triplet_config.max_dt_between + 1e-12);
                prop_assert!(dt_bc <= triplet_config.max_dt_between + 1e-12);

                // Angular constraints on (b, c).
                let dbc = ang_sep(obs_b.ra, obs_b.dec, obs_c.ra, obs_c.dec);
                prop_assert!(dbc <= triplet_config.max_pair_sep + 1e-12);

                // Flux constraint (trivial here: all equal, but keep the check).
                let flux_diff = (obs_b.flux - obs_c.flux).abs();
                prop_assert!(flux_diff <= triplet_config.max_flux_difference + 1e-6);

                // Linear prediction residual (recomputed as in the implementation).
                let cos_dec_a = obs_a.dec.cos();
                let (dx_ab, dy_ab) = planar_offset_fast(
                    obs_a.ra,
                    obs_a.dec,
                    cos_dec_a,
                    obs_b.ra,
                    obs_b.dec
                );
                let dt_ab_safe = dt_ab.max(1e-12);
                let vx = dx_ab / dt_ab_safe;
                let vy = dy_ab / dt_ab_safe;
                let dt_ac = obs_c.mjd_tt - obs_a.mjd_tt;
                let ra_pred = obs_a.ra + vx * dt_ac / cos_dec_a.max(1e-12);
                let dec_pred = obs_a.dec + vy * dt_ac;

                let (dx_actual, dy_actual) = planar_offset_fast(
                    obs_a.ra,
                    obs_a.dec,
                    cos_dec_a,
                    obs_c.ra,
                    obs_c.dec,
                );
                let (dx_pred, dy_pred) = planar_offset_fast(
                    obs_a.ra,
                    obs_a.dec,
                    cos_dec_a,
                    ra_pred,
                    dec_pred,
                );

                let residual = ((dx_actual - dx_pred).powi(2) + (dy_actual - dy_pred).powi(2)).sqrt();
                prop_assert!(residual <= triplet_config.max_predicted_residual + 1e-12);

                // Ensure (a, b) originates from the input pair list.
                let (i, j) = if a < b { (a, b) } else { (b, a) };
                prop_assert!(pair_set.contains(&(i, j)));
            }
        }
    }

    /* ---------------------- extract_triplet_features tests ---------------------- */

    /// Unit test for `extract_triplet_features`:
    /// - preserves input order,
    /// - assigns incremental `SeedId` starting at 0,
    /// - sets `members` to the triplet ids in time order,
    /// - sets `n_obs == 3`.
    #[test]
    fn extract_triplet_features_order_and_members() {
        use crate::alerts::AlertStore;

        let t0 = 60000.0;
        let dec0 = 0.3;

        // Build three alerts forming an increasing-time triplet.
        let a = mk_alert(AlertId::new(0), 1.0, dec0, t0, 1, 1000.0);
        let b = mk_alert(
            AlertId::new(1),
            1.0 + arcsec_to_rad(6.0) / dec0.cos(),
            dec0,
            t0 + 10.0 / 1440.0,
            1,
            1005.0,
        );
        let c = mk_alert(
            AlertId::new(2),
            1.0 + 2.0 * arcsec_to_rad(6.0) / dec0.cos(),
            dec0,
            t0 + 20.0 / 1440.0,
            1,
            1002.0,
        );

        let alerts = vec![a.clone(), b.clone(), c.clone()];
        let store = AlertStore::new(t0.floor(), alerts);

        // Two triplets; second one uses different ids but same night/order idea.
        let trips = vec![
            Triplet {
                a: a.id,
                b: b.id,
                c: c.id,
            },
            Triplet {
                a: AlertId::new(0),
                b: AlertId::new(2),
                c: AlertId::new(1),
            }, // intentionally out-of-time order to show we don't reorder members: we keep provided order in SeedNode.members
        ];

        let night_id = NightId::new(99);
        let seeds = extract_triplet_features(&store, &trips, night_id);

        assert_eq!(seeds.len(), 2);

        // SeedId assignment follows input order.
        assert_eq!(seeds[0].seed_id, SeedId::new(0));
        assert_eq!(seeds[1].seed_id, SeedId::new(1));

        // Night id propagation.
        assert_eq!(seeds[0].night_id, night_id);
        assert_eq!(seeds[1].night_id, night_id);

        // Members are the triplet ids as provided and n_obs == 3.
        assert_eq!(seeds[0].members, vec![a.id, b.id, c.id]);
        assert_eq!(seeds[0].n_obs, 3);

        assert_eq!(
            seeds[1].members,
            vec![AlertId::new(0), AlertId::new(2), AlertId::new(1)]
        );
        assert_eq!(seeds[1].n_obs, 3);
    }

    /// Proptest for `extract_triplet_features`:
    /// with arbitrary triplets and alerts, checks 1:1 mapping and invariants.
    mod prop_extract_triplet_features {
        use super::*;

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
            fn prop_extract_triplet_features_1to1_mapping(
                samples in proptest::collection::vec((ra_strategy(), dec_strategy(), time_strategy()), 3..60)
            ) {
                // Build alerts from samples.
                let alerts: Vec<Alert> = samples.iter().enumerate().map(|(i, (ra, dec, t))| Alert {
                    id: AlertId::new(i as u32),
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

                // Build a simple list of triplets using consecutive ids where times increase.
                // Use multiple windows to generate several triplets.
                let mut trips: Triplets = Vec::new();
                for i in 0..alerts.len().saturating_sub(2) {
                    let a = AlertId::new(i as u32);
                    let b = AlertId::new((i+1) as u32);
                    let c = AlertId::new((i+2) as u32);
                    let ta = alerts[i].mjd_tt;
                    let tb = alerts[i+1].mjd_tt;
                    let tc = alerts[i+2].mjd_tt;
                    if ta < tb && tb < tc {
                        trips.push(Triplet { a, b, c });
                    }
                }

                let night_id = NightId::new(7);
                let seeds = extract_triplet_features(&store, &trips, night_id);

                // 1:1 mapping: each triplet produces a seed.
                prop_assert_eq!(seeds.len(), trips.len());

                // Check invariants per seed.
                for (k, seed) in seeds.iter().enumerate() {
                    prop_assert_eq!(seed.seed_id, SeedId::new(k as u64));
                    prop_assert_eq!(seed.n_obs, 3);
                    let Triplet { a, b, c } = trips[k];
                    prop_assert_eq!(seed.members.clone(), vec![a, b, c]);
                    prop_assert_eq!(seed.night_id, night_id);

                    // Resolved members match alerts.
                    let resolved = seed.resolve_seed_members(&store).expect("ids valid");
                    prop_assert_eq!(resolved.len(), 3);
                    prop_assert_eq!(resolved[0].id, a);
                    prop_assert_eq!(resolved[1].id, b);
                    prop_assert_eq!(resolved[2].id, c);
                }
            }
        }
    }
}
