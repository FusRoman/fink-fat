//! Pair generation `(a, b)` for short-baseline intra-night motion filtering.
//!
//! This module builds ordered detection pairs `(a, b)` from an alert stream
//! already indexed in spatio-temporal buckets ([`BucketIndex`]).
//!
//! Goal
//! ----
//! Efficiently enumerate only “plausible” intra-night links by applying cheap,
//! cadence-aware constraints before any heavier downstream processing
//! (triplet building, seed fitting, graph edges, ML, etc.).
//!
//! Key constraints
//! ---------------
//! For each candidate pair `(a, b)`:
//! - **Time ordering:** `t_b > t_a`
//! - **Maximum time separation:** `t_b - t_a ≤ max_dt`
//! - **Flux similarity:** `|flux_a - flux_b| ≤ max_flux_difference`
//! - **Angular-speed constraint:** `angular_separation_vincenty(a, b) / (t_b - t_a) ≤ max_angular_speed`
//!
//! The angular-speed constraint is implemented via a dot-product threshold
//! (no `acos`):
//! - let `Δθ_max = max_angular_speed · Δt` (clamped to π),
//! - accept iff `cos(Δθ) ≥ cos(Δθ_max)`,
//! - where `cos(Δθ)` is computed as `dot3(unit_vec(ra_a, dec_a), unit_vec(ra_b, dec_b))`.
//!
//! Indexing strategy
//! -----------------
//! Alerts are grouped into buckets keyed by `(SpatialKey, TimeBin)`.
//! For each anchor bucket, we only search a small neighborhood:
//! - **Spatial neighborhood:** `neighbors(space_key, search_radius)`
//! - **Temporal neighborhood:** `time_targets(base_bin, max_dt, allow_same_timebin)`
//!
//! Because bucket members are stored **sorted by time**, we can binary-search to
//! skip `t_b ≤ t_a` within each candidate bucket.
//!
//! Deduplication
//! -------------
//! Neighbor scans can overlap, producing duplicate `(a, b)` candidates.
//! We deduplicate by pointer identity: `(ptr(a), ptr(b))`.
//!
//! Determinism
//! -----------
//! Output pairs are sorted at the end to provide deterministic ordering for
//! tests and reproducible benchmarks.
//!
//! Lifetimes
//! ---------
//! The module operates on borrowed alerts (`&'alert_lf Alert`) and returns
//! pairs and seeds borrowing the same alerts. The alert storage must outlive
//! the returned values.
//!
//! See also
//! --------
//! - [`PairConfig`] – configuration of time/flux/speed constraints.
//! - [`BucketIndex`] – bucketed storage used for accelerated neighbor scans.

use ahash::{AHashMap, AHashSet};
use photom::{
    coordinates::cartesian::CartesianCoord, observation_dataset::observation::Observation,
};

use crate::{
    engine_config::pair_config::PairConfig,
    spacetime_bucket::{
        bucket::{BucketIndex, BucketKey},
        spatial_binner::{SpatialBinner, SpatialKey},
        time_binner::{TimeBin, TimeBinner, time_targets},
    },
};

use crate::logging::LogTarget;

/// Structured log events for intra-night pairing/tracklet linking (shared by
/// [`pairs`](self), [`triplets`](super::triplets) and
/// [`tracklet_linker`](super::tracklet_linker)). See [`crate::logging`] for
/// the `.emit()` pattern.
pub enum SeedingEvent {
    PairsStart {
        n_buckets: usize,
        max_dt: f64,
        max_angular_speed: f64,
        max_mag_difference: f64,
        allow_same_timebin: bool,
        sep_cap: f64,
        spatial_search_radius: f64,
    },
    PairsComplete {
        n_pairs: usize,
        n_rejected_flux: u64,
        n_rejected_speed: u64,
        n_dedup_skipped: u64,
    },
    TripletsStart {
        n_input_pairs: usize,
        max_dt_between: f64,
        max_pair_sep: f64,
        max_predicted_residual: f64,
        max_mag_difference: f64,
        enforce_time_order: bool,
        search_radius: f64,
    },
    TripletsComplete {
        n_triplets: usize,
        n_skipped_time_order: u64,
        n_rejected_flux: u64,
        n_rejected_angular: u64,
        n_rejected_residual: u64,
        n_dedup_skipped: u64,
    },
}

crate::impl_log_target!(
    SeedingEvent,
    "seeding",
    "Intra-night observation pairing and tracklet linking (pairs/triplets)",
    [tracing::Level::DEBUG]
);

impl SeedingEvent {
    pub fn emit(&self) {
        use SeedingEvent::*;
        match self {
            PairsStart {
                n_buckets,
                max_dt,
                max_angular_speed,
                max_mag_difference,
                allow_same_timebin,
                sep_cap,
                spatial_search_radius,
            } => tracing::debug!(
                target: SeedingEvent::TARGET, n_buckets, max_dt, max_angular_speed, max_mag_difference, allow_same_timebin, sep_cap, spatial_search_radius,
                "generate_pairs starting"
            ),
            PairsComplete {
                n_pairs,
                n_rejected_flux,
                n_rejected_speed,
                n_dedup_skipped,
            } => tracing::debug!(
                target: SeedingEvent::TARGET, n_pairs, n_rejected_flux, n_rejected_speed, n_dedup_skipped,
                "generate_pairs complete"
            ),
            TripletsStart {
                n_input_pairs,
                max_dt_between,
                max_pair_sep,
                max_predicted_residual,
                max_mag_difference,
                enforce_time_order,
                search_radius,
            } => tracing::debug!(
                target: SeedingEvent::TARGET, n_input_pairs, max_dt_between, max_pair_sep, max_predicted_residual, max_mag_difference, enforce_time_order, search_radius,
                "generate_triplets_from_pairs starting"
            ),
            TripletsComplete {
                n_triplets,
                n_skipped_time_order,
                n_rejected_flux,
                n_rejected_angular,
                n_rejected_residual,
                n_dedup_skipped,
            } => tracing::debug!(
                target: SeedingEvent::TARGET, n_triplets, n_skipped_time_order, n_rejected_flux, n_rejected_angular, n_rejected_residual, n_dedup_skipped,
                "generate_triplets_from_pairs complete"
            ),
        }
    }
}

/// A time-ordered detection pair `(a, b)` with `t_b > t_a`.
///
/// The pair stores references to alerts (no copying).
/// In this module, pairs are constructed such that:
/// - `a` is the anchor detection,
/// - `b` is a candidate detection at a later epoch within `max_dt`.
///
/// # Notes
///
/// The ordering is semantically meaningful (directed in time) and is used
/// downstream when fitting a linear seed model.
#[derive(Copy, Clone, Debug)]
pub struct Pair<'alert_lf> {
    /// Anchor detection (earlier epoch).
    pub a: &'alert_lf Observation,
    /// Candidate detection (later epoch).
    pub b: &'alert_lf Observation,
}

/// Convenience alias: a flat list of time-ordered detection pairs.
pub type Pairs<'alert_lf> = Vec<Pair<'alert_lf>>;

/// Cached spatial neighbors for a given `SpatialKey`.
///
/// Computing `SpatialBinner::neighbors` can be non-trivial (HEALPix ring queries,
/// neighbor expansion, etc.). For pair generation we call it many times with the
/// same bucket keys, so we cache the results.
///
/// # Arguments
///
/// - `cache` — cache map keyed by the target cell.
/// - `spatial_binner` — spatial discretization backend (e.g. HEALPix).
/// - `target_space_key` — space cell key of the anchor bucket.
/// - `search_radius` — cone radius (radians) used to include neighboring cells.
///
/// # Returns
///
/// Sorted, deduplicated list of neighboring cell keys including
/// `target_space_key` itself if returned by the binner.
///
/// # Notes
///
/// The vector is sorted (`sort_unstable`) and deduplicated (`dedup`)
/// to ensure deterministic behavior and to avoid redundant scans.
#[inline]
fn cached_spatial_neighbors<'cache, Bs: SpatialBinner>(
    cache: &'cache mut AHashMap<SpatialKey, Vec<SpatialKey>>,
    spatial_binner: &Bs,
    target_space_key: SpatialKey,
    search_radius: f64,
) -> &'cache Vec<SpatialKey> {
    cache.entry(target_space_key).or_insert_with(|| {
        let mut neighbors = spatial_binner.neighbors(target_space_key, search_radius);
        neighbors.sort_unstable();
        neighbors.dedup();
        neighbors
    })
}

/// Cached time-bin targets for a given `TimeBin`.
///
/// This wraps [`time_targets`] and caches the resulting target bins.
/// The target set depends on:
/// - the base bin,
/// - `config.max_dt`,
/// - `config.allow_same_timebin`.
///
/// # Arguments
///
/// - `cache` — cache map keyed by the base time bin.
/// - `time_binner` — time discretization backend.
/// - `base_bin` — time bin of the anchor bucket.
/// - `config` — pair generation configuration.
///
/// # Returns
///
/// List of time bins to consider as candidate buckets.
#[inline]
fn cached_time_targets<'cache, Bt: TimeBinner>(
    cache: &'cache mut AHashMap<TimeBin, Vec<TimeBin>>,
    time_binner: &Bt,
    base_bin: TimeBin,
    config: &PairConfig,
) -> &'cache Vec<TimeBin> {
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

/// Find the first index `k` such that `members[k].mjd_tt > t0`.
///
/// This is a strict lower-bound search for times greater than `t0`.
/// It is used to skip all candidate alerts `b` with `t_b ≤ t_a` inside a bucket,
/// assuming `members` is sorted by increasing `mjd_tt`.
///
/// # Arguments
///
/// - `members` — bucket members, sorted by `mjd_tt` ascending.
/// - `t0` — threshold epoch (MJD TT).
///
/// # Returns
///
/// Index of the first element with `mjd_tt > t0` (may be `members.len()`).
#[inline]
fn lower_bound_gt_time(members: &[&Observation], t0: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = members.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if members[mid].mjd_tt() <= t0 {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Generate all valid `(a, b)` pairs according to [`PairConfig`].
///
/// The algorithm iterates anchor buckets and anchor alerts `a`, enumerates nearby
/// buckets using cached spatial neighbors and cached time targets, and within each
/// candidate bucket binary-searches to skip `t_b ≤ t_a`, scans forward until
/// `t_b > t_a + max_dt`, applies flux and angular-speed constraints, and deduplicates
/// by `(ptr(a), ptr(b))`.
///
/// # Arguments
///
/// - `bucket_index` — spatio-temporal bucket index holding observations; each bucket's
///   `members` must be sorted by time (`mjd_tt`).
/// - `spatial_binner` — spatial discretization backend used to build neighbor sets.
/// - `time_binner` — time discretization backend used to map `max_dt` to candidate time bins.
/// - `config` — pair-generation parameters: `max_dt` (days), `max_angular_speed` (rad/day),
///   `max_flux_difference` (flux units), `allow_same_timebin` (bool).
///
/// # Returns
///
/// A deterministic, time-ordered list of unique pairs `(a, b)`.
///
/// # Notes
///
/// **Spatial search radius:** `search_radius = max_sep + cell_radius`, where
/// `max_sep = max_angular_speed * max_dt`. This ensures all potentially intersecting
/// cells are scanned even when a bucket boundary cuts through the geometric cone.
///
/// **Angular-speed test:** `acos` is avoided by comparing dot products —
/// accept iff `dot3(unit_vec(a), unit_vec(b)) >= cos(max_angular_speed * Δt)`.
///
/// **Deduplication:** pairs are keyed by pointer identity `(ptr(a), ptr(b))`; this
/// assumes the same `Alert` object is not duplicated in memory.
///
/// Output is sorted at the end using `(a, b)` ordering for reproducibility.
/// This stage is intentionally permissive — it is a pre-filter before seed
/// fitting and later graph construction.
///
/// # See also
///
/// - [`extract_pair_features`] — converts valid pairs into [`SeedNode`] objects.
pub fn generate_pairs<'alert_lf, Bs: SpatialBinner, Bt: TimeBinner>(
    bucket_index: &BucketIndex<&'alert_lf Observation>,
    spatial_binner: &Bs,
    time_binner: &Bt,
    config: &PairConfig,
) -> Pairs<'alert_lf> {
    // Spatial search radius: cap + cell radius.
    let sep_cap = (config.max_angular_speed * config.max_dt).max(0.0);
    let spatial_search_radius = sep_cap + spatial_binner.cell_radius();

    SeedingEvent::PairsStart {
        n_buckets: bucket_index.buckets.len(),
        max_dt: config.max_dt,
        max_angular_speed: config.max_angular_speed,
        max_mag_difference: config.max_mag_difference,
        allow_same_timebin: config.allow_same_timebin,
        sep_cap,
        spatial_search_radius,
    }
    .emit();

    let mut spatial_neighbor_cache = AHashMap::<SpatialKey, Vec<SpatialKey>>::new();
    let mut timebin_target_cache = AHashMap::<TimeBin, Vec<TimeBin>>::new();

    // Deduplicate pairs created through overlapping neighbor scans.
    // Key is (ptr(a), ptr(b)).
    let mut seen: AHashSet<(usize, usize)> = AHashSet::new();

    let mut out: Pairs<'alert_lf> = Vec::new();

    // Rejection counters (reported at DEBUG level at the end).
    let mut n_rejected_flux: u64 = 0;
    let mut n_rejected_speed: u64 = 0;
    let mut n_dedup_skipped: u64 = 0;

    for (bucket_key, bucket) in &bucket_index.buckets {
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

        // Anchors `a` from this bucket
        for &a in bucket.members.iter() {
            let t_a = a.mjd_tt();
            let t_upper = t_a + config.max_dt;
            let magnitude_a = a.photometry().magnitude;

            // Precompute direction vector of `a` to amortize dot products.
            let u_a: CartesianCoord = a.equ_coord().into();

            for &time_bin in time_targets {
                for &space_key in spatial_neighbors {
                    let Some(cand_bucket) = bucket_index.buckets.get(&BucketKey {
                        space_key,
                        time_bin,
                    }) else {
                        continue;
                    };

                    let members = cand_bucket.members.as_slice(); // sorted by time
                    let mut idx = lower_bound_gt_time(members, t_a);

                    while idx < members.len() {
                        let b = members[idx];
                        idx += 1;

                        let t_b = b.mjd_tt();
                        if t_b > t_upper {
                            break;
                        }

                        // (Very rare) same object reference
                        if core::ptr::eq(a, b) {
                            continue;
                        }

                        // Magnitude similarity
                        if (magnitude_a - b.photometry().magnitude).abs()
                            > config.max_mag_difference
                        {
                            n_rejected_flux += 1;
                            continue;
                        }

                        // Angular-speed constraint via dot product
                        let dt = t_b - t_a; // dt > 0
                        let max_sep_dt = (config.max_angular_speed * dt).min(core::f64::consts::PI);
                        let cos_thresh = max_sep_dt.cos();

                        let u_b: CartesianCoord = b.equ_coord().into();
                        if u_a.dot(&u_b) < cos_thresh {
                            n_rejected_speed += 1;
                            continue;
                        }

                        // Dedup + push
                        let key = (
                            a as *const Observation as usize,
                            b as *const Observation as usize,
                        );
                        if seen.insert(key) {
                            out.push(Pair { a, b });
                        } else {
                            n_dedup_skipped += 1;
                        }
                    }
                }
            }
        }
    }

    // Deterministic ordering (handy for tests / reproducibility)
    out.sort_unstable_by(|p1, p2| p1.a.cmp(p2.a).then_with(|| p1.b.cmp(p2.b)));

    SeedingEvent::PairsComplete {
        n_pairs: out.len(),
        n_rejected_flux,
        n_rejected_speed,
        n_dedup_skipped,
    }
    .emit();

    out
}

#[cfg(test)]
mod pair_gen_tests {
    use super::*;
    use std::collections::HashSet;
    use std::f64::consts::PI;

    use photom::{
        Arcseconds, Radians,
        coordinates::equatorial::EquCoord,
        observation_dataset::{
            ObsDataset,
            observation::{Observation, ObservationInput},
        },
        photometry::{Filter, Photometry as PhotomPhotometry},
    };

    use crate::engine_config::pair_config::PairConfig;
    use crate::spacetime_bucket::bucket::{BucketKey, build_alert_bucket_index};
    use crate::spacetime_bucket::healpix_binner::HealpixBinner;
    use crate::spacetime_bucket::uniform_time_binner::UniformTimeBinner;

    /* ------------------------- helpers ------------------------- */

    #[inline]
    pub fn arcsec_to_rad(x: Arcseconds) -> Radians {
        x * PI / (180.0 * 3600.0)
    }

    /// Construct a minimal `Observation` for testing.
    fn mk_observation(id: u64, ra: f64, dec: f64, mjd_tt: f64, band: u8, flux: f64) -> Observation {
        let obs_dataset = ObsDataset::empty();

        let pos_err = arcsec_to_rad(0.5);
        let equ_coord = EquCoord::new(ra, pos_err, dec, pos_err);
        let photometry = PhotomPhotometry {
            magnitude: flux,
            error: 0.0,
            filter: Filter::Int(band as u32),
        };
        let input = ObservationInput::new(id, equ_coord, photometry, mjd_tt, None);
        let (obs_dataset, obs_id) = obs_dataset.push_observation(vec![input]).unwrap();
        let observation = obs_dataset
            .get_obs_by_index(*obs_id.get(0).unwrap())
            .unwrap()
            .clone();
        observation
    }

    fn idx_of(alerts: &[Observation], a: &Observation) -> usize {
        alerts
            .iter()
            .position(|x| core::ptr::eq(x, a))
            .expect("observation ref not found in slice")
    }

    /* ------------------------- unit tests ------------------------- */

    /// Basic sanity: one valid pair in time/angle, plus a distant outlier.
    #[test]
    fn pairs_basic_one_pair() {
        let spatial_binner = HealpixBinner::new(10); // NSIDE=1024
        let time_binner = UniformTimeBinner::new(60000.0, 10.0 / 1440.0); // 10 min bins

        let t0 = 60000.10;
        let dec0 = 0.2;

        // Two observations ~5" apart and 8 min apart.
        let a1 = mk_observation(0, 1.0, dec0, t0, 1, 1000.0);
        let a2 = mk_observation(
            1,
            1.0 + arcsec_to_rad(5.0) / dec0.cos(),
            dec0,
            t0 + 8.0 / 1440.0,
            1,
            1002.0,
        );

        // A distant outlier (must not match).
        let a3 = mk_observation(2, 2.0, -0.3, t0 + 5.0 / 1440.0, 1, 900.0);

        let alerts = vec![a1, a2, a3];
        let bucket_index = build_alert_bucket_index(&alerts, &spatial_binner, &time_binner);

        // Allow up to ~10" over 10 minutes.
        let max_dt = 10.0 / 1440.0;
        let max_sep = arcsec_to_rad(10.0);
        let omega = max_sep / max_dt;

        let config = PairConfig {
            max_dt,
            max_angular_speed: omega,
            allow_same_timebin: false,
            max_mag_difference: 10.0,
            acc_prior_var: 1e-5,
            ..Default::default()
        };

        let pairs = generate_pairs(&bucket_index, &spatial_binner, &time_binner, &config);

        assert_eq!(pairs.len(), 1);

        let p = pairs[0];
        let ia = idx_of(&alerts, p.a);
        let ib = idx_of(&alerts, p.b);

        assert_eq!(ia, 0);
        assert_eq!(ib, 1);
        assert!(p.b.mjd_tt() > p.a.mjd_tt());
    }

    /// Check behavior of `allow_same_timebin`.
    #[test]
    fn pairs_same_timebin_behavior() {
        let spatial_binner = HealpixBinner::new(9);
        let time_binner = UniformTimeBinner::new(60000.0, 20.0 / 1440.0); // 20 min bins

        let t0 = 60000.25;
        let dec0 = 0.1;

        // Two observations in the same time bin (Δt = 5 min < 20 min).
        let a1 = mk_observation(0, 1.5, dec0, t0, 1, 1000.0);
        let a2 = mk_observation(
            1,
            1.5 + arcsec_to_rad(4.0) / dec0.cos(),
            dec0,
            t0 + 5.0 / 1440.0,
            1,
            1001.0,
        );

        let alerts = vec![a1, a2];
        let bucket_index = build_alert_bucket_index(&alerts, &spatial_binner, &time_binner);

        let max_dt = 5.0 / 1440.0;
        let max_sep = arcsec_to_rad(4.0);
        let omega = max_sep / max_dt * 1.1;

        let config_no_same = PairConfig {
            max_dt,
            max_angular_speed: omega,
            allow_same_timebin: false,
            max_mag_difference: 10.0,
            acc_prior_var: 1e-5,
            ..Default::default()
        };

        let pairs_no_same = generate_pairs(
            &bucket_index,
            &spatial_binner,
            &time_binner,
            &config_no_same,
        );
        assert!(pairs_no_same.is_empty());

        let config_same = PairConfig {
            max_dt,
            max_angular_speed: omega,
            allow_same_timebin: true,
            max_mag_difference: 10.0,
            acc_prior_var: 1e-5,
            ..Default::default()
        };

        let pairs_same = generate_pairs(&bucket_index, &spatial_binner, &time_binner, &config_same);
        assert_eq!(pairs_same.len(), 1);

        let ia = idx_of(&alerts, pairs_same[0].a);
        let ib = idx_of(&alerts, pairs_same[0].b);
        assert_eq!(ia, 0);
        assert_eq!(ib, 1);
    }

    /// Ensure all pairs respect time ordering and the angular-speed constraint.
    #[test]
    fn pairs_time_order_and_constraints() {
        let spatial_binner = HealpixBinner::new(8);
        let time_binner = UniformTimeBinner::new(60000.0, 5.0 / 1440.0); // 5 min bins

        let t0 = 60000.0;
        let dec0 = 0.3;

        let a0 = mk_observation(0, 1.0, dec0, t0, 1, 1000.0);
        let a1 = mk_observation(
            1,
            1.0 + arcsec_to_rad(5.0) / dec0.cos(),
            dec0,
            t0 + 5.0 / 1440.0,
            1,
            1000.0,
        );
        let a2 = mk_observation(
            2,
            1.0 + arcsec_to_rad(9.0) / dec0.cos(),
            dec0,
            t0 + 10.0 / 1440.0,
            1,
            1000.0,
        );

        let alerts = vec![a0, a1, a2];
        let bucket_index = build_alert_bucket_index(&alerts, &spatial_binner, &time_binner);

        let dt01 = 5.0 / 1440.0;
        let dt02 = 10.0 / 1440.0;
        let sep01 = arcsec_to_rad(5.0);
        let sep02 = arcsec_to_rad(9.0);

        let omega_max = (sep01 / dt01).max(sep02 / dt02);
        let omega = 1.1 * omega_max;

        let config = PairConfig {
            max_dt: 15.0 / 1440.0,
            max_angular_speed: omega,
            allow_same_timebin: true,
            max_mag_difference: 1e6,
            acc_prior_var: 1e-5,
            ..Default::default()
        };

        let pairs = generate_pairs(&bucket_index, &spatial_binner, &time_binner, &config);

        for Pair { a, b } in &pairs {
            assert!(b.mjd_tt() > a.mjd_tt(), "t_b must be > t_a");
            assert!(
                (b.mjd_tt() - a.mjd_tt()) <= config.max_dt + 1e-15,
                "Δt must be <= max_dt"
            );

            let dt = b.mjd_tt() - a.mjd_tt();
            let d = a.equ_coord().angular_separation(&b.equ_coord());

            assert!(
                d <= config.max_angular_speed * dt + 1e-12,
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

        let a0 = mk_observation(0, 0.5, dec0, t0, 1, 1000.0);
        let a1 = mk_observation(
            1,
            0.5 + arcsec_to_rad(4.0) / dec0.cos(),
            dec0,
            t0 + 1.0 / 1440.0,
            1,
            1005.0,
        );
        let a2 = mk_observation(
            2,
            0.5 + arcsec_to_rad(7.0) / dec0.cos(),
            dec0,
            t0 + 2.0 / 1440.0,
            1,
            1002.0,
        );

        let alerts = vec![a0, a1, a2];
        let bucket_index = build_alert_bucket_index(&alerts, &spatial_binner, &time_binner);

        let max_dt = 10.0 / 1440.0;
        let max_sep = arcsec_to_rad(15.0);
        let omega = max_sep / max_dt;

        let config = PairConfig {
            max_dt,
            max_angular_speed: omega,
            allow_same_timebin: true,
            max_mag_difference: 10.0,
            acc_prior_var: 1e-5,
            ..Default::default()
        };

        let pairs = generate_pairs(&bucket_index, &spatial_binner, &time_binner, &config);

        // Uniqueness by pointer identity
        let mut set: HashSet<(usize, usize)> = HashSet::new();
        for p in &pairs {
            let k = (
                p.a as *const Observation as usize,
                p.b as *const Observation as usize,
            );
            assert!(set.insert(k), "duplicate pair produced");
        }
    }

    /* ---------------- integration-style test ---------------- */

    #[test]
    fn pairs_integration_small_track_with_noise() {
        let spatial_binner = HealpixBinner::new(8);
        let time_binner = UniformTimeBinner::new(61000.0, 5.0 / 1440.0);

        let t0 = 61000.0;
        let dec0: f64 = 0.25;
        let dr = arcsec_to_rad(6.0) / dec0.cos();

        let a = mk_observation(0, 2.0, dec0, t0, 1, 1000.0);
        let b = mk_observation(1, 2.0 + dr, dec0, t0 + 5.0 / 1440.0, 1, 1002.0);
        let c = mk_observation(2, 2.0 + 2.0 * dr, dec0, t0 + 10.0 / 1440.0, 1, 1004.0);

        let n1 = mk_observation(3, 3.0, -0.1, t0 + 3.0 / 1440.0, 1, 500.0);
        let n2 = mk_observation(4, 1.0, 0.8, t0 + 6.0 / 1440.0, 1, 800.0);

        let alerts = vec![a, b, c, n1, n2];
        let bucket_index = build_alert_bucket_index(&alerts, &spatial_binner, &time_binner);

        let max_dt = 15.0 / 1440.0;
        let max_sep = arcsec_to_rad(20.0);
        let omega = max_sep / max_dt;

        let config = PairConfig {
            max_dt,
            max_angular_speed: omega,
            allow_same_timebin: true,
            max_mag_difference: 100.0,
            acc_prior_var: 1e-5,
            ..Default::default()
        };

        let pairs = generate_pairs(&bucket_index, &spatial_binner, &time_binner, &config);

        // Build canonical set (by slice indices) for easy checking.
        let mut pair_set = HashSet::new();
        for p in &pairs {
            let i = idx_of(&alerts, p.a);
            let j = idx_of(&alerts, p.b);
            pair_set.insert((i.min(j), i.max(j)));
        }

        assert!(pair_set.contains(&(0, 1)));
        assert!(pair_set.contains(&(1, 2)));
        assert!(pair_set.contains(&(0, 2)));

        for p in &pairs {
            let dt = p.b.mjd_tt() - p.a.mjd_tt();
            let d = p.a.equ_coord().angular_separation(&p.b.equ_coord());
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

            #[test]
            fn prop_pairs_respect_constraints_and_buckets(
                triples in proptest::collection::vec((ra_strategy(), dec_strategy(), time_strategy()), 0..120)
            ) {
                let spatial_binner = HealpixBinner::new(8);
                let time_binner = UniformTimeBinner::new(60000.0, 10.0 / 1440.0); // 10 min bins

                let max_dt = 30.0 / 1440.0;
                let max_sep = arcsec_to_rad(20.0);
                let omega = max_sep / max_dt;

                let config = PairConfig {
                    max_dt,
                    max_angular_speed: omega,
                    allow_same_timebin: false,
                    max_mag_difference: 1e6,
                    acc_prior_var: 1e-5,
                    ..Default::default()
                };

                let sep_cap = config.max_angular_speed * config.max_dt;
                let search_radius = sep_cap + spatial_binner.cell_radius();

                let alerts: Vec<Observation> = triples.iter().enumerate()
                    .map(|(i, (ra, dec, t))| mk_observation(i as u64, *ra, *dec, *t, 1, 1000.0))
                    .collect();

                let bucket_index = build_alert_bucket_index(&alerts, &spatial_binner, &time_binner);

                let pairs = generate_pairs(&bucket_index, &spatial_binner, &time_binner, &config);

                // Pairs must be unique (pointer identity).
                let mut set: HashSet<(usize, usize)> = HashSet::new();
                for p in &pairs {
                    let k = (
                        p.a as *const Observation as usize,
                        p.b as *const Observation as usize,
                    );
                    prop_assert!(set.insert(k));
                }

                for Pair { a, b } in pairs {
                    prop_assert!(b.mjd_tt() > a.mjd_tt());
                    prop_assert!((b.mjd_tt() - a.mjd_tt()) <= config.max_dt + 1e-15);

                    let dt = b.mjd_tt() - a.mjd_tt();
                    let d = a.equ_coord().angular_separation(&b.equ_coord());
                    prop_assert!(d <= config.max_angular_speed * dt + 1e-12);
                    prop_assert!(d <= sep_cap + 1e-12);

                    // Bucket compatibility (same logic as before).
                    let key_a = BucketKey {
                        space_key: spatial_binner.key_for(a.equ_coord()),
                        time_bin: time_binner.bin_for(a.mjd_tt()),
                    };
                    let key_b = BucketKey {
                        space_key: spatial_binner.key_for(b.equ_coord()),
                        time_bin: time_binner.bin_for(b.mjd_tt()),
                    };

                    let spatial_neighbors = spatial_binner.neighbors(key_a.space_key, search_radius);
                    prop_assert!(spatial_neighbors.into_iter().any(|k| k == key_b.space_key));

                    // Allowed time bins when allow_same_timebin=false.
                    let bin_width = time_binner.bin_width().max(1e-12);
                    let max_steps = (config.max_dt / bin_width).ceil().max(0.0) as i64;
                    let allowed_bins: HashSet<i64> =
                        (1..=max_steps).map(|dk| key_a.time_bin.0 + dk).collect();

                    prop_assert!(allowed_bins.contains(&key_b.time_bin.0));
                }
            }
        }
    }
}
