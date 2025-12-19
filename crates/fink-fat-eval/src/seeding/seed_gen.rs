//! Seed generation helpers for evaluation workloads.
//!
//! Overview
//! --------
//! This module bridges `fink-fat-eval` ingestion (`AlertStoreWithTruth`) and the
//! `fink-fat-engine` intra-night seeding pipeline.
//!
//! It provides a small, reusable API to:
//! - build a spatio-temporal bucket index,
//! - generate `(a, b)` pairs,
//! - generate `(a, b, c)` triplets from pairs,
//! - optionally extract `SeedNode` features for pairs and triplets.
//!
//! This module is designed to be called from binaries (`src/bin/*.rs`) or examples,
//! but lives in the library so experiments remain composable and testable.

use std::time::{Duration, Instant};

use anyhow::Result;

use fink_fat_engine::{
    AlertId,
    engine_config::{pair_config::PairConfig, triplet_config::TripletConfig},
    night_id::NightId,
    seeding::{
        pairs::{Pairs, extract_pair_features, generate_pairs},
        seed_node::SeedNode,
        triplets::{Triplets, extract_triplet_features, generate_triplets_from_pairs},
    },
    spacetime_bucket::{
        bucket::{BucketIndex, build_bucket_index},
        spatial_binner::SpatialBinner,
        time_binner::TimeBinner,
    },
};

use crate::dataset::ztf_alerts::AlertStoreWithTruth;

/// Timings for each step of the seeding pipeline.
#[derive(Debug, Clone, Copy, Default)]
pub struct SeedGenTimings {
    pub bucket_index: Duration,
    pub pairs: Duration,
    pub pair_features: Duration,
    pub triplets: Duration,
    pub triplet_features: Duration,
}

/// Outputs of the intra-night seeding pipeline for evaluation.
#[derive(Debug)]
pub struct SeedGenOutput {
    /// Spatio-temporal bucket index used for the generation.
    pub bucket_index: BucketIndex<AlertId>,
    /// Generated ordered pairs `(a, b)`.
    pub pairs: Pairs,
    /// Generated ordered triplets `(a, b, c)`.
    pub triplets: Triplets,
    /// Feature vectors for pairs (one `SeedNode` per retained pair).
    pub pair_seeds: Vec<SeedNode>,
    /// Feature vectors for triplets (one `SeedNode` per triplet).
    pub triplet_seeds: Vec<SeedNode>,
    /// Per-step timings (useful in eval binaries).
    pub timings: SeedGenTimings,
}

/// Generate pairs + triplets from an [`AlertStoreWithTruth`].
///
/// Parameters
/// ----------
/// store : &AlertStoreWithTruth
///     Engine-ready alert store plus a truth sidecar.
/// night_id : NightId
///     Night identifier attached to produced `SeedNode`s.
/// spatial_binner : &Bs
///     Spatial binning strategy (e.g. HEALPix).
/// time_binner : &Bt
///     Time binning strategy (e.g. uniform bins).
/// pair_cfg : &PairConfig
///     Pair-generation thresholds.
/// triplet_cfg : &TripletConfig
///     Triplet-generation thresholds.
/// max_speed_rad_per_day : Option<f64>
///     Optional max angular speed for pair-feature extraction (filters seeds).
///
/// Returns
/// -------
/// SeedGenOutput
///     Bucket index, pairs, triplets, extracted seed features, and timings.
///
/// Notes
/// -----
/// - This function does *not* require truth association (`trajectory_id`) to exist,
///   but it is designed for evaluation workloads where it usually does.
/// - The bucket index is built once and reused for pairs and triplets.
/// - Pair and triplet feature extraction is delegated to the engine:
///   [`SeedNode::from_pair`] and [`SeedNode::from_triplet`].
pub fn generate_pairs_and_triplets<Bs: SpatialBinner, Bt: TimeBinner>(
    store: &AlertStoreWithTruth,
    night_id: NightId,
    spatial_binner: &Bs,
    time_binner: &Bt,
    pair_cfg: &PairConfig,
    triplet_cfg: &TripletConfig,
    max_speed_rad_per_day: Option<f64>,
) -> Result<SeedGenOutput> {
    // ---- 1) Bucket index
    let t0 = Instant::now();
    let bucket_index =
        build_bucket_index(store.store.alerts.as_slice(), spatial_binner, time_binner);
    let dt_bucket = t0.elapsed();

    // ---- 2) Pairs
    let t1 = Instant::now();
    let pairs = generate_pairs(
        &bucket_index,
        store.store.alerts.as_slice(),
        spatial_binner,
        time_binner,
        pair_cfg,
    );
    let dt_pairs = t1.elapsed();

    // ---- 3) Pair features
    let t2 = Instant::now();
    let pair_seeds = extract_pair_features(&store.store, &pairs, night_id, max_speed_rad_per_day);
    let dt_pair_feat = t2.elapsed();

    // ---- 4) Triplets
    let t3 = Instant::now();
    let triplets = generate_triplets_from_pairs(
        &bucket_index,
        store.store.alerts.as_slice(),
        spatial_binner,
        time_binner,
        triplet_cfg,
        &pairs,
    );
    let dt_trips = t3.elapsed();

    // ---- 5) Triplet features
    let t4 = Instant::now();
    let triplet_seeds = extract_triplet_features(&store.store, &triplets, night_id);
    let dt_trip_feat = t4.elapsed();

    Ok(SeedGenOutput {
        bucket_index,
        pairs,
        triplets,
        pair_seeds,
        triplet_seeds,
        timings: SeedGenTimings {
            bucket_index: dt_bucket,
            pairs: dt_pairs,
            pair_features: dt_pair_feat,
            triplets: dt_trips,
            triplet_features: dt_trip_feat,
        },
    })
}

/// Convenience helper: run seeding, but return only (pairs, triplets).
///
/// This is useful if you want to benchmark generation only, without the cost of
/// building `SeedNode`s.
pub fn generate_pairs_and_triplets_ids_only<Bs: SpatialBinner, Bt: TimeBinner>(
    store: &AlertStoreWithTruth,
    spatial_binner: &Bs,
    time_binner: &Bt,
    pair_cfg: &PairConfig,
    triplet_cfg: &TripletConfig,
) -> (Pairs, Triplets, SeedGenTimings) {
    let t0 = Instant::now();
    let bucket_index =
        build_bucket_index(store.store.alerts.as_slice(), spatial_binner, time_binner);
    let dt_bucket = t0.elapsed();

    let t1 = Instant::now();
    let pairs = generate_pairs(
        &bucket_index,
        store.store.alerts.as_slice(),
        spatial_binner,
        time_binner,
        pair_cfg,
    );
    let dt_pairs = t1.elapsed();

    let t2 = Instant::now();
    let triplets = generate_triplets_from_pairs(
        &bucket_index,
        store.store.alerts.as_slice(),
        spatial_binner,
        time_binner,
        triplet_cfg,
        &pairs,
    );
    let dt_trips = t2.elapsed();

    (
        pairs,
        triplets,
        SeedGenTimings {
            bucket_index: dt_bucket,
            pairs: dt_pairs,
            triplets: dt_trips,
            ..Default::default()
        },
    )
}
