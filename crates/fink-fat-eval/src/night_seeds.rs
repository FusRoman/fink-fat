use std::{
    collections::HashMap,
    fmt,
    ops::{Deref, DerefMut},
};

use ahash::AHashMap;
use camino::Utf8Path;
use fink_fat_engine::{
    Alert,
    engine_config::{EngineConfig, score_config::ScoreConfig},
    graph::edge::{Edge, edge_id::EdgeId, score::ScoredEdge},
    night_id::NightId,
    seeding::{seed_id::SeedId, seed_node::SeedNode},
    spacetime_bucket::{healpix_binner::HealpixBinner, uniform_time_binner::UniformTimeBinner},
};

use crate::{
    cli::scoring::Cli,
    dataset::{ParquetSource, ingest_config::AlertIngestConfig, ztf_alerts::collect_nights},
    io::{read_bin, write_bin},
};
use anyhow::{Context, Result};
use rand::{Rng, RngCore, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

use crate::{
    bin_utils::infer_t0_mjd_tt,
    dataset::ztf_alerts::{AlertStoreWithTruth, NightStore},
    seeding::seed_gen::generate_pairs_and_triplets,
};

use rayon::prelude::*;

/// Container holding all intra-night seeds and their associated truth information.
///
/// `NightSeeds` represents the **complete seeding output for a single night**
/// and serves as the primary unit exchanged between:
/// - intra-night seeding,
/// - inter-night edge generation,
/// - scoring and evaluation pipelines.
///
/// It bundles together:
/// - the ordered list of generated seeds,
/// - per-seed truth labels (when available),
/// - pre-aggregated statistics derived from truth labels for fast evaluation.
///
/// This structure is intentionally **read-only after construction** and is
/// designed to be cheaply shared across evaluation routines.
///
/// Invariants
/// ----------
/// - All `seeds` belong to the same night `nid`.
/// - `seeds` are sorted by increasing `plane.epoch_mid`.
/// - `truth` contains **exactly one entry per seed**.
/// - `truth_counts` is consistent with `truth`:
///   for any truth id `t`,
///   `truth_counts[t] == number of seeds s such that truth[s.seed_id] == Some(t)`.
///
/// Notes
/// -----
/// - A seed with `truth = None` corresponds to:
///   - either an unassociated detection set,
///   - or a seed whose members have inconsistent truth labels.
/// - Truth information is used **only for evaluation and calibration**;
///   it must never influence the operational linking or scoring logic.
///
/// Typical usage
/// -------------
/// - Computing the number of theoretically possible true edges between nights.
/// - Evaluating recall ceilings for a given seeding or gating configuration.
/// - Producing labeled datasets for score and gate optimization.
///
/// See also
/// --------
/// - [`SeedNode`] – compact representation of an intra-night seed.
/// - [`generate_seeds_from_store`] – construction of `NightSeeds` from alerts.
/// - [`nb_true_possible_edges`] – combinatorial count of true inter-night edges.
#[derive(Debug, Serialize, Deserialize)]
pub struct NightSeeds {
    /// Night identifier shared by all seeds in this container.
    ///
    /// This is typically the survey-specific night index (e.g. LSST night),
    /// and is used to:
    /// - enforce temporal ordering,
    /// - group seeds for inter-night linking,
    /// - label outputs during evaluation and logging.
    pub nid: NightId,

    /// Ordered list of all intra-night seeds generated for this night.
    ///
    /// Each [`SeedNode`] represents a pair or triplet of detections fitted
    /// by a local tangent-plane kinematic model.
    ///
    /// Ordering
    /// --------
    /// Seeds are sorted by increasing `plane.epoch_mid`.  
    /// This ordering is **required** by downstream components such as:
    /// - temporal gating,
    /// - directed edge generation,
    /// - reproducible iteration during evaluation.
    pub seeds: Vec<SeedNode>,

    /// Per-seed truth association map.
    ///
    /// Maps each [`SeedId`] to:
    /// - `Some(truth_id)` if the seed is fully associated with a known object,
    /// - `None` if the seed has no valid or consistent truth association.
    ///
    /// Semantics
    /// ---------
    /// - `truth_id` is an opaque integer label (typically an asteroid identifier).
    /// - Two seeds sharing the same `truth_id` are considered to belong to the
    ///   same physical object.
    ///
    /// Usage
    /// -----
    /// This map is used exclusively for:
    /// - labeling candidate edges as true / false,
    /// - computing evaluation metrics (ROC, PR, recall ceilings),
    /// - generating frozen datasets for optimization.
    pub truth: AHashMap<SeedId, Option<i32>>,

    /// Pre-aggregated counts of seeds per truth identifier.
    ///
    /// For each `truth_id = t`, this map stores:
    /// ```
    /// truth_counts[t] = number of seeds s such that truth[s.seed_id] == Some(t)
    /// ```
    ///
    /// Motivation
    /// ----------
    /// This field exists to accelerate evaluation routines that require
    /// **combinatorial counts**, such as:
    /// - the number of theoretically possible true edges between two nights,
    /// - upper bounds on achievable recall.
    ///
    /// By caching these counts at construction time, expensive per-call
    /// recomputation over all seeds is avoided.
    ///
    /// Notes
    /// -----
    /// - Seeds with `truth = None` are not represented in this map.
    /// - This field must remain consistent with `truth`; it should be computed
    ///   once during construction and treated as immutable.
    pub truth_counts: AHashMap<i32, u64>,
}

impl fmt::Display for NightSeeds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n_seeds = self.seeds.len();

        let (n_truth, n_none) = self
            .truth
            .iter()
            .fold((0usize, 0usize), |(t, n), v| match v {
                (_, Some(_)) => (t + 1, n),
                (_, None) => (t, n + 1),
            });

        write!(
            f,
            "NightSeeds(nid={}, seeds={}, truth: {} matched / {} unknown)",
            self.nid, n_seeds, n_truth, n_none,
        )
    }
}

impl NightSeeds {
    pub fn get_truth(&self, seedid: &SeedId) -> Option<i32> {
        self.truth.get(seedid).copied().flatten()
    }

    pub fn edge_truth(&self, right: &NightSeeds, edge: &Edge) -> Option<bool> {
        let t_from = self.get_truth(&edge.from.seed_id)?;
        let t_to = right.get_truth(&edge.to.seed_id)?;

        Some(t_from == t_to)
    }

    fn build_truth_counts(truth: &AHashMap<SeedId, Option<i32>>) -> AHashMap<i32, u64> {
        let mut m = AHashMap::new();
        for tid in truth.values().copied().flatten() {
            *m.entry(tid).or_insert(0) += 1;
        }
        m
    }

    /// Compute the number of *theoretically possible* true inter-night edges.
    ///
    /// This function counts how many **true edges could exist in principle**
    /// between the seeds of `self` (left night) and `right` (right night),
    /// assuming no geometric, temporal, or kinematic gating.
    ///
    /// Definition
    /// ----------
    /// Two seeds form a *true edge* if they share the same truth identifier
    /// (`truth_id`). For a given `truth_id = t`:
    ///
    /// ```text
    /// possible_true_edges(t) = n_left(t) × n_right(t)
    /// ```
    ///
    /// where:
    /// - `n_left(t)`  is the number of seeds in `self` associated with `t`,
    /// - `n_right(t)` is the number of seeds in `right` associated with `t`.
    ///
    /// The total number of possible true edges is the sum over all shared
    /// truth identifiers:
    ///
    /// ```text
    /// Σ_t n_left(t) × n_right(t)
    /// ```
    ///
    /// Purpose
    /// -------
    /// This quantity represents an **upper bound on recall** for any inter-night
    /// linking or scoring configuration:
    /// - if a pipeline recovers `N_true` edges,
    /// - the maximum achievable recall is `N_true / nb_true_possible_edges`.
    ///
    /// It is therefore used exclusively for:
    /// - evaluation and benchmarking,
    /// - diagnostic reporting,
    /// - calibration of gates and scores.
    ///
    /// Performance
    /// -----------
    /// - Runs in `O(min(U_left, U_right))`, where `U_*` is the number of distinct
    ///   truth identifiers present in each night.
    /// - Uses pre-aggregated [`truth_counts`] to avoid scanning individual seeds.
    /// - Iterates over the smaller map to minimize hash lookups.
    ///
    /// Notes
    /// -----
    /// - Seeds with `truth = None` are ignored by construction.
    /// - This function performs **no allocation**.
    /// - The result depends only on truth labels and is independent of
    ///   spatial or temporal constraints.
    ///
    /// Parameters
    /// ----------
    /// right : &NightSeeds
    ///     The seed container of the later night.
    ///
    /// Returns
    /// -------
    /// u64
    ///     The total number of theoretically possible true edges between the two nights.
    ///
    /// See also
    /// --------
    /// - [`truth_counts`] – cached per-night counts of seeds per truth identifier.
    /// - [`EdgeSeparationMetrics::n_true_possible`] – usage in evaluation summaries.
    pub fn nb_true_possible_edges(&self, right: &NightSeeds) -> u64 {
        // Iterate over the smaller truth-count map to reduce the number
        // of hash lookups and improve cache locality.
        if self.truth_counts.len() <= right.truth_counts.len() {
            // For each truth_id present in the left night:
            // - retrieve how many seeds share the same truth_id in the right night,
            // - add the Cartesian product cl × cr to the total.
            self.truth_counts
                .iter()
                .map(|(tid, &cl)| {
                    // If the truth_id does not exist in the right night,
                    // there are zero possible true edges for this id.
                    right.truth_counts.get(tid).map_or(0, |&cr| cl * cr)
                })
                .sum()
        } else {
            // Symmetric case: iterate over the right night if it has fewer
            // distinct truth identifiers.
            right
                .truth_counts
                .iter()
                .map(|(tid, &cr)| self.truth_counts.get(tid).map_or(0, |&cl| cl * cr))
                .sum()
        }
    }

    /// Generate seeds for a single night from its alert store.
    ///
    /// Parameters
    /// ----------
    /// store : &AlertStoreWithTruth
    ///     The alert store for the night.
    /// engine_cfg : &EngineConfig
    ///     The engine configuration.
    /// nid : i32
    ///     The night ID.
    /// healpix_depth : u8
    ///     The HEALPix depth for spatial binning.
    /// time_bin_days : f64
    ///     The time bin size in days for temporal binning.
    /// pairs_only : bool
    ///     If true, only generate pair seeds; otherwise, generate both pairs and triplets.
    ///
    /// Returns
    /// -------
    /// Result<NightSeeds>
    ///     The generated NightSeeds.
    pub fn generate_seeds_from_store(
        store: &AlertStoreWithTruth,
        engine_cfg: &EngineConfig,
        nid: NightId,
        healpix_depth: u8,
        time_bin_days: f64,
        pairs_only: bool,
    ) -> Result<Self> {
        let t0 = infer_t0_mjd_tt(&store);
        let spatial = HealpixBinner::new(healpix_depth);
        let time = UniformTimeBinner::new(time_bin_days, t0);

        let out = generate_pairs_and_triplets(
            &store,
            nid,
            &spatial,
            &time,
            &engine_cfg.pairs,
            &engine_cfg.triplets,
            None,
        )?;

        let mut seeds = out.pair_seeds;
        if !pairs_only {
            seeds.extend(out.triplet_seeds);
        }
        // Sort seeds by observation time.
        // very important for edge generation as it suppose a time ordering
        seeds.sort_by(|a, b| a.plane.epoch_mid.total_cmp(&b.plane.epoch_mid));

        let truth: AHashMap<SeedId, Option<i32>> = seeds
            .iter()
            .map(|s| (s.seed_id, store.seed_truth_id(s)))
            .collect();

        let truth_counts = NightSeeds::build_truth_counts(&truth);

        Ok(NightSeeds {
            nid,
            seeds,
            truth,
            truth_counts,
        })
    }

    /// Build `NightSeeds` for one night.
    fn seed_store_one_night_from_truth(
        nid: NightId,
        night: &AlertStoreWithTruth,
        include_triplets: bool,
        false_to_true_ratio: f64,
        seed: Option<u64>,
        max_speed_rad_per_day: Option<f64>,
        next_seed_id: &mut u64,
    ) -> Self {
        // Group alerts by truth trajectory_id (> 0).
        let mut by_traj: HashMap<i32, Vec<&Alert>> = HashMap::new();
        for (idx, alert) in night.store.alerts.iter().enumerate() {
            let tid = night.trajectory_id[idx];
            if tid > 0 {
                by_traj.entry(tid).or_default().push(alert);
            }
        }

        // Sort each trajectory by observation time.
        for alerts in by_traj.values_mut() {
            alerts.sort_by(|a, b| a.mjd_tt.partial_cmp(&b.mjd_tt).unwrap());
        }

        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };

        // Build all true seeds (consecutive pairs + optional triplets).
        let mut seeds: Vec<SeedNode> = Vec::new();
        let mut truth: AHashMap<SeedId, Option<i32>> = AHashMap::new();

        for (tid, alerts) in by_traj.iter() {
            if alerts.len() >= 2 {
                for w in alerts.windows(2) {
                    let sid = SeedId(*next_seed_id);
                    *next_seed_id += 1;

                    if let Some(node) =
                        SeedNode::from_pair(sid, nid, w[0], w[1], max_speed_rad_per_day)
                    {
                        seeds.push(node);
                        truth.insert(sid, Some(*tid));
                    }
                }
            }

            if include_triplets && alerts.len() >= 3 {
                for w in alerts.windows(3) {
                    let sid = SeedId(*next_seed_id);
                    *next_seed_id += 1;

                    let node = SeedNode::from_triplet(sid, nid, w[0], w[1], w[2]);
                    seeds.push(node);
                    truth.insert(sid, Some(*tid));
                }
            }
        }

        let n_true = seeds.len();
        let target_false = ((false_to_true_ratio.max(0.0)) * (n_true as f64)).round() as usize;

        // Prepare pools for false seeds.
        let traj_keys: Vec<i32> = by_traj.keys().copied().collect();

        let mut n_false_added = 0usize;
        let mut attempts = 0usize;
        let max_attempts = target_false.saturating_mul(50).max(10_000);

        while n_false_added < target_false && attempts < max_attempts {
            attempts += 1;

            // Need at least 2 distinct real trajectories to build mismatched seeds.
            if traj_keys.len() < 2 {
                break;
            }

            // Pick two different trajectories.
            let t1 = traj_keys[rng.random_range(0..traj_keys.len())];
            let mut t2 = traj_keys[rng.random_range(0..traj_keys.len())];
            while t2 == t1 {
                t2 = traj_keys[rng.random_range(0..traj_keys.len())];
            }

            let a_list = &by_traj[&t1];
            let b_list = &by_traj[&t2];
            if a_list.is_empty() || b_list.is_empty() {
                continue;
            }

            // Sample a false pair or triplet.
            if !include_triplets || rng.random_bool(0.5) {
                // False pair: one alert from t1, one from t2.
                let a = a_list[rng.random_range(0..a_list.len())];
                let b = b_list[rng.random_range(0..b_list.len())];

                let sid = SeedId(*next_seed_id);
                *next_seed_id += 1;

                if let Some(node) = SeedNode::from_pair(sid, nid, a, b, max_speed_rad_per_day) {
                    seeds.push(node);
                    truth.insert(sid, None);
                    n_false_added += 1;
                }
            } else {
                // False triplet: 2 alerts from t1 + 1 alert from t2.
                if a_list.len() < 2 {
                    continue;
                }

                let a1 = a_list[rng.random_range(0..a_list.len())];
                let a2 = a_list[rng.random_range(0..a_list.len())];
                if a1.id == a2.id {
                    continue;
                }
                let b = b_list[rng.random_range(0..b_list.len())];

                // Ensure members are sorted by time.
                let mut trip = [a1, a2, b];
                trip.sort_by(|x, y| x.mjd_tt.partial_cmp(&y.mjd_tt).unwrap());

                let sid = SeedId(*next_seed_id);
                *next_seed_id += 1;

                let node = SeedNode::from_triplet(sid, nid, trip[0], trip[1], trip[2]);
                seeds.push(node);
                truth.insert(sid, None);
                n_false_added += 1;
            }
        }

        let truth_counts = NightSeeds::build_truth_counts(&truth);

        NightSeeds {
            nid,
            seeds,
            truth,
            truth_counts,
        }
    }

    /// Get all true seeds in this night.
    ///
    /// Returns
    /// -------
    /// Vec<&SeedNode>
    ///    All seeds with known truth (trajectory_id > 0).
    pub fn get_true_seeds(&self) -> Vec<&SeedNode> {
        self.seeds
            .iter()
            .filter_map(|s| {
                if self.truth.get(&s.seed_id).is_some() {
                    Some(s)
                } else {
                    None
                }
            })
            .collect()
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct SeedStore {
    inner: HashMap<NightId, NightSeeds>,
}

impl Deref for SeedStore {
    type Target = HashMap<NightId, NightSeeds>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for SeedStore {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl fmt::Display for SeedStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n_nights = self.inner.len();

        let mut total_seeds = 0usize;
        let mut total_true = 0usize;
        let mut total_false = 0usize;
        let mut total_pairs = 0usize;
        let mut total_triplets = 0usize;

        // Sort nights for deterministic output
        let mut nights: Vec<(&NightId, &NightSeeds)> = self.inner.iter().collect();
        nights.sort_by_key(|(nid, _)| *nid);

        for (_, ns) in nights.iter() {
            total_seeds += ns.seeds.len();
            for seed in ns.seeds.iter() {
                if ns.truth.get(&seed.seed_id).is_some() {
                    total_true += 1;
                } else {
                    total_false += 1;
                }
                match seed.n_obs {
                    2 => total_pairs += 1,
                    3 => total_triplets += 1,
                    _ => {}
                }
            }
        }

        writeln!(f, "SeedStore summary")?;
        writeln!(f, "-----------------")?;
        writeln!(f, "Nights          : {}", n_nights)?;
        writeln!(f, "Total seeds     : {}", total_seeds)?;
        writeln!(f, "  True seeds    : {}", total_true)?;
        writeln!(f, "  False seeds   : {}", total_false)?;
        writeln!(f, "  Pairs         : {}", total_pairs)?;
        writeln!(f, "  Triplets      : {}", total_triplets)?;

        if n_nights > 0 {
            writeln!(f)?;
            writeln!(f, "Per-night breakdown")?;
            writeln!(f, "-------------------")?;

            for (nid, ns) in nights {
                let n = ns.seeds.len();
                let n_true = ns.truth.iter().filter(|(_, t)| t.is_some()).count();
                let n_false = n - n_true;
                let n_pairs = ns.seeds.iter().filter(|s| s.n_obs == 2).count();
                let n_triplets = ns.seeds.iter().filter(|s| s.n_obs == 3).count();

                writeln!(
                    f,
                    "Night {:>6} : {:>6} seeds  (true {:>5}, false {:>5})  | pairs {:>5}, triplets {:>5}",
                    nid.0, n, n_true, n_false, n_pairs, n_triplets
                )?;
            }
        }

        Ok(())
    }
}

impl SeedStore {
    /// Build a `SeedStore` from a `NightStore` (multiple nights), keeping all *true* seeds
    /// and sampling *false* seeds per night using ground-truth.
    ///
    /// Strategy (per night)
    /// --------------------
    /// - **True seeds**: for each `trajectory_id > 0`, sort alerts by `mjd_tt` and create
    ///   **consecutive** pairs `(i, i+1)` and optional triplets `(i, i+1, i+2)`.
    /// - **False seeds**: sample random pairs/triplets whose members come from **different**
    ///   trajectories, until reaching a target budget:
    ///   `target_false = round(false_to_true_ratio * n_true)`.
    ///
    /// Arguments
    /// ---------
    /// * `nights` – Per-night alert stores with truth (`trajectory_id` aligned with dense alerts).
    /// * `include_triplets` – If true, generate true triplets and sample false triplets.
    /// * `false_to_true_ratio` – Number of false seeds to sample per night relative to true seeds.
    /// * `seed` – Optional RNG seed for deterministic sampling (split per-night).
    /// * `max_speed_rad_per_day` – Optional speed sanity check passed to `SeedNode::from_pair`.
    ///
    /// Return
    /// ------
    /// * `SeedStore` – A map `NightId -> NightSeeds` containing seeds for all nights.
    ///
    /// Notes
    /// -----
    /// * Sampling is performed **independently per night** (balanced locally).
    /// * Nights with fewer than 2 distinct `trajectory_id > 0` cannot produce mismatched false seeds;
    ///   such nights will contain only true seeds.
    pub fn seed_store_from_night_store_truth(
        nights: &NightStore,
        include_triplets: bool,
        false_to_true_ratio: f64,
        seed: Option<u64>,
        max_speed_rad_per_day: Option<f64>,
    ) -> Self {
        let mut out = HashMap::with_capacity(nights.len());

        // Deterministic order to ensure stable SeedId assignment across runs.
        let mut night_ids: Vec<NightId> = nights.keys().copied().collect();
        night_ids.sort_unstable();

        let mut next_seed_id: u64 = 0;

        // Deterministic per-night RNG split: base_seed XOR hash(nid)
        // (stable across runs as long as NightId is stable).
        for (&nid, night) in nights.iter() {
            let night_seed = seed.map(|s| s ^ (nid.0 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let night_seeds = NightSeeds::seed_store_one_night_from_truth(
                nid,
                night,
                include_triplets,
                false_to_true_ratio,
                night_seed,
                max_speed_rad_per_day,
                &mut next_seed_id,
            );
            out.insert(nid, night_seeds);
        }

        Self { inner: out }
    }

    /// Generate a store of NightSeeds for multiple nights from their alert stores.
    ///
    /// Parameters
    /// ----------
    /// night_store : HashMap<i32, AlertStoreWithTruth>
    ///     A mapping from night IDs to their corresponding alert stores.
    /// engine_cfg : &EngineConfig
    ///     The engine configuration.
    /// healpix_depth : u8
    ///     The HEALPix depth for spatial binning.
    /// time_bin_days : f64
    ///     The time bin size in days for temporal binning.
    /// pairs_only : bool
    ///     If true, only generate pair seeds; otherwise, generate both pairs and triplets.
    ///
    /// Returns
    /// -------
    /// Result<HashMap<i32, NightSeeds>>
    ///     A mapping from night IDs to their generated NightSeeds.
    pub fn generate_nightseed_store(
        night_store: &NightStore,
        engine_cfg: &EngineConfig,
        healpix_depth: u8,
        time_bin_days: f64,
        pairs_only: bool,
    ) -> Result<Self> {
        let seed_store = night_store
            .par_iter()
            .map(|(&nid, store)| {
                let seeds = NightSeeds::generate_seeds_from_store(
                    store,
                    &engine_cfg,
                    nid,
                    healpix_depth,
                    time_bin_days,
                    pairs_only,
                )?;

                Ok((nid, seeds))
            })
            .collect::<Result<HashMap<NightId, NightSeeds>>>()?;

        Ok(Self { inner: seed_store })
    }

    /// Total number of seeds across all nights.
    pub fn total_seeds(&self) -> usize {
        self.inner.values().map(|ns| ns.seeds.len()).sum()
    }

    /// Total number of true seeds across all nights.
    pub fn total_true_seeds(&self) -> usize {
        self.inner
            .values()
            .flat_map(|ns| ns.truth.iter())
            .filter(|(_, t)| t.is_some())
            .count()
    }

    pub fn total_false_seeds(&self) -> usize {
        self.inner
            .values()
            .flat_map(|ns| ns.truth.iter())
            .filter(|(_, t)| t.is_none())
            .count()
    }

    /// Iterate over all seeds with their night id.
    pub fn iter_seeds(&self) -> impl Iterator<Item = (NightId, &SeedNode, Option<i32>)> {
        self.inner.iter().flat_map(|(&nid, ns)| {
            ns.seeds
                .iter()
                .filter_map(|s| ns.truth.get(&s.seed_id).map(|t| (s, t)))
                .map(move |(s, t)| (nid, s, *t))
        })
    }

    /// Get a seed by night id and seed index.
    ///
    /// Parameters
    /// ----------
    /// nid : NightId
    ///     The night ID.
    /// seed_id : usize
    ///     The index of the seed within the night.
    ///
    /// Returns
    /// -------
    /// &SeedNode
    ///     A reference to the requested SeedNode.
    pub fn get_seeds_from_seed_store(&self, nid: NightId, seed_id: usize) -> &SeedNode {
        self.get(&nid).unwrap().seeds.get(seed_id).unwrap()
    }

    /// Build labeled, scored inter-night edges from this `SeedStore`.
    ///
    /// This constructs **directed** edges `i → j` where `i` belongs to night `n`
    /// and `j` belongs to night `n + delta`, for `delta = 1..=horizon`.
    /// The score is computed with [`ScoredEdge::score`], so all gates are applied.
    ///
    /// Arguments
    /// ---------
    /// * `cfg` – Scoring configuration (gates + weights).
    /// * `horizon` – Maximum night separation to consider (in days / `NightId` units).
    /// * `balance_per_delta` – If true, keep the dataset balanced per delta by
    ///   downsampling the majority class (true vs false) after scoring.
    /// * `max_edges_per_class` – Optional cap per delta and per class (true/false)
    ///   applied **after** balancing (useful to bound memory).
    /// * `seed` – Optional RNG seed for deterministic downsampling.
    ///
    /// Return
    /// ------
    /// * `HashMap<delta, Vec<LabeledEdge>>` – Accepted edges, bucketed by delta.
    ///
    /// Notes
    /// -----
    /// * Truth label:
    ///   - `same=true` iff both endpoints have `Some(traj_id)` and they are equal.
    ///   - Otherwise `same=false`.
    /// * Nights that do not exist in the store for a given `(n, n+delta)` are skipped.
    /// * Complexity can be large: per delta, this is O(|S_n| × |S_{n+delta}|) scoring.
    pub fn labeled_edges_by_delta(
        &'_ self,
        cfg: &ScoreConfig,
        horizon: u32,
        balance_per_delta: bool,
        max_edges_per_class: Option<usize>,
        seed: Option<u64>,
    ) -> LabeledEdgesByDelta<'_> {
        let mut out: LabeledEdgesByDelta = HashMap::new();

        // Deterministic iteration order.
        let mut night_ids: Vec<NightId> = self.inner.keys().copied().collect();
        night_ids.sort_unstable();

        // Helper to fetch a night and quickly access its seeds & truth.
        let get_night = |nid: NightId| -> Option<&NightSeeds> { self.inner.get(&nid) };

        // We'll use one RNG per delta for deterministic downsampling.
        let mut base_rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };

        for delta in 1..=horizon {
            println!("  Scoring edges with delta = {}...\n", delta);

            let mut edges_true: Vec<LabeledEdge> = Vec::new();
            let mut edges_false: Vec<LabeledEdge> = Vec::new();

            for &nid in night_ids.iter() {
                // NightId is a newtype over u32 days, so we can step by +delta safely.
                let nid_to = NightId(nid.0.saturating_add(delta));
                let Some(src) = get_night(nid) else { continue };
                let Some(dst) = get_night(nid_to) else {
                    continue;
                };

                let nb_true_before = edges_true.len();

                // 1) First pass: compute ONLY true edges
                push_true_edges_only(src, dst, cfg, delta, &mut edges_true);

                let n_new_true = edges_true.len() - nb_true_before;

                // 2) Second pass: sample false edges (e.g. balanced with true edges)
                sample_false_edges(
                    src,
                    dst,
                    cfg,
                    delta,
                    n_new_true,
                    &mut base_rng,
                    &mut edges_false,
                );
            }

            // Optional balancing + capping.
            if balance_per_delta {
                // Downsample to min(true,false)
                let k = edges_true.len().min(edges_false.len());
                if k == 0 {
                    // Keep empty or all-one-class as-is (but usually this means no usable supervision)
                    let mut merged = Vec::new();
                    merged.extend(edges_true);
                    merged.extend(edges_false);
                    out.insert(delta, merged);
                    continue;
                }

                // Deterministic RNG stream per delta.
                let delta_seed =
                    base_rng.next_u64() ^ (delta as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let mut rng = StdRng::seed_from_u64(delta_seed);

                downsample_in_place(&mut edges_true, k, &mut rng);
                downsample_in_place(&mut edges_false, k, &mut rng);

                if let Some(cap) = max_edges_per_class {
                    let cap = cap.min(k);
                    edges_true.truncate(cap);
                    edges_false.truncate(cap);
                }
            } else if let Some(cap) = max_edges_per_class {
                // Apply cap independently (no balancing)
                edges_true.truncate(cap);
                edges_false.truncate(cap);
            }

            // Merge (keeping some structure is often useful, but caller asked for "ensemble").
            let mut merged = Vec::with_capacity(edges_true.len() + edges_false.len());
            merged.extend(edges_true);
            merged.extend(edges_false);

            out.insert(delta, merged);
        }

        out
    }

    /// Write the `SeedStore` to a binary file.
    ///
    /// Parameters
    /// ----------
    /// path : &Utf8Path
    ///     The file path to write the binary data to.
    ///
    /// Returns
    /// -------
    /// anyhow::Result<()>
    ///     An empty result indicating success or failure.
    pub fn write(&self, path: &Utf8Path) -> anyhow::Result<()> {
        write_bin(path, self)
    }

    /// Read a `SeedStore` from a binary file.
    ///
    /// Parameters
    /// ----------
    /// path : &Utf8Path
    ///     The file path to read the binary data from.
    ///
    /// Returns
    /// -------
    /// anyhow::Result<SeedStore>
    ///     The read `SeedStore` or an error.
    pub fn read(path: &Utf8Path) -> anyhow::Result<Self> {
        let store: SeedStore = read_bin(path)?;
        Ok(store)
    }
}

pub fn generate_seed_store(cli: &Cli, engine_cfg: &EngineConfig) -> Result<SeedStore> {
    let source = ParquetSource::new(&cli.scan.parquet)
        .with_context(|| format!("failed to open parquet source: {}", cli.scan.parquet))?;

    let ingest_cfg = AlertIngestConfig::default();
    let night_store: NightStore =
        collect_nights(&source, cli.scan.mode.into(), cli.scan.minimal, &ingest_cfg)?;

    SeedStore::generate_nightseed_store(&night_store, &engine_cfg, 8, 1.0, false)
}

fn push_true_edges_only<'a>(
    src: &'a NightSeeds,
    dst: &'a NightSeeds,
    cfg: &ScoreConfig,
    delta: u32,
    edges_true: &mut Vec<LabeledEdge<'a>>,
) {
    // Group seeds by trajectory id (only Some(tid))
    let mut src_by_tid: HashMap<i32, Vec<&'a SeedNode>> = HashMap::new();
    for (s, (_, truth)) in src.seeds.iter().zip(src.truth.iter()) {
        if let Some(truth) = truth {
            src_by_tid.entry(*truth).or_default().push(s);
        }
    }

    let mut dst_by_tid: HashMap<i32, Vec<&'a SeedNode>> = HashMap::new();
    for (s, (_, truth)) in dst.seeds.iter().zip(dst.truth.iter()) {
        if let Some(truth) = truth {
            dst_by_tid.entry(*truth).or_default().push(s);
        }
    }

    let mut edge_id = 0u64; // dummy, will be overwritten

    // Only score matching tid groups
    for (tid, src_list) in src_by_tid.iter() {
        let Some(dst_list) = dst_by_tid.get(tid) else {
            continue;
        };

        for i in src_list.iter() {
            for j in dst_list.iter() {
                let Some(edge) = ScoredEdge::score(i, j, cfg, delta) else {
                    continue;
                };
                let edge = Edge::new(EdgeId(edge_id), i, j, edge.cost, edge.dt_days);
                edge_id += 1;
                edges_true.push(LabeledEdge { same: true, edge });
            }
        }
    }
}

fn sample_false_edges<'a>(
    src: &'a NightSeeds,
    dst: &'a NightSeeds,
    cfg: &ScoreConfig,
    delta: u32,
    target_false: usize,
    rng: &mut StdRng,
    edges_false: &mut Vec<LabeledEdge<'a>>,
) {
    // Build pools restricted to Some(tid) to ensure “different asteroid”.
    let src_pool: Vec<(&'a SeedNode, i32)> = src
        .seeds
        .iter()
        .zip(src.truth.iter())
        .filter_map(|(s, (_, truth))| truth.map(|tid| (s, tid)))
        .collect();

    let dst_pool: Vec<(&'a SeedNode, i32)> = dst
        .seeds
        .iter()
        .zip(dst.truth.iter())
        .filter_map(|(s, (_, truth))| truth.map(|tid| (s, tid)))
        .collect();

    if src_pool.is_empty() || dst_pool.is_empty() || target_false == 0 {
        return;
    }

    let mut accepted = 0usize;
    let mut attempts = 0usize;

    // Rejection sampling: scoring may gate out many candidates.
    // Keep this bounded.
    let max_attempts = (target_false.saturating_mul(50)).max(10_000);

    let mut edge_id = 0u64; // dummy, will be overwritten

    while accepted < target_false && attempts < max_attempts {
        attempts += 1;

        let (i, ti) = src_pool[rng.random_range(0..src_pool.len())];
        let (j, tj) = dst_pool[rng.random_range(0..dst_pool.len())];

        if ti == tj {
            continue; // would be true edge, skip
        }

        let Some(edge) = ScoredEdge::score(i, j, cfg, delta) else {
            continue;
        };

        let edge = Edge::new(EdgeId(edge_id), i, j, edge.cost, edge.dt_days);
        edge_id += 1;
        edges_false.push(LabeledEdge { same: false, edge });
        accepted += 1;
    }
}

/// Randomly downsample `v` to exactly `k` elements (in-place).
///
/// This uses a partial Fisher-Yates shuffle (O(n) but efficient enough)
/// and then truncates.
fn downsample_in_place<T>(v: &mut Vec<T>, k: usize, rng: &mut StdRng) {
    if v.len() <= k {
        return;
    }
    // Partial shuffle: shuffle first k items into front.
    let n = v.len();
    for i in 0..k {
        let j = rng.random_range(i..n);
        v.swap(i, j);
    }
    v.truncate(k);
}

/// Scored edge with a truth label.
///
/// Notes
/// -----
/// `same=true` means both endpoints have a truth id and they match.
#[derive(Clone, Debug)]
pub struct LabeledEdge<'a> {
    pub same: bool,
    pub edge: Edge<'a>,
}

impl<'a> LabeledEdge<'a> {
    pub fn from_edge(edge: Edge<'a>, left: &NightSeeds, right: &NightSeeds) -> Self {
        let same = match (
            left.get_truth(&edge.from.seed_id),
            right.get_truth(&edge.to.seed_id),
        ) {
            (Some(tid1), Some(tid2)) => tid1 == tid2,
            _ => false,
        };
        Self { same, edge }
    }
}

/// Buckets of labeled edges by night separation `delta` (1..=horizon).
pub type LabeledEdgesByDelta<'a> = HashMap<u32, Vec<LabeledEdge<'a>>>;

/// Display-friendly wrapper around `LabeledEdgesByDelta`.
#[derive(Debug, Clone)]
pub struct LabeledEdgesByDeltaDisplay<'a>(pub &'a LabeledEdgesByDelta<'a>);
impl<'a> fmt::Display for LabeledEdgesByDeltaDisplay<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let map = self.0;

        let mut deltas: Vec<u32> = map.keys().copied().collect();
        deltas.sort_unstable();

        let mut grand_total = 0usize;
        let mut grand_true = 0usize;
        let mut grand_false = 0usize;

        let mut cost_sum = 0.0f64;
        let mut dt_sum = 0.0f64;

        let mut cost_min = f64::INFINITY;
        let mut cost_max = f64::NEG_INFINITY;
        let mut dt_min = f64::INFINITY;
        let mut dt_max = f64::NEG_INFINITY;

        for &d in deltas.iter() {
            let edges = &map[&d];
            grand_total += edges.len();

            for e in edges.iter() {
                if e.same {
                    grand_true += 1;
                } else {
                    grand_false += 1;
                }

                let c = e.edge.cost;
                let dt = e.edge.dt_days;

                // Ignore non-finite defensively (should not happen if scoring is sane).
                if c.is_finite() {
                    cost_sum += c;
                    cost_min = cost_min.min(c);
                    cost_max = cost_max.max(c);
                }
                if dt.is_finite() {
                    dt_sum += dt;
                    dt_min = dt_min.min(dt);
                    dt_max = dt_max.max(dt);
                }
            }
        }

        writeln!(f, "LabeledEdgesByDelta summary")?;
        writeln!(f, "---------------------------")?;
        writeln!(f, "Buckets (delta) : {}", deltas.len())?;
        writeln!(f, "Total edges     : {}", grand_total)?;
        writeln!(f, "  True edges    : {}", grand_true)?;
        writeln!(f, "  False edges   : {}", grand_false)?;

        if grand_total > 0 {
            let cost_mean = cost_sum / (grand_total as f64);
            let dt_mean = dt_sum / (grand_total as f64);

            writeln!(
                f,
                "Cost  (min/mean/max) : {:.6} / {:.6} / {:.6}",
                cost_min, cost_mean, cost_max
            )?;
            writeln!(
                f,
                "dt    (min/mean/max) : {:.6} / {:.6} / {:.6} days",
                dt_min, dt_mean, dt_max
            )?;
        }

        if !deltas.is_empty() {
            writeln!(f)?;
            writeln!(f, "Per-delta breakdown")?;
            writeln!(f, "-------------------")?;

            for d in deltas {
                let edges = &map[&d];
                let n = edges.len();
                let n_true = edges.iter().filter(|e| e.same).count();
                let n_false = n - n_true;

                let mut cmin = f64::INFINITY;
                let mut cmax = f64::NEG_INFINITY;
                let mut csum = 0.0f64;

                let mut dtmin = f64::INFINITY;
                let mut dtmax = f64::NEG_INFINITY;
                let mut dtsum = 0.0f64;

                for e in edges.iter() {
                    let c = e.edge.cost;
                    let dt = e.edge.dt_days;

                    if c.is_finite() {
                        csum += c;
                        cmin = cmin.min(c);
                        cmax = cmax.max(c);
                    }
                    if dt.is_finite() {
                        dtsum += dt;
                        minmax_update(&mut dtmin, &mut dtmax, dt);
                        (&mut dtmin, &mut dtmax, dt);
                    }
                }

                let cmean = if n > 0 { csum / (n as f64) } else { f64::NAN };
                let dtmean = if n > 0 { dtsum / (n as f64) } else { f64::NAN };

                writeln!(
                    f,
                    "Δ={:>2} : {:>8} edges  (true {:>8}, false {:>8})  | cost {:.6}/{:.6}/{:.6}  | dt {:.6}/{:.6}/{:.6} d",
                    d, n, n_true, n_false, cmin, cmean, cmax, dtmin, dtmean, dtmax
                )?;
            }
        }

        Ok(())
    }
}

#[inline]
fn minmax_update(minv: &mut f64, maxv: &mut f64, x: f64) {
    *minv = (*minv).min(x);
    *maxv = (*maxv).max(x);
}

#[cfg(test)]
mod night_seeds_tests {
    use super::*;
    use ahash::AHashMap;
    use prop_test::prelude::{Strategy, prop, prop_assert_eq, proptest};

    /// Build a minimal `NightSeeds` suitable for testing `nb_true_possible_edges`.
    ///
    /// The function under test only depends on `truth_counts`, so we can keep
    /// `seeds` and `truth` empty.
    fn mk_night_with_counts(nid: u32, counts: &[(i32, u64)]) -> NightSeeds {
        let mut truth_counts: AHashMap<i32, u64> = AHashMap::new();
        for &(tid, c) in counts {
            truth_counts.insert(tid, c);
        }
        NightSeeds {
            nid: NightId::new(nid),
            seeds: Vec::new(),
            truth: AHashMap::new(),
            truth_counts,
        }
    }

    /// Slow but obviously-correct reference implementation.
    fn expected_nb_true_possible_edges(
        left: &AHashMap<i32, u64>,
        right: &AHashMap<i32, u64>,
    ) -> u64 {
        let mut total = 0u64;
        for (tid, &cl) in left.iter() {
            if let Some(&cr) = right.get(tid) {
                total = total.saturating_add(cl.saturating_mul(cr));
            }
        }
        total
    }

    #[test]
    fn nb_true_possible_edges_empty_both_is_zero() {
        let left = mk_night_with_counts(1, &[]);
        let right = mk_night_with_counts(2, &[]);
        assert_eq!(left.nb_true_possible_edges(&right), 0);
    }

    #[test]
    fn nb_true_possible_edges_disjoint_truth_ids_is_zero() {
        let left = mk_night_with_counts(1, &[(10, 3), (11, 2)]);
        let right = mk_night_with_counts(2, &[(20, 7), (21, 1)]);
        assert_eq!(left.nb_true_possible_edges(&right), 0);
    }

    #[test]
    fn nb_true_possible_edges_single_overlap_matches_product() {
        let left = mk_night_with_counts(1, &[(42, 3)]);
        let right = mk_night_with_counts(2, &[(42, 5)]);
        assert_eq!(left.nb_true_possible_edges(&right), 15);
    }

    #[test]
    fn nb_true_possible_edges_multiple_overlaps_sum_of_products() {
        // Overlap on {1, 3}; disjoint on {2} and {4}
        let left = mk_night_with_counts(1, &[(1, 2), (2, 10), (3, 4)]);
        let right = mk_night_with_counts(2, &[(1, 7), (3, 1), (4, 99)]);
        // expected: 2*7 + 4*1 = 18
        assert_eq!(left.nb_true_possible_edges(&right), 18);
    }

    #[test]
    fn nb_true_possible_edges_is_symmetric() {
        let left = mk_night_with_counts(1, &[(1, 2), (2, 3), (3, 4)]);
        let right = mk_night_with_counts(2, &[(2, 10), (3, 1)]);
        assert_eq!(
            left.nb_true_possible_edges(&right),
            right.nb_true_possible_edges(&left)
        );
    }

    #[test]
    fn nb_true_possible_edges_same_inputs_equals_sum_of_squares() {
        let left = mk_night_with_counts(1, &[(5, 2), (7, 3)]);
        // expected: 2*2 + 3*3 = 13
        assert_eq!(left.nb_true_possible_edges(&left), 13);
    }

    // -------------------------
    // Property-based tests
    // -------------------------

    /// Strategy: build small maps {truth_id -> count} with bounded values to avoid overflow.
    fn truth_counts_strategy() -> impl Strategy<Value = AHashMap<i32, u64>> {
        // Distinct keys, size 0..50, counts 0..2000
        prop::collection::hash_map(-1000i32..1000i32, 0u64..2000u64, 0..50)
            .prop_map(|hm| hm.into_iter().collect::<AHashMap<_, _>>())
    }

    proptest! {
        #[test]
        fn prop_matches_reference(left in truth_counts_strategy(), right in truth_counts_strategy()) {
            let ln = NightSeeds {
                nid: NightId::new(1),
                seeds: Vec::new(),
                truth: AHashMap::new(),
                truth_counts: left.clone(),
            };
            let rn = NightSeeds {
                nid: NightId::new(2),
                seeds: Vec::new(),
                truth: AHashMap::new(),
                truth_counts: right.clone(),
            };

            let got = ln.nb_true_possible_edges(&rn);
            let exp = expected_nb_true_possible_edges(&left, &right);
            prop_assert_eq!(got, exp);
        }

        #[test]
        fn prop_is_symmetric(left in truth_counts_strategy(), right in truth_counts_strategy()) {
            let ln = NightSeeds {
                nid: NightId::new(1),
                seeds: Vec::new(),
                truth: AHashMap::new(),
                truth_counts: left,
            };
            let rn = NightSeeds {
                nid: NightId::new(2),
                seeds: Vec::new(),
                truth: AHashMap::new(),
                truth_counts: right,
            };

            prop_assert_eq!(ln.nb_true_possible_edges(&rn), rn.nb_true_possible_edges(&ln));
        }

        #[test]
        fn prop_empty_right_gives_zero(left in truth_counts_strategy()) {
            let ln = NightSeeds {
                nid: NightId::new(1),
                seeds: Vec::new(),
                truth: AHashMap::new(),
                truth_counts: left,
            };
            let rn = NightSeeds {
                nid: NightId::new(2),
                seeds: Vec::new(),
                truth: AHashMap::new(),
                truth_counts: AHashMap::new(),
            };

            prop_assert_eq!(ln.nb_true_possible_edges(&rn), 0);
        }
    }
}
