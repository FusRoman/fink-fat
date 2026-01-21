use std::{
    collections::HashMap,
    fmt,
    ops::{Deref, DerefMut},
};

use camino::Utf8Path;
use fink_fat_engine::{
    Alert,
    engine_config::{EngineConfig, score_config::ScoreConfig},
    graph::score::ScoredEdge,
    night_id::NightId,
    seeding::{seed_id::SeedId, seed_node::SeedNode},
    spacetime_bucket::{healpix_binner::HealpixBinner, uniform_time_binner::UniformTimeBinner},
};

use crate::io::{read_bin, write_bin};
use anyhow::Result;
use rand::{Rng, RngCore, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

use crate::{
    bin_utils::infer_t0_mjd_tt,
    dataset::ztf_alerts::{AlertStoreWithTruth, NightStore},
    seeding::seed_gen::generate_pairs_and_triplets,
};

use rayon::prelude::*;

#[derive(Debug, Serialize, Deserialize)]
pub struct NightSeeds {
    pub nid: NightId,
    pub seeds: Vec<SeedNode>,
    pub truth: Vec<Option<i32>>,
}

impl fmt::Display for NightSeeds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n_seeds = self.seeds.len();

        let (n_truth, n_none) = self
            .truth
            .iter()
            .fold((0usize, 0usize), |(t, n), v| match v {
                Some(_) => (t + 1, n),
                None => (t, n + 1),
            });

        write!(
            f,
            "NightSeeds(nid={}, seeds={}, truth: {} matched / {} unknown)",
            self.nid, n_seeds, n_truth, n_none,
        )
    }
}

impl NightSeeds {
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

        let truth: Vec<Option<i32>> = seeds.iter().map(|s| store.seed_truth_id(s)).collect();

        Ok(NightSeeds { nid, seeds, truth })
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
        let mut truth: Vec<Option<i32>> = Vec::new();

        for (tid, alerts) in by_traj.iter() {
            if alerts.len() >= 2 {
                for w in alerts.windows(2) {
                    let sid = SeedId(*next_seed_id);
                    *next_seed_id += 1;

                    if let Some(node) =
                        SeedNode::from_pair(sid, nid, w[0], w[1], max_speed_rad_per_day)
                    {
                        seeds.push(node);
                        truth.push(Some(*tid));
                    }
                }
            }

            if include_triplets && alerts.len() >= 3 {
                for w in alerts.windows(3) {
                    let sid = SeedId(*next_seed_id);
                    *next_seed_id += 1;

                    let node = SeedNode::from_triplet(sid, nid, w[0], w[1], w[2]);
                    seeds.push(node);
                    truth.push(Some(*tid));
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
                    truth.push(None);
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
                truth.push(None);
                n_false_added += 1;
            }
        }

        NightSeeds { nid, seeds, truth }
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
            .zip(self.truth.iter())
            .filter_map(|(s, t)| if t.is_some() { Some(s) } else { None })
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
            for (seed, truth) in ns.seeds.iter().zip(ns.truth.iter()) {
                if truth.is_some() {
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
                let n_true = ns.truth.iter().filter(|t| t.is_some()).count();
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
            .filter(|t| t.is_some())
            .count()
    }

    pub fn total_false_seeds(&self) -> usize {
        self.inner
            .values()
            .flat_map(|ns| ns.truth.iter())
            .filter(|t| t.is_none())
            .count()
    }

    /// Iterate over all seeds with their night id.
    pub fn iter_seeds(&self) -> impl Iterator<Item = (NightId, &SeedNode, Option<i32>)> {
        self.inner.iter().flat_map(|(&nid, ns)| {
            ns.seeds
                .iter()
                .zip(ns.truth.iter())
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
        &self,
        cfg: &ScoreConfig,
        horizon: u32,
        balance_per_delta: bool,
        max_edges_per_class: Option<usize>,
        seed: Option<u64>,
    ) -> LabeledEdgesByDelta {
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

fn push_true_edges_only(
    src: &NightSeeds,
    dst: &NightSeeds,
    cfg: &ScoreConfig,
    delta: u32,
    edges_true: &mut Vec<LabeledEdge>,
) {
    // Group seeds by trajectory id (only Some(tid))
    let mut src_by_tid: HashMap<i32, Vec<&SeedNode>> = HashMap::new();
    for (s, t) in src.seeds.iter().zip(src.truth.iter()) {
        if let Some(tid) = t {
            src_by_tid.entry(*tid).or_default().push(s);
        }
    }

    let mut dst_by_tid: HashMap<i32, Vec<&SeedNode>> = HashMap::new();
    for (s, t) in dst.seeds.iter().zip(dst.truth.iter()) {
        if let Some(tid) = t {
            dst_by_tid.entry(*tid).or_default().push(s);
        }
    }

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
                edges_true.push(LabeledEdge { same: true, edge });
            }
        }
    }
}

fn sample_false_edges(
    src: &NightSeeds,
    dst: &NightSeeds,
    cfg: &ScoreConfig,
    delta: u32,
    target_false: usize,
    rng: &mut StdRng,
    edges_false: &mut Vec<LabeledEdge>,
) {
    // Build pools restricted to Some(tid) to ensure “different asteroid”.
    let src_pool: Vec<(&SeedNode, i32)> = src
        .seeds
        .iter()
        .zip(src.truth.iter())
        .filter_map(|(s, t)| t.map(|tid| (s, tid)))
        .collect();

    let dst_pool: Vec<(&SeedNode, i32)> = dst
        .seeds
        .iter()
        .zip(dst.truth.iter())
        .filter_map(|(s, t)| t.map(|tid| (s, tid)))
        .collect();

    if src_pool.is_empty() || dst_pool.is_empty() || target_false == 0 {
        return;
    }

    let mut accepted = 0usize;
    let mut attempts = 0usize;

    // Rejection sampling: scoring may gate out many candidates.
    // Keep this bounded.
    let max_attempts = (target_false.saturating_mul(50)).max(10_000);

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
pub struct LabeledEdge {
    pub same: bool,
    pub edge: ScoredEdge,
}

/// Buckets of labeled edges by night separation `delta` (1..=horizon).
pub type LabeledEdgesByDelta = HashMap<u32, Vec<LabeledEdge>>;

/// Display-friendly wrapper around `LabeledEdgesByDelta`.
#[derive(Debug, Clone)]
pub struct LabeledEdgesByDeltaDisplay<'a>(pub &'a LabeledEdgesByDelta);

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
