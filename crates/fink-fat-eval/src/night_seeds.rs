use std::{
    collections::HashMap,
    fmt,
    ops::{Deref, DerefMut},
};

use fink_fat_engine::{
    Alert,
    engine_config::EngineConfig,
    night_id::NightId,
    seeding::{seed_id::SeedId, seed_node::SeedNode},
    spacetime_bucket::{healpix_binner::HealpixBinner, uniform_time_binner::UniformTimeBinner},
};

use anyhow::Result;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{
    bin_utils::{fmt_ms, infer_t0_mjd_tt},
    dataset::ztf_alerts::{AlertStoreWithTruth, NightStore},
    seeding::seed_gen::generate_pairs_and_triplets,
};

use rayon::prelude::*;

#[derive(Debug)]
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
        let n_alerts = store.store.alerts.len();
        eprintln!("  alerts: {}", n_alerts);

        let t0 = infer_t0_mjd_tt(&store);
        let spatial = HealpixBinner::new(healpix_depth);
        let time = UniformTimeBinner::new(time_bin_days, t0);

        let t_gen = std::time::Instant::now();
        let out = generate_pairs_and_triplets(
            &store,
            nid,
            &spatial,
            &time,
            &engine_cfg.pairs,
            &engine_cfg.triplets,
            None,
        )?;

        eprintln!(
            "  seeding: bucket={:.3}ms pairs={:.3}ms pair_feat={:.3}ms triplets={:.3}ms trip_feat={:.3}ms",
            fmt_ms(out.timings.bucket_index),
            fmt_ms(out.timings.pairs),
            fmt_ms(out.timings.pair_features),
            fmt_ms(out.timings.triplets),
            fmt_ms(out.timings.triplet_features),
        );
        eprintln!(
            "  seeds: pairs={} triplets={} (elapsed {:.3} ms)",
            out.pair_seeds.len(),
            out.triplet_seeds.len(),
            fmt_ms(t_gen.elapsed())
        );

        let mut seeds = out.pair_seeds;
        if !pairs_only {
            seeds.extend(out.triplet_seeds);
        }

        let truth: Vec<Option<i32>> = seeds.iter().map(|s| store.seed_truth_id(s)).collect();

        Ok(NightSeeds { nid, seeds, truth })
    }
}

#[derive(Debug, Default)]
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

        // Deterministic per-night RNG split: base_seed XOR hash(nid)
        // (stable across runs as long as NightId is stable).
        for (&nid, night) in nights.iter() {
            let night_seed = seed.map(|s| s ^ (nid.0 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let night_seeds = seed_store_one_night_from_truth(
                nid,
                night,
                include_triplets,
                false_to_true_ratio,
                night_seed,
                max_speed_rad_per_day,
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
}

/// Build `NightSeeds` for one night.
fn seed_store_one_night_from_truth(
    nid: NightId,
    night: &AlertStoreWithTruth,
    include_triplets: bool,
    false_to_true_ratio: f64,
    seed: Option<u64>,
    max_speed_rad_per_day: Option<f64>,
) -> NightSeeds {
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
    let mut next_seed_id: u64 = 0;

    for (tid, alerts) in by_traj.iter() {
        if alerts.len() >= 2 {
            for w in alerts.windows(2) {
                let sid = SeedId(next_seed_id);
                next_seed_id += 1;

                if let Some(node) = SeedNode::from_pair(sid, nid, w[0], w[1], max_speed_rad_per_day)
                {
                    seeds.push(node);
                    truth.push(Some(*tid));
                }
            }
        }

        if include_triplets && alerts.len() >= 3 {
            for w in alerts.windows(3) {
                let sid = SeedId(next_seed_id);
                next_seed_id += 1;

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

            let sid = SeedId(next_seed_id);
            next_seed_id += 1;

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

            let sid = SeedId(next_seed_id);
            next_seed_id += 1;

            let node = SeedNode::from_triplet(sid, nid, trip[0], trip[1], trip[2]);
            seeds.push(node);
            truth.push(None);
            n_false_added += 1;
        }
    }

    NightSeeds { nid, seeds, truth }
}
