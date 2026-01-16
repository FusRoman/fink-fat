use std::{collections::HashMap, fmt};

use fink_fat_engine::{
    engine_config::EngineConfig,
    night_id::NightId,
    seeding::seed_node::SeedNode,
    spacetime_bucket::{healpix_binner::HealpixBinner, uniform_time_binner::UniformTimeBinner},
};

use anyhow::Result;

use crate::{
    bin_utils::{fmt_ms, infer_t0_mjd_tt},
    dataset::ztf_alerts::{AlertStoreWithTruth, NightStore},
    seeding::seed_gen::generate_pairs_and_triplets,
};

use rayon::prelude::*;

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

pub type SeedStore = HashMap<NightId, NightSeeds>;

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
    ) -> Result<SeedStore> {
        night_store
            .par_iter()
            .map(|(&nid, store)| {
                println!("\nProcessing nid={}", nid);

                let seeds = Self::generate_seeds_from_store(
                    store,
                    &engine_cfg,
                    nid,
                    healpix_depth,
                    time_bin_days,
                    pairs_only,
                )?;

                Ok((nid, seeds))
            })
            .collect::<Result<SeedStore>>()
    }
}

pub fn get_seeds_from_seed_store(
    seed_store: &SeedStore,
    nid: NightId,
    seed_id: usize,
) -> &SeedNode {
    seed_store.get(&nid).unwrap().seeds.get(seed_id).unwrap()
}
