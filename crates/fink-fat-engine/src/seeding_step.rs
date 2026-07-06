use photom::{MJDTT, observation_dataset::observation::Observation};

use crate::{
    engine_config::EngineConfig,
    error::{EngineError, OptionExt},
    pipeline::PipelineStage,
    seeding::{pairs, triplets},
    spacetime_bucket::{
        bucket::build_alert_bucket_index, healpix_binner::HealpixBinner,
        uniform_time_binner::UniformTimeBinner,
    },
    tracklet::{Tracklet, track_storage::TrackId},
};

/// Process one observation night: bucketize, generate pairs and triplets, extract seed features.
///
/// Uses a thread-local [`SeedStore`] for provisional key allocation. Resulting
/// seeds carry placeholder keys that must be replaced with real globally-unique
/// keys by [`finalize_night_seeds`] before insertion into the pipeline seed store.
///
/// Arguments
/// ---------
/// * `night_id` – Identifier of the night being processed.
/// * `alerts` – Alert slice for this night.
/// * `spatial_binner` – Spatial partitioner for (space, time) bucket assignment.
/// * `pair_cfg` – Pair generation configuration.
/// * `triplet_cfg` – Triplet generation configuration.
/// * `time_binner_width` – Time bin width in days.
/// * `night_sink` – Progress sink for the per-night sub-scope (five milestones).
///
/// Return
/// ------
/// * `Ok(NightSeedResult)` – Alert, pair and triplet counts plus sorted seeds
///   with provisional keys.
/// * `Err(EngineError::StageFailed)` – If the alert slice is empty
///   (cannot determine `t0` for time binning).
pub(crate) fn process_one_night(
    alerts: &[Observation],
    spatial_binner: &HealpixBinner,
    cfg: &EngineConfig,
    time_binner_width: f64,
) -> Result<Vec<Tracklet>, EngineError> {
    let t0: MJDTT = alerts
        .iter()
        .min()
        .stage_err(
            PipelineStage::BuildSeeds,
            "cannot determine t0 for time binning: night contains zero alerts",
        )?
        .mjd_tt();

    let time_binner = UniformTimeBinner::new(t0, time_binner_width);
    tracing::trace!(t0, time_binner_width, "t0 and time binner initialised");

    let bucket_index = build_alert_bucket_index(alerts, spatial_binner, &time_binner);
    tracing::trace!(n_buckets = bucket_index.buckets.len(), "bucket index built");

    let ps = pairs::generate_pairs(&bucket_index, spatial_binner, &time_binner, &cfg.pairs);

    let n_pairs = ps.len() as u64;
    tracing::debug!(n_pairs, "pairs generated");

    let seed_pairs = ps
        .iter()
        .enumerate()
        .map(|(i, pair)| {
            Tracklet::seed_from_pairs(
                TrackId(i as u32),
                pair,
                cfg.pairs.acc_prior_var,
                cfg.pairs.max_angular_speed,
                cfg.process_noise_q,
                cfg.singer_params.clone(),
            )
        })
        .flatten()
        .collect::<Vec<_>>();

    let nb_pair_seeds = seed_pairs.len() as u64;
    tracing::trace!(n_pair_seeds = nb_pair_seeds, "pair features extracted");

    let ts = triplets::generate_triplets_from_pairs(
        &bucket_index,
        spatial_binner,
        &time_binner,
        &cfg.triplets,
        &ps,
    );
    let n_triplets = ts.len() as u64;
    tracing::debug!(n_triplets, "triplets generated");

    let seed_triplets = ts
        .iter()
        .enumerate()
        .map(|(i, triplet)| {
            Tracklet::seed_from_triplet(
                TrackId((i as u64 + n_pairs) as u32),
                triplet,
                cfg.triplets.max_angular_speed,
                cfg.process_noise_q,
                cfg.singer_params.clone(),
            )
        })
        .flatten()
        .collect::<Vec<_>>();
    tracing::trace!(
        n_triplet_seeds = seed_triplets.len(),
        "triplet features extracted"
    );

    let mut all_seeds = {
        let mut v = seed_pairs;
        v.extend(seed_triplets);
        v
    };

    all_seeds.sort_by(|a, b| a.epoch().partial_cmp(&b.epoch()).unwrap());

    tracing::debug!(n_night_seeds = all_seeds.len(), "seeds combined and sorted");

    Ok(all_seeds)
}
