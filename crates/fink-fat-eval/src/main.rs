use std::time::Instant;

use ahash::{HashSet, RandomState};
use anyhow::{Context, Result};
use camino::{Utf8Path, Utf8PathBuf};
use clap::Parser;

use fink_fat_engine::{
    engine_config::{EngineConfig, load_engine_config_validated},
    error::EngineError,
    tracklet::{
        Tracklet, generate_candidate::materialize_contiguous_night, track_storage::TrackStorage,
    },
};
use hifitime::ut1::Ut1Provider;
use outfit::{DifferentialCorrectionConfig, IODParams, JPLEphem};
use photom::{
    NightId,
    io::polars::{ContiguousChoice, FromPolarsArgs},
    observation_dataset::ObsDataset,
    observer::error_model::ObsErrorModel,
};

use polars::prelude::{DataFrame, LazyFrame, ScanArgsParquet};
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Metrics types
// ---------------------------------------------------------------------------

/// Wall-clock time (milliseconds) spent in each major pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageTimings {
    /// Materialization of the night's observations from the dataset.
    pub materialize_ms: u64,
    /// Candidate generation and association to existing tracklets.
    /// `None` if the store was empty (seeding-only path).
    pub association_ms: Option<u64>,
    /// Kalman filter update from accepted associations.
    /// `None` if the store was empty.
    pub kalman_update_ms: Option<u64>,
    /// Seed generation from unassociated observations.
    /// `None` if all observations were associated.
    pub seeding_ms: Option<u64>,
    /// Orbit promotion step (IOD + differential correction).
    /// `None` if the store was empty.
    pub orbit_promotion_ms: Option<u64>,
    /// Pruning of lost tracklets beyond the time limit.
    /// `None` if the store was empty.
    pub clear_lost_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NightMetrics {
    pub night_id: String,
    pub night_index: usize,
    pub n_observations: usize,
    pub n_associated: usize,
    pub n_associations_per_tracklets_mean: f64,
    pub n_associations_per_tracklets_std: f64,
    pub n_new_seeds: usize,
    pub tracklet_counts: TrackletCounts,
    /// Breakdown of wall-clock time per pipeline stage.
    pub stage_timings: StageTimings,
    /// Total wall-clock time for this night (milliseconds).
    pub elapsed_ms: u64,
}

/// Snapshot of tracklet type distribution in the store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackletCounts {
    pub n_seed: usize,
    pub n_filter: usize,
    pub n_orbit: usize,
}

impl TrackletCounts {
    pub fn total(&self) -> usize {
        self.n_seed + self.n_filter + self.n_orbit
    }
}

/// Aggregated metrics over the full pipeline run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMetrics {
    /// Ordered per-night metrics.
    pub nights: Vec<NightMetrics>,
    /// Total number of nights processed.
    pub total_nights: usize,
    /// Total number of observations processed across all nights.
    pub total_observations: usize,
    /// Total number of associations made across all nights.
    pub total_associated: usize,
    /// Total number of seeds generated across all nights.
    pub total_new_seeds: usize,
    /// Final tracklet counts at the end of the run.
    pub final_tracklet_counts: TrackletCounts,
    /// Total wall-clock time for the full pipeline (milliseconds).
    pub total_elapsed_ms: u64,
    /// Mean per-night processing time (milliseconds).
    pub mean_night_elapsed_ms: f64,
    /// Slowest night processing time (milliseconds).
    pub max_night_elapsed_ms: u64,
    /// Fastest night processing time (milliseconds).
    pub min_night_elapsed_ms: u64,
}

impl PipelineMetrics {
    pub fn aggregate(nights: &Vec<NightMetrics>, final_counts: TrackletCounts) -> Self {
        let total_nights = nights.len();
        let total_observations = nights.iter().map(|n| n.n_observations).sum();
        let total_associated = nights.iter().map(|n| n.n_associated).sum();
        let total_new_seeds = nights.iter().map(|n| n.n_new_seeds).sum();
        let total_elapsed_ms = nights.iter().map(|n| n.elapsed_ms).sum();
        let mean_night_elapsed_ms = if total_nights == 0 {
            0.0
        } else {
            total_elapsed_ms as f64 / total_nights as f64
        };
        let max_night_elapsed_ms = nights.iter().map(|n| n.elapsed_ms).max().unwrap_or(0);
        let min_night_elapsed_ms = nights.iter().map(|n| n.elapsed_ms).min().unwrap_or(0);

        let night_clone = nights.clone();
        Self {
            nights: night_clone,
            total_nights,
            total_observations,
            total_associated,
            total_new_seeds,
            final_tracklet_counts: final_counts,
            total_elapsed_ms,
            mean_night_elapsed_ms,
            max_night_elapsed_ms,
            min_night_elapsed_ms,
        }
    }

    /// Write aggregated metrics to a JSON file at `path`.
    pub fn write_json(&self, path: impl AsRef<Utf8Path>) -> Result<()> {
        let file = std::fs::File::create(path.as_ref())
            .with_context(|| format!("failed to create metrics file: {}", path.as_ref()))?;
        serde_json::to_writer_pretty(file, self)
            .context("failed to serialize pipeline metrics to JSON")?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn count_tracklets(track_storage: &TrackStorage) -> TrackletCounts {
    track_storage.iter_tracklets().fold(
        TrackletCounts {
            n_seed: 0,
            n_filter: 0,
            n_orbit: 0,
        },
        |mut acc, t| {
            match t {
                Tracklet::Seed(_) => acc.n_seed += 1,
                Tracklet::Filter(_) => acc.n_filter += 1,
                Tracklet::Orbit(_) => acc.n_orbit += 1,
            }
            acc
        },
    )
}

// ---------------------------------------------------------------------------
// Existing functions (load_data, load_config unchanged)
// ---------------------------------------------------------------------------

pub fn load_data(parquet_path: impl AsRef<Utf8Path>) -> (DataFrame, ObsDataset) {
    let path = parquet_path.as_ref().as_str();
    let args = ScanArgsParquet {
        rechunk: true,
        ..Default::default()
    };
    let lf = LazyFrame::scan_parquet(path.into(), args).expect("scan_parquet must succeed");
    let obs_dataset = ObsDataset::from_lazy(
        lf.clone(),
        FromPolarsArgs {
            do_rechunk: Some(false),
            error_model: Some(ObsErrorModel::FCCT14),
            contiguous_choice: Some(ContiguousChoice::ContiguousNight),
        },
    )
    .expect("from_lazy must succeed for int file");

    let df = lf.collect().expect("collect must succeed");
    (df, obs_dataset)
}

/// FINK-FAT: Fink Asteroid Tracker
#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Path to the file containing the night's alerts
    #[arg(short, long, value_name = "ALERTS_FILE")]
    pub alerts: Utf8PathBuf,

    /// Path to the fink-fat configuration file
    #[arg(short, long, value_name = "CONFIG_FILE")]
    pub config: Utf8PathBuf,
}

pub fn load_config(config_path: &Utf8Path) -> Result<EngineConfig> {
    load_engine_config_validated(config_path).context("failed to load engine config")
}

// ---------------------------------------------------------------------------
// Night pipeline — now returns metrics alongside the updated store
// ---------------------------------------------------------------------------

pub fn night_pipeline(
    track_store: TrackStorage,
    night_id: &NightId,
    night_index: usize,
    obs_dataset: &ObsDataset,
    engine_config: &EngineConfig,
    error_model: ObsErrorModel,
    jpl: &JPLEphem,
    ut1_provider: &Ut1Provider,
    iod_params: &IODParams,
    diff_cor_config: &DifferentialCorrectionConfig,
    track_time_limit: f64,
    rng: &mut impl rand::Rng,
) -> Result<(TrackStorage, NightMetrics), EngineError> {
    let total_start = Instant::now();

    println!("Start fink-fat pipeline for night : {}", night_id);

    // --- Materialize night observations ----------------------------------
    let t = Instant::now();
    let night_obs = materialize_contiguous_night(obs_dataset, night_id)?;
    let materialize_ms = t.elapsed().as_millis() as u64;

    if night_obs.is_empty() {
        let metrics = NightMetrics {
            night_id: night_id.to_string(),
            night_index,
            n_observations: 0,
            n_associated: 0,
            n_associations_per_tracklets_mean: 0.,
            n_associations_per_tracklets_std: 0.,
            n_new_seeds: 0,
            tracklet_counts: count_tracklets(&track_store),
            stage_timings: StageTimings {
                materialize_ms,
                association_ms: None,
                kalman_update_ms: None,
                seeding_ms: None,
                orbit_promotion_ms: None,
                clear_lost_ms: None,
            },
            elapsed_ms: total_start.elapsed().as_millis() as u64,
        };
        return Ok((track_store, metrics));
    }

    let n_observations = night_obs.len();
    println!("Number of observations in this night : {}", n_observations);
    let current_night_time = night_obs[0].mjd_tt();

    // --- Empty store: seeding only ---------------------------------------
    if track_store.is_empty() {
        println!("TrackStore is empty, generate seeds to start tracklets");

        let t = Instant::now();
        let (night_tracklets, nb_new_tracklet) =
            track_store.generate_seeds(engine_config, night_obs)?;
        let seeding_ms = t.elapsed().as_millis() as u64;

        println!("Generate {} seeds", nb_new_tracklet);

        let metrics = NightMetrics {
            night_id: night_id.to_string(),
            night_index,
            n_observations,
            n_associated: 0,
            n_associations_per_tracklets_mean: 0.,
            n_associations_per_tracklets_std: 0.,
            n_new_seeds: nb_new_tracklet,
            tracklet_counts: count_tracklets(&night_tracklets),
            stage_timings: StageTimings {
                materialize_ms,
                association_ms: None,
                kalman_update_ms: None,
                seeding_ms: Some(seeding_ms),
                orbit_promotion_ms: None,
                clear_lost_ms: None,
            },
            elapsed_ms: total_start.elapsed().as_millis() as u64,
        };
        return Ok((night_tracklets, metrics));
    }

    // --- Association step ------------------------------------------------
    println!(
        "Track storage contains {} tracks, propagation step will go ...",
        track_store.len()
    );

    let t = Instant::now();
    let new_assoc = track_store.generate_candidates(night_obs, engine_config)?;
    let association_ms = t.elapsed().as_millis() as u64;
    let mut set_assoc_id = HashSet::with_hasher(RandomState::new());

    let mut count = 0_u64;
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;

    for (_, obs_vec) in &new_assoc {
        let n = obs_vec.len() as f64;

        // Welford online update
        count += 1;
        let delta = n - mean;
        mean += delta / count as f64;
        let delta2 = n - mean;
        m2 += delta * delta2;

        for obs in obs_vec {
            set_assoc_id.insert(obs.id());
        }
    }

    let nb_assoc_per_track_mean = mean;
    let nb_assoc_per_track_std = if count > 1 {
        (m2 / (count - 1) as f64).sqrt()
    } else {
        0.0
    };

    let n_associated = set_assoc_id.len();
    println!(
        "Found {} associations, remove them for the seeding step",
        n_associated
    );

    // --- Kalman update ---------------------------------------------------
    let t = Instant::now();
    let updated_tracklet_storage = track_store.update_from_associations(new_assoc)?;
    let kalman_update_ms = t.elapsed().as_millis() as u64;

    println!("Candidates generations and kalman update done.");

    let obs_without_assoc: Vec<_> = night_obs
        .iter()
        .filter(|obs| !set_assoc_id.contains(obs.id()))
        .cloned()
        .collect();

    // --- Seeding from unassociated observations --------------------------
    let (store_after_seed, nb_new_seeds, seeding_ms) = if obs_without_assoc.is_empty() {
        println!("No remaining observations for seeding, skip seeding step.");
        (updated_tracklet_storage, 0, None)
    } else {
        let t = Instant::now();
        let (new_seed_store, nb_new_tracklet) =
            updated_tracklet_storage.generate_seeds(engine_config, obs_without_assoc.as_slice())?;
        let seeding_ms = t.elapsed().as_millis() as u64;
        println!(
            "Generate {} new tracks, start the orbit promotion",
            nb_new_tracklet
        );
        (new_seed_store, nb_new_tracklet, Some(seeding_ms))
    };

    // --- Orbit promotion -------------------------------------------------
    let t = Instant::now();
    let promoted = store_after_seed.promote_to_orbit(
        obs_dataset,
        error_model,
        jpl,
        ut1_provider,
        iod_params,
        diff_cor_config,
        rng,
    )?;
    let orbit_promotion_ms = t.elapsed().as_millis() as u64;

    // --- Clear lost tracklets --------------------------------------------
    let t = Instant::now();
    let clean_track_store = promoted.clear_lost_tracklet(current_night_time, track_time_limit);
    let clear_lost_ms = t.elapsed().as_millis() as u64;

    let metrics = NightMetrics {
        night_id: night_id.to_string(),
        night_index,
        n_observations,
        n_associated,
        n_associations_per_tracklets_mean: nb_assoc_per_track_mean,
        n_associations_per_tracklets_std: nb_assoc_per_track_std,
        n_new_seeds: nb_new_seeds,
        tracklet_counts: count_tracklets(&clean_track_store),
        stage_timings: StageTimings {
            materialize_ms,
            association_ms: Some(association_ms),
            kalman_update_ms: Some(kalman_update_ms),
            seeding_ms,
            orbit_promotion_ms: Some(orbit_promotion_ms),
            clear_lost_ms: Some(clear_lost_ms),
        },
        elapsed_ms: total_start.elapsed().as_millis() as u64,
    };

    Ok((clean_track_store, metrics))
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn main() -> Result<()> {
    let cli = Cli::parse();

    let mut rng = StdRng::seed_from_u64(42);
    let error_model = ObsErrorModel::FCCT14;
    let jpl_ephem: JPLEphem = "horizon:DE440"
        .try_into()
        .expect("Failed to load JPL ephemeris");

    let ut1_provider = Ut1Provider::download_from_jpl("latest_eop2.long")
        .expect("Download of the JPL short time scale UT1 data failed");

    let iod_params = IODParams::builder()
        .n_noise_realizations(5)
        .noise_scale(1.1)
        .build()
        .unwrap();

    let diff_cor_config = DifferentialCorrectionConfig::default();

    let engine_config = load_config(&cli.config)?;

    let (_, obs_dataset) = load_data(&cli.alerts);

    let mut night_ids: Vec<_> = obs_dataset.iter_night_id().unwrap().collect();
    night_ids.sort();

    println!("Number of nights in the dataset: {}", night_ids.len());

    let mut track_storage = TrackStorage::new();
    let mut all_night_metrics: Vec<NightMetrics> = Vec::with_capacity(night_ids.len());
    let track_time_limit = 10.0;

    for (it, night_id) in night_ids.iter().enumerate() {
        let (new_store, night_metrics) = night_pipeline(
            track_storage,
            night_id,
            it,
            &obs_dataset,
            &engine_config,
            error_model,
            &jpl_ephem,
            &ut1_provider,
            &iod_params,
            &diff_cor_config,
            track_time_limit,
            &mut rng,
        )?;
        track_storage = new_store;

        println!(
            "=== Generation numbers ===\nnb seed = {}, nb filter = {}, nb orbit = {}\n",
            night_metrics.tracklet_counts.n_seed,
            night_metrics.tracklet_counts.n_filter,
            night_metrics.tracklet_counts.n_orbit,
        );

        all_night_metrics.push(night_metrics);

        if it % 10 == 0 {
            println!("writing snapshot track_store");
            track_storage.write_parquet(format!("track_storage_v{}.parquet", it))?;

            let final_counts = count_tracklets(&track_storage);
            let pipeline_metrics = PipelineMetrics::aggregate(&all_night_metrics, final_counts);

            println!("write snapshot pipeline metrics to disk");
            pipeline_metrics.write_json("pipeline_metrics.json")?;

            track_storage = track_storage.filter_orbit_out()
        }
    }

    println!("End of the pipeline\n ________________________\n\n");

    println!("write the final track_store to disk");
    track_storage.write_parquet("track_storage.parquet")?;

    let final_counts = count_tracklets(&track_storage);
    let pipeline_metrics = PipelineMetrics::aggregate(&all_night_metrics, final_counts);

    println!("write the final pipeline metrics to disk");
    pipeline_metrics.write_json("pipeline_metrics.json")?;

    Ok(())
}
