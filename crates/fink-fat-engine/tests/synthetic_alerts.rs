//! Synthetic alert dataset generator for integration tests.
//!
//! This module provides utilities to generate realistic synthetic asteroid
//! alert datasets spanning multiple nights, with coherent trajectories
//! belonging to known Solar System populations.
//!
//! # Supported populations
//!
//! | Population           | Angular speed (°/day)  | Typical mag range |
//! |----------------------|------------------------|-------------------|
//! | Near-Earth (NEA)     | 0.5 – 2.0             | 18 – 22           |
//! | Main Belt            | 0.2 – 0.5             | 19 – 23           |
//! | Jupiter Trojans      | 0.05 – 0.15           | 20 – 24           |
//! | Trans-Neptunians     | 0.01 – 0.03           | 23 – 26           |
//! | Kuiper Belt          | 0.005 – 0.02          | 23 – 27           |
//!
//! # Design
//!
//! Each trajectory is generated as a linear-motion track on the sky:
//!
//! - A random starting position `(ra₀, dec₀)` in radians.
//! - A random velocity vector `(v_ra, v_dec)` in rad/day, drawn from
//!   the population's characteristic speed range.
//! - On each night, `obs_per_night` (≥ 2) observations are emitted with
//!   small intra-night time offsets and Gaussian positional noise.
//! - Each observation gets a random LSST band, realistic mag/mag_err,
//!   and a unique `dia_source_id`.
//!
//! The generator is deterministic (seeded RNG) so tests are reproducible.
//!
//! # Units
//!
//! All generated fields respect the engine's conventions:
//!
//! - `ra`, `dec`, `ra_err`, `dec_err` → **radians**
//! - `mjd_tt` → **MJD TT** (days)
//! - `mag`, `mag_err` → apparent magnitude and 1σ uncertainty
//! - `band` → `u8` LSST band code (u=0, g=1, r=2, i=3, z=4, y=5)
//!
//! # Usage
//!
//! ```rust,ignore
//! let dataset = SyntheticDatasetBuilder::new()
//!     .population(AsteroidPopulation::MainBelt, 10)
//!     .population(AsteroidPopulation::NearEarth, 5)
//!     .n_nights(5)
//!     .obs_per_night(3)
//!     .seed(42)
//!     .build();
//!
//! let store = dataset.into_alert_store();
//! let truth = dataset.ground_truth();
//! ```
//!
//! # Importing from other integration tests
//!
//! ```rust,ignore
//! #[path = "synthetic_alerts.rs"]
//! #[allow(dead_code)]
//! mod synthetic_alerts;
//! use synthetic_alerts::*;
//! ```

#![allow(dead_code)]

use std::f64::consts::PI;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, Float64Array, RecordBatch, StringArray, UInt8Array, UInt32Array, UInt64Array,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::parquet::arrow::ArrowWriter;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use fink_fat_engine::{
    Alert, AlertKey, AlertStore, night_id::NightId,
    pipeline::stages::alert_inputs::input_uri::InputUri,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEG_TO_RAD: f64 = PI / 180.0;
const ARCMIN_TO_RAD: f64 = DEG_TO_RAD / 60.0;
const ARCSEC_TO_RAD: f64 = ARCMIN_TO_RAD / 60.0;

/// LSST photometric band codes.
///
/// Convention: `u=0, g=1, r=2, i=3, z=4, y=5`.
pub mod lsst_bands {
    pub const U: u8 = 0;
    pub const G: u8 = 1;
    pub const R: u8 = 2;
    pub const I: u8 = 3;
    pub const Z: u8 = 4;
    pub const Y: u8 = 5;

    pub const ALL: [u8; 6] = [U, G, R, I, Z, Y];
}

// ---------------------------------------------------------------------------
// Asteroid populations
// ---------------------------------------------------------------------------

/// Known asteroid population for trajectory generation.
///
/// Each variant encodes population-level kinematic and photometric priors
/// used to draw realistic trajectory parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AsteroidPopulation {
    /// Near-Earth Asteroids: fast movers, relatively bright.
    NearEarth,
    /// Main Belt Asteroids: moderate speed.
    MainBelt,
    /// Jupiter Trojans: slow-moving, faint.
    Trojan,
    /// Trans-Neptunian Objects: very slow, very faint.
    TransNeptunian,
    /// Kuiper Belt Objects: slowest, deep observations required.
    KuiperBelt,
}

impl AsteroidPopulation {
    /// All known populations.
    pub const ALL: [AsteroidPopulation; 5] = [
        Self::NearEarth,
        Self::MainBelt,
        Self::Trojan,
        Self::TransNeptunian,
        Self::KuiperBelt,
    ];

    /// Typical angular speed range in **radians/day**.
    fn speed_range_rad_per_day(self) -> (f64, f64) {
        match self {
            Self::NearEarth => (0.5 * DEG_TO_RAD, 2.0 * DEG_TO_RAD),
            Self::MainBelt => (0.2 * DEG_TO_RAD, 0.5 * DEG_TO_RAD),
            Self::Trojan => (0.05 * DEG_TO_RAD, 0.15 * DEG_TO_RAD),
            Self::TransNeptunian => (0.01 * DEG_TO_RAD, 0.03 * DEG_TO_RAD),
            Self::KuiperBelt => (0.005 * DEG_TO_RAD, 0.02 * DEG_TO_RAD),
        }
    }

    /// Typical apparent magnitude range.
    fn magnitude_range(self) -> (f64, f64) {
        match self {
            Self::NearEarth => (18.0, 22.0),
            Self::MainBelt => (19.0, 23.0),
            Self::Trojan => (20.0, 24.0),
            Self::TransNeptunian => (23.0, 26.0),
            Self::KuiperBelt => (23.0, 27.0),
        }
    }

    /// Typical 1σ positional error range in **radians**.
    fn position_error_rad(self) -> (f64, f64) {
        match self {
            Self::NearEarth => (0.05 * ARCSEC_TO_RAD, 0.2 * ARCSEC_TO_RAD),
            Self::MainBelt => (0.05 * ARCSEC_TO_RAD, 0.3 * ARCSEC_TO_RAD),
            Self::Trojan => (0.1 * ARCSEC_TO_RAD, 0.5 * ARCSEC_TO_RAD),
            Self::TransNeptunian => (0.2 * ARCSEC_TO_RAD, 1.0 * ARCSEC_TO_RAD),
            Self::KuiperBelt => (0.2 * ARCSEC_TO_RAD, 1.0 * ARCSEC_TO_RAD),
        }
    }

    /// Short name for display / debug.
    pub fn label(self) -> &'static str {
        match self {
            Self::NearEarth => "NEA",
            Self::MainBelt => "MBA",
            Self::Trojan => "Trojan",
            Self::TransNeptunian => "TNO",
            Self::KuiperBelt => "KBO",
        }
    }
}

// ---------------------------------------------------------------------------
// Ground truth
// ---------------------------------------------------------------------------

/// Ground truth for a single synthetic trajectory.
///
/// This struct stores the generative parameters and the list of alert IDs
/// that belong to this trajectory, enabling verification in tests.
#[derive(Clone, Debug)]
pub struct TrajectoryTruth {
    /// Index of this trajectory in the dataset (0-based).
    pub trajectory_id: usize,
    /// Population this trajectory was drawn from.
    pub population: AsteroidPopulation,
    /// `dia_source_id` values for every observation in this trajectory.
    pub dia_source_ids: Vec<u64>,
    /// Night IDs where this trajectory was observed.
    pub night_ids: Vec<u32>,
    /// True RA at origin epoch (radians).
    pub ra0: f64,
    /// True Dec at origin epoch (radians).
    pub dec0: f64,
    /// True RA velocity (radians/day, coordinate speed including cos(dec) correction).
    pub vra: f64,
    /// True Dec velocity (radians/day).
    pub vdec: f64,
    /// Base apparent magnitude.
    pub magnitude: f64,
    /// MPC observatory code used for all observations in this trajectory.
    pub observer_mpc_code: String,
}

// ---------------------------------------------------------------------------
// Synthetic dataset
// ---------------------------------------------------------------------------

/// A synthetic dataset of alerts with ground truth trajectory associations.
pub struct SyntheticDataset {
    alerts: Vec<Alert>,
    ground_truth: Vec<TrajectoryTruth>,
}

impl SyntheticDataset {
    /// All generated alerts (sorted by night, then by time).
    pub fn alerts(&self) -> &[Alert] {
        &self.alerts
    }

    /// Ground truth trajectory associations.
    pub fn ground_truth(&self) -> &[TrajectoryTruth] {
        &self.ground_truth
    }

    /// Total number of alerts in the dataset.
    pub fn n_alerts(&self) -> usize {
        self.alerts.len()
    }

    /// Number of distinct nights in the dataset.
    pub fn n_nights(&self) -> usize {
        let mut nights: Vec<u32> = self.alerts.iter().map(|a| a.key.night_id.0).collect();
        nights.sort_unstable();
        nights.dedup();
        nights.len()
    }

    /// Number of trajectories in the dataset.
    pub fn n_trajectories(&self) -> usize {
        self.ground_truth.len()
    }

    /// Consume self and return all alerts as a `Vec<Alert>`.
    pub fn into_alerts(self) -> Vec<Alert> {
        self.alerts
    }

    /// Convert into an `AlertStore` grouped by night.
    pub fn into_alert_store(self) -> AlertStore {
        let mut store = AlertStore::new();
        for alert in self.alerts {
            store
                .insert_alert(alert)
                .expect("no duplicate dia_source_id in synthetic data");
        }
        store
    }

    /// Build an `AlertStore` from a reference (clones alerts).
    pub fn to_alert_store(&self) -> AlertStore {
        let mut store = AlertStore::new();
        for alert in &self.alerts {
            store
                .insert_alert(alert.clone())
                .expect("no duplicate dia_source_id in synthetic data");
        }
        store
    }

    /// Write the dataset to a Parquet file compatible with the engine's
    /// `AlertParquetColumns::default()` schema.
    ///
    /// Returns a `file://` URI suitable for `InputUri`.
    pub fn write_parquet(&self, path: &Path) -> InputUri {
        let schema = parquet_alert_schema();
        let file = std::fs::File::create(path).expect("create parquet file");
        let mut writer =
            ArrowWriter::try_new(file, schema.clone(), None).expect("create ArrowWriter");
        let batch = self.to_record_batch(&schema);
        writer.write(&batch).expect("write batch");
        writer.close().expect("close parquet writer");

        InputUri(format!("file://{}", path.to_str().unwrap()))
    }

    /// Convert alerts into an Arrow `RecordBatch`.
    fn to_record_batch(&self, schema: &Arc<Schema>) -> RecordBatch {
        let n = self.alerts.len();

        let mut night_ids = Vec::with_capacity(n);
        let mut dia_source_ids = Vec::with_capacity(n);
        let mut ras = Vec::with_capacity(n);
        let mut ra_errs = Vec::with_capacity(n);
        let mut decs = Vec::with_capacity(n);
        let mut dec_errs = Vec::with_capacity(n);
        let mut mjd_tts = Vec::with_capacity(n);
        let mut mags = Vec::with_capacity(n);
        let mut mag_errs = Vec::with_capacity(n);
        let mut bands = Vec::with_capacity(n);
        let mut observer_codes: Vec<String> = Vec::with_capacity(n);

        for alert in &self.alerts {
            night_ids.push(alert.key.night_id.0);
            dia_source_ids.push(alert.key.dia_source_id);
            ras.push(alert.ra);
            ra_errs.push(alert.ra_err);
            decs.push(alert.dec);
            dec_errs.push(alert.dec_err);
            mjd_tts.push(alert.mjd_tt);
            mags.push(alert.mag);
            mag_errs.push(alert.mag_err);
            bands.push(alert.band);
            observer_codes.push((*alert.observer_mpc_code).clone());
        }

        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(night_ids)) as ArrayRef,
                Arc::new(UInt64Array::from(dia_source_ids)) as ArrayRef,
                Arc::new(Float64Array::from(ras)) as ArrayRef,
                Arc::new(Float64Array::from(ra_errs)) as ArrayRef,
                Arc::new(Float64Array::from(decs)) as ArrayRef,
                Arc::new(Float64Array::from(dec_errs)) as ArrayRef,
                Arc::new(Float64Array::from(mjd_tts)) as ArrayRef,
                Arc::new(Float64Array::from(mags)) as ArrayRef,
                Arc::new(Float64Array::from(mag_errs)) as ArrayRef,
                Arc::new(UInt8Array::from(bands)) as ArrayRef,
                Arc::new(StringArray::from(observer_codes)) as ArrayRef,
            ],
        )
        .expect("build record batch")
    }
}

/// Arrow schema compatible with `AlertParquetColumns::default()`.
fn parquet_alert_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("night_id", DataType::UInt32, false),
        Field::new("dia_source_id", DataType::UInt64, false),
        Field::new("ra", DataType::Float64, false),
        Field::new("ra_err", DataType::Float64, false),
        Field::new("dec", DataType::Float64, false),
        Field::new("dec_err", DataType::Float64, false),
        Field::new("mjd_tt", DataType::Float64, false),
        Field::new("mag", DataType::Float64, false),
        Field::new("mag_err", DataType::Float64, false),
        Field::new("band", DataType::UInt8, false),
        Field::new("observer_mpc_code", DataType::Utf8, false),
    ]))
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Population request in the builder.
struct PopulationRequest {
    population: AsteroidPopulation,
    count: usize,
}

/// Builder for creating synthetic alert datasets.
///
/// # Defaults
///
/// | Parameter             | Default       |
/// |-----------------------|---------------|
/// | `n_nights`            | 3             |
/// | `obs_per_night`       | 2             |
/// | `start_night_id`      | 60000         |
/// | `start_mjd`           | 60000.5       |
/// | `night_gap_days`      | 1.0           |
/// | `intra_night_gap_days`| 0.02 (~29 min)|
/// | `observer_mpc_code`   | `"I41"`       |
/// | `seed`                | 42            |
pub struct SyntheticDatasetBuilder {
    populations: Vec<PopulationRequest>,
    n_nights: usize,
    obs_per_night: usize,
    start_night_id: u32,
    start_mjd: f64,
    night_gap_days: f64,
    intra_night_gap_days: f64,
    observer_mpc_code: String,
    rng_seed: u64,
}

impl Default for SyntheticDatasetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SyntheticDatasetBuilder {
    /// Create a new builder with default parameters.
    pub fn new() -> Self {
        Self {
            populations: Vec::new(),
            n_nights: 3,
            obs_per_night: 2,
            start_night_id: 60000,
            start_mjd: 60000.5, // middle of the first night
            night_gap_days: 1.0,
            intra_night_gap_days: 0.02, // ~29 minutes
            observer_mpc_code: "I41".to_string(),
            rng_seed: 42,
        }
    }

    /// Add `count` trajectories from the given population.
    ///
    /// Can be called multiple times to mix populations.
    pub fn population(mut self, pop: AsteroidPopulation, count: usize) -> Self {
        self.populations.push(PopulationRequest {
            population: pop,
            count,
        });
        self
    }

    /// Number of nights to observe (default: 3).
    pub fn n_nights(mut self, n: usize) -> Self {
        self.n_nights = n;
        self
    }

    /// Number of observations per trajectory per night (default: 2).
    ///
    /// Must be ≥ 2 so that intra-night pairs/seeds can be formed.
    pub fn obs_per_night(mut self, n: usize) -> Self {
        assert!(n >= 2, "obs_per_night must be >= 2 to form pairs/seeds");
        self.obs_per_night = n;
        self
    }

    /// Starting night ID (default: 60000).
    pub fn start_night_id(mut self, id: u32) -> Self {
        self.start_night_id = id;
        self
    }

    /// Starting MJD TT epoch (default: 60000.5).
    pub fn start_mjd(mut self, mjd: f64) -> Self {
        self.start_mjd = mjd;
        self
    }

    /// Gap between consecutive nights in days (default: 1.0).
    pub fn night_gap_days(mut self, gap: f64) -> Self {
        assert!(gap > 0.0, "night_gap_days must be positive");
        self.night_gap_days = gap;
        self
    }

    /// Intra-night gap between observations in days (default: 0.02 ≈ 29 min).
    pub fn intra_night_gap_days(mut self, gap: f64) -> Self {
        assert!(gap > 0.0, "intra_night_gap_days must be positive");
        self.intra_night_gap_days = gap;
        self
    }

    /// MPC observatory code assigned to all generated alerts (default: `"I41"`).
    pub fn observer_mpc_code(mut self, code: impl Into<String>) -> Self {
        self.observer_mpc_code = code.into();
        self
    }

    /// RNG seed for reproducibility (default: 42).
    pub fn seed(mut self, s: u64) -> Self {
        self.rng_seed = s;
        self
    }

    /// Build the synthetic dataset.
    ///
    /// # Panics
    ///
    /// - If no population was added.
    /// - If `obs_per_night < 2`.
    pub fn build(self) -> SyntheticDataset {
        assert!(
            !self.populations.is_empty(),
            "at least one population must be added via .population()"
        );
        assert!(self.n_nights >= 1, "at least one night required");
        assert!(
            self.obs_per_night >= 2,
            "at least 2 observations per night required"
        );

        let mut rng = StdRng::seed_from_u64(self.rng_seed);
        let mut all_alerts: Vec<Alert> = Vec::new();
        let mut ground_truth: Vec<TrajectoryTruth> = Vec::new();
        let mut next_dia_source_id: u64 = 1;
        let mut trajectory_id: usize = 0;
        let observer_arc = Arc::new(self.observer_mpc_code.clone());

        for req in &self.populations {
            for _ in 0..req.count {
                let gen_params = TrajectoryGenParams {
                    population: req.population,
                    n_nights: self.n_nights,
                    obs_per_night: self.obs_per_night,
                    start_night_id: self.start_night_id,
                    start_mjd: self.start_mjd,
                    night_gap_days: self.night_gap_days,
                    intra_night_gap_days: self.intra_night_gap_days,
                    observer_mpc_code: &observer_arc,
                };
                let (alerts, truth) = generate_trajectory(
                    &mut rng,
                    trajectory_id,
                    &mut next_dia_source_id,
                    &gen_params,
                );
                all_alerts.extend(alerts);
                ground_truth.push(truth);
                trajectory_id += 1;
            }
        }

        // Sort alerts by (night_id, mjd_tt) for deterministic ordering.
        all_alerts.sort_by(|a, b| {
            a.key
                .night_id
                .cmp(&b.key.night_id)
                .then(a.mjd_tt.total_cmp(&b.mjd_tt))
        });

        SyntheticDataset {
            alerts: all_alerts,
            ground_truth,
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

/// Create a quick dataset with a mix of all five populations.
///
/// Each population gets `n_per_pop` trajectories, observed over `n_nights` nights
/// with 2 observations per night. Uses a fixed seed for reproducibility.
pub fn quick_mixed_dataset(n_per_pop: usize, n_nights: usize) -> SyntheticDataset {
    SyntheticDatasetBuilder::new()
        .population(AsteroidPopulation::NearEarth, n_per_pop)
        .population(AsteroidPopulation::MainBelt, n_per_pop)
        .population(AsteroidPopulation::Trojan, n_per_pop)
        .population(AsteroidPopulation::TransNeptunian, n_per_pop)
        .population(AsteroidPopulation::KuiperBelt, n_per_pop)
        .n_nights(n_nights)
        .build()
}

/// Create a minimal dataset from a single population.
///
/// Useful for targeted tests on one population type.
pub fn single_population_dataset(
    pop: AsteroidPopulation,
    n_trajectories: usize,
    n_nights: usize,
    obs_per_night: usize,
) -> SyntheticDataset {
    SyntheticDatasetBuilder::new()
        .population(pop, n_trajectories)
        .n_nights(n_nights)
        .obs_per_night(obs_per_night)
        .build()
}

// ---------------------------------------------------------------------------
// Internal: trajectory generation
// ---------------------------------------------------------------------------

struct TrajectoryGenParams<'a> {
    population: AsteroidPopulation,
    n_nights: usize,
    obs_per_night: usize,
    start_night_id: u32,
    start_mjd: f64,
    night_gap_days: f64,
    intra_night_gap_days: f64,
    observer_mpc_code: &'a Arc<String>,
}

/// Generate one coherent trajectory with alerts on every night.
fn generate_trajectory(
    rng: &mut StdRng,
    trajectory_id: usize,
    next_id: &mut u64,
    params: &TrajectoryGenParams<'_>,
) -> (Vec<Alert>, TrajectoryTruth) {
    let population = params.population;
    let (speed_lo, speed_hi) = population.speed_range_rad_per_day();
    let (mag_lo, mag_hi) = population.magnitude_range();
    let (err_lo, err_hi) = population.position_error_rad();

    // -- Random initial sky position --
    // RA uniform in [0, 2π), Dec in [-60°, +60°] to avoid polar singularities.
    let ra0: f64 = rng.random_range(0.0..2.0 * PI);
    let dec0: f64 = rng.random_range(-60.0_f64.to_radians()..60.0_f64.to_radians());

    // -- Random velocity --
    // Total angular speed drawn from population range, direction uniform.
    let speed = rng.random_range(speed_lo..speed_hi);
    let direction: f64 = rng.random_range(0.0..2.0 * PI);

    // RA coordinate speed: angular speed projected onto RA axis, corrected for cos(dec).
    // We clamp cos(dec) to avoid blowup near poles (already mitigated by dec range).
    let cos_dec = dec0.cos().abs().max(0.1);
    let vra = speed * direction.cos() / cos_dec;
    let vdec = speed * direction.sin();

    // -- Photometry --
    let magnitude = rng.random_range(mag_lo..mag_hi);
    let base_mag = magnitude;
    let base_mag_err: f64 = rng.random_range(0.05..0.15);

    // -- Position noise --
    let pos_err = rng.random_range(err_lo..err_hi);

    // -- Generate observations --
    let capacity = params.n_nights * params.obs_per_night;
    let mut alerts = Vec::with_capacity(capacity);
    let mut dia_source_ids = Vec::with_capacity(capacity);
    let mut night_ids_set = Vec::with_capacity(params.n_nights);

    for night_idx in 0..params.n_nights {
        let night_id = params.start_night_id + night_idx as u32;
        night_ids_set.push(night_id);

        // Base MJD for this night.
        let night_base_mjd = params.start_mjd + (night_idx as f64) * params.night_gap_days;

        for obs_idx in 0..params.obs_per_night {
            // Intra-night time offset with small jitter (±2 min).
            let jitter: f64 = rng.random_range(-0.0014..0.0014); // ±2 min in days
            let dt_intra = (obs_idx as f64) * params.intra_night_gap_days + jitter;
            let mjd_tt = night_base_mjd + dt_intra.max(0.0);

            // True position at this epoch (linear motion from origin).
            let dt_from_start = mjd_tt - params.start_mjd;
            let true_ra = ra0 + vra * dt_from_start;
            let true_dec = dec0 + vdec * dt_from_start;

            // Add Gaussian noise to observed position.
            let (noise_ra, noise_dec) = gaussian_noise_pair(rng, pos_err);
            let observed_ra = wrap_ra(true_ra + noise_ra);
            let observed_dec = clamp_dec(true_dec + noise_dec);

            // Magnitude with per-observation scatter.
            let mag_err = base_mag_err;
            let mag_scatter: f64 = rng.random_range(-1.0..1.0) * mag_err;
            let mag = base_mag + mag_scatter;

            // Random LSST band.
            let band_idx: usize = rng.random_range(0..lsst_bands::ALL.len());
            let band = lsst_bands::ALL[band_idx];

            // Unique alert identifier.
            let dia_source_id = *next_id;
            *next_id += 1;
            dia_source_ids.push(dia_source_id);

            alerts.push(Alert {
                key: AlertKey {
                    night_id: NightId(night_id),
                    dia_source_id,
                },
                ra: observed_ra,
                ra_err: pos_err,
                dec: observed_dec,
                dec_err: pos_err,
                mjd_tt,
                mag,
                mag_err,
                band,
                observer_mpc_code: Arc::clone(params.observer_mpc_code),
            });
        }
    }

    let truth = TrajectoryTruth {
        trajectory_id,
        population,
        dia_source_ids,
        night_ids: night_ids_set,
        ra0,
        dec0,
        vra,
        vdec,
        magnitude,
        observer_mpc_code: (**params.observer_mpc_code).clone(),
    };

    (alerts, truth)
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/// Generate a pair of independent Gaussian-distributed noise values (Box-Muller).
fn gaussian_noise_pair(rng: &mut StdRng, sigma: f64) -> (f64, f64) {
    // Box-Muller transform: two uniform → two independent Gaussians.
    let u1: f64 = rng.random_range(1e-10_f64..1.0);
    let u2: f64 = rng.random_range(0.0..2.0 * PI);
    let r = (-2.0 * u1.ln()).sqrt() * sigma;
    (r * u2.cos(), r * u2.sin())
}

/// Wrap RA into `[0, 2π)`.
fn wrap_ra(ra: f64) -> f64 {
    let mut r = ra % (2.0 * PI);
    if r < 0.0 {
        r += 2.0 * PI;
    }
    r
}

/// Clamp Dec into `[-π/2, +π/2]`.
fn clamp_dec(dec: f64) -> f64 {
    dec.clamp(-PI / 2.0, PI / 2.0)
}

// ---------------------------------------------------------------------------
// Smoke tests (run with `cargo test --test synthetic_alerts`)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod synthetic_alerts_tests {
    use super::*;

    #[test]
    fn builder_produces_expected_counts() {
        let ds = SyntheticDatasetBuilder::new()
            .population(AsteroidPopulation::MainBelt, 3)
            .population(AsteroidPopulation::NearEarth, 2)
            .n_nights(4)
            .obs_per_night(2)
            .build();

        // 5 trajectories × 4 nights × 2 obs = 40 alerts
        assert_eq!(ds.n_alerts(), 40);
        assert_eq!(ds.n_trajectories(), 5);
        assert_eq!(ds.n_nights(), 4);
    }

    #[test]
    fn all_alerts_have_valid_coordinates() {
        let ds = quick_mixed_dataset(2, 3);
        for alert in ds.alerts() {
            assert!(alert.ra >= 0.0, "RA must be >= 0: {}", alert.ra);
            assert!(alert.ra < 2.0 * PI, "RA must be < 2π: {}", alert.ra);
            assert!(alert.dec >= -PI / 2.0, "Dec must be >= -π/2: {}", alert.dec);
            assert!(alert.dec <= PI / 2.0, "Dec must be <= π/2: {}", alert.dec);
            assert!(alert.ra_err > 0.0, "ra_err must be positive");
            assert!(alert.dec_err > 0.0, "dec_err must be positive");
            assert!(alert.mag_err > 0.0, "mag_err must be positive");
        }
    }

    #[test]
    fn dia_source_ids_are_unique() {
        let ds = quick_mixed_dataset(5, 5);
        let mut ids: Vec<u64> = ds.alerts().iter().map(|a| a.key.dia_source_id).collect();
        let n = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), n, "all dia_source_ids must be unique");
    }

    #[test]
    fn each_trajectory_has_at_least_two_obs_per_night() {
        let ds = SyntheticDatasetBuilder::new()
            .population(AsteroidPopulation::MainBelt, 3)
            .n_nights(4)
            .obs_per_night(3)
            .build();

        for truth in ds.ground_truth() {
            // Total observations = n_nights × obs_per_night
            assert_eq!(truth.dia_source_ids.len(), 4 * 3);

            // Each night should have exactly 3 observations.
            for &nid in &truth.night_ids {
                let count = truth
                    .dia_source_ids
                    .iter()
                    .filter(|&&did| {
                        ds.alerts()
                            .iter()
                            .any(|a| a.key.dia_source_id == did && a.key.night_id.0 == nid)
                    })
                    .count();
                assert!(
                    count >= 2,
                    "trajectory {} has only {} obs on night {} (need >= 2)",
                    truth.trajectory_id,
                    count,
                    nid
                );
            }
        }
    }

    #[test]
    fn into_alert_store_groups_by_night() {
        let ds = SyntheticDatasetBuilder::new()
            .population(AsteroidPopulation::NearEarth, 2)
            .n_nights(3)
            .obs_per_night(2)
            .build();

        let store = ds.into_alert_store();
        assert_eq!(store.n_nights(), 3);
        // 2 trajectories × 2 obs = 4 alerts per night
        assert_eq!(store.n_alerts(), 12);
    }

    #[test]
    fn deterministic_with_same_seed() {
        let ds1 = SyntheticDatasetBuilder::new()
            .population(AsteroidPopulation::MainBelt, 3)
            .n_nights(2)
            .seed(123)
            .build();

        let ds2 = SyntheticDatasetBuilder::new()
            .population(AsteroidPopulation::MainBelt, 3)
            .n_nights(2)
            .seed(123)
            .build();

        assert_eq!(ds1.n_alerts(), ds2.n_alerts());
        for (a, b) in ds1.alerts().iter().zip(ds2.alerts().iter()) {
            assert_eq!(a.key.dia_source_id, b.key.dia_source_id);
            assert_eq!(a.ra.to_bits(), b.ra.to_bits());
            assert_eq!(a.dec.to_bits(), b.dec.to_bits());
            assert_eq!(a.mjd_tt.to_bits(), b.mjd_tt.to_bits());
        }
    }

    #[test]
    fn different_seeds_produce_different_data() {
        let ds1 = SyntheticDatasetBuilder::new()
            .population(AsteroidPopulation::MainBelt, 3)
            .seed(1)
            .build();

        let ds2 = SyntheticDatasetBuilder::new()
            .population(AsteroidPopulation::MainBelt, 3)
            .seed(2)
            .build();

        // RA of first alert should differ (with overwhelming probability).
        assert_ne!(ds1.alerts()[0].ra.to_bits(), ds2.alerts()[0].ra.to_bits());
    }

    #[test]
    fn bands_are_valid_lsst() {
        let ds = quick_mixed_dataset(3, 3);
        for alert in ds.alerts() {
            assert!(
                lsst_bands::ALL.contains(&alert.band),
                "band {} is not a valid LSST band",
                alert.band
            );
        }
    }

    #[test]
    fn write_parquet_roundtrip() {
        let ds = SyntheticDatasetBuilder::new()
            .population(AsteroidPopulation::MainBelt, 2)
            .n_nights(2)
            .build();

        let dir = tempfile::TempDir::new().unwrap();
        let parquet_path = dir.path().join("synthetic.parquet");
        let uri = ds.write_parquet(&parquet_path);

        assert!(parquet_path.exists());
        assert!(uri.0.starts_with("file://"));

        // Verify we can load it back through the engine's loader.
        use fink_fat_engine::pipeline::stages::alert_inputs::alert_loader::{
            AlertParquetColumns, load_alerts_sync,
        };
        let store =
            load_alerts_sync(&uri, AlertParquetColumns::default()).expect("load back from parquet");
        assert_eq!(store.n_alerts(), ds.n_alerts());
        assert_eq!(store.n_nights(), 2);
    }

    #[test]
    fn single_population_helper_works() {
        let ds = single_population_dataset(AsteroidPopulation::Trojan, 4, 3, 2);
        assert_eq!(ds.n_trajectories(), 4);
        assert_eq!(ds.n_nights(), 3);
        assert_eq!(ds.n_alerts(), 4 * 3 * 2);
        for truth in ds.ground_truth() {
            assert_eq!(truth.population, AsteroidPopulation::Trojan);
        }
    }

    #[test]
    fn quick_mixed_helper_has_all_populations() {
        let ds = quick_mixed_dataset(2, 3);
        assert_eq!(ds.n_trajectories(), 10); // 2 × 5 populations
        let pops: Vec<AsteroidPopulation> =
            ds.ground_truth().iter().map(|t| t.population).collect();
        for pop in AsteroidPopulation::ALL {
            assert!(pops.contains(&pop), "missing population {:?}", pop);
        }
    }
}
