//! # Engine configuration (`EngineConfig`) and loading
//!
//! This module defines the **root configuration** for the engine and the
//! **validated loading** routine used by CLI / applications.
//!
//! The engine configuration is designed around three constraints:
//! - **Schema stability**: a `version` field enables forward compatibility.
//! - **Strictness**: unknown keys are rejected (`deny_unknown_fields`) to catch
//!   YAML typos early.
//! - **Ergonomics**: missing fields fall back to Rust defaults
//!   (`serde(default)` + `Default` impls).
//!
//! In the runtime pipeline, the main sections map to major engine stages:
//!
//! - [`PairConfig`](crate::engine_config::pair_config::PairConfig):
//!   intra-night pair generation pre-filter.
//! - [`TripletConfig`](crate::engine_config::triplet_config::TripletConfig):
//!   intra-night triplet generation, producing higher-quality seeds.
//! - [`EdgeConfig`](crate::engine_config::edge_config::EdgeConfig):
//!   inter-night edge construction (candidate retrieval + features + optional ML Top-K).
//! - [`SolverConfig`](crate::engine_config::solver_config::SolverConfig):
//!   solver selection and solver-specific policies.
//!
//! Additional global knobs:
//! - `max_gap_nights`: maximum inter-night gap considered for linking.
//! - `storage_path`: root directory for on-disk persistence / artifacts.
//!
//! -----------------------------------------------------------------------------
//! Configuration sources and precedence
//! -----------------------------------------------------------------------------
//!
//! The loader [`load_engine_config_validated`] merges multiple sources using
//! the `config` crate, with the following order (later sources override earlier):
//!
//! 1) **Rust defaults** (`EngineConfig::default()`).
//! 2) **YAML file** at the provided path (required).
//! 3) **Environment overrides** (optional), using prefix `FINK_FAT` and separator `__`.
//!
//! This produces a single `EngineConfig` instance which is then validated by
//! [`EngineConfig::validate`].
//!
//! -----------------------------------------------------------------------------
//! Environment override naming convention
//! -----------------------------------------------------------------------------
//!
//! The environment loader is configured as:
//! - prefix: `FINK_FAT`
//! - separator: `__`
//! - parsing: `try_parsing(true)`
//!
//! This implies that nested keys are addressed with double underscores.
//! Example overrides (shell):
//!
//! ```bash
//! # Override an integer field
//! export FINK_FAT__MAX_GAP_NIGHTS=4
//!
//! # Override a nested field (if it is a plain numeric type)
//! export FINK_FAT__EDGES__TOP_K_PER_LEFT=64
//! ```
//!
//! Notes
//! -----
//! - Some fields use custom deserializers (e.g. unit parsing in pairs/triplets).
//!   For those, providing a string value in the env may work, but the actual
//!   behavior depends on the serde implementation of the corresponding field.
//! - Because `deny_unknown_fields` is enabled, typos in env keys will fail
//!   during deserialization.
//!
//! -----------------------------------------------------------------------------
//! Validation strategy
//! -----------------------------------------------------------------------------
//!
//! Validation is intentionally split into:
//! - **schema version check** (`version`),
//! - **local section validation** (pairs, triplets, edges, predictor),
//! - **cross-field consistency** (global invariants).
//!
//! The current implementation performs:
//! - `version == 1` (else [`ConfigError::UnsupportedVersion`]),
//! - `pairs.validate()` and `triplets.validate()` (propagated as [`ConfigError::Seed`]),
//! - `edges.validate()` (propagated as [`ConfigError::Edges`]).
//!
//! Extending validation is expected as new fields are added, e.g.:
//! - validate the predictor configuration (`edges.predictor_config.validate()`),
//! - validate global limits (e.g. `max_gap_nights > 0`),
//! - validate storage path constraints.
//!
//! -----------------------------------------------------------------------------
//! YAML example (minimal, with solver configuration filled)
//! -----------------------------------------------------------------------------
//!
//! ```yaml
//! version: 1
//!
//! storage_path: "./storage"
//! max_gap_nights: 3
//!
//! pairs:
//!   max_dt: "86.4 min"
//!   max_angular_speed: "35 arcmin/day"
//!   max_flux_difference: 5.0
//!   allow_same_timebin: true
//!
//! triplets:
//!   max_dt_between: "57.6 min"
//!   max_pair_sep: "8.6 arcmin"
//!   max_predicted_residual: "2.75 arcmin"
//!   enforce_time_order: true
//!   max_flux_difference: 5.0
//!
//! edges:
//!   emit_all_edges: false
//!   edge_ranking_model_path: "edge_ranker.onnx"
//!   top_k_per_left: 32
//!   onnx_batch_size: 128
//!   parallel_left_batches: true
//!   parallel_left_batch_size: 512
//!   predictor_config:
//!     k_sigma: 3.0
//!     noise:
//!       variance_floor: 0.0
//!       drift_per_day: 0.0
//!       curvature_per_day2: 0.0
//!     pad_cell_radius: true
//!     time_bin_dt: 0.021
//!     v_slack: 0.0
//!
//! solver_config:
//!   policy:
//!     routing: Heuristics
//!     trivial_max_nodes: 8
//!     trivial_max_active_edges: 16
//!     mcf_budget_s: 0.05
//!     k_mcf_s_per_edge_logn: 1.0e-8
//!     max_night_span_for_mcf: 4
//!   bounded_beam:
//!     max_tracks: 16
//!     min_nodes: 3
//!     beam_width: 64
//!     max_out_per_node: 8
//!     max_tracks_per_source: 8
//!     max_expansions: 50000
//! ```
//!
//! Notes
//! -----
//! - The `policy.routing` field is serialized using Serde’s default enum encoding:
//!   - `Heuristics` is written as the plain variant name,
//!   - `Force(choice)` is written as a map like `{ Force: BoundedBeam }`.
//! - The `bounded_beam` block corresponds to [`BoundedBeamConfig`] and provides
//!   hard guardrails on exploration and output size.
//! - If additional solver families are added later, `solver_config` may grow
//!   with extra sub-sections; keep the routing policy independent from solver
//!   internal knobs.
//!
//! -----------------------------------------------------------------------------
//! See also
//! -----------------------------------------------------------------------------
//!
//! - [`crate::engine_config::units`]: human-friendly unit parsing for YAML fields.
//! - [`crate::engine_config::error::ConfigError`]: unified error type returned by the loader.
//! - [`PairConfig`], [`TripletConfig`], [`EdgeConfig`], [`SolverConfig`] for detailed section docs.

pub mod edge_config;
pub mod error;
pub mod pair_config;
pub mod propagator_config;
pub mod solver_config;
pub mod triplet_config;
pub mod units;

use camino::{Utf8Path, Utf8PathBuf};
use config::{Config, Environment, File};
use serde::{Deserialize, Serialize};

use crate::{
    MJDTT,
    engine_config::{
        edge_config::EdgeConfig, error::ConfigError, pair_config::PairConfig,
        solver_config::SolverConfig, triplet_config::TripletConfig, units::de_time_days,
    },
};

/// Root configuration for the engine (serde-friendly).
///
/// This struct is the single entry point for configuration consumed by the
/// engine at runtime.
///
/// Behavior
/// --------
/// - Unknown keys are rejected (`deny_unknown_fields`) to catch YAML typos early.
/// - Missing fields are filled from defaults (`serde(default)` + [`Default`]).
///
/// Fields
/// ------
/// - `version`: schema version. Must match the expected version in
///   [`EngineConfig::validate`].
/// - `pairs`: configuration for intra-night pair generation.
/// - `triplets`: configuration for intra-night triplet generation.
/// - `edges`: configuration for inter-night edge construction.
/// - `solver_config`: solver policy and solver-specific knobs.
/// - `max_gap_nights`: maximum inter-night gap considered when linking nights.
/// - `storage_path`: root directory for on-disk artifacts produced by the pipeline.
///
/// Notes
/// -----
/// - `max_gap_nights` and `storage_path` are stored as private fields and exposed
///   through accessors to keep the public API stable.
/// - The storage path is stored as a UTF-8 string and exposed as `Utf8Path`
///   for ergonomics and OS-independent handling.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EngineConfig {
    /// Schema version for forward compatibility.
    pub version: u32,

    /// Intra-night pair generation configuration.
    pub pairs: PairConfig,

    /// Intra-night triplet generation configuration.
    pub triplets: TripletConfig,

    /// Inter-night edge construction configuration.
    pub edges: EdgeConfig,

    /// Solver selection and solver-specific configuration.
    pub solver_config: SolverConfig,

    /// Maximum number of nights that can be skipped when linking (`gap` constraint).
    ///
    /// Interpretation
    /// --------------
    /// This parameter limits how far the engine is allowed to link forward in time.
    /// For example, if `max_gap_nights = 3`, edges may connect seeds separated by
    /// up to 3 night boundaries (implementation-dependent: inclusive/exclusive
    /// gap semantics are defined by the edge builder).
    ///
    /// Notes
    /// -----
    /// Keeping this small reduces candidate fan-out and runtime.
    max_gap_nights: u8,

    /// Healpix depth used for spatial binning (nested representation).
    ///
    /// Overview
    /// --------
    /// Controls the angular resolution of the HEALPix tessellation used to
    /// index sky positions (via `cdshealpix`, nested scheme).
    ///
    /// The subdivision follows:
    /// - `nside = 2^depth`
    /// - `npix = 12 × nside²`
    ///
    /// Increasing `depth`:
    /// - increases the number of pixels,
    /// - decreases pixel angular size,
    /// - improves spatial selectivity.
    ///
    /// Role in the engine
    /// ------------------
    /// Used for:
    /// - intra-night alert/seed spatial binning,
    /// - spatial pre-filtering during inter-night edge construction.
    ///
    /// Trade-off
    /// ---------
    /// - Low depth → coarse grid → more candidates per pixel (higher fan-out).
    /// - High depth → fine grid → better pruning but more indexing overhead.
    ///
    /// Typical values in astronomical use cases are between 5 and 12.
    ///
    /// Must remain consistent across pipeline stages to preserve deterministic
    /// graph construction.
    pub healpix_depth: u8,

    /// Time bin width (days, MJD TT) used for temporal binning.
    ///
    /// Overview
    /// --------
    /// Controls the resolution of the uniform time binning scheme
    /// (`UniformTimeBinner`) used during inter-night edge construction.
    ///
    /// Time is partitioned into fixed-width bins:
    /// - width = `time_binner_width` (days, MJD TT),
    /// - bin index: `k = floor((t - t0) / dt)`.
    ///
    /// Role in the engine
    /// ------------------
    /// Used to:
    /// - index seeds by time,
    /// - restrict candidate searches to compatible temporal windows,
    /// - reduce combinatorial explosion in edge generation.
    ///
    /// Trade-off
    /// ---------
    /// - Large width → coarse temporal grouping → more candidates per bin.
    /// - Small width → finer pruning → more bins and indexing overhead.
    ///
    /// Must be strictly positive. Very small values increase index fragmentation
    /// without significant gain beyond typical astrometric timing precision.
    #[serde(deserialize_with = "de_time_days")]
    pub time_binner_width: MJDTT,

    /// Root directory used for persistence (seeds, edges, trajectories, logs).
    ///
    /// The directory is stored as a UTF-8 string and exposed via:
    /// - [`EngineConfig::storage_path`] (`&Utf8Path`)
    /// - [`EngineConfig::storage_path_buf`] (`Utf8PathBuf`)
    storage_path: String,

    /// Number of delta steps after which the graph is compacted.
    /// This is a safeguard to keep load times bounded by preventing an unbounded number of deltas.
    ///
    /// When the number of delta files in the journal exceeds this threshold,
    /// the stage triggers a full compaction (snapshot rebuild + delta pruning).
    pub compact_graph_every_delta: usize,
}

impl Default for EngineConfig {
    /// Default engine configuration.
    ///
    /// Defaults are chosen to be safe and conservative for typical pipelines:
    /// - version 1 schema,
    /// - LSST/ZTF-like seeding defaults for pairs/triplets,
    /// - ML Top-K edge construction defaults,
    /// - a small `max_gap_nights` for bounded fan-out,
    /// - `./storage` as the persistence root.
    fn default() -> Self {
        Self {
            version: 1,
            pairs: PairConfig::default(),
            triplets: TripletConfig::default(),
            edges: EdgeConfig::default(),
            solver_config: SolverConfig::default(),
            max_gap_nights: 3,
            healpix_depth: 8,
            time_binner_width: 0.021, // ~30 min in days
            storage_path: "./storage".to_string(),
            compact_graph_every_delta: 20,
        }
    }
}

impl EngineConfig {
    /// Validate numeric ranges and cross-field consistency.
    ///
    /// Validation performed
    /// --------------------
    /// - Schema version:
    ///   - `version` must be `1`.
    /// - Seeding section:
    ///   - [`PairConfig::validate`],
    ///   - [`TripletConfig::validate`].
    /// - Edge section:
    ///   - [`EdgeConfig::validate`].
    ///
    /// Return
    /// ------
    /// - `Ok(())` if the configuration is valid.
    /// - `Err(ConfigError)` if any validation step fails.
    ///
    /// Errors
    /// ------
    /// - [`ConfigError::UnsupportedVersion`] if `version != 1`.
    /// - [`ConfigError::Seed`] for pairs/triplets validation errors.
    /// - [`ConfigError::Edges`] for edge validation errors.
    ///
    /// Notes
    /// -----
    /// This function currently does not validate `max_gap_nights` nor the
    /// predictor configuration embedded in `edges`. If those are operationally
    /// required invariants, add checks here to centralize validation.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.version != 1 {
            return Err(ConfigError::UnsupportedVersion(self.version));
        }

        let storage_path = self.storage_path();
        if storage_path.as_str().is_empty() {
            return Err(ConfigError::Invalid {
                msg: "storage_path must not be empty".to_string(),
            });
        }

        if storage_path.is_file() {
            return Err(ConfigError::Invalid {
                msg: format!(
                    "storage_path must be a directory, got file path '{}'",
                    storage_path
                ),
            });
        }

        if self.healpix_depth > 29 {
            return Err(ConfigError::Invalid {
                msg: format!(
                    "healpix_depth must be between 0 and 29, got {}",
                    self.healpix_depth
                ),
            });
        }

        if self.time_binner_width <= 0.0 {
            return Err(ConfigError::Invalid {
                msg: format!(
                    "time_binner_width must be positive, got {}",
                    self.time_binner_width
                ),
            });
        }

        // SeedError -> ConfigError via #[from]
        self.pairs.validate()?;
        self.triplets.validate()?;

        // EdgeConfigError -> ConfigError via #[from]
        self.edges.validate()?;

        Ok(())
    }

    /// Return the storage root as a borrowed UTF-8 path.
    ///
    /// This is the recommended accessor for read-only operations.
    pub fn storage_path(&self) -> &Utf8Path {
        Utf8Path::new(&self.storage_path)
    }

    /// Return the storage root as an owned UTF-8 path buffer.
    ///
    /// This is useful when the caller needs to join paths or store the result.
    pub fn storage_path_buf(&self) -> Utf8PathBuf {
        Utf8PathBuf::from(&self.storage_path)
    }

    /// Return the configured maximum inter-night gap (in nights).
    pub fn max_gap_nights(&self) -> u8 {
        self.max_gap_nights
    }
}

/// Load and validate an [`EngineConfig`] from a YAML file plus optional environment overrides.
///
/// Behavior
/// --------
/// This function builds a `config::Config` by merging the following sources:
///
/// 1) Rust defaults from [`EngineConfig::default`].
/// 2) A YAML file at `path` (required).
/// 3) Environment overrides with prefix `FINK_FAT`, nested separator `__`.
///
/// The merged config is deserialized into [`EngineConfig`], then validated with
/// [`EngineConfig::validate`].
///
/// Arguments
/// ---------
/// - `path`: path to a required YAML file.
///
/// Return
/// ------
/// - `Ok(EngineConfig)` if loading + deserialization + validation succeed.
/// - `Err(ConfigError)` otherwise.
///
/// Errors
/// ------
/// - [`ConfigError::ConfigRs`] for load/parse/deserialization failures.
/// - [`ConfigError::UnsupportedVersion`] for version mismatch.
/// - Section validation errors propagated via `ConfigError` (`Seed`, `Predictor`, `Edges`).
///
/// Notes
/// -----
/// - Unknown keys in YAML (or env) are rejected because `EngineConfig` and most
///   nested structs use `deny_unknown_fields`.
/// - This function is intended to be the single entry point for config loading
///   in binaries to ensure consistent validation behavior.
pub fn load_engine_config_validated(path: &Utf8Path) -> Result<EngineConfig, ConfigError> {
    let cfg: EngineConfig = Config::builder()
        // defaults from Rust
        .add_source(Config::try_from(&EngineConfig::default())?)
        // YAML file
        .add_source(File::from(path.as_std_path()).required(true))
        // optional env overrides
        .add_source(
            Environment::with_prefix("FINK_FAT")
                .separator("__")
                .try_parsing(true),
        )
        .build()?
        .try_deserialize()?;

    cfg.validate()?;
    Ok(cfg)
}

#[cfg(test)]
mod engine_config_tests {
    use crate::{engine_config::error::EdgeConfigError, error::PredictorParamError};

    use super::*;

    use std::{
        collections::HashMap,
        env, fs,
        sync::{Mutex, OnceLock},
    };

    use approx::{assert_relative_eq, assert_ulps_eq};
    use camino::Utf8PathBuf;
    use proptest::prelude::*;

    /* --------------------------------------------------------------------- */
    /*  Global env lock (env vars are process-global; tests must not race)    */
    /* --------------------------------------------------------------------- */

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    /// RAII guard: temporarily set env vars and restore previous values on drop.
    struct EnvGuard {
        saved: HashMap<String, Option<String>>,
    }

    impl EnvGuard {
        fn set(vars: &[(&str, &str)]) -> Self {
            let mut saved = HashMap::new();
            for (k, v) in vars {
                let key = (*k).to_string();
                let prev = env::var(k).ok();
                saved.insert(key.clone(), prev);
                unsafe { env::set_var(k, v) };
            }
            Self { saved }
        }

        /// Clear all env vars with a given prefix (e.g. "FINK_FAT__") and restore on drop.
        fn clear(prefix: &str) -> Self {
            let mut saved = HashMap::new();
            for (k, v) in env::vars() {
                if k.starts_with(prefix) {
                    saved.insert(k.clone(), Some(v));
                    unsafe { env::remove_var(k) };
                }
            }
            Self { saved }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (k, prev) in self.saved.drain() {
                match prev {
                    Some(v) => unsafe { env::set_var(k, v) },
                    None => unsafe { env::remove_var(k) },
                }
            }
        }
    }

    /* --------------------------------------------------------------------- */
    /*  Helpers                                                              */
    /* --------------------------------------------------------------------- */

    fn write_tmp_yaml(contents: &str) -> Utf8PathBuf {
        let mut p = std::env::temp_dir();
        let fname = format!(
            "fink_fat_engine_config_test_{}_{}.yaml",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        p.push(fname);
        fs::write(&p, contents).expect("write temp yaml");
        Utf8PathBuf::from_path_buf(p).expect("temp path should be valid UTF-8")
    }

    fn load_from_yaml_str(yaml: &str) -> Result<EngineConfig, ConfigError> {
        let _guard = env_lock().lock().unwrap();
        let _clear = EnvGuard::clear("FINK_FAT__");
        let path = write_tmp_yaml(yaml);
        load_engine_config_validated(&path)
    }

    /* --------------------------------------------------------------------- */
    /*  Unit helpers (expected values)                                        */
    /* --------------------------------------------------------------------- */

    fn days_from_minutes(min: f64) -> f64 {
        min / 1440.0
    }

    fn rad_from_arcmin(arcmin: f64) -> f64 {
        (arcmin / 60.0).to_radians()
    }

    fn rad_per_day_from_arcsec_per_hour(arcsec_per_hour: f64) -> f64 {
        // arcsec/hour * 24 = arcsec/day ; /3600 = deg/day ; deg -> rad
        ((arcsec_per_hour * 24.0) / 3600.0).to_radians()
    }

    /* --------------------------------------------------------------------- */
    /*  Deterministic unit tests                                              */
    /* --------------------------------------------------------------------- */

    #[test]
    fn default_engine_config_is_valid() {
        let cfg = EngineConfig::default();
        cfg.validate()
            .expect("EngineConfig::default() must be valid");
    }

    #[test]
    fn load_yaml_minimal_uses_defaults_and_validates() {
        let cfg = load_from_yaml_str(
            r#"
version: 1
"#,
        )
        .expect("config should load");

        assert_eq!(cfg.version, 1);

        // A few stable default checks:
        assert_eq!(cfg.max_gap_nights(), 3);
        assert_eq!(cfg.storage_path(), Utf8Path::new("./storage"));

        assert_ulps_eq!(cfg.pairs.max_dt, 0.06, max_ulps = 0);
        assert_eq!(cfg.edges.top_k_per_left, 32);

        // Predictor defaults are expected valid.
        assert!(cfg.edges.predictor_config.k_sigma > 0.0);
        cfg.edges
            .predictor_config
            .validate()
            .expect("default predictor_config must validate");
    }

    #[test]
    fn load_yaml_rejects_unknown_top_level_keys() {
        let err = load_from_yaml_str(
            r#"
version: 1
unknown_key: 123
"#,
        )
        .unwrap_err();

        matches!(err, ConfigError::ConfigRs(_))
            .then_some(())
            .expect("expected ConfigError::ConfigRs (serde deny_unknown_fields)");
    }

    #[test]
    fn load_yaml_rejects_unknown_nested_keys() {
        // `pairs` has `deny_unknown_fields`.
        let err = load_from_yaml_str(
            r#"
version: 1
pairs:
  max_dt: "86.4 min"
  not_a_real_field: 1
"#,
        )
        .unwrap_err();

        matches!(err, ConfigError::ConfigRs(_))
            .then_some(())
            .expect("expected ConfigError::ConfigRs (nested deny_unknown_fields)");
    }

    #[test]
    fn validate_rejects_unsupported_version() {
        let err = load_from_yaml_str(
            r#"
version: 2
"#,
        )
        .unwrap_err();

        match err {
            ConfigError::UnsupportedVersion(2) => {}
            _ => panic!("expected UnsupportedVersion(2), got {err:?}"),
        }
    }

    #[test]
    fn validate_rejects_invalid_pairs() {
        // max_dt < 0
        let err = load_from_yaml_str(
            r#"
version: 1
pairs:
  max_dt: -0.01
"#,
        )
        .unwrap_err();

        match err {
            ConfigError::Seed(_) => {}
            _ => panic!("expected ConfigError::Seed, got {err:?}"),
        }
    }

    #[test]
    fn validate_rejects_invalid_triplets_inconsistent_residual() {
        // max_predicted_residual > max_pair_sep
        let err = load_from_yaml_str(
            r#"
version: 1
triplets:
  max_pair_sep: "1 arcmin"
  max_predicted_residual: "2 arcmin"
"#,
        )
        .unwrap_err();

        match err {
            ConfigError::Seed(_) => {}
            _ => panic!("expected ConfigError::Seed, got {err:?}"),
        }
    }

    #[test]
    fn validate_rejects_invalid_edge_config() {
        // top_k_per_left == 0 is rejected by EdgeConfig::validate()
        let err = load_from_yaml_str(
            r#"
version: 1
edges:
  top_k_per_left: 0
"#,
        )
        .unwrap_err();

        match err {
            ConfigError::Edges(_) => {}
            _ => panic!("expected ConfigError::Edges, got {err:?}"),
        }
    }

    #[test]
    fn validate_rejects_invalid_predictor_config() {
        // requires EngineConfig::validate() to call predictor_config.validate()
        let err = load_from_yaml_str(
            r#"
version: 1
edges:
  predictor_config:
    k_sigma: 0.0
"#,
        )
        .unwrap_err();

        match err {
            ConfigError::Edges(EdgeConfigError::PredictorConfig(
                PredictorParamError::InvalidKSigma(0.0),
            )) => {}
            _ => panic!("expected ConfigError::Predictor, got {err:?}"),
        }
    }

    #[test]
    fn env_overrides_yaml_and_defaults() {
        let _guard = env_lock().lock().unwrap();
        let _clear = EnvGuard::clear("FINK_FAT__");

        let yaml = r#"
version: 1
pairs:
  max_dt: 0.06
edges:
  top_k_per_left: 10
"#;
        let path = write_tmp_yaml(yaml);

        // env > yaml > defaults
        let _env = EnvGuard::set(&[
            ("FINK_FAT__PAIRS__MAX_DT", "0.05"),
            ("FINK_FAT__EDGES__TOP_K_PER_LEFT", "42"),
            ("FINK_FAT__PAIRS__ALLOW_SAME_TIMEBIN", "false"),
        ]);

        let cfg =
            load_engine_config_validated(&path).expect("config should load with env overrides");

        assert_relative_eq!(cfg.pairs.max_dt, 0.05, epsilon = 1e-15);
        assert_eq!(cfg.edges.top_k_per_left, 42);
        assert_eq!(cfg.pairs.allow_same_timebin, false);
    }

    #[test]
    fn env_bad_value_fails_to_load() {
        let _guard = env_lock().lock().unwrap();
        let _clear = EnvGuard::clear("FINK_FAT__");

        let yaml = r#"
version: 1
"#;
        let path = write_tmp_yaml(yaml);

        let _env = EnvGuard::set(&[("FINK_FAT__PAIRS__MAX_DT", "not-a-number")]);

        let err = load_engine_config_validated(&path).unwrap_err();
        match err {
            ConfigError::ConfigRs(_) => {}
            _ => panic!("expected ConfigError::ConfigRs, got {err:?}"),
        }
    }

    #[test]
    fn yaml_accepts_units_for_pairs_and_triplets() {
        // pairs.max_dt: "86.4 min" = 0.06 day
        // pairs.max_angular_speed: "180 arcsec/hour"
        //   = 180 arcsec * 24 = 4320 arcsec/day = 1.2 deg/day
        let cfg = load_from_yaml_str(
            r#"
version: 1

pairs:
  max_dt: "86.4 min"
  max_angular_speed: "180 arcsec/hour"
  allow_same_timebin: true

triplets:
  max_dt_between: "57.6 min"
  max_pair_sep: "9 arcmin"
  max_predicted_residual: "48 arcsec"
  enforce_time_order: true
"#,
        )
        .expect("config should load with unit strings");

        // Time → days
        assert_relative_eq!(cfg.pairs.max_dt, 0.06, epsilon = 1e-15);
        assert_relative_eq!(cfg.triplets.max_dt_between, 0.04, epsilon = 1e-15);

        // Angles → radians
        let expected_pair_sep = (9.0_f64 / 60.0).to_radians();
        assert_relative_eq!(
            cfg.triplets.max_pair_sep,
            expected_pair_sep,
            max_relative = 1e-13
        );

        let expected_residual = (48.0_f64 / 3600.0).to_radians();
        assert_relative_eq!(
            cfg.triplets.max_predicted_residual,
            expected_residual,
            max_relative = 1e-13
        );

        // Angular speed → rad/day
        let expected_speed = (1.2_f64).to_radians();
        assert_relative_eq!(
            cfg.pairs.max_angular_speed,
            expected_speed,
            max_relative = 1e-13
        );
    }

    #[test]
    fn yaml_rejects_invalid_unit_strings() {
        let err = load_from_yaml_str(
            r#"
version: 1
triplets:
  max_pair_sep: "10 parsec"
"#,
        )
        .unwrap_err();

        match err {
            ConfigError::ConfigRs(_) => {}
            _ => panic!("expected ConfigError::ConfigRs, got {err:?}"),
        }
    }

    #[test]
    fn yaml_rejects_invalid_rate_syntax() {
        // Missing "/day" or "/hour" etc.
        let err = load_from_yaml_str(
            r#"
version: 1
pairs:
  max_angular_speed: "10 arcmin"
"#,
        )
        .unwrap_err();

        match err {
            ConfigError::ConfigRs(_) => {}
            _ => panic!("expected ConfigError::ConfigRs, got {err:?}"),
        }
    }

    #[test]
    fn numeric_values_still_pass_through_unchanged() {
        // For numeric YAML, we expect an exact float parse for this literal in practice;
        // but to avoid brittle parsing edge cases, we validate at 0 ulps for the exact literal.
        let cfg = load_from_yaml_str(
            r#"
version: 1
pairs:
  max_dt: 0.123456
"#,
        )
        .expect("numeric values should still be accepted");

        assert_ulps_eq!(cfg.pairs.max_dt, 0.123456, max_ulps = 0);
    }

    /* --------------------------------------------------------------------- */
    /*  Property-based tests (proptest)                                       */
    /* --------------------------------------------------------------------- */

    proptest! {
        // Keep the number of cases reasonable to avoid slow CI.
        #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]

        #[test]
        fn prop_units_minutes_to_days_pairs_max_dt(minutes in 0u32..(10_000u32)) {
            let minutes_f = minutes as f64;

            let yaml = format!(r#"
version: 1
pairs:
  max_dt: "{} min"
"#, minutes);

            let cfg = load_from_yaml_str(&yaml).expect("config should load");
            let expected = days_from_minutes(minutes_f);

            // Small epsilon: exact rational / 1440 in f64.
            assert_relative_eq!(cfg.pairs.max_dt, expected, epsilon = 1e-15);
        }

        #[test]
        fn prop_units_arcsec_per_hour_to_rad_per_day_pairs_max_angular_speed(arcsec_per_hour in 0u32..(50_000u32)) {
            let x = arcsec_per_hour as f64;

            let yaml = format!(r#"
version: 1
pairs:
  max_angular_speed: "{} arcsec/hour"
"#, arcsec_per_hour);

            let cfg = load_from_yaml_str(&yaml).expect("config should load");
            let expected = rad_per_day_from_arcsec_per_hour(x);

            assert_relative_eq!(cfg.pairs.max_angular_speed, expected, max_relative = 1e-13);
        }

        #[test]
        fn prop_units_arcmin_to_rad_triplets_max_pair_sep(arcmin in 0u32..(60_000u32)) {
            let arcmin_f = arcmin as f64;

            // Ensure config remains valid by also setting residual <= pair_sep.
            // We use half the sep (integer division ok).
            let residual_arcmin = (arcmin / 2) as u32;

            let yaml = format!(r#"
version: 1
triplets:
  max_pair_sep: "{} arcmin"
  max_predicted_residual: "{} arcmin"
"#, arcmin, residual_arcmin);

            let cfg = load_from_yaml_str(&yaml).expect("config should load");
            let expected = rad_from_arcmin(arcmin_f);

            assert_relative_eq!(cfg.triplets.max_pair_sep, expected, max_relative = 1e-13);
        }

        #[test]
        fn prop_env_overrides_yaml_top_k_per_left(top_k_yaml in 1usize..512usize, top_k_env in 1usize..512usize) {
            let _guard = env_lock().lock().unwrap();
            let _clear = EnvGuard::clear("FINK_FAT__");

            let yaml = format!(r#"
version: 1
edges:
  top_k_per_left: {}
"#, top_k_yaml);

            let path = write_tmp_yaml(&yaml);

            let _env = EnvGuard::set(&[
                ("FINK_FAT__EDGES__TOP_K_PER_LEFT", &top_k_env.to_string()),
            ]);

            let cfg = load_engine_config_validated(&path).expect("config should load");
            prop_assert_eq!(cfg.edges.top_k_per_left, top_k_env);
        }
    }
}
