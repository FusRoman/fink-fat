//! # Engine configuration (`engine_config`)
//!
//! This module is the engine's serde-based YAML configuration schema, rooted
//! at [`EngineConfig`]. Loading and validating a configuration file is a
//! three-stage pipeline:
//!
//! 1. **Load / merge** — [`load_engine_config_validated`] builds a
//!    `config::Config` from [`EngineConfig::default`], overlays the YAML
//!    file at the given path, then overlays `FINK_FAT__`-prefixed
//!    environment variables (nested separator `__`), and deserializes the
//!    result into [`EngineConfig`].
//! 2. **Deserialize** — most nested sections use `#[serde(default,
//!    deny_unknown_fields)]` so missing keys fall back to Rust defaults and
//!    unknown keys are rejected as YAML typos rather than silently ignored.
//! 3. **Validate** — [`EngineConfig::validate`] checks numeric ranges and
//!    cross-field invariants not expressible through types alone; see
//!    [`error::ConfigError`] for the resulting error taxonomy.
//!
//! Submodules
//! ----------
//! - [`pair_config`] / [`triplet_config`]: intra-night pair/triplet seeding.
//! - [`kalman_context`] / [`single_kalman_config`]: shared ephemeris state
//!   and per-hypothesis Kalman filter tuning.
//! - [`kf_bank_config`] / [`hypothesis_cap`]: hypothesis-bank pruning/merging
//!   and the live-hypothesis-count decay schedule.
//! - [`grid_population`]: the `(ρ, ρ̇)` admissible-region seeding grid and
//!   its dynamical population priors.
//! - [`night_advance_params`]: tuning for advancing all banks by one night.
//! - [`log_level`]: the engine's tracing verbosity setting.
//! - [`error`]: the error types returned by the load/validate pipeline.
//! - [`units`]: human-friendly YAML unit parsing shared by the fields above
//!   (e.g. `"35 arcmin/day"`, `"86.4 min"`, `"0.02 au"`) — see that module's
//!   doc for the full list of supported quantities and unit tokens.

pub mod error;
pub mod grid_population;
pub mod hypothesis_cap;
pub mod kalman_context;
pub mod kf_bank_config;
pub mod log_level;
pub mod night_advance_params;
pub mod pair_config;
pub mod single_kalman_config;
pub mod triplet_config;
pub mod units;

use camino::{Utf8Path, Utf8PathBuf};
use config::{Config, Environment, File};
use photom::MJDTT;
use serde::{Deserialize, Serialize};

use crate::engine_config::{
    error::ConfigError,
    grid_population::GridConfig,
    kalman_context::{KalmanContext, KalmanContextConfig},
    kf_bank_config::KFBankConfig,
    log_level::LogLevel,
    night_advance_params::NightAdvanceParams,
    pair_config::PairConfig,
    triplet_config::TripletConfig,
    units::de_time_days,
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
/// - `pairs`: configuration for intra-night pair generation ([`PairConfig`]).
/// - `triplets`: configuration for intra-night triplet generation
///   ([`TripletConfig`]).
/// - `kalman_shared_context`: shared ephemeris/UT1 state and per-hypothesis
///   Kalman tuning ([`KalmanContextConfig`]).
/// - `kfbank_config`: hypothesis-bank pruning/merging tuning ([`KFBankConfig`]).
/// - `seeding_grid_config`: `(ρ, ρ̇)` admissible-region seeding grid
///   ([`GridConfig`]).
/// - `advance_params`: tuning for advancing all banks by one night
///   ([`NightAdvanceParams`]).
/// - `healpix_depth`: HEALPix tessellation depth for spatial binning.
/// - `time_binner_width`: time bin width for temporal binning.
/// - `storage_path`: root directory for on-disk artifacts produced by the pipeline.
/// - `log_level`: minimum tracing/log level for the CLI subscriber.
///
/// Notes
/// -----
/// - `storage_path` is stored as a private field and exposed through
///   accessors to keep the public API stable.
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

    /// Shared ephemeris/UT1 state and per-hypothesis Kalman filter tuning,
    /// built once at startup via [`EngineConfig::build_context`]. See
    /// [`KalmanContextConfig`].
    pub kalman_shared_context: KalmanContextConfig,

    /// Hypothesis-bank pruning, merging and search-region tuning. See
    /// [`KFBankConfig`].
    pub kfbank_config: KFBankConfig,

    /// `(ρ, ρ̇)` admissible-region seeding grid and dynamical population
    /// priors used to initialize new tracklet hypotheses. See [`GridConfig`].
    pub seeding_grid_config: GridConfig,

    /// Tuning parameters for advancing all hypothesis banks by one night.
    /// See [`NightAdvanceParams`].
    pub advance_params: NightAdvanceParams,

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

    /// Minimum log level that the CLI subscriber will record.
    ///
    /// Accepted YAML values: `"trace"`, `"debug"`, `"info"`, `"warn"`, `"error"`.
    /// Defaults to `"info"`. This value is only read by the CLI; the engine
    /// itself only emits tracing events and does not install any subscriber.
    pub log_level: LogLevel,
}

impl Default for EngineConfig {
    /// Default engine configuration.
    ///
    /// Defaults are chosen to be safe and conservative for typical pipelines:
    /// - version 1 schema,
    /// - LSST/ZTF-like seeding defaults for pairs/triplets,
    /// - defaults for the Kalman context, hypothesis-bank and seeding-grid
    ///   sections as documented on their respective types,
    /// - `./storage` as the persistence root.
    fn default() -> Self {
        Self {
            version: 1,
            pairs: PairConfig::default(),
            triplets: TripletConfig::default(),
            kalman_shared_context: KalmanContextConfig::default(),
            kfbank_config: KFBankConfig::default(),
            seeding_grid_config: GridConfig::default(),
            advance_params: NightAdvanceParams::default(),
            time_binner_width: 0.021, // ~30 min in days
            healpix_depth: 8,
            storage_path: "./storage".to_string(),
            log_level: LogLevel::default(),
        }
    }
}

impl EngineConfig {
    pub fn build_context(&self) -> KalmanContext {
        self.kalman_shared_context.build()
    }

    /// Validate numeric ranges and cross-field consistency.
    ///
    /// Validation performed
    /// --------------------
    /// - Schema version:
    ///   - `version` must be `1`.
    /// - `storage_path`:
    ///   - must be non-empty and must not already exist as a file.
    /// - `healpix_depth`:
    ///   - must be `≤ 29`.
    /// - Seeding section:
    ///   - [`PairConfig::validate`],
    ///   - [`TripletConfig::validate`].
    ///
    /// Return
    /// ------
    /// - `Ok(())` if the configuration is valid.
    /// - `Err(ConfigError)` if any validation step fails.
    ///
    /// Errors
    /// ------
    /// - [`ConfigError::UnsupportedVersion`] if `version != 1`.
    /// - [`ConfigError::Invalid`] for the `storage_path`/`healpix_depth` checks above.
    /// - [`ConfigError::Seed`] for pairs/triplets validation errors.
    ///
    /// Notes
    /// -----
    /// This function does not validate `kalman_shared_context`,
    /// `kfbank_config`, `seeding_grid_config`, `advance_params`, or
    /// `time_binner_width` — those types currently have no `validate()`
    /// method of their own (see each type's own doc for the numeric-range
    /// expectations that are documented but not enforced at load time).
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

        // SeedError -> ConfigError via #[from]
        self.pairs.validate()?;
        self.triplets.validate()?;

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
        assert_eq!(cfg.storage_path(), Utf8Path::new("./storage"));

        assert_ulps_eq!(cfg.pairs.max_dt, 0.06, max_ulps = 0);
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
    fn env_overrides_yaml_and_defaults() {
        let _guard = env_lock().lock().unwrap();
        let _clear = EnvGuard::clear("FINK_FAT__");

        let yaml = r#"
version: 1
pairs:
  max_dt: 0.06
"#;
        let path = write_tmp_yaml(yaml);

        // env > yaml > defaults
        let _env = EnvGuard::set(&[
            ("FINK_FAT__PAIRS__MAX_DT", "0.05"),
            ("FINK_FAT__PAIRS__ALLOW_SAME_TIMEBIN", "false"),
        ]);

        let cfg =
            load_engine_config_validated(&path).expect("config should load with env overrides");

        assert_relative_eq!(cfg.pairs.max_dt, 0.05, epsilon = 1e-15);
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
    }
}
