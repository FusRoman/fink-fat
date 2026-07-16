use camino::{Utf8Path, Utf8PathBuf};
use config::{Config, Environment, File};
use photom::MJDTT;
use serde::{Deserialize, Serialize};

use crate::engine_config::{
    CONFIGURATION_VERSION, Validate,
    error::{ConfigError, FieldError, ValidationErrors, prefix_errors},
    grid_population::GridConfig,
    kalman_context::{KalmanContext, KalmanContextConfig},
    kf_bank_config::KFBankConfig,
    log_level::LogLevel,
    night_advance_params::NightAdvanceParams,
    pair_config::PairConfig,
    triplet_config::TripletConfig,
    units::de_time_days,
    validate_helpers::{check_finite_positive, check_non_empty},
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

    /// Per-target log level overrides (tracing target name → level). Empty by
    /// default — every target then falls back to `log_level`. See
    /// [`crate::logging::registry::all_targets`] for the list of valid target
    /// names, and [`crate::logging::registry::build_env_filter_directive`] for
    /// how this combines with `log_level` into an `EnvFilter` directive.
    #[serde(default)]
    pub log_targets: std::collections::BTreeMap<String, LogLevel>,

    /// Number of daily log files to keep on disk when `--logs` is enabled —
    /// the oldest files beyond this count are deleted automatically on
    /// rotation. Log rotation is daily, so this is equivalently "how many
    /// days of logs to retain". `0` disables deletion (unlimited retention).
    /// Defaults to 5. Only read by the CLI; the engine itself never installs
    /// a subscriber or touches the filesystem for logging.
    #[serde(default = "default_log_retention_days")]
    pub log_retention_days: usize,
}

fn default_log_retention_days() -> usize {
    5
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
            version: CONFIGURATION_VERSION,
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
            log_targets: std::collections::BTreeMap::new(),
            log_retention_days: default_log_retention_days(),
        }
    }
}

impl EngineConfig {
    pub fn build_context(&self) -> KalmanContext {
        self.kalman_shared_context.build()
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

    /// Path the `BranchCollection` snapshot is (or would be) written to —
    /// `storage_path_buf().join(SNAPSHOT_FILENAME)`.
    pub fn snapshot_path(&self) -> Utf8PathBuf {
        self.storage_path_buf()
            .join(crate::topocentric_kf::branching::SNAPSHOT_FILENAME)
    }

    /// Load and validate an [`EngineConfig`] from a YAML file plus optional environment overrides.
    ///
    /// Behavior
    /// --------
    /// This function builds a `config::Config` by merging the following sources:
    ///
    /// 1) A YAML file at `path` (required).
    /// 2) Environment overrides with prefix `FINK_FAT`, nested separator `__`.
    ///
    /// The merged config is deserialized into [`EngineConfig`], then validated with
    /// [`EngineConfig::validate`].
    ///
    /// Missing sections/fields fall back to their Rust [`Default`] purely
    /// through each type's `#[serde(default)]` attribute — there is
    /// deliberately **no** separate "defaults" source merged in ahead of the
    /// YAML file (unlike an earlier version of this function). Layering a
    /// `Config::try_from(&EngineConfig::default())` source ahead of the YAML
    /// file made every externally-tagged enum field (e.g.
    /// `kfbank_config.cap_schedule`, `advance_params.top_k`,
    /// `advance_params.radius_strategy`) impossible to override to a
    /// different variant: `config` merges nested tables key-by-key rather
    /// than replacing them wholesale, so the default's single-key table
    /// (e.g. `{"Fixed": 1000}`) and the YAML override's single-key table
    /// (e.g. `{"Logarithmic": {...}}`) would merge into a two-key table,
    /// which `config`'s enum deserializer rejects (it requires a table with
    /// exactly one key, or a bare string for unit variants).
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
    /// - [`ConfigError::Validation`] for any semantic invariant violated
    ///   anywhere in the configuration tree (schema version, numeric ranges,
    ///   cross-field consistency, ...) — see [`ValidationErrors`] for every
    ///   field that failed, not just the first one.
    ///
    /// Notes
    /// -----
    /// - Unknown keys in YAML (or env) are rejected because `EngineConfig` and
    ///   every nested struct use `deny_unknown_fields`.
    /// - This function is intended to be the single entry point for config loading
    ///   in binaries to ensure consistent validation behavior.
    pub fn load_engine_config_validated(path: impl AsRef<Utf8Path>) -> Result<Self, ConfigError> {
        let cfg: EngineConfig = Config::builder()
            // YAML file
            .add_source(File::from(path.as_ref().as_std_path()).required(true))
            // optional env overrides
            .add_source(
                Environment::with_prefix("FINK_FAT")
                    .separator("__")
                    .try_parsing(true),
            )
            .build()?
            .try_deserialize()?;

        cfg.validate()
            .map_err(|errs| ConfigError::Validation(ValidationErrors(errs)))?;
        Ok(cfg)
    }
}

impl Validate for EngineConfig {
    /// Validate numeric ranges and cross-field consistency across the whole
    /// configuration tree, accumulating **every** failure instead of
    /// stopping at the first one.
    ///
    /// Validation performed
    /// --------------------
    /// - Schema version: `version` must be `1`.
    /// - `storage_path`: must be non-empty and must not already exist as a file.
    /// - `healpix_depth`: must be `≤ 29`.
    /// - `time_binner_width`: must be finite and strictly positive.
    /// - Every nested section, delegated to its own
    ///   [`Validate`] implementation and re-prefixed with the field name
    ///   (`pairs`, `triplets`, `kalman_shared_context`, `kfbank_config`,
    ///   `seeding_grid_config`, `advance_params`).
    ///
    /// Return
    /// ------
    /// - `Ok(())` if the configuration is valid.
    /// - `Err(Vec<FieldError>)` listing every invalid field found, if any.
    fn validate(&self) -> Result<(), Vec<FieldError>> {
        let mut errors = Vec::new();

        if self.version != CONFIGURATION_VERSION {
            errors.push(
                FieldError::new(
                    "version",
                    format!("unsupported schema version {}", self.version),
                )
                .with_hint("set version: 1, the only schema version currently supported"),
            );
        }

        let storage_path = self.storage_path();
        if let Some(e) = check_non_empty(
            "storage_path",
            storage_path.as_str(),
            "set storage_path to a directory path, e.g. \"./storage\"",
        ) {
            errors.push(e);
        } else if storage_path.is_file() {
            errors.push(
                FieldError::new(
                    "storage_path",
                    format!("must be a directory, got file path '{storage_path}'"),
                )
                .with_hint("point storage_path at a directory (existing or not), not a file"),
            );
        }

        if self.healpix_depth > 29 {
            errors.push(
                FieldError::new(
                    "healpix_depth",
                    format!("must be between 0 and 29, got {}", self.healpix_depth),
                )
                .with_hint("set healpix_depth to a value in [0, 29], e.g. 8"),
            );
        }

        if let Some(e) = check_finite_positive(
            "time_binner_width",
            self.time_binner_width,
            "set time_binner_width to a strictly positive duration, e.g. 0.021 (~30 min, days)",
        ) {
            errors.push(e);
        }

        if let Err(e) = self.pairs.validate() {
            errors.extend(prefix_errors(e, "pairs"));
        }
        if let Err(e) = self.triplets.validate() {
            errors.extend(prefix_errors(e, "triplets"));
        }
        if let Err(e) = self.kalman_shared_context.validate() {
            errors.extend(prefix_errors(e, "kalman_shared_context"));
        }
        if let Err(e) = self.kfbank_config.validate() {
            errors.extend(prefix_errors(e, "kfbank_config"));
        }
        if let Err(e) = self.seeding_grid_config.validate() {
            errors.extend(prefix_errors(e, "seeding_grid_config"));
        }
        if let Err(e) = self.advance_params.validate() {
            errors.extend(prefix_errors(e, "advance_params"));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
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

    use crate::engine_config::hypothesis_cap::HypothesisCapSchedule;
    use crate::topocentric_kf::kalman_bank::ellipse_region_finder::top_k::TopK;

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
        EngineConfig::load_engine_config_validated(&path)
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

        assert_eq!(cfg.version, CONFIGURATION_VERSION);

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
            ConfigError::Validation(errs) => {
                assert!(errs.0.iter().any(|e| e.field == "version"));
            }
            _ => panic!("expected ConfigError::Validation, got {err:?}"),
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
            ConfigError::Validation(errs) => {
                assert!(errs.0.iter().any(|e| e.field == "pairs.max_dt"));
            }
            _ => panic!("expected ConfigError::Validation, got {err:?}"),
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
            ConfigError::Validation(errs) => {
                assert!(
                    errs.0
                        .iter()
                        .any(|e| e.field == "triplets.max_predicted_residual")
                );
            }
            _ => panic!("expected ConfigError::Validation, got {err:?}"),
        }
    }

    #[test]
    fn yaml_can_override_enum_fields_to_a_non_default_variant() {
        // Regression test: overriding an externally-tagged enum field to a
        // variant *other* than its Rust default used to fail with "value of
        // enum ... should be represented by either string or table with
        // exactly one key", because `load_engine_config_validated` used to
        // merge a `Config::try_from(&EngineConfig::default())` source ahead
        // of the YAML file. `config` merges nested tables key-by-key rather
        // than replacing them, so the default's single-key table (e.g.
        // `{"Fixed": 1000}`) and the override's single-key table (e.g.
        // `{"Logarithmic": {...}}`) merged into an invalid two-key table.
        let cfg = load_from_yaml_str(
            r#"
version: 1
kfbank_config:
  cap_schedule:
    Logarithmic:
      start: 500
      end: 5
      n_obs_full: 15
advance_params:
  top_k:
    Best: 3
"#,
        )
        .expect("overriding cap_schedule/top_k to a non-default variant should load");

        match cfg.kfbank_config.cap_schedule {
            HypothesisCapSchedule::Logarithmic {
                start,
                end,
                n_obs_full,
            } => {
                assert_eq!(start, 500);
                assert_eq!(end, 5);
                assert_eq!(n_obs_full, 15);
            }
            other => panic!("expected Logarithmic, got {other:?}"),
        }

        match cfg.advance_params.top_k {
            TopK::Best(3) => {}
            other => panic!("expected Best(3), got {other:?}"),
        }

        // Fields not present in the YAML section must still fall back to
        // their Rust defaults (purely via serde(default), with no merged
        // "defaults" source involved anymore).
        assert_relative_eq!(cfg.kfbank_config.gate_chi2, 23.0, epsilon = 1e-15);
    }

    #[test]
    fn validate_accumulates_every_error_across_the_whole_tree() {
        // Three independently-invalid fields at once: schema version, a
        // pairs.* field, and a kfbank_config.* field. All three must show up
        // in the same Err, not just the first one encountered.
        let err = load_from_yaml_str(
            r#"
version: 2
pairs:
  max_dt: -0.01
kfbank_config:
  gate_chi2: -5.0
"#,
        )
        .unwrap_err();

        match err {
            ConfigError::Validation(errs) => {
                let fields: Vec<&str> = errs.0.iter().map(|e| e.field.as_str()).collect();
                assert!(fields.contains(&"version"), "fields = {fields:?}");
                assert!(fields.contains(&"pairs.max_dt"), "fields = {fields:?}");
                assert!(
                    fields.contains(&"kfbank_config.gate_chi2"),
                    "fields = {fields:?}"
                );
            }
            _ => panic!("expected ConfigError::Validation, got {err:?}"),
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

        let cfg = EngineConfig::load_engine_config_validated(&path)
            .expect("config should load with env overrides");

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

        let err = EngineConfig::load_engine_config_validated(&path).unwrap_err();
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
