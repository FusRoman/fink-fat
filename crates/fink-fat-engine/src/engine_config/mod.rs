pub mod edge_config;
pub mod error;
pub mod pair_config;
pub mod propagator_config;
pub mod triplet_config;
pub mod units;
pub mod solver_config;

use camino::{Utf8Path, Utf8PathBuf};
use config::{Config, Environment, File};
use serde::{Deserialize, Serialize};

use crate::engine_config::{
    edge_config::EdgeConfig, error::ConfigError, pair_config::PairConfig,
    triplet_config::TripletConfig,
};

/// Root configuration for the engine (serde-friendly).
///
/// Notes
/// -----
/// - Unknown keys are rejected (`deny_unknown_fields`) to catch YAML typos early.
/// - Missing fields are filled from defaults (`serde(default)` + `Default` impls).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EngineConfig {
    /// Schema version for forward compatibility.
    pub version: u32,

    pub pairs: PairConfig,
    pub triplets: TripletConfig,
    pub edges: EdgeConfig,
    max_gap_nights: u8,
    storage_path: String,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            version: 1,
            pairs: PairConfig::default(),
            triplets: TripletConfig::default(),
            edges: EdgeConfig::default(),
            max_gap_nights: 3,
            storage_path: "./storage".to_string(),
        }
    }
}

impl EngineConfig {
    /// Validate numeric ranges and cross-field consistency.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.version != 1 {
            return Err(ConfigError::UnsupportedVersion(self.version));
        }

        // SeedError -> ConfigError via #[from]
        self.pairs.validate()?;
        self.triplets.validate()?;

        // EdgeConfigError -> ConfigError via #[from]
        self.edges.validate()?;

        Ok(())
    }

    pub fn storage_path(&self) -> &Utf8Path {
        Utf8Path::new(&self.storage_path)
    }

    pub fn storage_path_buf(&self) -> Utf8PathBuf {
        Utf8PathBuf::from(&self.storage_path)
    }

    pub fn max_gap_nights(&self) -> u8 {
        self.max_gap_nights
    }
}

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

    use camino::Utf8PathBuf;

    use approx::{assert_relative_eq, assert_ulps_eq};

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

    /* --------------------------------------------------------------------- */
    /*  Tests                                                                */
    /* --------------------------------------------------------------------- */

    #[test]
    fn default_engine_config_is_valid() {
        // This test will FAIL until PredictorParams::default() is fixed
        // (k_sigma must be > 0).
        let cfg = EngineConfig::default();
        cfg.validate()
            .expect("EngineConfig::default() must be valid");
    }

    #[test]
    fn load_yaml_minimal_uses_defaults_and_validates() {
        let _guard = env_lock().lock().unwrap();
        let _clear = EnvGuard::clear("FINK_FAT__");

        // Minimal YAML: only version. Everything else should come from defaults.
        let yaml = r#"
version: 1
"#;
        let path = write_tmp_yaml(yaml);

        let cfg = load_engine_config_validated(&path).expect("config should load");

        assert_eq!(cfg.version, 1);

        // A few stable default checks:
        assert_eq!(cfg.edges.top_k_per_left, 32);

        // Predictor must be valid
        assert!(cfg.edges.predictor_config.k_sigma > 0.0);
    }

    #[test]
    fn load_yaml_rejects_unknown_top_level_keys() {
        let _guard = env_lock().lock().unwrap();
        let _clear = EnvGuard::clear("FINK_FAT__");

        let yaml = r#"
version: 1
unknown_key: 123
"#;
        let path = write_tmp_yaml(yaml);

        let err = load_engine_config_validated(&path).unwrap_err();
        match err {
            ConfigError::ConfigRs(_) => {}
            _ => panic!("expected ConfigError::ConfigRs, got {err:?}"),
        }
    }

    #[test]
    fn load_yaml_rejects_unknown_nested_keys() {
        let _guard = env_lock().lock().unwrap();
        let _clear = EnvGuard::clear("FINK_FAT__");

        // ScoreConfig has deny_unknown_fields; unknown nested key should fail.
        let yaml = r#"
version: 1
scoring:
  position:
    max_d2: 25.0
    i_am_not_real: 1
"#;
        let path = write_tmp_yaml(yaml);

        let err = load_engine_config_validated(&path).unwrap_err();
        match err {
            ConfigError::ConfigRs(_) => {}
            _ => panic!("expected ConfigError::ConfigRs, got {err:?}"),
        }
    }

    #[test]
    fn validate_rejects_unsupported_version() {
        let _guard = env_lock().lock().unwrap();
        let _clear = EnvGuard::clear("FINK_FAT__");

        let yaml = r#"
version: 2
"#;
        let path = write_tmp_yaml(yaml);

        let err = load_engine_config_validated(&path).unwrap_err();
        match err {
            ConfigError::UnsupportedVersion(2) => {}
            _ => panic!("expected UnsupportedVersion(2), got {err:?}"),
        }
    }

    #[test]
    fn validate_rejects_invalid_pairs() {
        let _guard = env_lock().lock().unwrap();
        let _clear = EnvGuard::clear("FINK_FAT__");

        let yaml = r#"
version: 1
pairs:
  max_dt: -0.01
"#;
        let path = write_tmp_yaml(yaml);

        let err = load_engine_config_validated(&path).unwrap_err();
        match err {
            ConfigError::Seed(_) => {}
            _ => panic!("expected ConfigError::Seed, got {err:?}"),
        }
    }

    #[test]
    fn validate_rejects_invalid_predictor() {
        let _guard = env_lock().lock().unwrap();
        let _clear = EnvGuard::clear("FINK_FAT__");

        // k_sigma <= 0 => Predictor error
        let yaml = r#"
version: 1
predictor:
  k_sigma: 0.0
"#;
        let path = write_tmp_yaml(yaml);

        let err = load_engine_config_validated(&path).unwrap_err();
        match err {
            ConfigError::Predictor(_) => {}
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

        // env > yaml
        let _env = EnvGuard::set(&[
            ("FINK_FAT__PAIRS__MAX_DT", "0.05"),
            ("FINK_FAT__SCORING__POSITION__MAX_D2", "9.0"),
            ("FINK_FAT__EDGES__TOP_K_PER_LEFT", "42"),
        ]);

        let cfg =
            load_engine_config_validated(&path).expect("config should load with env overrides");

        assert!((cfg.pairs.max_dt - 0.05).abs() < 1e-12);
        assert_eq!(cfg.edges.top_k_per_left, 42);
    }

    #[test]
    fn env_can_override_bool_and_option() {
        let _guard = env_lock().lock().unwrap();
        let _clear = EnvGuard::clear("FINK_FAT__");

        let yaml = r#"
version: 1
"#;
        let path = write_tmp_yaml(yaml);

        let _env = EnvGuard::set(&[
            ("FINK_FAT__PAIRS__ALLOW_SAME_TIMEBIN", "false"),
            ("FINK_FAT__EDGES__MAX_TOTAL_EDGES", "12345"),
            ("FINK_FAT__PREDICTOR__PAD_CELL_RADIUS", "false"),
        ]);

        let cfg =
            load_engine_config_validated(&path).expect("config should load with env overrides");

        assert_eq!(cfg.pairs.allow_same_timebin, false);
        assert_eq!(cfg.edges.predictor_config.pad_cell_radius, false);
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
    fn to_edge_config_is_consistent() {
        let _guard = env_lock().lock().unwrap();
        let _clear = EnvGuard::clear("FINK_FAT__");

        let yaml = r#"
version: 1
edges:
  top_k_per_left: 42
"#;
        let path = write_tmp_yaml(yaml);

        let cfg = load_engine_config_validated(&path).expect("config should load");
        let e = cfg.edges;

        assert_eq!(e.top_k_per_left, 42);
    }

    #[test]
    fn reject_invalid_edge_config() {
        let _guard = env_lock().lock().unwrap();
        let _clear = EnvGuard::clear("FINK_FAT__");

        let yaml = r#"
version: 1
edges:
  top_k_per_left: 0
"#;
        let path = write_tmp_yaml(yaml);

        let err = load_engine_config_validated(&path).unwrap_err();
        match err {
            ConfigError::Edges(_) => {}
            _ => panic!("expected ConfigError::Edges, got {err:?}"),
        }
    }

    #[test]
    fn yaml_accepts_units_for_pairs_and_triplets() {
        let _guard = env_lock().lock().unwrap();
        let _clear = EnvGuard::clear("FINK_FAT__");

        // pairs.max_dt: "86.4 min" = 0.06 day
        // pairs.max_angular_speed: "180 arcsec/hour"
        //   = 180 arcsec * 24 = 4320 arcsec/day = 1.2 deg/day
        let yaml = r#"
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
"#;

        let path = write_tmp_yaml(yaml);
        let cfg =
            load_engine_config_validated(&path).expect("config should load with unit strings");

        // Time → days
        assert_relative_eq!(cfg.pairs.max_dt, 0.06, epsilon = 1e-15);
        assert_relative_eq!(cfg.triplets.max_dt_between, 0.04, epsilon = 1e-15);

        // Angles → radians
        let expected_pair_sep = (9.0_f64 / 60.0_f64).to_radians();
        assert_relative_eq!(
            cfg.triplets.max_pair_sep,
            expected_pair_sep,
            max_relative = 1e-13
        );

        let expected_residual = (48.0_f64 / 3600.0_f64).to_radians();
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
        let _guard = env_lock().lock().unwrap();
        let _clear = EnvGuard::clear("FINK_FAT__");

        let yaml = r#"
version: 1
triplets:
  max_pair_sep: "10 parsec"
"#;

        let path = write_tmp_yaml(yaml);
        let err = load_engine_config_validated(&path).unwrap_err();

        match err {
            ConfigError::ConfigRs(_) => {}
            _ => panic!("expected ConfigError::ConfigRs, got {err:?}"),
        }
    }

    #[test]
    fn yaml_rejects_invalid_rate_syntax() {
        let _guard = env_lock().lock().unwrap();
        let _clear = EnvGuard::clear("FINK_FAT__");

        let yaml = r#"
version: 1
pairs:
  max_angular_speed: "10 arcmin"
"#;

        let path = write_tmp_yaml(yaml);
        let err = load_engine_config_validated(&path).unwrap_err();

        match err {
            ConfigError::ConfigRs(_) => {}
            _ => panic!("expected ConfigError::ConfigRs, got {err:?}"),
        }
    }

    #[test]
    fn numeric_values_still_pass_through_unchanged() {
        let _guard = env_lock().lock().unwrap();
        let _clear = EnvGuard::clear("FINK_FAT__");

        let yaml = r#"
version: 1
pairs:
  max_dt: 0.123456
"#;

        let path = write_tmp_yaml(yaml);
        let cfg =
            load_engine_config_validated(&path).expect("numeric values should still be accepted");

        // Numeric → exact passthrough
        assert_ulps_eq!(cfg.pairs.max_dt, 0.123456, max_ulps = 0);
    }
}
