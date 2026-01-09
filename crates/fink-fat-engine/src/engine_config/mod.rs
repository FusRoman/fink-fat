pub mod edge_config;
pub mod error;
pub mod pair_config;
pub mod propagator_config;
pub mod score_config;
pub mod triplet_config;

use camino::Utf8Path;
use config::{Config, Environment, File};
use serde::{Deserialize, Serialize};

use crate::engine_config::{
    edge_config::{EdgeConfig, EdgeConfigFile},
    error::ConfigError,
    pair_config::PairConfig,
    propagator_config::PredictorParams,
    score_config::ScoreConfig,
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
    pub predictor: PredictorParams,
    pub scoring: ScoreConfig,
    pub edges: EdgeConfigFile,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            version: 1,
            pairs: PairConfig::default(),
            triplets: TripletConfig::default(),
            predictor: PredictorParams::default(),
            scoring: ScoreConfig::default(),
            edges: EdgeConfigFile::default(),
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

        // PredictorParams -> ConfigError via #[from]
        self.predictor.validate()?;

        // ScoringConfigError -> ConfigError via #[from]
        self.scoring.validate()?;

        // EdgeConfigError -> ConfigError via #[from]
        self.edges.validate()?;

        Ok(())
    }

    /// Build the runtime edge configuration used by the linker/scorer.
    pub fn to_edge_config(&self) -> EdgeConfig {
        EdgeConfig {
            top_k_per_left: self.edges.top_k_per_left,
            max_total_edges: self.edges.max_total_edges,
            predictor_config: self.predictor,
            score_config: self.scoring.clone(),
        }
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
        assert_eq!(cfg.edges.max_total_edges, None);

        // Predictor must be valid
        assert!(cfg.predictor.k_sigma > 0.0);
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
scoring:
  position:
    max_d2: 25.0
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
        assert!((cfg.scoring.position.max_d2 - 9.0).abs() < 1e-12);
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
        assert_eq!(cfg.edges.max_total_edges, Some(12345));
        assert_eq!(cfg.predictor.pad_cell_radius, false);
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
  max_total_edges: 1000
"#;
        let path = write_tmp_yaml(yaml);

        let cfg = load_engine_config_validated(&path).expect("config should load");
        let e = cfg.to_edge_config();

        assert_eq!(e.top_k_per_left, 42);
        assert_eq!(e.max_total_edges, Some(1000));

        // predictor copied
        assert!((e.predictor_config.k_sigma - cfg.predictor.k_sigma).abs() < 1e-12);

        // scoring cloned
        assert!((e.score_config.position.max_d2 - cfg.scoring.position.max_d2).abs() < 1e-12);
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
}
