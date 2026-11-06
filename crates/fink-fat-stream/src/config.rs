//! [`StreamConfig`]: the YAML configuration of `fink-fat-stream`.
//!
//! ```yaml
//! engine_config: engine_config_lsst.best.yaml   # the tracker's own configuration
//! ingest:
//!   error_model: FCCT14                          # astrometric error model of the night datasets
//!   quality_cuts: { min_snr: 5.0, max_ra_err_arcsec: 1.0, max_dec_err_arcsec: 1.0 }
//!   conversion: { astrometric_floor_arcsec: 0.0 }
//! assembler:
//!   visit_idle_timeout: "90 sec"
//! clock:
//!   twilight_altitude: "-12 deg"
//!   twilight_grace: "30 min"
//!   rollover_grace: "30 min"
//!   site: { lon: "-70.749417 deg", lat: "-30.244639 deg" }
//! replay:
//!   path: alerts.parquet
//!   latency: "3 min"
//!   batch_size: 1000
//!   columns: { dia_source_id: diaSourceId }      # dump column names, see ColumnMap
//! database:                                       # optional: no section, no Postgres
//!   url: postgres://engine@host/finkfat           # or env FINK_FAT_DATABASE_URL (keeps the password out)
//!   connect_timeout: "10 sec"
//!   flush_backoff_min: "30 sec"
//!   flush_backoff_max: "15 min"
//!   swap_retries: 12
//!   swap_retry_delay: "10 sec"
//! ```
//!
//! Every section but `engine_config` and `replay` is optional. Paths are
//! taken as given (relative to the working directory). Durations and angles
//! accept units, through the engine's
//! [`units`](fink_fat_engine::engine_config::units) helpers.

use std::time::Duration;

use camino::{Utf8Path, Utf8PathBuf};
use fink_fat_engine::engine_config::units::de_time_days;
use photom::{
    io::lsst_alert::{LsstConversionParams, LsstQualityCuts},
    observer::error_model::ObsErrorModel,
};
use serde::Deserialize;

use crate::{
    clock::ClockParams,
    error::{Result, StreamError},
    source::ReplayConfig,
};

/// `ingest:` section — what turns an alert into a tracked observation.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IngestConfig {
    /// Astrometric error model of the night datasets, needed to resolve the
    /// Rubin (`X05`) observer (default `FCCT14`, as `fink-fat track`).
    pub error_model: ObsErrorModel,
    /// Science-quality cuts (default: none).
    pub quality_cuts: LsstQualityCuts,
    /// Conversion parameters (default: no astrometric floor).
    pub conversion: LsstConversionParams,
}

impl Default for IngestConfig {
    fn default() -> Self {
        Self {
            error_model: ObsErrorModel::FCCT14,
            quality_cuts: LsstQualityCuts::default(),
            conversion: LsstConversionParams::default(),
        }
    }
}

/// `assembler:` section — see [`VisitAssembler`](crate::assembler::VisitAssembler).
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AssemblerConfig {
    /// A visit with no new alert for this long is complete, days (default
    /// 90 s).
    #[serde(deserialize_with = "de_time_days")]
    pub visit_idle_timeout: f64,
}

impl Default for AssemblerConfig {
    fn default() -> Self {
        Self {
            visit_idle_timeout: 90.0 / 86_400.0,
        }
    }
}

/// Environment variable holding the database URL when
/// [`DatabaseConfig::url`] is not set — keeps a password out of the YAML.
pub const DATABASE_URL_ENV: &str = "FINK_FAT_DATABASE_URL";

/// Seconds-scale durations are configured in days, like every engine
/// duration (units accepted, e.g. `"10 sec"`); this converts them.
fn days_to_duration(days: f64) -> Duration {
    Duration::from_secs_f64((days * 86_400.0).max(0.0))
}

/// `database:` section — where and how the Postgres sink writes (see
/// [`crate::sink`]). Absent: the driver runs without a database.
#[derive(Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DatabaseConfig {
    /// Postgres connection string; when `None`, read from
    /// [`DATABASE_URL_ENV`] (see [`Self::resolve_url`]). Never logged.
    pub url: Option<String>,
    /// Give up connecting after this long, days (default 10 s) — the sink
    /// must not stall the tracker on an unreachable server.
    #[serde(deserialize_with = "de_time_days")]
    pub connect_timeout: f64,
    /// Wait before retrying a failed flush, days (default 30 s), doubled
    /// after each further failure…
    #[serde(deserialize_with = "de_time_days")]
    pub flush_backoff_min: f64,
    /// …up to this, days (default 15 min).
    #[serde(deserialize_with = "de_time_days")]
    pub flush_backoff_max: f64,
    /// Retries of the nightly swap when it times out on the explorer's locks
    /// (default 12).
    pub swap_retries: u32,
    /// Pause between two swap attempts, days (default 10 s).
    #[serde(deserialize_with = "de_time_days")]
    pub swap_retry_delay: f64,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            url: None,
            connect_timeout: 10.0 / 86_400.0,
            flush_backoff_min: 30.0 / 86_400.0,
            flush_backoff_max: 15.0 / 1_440.0,
            swap_retries: 12,
            swap_retry_delay: 10.0 / 86_400.0,
        }
    }
}

/// Hand-written so the URL — which may embed a password — never reaches a
/// log line through `{:?}`.
impl std::fmt::Debug for DatabaseConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DatabaseConfig")
            .field("url", &self.url.as_ref().map(|_| "<redacted>"))
            .field("connect_timeout", &self.connect_timeout)
            .field("flush_backoff_min", &self.flush_backoff_min)
            .field("flush_backoff_max", &self.flush_backoff_max)
            .field("swap_retries", &self.swap_retries)
            .field("swap_retry_delay", &self.swap_retry_delay)
            .finish()
    }
}

impl DatabaseConfig {
    /// The connection string: [`Self::url`], else [`DATABASE_URL_ENV`].
    ///
    /// # Errors
    /// [`StreamError::Config`] when neither is set.
    pub fn resolve_url(&self) -> Result<String> {
        self.url
            .clone()
            .or_else(|| std::env::var(DATABASE_URL_ENV).ok())
            .ok_or_else(|| StreamError::Config {
                path: Utf8PathBuf::from("database.url"),
                message: format!("no database URL: set `database.url` or ${DATABASE_URL_ENV}"),
            })
    }

    /// [`Self::connect_timeout`] as a [`Duration`].
    pub fn connect_timeout(&self) -> Duration {
        days_to_duration(self.connect_timeout)
    }

    /// [`Self::flush_backoff_min`] / [`Self::flush_backoff_max`] as
    /// [`Duration`]s.
    pub fn flush_backoff(&self) -> (Duration, Duration) {
        (
            days_to_duration(self.flush_backoff_min),
            days_to_duration(self.flush_backoff_max),
        )
    }

    /// The swap retry policy of [`fink_fat_sql::night_sync`].
    pub fn swap_retry_policy(&self) -> fink_fat_sql::night_sync::SwapRetryPolicy {
        fink_fat_sql::night_sync::SwapRetryPolicy {
            retries: self.swap_retries,
            delay: days_to_duration(self.swap_retry_delay),
        }
    }
}

/// Whole configuration of `fink-fat-stream` (see the [module doc](self)).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamConfig {
    /// Path of the engine configuration (YAML), which also sets the
    /// `storage_path` holding the snapshot, archive, journal and logs.
    pub engine_config: Utf8PathBuf,
    /// Ingestion.
    #[serde(default)]
    pub ingest: IngestConfig,
    /// Visit assembly.
    #[serde(default)]
    pub assembler: AssemblerConfig,
    /// Night clock.
    #[serde(default)]
    pub clock: ClockParams,
    /// Replay source.
    pub replay: ReplayConfig,
    /// Postgres sink; `None` (no `database:` section) runs without one.
    #[serde(default)]
    pub database: Option<DatabaseConfig>,
}

/// The part of the configuration the [`Driver`](crate::Driver) needs —
/// everything but the source.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DriverConfig {
    /// Ingestion.
    pub ingest: IngestConfig,
    /// Visit assembly.
    pub assembler: AssemblerConfig,
    /// Night clock.
    pub clock: ClockParams,
    /// Postgres sink, if any.
    pub database: Option<DatabaseConfig>,
}

impl StreamConfig {
    /// The driver's part of the configuration.
    pub fn driver_config(&self) -> DriverConfig {
        DriverConfig {
            ingest: self.ingest,
            assembler: self.assembler,
            clock: self.clock,
            database: self.database.clone(),
        }
    }

    /// Parse a configuration from YAML text.
    ///
    /// # Errors
    /// [`StreamError::Config`] (with `origin` as its path) on invalid YAML or
    /// an unknown field.
    pub fn from_yaml(yaml: &str, origin: &Utf8Path) -> Result<Self> {
        serde_yaml::from_str(yaml).map_err(|e| StreamError::Config {
            path: origin.to_owned(),
            message: e.to_string(),
        })
    }

    /// Read and parse the configuration file at `path`.
    ///
    /// # Errors
    /// [`StreamError::Config`] when the file cannot be read or parsed.
    pub fn load(path: &Utf8Path) -> Result<Self> {
        let yaml = std::fs::read_to_string(path).map_err(|e| StreamError::Config {
            path: path.to_owned(),
            message: e.to_string(),
        })?;
        Self::from_yaml(&yaml, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_minimal_configuration_takes_every_default() {
        let config = StreamConfig::from_yaml(
            "engine_config: engine.yaml\nreplay:\n  path: dump.parquet\n",
            Utf8Path::new("stream.yaml"),
        )
        .unwrap();
        assert_eq!(config.engine_config, "engine.yaml");
        assert_eq!(config.ingest, IngestConfig::default());
        assert_eq!(config.assembler, AssemblerConfig::default());
        assert_eq!(config.clock, ClockParams::default());
        assert_eq!(config.replay.path, "dump.parquet");
    }

    #[test]
    fn sections_override_defaults() {
        let config = StreamConfig::from_yaml(
            "engine_config: e.yaml\n\
             ingest:\n  error_model: VFCC17\n  quality_cuts:\n    min_snr: 5.0\n\
             assembler:\n  visit_idle_timeout: \"2 min\"\n\
             replay:\n  path: d.parquet\n",
            Utf8Path::new("stream.yaml"),
        )
        .unwrap();
        assert_eq!(config.ingest.error_model, ObsErrorModel::VFCC17);
        assert_eq!(config.ingest.quality_cuts.min_snr, Some(5.0));
        assert!((config.assembler.visit_idle_timeout - 2.0 / 1_440.0).abs() < 1e-12);
    }

    #[test]
    fn the_database_section_is_optional_and_defaults() {
        let base = "engine_config: e.yaml\nreplay:\n  path: d.parquet\n";
        let config = StreamConfig::from_yaml(base, Utf8Path::new("s.yaml")).unwrap();
        assert_eq!(config.database, None);

        let config = StreamConfig::from_yaml(
            &format!("{base}database:\n  swap_retries: 3\n  connect_timeout: \"5 sec\"\n"),
            Utf8Path::new("s.yaml"),
        )
        .unwrap();
        let database = config.database.unwrap();
        assert_eq!(database.swap_retries, 3);
        let close = |d: Duration, secs: f64| (d.as_secs_f64() - secs).abs() < 1e-6;
        assert!(close(database.connect_timeout(), 5.0));
        let (min, max) = database.flush_backoff();
        assert!(close(min, 30.0) && close(max, 900.0), "{min:?} {max:?}");
    }

    #[test]
    fn the_database_url_never_shows_in_debug_output() {
        let config = DatabaseConfig {
            url: Some("postgres://engine:secret@host/db".into()),
            ..DatabaseConfig::default()
        };
        let debug = format!("{config:?}");
        assert!(!debug.contains("secret"), "{debug}");
        assert!(debug.contains("<redacted>"));
    }

    #[test]
    fn unknown_fields_are_refused() {
        let err = StreamConfig::from_yaml(
            "engine_config: e.yaml\nreplay:\n  path: d.parquet\ntypo: 1\n",
            Utf8Path::new("stream.yaml"),
        )
        .unwrap_err();
        assert!(matches!(err, StreamError::Config { .. }));
    }
}
