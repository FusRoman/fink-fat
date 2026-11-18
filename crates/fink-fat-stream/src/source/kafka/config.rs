//! [`KafkaConfig`]: the `source: {kind: kafka}` section of the stream
//! configuration, and the librdkafka properties it turns into
//! ([`KafkaConfig::client_properties`], pure).
//!
//! Compiled with or without the `kafka` feature, so a configuration naming
//! the Kafka source always parses; only building the consumer needs the
//! feature.

use std::collections::BTreeMap;

use camino::Utf8PathBuf;
use fink_fat_engine::engine_config::units::de_time_days;
use serde::Deserialize;

use crate::{
    config::days_to_duration,
    error::{Result, StreamError},
    source::ColumnMap,
};

/// Environment variable holding the SASL password when
/// [`SaslConfig::password`] is not set — keeps it out of the YAML.
pub const KAFKA_PASSWORD_ENV: &str = "FINK_FAT_KAFKA_PASSWORD";

/// librdkafka properties the source sets itself and [`KafkaConfig::extra`]
/// may not override: the offset handling the at-least-once guarantee rests
/// on, the identity of the consumer, and the credentials (which have their
/// own, redacted, fields).
pub const PROTECTED_PROPERTIES: [&str; 8] = [
    "bootstrap.servers",
    "group.id",
    "enable.auto.commit",
    "enable.auto.offset.store",
    "max.poll.interval.ms",
    "sasl.mechanisms",
    "sasl.username",
    "sasl.password",
];

fn default_mechanism() -> String {
    "SCRAM-SHA-512".into()
}

/// `source.sasl:` — SASL authentication (Fink: SCRAM-SHA-512 over
/// `sasl_plaintext`).
#[derive(Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SaslConfig {
    /// `sasl.mechanisms` (default `SCRAM-SHA-512`).
    #[serde(default = "default_mechanism")]
    pub mechanism: String,
    /// `sasl.username`.
    pub username: String,
    /// `sasl.password`; when `None`, read from [`KAFKA_PASSWORD_ENV`] (see
    /// [`Self::resolve_password`]). Never logged.
    #[serde(default)]
    pub password: Option<String>,
}

/// Hand-written so the password never reaches a log line through `{:?}`.
impl std::fmt::Debug for SaslConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SaslConfig")
            .field("mechanism", &self.mechanism)
            .field("username", &self.username)
            .field("password", &self.password.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

impl SaslConfig {
    /// The password: [`Self::password`], else [`KAFKA_PASSWORD_ENV`], else
    /// none (a SCRAM user may be set up without one).
    pub fn resolve_password(&self) -> Option<String> {
        self.password
            .clone()
            .or_else(|| std::env::var(KAFKA_PASSWORD_ENV).ok())
    }
}

fn default_record() -> String {
    "diaSource".into()
}

fn default_ingest_mjd() -> Option<String> {
    Some("brokerIngestMjd".into())
}

/// `source.layout:` — where the [`AlertRecord`](crate::AlertRecord) fields
/// live in a decoded Avro alert. The defaults match Fink's LSST alerts
/// (checked on real messages): a top-level `diaSource` record holding the
/// Rubin-named fields, and Fink's `brokerIngestMjd` at the top level.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AlertLayout {
    /// Top-level field holding the `DiaSource` record (default `diaSource`).
    pub record: String,
    /// Names of the fields inside that record (default: Rubin names).
    pub fields: ColumnMap,
    /// Top-level field holding the broker's ingestion time, MJD UTC
    /// (default `brokerIngestMjd`); the arrival time of a message without a
    /// Kafka timestamp. `null` disables it.
    pub ingest_mjd: Option<String>,
}

impl Default for AlertLayout {
    fn default() -> Self {
        Self {
            record: default_record(),
            fields: ColumnMap::default(),
            ingest_mjd: default_ingest_mjd(),
        }
    }
}

fn default_batch_size() -> usize {
    1_000
}

fn default_batch_timeout() -> f64 {
    1.0 / 86_400.0
}

fn default_max_poll_interval() -> f64 {
    30.0 / 1_440.0
}

fn default_caught_up_check_interval() -> f64 {
    30.0 / 86_400.0
}

fn default_watermark_timeout() -> f64 {
    5.0 / 86_400.0
}

fn default_max_buffered_per_partition() -> usize {
    10_000
}

fn default_latency() -> f64 {
    3.0 / 1_440.0
}

/// `source:` section with `kind: kafka` — the live alert stream (see
/// [`crate::source::kafka`]).
///
/// ```yaml
/// source:
///   kind: kafka
///   bootstrap_servers: kafka.example.org:24499
///   topics: [fink_sso_candidate_lsst]
///   group_id: my_group
///   sasl: { username: my_user }        # password: env FINK_FAT_KAFKA_PASSWORD
///   batch_size: 1000
///   batch_timeout: "1 sec"
/// ```
///
/// Durations accept units, like every duration of the configuration.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KafkaConfig {
    /// `bootstrap.servers`: `host:port[,host:port…]`.
    pub bootstrap_servers: String,
    /// Topics to subscribe to (at least one).
    pub topics: Vec<String>,
    /// `group.id`: the consumer group whose committed offsets say where a
    /// restart resumes.
    pub group_id: String,
    /// SASL authentication; `None` connects in plaintext without it (a
    /// local broker).
    #[serde(default)]
    pub sasl: Option<SaslConfig>,
    /// Most alerts per batch (default 1000).
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// Longest wait for a batch to fill, days (default 1 s): an empty or
    /// partial batch after it lets the driver's clock advance.
    #[serde(default = "default_batch_timeout", deserialize_with = "de_time_days")]
    pub batch_timeout: f64,
    /// librdkafka's `max.poll.interval.ms`, days (default 30 min): the
    /// longest the driver may go without polling — a night close runs the
    /// engine's discovery for minutes — before the broker evicts the
    /// consumer from its group.
    #[serde(
        default = "default_max_poll_interval",
        deserialize_with = "de_time_days"
    )]
    pub max_poll_interval: f64,
    /// Minimum time between two queries of the partitions' high watermark
    /// offsets, days (default 30 s). They tell whether a partition is
    /// consumed to its end — which says whether the source is caught up,
    /// and lets an idle partition stop holding back the merge of the
    /// others (see `source::kafka::watermark`, `kafka` feature).
    #[serde(
        default = "default_caught_up_check_interval",
        deserialize_with = "de_time_days"
    )]
    pub caught_up_check_interval: f64,
    /// Most fetched alerts one partition buffers, waiting for the others to
    /// catch up in time, before it is paused (default 10 000; see
    /// `source::kafka::watermark`, `kafka` feature).
    #[serde(default = "default_max_buffered_per_partition")]
    pub max_buffered_per_partition: usize,
    /// Timeout of one watermark query, days (default 5 s).
    #[serde(
        default = "default_watermark_timeout",
        deserialize_with = "de_time_days"
    )]
    pub watermark_timeout: f64,
    /// Where the alert fields are (default: Fink's LSST layout).
    #[serde(default)]
    pub layout: AlertLayout,
    /// Avro schema file used when a message key does not hold a parsable
    /// schema (Fink puts the writer schema in the key).
    #[serde(default)]
    pub schema_path: Option<Utf8PathBuf>,
    /// Last-resort arrival delay after the exposure mid-point, days
    /// (default 3 min), for a message with neither a Kafka timestamp nor an
    /// ingestion time.
    #[serde(default = "default_latency", deserialize_with = "de_time_days")]
    pub latency: f64,
    /// Further librdkafka properties (e.g. `security.protocol: sasl_ssl`,
    /// `auto.offset.reset: latest`), applied last; any of
    /// [`PROTECTED_PROPERTIES`] is refused.
    #[serde(default)]
    pub extra: BTreeMap<String, String>,
}

impl KafkaConfig {
    /// The librdkafka properties of the consumer, in the order they are set.
    ///
    /// * identity: `bootstrap.servers`, `group.id`;
    /// * offsets: `enable.auto.commit=false` and
    ///   `enable.auto.offset.store=false` — the source commits exactly the
    ///   offsets of the batches the driver acknowledged (journaled), never
    ///   more; `auto.offset.reset=earliest` — a new group starts at the
    ///   oldest retained alert rather than skipping the backlog;
    /// * `max.poll.interval.ms` from [`Self::max_poll_interval`];
    /// * SASL when [`Self::sasl`] is set: `security.protocol=sasl_plaintext`
    ///   (Fink's), `sasl.mechanisms`, `sasl.username`, and `sasl.password`
    ///   when `password` is `Some`;
    /// * [`Self::extra`], last, so it can override the defaults above that
    ///   are not [`PROTECTED_PROPERTIES`].
    ///
    /// # Arguments
    /// * `password` – The SASL password, already resolved
    ///   ([`SaslConfig::resolve_password`]); ignored without SASL.
    ///
    /// # Errors
    /// [`StreamError::Config`] when no topic is configured, or
    /// [`Self::extra`] sets a protected property.
    pub fn client_properties(&self, password: Option<&str>) -> Result<Vec<(String, String)>> {
        let invalid = |message: String| StreamError::Config {
            path: Utf8PathBuf::from("source"),
            message,
        };
        if self.topics.is_empty() {
            return Err(invalid(
                "`topics` is empty: subscribe to at least one".into(),
            ));
        }
        if let Some(key) = self
            .extra
            .keys()
            .find(|key| PROTECTED_PROPERTIES.contains(&key.as_str()))
        {
            return Err(invalid(format!(
                "`extra` may not set `{key}`, which the source controls (see its own field)"
            )));
        }

        let max_poll_ms = days_to_duration(self.max_poll_interval).as_millis();
        let mut properties: Vec<(String, String)> = vec![
            ("bootstrap.servers".into(), self.bootstrap_servers.clone()),
            ("group.id".into(), self.group_id.clone()),
            ("enable.auto.commit".into(), "false".into()),
            ("enable.auto.offset.store".into(), "false".into()),
            ("auto.offset.reset".into(), "earliest".into()),
            ("max.poll.interval.ms".into(), max_poll_ms.to_string()),
        ];
        if let Some(sasl) = &self.sasl {
            properties.push(("security.protocol".into(), "sasl_plaintext".into()));
            properties.push(("sasl.mechanisms".into(), sasl.mechanism.clone()));
            properties.push(("sasl.username".into(), sasl.username.clone()));
            if let Some(password) = password {
                properties.push(("sasl.password".into(), password.to_string()));
            }
        }
        properties.extend(self.extra.iter().map(|(k, v)| (k.clone(), v.clone())));
        Ok(properties)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(yaml: &str) -> KafkaConfig {
        serde_yaml::from_str(yaml).unwrap()
    }

    const MINIMAL: &str = "bootstrap_servers: localhost:19092\ntopics: [alerts]\ngroup_id: g\n";

    fn property<'a>(properties: &'a [(String, String)], key: &str) -> Option<&'a str> {
        properties
            .iter()
            .rev()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    #[test]
    fn a_minimal_section_takes_the_fink_defaults() {
        let config = config(MINIMAL);
        assert_eq!(config.batch_size, 1_000);
        assert_eq!(config.max_buffered_per_partition, 10_000);
        assert!((config.batch_timeout - 1.0 / 86_400.0).abs() < 1e-15);
        assert_eq!(config.layout.record, "diaSource");
        assert_eq!(config.layout.fields, ColumnMap::default());
        assert_eq!(config.layout.ingest_mjd.as_deref(), Some("brokerIngestMjd"));
        assert_eq!(config.sasl, None);
    }

    #[test]
    fn properties_disable_auto_commit_and_add_sasl() {
        let config = config(&format!(
            "{MINIMAL}sasl: {{ username: me }}\nmax_poll_interval: \"10 min\"\n"
        ));
        let properties = config.client_properties(Some("pw")).unwrap();
        assert_eq!(property(&properties, "enable.auto.commit"), Some("false"));
        assert_eq!(
            property(&properties, "enable.auto.offset.store"),
            Some("false")
        );
        assert_eq!(property(&properties, "auto.offset.reset"), Some("earliest"));
        assert_eq!(
            property(&properties, "max.poll.interval.ms"),
            Some("600000")
        );
        assert_eq!(
            property(&properties, "security.protocol"),
            Some("sasl_plaintext")
        );
        assert_eq!(
            property(&properties, "sasl.mechanisms"),
            Some("SCRAM-SHA-512")
        );
        assert_eq!(property(&properties, "sasl.username"), Some("me"));
        assert_eq!(property(&properties, "sasl.password"), Some("pw"));
    }

    #[test]
    fn extra_overrides_defaults_but_not_protected_properties() {
        let config = config(&format!(
            "{MINIMAL}extra: {{ security.protocol: sasl_ssl, auto.offset.reset: latest }}\n"
        ));
        let properties = config.client_properties(None).unwrap();
        assert_eq!(property(&properties, "auto.offset.reset"), Some("latest"));
        assert_eq!(property(&properties, "security.protocol"), Some("sasl_ssl"));

        let config = self::config(&format!(
            "{MINIMAL}extra: {{ enable.auto.commit: \"true\" }}\n"
        ));
        assert!(config.client_properties(None).is_err());
    }

    #[test]
    fn no_topic_is_refused() {
        let config = config("bootstrap_servers: b\ntopics: []\ngroup_id: g\n");
        assert!(config.client_properties(None).is_err());
    }

    #[test]
    fn the_password_never_shows_in_debug_output() {
        let config = config(&format!(
            "{MINIMAL}sasl: {{ username: me, password: secret }}\n"
        ));
        let debug = format!("{config:?}");
        assert!(!debug.contains("secret"), "{debug}");
        assert!(debug.contains("<redacted>"));
    }
}
