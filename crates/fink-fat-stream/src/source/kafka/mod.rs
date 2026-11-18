//! The Kafka source: Fink's livestream of Rubin alerts.
//!
//! | Module | Role | Pure | Feature |
//! |--------|------|------|---------|
//! | [`config`] | [`KafkaConfig`], the librdkafka properties | yes | always |
//! | `decode` | Avro message → [`AlertRecord`](crate::AlertRecord) | yes | `kafka` |
//! | `progress` | Offsets to commit, end-of-partition test | yes | `kafka` |
//! | `watermark` | Merges the partitions back into arrival order (watermark per partition) | yes | `kafka` |
//! | `consumer` | `KafkaSource`, the [`AlertSource`](crate::AlertSource) | I/O | `kafka` |
//!
//! The configuration is always compiled, so a `kind: kafka` source parses
//! in any build; the consumer (librdkafka, built from source) needs the
//! `kafka` feature, which the `fink-fat` binary enables.
//!
//! The source is synchronous — a librdkafka `BaseConsumer` polled with a
//! timeout — like the rest of the driver: librdkafka fetches in its own
//! threads, and an async runtime would only wrap blocking code (the
//! engine's night close, the Postgres sink).

pub mod config;
#[cfg(feature = "kafka")]
pub mod consumer;
#[cfg(feature = "kafka")]
pub mod decode;
#[cfg(feature = "kafka")]
pub mod progress;
#[cfg(feature = "kafka")]
pub mod watermark;

pub use config::{AlertLayout, KAFKA_PASSWORD_ENV, KafkaConfig, SaslConfig};
#[cfg(feature = "kafka")]
pub use consumer::{KafkaSource, KafkaSourceStats};
