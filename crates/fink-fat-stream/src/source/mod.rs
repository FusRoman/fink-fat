//! Where alerts come from: the [`AlertSource`] trait and its two
//! implementations — [`ReplaySource`], a parquet dump replayed as if live
//! (tests, validation), and the Kafka consumer of Fink's livestream
//! (`kafka::KafkaSource`, `kafka` feature), the production source.

pub mod columns;
pub mod kafka;
pub mod replay;

pub use columns::ColumnMap;
pub use kafka::KafkaConfig;
#[cfg(feature = "kafka")]
pub use kafka::KafkaSource;
pub use replay::{ReplayConfig, ReplaySource};

use serde::Deserialize;

use crate::{error::Result, record::AlertRecord};

/// `source:` section of the stream configuration: which [`AlertSource`]
/// the driver reads, selected by `kind`.
///
/// ```yaml
/// source:
///   kind: replay          # or: kafka (see KafkaConfig)
///   path: alerts.parquet
/// ```
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[allow(
    clippy::large_enum_variant,
    reason = "read once from the configuration file and consumed at start-up"
)]
pub enum SourceConfig {
    /// A parquet dump replayed as if live ([`ReplayConfig`]).
    Replay(ReplayConfig),
    /// Fink's Kafka livestream ([`KafkaConfig`]).
    Kafka(KafkaConfig),
}

/// A stream of alerts delivered in batches, *at least once*.
///
/// # Protocol
///
/// The driver loops on:
///
/// 1. [`Self::next_batch`];
/// 2. process the batch and write it durably to its journal;
/// 3. [`Self::ack`].
///
/// # Contract for implementers
///
/// * **At least once**: a batch that was not acknowledged may be delivered
///   again after a restart — the driver deduplicates by `diaSourceId` and
///   drops alerts of closed nights. Never *lose* an unacknowledged alert.
/// * **Order**: alerts should come in arrival order. The driver's clock is
///   the latest arrival seen, so an alert delivered well after later ones
///   finds its visit already advanced (it then only feeds the night's
///   discovery, not the visit's association) or its night already closed
///   (it is dropped as late). A source reading several queues must merge
///   them: the Kafka source does, across the topic's partitions (see
///   `kafka::watermark`, `kafka` feature).
/// * **`arrival_mjd`** of each record must be on UTC (see
///   [`crate::record`]): it drives visit completion and night closing.
/// * **`caught_up`** must only be `true` when no backlog is pending:
///   it allows the night to close at twilight.
pub trait AlertSource {
    /// The next batch of alerts, `Ok(None)` once the stream is over (a
    /// replay reached its end; a live source never ends). An empty batch
    /// means "nothing right now" and lets the driver advance its clock.
    ///
    /// # Errors
    /// The source failed; the driver stops.
    fn next_batch(&mut self) -> Result<Option<Vec<AlertRecord>>>;

    /// Acknowledge every batch returned so far: it is durably journaled
    /// and must not be redelivered.
    ///
    /// # Errors
    /// The acknowledgement failed; the driver stops.
    fn ack(&mut self) -> Result<()>;

    /// Whether the source has delivered everything available right now
    /// (no backlog). Required by the morning-twilight close (see
    /// [`crate::clock`]).
    fn caught_up(&self) -> bool;

    /// Wall-clock time, MJD UTC, for a live source; `None` for a source
    /// whose only clock is its alerts' event time (a replay).
    ///
    /// Only consulted while [`Self::caught_up`]: it lets the clock reach
    /// twilight after the night's last alert, when no event time advances
    /// it any more.
    fn wall_clock_mjd(&self) -> Option<f64> {
        None
    }
}
