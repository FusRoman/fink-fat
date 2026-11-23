//! [`KafkaSource`]: the [`AlertSource`] reading Fink's Kafka livestream —
//! the only part of the Kafka source that talks to a broker. Decoding
//! ([`super::decode`]), offset bookkeeping ([`super::progress`]) and the
//! merge of the partitions ([`super::watermark`]) are pure and tested
//! without one.

use std::{
    collections::BTreeSet,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use rdkafka::{
    ClientConfig, Message, Offset, TopicPartitionList,
    consumer::{BaseConsumer, CommitMode, Consumer},
    message::BorrowedMessage,
};

use super::{
    KafkaConfig,
    decode::{DecodeError, SchemaCache, decode_alert, decode_payload},
    progress::{OffsetTracker, TopicPartition, start_offset},
    watermark::{Fetched, PartitionMerger, Pushed},
};
use crate::{
    config::days_to_duration,
    error::{Result, StreamError},
    record::{AlertRecord, unix_ms_to_mjd_utc},
    source::AlertSource,
};

/// Counters of a [`KafkaSource`], for the logs.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct KafkaSourceStats {
    /// Messages fetched from the broker.
    pub n_messages: u64,
    /// Messages decoded into alerts (delivered or still buffered).
    pub n_decoded: u64,
    /// Messages skipped (no schema, undecodable, null required field).
    pub n_skipped: u64,
    /// Alerts delivered to the driver, in arrival order.
    pub n_delivered: u64,
    /// Messages fetched a second time after a rebalance, ignored.
    pub n_refetched: u64,
    /// Fetched messages dropped with a revoked partition (its next owner
    /// reads them again).
    pub n_dropped: u64,
    /// Messages buffered right now, waiting for the watermark.
    pub n_buffered: u64,
    /// Consumer errors reported by `poll` (librdkafka retries on its own).
    pub n_poll_errors: u64,
    /// Failed offset commits (retried at the next acknowledgement).
    pub n_commit_failures: u64,
}

/// A message just fetched, decoded or not, before it joins the merge.
struct Polled {
    /// Where it comes from.
    partition: TopicPartition,
    /// Its offset there.
    offset: i64,
    /// Its Kafka timestamp, MJD UTC.
    message_mjd: Option<f64>,
    /// The alert, `None` when the message was skipped.
    record: Option<AlertRecord>,
}

/// Fink's Kafka livestream as an [`AlertSource`].
///
/// # Delivery, in arrival order
/// Fetched messages do not go straight to the driver: they wait in a
/// [`PartitionMerger`], which releases them in arrival order across the
/// topic's partitions, once no partition can still hold an older alert —
/// see [`super::watermark`] for why. [`Self::next_batch`] polls until
/// [`KafkaConfig::batch_size`] alerts are **released**, or
/// [`KafkaConfig::batch_timeout`], whichever comes first; it may return an
/// empty batch (nothing releasable: the driver advances its clock) and never
/// ends the stream. A partition whose buffer reaches
/// [`KafkaConfig::max_buffered_per_partition`] is paused until half
/// drained.
///
/// A message that cannot be decoded is logged, counted and skipped — like
/// a replay row with a null — but goes through the merge all the same, so
/// that its offset is committed in order and it is not redelivered forever.
///
/// # At least once
/// Auto-commit is off. [`Self::ack`] — called by the driver once the batch
/// is journaled — commits, for each assigned partition, the offset
/// following its last **released** message, synchronously. Buffered
/// messages are not committed: after a stop or a crash they are fetched
/// again. A failed commit is logged and retried at the next `ack`, never
/// fatal: at worst a restart redelivers alerts the driver already holds,
/// which it drops as duplicates or late.
///
/// # Clock
/// Each alert arrives at its message timestamp (see [`decode_alert`]).
/// Every [`KafkaConfig::caught_up_check_interval`], the partitions' low and
/// high watermark offsets are queried from the broker (and, for a partition
/// not fetched from yet, the group's committed offset): they tell which
/// partitions are consumed to their end. [`Self::caught_up`] holds when all
/// are, with nothing buffered, and the last batch was not full.
/// [`Self::wall_clock_mjd`] is the system clock.
pub struct KafkaSource {
    /// The consumer, subscribed to [`KafkaConfig::topics`].
    consumer: BaseConsumer,
    /// The source configuration.
    config: KafkaConfig,
    /// Writer schemas of the messages.
    schemas: SchemaCache,
    /// Offsets to commit at the next acknowledgement.
    offsets: OffsetTracker,
    /// Fetched messages waiting for the watermark.
    merger: PartitionMerger,
    /// Partitions paused because their buffer is full.
    paused: BTreeSet<TopicPartition>,
    /// Whether the source is caught up; `None` before the first batch.
    caught_up: Option<bool>,
    /// When the partitions' offsets were last queried.
    last_offsets_query: Option<Instant>,
    /// Counters.
    stats: KafkaSourceStats,
}

impl std::fmt::Debug for KafkaSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KafkaSource")
            .field("config", &self.config)
            .field("caught_up", &self.caught_up)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

/// A broker-side failure of the source, as a [`StreamError::Source`].
fn source_error(context: &str, error: impl std::fmt::Display) -> StreamError {
    StreamError::Source(format!("{context}: {error}"))
}

/// `partitions` as a librdkafka list (failures to add one are logged).
fn partition_list<'a>(
    partitions: impl IntoIterator<Item = &'a TopicPartition>,
) -> TopicPartitionList {
    let mut list = TopicPartitionList::new();
    for (topic, partition) in partitions {
        list.add_partition(topic, *partition);
    }
    list
}

impl KafkaSource {
    /// Create the consumer and subscribe to the configured topics.
    ///
    /// Connecting happens in the background: a broker that is down is not
    /// an error here, the first polls just return nothing.
    ///
    /// # Errors
    /// * [`StreamError::Config`] for an invalid configuration (see
    ///   [`KafkaConfig::client_properties`]) or an unreadable / invalid
    ///   `schema_path`;
    /// * [`StreamError::Source`] when librdkafka refuses a property or the
    ///   subscription.
    pub fn new(config: KafkaConfig) -> Result<Self> {
        let password = config
            .sasl
            .as_ref()
            .and_then(|sasl| sasl.resolve_password());
        let mut client_config = ClientConfig::new();
        for (key, value) in config.client_properties(password.as_deref())? {
            client_config.set(key, value);
        }
        let consumer: BaseConsumer = client_config
            .create()
            .map_err(|e| source_error("cannot create the Kafka consumer", e))?;
        let topics: Vec<&str> = config.topics.iter().map(String::as_str).collect();
        consumer
            .subscribe(&topics)
            .map_err(|e| source_error("cannot subscribe to the Kafka topics", e))?;

        let fallback = config
            .schema_path
            .as_deref()
            .map(|path| {
                let invalid = |message: String| StreamError::Config {
                    path: path.to_owned(),
                    message,
                };
                let text = std::fs::read_to_string(path).map_err(|e| invalid(e.to_string()))?;
                apache_avro::Schema::parse_str(&text).map_err(|e| invalid(e.to_string()))
            })
            .transpose()?;

        tracing::info!(
            topics = ?config.topics,
            group_id = %config.group_id,
            "subscribed to the Kafka alert stream"
        );
        Ok(Self {
            consumer,
            schemas: SchemaCache::new(fallback),
            offsets: OffsetTracker::default(),
            merger: PartitionMerger::default(),
            paused: BTreeSet::new(),
            caught_up: None,
            last_offsets_query: None,
            stats: KafkaSourceStats::default(),
            config,
        })
    }

    /// The source's counters.
    pub fn stats(&self) -> KafkaSourceStats {
        self.stats
    }

    /// Count and decode a fetched message.
    ///
    /// An associated function on the source's parts rather than a method:
    /// `message` borrows the consumer while these are updated.
    fn decode_message(
        message: &BorrowedMessage<'_>,
        config: &KafkaConfig,
        schemas: &mut SchemaCache,
        stats: &mut KafkaSourceStats,
    ) -> Polled {
        stats.n_messages += 1;
        let message_mjd = message.timestamp().to_millis().map(unix_ms_to_mjd_utc);
        let decoded = (|| {
            let payload = message.payload().ok_or(DecodeError::NoPayload)?;
            let schema = schemas.writer_schema(message.key())?;
            let alert = decode_payload(schema, payload)?;
            decode_alert(&alert, &config.layout, message_mjd, config.latency)
        })();
        let record = match decoded {
            Ok(record) => {
                stats.n_decoded += 1;
                Some(record)
            }
            Err(error) => {
                stats.n_skipped += 1;
                tracing::warn!(
                    topic = message.topic(),
                    partition = message.partition(),
                    offset = message.offset(),
                    %error,
                    "skipped an undecodable Kafka message"
                );
                None
            }
        };
        Polled {
            partition: (message.topic().to_string(), message.partition()),
            offset: message.offset(),
            message_mjd,
            record,
        }
    }

    /// Hand a fetched message to the merger, after following the
    /// consumer's assignment if the message comes from a partition the
    /// merger does not know yet (a rebalance happened inside `poll`).
    fn buffer(&mut self, polled: Polled) {
        if !self.merger.is_assigned(&polled.partition) {
            self.sync_assignment();
        }
        // A skipped message has no alert time: its timestamp, else the
        // partition's last arrival, keeps it in place.
        let arrival = polled.record.as_ref().map_or_else(
            || {
                polled
                    .message_mjd
                    .or_else(|| self.merger.last_arrival(&polled.partition))
                    .unwrap_or(f64::NEG_INFINITY)
            },
            |record| record.arrival_mjd,
        );
        let fetched = Fetched {
            offset: polled.offset,
            arrival,
            record: polled.record,
        };
        if self.merger.push(&polled.partition, fetched) == Pushed::Refetched {
            self.stats.n_refetched += 1;
        }
    }

    /// Track the consumer's current assignment in the merger and the pause
    /// set (see [`PartitionMerger::assign`]).
    fn sync_assignment(&mut self) {
        let assigned: Vec<TopicPartition> = match self.consumer.assignment() {
            Ok(list) => list
                .elements()
                .iter()
                .map(|element| (element.topic().to_string(), element.partition()))
                .collect(),
            Err(error) => {
                tracing::warn!(%error, "cannot read the Kafka assignment");
                return;
            }
        };
        let dropped = self.merger.assign(&assigned);
        self.paused.retain(|partition| assigned.contains(partition));
        if dropped > 0 {
            self.stats.n_dropped += dropped as u64;
            tracing::info!(
                dropped,
                "partitions revoked: their buffered alerts go to their next owner"
            );
        }
    }

    /// Move up to `batch_size − batch.len()` released alerts into `batch`,
    /// recording every released offset (skipped messages included) for the
    /// next commit.
    fn release_into(&mut self, batch: &mut Vec<AlertRecord>, batch_size: usize) {
        let room = batch_size.saturating_sub(batch.len());
        for released in self.merger.release(room) {
            let (topic, partition) = &released.partition;
            self.offsets.record(topic, *partition, released.offset);
            if let Some(record) = released.record {
                self.stats.n_delivered += 1;
                batch.push(record);
            }
        }
    }

    /// Query the partitions' watermark offsets (and, for partitions not
    /// fetched from yet, the group's committed offsets) if
    /// [`KafkaConfig::caught_up_check_interval`] has elapsed since the last
    /// query. A failed query is logged at debug level: the partition then
    /// just keeps holding the merge back until the next one.
    fn query_offsets_if_due(&mut self) {
        let interval = days_to_duration(self.config.caught_up_check_interval);
        if self
            .last_offsets_query
            .is_some_and(|last| last.elapsed() < interval)
        {
            return;
        }
        self.last_offsets_query = Some(Instant::now());
        let timeout = days_to_duration(self.config.watermark_timeout);

        let partitions: Vec<TopicPartition> = self.merger.partitions().cloned().collect();
        let mut bounds = Vec::with_capacity(partitions.len());
        for partition in partitions {
            match self
                .consumer
                .fetch_watermarks(&partition.0, partition.1, timeout)
            {
                Ok((low, high)) => {
                    self.merger.set_offsets(&partition, low, high);
                    bounds.push((partition, low, high));
                }
                Err(error) => {
                    tracing::debug!(%error, ?partition, "cannot query the partition watermarks");
                }
            }
        }

        // Partitions never fetched from: where do they resume?
        let unfetched: Vec<&(TopicPartition, i64, i64)> = bounds
            .iter()
            .filter(|(partition, ..)| self.merger.unfetched().any(|p| p == partition))
            .collect();
        if unfetched.is_empty() {
            return;
        }
        let list = partition_list(unfetched.iter().map(|(partition, ..)| partition));
        let committed = match self.consumer.committed_offsets(list, timeout) {
            Ok(committed) => committed,
            Err(error) => {
                tracing::debug!(%error, "cannot query the committed offsets");
                return;
            }
        };
        let reset_to_latest = self
            .config
            .extra
            .get("auto.offset.reset")
            .is_some_and(|reset| matches!(reset.as_str(), "latest" | "largest" | "end"));
        for (partition, low, high) in unfetched {
            let committed = committed
                .find_partition(&partition.0, partition.1)
                .and_then(|element| match element.offset() {
                    Offset::Offset(offset) => Some(offset),
                    _ => None,
                });
            self.merger.set_start_offset(
                partition,
                start_offset(committed, *low, *high, reset_to_latest),
            );
        }
    }

    /// Pause the partitions whose buffer is full and resume the drained
    /// ones (see [`PartitionMerger::flow_control`]). A failure is logged:
    /// the buffer just grows, or the partition stays paused, until the
    /// next batch retries.
    fn apply_flow_control(&mut self) {
        let (to_pause, to_resume) = self
            .merger
            .flow_control(&self.paused, self.config.max_buffered_per_partition);
        if !to_pause.is_empty() {
            match self.consumer.pause(&partition_list(&to_pause)) {
                Ok(()) => {
                    tracing::debug!(?to_pause, "partitions far ahead in time paused");
                    self.paused.extend(to_pause);
                }
                Err(error) => tracing::warn!(%error, "cannot pause Kafka partitions"),
            }
        }
        if !to_resume.is_empty() {
            match self.consumer.resume(&partition_list(&to_resume)) {
                Ok(()) => {
                    tracing::debug!(?to_resume, "partitions resumed");
                    for partition in &to_resume {
                        self.paused.remove(partition);
                    }
                }
                Err(error) => tracing::warn!(%error, "cannot resume Kafka partitions"),
            }
        }
    }

    /// Update [`Self::caught_up`] after a poll loop that `filled` its batch
    /// or not, logging every change of state with the counters so far —
    /// including the first one, so a run starting on a backlog says so.
    fn update_caught_up(&mut self, filled: bool) {
        let caught_up = !filled && self.merger.is_caught_up();
        if self.caught_up != Some(caught_up) {
            let state = if caught_up {
                "caught up with the Kafka stream"
            } else {
                "behind the Kafka stream, consuming a backlog"
            };
            tracing::info!(stats = ?self.stats, "{state}");
        }
        self.caught_up = Some(caught_up);
    }
}

/// Logs the counters when the source goes away (end of `fink-fat stream`).
impl Drop for KafkaSource {
    fn drop(&mut self) {
        tracing::info!(stats = ?self.stats, "Kafka source closed");
    }
}

impl AlertSource for KafkaSource {
    /// Up to `batch_size` alerts in arrival order, polled for at most
    /// `batch_timeout`; never `None` (a live stream does not end).
    ///
    /// # Steps
    /// 1. Follow the consumer's assignment (rebalances).
    /// 2. Poll, buffering every message in the merger and releasing what
    ///    the watermark allows, until the batch is full or the time is up.
    /// 3. If due, query the partitions' offsets, and release what partitions
    ///    found at their end no longer hold back.
    /// 4. Pause / resume partitions per their buffer, then update
    ///    [`Self::caught_up`].
    fn next_batch(&mut self) -> Result<Option<Vec<AlertRecord>>> {
        let batch_size = self.config.batch_size.max(1);
        let deadline = Instant::now() + days_to_duration(self.config.batch_timeout);
        let mut batch = Vec::new();
        self.sync_assignment();
        loop {
            self.release_into(&mut batch, batch_size);
            if batch.len() >= batch_size {
                break;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining == Duration::ZERO {
                break;
            }
            let polled = match self.consumer.poll(remaining) {
                None => break,
                Some(Ok(message)) => Some(Self::decode_message(
                    &message,
                    &self.config,
                    &mut self.schemas,
                    &mut self.stats,
                )),
                Some(Err(error)) => {
                    // Broker or network trouble: librdkafka reconnects and
                    // retries by itself; the batch just stays short.
                    self.stats.n_poll_errors += 1;
                    tracing::warn!(%error, "Kafka consumer error");
                    None
                }
            };
            if let Some(polled) = polled {
                self.buffer(polled);
            }
        }
        let filled = batch.len() >= batch_size;
        self.query_offsets_if_due();
        self.release_into(&mut batch, batch_size);
        self.apply_flow_control();
        self.stats.n_buffered = self.merger.n_buffered() as u64;
        self.update_caught_up(filled);
        Ok(Some(batch))
    }

    /// Commit the offsets of every message released so far (see the type
    /// doc); a failure is logged, never returned.
    fn ack(&mut self) -> Result<()> {
        if !self.offsets.is_dirty() {
            return Ok(());
        }
        // Only partitions still ours: a revoked partition's offsets are
        // its new owner's to commit.
        match self.consumer.assignment() {
            Ok(assignment) => {
                let assigned: Vec<TopicPartition> = assignment
                    .elements()
                    .iter()
                    .map(|element| (element.topic().to_string(), element.partition()))
                    .collect();
                self.offsets
                    .retain(|partition| assigned.contains(partition));
            }
            Err(error) => {
                self.stats.n_commit_failures += 1;
                tracing::warn!(%error, "cannot read the Kafka assignment, commit postponed");
                return Ok(());
            }
        }
        let mut list = TopicPartitionList::new();
        for ((topic, partition), next) in self.offsets.offsets() {
            if let Err(error) = list.add_partition_offset(topic, *partition, Offset::Offset(next)) {
                tracing::warn!(%error, topic, partition, "cannot list an offset to commit");
            }
        }
        if list.count() == 0 {
            self.offsets.mark_committed();
            return Ok(());
        }
        match self.consumer.commit(&list, CommitMode::Sync) {
            Ok(()) => self.offsets.mark_committed(),
            Err(error) => {
                self.stats.n_commit_failures += 1;
                tracing::warn!(%error, "Kafka offset commit failed, retried at the next batch");
            }
        }
        Ok(())
    }

    fn caught_up(&self) -> bool {
        self.caught_up == Some(true)
    }

    /// The system clock, MJD UTC.
    fn wall_clock_mjd(&self) -> Option<f64> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?;
        Some(unix_ms_to_mjd_utc(i64::try_from(now.as_millis()).ok()?))
    }
}
