//! [`PartitionMerger`]: merges the partitions of the Kafka topic back into
//! **one stream in arrival order**, using a watermark per partition. Pure:
//! the consumer feeds it the messages it fetches and the broker's offsets,
//! and delivers what it releases.
//!
//! # The problem: partitions drift apart in time
//!
//! Fink spreads the alerts of each visit over all the partitions of its
//! topic (10 for `fink_is_solar_system_lsst`). Each partition is a queue
//! **ordered in time**, but librdkafka reads them in turns of a few
//! messages each: it balances partitions by *message count*, not by
//! *time*. A partition holding few alerts over some period therefore races
//! ahead in time, and a dense one falls behind. On the live stream, after
//! 1200 messages, some partitions had delivered alerts up to t = 366 s,
//! others only up to t = 240 s.
//!
//! The rest of the pipeline expects alerts in roughly arrival order: the
//! driver's clock is the **latest** arrival seen, and the visit assembler
//! declares a visit complete once that clock is 90 s past the visit's last
//! alert. Fed straight from the partitions, the clock follows the
//! *fastest* partition:
//!
//! ```text
//!   visit V published between t = 100 s and t = 160 s, on partitions P1 and P9
//!
//!   read order        P9: V (100–160 s) … next visits … 300 s
//!                     P1:                                       V (120–150 s)
//!   clock (max seen)  160 s ─────────────────────────► 300 s
//!   assembler         300 − 160 > 90 s: V complete, with P9's alerts only
//!   P1's V alerts     arrive after V was advanced: leftovers, no association
//! ```
//!
//! The first run on Fink's stream lost a quarter of its alerts this way
//! (607 leftovers out of 2293). At a night boundary it is worse: the fast
//! partition's first alert of the next night closes the current night, and
//! the slow partitions' last alerts of that night become *late* — not
//! tracked at all.
//!
//! # The fix: a watermark per partition
//!
//! An alert is released only once no partition can still hold an older
//! one. Each partition is ordered in time, so once partition P has
//! delivered an alert of time t, nothing P delivers later is older than t.
//! Each partition therefore gives a **bound** — the earliest time it may
//! still deliver:
//!
//! | Partition state | Bound |
//! |---|---|
//! | alerts waiting in its buffer | the time of the oldest one |
//! | nothing waiting, not consumed to its end | the time of the last alert fetched from it (`−∞` before the first) |
//! | nothing waiting, consumed to its end | none: it holds nothing back |
//!
//! The **watermark** is the smallest bound. Fetched alerts wait in a buffer
//! per partition; an alert is released — handed to the driver — only once
//! it is not later than the watermark, oldest first across all partitions.
//! On the same example:
//!
//! ```text
//!   P9 fetched 100, 160, 200, 300 s;  P1 fetched 90 s       watermark = 90 s
//!     released: 90 s (P1). All of P9 waits in its buffer; V is still open
//!   P1 fetches its V alerts, 120 and 150 s                  watermark = 150 s
//!     released: 100 (P9), 120, 150 (P1) — V's alerts, in time order
//!   P1 fetches 260 s                                        watermark = 260 s
//!     released: 160, 200 (P9), 260 (P1). The clock passes 160 + 90 s only
//!     now, and V is complete with every one of its alerts
//! ```
//!
//! Downstream, the driver sees what a single, time-ordered queue would
//! deliver: its clock, the visit assembler and the night boundaries work
//! unchanged, and the replay source needs nothing of this.
//!
//! # The trap: a partition with nothing to say
//!
//! A partition that receives nothing — the end of the night, a sparse
//! topic, an empty partition — keeps the bound of its last alert forever,
//! and would freeze the watermark. Hence the third row of the table: a
//! partition **consumed to its end** (fetched up to the high watermark the
//! broker reported, nothing waiting in its buffer) does not take part. The
//! broker is asked for those high watermarks periodically
//! ([`KafkaConfig::caught_up_check_interval`](super::KafkaConfig::caught_up_check_interval));
//! as soon as a partition is fetched past the last high watermark known,
//! that knowledge is stale and the partition holds the watermark again
//! until the next query.
//!
//! # Ties and imperfect order
//!
//! Alerts of one exposure often share their arrival time; ties are broken
//! by `diaSourceId` (`(arrival, diaSourceId)` is the merge key), as the
//! replay source does — a topic produced in that order is released in
//! exactly the replay's order. Fink's partitions are ordered by time only
//! to within a fraction of a second; an alert slightly older than its
//! predecessor in the same partition is simply released a little late,
//! well within the assembler's timeout.
//!
//! # Memory
//!
//! A partition far ahead in time accumulates alerts in its buffer while the
//! others catch up. [`PartitionMerger::flow_control`] tells the consumer to
//! pause a partition whose buffer is full and to resume it once half
//! drained, so a long backlog does not grow the buffers without bound. It
//! cannot stall the merge: the partition holding the watermark back either
//! has its oldest alert releasable right away, or has an empty buffer and
//! is never paused.
//!
//! # Rebalances
//!
//! Partitions revoked from the consumer are dropped with their buffers:
//! their alerts were never delivered, so never committed, and the
//! partition's next owner reads them again. A partition assigned back
//! restarts at its committed offset, behind what this merger already
//! fetched; those messages are recognised by their offset and ignored.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, VecDeque},
};

use super::progress::{TopicPartition, partition_caught_up};
use crate::record::AlertRecord;

/// Merge key: arrival time (MJD UTC), then `diaSourceId`.
type Key = (f64, i64);

/// Total order on [`Key`]s.
fn compare_keys(a: &Key, b: &Key) -> Ordering {
    a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))
}

/// A message fetched from a partition, waiting for the watermark.
#[derive(Debug, Clone, PartialEq)]
pub struct Fetched {
    /// Its offset in the partition.
    pub offset: i64,
    /// Its arrival time, MJD UTC (the decoded alert's, or the message
    /// timestamp of a skipped message).
    pub arrival: f64,
    /// The decoded alert; `None` for a skipped (undecodable) message, which
    /// still goes through the merge so that its offset is committed in
    /// order.
    pub record: Option<AlertRecord>,
}

impl Fetched {
    /// Its merge key; a skipped message sorts before the alerts of its
    /// arrival time.
    fn key(&self) -> Key {
        let id = self
            .record
            .as_ref()
            .map_or(i64::MIN, |record| record.dia_source_id);
        (self.arrival, id)
    }
}

/// A message released by [`PartitionMerger::release`].
#[derive(Debug, Clone, PartialEq)]
pub struct Released {
    /// The partition it came from.
    pub partition: TopicPartition,
    /// Its offset there — to commit once the driver acknowledged it.
    pub offset: i64,
    /// The alert; `None` for a skipped message (only its offset matters).
    pub record: Option<AlertRecord>,
}

/// What [`PartitionMerger::push`] did with a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pushed {
    /// Buffered, waiting for the watermark.
    Buffered,
    /// Already fetched before (the partition was assigned back after a
    /// rebalance and restarted behind): ignored.
    Refetched,
}

/// One partition: its buffer and what is known of its position.
#[derive(Debug, Default, Clone)]
struct PartitionState {
    /// Fetched messages not released yet, in fetch order.
    buffer: VecDeque<Fetched>,
    /// Offset following the last message fetched — or, before the first
    /// fetch, the offset consumption resumes from, when known (see
    /// [`PartitionMerger::set_start_offset`]).
    next_offset: Option<i64>,
    /// Key of the last message fetched (`None` before the first).
    last_key: Option<Key>,
    /// `(low, high)` watermark offsets from the last broker query; `None`
    /// before the first query, or once a fetch went past `high` (stale).
    offsets: Option<(i64, i64)>,
}

/// The constraint an empty partition puts on the release.
enum Bound {
    /// Consumed to its end: none.
    Free,
    /// Nothing may be released after this key.
    UpTo(Key),
    /// Nothing fetched yet, end unknown: nothing may be released.
    Blocked,
}

impl PartitionState {
    /// Whether the partition is consumed to its end: nothing buffered, and
    /// fetched up to the last high watermark known.
    fn at_end(&self) -> bool {
        self.buffer.is_empty()
            && self
                .offsets
                .is_some_and(|(low, high)| partition_caught_up(self.next_offset, low, high))
    }

    /// The constraint of an **empty** partition (see the
    /// [module doc](self#the-fix-a-watermark-per-partition)).
    fn bound(&self) -> Bound {
        if self.at_end() {
            return Bound::Free;
        }
        match self.last_key {
            Some(key) => Bound::UpTo(key),
            None => Bound::Blocked,
        }
    }
}

/// Merges the assigned partitions into one stream in arrival order (see
/// the [module doc](self)).
#[derive(Debug, Default, Clone)]
pub struct PartitionMerger {
    /// The assigned partitions.
    partitions: BTreeMap<TopicPartition, PartitionState>,
}

impl PartitionMerger {
    /// Track exactly `assigned`: new partitions start unknown (holding the
    /// watermark at `−∞` until they deliver or are known to be empty),
    /// revoked ones are dropped with their buffers (see the
    /// [module doc](self#rebalances)).
    ///
    /// # Returns
    /// The number of buffered messages dropped with revoked partitions.
    pub fn assign(&mut self, assigned: &[TopicPartition]) -> usize {
        let mut dropped = 0;
        self.partitions.retain(|partition, state| {
            let keep = assigned.contains(partition);
            if !keep {
                dropped += state.buffer.len();
            }
            keep
        });
        for partition in assigned {
            self.partitions.entry(partition.clone()).or_default();
        }
        dropped
    }

    /// Whether `partition` is tracked (assigned).
    pub fn is_assigned(&self, partition: &TopicPartition) -> bool {
        self.partitions.contains_key(partition)
    }

    /// The assigned partitions.
    pub fn partitions(&self) -> impl Iterator<Item = &TopicPartition> {
        self.partitions.keys()
    }

    /// The assigned partitions nothing was fetched from yet.
    pub fn unfetched(&self) -> impl Iterator<Item = &TopicPartition> {
        self.partitions
            .iter()
            .filter(|(_, state)| state.last_key.is_none())
            .map(|(partition, _)| partition)
    }

    /// Arrival time of the last message fetched from `partition`, if any —
    /// the arrival given to a skipped message without a timestamp, so that
    /// it neither jumps the queue nor holds the others back.
    pub fn last_arrival(&self, partition: &TopicPartition) -> Option<f64> {
        self.partitions
            .get(partition)
            .and_then(|state| state.last_key)
            .map(|(arrival, _)| arrival)
    }

    /// Buffer `message`, fetched from `partition`.
    ///
    /// The partition is tracked if it was not (the consumer only fetches
    /// from assigned partitions). A message at an offset already passed is
    /// a re-fetch after a rebalance and is ignored. Fetching at or past the
    /// last known high watermark makes it stale (see the
    /// [module doc](self#the-trap-a-partition-with-nothing-to-say)).
    pub fn push(&mut self, partition: &TopicPartition, message: Fetched) -> Pushed {
        let state = self.partitions.entry(partition.clone()).or_default();
        if state.last_key.is_some() && state.next_offset.is_some_and(|n| message.offset < n) {
            return Pushed::Refetched;
        }
        state.next_offset = Some(message.offset + 1);
        state.last_key = Some(message.key());
        if state
            .offsets
            .is_some_and(|(_, high)| message.offset >= high)
        {
            state.offsets = None;
        }
        state.buffer.push_back(message);
        Pushed::Buffered
    }

    /// Record the offset consumption of `partition` resumes from (its
    /// committed offset), before anything was fetched from it: an assigned
    /// partition with no new message is then known to be at its end at the
    /// next [`Self::set_offsets`]. Ignored once the partition delivered.
    pub fn set_start_offset(&mut self, partition: &TopicPartition, offset: i64) {
        if let Some(state) = self.partitions.get_mut(partition)
            && state.last_key.is_none()
        {
            state.next_offset = Some(offset);
        }
    }

    /// Record the broker's `(low, high)` watermark offsets of `partition`
    /// (oldest retained offset, offset of the next message to be written).
    /// Ignored for a partition not assigned.
    pub fn set_offsets(&mut self, partition: &TopicPartition, low: i64, high: i64) {
        if let Some(state) = self.partitions.get_mut(partition) {
            state.offsets = Some((low, high));
        }
    }

    /// Release up to `max` buffered messages that no partition can precede
    /// any more, oldest first across partitions (see the
    /// [module doc](self#the-fix-a-watermark-per-partition)).
    ///
    /// # Returns
    /// The released messages, in merge-key order.
    pub fn release(&mut self, max: usize) -> Vec<Released> {
        let mut released = Vec::new();
        while released.len() < max {
            // The oldest buffered message, across partitions.
            let Some((partition, head)) = self
                .partitions
                .iter()
                .filter_map(|(partition, state)| {
                    state.buffer.front().map(|head| (partition, head.key()))
                })
                .min_by(|a, b| compare_keys(&a.1, &b.1))
            else {
                break;
            };
            // Every empty partition must be unable to deliver anything
            // older (partitions with a buffer cannot: their own head is
            // not older than this one).
            let releasable = self
                .partitions
                .values()
                .filter(|state| state.buffer.is_empty())
                .all(|state| match state.bound() {
                    Bound::Free => true,
                    Bound::UpTo(bound) => compare_keys(&head, &bound).is_le(),
                    Bound::Blocked => false,
                });
            if !releasable {
                break;
            }
            let partition = partition.clone();
            let message = self
                .partitions
                .get_mut(&partition)
                .and_then(|state| state.buffer.pop_front())
                .expect("the head was just found");
            released.push(Released {
                partition,
                offset: message.offset,
                record: message.record,
            });
        }
        released
    }

    /// Messages buffered, all partitions.
    pub fn n_buffered(&self) -> usize {
        self.partitions.values().map(|s| s.buffer.len()).sum()
    }

    /// Whether every assigned partition is consumed to its end with nothing
    /// buffered: `false` with no partition assigned (the backlog is
    /// unknown).
    pub fn is_caught_up(&self) -> bool {
        !self.partitions.is_empty() && self.partitions.values().all(PartitionState::at_end)
    }

    /// Which partitions to pause and which to resume so that no buffer
    /// grows past `capacity` messages (see the [module doc](self#memory)).
    ///
    /// # Arguments
    /// * `paused` – The partitions currently paused.
    /// * `capacity` – Most messages a partition buffers before it is paused;
    ///   it is resumed once its buffer is at most half of that.
    ///
    /// # Returns
    /// `(to_pause, to_resume)`; a paused partition no longer assigned is in
    /// neither (the consumer forgets it).
    pub fn flow_control(
        &self,
        paused: &BTreeSet<TopicPartition>,
        capacity: usize,
    ) -> (Vec<TopicPartition>, Vec<TopicPartition>) {
        let capacity = capacity.max(1);
        let mut to_pause = Vec::new();
        let mut to_resume = Vec::new();
        for (partition, state) in &self.partitions {
            let len = state.buffer.len();
            let is_paused = paused.contains(partition);
            if !is_paused && len >= capacity {
                to_pause.push(partition.clone());
            } else if is_paused && len <= capacity / 2 {
                to_resume.push(partition.clone());
            }
        }
        (to_pause, to_resume)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(partition: i32) -> TopicPartition {
        ("alerts".to_string(), partition)
    }

    /// An alert of id `id` arriving at `arrival`, at `offset`.
    fn alert(offset: i64, arrival: f64, id: i64) -> Fetched {
        Fetched {
            offset,
            arrival,
            record: Some(AlertRecord {
                dia_source_id: id,
                visit: 1,
                midpoint_mjd_tai: 0.0,
                ra_deg: 0.0,
                dec_deg: 0.0,
                ra_err_deg: 0.0,
                dec_err_deg: 0.0,
                psf_flux_njy: 1.0,
                psf_flux_err_njy: 1.0,
                band: "r".into(),
                arrival_mjd: arrival,
            }),
        }
    }

    fn skipped(offset: i64, arrival: f64) -> Fetched {
        Fetched {
            offset,
            arrival,
            record: None,
        }
    }

    /// The `(partition, offset)` of released messages.
    fn origins(released: &[Released]) -> Vec<(i32, i64)> {
        released.iter().map(|r| (r.partition.1, r.offset)).collect()
    }

    fn ids(released: &[Released]) -> Vec<i64> {
        released
            .iter()
            .filter_map(|r| r.record.as_ref().map(|record| record.dia_source_id))
            .collect()
    }

    /// The module doc's example: P9 races ahead, P1 lags; visit V spans
    /// both. Nothing of V is released before P1's share of it is fetched,
    /// and the merged stream is in time order.
    #[test]
    fn the_module_doc_example() {
        // Ids follow arrival times, so the expected order reads as ids.
        let mut merger = PartitionMerger::default();
        merger.assign(&[p(1), p(9)]);
        // P9: V at 100 and 160 s, then later visits at 200 and 300 s.
        for (offset, t, id) in [(0, 100.0, 1), (1, 160.0, 4), (2, 200.0, 5), (3, 300.0, 7)] {
            merger.push(&p(9), alert(offset, t, id));
        }
        // P1 has only fetched an alert of the previous visit.
        merger.push(&p(1), alert(0, 90.0, 0));
        assert_eq!(ids(&merger.release(usize::MAX)), [0]);

        // P1 fetches its V alerts (120, 150 s).
        merger.push(&p(1), alert(1, 120.0, 2));
        merger.push(&p(1), alert(2, 150.0, 3));
        assert_eq!(
            ids(&merger.release(usize::MAX)),
            [1, 2, 3],
            "P9's 160 s waits for P1 past 150 s"
        );

        // P1 moves past V: everything up to 260 s goes, in time order.
        merger.push(&p(1), alert(3, 260.0, 6));
        assert_eq!(ids(&merger.release(usize::MAX)), [4, 5, 6]);
        assert_eq!(merger.n_buffered(), 1, "P9's 300 s waits for P1");
    }

    #[test]
    fn a_partition_never_fetched_blocks_until_known_empty() {
        let mut merger = PartitionMerger::default();
        merger.assign(&[p(0), p(1)]);
        merger.push(&p(0), alert(0, 10.0, 1));
        assert!(merger.release(10).is_empty(), "P1 may hold older alerts");

        // The broker says P1 is empty (high == low): it holds nothing back.
        merger.set_offsets(&p(1), 0, 0);
        assert_eq!(ids(&merger.release(10)), [1]);
    }

    #[test]
    fn a_partition_at_its_committed_end_holds_nothing_back() {
        let mut merger = PartitionMerger::default();
        merger.assign(&[p(0), p(1)]);
        merger.push(&p(0), alert(0, 10.0, 1));
        // P1 resumes at its committed offset 500, which is its end.
        merger.set_start_offset(&p(1), 500);
        merger.set_offsets(&p(1), 0, 500);
        assert_eq!(ids(&merger.release(10)), [1]);
    }

    #[test]
    fn a_committed_offset_behind_the_end_is_a_backlog() {
        let mut merger = PartitionMerger::default();
        merger.assign(&[p(0), p(1)]);
        merger.push(&p(0), alert(0, 10.0, 1));
        merger.set_start_offset(&p(1), 400);
        merger.set_offsets(&p(1), 0, 500);
        assert!(merger.release(10).is_empty(), "P1 has 100 unread messages");
    }

    #[test]
    fn a_partition_consumed_to_its_end_holds_nothing_back() {
        let mut merger = PartitionMerger::default();
        merger.assign(&[p(0), p(1)]);
        merger.push(&p(1), alert(0, 5.0, 1));
        merger.push(&p(0), alert(0, 10.0, 2));
        merger.push(&p(0), alert(1, 20.0, 3));
        assert_eq!(ids(&merger.release(10)), [1], "P1 may still hold 5–20 s");

        // P1's high watermark is 1: fetched to its end.
        merger.set_offsets(&p(1), 0, 1);
        assert_eq!(ids(&merger.release(10)), [2, 3]);
    }

    #[test]
    fn fetching_past_the_known_end_makes_it_stale() {
        let mut merger = PartitionMerger::default();
        merger.assign(&[p(0), p(1)]);
        merger.push(&p(1), alert(0, 5.0, 1));
        merger.set_offsets(&p(1), 0, 1);
        // A new message arrived on P1 after the query.
        merger.push(&p(1), alert(1, 30.0, 4));
        merger.push(&p(0), alert(0, 10.0, 2));
        merger.push(&p(0), alert(1, 40.0, 3));
        assert_eq!(ids(&merger.release(10)), [1, 2, 4]);
        assert_eq!(
            merger.n_buffered(),
            1,
            "P1 is no longer known at its end: 40 s waits"
        );
        assert!(!merger.is_caught_up());
    }

    #[test]
    fn ties_are_broken_by_dia_source_id() {
        let mut merger = PartitionMerger::default();
        merger.assign(&[p(0), p(1)]);
        merger.push(&p(0), alert(0, 10.0, 3));
        merger.push(&p(1), alert(0, 10.0, 1));
        merger.push(&p(1), alert(1, 10.0, 4));
        merger.push(&p(0), alert(1, 10.0, 6));
        merger.set_offsets(&p(0), 0, 2);
        merger.set_offsets(&p(1), 0, 2);
        assert_eq!(ids(&merger.release(10)), [1, 3, 4, 6]);
    }

    #[test]
    fn release_stops_at_max_and_resumes_in_order() {
        let mut merger = PartitionMerger::default();
        merger.assign(&[p(0)]);
        for i in 0..5 {
            merger.push(&p(0), alert(i, i as f64, i));
        }
        merger.set_offsets(&p(0), 0, 5);
        assert_eq!(ids(&merger.release(2)), [0, 1]);
        assert_eq!(ids(&merger.release(10)), [2, 3, 4]);
        assert!(merger.is_caught_up());
    }

    #[test]
    fn a_single_partition_releases_up_to_its_last_fetch() {
        let mut merger = PartitionMerger::default();
        merger.assign(&[p(0)]);
        merger.push(&p(0), alert(0, 1.0, 1));
        merger.push(&p(0), alert(1, 2.0, 2));
        // Nothing else can be older: a single partition is its own order.
        assert_eq!(ids(&merger.release(10)), [1, 2]);
    }

    #[test]
    fn skipped_messages_are_released_in_offset_order() {
        let mut merger = PartitionMerger::default();
        merger.assign(&[p(0)]);
        merger.push(&p(0), alert(0, 1.0, 1));
        merger.push(&p(0), skipped(1, 2.0));
        merger.push(&p(0), alert(2, 3.0, 2));
        let released = merger.release(10);
        assert_eq!(origins(&released), [(0, 0), (0, 1), (0, 2)]);
        assert_eq!(ids(&released), [1, 2]);
    }

    #[test]
    fn a_slightly_older_alert_in_the_same_partition_is_released_late() {
        // Fink's partitions are ordered to within a fraction of a second.
        let mut merger = PartitionMerger::default();
        merger.assign(&[p(0), p(1)]);
        merger.push(&p(0), alert(0, 10.0, 1));
        merger.push(&p(0), alert(1, 9.95, 2));
        merger.push(&p(1), alert(0, 20.0, 3));
        merger.set_offsets(&p(0), 0, 2);
        merger.set_offsets(&p(1), 0, 1);
        // Released by buffer order within a partition: nothing is lost.
        assert_eq!(ids(&merger.release(10)), [1, 2, 3]);
    }

    #[test]
    fn revoked_partitions_drop_their_buffer_and_stop_holding() {
        let mut merger = PartitionMerger::default();
        merger.assign(&[p(0), p(1)]);
        merger.push(&p(0), alert(0, 10.0, 1));
        merger.push(&p(1), alert(0, 50.0, 2));
        merger.push(&p(1), alert(1, 60.0, 3));
        assert_eq!(ids(&merger.release(10)), [1], "P0 holds at 10 s");
        assert_eq!(merger.assign(&[p(1)]), 0, "P0's buffer was empty");
        // Only P1 is left; its own buffer is its order.
        assert_eq!(ids(&merger.release(10)), [2, 3]);
    }

    #[test]
    fn revoking_drops_the_buffered_messages() {
        let mut merger = PartitionMerger::default();
        merger.assign(&[p(0), p(1)]);
        merger.push(&p(1), alert(0, 50.0, 2));
        merger.push(&p(1), alert(1, 60.0, 3));
        assert!(merger.release(10).is_empty(), "P0 never fetched");
        assert_eq!(merger.assign(&[p(0)]), 2);
        assert_eq!(merger.n_buffered(), 0);
        assert!(!merger.is_assigned(&p(1)));
    }

    #[test]
    fn a_partition_assigned_back_ignores_what_it_already_fetched() {
        let mut merger = PartitionMerger::default();
        merger.assign(&[p(0)]);
        merger.push(&p(0), alert(0, 1.0, 1));
        merger.push(&p(0), alert(1, 2.0, 2));
        // Restarted at the committed offset 0: offsets 0 and 1 again.
        assert_eq!(merger.push(&p(0), alert(0, 1.0, 1)), Pushed::Refetched);
        assert_eq!(merger.push(&p(0), alert(1, 2.0, 2)), Pushed::Refetched);
        assert_eq!(merger.push(&p(0), alert(2, 3.0, 3)), Pushed::Buffered);
        assert_eq!(ids(&merger.release(10)), [1, 2, 3]);
    }

    #[test]
    fn a_start_offset_does_not_reject_the_first_message() {
        let mut merger = PartitionMerger::default();
        merger.assign(&[p(0)]);
        merger.set_start_offset(&p(0), 7);
        assert_eq!(merger.push(&p(0), alert(7, 1.0, 1)), Pushed::Buffered);
        // Ignored once the partition delivered.
        merger.set_start_offset(&p(0), 0);
        assert_eq!(merger.unfetched().count(), 0);
        assert_eq!(merger.push(&p(0), alert(8, 2.0, 2)), Pushed::Buffered);
    }

    #[test]
    fn offsets_of_unassigned_partitions_are_ignored() {
        let mut merger = PartitionMerger::default();
        merger.assign(&[p(0)]);
        merger.set_offsets(&p(5), 0, 0);
        merger.set_start_offset(&p(5), 3);
        assert_eq!(merger.partitions().count(), 1);
    }

    #[test]
    fn caught_up_needs_every_partition_at_its_end_and_empty_buffers() {
        let mut merger = PartitionMerger::default();
        assert!(!merger.is_caught_up(), "nothing assigned: backlog unknown");
        merger.assign(&[p(0), p(1)]);
        assert!(!merger.is_caught_up());
        merger.push(&p(0), alert(0, 1.0, 1));
        merger.set_offsets(&p(0), 0, 1);
        merger.set_offsets(&p(1), 0, 0);
        assert!(!merger.is_caught_up(), "P0's alert still buffered");
        merger.release(10);
        assert!(merger.is_caught_up());
    }

    #[test]
    fn a_full_buffer_is_paused_and_resumed_once_half_drained() {
        let mut merger = PartitionMerger::default();
        merger.assign(&[p(0), p(1)]);
        for i in 0..4 {
            merger.push(&p(1), alert(i, 100.0 + i as f64, i));
        }
        let mut paused = BTreeSet::new();
        let (to_pause, to_resume) = merger.flow_control(&paused, 4);
        assert_eq!((to_pause.clone(), to_resume), (vec![p(1)], vec![]));
        paused.extend(to_pause);

        // P0 fetches up to 101.5 s and lets P1 drain to 2 = capacity / 2.
        merger.push(&p(0), alert(0, 101.5, 10));
        assert_eq!(merger.release(10).len(), 3);
        let (to_pause, to_resume) = merger.flow_control(&paused, 4);
        assert_eq!((to_pause, to_resume), (vec![], vec![p(1)]));
    }

    #[test]
    fn the_partition_holding_the_watermark_is_never_paused() {
        // P0 is behind with an empty buffer: it must keep fetching.
        let mut merger = PartitionMerger::default();
        merger.assign(&[p(0), p(1)]);
        merger.push(&p(0), alert(0, 1.0, 0));
        for i in 0..10 {
            merger.push(&p(1), alert(i, 100.0 + i as f64, i + 1));
        }
        merger.release(usize::MAX);
        let (to_pause, _) = merger.flow_control(&BTreeSet::new(), 5);
        assert_eq!(to_pause, [p(1)]);
    }
}
