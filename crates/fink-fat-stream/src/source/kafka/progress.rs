//! Consumer bookkeeping of the Kafka source — pure, no broker involved:
//! which offsets to commit ([`OffsetTracker`]), where a partition resumes
//! ([`start_offset`]), and whether it is consumed to its end
//! ([`partition_caught_up`]).

use std::collections::BTreeMap;

/// A topic partition.
pub type TopicPartition = (String, i32);

/// The offsets the next acknowledgement commits: for each partition, the
/// offset following the last message **delivered** to the driver.
///
/// Kafka commits mean "resume here", hence `offset + 1`. Every delivered
/// message is recorded, skipped (undecodable) ones included, so a poison
/// message is not redelivered forever. The driver acknowledges only after
/// journaling the batch, so committing what was delivered so far never
/// loses an alert.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct OffsetTracker {
    /// Next offset to resume from, per partition.
    next: BTreeMap<TopicPartition, i64>,
    /// Whether a message was recorded since the last successful commit.
    dirty: bool,
}

impl OffsetTracker {
    /// Record the delivery of the message at `offset` of `topic`/`partition`.
    ///
    /// Offsets only move forward: a redelivered older message (after a
    /// rebalance) does not move the commit point back.
    pub fn record(&mut self, topic: &str, partition: i32, offset: i64) {
        let next = self.next.entry((topic.to_string(), partition)).or_insert(0);
        if offset + 1 > *next {
            *next = offset + 1;
            self.dirty = true;
        }
    }

    /// Whether there is something new to commit.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Forget the partitions for which `keep` is false — those no longer
    /// assigned to this consumer, whose offsets are another member's to
    /// commit now.
    pub fn retain(&mut self, mut keep: impl FnMut(&TopicPartition) -> bool) {
        self.next.retain(|partition, _| keep(partition));
    }

    /// The offsets to commit, per partition.
    pub fn offsets(&self) -> impl Iterator<Item = (&TopicPartition, i64)> {
        self.next.iter().map(|(partition, next)| (partition, *next))
    }

    /// The offsets were committed: nothing new until the next record.
    pub fn mark_committed(&mut self) {
        self.dirty = false;
    }
}

/// Whether a partition is consumed to its end.
///
/// # Arguments
/// * `position` – The consumer's next offset to fetch in the partition,
///   `None` before its first fetch.
/// * `low`, `high` – The partition's low and high watermarks (oldest
///   retained offset, offset of the next message to be written).
///
/// # Returns
/// `position >= high`; before the first fetch, only an empty partition
/// (`high <= low`) is consumed to its end.
pub fn partition_caught_up(position: Option<i64>, low: i64, high: i64) -> bool {
    match position {
        Some(position) => position >= high,
        None => high <= low,
    }
}

/// Where consumption of a partition resumes, before its first fetch.
///
/// # Arguments
/// * `committed` – The group's committed offset for it, if any.
/// * `low`, `high` – The partition's low and high watermark offsets.
/// * `reset_to_latest` – Whether `auto.offset.reset` is `latest` (the
///   source's default is `earliest`).
///
/// # Returns
/// The committed offset, clamped into `[low, high]` (an offset older than
/// the retention restarts at `low`); without one, `low` — or `high` when
/// the group starts at the latest message.
pub fn start_offset(committed: Option<i64>, low: i64, high: i64, reset_to_latest: bool) -> i64 {
    match committed {
        Some(offset) => offset.clamp(low, high.max(low)),
        None if reset_to_latest => high,
        None => low,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_commit_point_follows_the_last_delivered_offset() {
        let mut tracker = OffsetTracker::default();
        assert!(!tracker.is_dirty());
        tracker.record("a", 0, 5);
        tracker.record("a", 0, 7);
        tracker.record("a", 1, 0);
        assert!(tracker.is_dirty());
        let offsets: Vec<_> = tracker
            .offsets()
            .map(|((t, p), o)| (t.clone(), *p, o))
            .collect();
        assert_eq!(offsets, [("a".into(), 0, 8), ("a".into(), 1, 1)]);

        tracker.mark_committed();
        tracker.record("a", 0, 3);
        assert!(
            !tracker.is_dirty(),
            "an older offset does not move the commit point"
        );
        assert_eq!(tracker.offsets().next().unwrap().1, 8);
    }

    #[test]
    fn revoked_partitions_are_forgotten() {
        let mut tracker = OffsetTracker::default();
        tracker.record("a", 0, 1);
        tracker.record("a", 1, 1);
        tracker.retain(|(_, partition)| *partition == 1);
        let partitions: Vec<i32> = tracker.offsets().map(|((_, p), _)| *p).collect();
        assert_eq!(partitions, [1]);
    }

    #[test]
    fn a_partition_is_caught_up_at_its_high_watermark() {
        assert!(partition_caught_up(Some(10), 0, 10));
        assert!(!partition_caught_up(Some(9), 0, 10));
        assert!(
            !partition_caught_up(None, 0, 10),
            "not fetched yet, backlog"
        );
        assert!(partition_caught_up(None, 4, 4), "empty partition");
    }

    #[test]
    fn a_partition_resumes_at_its_committed_offset_or_per_the_reset_policy() {
        assert_eq!(start_offset(Some(7), 0, 10, false), 7);
        assert_eq!(start_offset(Some(7), 0, 10, true), 7, "a commit wins");
        assert_eq!(start_offset(Some(2), 5, 10, false), 5, "past retention");
        assert_eq!(start_offset(Some(12), 0, 10, false), 10);
        assert_eq!(start_offset(None, 3, 10, false), 3, "earliest");
        assert_eq!(start_offset(None, 3, 10, true), 10, "latest");
    }
}
