//! [`NightJournal`]: the append-only log of the open night, from which a
//! restarted driver rebuilds it.
//!
//! # Why
//!
//! The branch-collection snapshot is only written when a night closes. Until
//! then, everything the driver did during the night — alerts accepted,
//! visits folded into the branches — lives in memory only, and a crash at
//! 4 a.m. would lose half a night of tracking. Re-reading the source from
//! the start of the night is not an option for a live stream (Kafka offsets
//! are committed as batches are acknowledged), so the driver keeps its own
//! record of the night.
//!
//! # Content
//!
//! The journal holds the open night as a sequence of [`JournalEntry`]s, in
//! the exact order the driver performed them:
//!
//! * [`JournalEntry::Alert`] for every alert accepted into the night (raw
//!   record, arrival time included);
//! * [`JournalEntry::VisitEmitted`] each time a visit is handed to the
//!   tracker — written *after* that visit's alerts, *before* any alert that
//!   arrived after the visit was emitted.
//!
//! For instance:
//!
//! ```text
//!   Alert(v1,a)  Alert(v1,b)  Alert(v2,c)  VisitEmitted(v1)  Alert(v1,d)  Alert(v2,e)
//! ```
//!
//! reads: `a`, `b`, `c` arrived; visit 1 was advanced with `{a, b}`; `d`
//! arrived too late for it (a leftover); `e` joined the still-open visit 2.
//!
//! # Replay
//!
//! Replaying the entries in order rebuilds the night exactly (see the
//! driver's [restart](crate::driver#restart)): each alert is re-ingested and
//! pushed into a fresh [`VisitAssembler`](crate::assembler::VisitAssembler);
//! each marker forces its visit out of the assembler
//! ([`take`](crate::assembler::VisitAssembler::take)) and advances it. The
//! markers are what makes the replay **independent of time**: the visit is
//! re-emitted where the interrupted run emitted it, with the same alerts,
//! not whenever the restarted clock would have. Alerts arriving after a
//! marker for their visit become leftovers again, as they were live.
//!
//! Only accepted alerts are journaled — rejected, duplicate or late ones
//! would be rejected again — and only the open night: the journal is
//! emptied when the night closes.
//!
//! # Durability
//!
//! Entries are written as [`framed_log`](fink_fat_engine::framed_log)
//! frames, one per call to [`NightJournal::append`] (one per source batch).
//! The driver [`syncs`](NightJournal::sync) the journal **before
//! acknowledging** the batch to the source: an acknowledged alert is always
//! on disk, and whatever a crash loses was never acknowledged, so the
//! source redelivers it.
//!
//! A crash during a write leaves at most a torn last frame (an unacknowledged
//! batch); [`NightJournal::open`] cuts it off before anything is appended
//! after it, otherwise its stale length prefix would swallow the next
//! frames.
//!
//! Once the night is closed and its snapshot written, the journal is
//! [`truncated`](NightJournal::truncate).

use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
};

use camino::Utf8Path;
use fink_fat_engine::framed_log::{read_frames, write_frame};

use crate::{error::Result, record::AlertRecord};

/// Filename of the journal under the engine's `storage_path`.
pub const JOURNAL_FILENAME: &str = "night_journal.rkyvlog";

/// One step of the open night, as recorded in the journal (see the
/// [module doc](self#content)).
#[derive(Debug, Clone, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum JournalEntry {
    /// An alert accepted into the night: it passed the cuts and conversion,
    /// belongs to the open night and was not a duplicate. Stored raw, so the
    /// replay re-ingests it under the current configuration.
    Alert(AlertRecord),
    /// The visit was handed to the tracker at this point of the alert
    /// sequence, with every alert of it journaled before this entry (and
    /// not already handed back as a leftover).
    VisitEmitted {
        /// Rubin `visit` identifier.
        visit: i64,
    },
}

/// Writer of the open night's journal.
///
/// Reading only happens once, at [`Self::open`]; afterwards the journal is
/// write-only: [`Self::append`] (buffered), [`Self::sync`] (durable),
/// [`Self::truncate`] (night closed).
#[derive(Debug)]
pub struct NightJournal {
    /// The journal file, opened in append mode: every write lands at its
    /// current end, including after a truncation.
    writer: BufWriter<File>,
}

impl NightJournal {
    /// Open (or create) the journal at `path` and read back its entries.
    ///
    /// # Steps
    /// 1. Read every complete frame ([`read_frames`]); a missing file is an
    ///    empty journal.
    /// 2. Open the file in append mode (creating it if needed).
    /// 3. If the file ends with a torn frame (the process died mid-write),
    ///    truncate it to the end of the last complete frame and fsync, so
    ///    the next frames append right after valid data.
    ///
    /// # Arguments
    /// * `path` – Journal file, usually
    ///   [`StreamPaths::journal`](crate::driver::StreamPaths::journal).
    ///
    /// # Returns
    /// The journal, ready to append, and its complete entries in write
    /// order (empty for a new journal or one truncated at the last close).
    ///
    /// # Errors
    /// I/O errors, or a complete frame that fails `rkyv` validation (a
    /// corrupt file, as opposed to a torn tail).
    pub fn open(path: &Utf8Path) -> Result<(Self, Vec<JournalEntry>)> {
        let read = read_frames::<Vec<JournalEntry>>(path)?;
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        if read.torn_tail {
            tracing::warn!(
                %path,
                valid_len = read.valid_len,
                "night journal ends with a torn frame, truncated"
            );
            file.set_len(read.valid_len)?;
            file.sync_all()?;
        }
        let entries = read.frames.into_iter().flatten().collect();
        Ok((
            Self {
                writer: BufWriter::new(file),
            },
            entries,
        ))
    }

    /// Append `entries` as one frame.
    ///
    /// Buffered: nothing is guaranteed on disk before [`Self::sync`]. Taking
    /// the entries by value lets the driver hand over its per-batch queue
    /// (`std::mem::take`) without a copy. A no-op when `entries` is empty, so
    /// a batch that accepted nothing does not grow the file.
    ///
    /// # Arguments
    /// * `entries` – The batch's entries, in the order they happened.
    ///
    /// # Errors
    /// Serialization or I/O error.
    pub fn append(&mut self, entries: Vec<JournalEntry>) -> Result<()> {
        if !entries.is_empty() {
            write_frame(&mut self.writer, &entries)?;
        }
        Ok(())
    }

    /// Flush the buffer and fsync the file: every entry appended so far is
    /// durable on return. The driver calls it once per batch, right before
    /// acknowledging the batch to the source.
    ///
    /// `sync_data` (not `sync_all`): only the content and length matter, not
    /// the file's timestamps.
    ///
    /// # Errors
    /// I/O error.
    pub fn sync(&mut self) -> Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_data()?;
        Ok(())
    }

    /// Empty the journal durably, because the night it described is closed
    /// and persisted in the snapshot.
    ///
    /// Buffered entries are flushed first (then truncated with the rest),
    /// so none lands *after* the truncation. Because the file is in append
    /// mode, the next [`Self::append`] writes at offset 0.
    ///
    /// # Errors
    /// I/O error.
    pub fn truncate(&mut self) -> Result<()> {
        self.writer.flush()?;
        let file = self.writer.get_ref();
        file.set_len(0)?;
        file.sync_all()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::tests::sample_record;
    use camino::Utf8PathBuf;
    use tempfile::TempDir;

    fn journal_path(dir: &TempDir) -> Utf8PathBuf {
        Utf8PathBuf::from_path_buf(dir.path().join(JOURNAL_FILENAME)).unwrap()
    }

    fn entries() -> Vec<JournalEntry> {
        vec![
            JournalEntry::Alert(sample_record()),
            JournalEntry::VisitEmitted {
                visit: sample_record().visit,
            },
        ]
    }

    #[test]
    fn entries_round_trip_across_reopenings() {
        let dir = TempDir::new().unwrap();
        let path = journal_path(&dir);

        let (mut journal, replayed) = NightJournal::open(&path).unwrap();
        assert!(replayed.is_empty());
        journal.append(entries()).unwrap();
        journal.append(Vec::new()).unwrap();
        journal.sync().unwrap();
        drop(journal);

        let (mut journal, replayed) = NightJournal::open(&path).unwrap();
        assert_eq!(replayed, entries());
        journal.append(entries()[..1].to_vec()).unwrap();
        journal.sync().unwrap();
        drop(journal);

        let (_, replayed) = NightJournal::open(&path).unwrap();
        assert_eq!(replayed.len(), 3);
        assert_eq!(replayed[2], entries()[0]);
    }

    #[test]
    fn a_torn_frame_is_cut_before_appending() {
        let dir = TempDir::new().unwrap();
        let path = journal_path(&dir);
        let (mut journal, _) = NightJournal::open(&path).unwrap();
        journal.append(entries()).unwrap();
        journal.append(entries()).unwrap();
        journal.sync().unwrap();
        drop(journal);

        // Simulate a crash in the middle of the second frame.
        let len = std::fs::metadata(&path).unwrap().len();
        let file = OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(len - 5).unwrap();
        drop(file);

        let (mut journal, replayed) = NightJournal::open(&path).unwrap();
        assert_eq!(replayed, entries(), "only the complete frame survives");
        journal.append(entries()).unwrap();
        journal.sync().unwrap();
        drop(journal);

        let (_, replayed) = NightJournal::open(&path).unwrap();
        assert_eq!(replayed.len(), 4, "appends land after the cut");
    }

    #[test]
    fn truncate_empties_the_journal() {
        let dir = TempDir::new().unwrap();
        let path = journal_path(&dir);
        let (mut journal, _) = NightJournal::open(&path).unwrap();
        journal.append(entries()).unwrap();
        journal.truncate().unwrap();
        journal.append(entries()[1..].to_vec()).unwrap();
        journal.sync().unwrap();
        drop(journal);

        let (_, replayed) = NightJournal::open(&path).unwrap();
        assert_eq!(replayed, entries()[1..].to_vec());
    }
}
