//! [`SqlOutbox`]: the durable queue of rows not yet in Postgres.
//!
//! # Why
//!
//! The tracker never waits for the database: observations and archived
//! trajectories bound for Postgres are first appended here, durably, and
//! inserted whenever Postgres is reachable. A database outage of hours (or
//! nights) only grows this file; nothing is lost, and the inserts catch up
//! at the next successful flush.
//!
//! # Format and lifecycle
//!
//! A [`framed_log`](fink_fat_engine::framed_log) of [`OutboxEntry`]
//! batches, mirrored in memory (so a flush never re-reads the file):
//!
//! ```text
//! append(entry) ×N ─► sync() ─► … ─► [flush succeeded] ─► clear()
//!   (buffered)        (fsync)                              (truncate)
//! ```
//!
//! A flush always inserts **everything** pending, idempotently (see
//! [`fink_fat_sql::incremental`]): if the process dies after the database
//! committed but before [`SqlOutbox::clear`], the next flush inserts the
//! same rows again and they are skipped. No cursor to keep consistent.

use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
};

use camino::Utf8Path;
use fink_fat_engine::{
    framed_log::{read_frames, write_frame},
    topocentric_kf::branching::ArchivedTrajectory,
};
use fink_fat_sql::rows::ObservationRow;

use crate::error::Result;

/// Filename of the outbox under the engine's `storage_path`.
pub const SQL_OUTBOX_FILENAME: &str = "sql_outbox.rkyvlog";

/// One batch of rows bound for Postgres.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum OutboxEntry {
    /// Observations of one source batch, for `observations`.
    Observations(Vec<ObservationRow>),
    /// Trajectories archived at one night close, for
    /// `archived_trajectories`.
    Archived(Vec<ArchivedTrajectory>),
}

/// The durable queue (see the [module doc](self)).
#[derive(Debug)]
pub struct SqlOutbox {
    /// The outbox file, in append mode.
    writer: BufWriter<File>,
    /// Every entry in the file, in append order.
    pending: Vec<OutboxEntry>,
}

impl SqlOutbox {
    /// Open (or create) the outbox at `path` and load what it holds — rows
    /// a previous run could not insert.
    ///
    /// A torn last frame (crash mid-write) is cut off, as for the night
    /// journal: those rows were never acknowledged to the source, which
    /// redelivers them.
    ///
    /// # Errors
    /// I/O errors, or a corrupt complete frame.
    pub fn open(path: &Utf8Path) -> Result<Self> {
        let read = read_frames::<OutboxEntry>(path)?;
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        if read.torn_tail {
            tracing::warn!(%path, "SQL outbox ends with a torn frame, truncated");
            file.set_len(read.valid_len)?;
            file.sync_all()?;
        }
        Ok(Self {
            writer: BufWriter::new(file),
            pending: read.frames,
        })
    }

    /// Queue `entry` (buffered: durable after [`Self::sync`]). An empty
    /// batch is ignored.
    ///
    /// # Errors
    /// Serialization or I/O error.
    pub fn append(&mut self, entry: OutboxEntry) -> Result<()> {
        let is_empty = match &entry {
            OutboxEntry::Observations(rows) => rows.is_empty(),
            OutboxEntry::Archived(trajectories) => trajectories.is_empty(),
        };
        if !is_empty {
            write_frame(&mut self.writer, &entry)?;
            self.pending.push(entry);
        }
        Ok(())
    }

    /// Flush and fsync: every queued entry is durable on return.
    ///
    /// # Errors
    /// I/O error.
    pub fn sync(&mut self) -> Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_data()?;
        Ok(())
    }

    /// Every pending entry, in append order.
    pub fn pending(&self) -> &[OutboxEntry] {
        &self.pending
    }

    /// Whether nothing is pending.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Empty the outbox, durably: everything pending is in Postgres.
    ///
    /// # Errors
    /// I/O error.
    pub fn clear(&mut self) -> Result<()> {
        self.writer.flush()?;
        let file = self.writer.get_ref();
        file.set_len(0)?;
        file.sync_all()?;
        self.pending.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use camino::Utf8PathBuf;
    use tempfile::TempDir;

    fn row(id: i64) -> ObservationRow {
        ObservationRow {
            id,
            night_id: 61_344,
            object_id: id.to_string(),
            magnitude: 20.0,
            mag_err: 0.1,
            filter: 2,
            mpc_code_obs: "X05".into(),
            ra: 1.0,
            ra_err: 1e-6,
            dec: -0.5,
            dec_err: 1e-6,
            mjd_tt: 61_345.1,
        }
    }

    /// Ids of the pending observation rows, in order.
    fn pending_ids(outbox: &SqlOutbox) -> Vec<i64> {
        outbox
            .pending()
            .iter()
            .flat_map(|entry| match entry {
                OutboxEntry::Observations(rows) => rows.iter().map(|r| r.id).collect(),
                OutboxEntry::Archived(_) => Vec::new(),
            })
            .collect()
    }

    fn outbox_path(dir: &TempDir) -> Utf8PathBuf {
        Utf8PathBuf::from_path_buf(dir.path().join(SQL_OUTBOX_FILENAME)).unwrap()
    }

    #[test]
    fn pending_entries_survive_a_restart_until_cleared() {
        let dir = TempDir::new().unwrap();
        let path = outbox_path(&dir);

        let mut outbox = SqlOutbox::open(&path).unwrap();
        outbox
            .append(OutboxEntry::Observations(vec![row(1), row(2)]))
            .unwrap();
        outbox.append(OutboxEntry::Observations(vec![])).unwrap();
        outbox.sync().unwrap();
        assert_eq!(outbox.pending().len(), 1, "empty batches are ignored");
        drop(outbox);

        let mut outbox = SqlOutbox::open(&path).unwrap();
        assert_eq!(pending_ids(&outbox), vec![1, 2]);
        outbox.clear().unwrap();
        assert!(outbox.is_empty());
        outbox
            .append(OutboxEntry::Observations(vec![row(3)]))
            .unwrap();
        outbox.sync().unwrap();
        drop(outbox);

        let outbox = SqlOutbox::open(&path).unwrap();
        assert_eq!(pending_ids(&outbox), vec![3]);
    }

    #[test]
    fn a_torn_frame_is_dropped() {
        let dir = TempDir::new().unwrap();
        let path = outbox_path(&dir);
        let mut outbox = SqlOutbox::open(&path).unwrap();
        outbox
            .append(OutboxEntry::Observations(vec![row(1)]))
            .unwrap();
        outbox
            .append(OutboxEntry::Observations(vec![row(2)]))
            .unwrap();
        outbox.sync().unwrap();
        drop(outbox);

        let len = std::fs::metadata(&path).unwrap().len();
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(len - 3)
            .unwrap();

        let outbox = SqlOutbox::open(&path).unwrap();
        assert_eq!(pending_ids(&outbox), vec![1]);
    }
}
