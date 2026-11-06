//! [`Driver`]: the streaming loop — source → ingestion → journal → visit
//! assembly → [`LiveNight`] → night close.
//!
//! The driver is the only stateful, I/O-performing part of the crate. Every
//! decision it takes is delegated to a pure building block — the
//! [`Ingestor`] (is this alert usable? already seen?), the
//! [`VisitAssembler`] (is this visit complete?), the [`NightClock`] (is the
//! night over?) — and every tracking computation to [`LiveNight`]. What the
//! driver itself owns is the **ordering** of those calls and the
//! **durability** of their effects.
//!
//! # State machine
//!
//! ```text
//!                  first convertible alert of a night not closed yet
//!   ┌────────────────────┐ ───────────────────────────────► ┌────────────────────┐
//!   │        Idle        │                                  │        Open        │
//!   │  BranchCollection  │                                  │  LiveNight         │
//!   │                    │                                  │  NightClock        │
//!   │                    │ ◄─────────────────────────────── │  VisitAssembler    │
//!   └────────────────────┘   close: newer-night alert,      └────────────────────┘
//!                            twilight, rollover, end of stream
//! ```
//!
//! * **Idle** — between two nights, the driver holds the
//!   [`BranchCollection`] returned by the last close (or loaded from the
//!   snapshot).
//! * **Open** — during a night, the collection's branches are owned by the
//!   [`LiveNight`]'s engine session; the driver holds that night, its clock
//!   and its visit assembler.
//!
//! Exactly one night is open at a time. A night opens on the first convertible
//! alert that belongs to it, and closes on one of the clock's triggers (see
//! [`crate::clock`]); a closed night never reopens (its alerts become *late*).
//!
//! # Life of one alert
//!
//! | Step | Done by | Outcome when it stops there |
//! |------|---------|-----------------------------|
//! | 1. cuts + conversion | [`Ingestor::convert`] | counted as cut / unconvertible |
//! | 2. find its night; close the open one if the alert is newer; open its night if needed | `make_night_current` (private) | counted as *late* if its night is closed |
//! | 3. deduplication within the night | [`Ingestor::accept`] | counted as duplicate |
//! | 4. queued as [`JournalEntry::Alert`] | driver | — |
//! | 5. buffered in its visit, or added as a leftover if that visit was already advanced | [`VisitAssembler::push`], [`LiveNight::add_leftovers`] | — |
//! | 6. advanced with its whole visit once the visit is idle | [`VisitAssembler::poll`], [`LiveNight::advance_visit`] | — |
//! | 7. part of the night's discovery pool at close | [`LiveNight::close`] | — |
//!
//! # One batch
//!
//! 1. Steps 1–5 above for every alert of the batch, in order.
//! 2. Every visit the assembler declares complete at the driver's clock is
//!    advanced, a [`JournalEntry::VisitEmitted`] marker being queued before
//!    each.
//! 3. The clock's close triggers are checked (twilight, rollover).
//! 4. The queued journal entries are written as one frame and **fsynced**.
//! 5. Only then is the batch **acknowledged** to the source.
//!
//! Because of 4-before-5, every acknowledged alert is on disk; anything
//! lost in a crash was not acknowledged, and the source redelivers it.
//!
//! # Clock
//!
//! The driver's clock (the private `now` method) is **event time**: the
//! latest [`arrival_mjd`](AlertRecord::arrival_mjd) seen. While the source
//! is caught up, a live source's wall clock may push it further (so that
//! twilight is reached after the night's last alert, when no new alert
//! advances event time any more); while the source catches up on a backlog,
//! only the backlog's own times count, so a night replayed hours late still
//! closes on its own timeline.
//!
//! # Closing a night, durably
//!
//! Remaining visits are drained and advanced, the engine closes the night
//! (pruning, stale purge, discovery), then:
//!
//! 1. the night's archived trajectories are appended to the archive log and
//!    fsynced (with a SQL sink, they are also queued in its outbox, with the
//!    batch's pending `observations` rows, and fsynced);
//! 2. the branch-collection snapshot is written (atomically, carrying the
//!    night as its last closed one);
//! 3. the journal is truncated;
//! 4. with a SQL sink, the new collection is published to Postgres — best
//!    effort, see [Postgres](self#postgres).
//!
//! | Crash … | On restart |
//! |---------|------------|
//! | before 1 | the snapshot still says the night is open: it is replayed from the journal and closed again |
//! | between 1 and 2 | same, so the night's archived batch is appended a **second time** (identical trajectories, same designations — deduplicable by designation). The reverse order would *lose* the batch instead. |
//! | between 2 and 3 | the journal describes a night the snapshot says is closed: it is discarded |
//! | during 4 | the snapshot and the outbox are on disk; the next start-up flushes the outbox and publishes the collection |
//!
//! # Postgres
//!
//! With a `database:` configuration, the driver also feeds the explorer's
//! database through a [`SqlSink`] (see [`crate::sink`]): accepted alerts
//! are queued for `observations` at every batch and flushed when Postgres
//! is reachable; at every night close the archived trajectories are queued
//! and the trajectory tables are swapped for the new collection. The
//! database is a *view* of the driver's local state: its failures are
//! logged and counted ([`DriverStats`]), never fatal, and caught up later.
//!
//! # Restart
//!
//! [`Driver::new`] loads the snapshot, reads the journal and, if it
//! describes a night not closed yet, reopens that night and replays it (see
//! [`crate::journal`]). The source then resumes from wherever it restarts:
//! redelivered alerts of the open night are duplicates, alerts of closed
//! nights are late, and nothing is counted twice.

use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    time::Instant,
};

use camino::Utf8PathBuf;
use fink_fat_engine::{
    engine_config::{EngineConfig, kalman_context::KalmanContext},
    topocentric_kf::branching::{BranchCollection, write_archived_batch},
};
use fink_fat_sql::rows::ObservationRow;
use photom::{NightId, observation_dataset::observation::ObservationInput};

use crate::{
    assembler::{EmittedVisit, Placement, VisitAssembler},
    clock::{AlertNight, CloseTrigger, NightClock},
    config::DriverConfig,
    error::Result,
    events::StreamEvent,
    ingest::{Ingestor, Rejection},
    journal::{JOURNAL_FILENAME, JournalEntry, NightJournal},
    night::LiveNight,
    record::AlertRecord,
    sink::{FlushOutcome, SQL_OUTBOX_FILENAME, SqlSink, SyncOutcome, observation_row},
    source::AlertSource,
};

/// Files the driver keeps under the engine's `storage_path`.
///
/// The snapshot and the archive log use the same names and formats as
/// `fink-fat track`, so the batch tools (`fink-fat convert`,
/// `tracking_analysis`) read a streaming run's state unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamPaths {
    /// Branch-collection snapshot, rewritten at every night close.
    pub snapshot: Utf8PathBuf,
    /// Append-only log of closed trajectories, one frame per night.
    pub archive: Utf8PathBuf,
    /// Journal of the open night (see [`crate::journal`]).
    pub journal: Utf8PathBuf,
    /// Rows waiting for Postgres (see [`crate::sink::outbox`]); only used
    /// with a `database:` configuration.
    pub sql_outbox: Utf8PathBuf,
}

impl StreamPaths {
    /// The driver's files under `engine_config`'s `storage_path`.
    ///
    /// # Arguments
    /// * `engine_config` – Supplies `storage_path` and the shared snapshot /
    ///   archive file names.
    pub fn under(engine_config: &EngineConfig) -> Self {
        Self {
            snapshot: engine_config.snapshot_path(),
            archive: engine_config.archive_log_path(),
            journal: engine_config.storage_path_buf().join(JOURNAL_FILENAME),
            sql_outbox: engine_config.storage_path_buf().join(SQL_OUTBOX_FILENAME),
        }
    }
}

/// Running counters of a driver (see [`Driver::stats`]).
///
/// Every received alert ends in exactly one of `n_accepted`, `n_cut`,
/// `n_unconvertible`, `n_duplicates` or `n_late`, so after a run without
/// restart `n_received` is their sum. After a restart, `n_accepted` also
/// counts the alerts replayed from the journal, which `n_received` does not.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DriverStats {
    /// Alerts received from the source (redeliveries included).
    pub n_received: usize,
    /// Alerts handed to the tracker (journal replays included).
    pub n_accepted: usize,
    /// Alerts failing a quality cut.
    pub n_cut: usize,
    /// Alerts that could not be converted.
    pub n_unconvertible: usize,
    /// Redelivered alerts, dropped.
    pub n_duplicates: usize,
    /// Alerts of a night already closed, dropped.
    pub n_late: usize,
    /// Accepted alerts of a visit already advanced: they skipped
    /// association and only feed the night's discovery. Included in
    /// `n_accepted`.
    pub n_leftovers: usize,
    /// Non-empty visits advanced (journal replays included).
    pub n_visits: usize,
    /// Of those, visits whose observer state did not resolve (skipped by the
    /// engine, every lineage carried over unchanged).
    pub n_unresolved_visits: usize,
    /// Nights closed.
    pub n_nights_closed: usize,
    /// Trajectories appended to the archive log.
    pub n_archived: usize,
    /// Rows inserted into `observations` (already-present rows excluded).
    pub n_observations_inserted: u64,
    /// Rows inserted into `archived_trajectories`.
    pub n_archived_inserted: u64,
    /// Failed flushes of the SQL outbox (each retried later).
    pub n_flush_failures: usize,
    /// Successful swaps of the trajectory tables.
    pub n_nights_synced: usize,
    /// Failed syncs of the trajectory tables (each retried at the next
    /// close or start-up).
    pub n_sync_failures: usize,
}

impl DriverStats {
    /// Increment the counter matching `rejection`.
    fn record_rejection(&mut self, rejection: &Rejection) {
        match rejection {
            Rejection::Cut(_) => self.n_cut += 1,
            Rejection::Unconvertible(_) => self.n_unconvertible += 1,
            Rejection::Duplicate => self.n_duplicates += 1,
        }
    }
}

/// Everything that exists only while a night is open.
struct OpenNight<'ctx, 'cfg> {
    /// The engine session and the night's own dataset.
    live: LiveNight<'ctx, 'cfg>,
    /// The night's close deadlines.
    clock: NightClock,
    /// Alerts accepted tonight whose visit has not been advanced yet.
    assembler: VisitAssembler<ObservationInput>,
}

/// Where the driver stands (see the [module doc](self#state-machine)).
///
/// The [`BranchCollection`] is moved, never cloned (it is the largest piece
/// of state): into the [`LiveNight`] when a night opens, and back out of it
/// when the night closes.
enum Phase<'ctx, 'cfg> {
    /// No open night; the collection the next night will start from.
    Idle(BranchCollection<'ctx, 'cfg>),
    /// A night is open (boxed: much larger than the other variants).
    Open(Box<OpenNight<'ctx, 'cfg>>),
    /// Placeholder left by `std::mem::replace` while a transition moves the
    /// collection from one variant to the other. Never observed between two
    /// method calls — unless the transition failed with an error, after which
    /// the driver must be dropped and restarted from storage.
    Poisoned,
}

/// The streaming loop (see the [module doc](self)).
///
/// `'ctx` / `'cfg` are the lifetimes of the engine's [`KalmanContext`] and
/// [`EngineConfig`], borrowed for the driver's whole life (every branch
/// borrows them); `S` is the alert source.
pub struct Driver<'ctx, 'cfg, S> {
    /// Where alerts come from.
    source: S,
    /// Ingestion, assembly and clock parameters.
    config: DriverConfig,
    /// Engine context, lent to every night.
    kalman_context: &'ctx KalmanContext,
    /// Engine configuration, lent to every night.
    engine_config: &'cfg EngineConfig,
    /// Snapshot / archive / journal locations.
    paths: StreamPaths,
    /// Journal of the open night.
    journal: NightJournal,
    /// Archive log, open in append mode for the driver's whole life.
    archive: BufWriter<File>,
    /// Cuts, conversion, and the set of `diaSourceId`s accepted tonight.
    ingestor: Ingestor,
    /// Journal entries of the current batch, in the order the driver
    /// performed them; written as one frame (and fsynced) before the batch
    /// is acknowledged, or dropped if the night closes first (the snapshot
    /// then supersedes them).
    pending: Vec<JournalEntry>,
    /// Postgres writer, `None` without a `database:` configuration.
    sink: Option<SqlSink>,
    /// `observations` rows of the current batch's accepted alerts, queued
    /// in the SQL outbox before the batch is journaled (see [`Self::step`]).
    /// Always empty without a sink.
    pending_rows: Vec<ObservationRow>,
    /// Idle or open (see [`Phase`]).
    phase: Phase<'ctx, 'cfg>,
    /// Latest `arrival_mjd` seen, MJD UTC (`-∞` before the first alert).
    event_time: f64,
    /// Running counters.
    stats: DriverStats,
}

impl<'ctx, 'cfg, S: AlertSource> Driver<'ctx, 'cfg, S> {
    /// Start a driver over `source`, resuming from whatever state lies under
    /// the engine's `storage_path`.
    ///
    /// # Steps
    /// 1. Create `storage_path` if missing.
    /// 2. Load the snapshot if there is one, else start from an empty
    ///    collection (a first run).
    /// 3. Open the archive log in append mode.
    /// 4. Open the journal (cutting a torn last frame) and read its entries.
    /// 5. With a `database:` configuration: open the SQL sink (and its
    ///    outbox), then publish the loaded collection if the database is
    ///    behind it — the only moment the idle collection is at hand before
    ///    a night opens (see [`crate::sink`]). A database failure is logged,
    ///    not fatal.
    /// 6. Replay those entries if they describe a night not closed yet (see
    ///    the private `recover`): the driver then
    ///    starts with that night open, exactly as the interrupted run left
    ///    it at its last fsync.
    ///
    /// The source is not touched: the first [`Self::step`] reads from it.
    ///
    /// # Arguments
    /// * `source` – Alert source, read from by [`Self::step`].
    /// * `config` – Ingestion, assembly and clock parameters.
    /// * `kalman_context`, `engine_config` – Engine context, borrowed for the
    ///   driver's whole life; `engine_config` also sets the storage location.
    ///
    /// # Errors
    /// The storage cannot be created, read or written, the snapshot is
    /// invalid, or replaying the journal fails (an engine error while
    /// re-advancing a visit).
    pub fn new(
        source: S,
        config: DriverConfig,
        kalman_context: &'ctx KalmanContext,
        engine_config: &'cfg EngineConfig,
    ) -> Result<Self> {
        std::fs::create_dir_all(engine_config.storage_path())?;
        let paths = StreamPaths::under(engine_config);
        let collection = if paths.snapshot.exists() {
            BranchCollection::load_snapshot_from_disk(
                &paths.snapshot,
                kalman_context,
                engine_config,
            )?
        } else {
            BranchCollection::empty()
        };
        let archive = BufWriter::new(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&paths.archive)?,
        );
        let (journal, entries) = NightJournal::open(&paths.journal)?;
        let ingestor = Ingestor::new(config.ingest.quality_cuts, config.ingest.conversion);
        let sink = config
            .database
            .as_ref()
            .map(|database| SqlSink::open(database, &paths.sql_outbox))
            .transpose()?;

        let mut driver = Self {
            source,
            config,
            kalman_context,
            engine_config,
            paths,
            journal,
            archive,
            ingestor,
            pending: Vec::new(),
            sink,
            pending_rows: Vec::new(),
            phase: Phase::Idle(collection),
            event_time: f64::NEG_INFINITY,
            stats: DriverStats::default(),
        };
        driver.sync_idle_collection()?;
        driver.recover(entries)?;
        Ok(driver)
    }

    /// Counters so far (see [`DriverStats`]).
    pub fn stats(&self) -> DriverStats {
        self.stats
    }

    /// The open night, or `None` between two nights.
    pub fn open_night_id(&self) -> Option<NightId> {
        match &self.phase {
            Phase::Open(night) => Some(night.live.night_id()),
            Phase::Idle(_) | Phase::Poisoned => None,
        }
    }

    /// Process one batch of the source — the five steps of the
    /// [module doc](self#one-batch).
    ///
    /// A batch may close a night anywhere inside it (an alert of a newer
    /// night arrives mid-batch, or a clock trigger fires at its end), and
    /// may open the next one; the journal frame written at the end then only
    /// holds the new night's entries.
    ///
    /// With a SQL sink, the batch's `observations` rows are queued and
    /// fsynced in the outbox **before** the journal: if the process dies in
    /// between, the batch is not acknowledged, the source redelivers it, and
    /// the alerts — absent from the journal, hence not duplicates — are
    /// accepted and queued again (an idempotent insert). The reverse order
    /// would lose them: journaled, hence duplicates on redelivery, but never
    /// queued. After the acknowledgement, the outbox is flushed if its
    /// backoff allows; a database failure never fails the step.
    ///
    /// # Returns
    /// `true` when a batch was processed (possibly empty), `false` once the
    /// source is exhausted — in which case nothing was done, and the caller
    /// should [`Self::finish`].
    ///
    /// # Errors
    /// Any source, engine, journal or archive failure. The in-memory state
    /// is then undefined; drop the driver and restart from storage, which
    /// is consistent as of the last successful `step`.
    pub fn step(&mut self) -> Result<bool> {
        let Some(batch) = self.source.next_batch()? else {
            return Ok(false);
        };
        for record in batch {
            self.handle_alert(record)?;
        }
        self.advance_ready_visits()?;
        self.check_clock()?;
        // Durability point: everything this batch did is on disk before the
        // source is told it may forget the batch. Outbox first (see above).
        self.queue_pending_rows()?;
        if let Some(sink) = &mut self.sink {
            sink.sync_outbox()?;
        }
        self.journal.append(std::mem::take(&mut self.pending))?;
        self.journal.sync()?;
        self.source.ack()?;
        self.flush_sink(false)?;
        Ok(true)
    }

    /// [`Self::step`] until the source is exhausted, then [`Self::finish`].
    ///
    /// A live source never ends, so this only returns for a replay (or on
    /// error).
    ///
    /// # Returns
    /// The final counters.
    ///
    /// # Errors
    /// As [`Self::step`] and [`Self::finish`].
    pub fn run(mut self) -> Result<DriverStats> {
        while self.step()? {}
        self.finish()
    }

    /// End of stream: close the open night, if any, with
    /// [`CloseTrigger::EndOfStream`] — its remaining visits are drained and
    /// it is persisted like any other night.
    ///
    /// Not calling it (dropping the driver) is equivalent to a crash: the
    /// open night stays in the journal and is resumed by the next
    /// [`Self::new`].
    ///
    /// With a SQL sink, a last flush is forced, so a replay ends with the
    /// database up to date whenever Postgres is reachable.
    ///
    /// # Returns
    /// The final counters.
    ///
    /// # Errors
    /// As a night close (see the [module doc](self#closing-a-night-durably)).
    pub fn finish(mut self) -> Result<DriverStats> {
        if matches!(self.phase, Phase::Open(_)) {
            self.close_night(CloseTrigger::EndOfStream)?;
        }
        self.flush_sink(true)?;
        Ok(self.stats)
    }

    /// Move the batch's `observations` rows into the SQL outbox (buffered;
    /// the caller fsyncs). A no-op without a sink.
    ///
    /// # Errors
    /// Local I/O error on the outbox.
    fn queue_pending_rows(&mut self) -> Result<()> {
        let rows = std::mem::take(&mut self.pending_rows);
        match &mut self.sink {
            Some(sink) => sink.queue_observations(rows),
            None => Ok(()),
        }
    }

    /// Flush the SQL outbox (see [`SqlSink::flush`]) and record the outcome
    /// in the stats and the log. A no-op without a sink.
    ///
    /// # Arguments
    /// * `force` – Ignore the backoff.
    ///
    /// # Errors
    /// Only a local I/O error on the outbox; Postgres failures are logged.
    fn flush_sink(&mut self, force: bool) -> Result<()> {
        let Some(sink) = &mut self.sink else {
            return Ok(());
        };
        match sink.flush(Instant::now(), force)? {
            FlushOutcome::Empty | FlushOutcome::Deferred => {}
            FlushOutcome::Flushed {
                n_observations,
                n_archived,
            } => {
                self.stats.n_observations_inserted += n_observations;
                self.stats.n_archived_inserted += n_archived;
                StreamEvent::SqlFlushed {
                    n_observations,
                    n_archived,
                }
                .emit();
            }
            FlushOutcome::Failed { failure, retry_in } => {
                self.stats.n_flush_failures += 1;
                StreamEvent::SqlFlushFailed {
                    failure: &failure,
                    retry_in,
                }
                .emit();
            }
        }
        Ok(())
    }

    /// Publish `collection` to the database if it is behind (see
    /// [`SqlSink::sync_night`]) and record the outcome. A no-op without a
    /// sink.
    ///
    /// The caller must first force a flush ([`Self::flush_sink`]`(true)`),
    /// which records its own outcome: the sync refuses to run while rows
    /// are pending.
    ///
    /// A free-standing helper over the fields it needs, so the caller can
    /// pass a collection it owns while the driver is borrowed.
    fn sync_collection(
        sink: Option<&mut SqlSink>,
        collection: &BranchCollection<'ctx, 'cfg>,
        stats: &mut DriverStats,
    ) {
        let Some(sink) = sink else {
            return;
        };
        let night_id = collection.last_closed_night_id;
        match sink.sync_night(collection) {
            SyncOutcome::UpToDate => {}
            SyncOutcome::Synced {
                n_branches,
                duration,
            } => {
                stats.n_nights_synced += 1;
                StreamEvent::NightSynced {
                    night_id,
                    n_branches,
                    duration,
                }
                .emit();
            }
            SyncOutcome::Failed(failure) => {
                stats.n_sync_failures += 1;
                StreamEvent::NightSyncFailed {
                    night_id,
                    failure: &failure,
                }
                .emit();
            }
        }
    }

    /// At start-up, publish the idle collection loaded from the snapshot if
    /// the database is behind (step 5 of [`Self::new`]).
    ///
    /// # Errors
    /// Only a local I/O error on the outbox.
    fn sync_idle_collection(&mut self) -> Result<()> {
        if self.sink.is_none() {
            return Ok(());
        }
        self.flush_sink(true)?;
        if let Phase::Idle(collection) = &self.phase {
            Self::sync_collection(self.sink.as_mut(), collection, &mut self.stats);
        }
        Ok(())
    }

    /// The driver's clock, MJD UTC (see the [module doc](self#clock)).
    ///
    /// `event_time`, raised to the source's wall clock only while the source
    /// is caught up — a backlog must not be closed by the wall clock before
    /// its own alerts are processed.
    fn now(&self) -> f64 {
        let wall = self
            .source
            .caught_up()
            .then(|| self.source.wall_clock_mjd())
            .flatten();
        wall.map_or(self.event_time, |wall| wall.max(self.event_time))
    }

    /// Take one alert through steps 1–5 of the
    /// [module doc](self#life-of-one-alert).
    ///
    /// The order matters:
    ///
    /// * conversion comes first because it yields the alert's night;
    /// * the night is made current **before** deduplication, because the
    ///   dedup set is per night — an alert starting a newer night must be
    ///   accepted into the *new* night's set, after the old night closed and
    ///   reset it;
    /// * the journal entry is queued before the alert reaches the
    ///   assembler, so that in the journal an alert always precedes the
    ///   [`JournalEntry::VisitEmitted`] marker of its visit.
    ///
    /// Updates `event_time` for every received alert, even one dropped
    /// afterwards: its arrival is still a fact about the stream's time.
    ///
    /// # Errors
    /// Closing or opening a night failed, or the night's dataset refused
    /// the observation (see [`LiveNight::add_leftovers`]).
    fn handle_alert(&mut self, record: AlertRecord) -> Result<()> {
        self.stats.n_received += 1;
        self.event_time = self.event_time.max(record.arrival_mjd);

        // 1. Cuts + conversion.
        let observation = match self.ingestor.convert(&record) {
            Ok(observation) => observation,
            Err(rejection) => {
                self.reject(&record, rejection);
                return Ok(());
            }
        };
        // 2. Its night must be the open one (possibly closing the previous).
        if !self.make_night_current(observation.night_id)? {
            self.stats.n_late += 1;
            StreamEvent::LateAlert {
                dia_source_id: record.dia_source_id,
                night_id: observation.night_id,
            }
            .emit();
            return Ok(());
        }
        // 3. Deduplication within that night.
        if let Err(rejection) = self.ingestor.accept(record.dia_source_id) {
            self.reject(&record, rejection);
            return Ok(());
        }

        // 4. Journal (and SQL outbox), then 5. assembler.
        self.stats.n_accepted += 1;
        if self.sink.is_some() {
            self.pending_rows.push(observation_row(&observation));
        }
        let arrival = record.arrival_mjd;
        self.pending.push(JournalEntry::Alert(record));
        self.place(observation.visit, arrival, observation.input)
    }

    /// Ensure `night_id` is the open night, if it still can be.
    ///
    /// | Phase | `night_id` vs open / last closed night | Action | Returns |
    /// |-------|---------------------------------------|--------|---------|
    /// | Open | same night | none | `true` |
    /// | Open | earlier | none (the alert is late) | `false` |
    /// | Open | later | close the open night ([`CloseTrigger::NewerNight`]), then open `night_id` | `true` |
    /// | Idle | at or before the last closed night | none (late) | `false` |
    /// | Idle | after it | open `night_id` | `true` |
    ///
    /// Note that nights are not required to be consecutive: a gap (bad
    /// weather, downtime) simply opens the next night that has alerts.
    ///
    /// # Errors
    /// Closing or opening a night failed.
    fn make_night_current(&mut self, night_id: NightId) -> Result<bool> {
        if let Phase::Open(night) = &self.phase {
            match night.clock.classify(night_id) {
                AlertNight::Current => return Ok(true),
                AlertNight::Older => return Ok(false),
                // Falls through to the Idle branch below once closed.
                AlertNight::Newer => self.close_night(CloseTrigger::NewerNight)?,
            }
        }
        if let Phase::Idle(collection) = &self.phase
            && collection.is_night_closed(night_id)
        {
            return Ok(false);
        }
        self.open_night(night_id)?;
        Ok(true)
    }

    /// Count `rejection` in the stats and log it at debug level.
    fn reject(&mut self, record: &AlertRecord, rejection: Rejection) {
        self.stats.record_rejection(&rejection);
        StreamEvent::AlertRejected {
            dia_source_id: record.dia_source_id,
            rejection: &rejection,
        }
        .emit();
    }

    /// Hand an accepted observation to the open night (step 5 of the
    /// [module doc](self#life-of-one-alert)).
    ///
    /// The assembler buffers it with its visit, unless that visit was
    /// already advanced: the observation then arrived too late for the
    /// visit's association and is added to the night's dataset as a
    /// *leftover* right away, so that the night's discovery still sees it.
    ///
    /// # Arguments
    /// * `visit` – Rubin `visit` of the observation.
    /// * `arrival` – Its arrival time, MJD UTC (drives the visit's idle
    ///   timeout).
    /// * `input` – The converted observation.
    ///
    /// # Errors
    /// The night's dataset refused the leftover (duplicate id).
    ///
    /// # Panics
    /// When no night is open — callers only place alerts they just routed
    /// to the open night.
    fn place(&mut self, visit: i64, arrival: f64, input: ObservationInput) -> Result<()> {
        let Phase::Open(night) = &mut self.phase else {
            unreachable!("an accepted alert always has an open night");
        };
        if let Placement::Leftover(input) = night.assembler.push(visit, arrival, input) {
            night.live.add_leftovers(vec![input])?;
            self.stats.n_leftovers += 1;
        }
        Ok(())
    }

    /// Advance every visit the assembler declares complete at the driver's
    /// clock, in `visit` order (step 2 of [`Self::step`]).
    ///
    /// Each visit's [`JournalEntry::VisitEmitted`] marker is queued
    /// *before* the visit is advanced, after the visit's own alerts: a
    /// replay of the journal then re-emits the visit at exactly this point
    /// of the alert sequence, with exactly the same alerts.
    ///
    /// A no-op when no night is open.
    ///
    /// # Errors
    /// The engine failed to advance a visit.
    fn advance_ready_visits(&mut self) -> Result<()> {
        let now = self.now();
        let Phase::Open(night) = &mut self.phase else {
            return Ok(());
        };
        for visit in night.assembler.poll(now) {
            self.pending
                .push(JournalEntry::VisitEmitted { visit: visit.visit });
            advance(&mut night.live, visit, &mut self.stats)?;
        }
        Ok(())
    }

    /// Close the open night if the clock says so (step 3 of
    /// [`Self::step`]): twilight passed with the source caught up and no
    /// visit pending, or the rollover safety net (see
    /// [`NightClock::close_trigger`]). The newer-night trigger is handled
    /// per alert, in `handle_alert`.
    ///
    /// Called after `advance_ready_visits`,
    /// so "no visit pending" reflects the visits that just became complete.
    /// A no-op when no night is open.
    ///
    /// # Errors
    /// As a night close.
    fn check_clock(&mut self) -> Result<()> {
        let Phase::Open(night) = &self.phase else {
            return Ok(());
        };
        let trigger = night.clock.close_trigger(
            self.now(),
            self.source.caught_up(),
            !night.assembler.is_empty(),
        );
        match trigger {
            Some(trigger) => self.close_night(trigger),
            None => Ok(()),
        }
    }

    /// Transition Idle → Open: move the idle collection into a new
    /// [`LiveNight`] for `night_id`, with a fresh clock and assembler.
    ///
    /// # Errors
    /// The engine refused the night (it is not after the collection's last
    /// closed night). The collection is put back and the driver stays
    /// Idle, but callers check [`BranchCollection::is_night_closed`] first,
    /// so this signals an inconsistency.
    ///
    /// # Panics
    /// When a night is already open.
    fn open_night(&mut self, night_id: NightId) -> Result<()> {
        let Phase::Idle(collection) = std::mem::replace(&mut self.phase, Phase::Poisoned) else {
            unreachable!("a night is only opened from the idle phase");
        };
        let live = match LiveNight::begin(
            collection,
            night_id,
            self.config.ingest.error_model,
            self.kalman_context,
            self.engine_config,
        ) {
            Ok(live) => live,
            Err(refused) => {
                self.phase = Phase::Idle(*refused.collection);
                return Err(refused.error.into());
            }
        };
        self.phase = Phase::Open(Box::new(OpenNight {
            live,
            clock: NightClock::open(night_id, &self.config.clock),
            assembler: VisitAssembler::new(self.config.assembler.visit_idle_timeout),
        }));
        StreamEvent::NightOpened { night_id }.emit();
        Ok(())
    }

    /// Transition Open → Idle: finish, close and persist the open night.
    ///
    /// # Steps
    /// 1. Drain the assembler and advance every visit still buffered, in
    ///    `visit` order (they are complete: the night is over).
    /// 2. Close the night in the engine ([`LiveNight::close`]): N-scan
    ///    pruning, stale purge, discovery over all the night's observations.
    ///    The night's dataset is dropped.
    /// 3. Persist, in the order of the
    ///    [module doc](self#closing-a-night-durably): archive (fsynced),
    ///    SQL outbox (the batch's pending `observations` rows and the
    ///    night's archived trajectories, fsynced — before the snapshot, so a
    ///    crash replays the close and queues them again, idempotently),
    ///    snapshot, journal truncation. Queued journal entries of this night
    ///    are dropped — the snapshot supersedes them.
    /// 4. Reset the per-night dedup set and go Idle with the new collection.
    /// 5. Publish the new collection to the database (flush, then swap; see
    ///    [`SqlSink::sync_night`]). Best effort: a failure is logged and the
    ///    next close or start-up catches up.
    ///
    /// Drained visits get no [`JournalEntry::VisitEmitted`] marker: if the
    /// process dies before the snapshot, the replay leaves them in the
    /// assembler and the next close drains them in the same order.
    ///
    /// # Arguments
    /// * `trigger` – Why the night closes (logged only).
    ///
    /// # Errors
    /// Engine, archive, snapshot or journal failure; the driver is then left
    /// `Phase::Poisoned` and must be dropped.
    ///
    /// # Panics
    /// When no night is open.
    fn close_night(&mut self, trigger: CloseTrigger) -> Result<()> {
        let Phase::Open(night) = std::mem::replace(&mut self.phase, Phase::Poisoned) else {
            unreachable!("only an open night is closed");
        };
        let OpenNight {
            mut live,
            mut assembler,
            ..
        } = *night;

        // 1. Remaining visits.
        for visit in assembler.drain() {
            advance(&mut live, visit, &mut self.stats)?;
        }
        let (night_id, n_visits, n_unresolved_visits, n_observations, n_leftovers) = (
            live.night_id(),
            live.n_visits(),
            live.n_unresolved_visits(),
            live.n_observations(),
            live.n_leftovers(),
        );

        // 2. Engine close.
        let mut collection = live.close()?;

        // 3. Persistence: archive, SQL outbox, snapshot, journal — in this
        // order.
        let n_archived = collection.archived.len();
        write_archived_batch(&mut self.archive, &collection.archived)?;
        self.archive.flush()?;
        self.archive.get_ref().sync_data()?;
        // `archived` is this night's batch only; once on disk (and queued
        // for Postgres), no need to carry it in RAM, nor in the snapshot.
        let archived = std::mem::take(&mut collection.archived);
        self.queue_pending_rows()?;
        if let Some(sink) = &mut self.sink {
            sink.queue_archived(archived)?;
            sink.sync_outbox()?;
        }
        collection.save_snapshot_to_disk(&self.paths.snapshot)?;
        self.pending.clear();
        self.journal.truncate()?;

        // 4. Back to Idle.
        self.ingestor.reset();
        self.stats.n_nights_closed += 1;
        self.stats.n_archived += n_archived;
        StreamEvent::NightClosed {
            night_id,
            trigger,
            n_visits,
            n_unresolved_visits,
            n_observations,
            n_leftovers,
            n_branches: collection.branches.len(),
            n_archived,
        }
        .emit();

        // 5. Database (best effort): forced flush (counted), then sync.
        self.flush_sink(true)?;
        Self::sync_collection(self.sink.as_mut(), &collection, &mut self.stats);
        self.phase = Phase::Idle(collection);
        Ok(())
    }

    /// Rebuild the open night from the journal's `entries`, at start-up.
    ///
    /// # Cases
    /// * **No alert entry** (empty journal, or only markers — impossible in
    ///   practice): nothing to resume; a non-empty journal is truncated.
    /// * **The journal's night is already closed** in the snapshot (crash
    ///   between the snapshot and the journal truncation): the journal is
    ///   stale and truncated.
    /// * **Otherwise**: the night is opened and the entries are replayed in
    ///   order — each [`JournalEntry::Alert`] re-ingested and placed as it
    ///   was live (see `replay_alert`), each
    ///   [`JournalEntry::VisitEmitted`] forcing that visit out of the
    ///   assembler ([`VisitAssembler::take`]) and advancing it.
    ///
    /// Markers make the replay independent of time: a visit is re-emitted
    /// where the interrupted run emitted it, not when the restarted clock
    /// would decide, so the rebuilt night is the one the interrupted run had
    /// at its last fsync. Visits that were still being assembled stay in the
    /// assembler, with their alerts' original arrival times.
    ///
    /// Replayed entries are not journaled again (they are already there).
    ///
    /// # Errors
    /// An alert entry with an invalid `visit` (a corrupt journal), opening
    /// the night, or re-advancing a visit failed.
    fn recover(&mut self, entries: Vec<JournalEntry>) -> Result<()> {
        // A journal holds one night: the night of its first alert.
        let night_id = entries.iter().find_map(|entry| match entry {
            JournalEntry::Alert(record) => Some(record.night_id()),
            JournalEntry::VisitEmitted { .. } => None,
        });
        let Some(night_id) = night_id.transpose()? else {
            if !entries.is_empty() {
                self.journal.truncate()?;
            }
            return Ok(());
        };
        if let Phase::Idle(collection) = &self.phase
            && collection.is_night_closed(night_id)
        {
            StreamEvent::JournalDiscarded { night_id }.emit();
            return self.journal.truncate();
        }

        self.open_night(night_id)?;
        let n_entries = entries.len();
        for entry in entries {
            match entry {
                JournalEntry::Alert(record) => self.replay_alert(&record, night_id)?,
                JournalEntry::VisitEmitted { visit } => {
                    let Phase::Open(night) = &mut self.phase else {
                        unreachable!("opened above");
                    };
                    let visit = night.assembler.take(visit);
                    advance(&mut night.live, visit, &mut self.stats)?;
                }
            }
        }
        StreamEvent::Recovered {
            night_id,
            n_entries,
        }
        .emit();
        Ok(())
    }

    /// Replay one journaled alert of `night_id`: the live path's steps 1, 3
    /// and 5 (no night routing — the night is already open — and no new
    /// journal entry).
    ///
    /// The alert passed ingestion when it was journaled, but it is
    /// re-ingested under the *current* configuration: if the cuts were
    /// tightened between the crash and the restart, it is now rejected and
    /// counted as such. Re-ingestion also refills the dedup set, so the
    /// redelivered copies that follow are recognised as duplicates.
    ///
    /// # Errors
    /// The night's dataset refused the observation (see `place`).
    fn replay_alert(&mut self, record: &AlertRecord, night_id: NightId) -> Result<()> {
        self.event_time = self.event_time.max(record.arrival_mjd);
        let observation = match self.ingestor.ingest(record) {
            Ok(observation) if observation.night_id == night_id => observation,
            // Cannot happen with a journal this driver wrote (it is
            // truncated at every close), but never let it reach the night.
            Ok(_) => {
                tracing::warn!(
                    dia_source_id = record.dia_source_id,
                    "journaled alert of another night, skipped"
                );
                return Ok(());
            }
            Err(rejection) => {
                self.reject(record, rejection);
                return Ok(());
            }
        };
        self.stats.n_accepted += 1;
        self.place(observation.visit, record.arrival_mjd, observation.input)
    }
}

/// Fold one emitted visit into `live` (step 6 of the
/// [module doc](self#life-of-one-alert)), count it and log its outcome.
///
/// A free function rather than a method so it can borrow the open night and
/// the stats at the same time (both are fields of the driver). An empty
/// visit — [`VisitAssembler::take`] of a visit with nothing buffered — is
/// skipped and not counted.
///
/// # Errors
/// The night's dataset refused an observation (see
/// [`LiveNight::advance_visit`]).
fn advance(
    live: &mut LiveNight<'_, '_>,
    visit: EmittedVisit<ObservationInput>,
    stats: &mut DriverStats,
) -> Result<()> {
    if visit.items.is_empty() {
        return Ok(());
    }
    let n_observations = visit.items.len();
    let outcome = live.advance_visit(visit.items)?;
    stats.n_visits += 1;
    if !outcome.observer_resolved {
        stats.n_unresolved_visits += 1;
    }
    StreamEvent::VisitAdvanced {
        visit: visit.visit,
        n_observations,
        n_lineages_gated: outcome.n_lineages_gated,
        n_branches: outcome.n_branches,
        observer_resolved: outcome.observer_resolved,
    }
    .emit();
    Ok(())
}
