//! [`SqlSink`]: writes the driver's results to the explorer's Postgres
//! database, without ever holding the tracker back.
//!
//! # What goes where
//!
//! | Data | When | How |
//! |------|------|-----|
//! | accepted alerts → `observations` | every source batch | queued in the [`SqlOutbox`], [flushed](SqlSink::flush) idempotently when Postgres is reachable |
//! | archived trajectories → `archived_trajectories` | every night close | same queue |
//! | live branches → the 6 trajectory tables | every night close | [`SqlSink::sync_night`]: atomic swap ([`fink_fat_sql::night_sync`]) |
//!
//! # Never blocking the tracker
//!
//! The tracker's own state (snapshot, archive log, night journal) is its
//! source of truth; the database is a view of it for the explorer. So:
//!
//! * nothing is written to Postgres synchronously with tracking — rows go
//!   to the durable [`SqlOutbox`] first;
//! * a Postgres failure is **reported, never propagated**: the flush and
//!   sync outcomes ([`FlushOutcome`], [`SyncOutcome`]) carry the error for
//!   the driver to log and count; only local I/O errors (the outbox file)
//!   are errors;
//! * after a failure, flushes are spaced by an exponential [`Backoff`], and
//!   the connection is re-established from scratch.
//!
//! # Catching up
//!
//! * **Observations and archives**: everything queued stays in the outbox
//!   until a flush succeeds — through restarts, for as long as the outage
//!   lasts.
//! * **Trajectories**: the swap replaces the whole tables, so only the
//!   latest state matters. A failed sync is not retried during the night
//!   (the branches are inside the open night by then); the next close, or
//!   the next start-up (the driver syncs its idle collection before resuming
//!   a night), syncs the newer state. [`fink_fat_sql::night_sync::last_synced_night`]
//!   tells whether a sync is needed — and is empty after a Postgres crash
//!   (UNLOGGED tables), which forces one.
//!
//! # Ordering constraint
//!
//! `branch_observations.obs_id` references `observations(id)`: the caller
//! forces a [flush](SqlSink::flush) right before each
//! [sync](SqlSink::sync_night), which refuses to run while rows are still
//! pending — so every observation a branch references is in the database.

pub mod backoff;
pub mod outbox;
pub mod rows;

use std::time::{Duration, Instant};

use camino::Utf8Path;
use fink_fat_engine::topocentric_kf::branching::{ArchivedTrajectory, BranchCollection};
use fink_fat_sql::{
    SqlError,
    build::{TrajectoryRows, build_archived_rows, referenced_obs_ids},
    incremental, night_sync,
    rows::ObservationRow,
    schema,
};
use postgres::{Client, NoTls};
use thiserror::Error;

pub use backoff::Backoff;
pub use outbox::{OutboxEntry, SQL_OUTBOX_FILENAME, SqlOutbox};
pub use rows::observation_row;

use crate::{config::DatabaseConfig, error::Result};

/// Why a flush or a sync did not go through. Reported, not propagated (see
/// the [module doc](self#never-blocking-the-tracker)).
#[derive(Debug, Error)]
pub enum SinkFailure {
    /// Connecting, or a statement, failed.
    #[error(transparent)]
    Sql(#[from] SqlError),
    /// Some observations the branches reference are not in `observations`
    /// (e.g. the table was emptied by hand): the swap's foreign key would
    /// fail. Needs a `convert`, or the missing rows, to recover.
    #[error("{missing} observations referenced by the branches are missing from `observations`")]
    MissingObservations {
        /// How many referenced ids are missing.
        missing: usize,
    },
    /// The outbox still holds rows (its last flush failed): the swap's
    /// foreign key needs them in the database first.
    #[error("the SQL outbox is not flushed; the sync waits for the rows to reach Postgres")]
    PendingRows,
}

impl From<postgres::Error> for SinkFailure {
    fn from(e: postgres::Error) -> Self {
        Self::Sql(SqlError::Postgres(e))
    }
}

/// What [`SqlSink::flush`] did.
#[derive(Debug)]
pub enum FlushOutcome {
    /// Nothing was pending.
    Empty,
    /// Something is pending, but the backoff forbids an attempt yet.
    Deferred,
    /// Everything pending was inserted (rows already present skipped).
    Flushed {
        /// Rows inserted into `observations`.
        n_observations: u64,
        /// Rows inserted into `archived_trajectories`.
        n_archived: u64,
    },
    /// The attempt failed; the rows stay queued.
    Failed {
        /// Why.
        failure: SinkFailure,
        /// No new attempt before this long.
        retry_in: Duration,
    },
}

/// What [`SqlSink::sync_night`] did.
#[derive(Debug)]
pub enum SyncOutcome {
    /// The database already holds this night's (or a later) state, or no
    /// night was ever closed.
    UpToDate,
    /// The trajectory tables now hold the collection.
    Synced {
        /// Rows loaded into `branches`.
        n_branches: usize,
        /// Wall-clock time of the sync (staging build + swap).
        duration: Duration,
    },
    /// The flush or the sync failed; retried at the next close or start-up.
    Failed(SinkFailure),
}

/// The Postgres writer of the driver (see the [module doc](self)).
pub struct SqlSink {
    /// Connection string, never logged.
    url: String,
    /// Connection timeout, swap policy, backoff bounds.
    config: DatabaseConfig,
    /// Rows waiting for Postgres.
    outbox: SqlOutbox,
    /// Live connection, `None` until the first attempt or after a failure.
    client: Option<Client>,
    /// Spacing of flush attempts after failures.
    backoff: Backoff,
}

impl SqlSink {
    /// Sink writing to the database of `config`, queueing in the outbox at
    /// `outbox_path` (opened — and its pending rows loaded — now). No
    /// connection is made before the first flush.
    ///
    /// # Errors
    /// No database URL configured (see [`DatabaseConfig::resolve_url`]), or
    /// the outbox cannot be opened.
    pub fn open(config: &DatabaseConfig, outbox_path: &Utf8Path) -> Result<Self> {
        let (min, max) = config.flush_backoff();
        Ok(Self {
            url: config.resolve_url()?,
            config: config.clone(),
            outbox: SqlOutbox::open(outbox_path)?,
            client: None,
            backoff: Backoff::new(min, max),
        })
    }

    /// Queue `rows` for `observations` (buffered, see [`Self::sync_outbox`]).
    ///
    /// # Errors
    /// Local I/O error on the outbox.
    pub fn queue_observations(&mut self, rows: Vec<ObservationRow>) -> Result<()> {
        self.outbox.append(OutboxEntry::Observations(rows))
    }

    /// Queue `archived` for `archived_trajectories` (buffered).
    ///
    /// # Errors
    /// Local I/O error on the outbox.
    pub fn queue_archived(&mut self, archived: Vec<ArchivedTrajectory>) -> Result<()> {
        self.outbox.append(OutboxEntry::Archived(archived))
    }

    /// Make everything queued durable (fsync of the outbox).
    ///
    /// # Errors
    /// Local I/O error on the outbox.
    pub fn sync_outbox(&mut self) -> Result<()> {
        self.outbox.sync()
    }

    /// Whether rows are waiting for Postgres.
    pub fn has_pending(&self) -> bool {
        !self.outbox.is_empty()
    }

    /// Try to insert everything queued, in one transaction.
    ///
    /// # Steps
    /// 1. Nothing pending → [`FlushOutcome::Empty`]; backoff not elapsed
    ///    (and not `force`d) → [`FlushOutcome::Deferred`].
    /// 2. Connect if needed (creating the tables on a fresh connection,
    ///    idempotently), insert every pending entry in append order
    ///    ([`incremental`]), commit.
    /// 3. On success, empty the outbox and reset the backoff; on failure,
    ///    drop the connection and back off.
    ///
    /// # Arguments
    /// * `now` – Current instant, for the backoff.
    /// * `force` – Ignore the backoff (a night close needs the flush before
    ///   its sync).
    ///
    /// # Errors
    /// Only a local I/O error emptying the outbox after a successful
    /// insert (harmless to retry: the inserts are idempotent). Postgres
    /// failures are in [`FlushOutcome::Failed`].
    pub fn flush(&mut self, now: Instant, force: bool) -> Result<FlushOutcome> {
        if self.outbox.is_empty() {
            return Ok(FlushOutcome::Empty);
        }
        if !force && !self.backoff.ready(now) {
            return Ok(FlushOutcome::Deferred);
        }
        let inserted = connect_if_needed(&mut self.client, &self.url, &self.config)
            .and_then(|client| insert_pending(client, self.outbox.pending()));
        match inserted {
            Ok((n_observations, n_archived)) => {
                self.outbox.clear()?;
                self.backoff.succeeded();
                Ok(FlushOutcome::Flushed {
                    n_observations,
                    n_archived,
                })
            }
            Err(failure) => {
                self.client = None;
                let retry_in = self.backoff.failed(now);
                Ok(FlushOutcome::Failed { failure, retry_in })
            }
        }
    }

    /// Bring the database's trajectory tables to `collection`'s state, if
    /// they are behind.
    ///
    /// **Precondition**: the outbox is flushed — the foreign key of
    /// `branch_observations` needs every observation in the database. The
    /// caller runs a forced [`Self::flush`] first (and records its
    /// outcome); this method only checks it succeeded.
    ///
    /// # Steps
    /// 1. No night closed in `collection` → [`SyncOutcome::UpToDate`].
    /// 2. Rows still pending → [`SyncOutcome::Failed`] with
    ///    [`SinkFailure::PendingRows`].
    /// 3. If [`night_sync::last_synced_night`] is already at or after the
    ///    collection's last closed night → [`SyncOutcome::UpToDate`].
    /// 4. Read the `(night_id, mjd_tt)` of every observation the branches
    ///    reference ([`night_sync::fetch_obs_index`]); fail if some are
    ///    missing.
    /// 5. Build the rows ([`TrajectoryRows::build`]) and swap them in
    ///    ([`night_sync::sync_trajectories`]).
    ///
    /// Blocks for the duration of the sync (minutes at LSST scale): called
    /// at a night close, when no alert is expected.
    ///
    /// # Arguments
    /// * `collection` – The state to publish, with its last closed night.
    ///
    /// # Returns
    /// The outcome; Postgres failures are in [`SyncOutcome::Failed`], never
    /// an error.
    pub fn sync_night(&mut self, collection: &BranchCollection<'_, '_>) -> SyncOutcome {
        let Some(night_id) = collection.last_closed_night_id else {
            return SyncOutcome::UpToDate;
        };
        if self.has_pending() {
            return SyncOutcome::Failed(SinkFailure::PendingRows);
        }
        let started = Instant::now();
        let synced =
            connect_if_needed(&mut self.client, &self.url, &self.config).and_then(|client| {
                sync_collection(
                    client,
                    collection,
                    i64::from(night_id.0),
                    self.config.swap_retry_policy(),
                )
            });
        match synced {
            Ok(Some(n_branches)) => SyncOutcome::Synced {
                n_branches,
                duration: started.elapsed(),
            },
            Ok(None) => SyncOutcome::UpToDate,
            Err(failure) => {
                self.client = None;
                SyncOutcome::Failed(failure)
            }
        }
    }
}

/// The open connection in `client`, opening one (and creating the tables)
/// if there is none.
///
/// A new connection runs [`schema::create_tables`] once: idempotent, and it
/// makes a fresh (or wiped) database usable without a prior `convert`.
fn connect_if_needed<'c>(
    client: &'c mut Option<Client>,
    url: &str,
    config: &DatabaseConfig,
) -> std::result::Result<&'c mut Client, SinkFailure> {
    if client.is_none() {
        let mut pg_config: postgres::Config = url.parse()?;
        pg_config.connect_timeout(config.connect_timeout());
        let mut fresh = pg_config.connect(NoTls)?;
        let mut transaction = fresh.transaction()?;
        schema::create_tables(&mut transaction)?;
        transaction.commit()?;
        *client = Some(fresh);
    }
    Ok(client.as_mut().expect("connected above"))
}

/// Insert every entry of `pending`, in order, in one transaction.
///
/// # Returns
/// Rows inserted into `observations` and into `archived_trajectories`.
fn insert_pending(
    client: &mut Client,
    pending: &[OutboxEntry],
) -> std::result::Result<(u64, u64), SinkFailure> {
    let mut transaction = client.transaction()?;
    let (mut n_observations, mut n_archived) = (0, 0);
    for entry in pending {
        match entry {
            OutboxEntry::Observations(rows) => {
                n_observations += incremental::insert_observations(&mut transaction, rows)?;
            }
            OutboxEntry::Archived(trajectories) => {
                let rows = build_archived_rows(trajectories);
                n_archived += incremental::insert_archived(&mut transaction, &rows)?;
            }
        }
    }
    transaction.commit()?;
    Ok((n_observations, n_archived))
}

/// Steps 3–5 of [`SqlSink::sync_night`].
///
/// # Returns
/// `Some(n_branches)` after a swap, `None` when the database was already up
/// to date.
fn sync_collection(
    client: &mut Client,
    collection: &BranchCollection<'_, '_>,
    night_id: i64,
    policy: night_sync::SwapRetryPolicy,
) -> std::result::Result<Option<usize>, SinkFailure> {
    if night_sync::last_synced_night(client)?.is_some_and(|last| last >= night_id) {
        return Ok(None);
    }
    let ids = referenced_obs_ids(collection);
    let obs_index = night_sync::fetch_obs_index(client, &ids)?;
    let missing = ids.len() - obs_index.len();
    if missing > 0 {
        return Err(SinkFailure::MissingObservations { missing });
    }
    let rows = TrajectoryRows::build(collection, &obs_index);
    night_sync::sync_trajectories(client, &rows, night_id, policy)?;
    Ok(Some(rows.branches.len()))
}
