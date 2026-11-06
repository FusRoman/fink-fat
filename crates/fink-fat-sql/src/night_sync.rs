//! Nightly replacement of the 6 trajectory tables by an atomic schema swap,
//! for the streaming pipeline.
//!
//! # Why a swap
//!
//! `convert` replaces the trajectory tables with `TRUNCATE` + `COPY` in one
//! transaction: correct, but the `TRUNCATE` holds `ACCESS EXCLUSIVE` for the
//! whole load (minutes at LSST scale), during which every explorer page
//! touching these tables hangs. The streaming pipeline replaces them every
//! night while the explorer is in use, so it builds the next generation
//! aside and swaps it in with a few catalogue updates instead.
//!
//! # Steps ([`sync_trajectories`])
//!
//! 1. **Build** (one long transaction, invisible to the explorer): the
//!    staging schema [`STAGING_SCHEMA`] is created (emptied of an
//!    interrupted sync's leftovers, as is [`RETIRED_SCHEMA`]), and the 6
//!    tables are created and loaded
//!    there by the very scripts and loaders of `convert`, run with
//!    `search_path = fink_fat_staging, public` (see
//!    [`crate::schema`'s module doc](crate::schema#schema-independent-scripts)
//!    and [`crate::load_trajectory_tables`]).
//! 2. **Swap** (one short transaction, retried on lock timeout): the 6
//!    `public` tables move to [`RETIRED_SCHEMA`] and the 6 staging tables to
//!    `public`; [`SYNC_STATE_TABLE`] records the night.
//! 3. **Cleanup**: the retired tables are dropped.
//!
//! The explorer resolves `branches` by name, so after step 2 it reads the
//! new tables, and its homepage change check (the filenode behind the name
//! `branches`) sees a new relation and rebuilds its snapshot. A caller that
//! inserts the night's archived trajectories ([`crate::incremental`]) right
//! before the sync has them counted by that rebuild.
//!
//! # Preconditions
//!
//! * The tables exist ([`crate::schema::create_tables`]).
//! * `observations` already holds every observation the branches reference:
//!   `branch_observations.obs_id`'s foreign key is validated in step 1.
//!
//! # Failure
//!
//! Each step is its own transaction. A failure in step 1 leaves `public`
//! untouched (the staging leftovers are cleaned by the next sync); a
//! failure in step 2 too (rolled back); a failure in step 3 leaves retired
//! tables behind, dropped by the next sync's step 1.

use std::{thread, time::Duration};

use indicatif::ProgressBar;
use postgres::{Client, GenericClient, error::SqlState};

use crate::{
    SqlError,
    build::{ObsIndex, TrajectoryRows},
    load::load_trajectory_tables,
    schema,
};

/// Schema the next generation of the trajectory tables is built in.
pub const STAGING_SCHEMA: &str = "fink_fat_staging";

/// Schema the previous generation is moved to by the swap, then dropped.
pub const RETIRED_SCHEMA: &str = "fink_fat_retired";

/// Single-row table holding the last night swapped in (see
/// `sql/tables/engine/sync_state.sql`).
pub const SYNC_STATE_TABLE: &str = "fink_fat_sync_state";

/// How the swap transaction copes with the explorer's locks.
///
/// The swap needs `ACCESS EXCLUSIVE` on the 6 public tables, so it waits for
/// every running query on them (a homepage snapshot build, a bulk job…).
/// Rather than queueing — which would block every explorer query arriving
/// after it — it gives up after the `lock_timeout` set in
/// `sql/night_sync/swap_trajectory_tables.sql` (5 s), and is retried.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwapRetryPolicy {
    /// Retries after the first attempt.
    pub retries: u32,
    /// Pause between two attempts.
    pub delay: Duration,
}

impl Default for SwapRetryPolicy {
    /// 12 retries, 10 s apart: about 3 minutes of tries in all.
    fn default() -> Self {
        Self {
            retries: 12,
            delay: Duration::from_secs(10),
        }
    }
}

/// `(night_id, mjd_tt)` of the given observations, read from `observations`
/// — the [`ObsIndex`] [`TrajectoryRows::build`] needs to compute each
/// branch's statistics when the observations themselves are not in memory
/// (the streaming pipeline keeps only the open night's).
///
/// # Arguments
///
/// * `client` — a connection (or transaction).
/// * `ids` — observation ids to look up, typically
///   [`crate::build::referenced_obs_ids`]. Ids absent from the table are
///   simply missing from the result.
///
/// # Errors
///
/// [`SqlError::Postgres`] on query failure.
pub fn fetch_obs_index(client: &mut impl GenericClient, ids: &[i64]) -> Result<ObsIndex, SqlError> {
    let rows = client.query(
        "SELECT id, night_id, mjd_tt FROM observations WHERE id = ANY($1)",
        &[&ids],
    )?;
    Ok(rows
        .iter()
        .map(|row| (row.get(0), (row.get(1), row.get(2))))
        .collect())
}

/// The last night swapped in, `None` if no sync ever completed — or if
/// Postgres crashed since: [`SYNC_STATE_TABLE`] is UNLOGGED like the
/// trajectory tables, so both are emptied together and the caller knows it
/// must sync again.
///
/// # Arguments
///
/// * `client` — a connection (or transaction).
///
/// # Errors
///
/// [`SqlError::Postgres`] on query failure (e.g. the table does not exist:
/// run [`crate::schema::create_tables`] first).
pub fn last_synced_night(client: &mut impl GenericClient) -> Result<Option<i64>, SqlError> {
    Ok(client
        .query_opt(
            &format!("SELECT last_synced_night FROM {SYNC_STATE_TABLE}"),
            &[],
        )?
        .map(|row| row.get(0)))
}

/// Replace the public trajectory tables by `rows` (see the
/// [module doc](self)).
///
/// # Arguments
///
/// * `client` — a connection, not in a transaction (each step opens its
///   own).
/// * `rows` — the new content of the 6 trajectory tables.
/// * `night_id` — night recorded in [`SYNC_STATE_TABLE`].
/// * `policy` — retries of the swap on lock timeout.
///
/// # Errors
///
/// [`SqlError::Postgres`] when a step fails, or when the swap still times
/// out on locks after `policy.retries` retries. `public` is unchanged
/// unless the swap committed.
pub fn sync_trajectories(
    client: &mut Client,
    rows: &TrajectoryRows,
    night_id: i64,
    policy: SwapRetryPolicy,
) -> Result<(), SqlError> {
    build_staging(client, rows)?;
    let mut attempt = 0;
    loop {
        match swap(client, night_id) {
            Ok(()) => break,
            Err(SqlError::Postgres(e))
                if e.code() == Some(&SqlState::LOCK_NOT_AVAILABLE) && attempt < policy.retries =>
            {
                attempt += 1;
                tracing::warn!(
                    attempt,
                    retries = policy.retries,
                    "trajectory swap timed out on the explorer's locks, retrying"
                );
                thread::sleep(policy.delay);
            }
            Err(e) => return Err(e),
        }
    }
    let mut transaction = client.transaction()?;
    transaction.batch_execute(schema::DROP_RETIRED_TRAJECTORY_TABLES_SQL)?;
    transaction.commit()?;
    Ok(())
}

/// Step 1: (re)create the staging schema and load `rows` into it.
///
/// Leftovers of an interrupted sync are removed first, in both private
/// schemas: the staging tables (rebuilt here) and retired tables a failed
/// cleanup left behind — the swap could not move the current tables there
/// otherwise.
fn build_staging(client: &mut Client, rows: &TrajectoryRows) -> Result<(), SqlError> {
    let mut transaction = client.transaction()?;
    transaction.batch_execute(schema::DROP_RETIRED_TRAJECTORY_TABLES_SQL)?;
    transaction.batch_execute(schema::PREPARE_STAGING_SCHEMA_SQL)?;
    // From here on, unqualified names resolve to the staging schema first:
    // new tables are created there, and every later statement finds the
    // staging table (or index) of a name before the public one — the
    // staging copies all exist once the creation script ran, so the
    // `DROP … IF EXISTS` of `load_trajectory_tables` never reaches `public`.
    // Only `observations`, which has no staging copy, resolves to `public`
    // (referenced by `branch_observations`).
    transaction.batch_execute(&format!(
        "SET LOCAL search_path = {STAGING_SCHEMA}, public;"
    ))?;
    transaction.batch_execute(schema::CREATE_TRAJECTORY_TABLES_SQL)?;
    load_trajectory_tables(&mut transaction, rows, &ProgressBar::hidden())?;
    transaction.commit()?;
    Ok(())
}

/// Step 2: swap the tables and record `night_id` — one transaction.
fn swap(client: &mut Client, night_id: i64) -> Result<(), SqlError> {
    let mut transaction = client.transaction()?;
    transaction.batch_execute(schema::SWAP_TRAJECTORY_TABLES_SQL)?;
    transaction.execute(
        &format!(
            "INSERT INTO {SYNC_STATE_TABLE} (last_synced_night, synced_at) VALUES ($1, now()) \
             ON CONFLICT (singleton) DO UPDATE \
             SET last_synced_night = EXCLUDED.last_synced_night, synced_at = EXCLUDED.synced_at"
        ),
        &[&night_id],
    )?;
    transaction.commit()?;
    Ok(())
}
