//! Idempotent incremental inserts into the two append-only tables the
//! streaming pipeline writes between full reloads: `observations` and
//! `archived_trajectories`.
//!
//! # Why idempotent
//!
//! The streaming pipeline delivers its rows *at least once*: a batch whose
//! insert succeeded but whose acknowledgement was lost (crash, network cut
//! after the commit) is inserted again on retry. Both functions therefore
//! skip rows whose key is already present (`ON CONFLICT … DO NOTHING`)
//! instead of failing the whole batch.
//!
//! # How
//!
//! `INSERT … ON CONFLICT` cannot read from a `COPY` stream, and a row-by-row
//! `INSERT` would cost one round trip per row. Each function therefore:
//!
//! 1. creates a temporary table shaped like the target (`LIKE`), dropped at
//!    the end of the call;
//! 2. bulk-loads the rows into it with the same binary `COPY` writer as the
//!    full reload ([`crate::copy`]);
//! 3. moves them into the target with one `INSERT … SELECT … ON CONFLICT DO
//!    NOTHING`.

use indicatif::ProgressBar;
use postgres::Transaction;

use crate::{
    SqlError, copy,
    rows::{ArchivedRow, ObservationRow},
};

/// Temporary table [`insert_observations`] stages its rows in.
const OBSERVATIONS_IN: &str = "fink_fat_observations_in";

/// Temporary table [`insert_archived`] stages its rows in.
const ARCHIVED_IN: &str = "fink_fat_archived_in";

/// Inserts `rows` into `observations`, skipping any whose `id` is already
/// there (see the [module doc](self)).
///
/// Duplicates *within* `rows` are skipped too (the first one wins).
///
/// # Arguments
///
/// * `transaction` — the transaction to insert in; nothing is visible to
///   other sessions before it commits.
/// * `rows` — observations to insert.
///
/// # Returns
///
/// The number of rows actually inserted (excluding skipped duplicates).
///
/// # Errors
///
/// [`SqlError::Postgres`] on any statement failure.
pub fn insert_observations(
    transaction: &mut Transaction<'_>,
    rows: &[ObservationRow],
) -> Result<u64, SqlError> {
    insert_via_temp_table(
        transaction,
        "observations",
        OBSERVATIONS_IN,
        "id",
        |transaction, table| {
            copy::copy_observations(transaction, table, rows, &ProgressBar::hidden())
        },
    )
}

/// Inserts `rows` into `archived_trajectories`, skipping any whose
/// `designation` is already there (see the [module doc](self)).
///
/// Each row's dynamical family is classified on the way, as in the full
/// reload ([`copy::copy_archived_trajectories`]).
///
/// # Arguments
///
/// * `transaction` — the transaction to insert in.
/// * `rows` — archived trajectories to insert.
///
/// # Returns
///
/// The number of rows actually inserted.
///
/// # Errors
///
/// [`SqlError::Postgres`] on any statement failure.
pub fn insert_archived(
    transaction: &mut Transaction<'_>,
    rows: &[ArchivedRow],
) -> Result<u64, SqlError> {
    insert_via_temp_table(
        transaction,
        "archived_trajectories",
        ARCHIVED_IN,
        "designation",
        |transaction, table| {
            copy::copy_archived_trajectories(transaction, table, rows, &ProgressBar::hidden())
        },
    )
}

/// Steps 1–3 of the [module doc](self#how), for one target table.
///
/// # Arguments
///
/// * `transaction` — the transaction to insert in.
/// * `target` — the table to insert into.
/// * `temp` — name of the temporary table to stage the rows in.
/// * `key` — `target`'s conflict column.
/// * `copy_into` — `COPY`s the rows into the table it is given.
///
/// All names are internal constants (they are interpolated into SQL).
///
/// # Returns
///
/// The number of rows inserted into `target`.
///
/// # Errors
///
/// [`SqlError::Postgres`] on any statement failure.
fn insert_via_temp_table(
    transaction: &mut Transaction<'_>,
    target: &str,
    temp: &str,
    key: &str,
    copy_into: impl FnOnce(&mut Transaction<'_>, &str) -> Result<(), postgres::Error>,
) -> Result<u64, SqlError> {
    // `ON COMMIT DROP` is only a safety net: the explicit `DROP` below lets
    // the same transaction call this function again.
    transaction.batch_execute(&format!(
        "CREATE TEMP TABLE {temp} (LIKE {target} INCLUDING DEFAULTS) ON COMMIT DROP;"
    ))?;
    copy_into(transaction, temp)?;
    let inserted = transaction.execute(
        &format!("INSERT INTO {target} SELECT * FROM {temp} ON CONFLICT ({key}) DO NOTHING"),
        &[],
    )?;
    transaction.batch_execute(&format!("DROP TABLE {temp};"))?;
    Ok(inserted)
}
