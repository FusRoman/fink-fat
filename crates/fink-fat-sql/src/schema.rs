//! Schema of every table this crate writes: versioned migrations of the
//! persistent tables (refinery), regeneration of the trajectory tables, and
//! the drop/rebuild of secondary indexes and foreign keys around bulk loads.
//!
//! The DDL lives in plain `.sql` files under the crate's `sql/` directory,
//! embedded at compile time ([`include_str!`], or refinery's
//! `embed_migrations!` for `sql/migrations/`) — kept as `.sql` rather than
//! Rust string literals so the schema can be read, diffed, linted (SQLFluff,
//! Squawk) and run directly with `psql`. Each function is a thin wrapper
//! running one (or a few) of these scripts.
//!
//! # Script layout
//!
//! ```text
//! sql/
//! ├── migrations/        refinery migrations (V{n}__{name}.sql) of the persistent tables
//! ├── tables/engine/     trajectory_tables.sql: the 6 regenerated trajectory tables
//! ├── bulk_load/         drop / restore of secondary indexes, FKs and trajectory tables
//! │                      around the COPYs
//! └── night_sync/        staging, swap and cleanup of the streaming pipeline's nightly sync
//! ```
//!
//! # Table groups
//!
//! | Group | Schema managed by | Persistence | Written by |
//! |-------|-------------------|-------------|-----------|
//! | `observations` | migrations | LOGGED | `convert` (full reload), stream (live inserts) |
//! | `archived_trajectories` | migrations | UNLOGGED | `convert` (full reload), stream (inserts) |
//! | `fink_fat_sync_state` | migrations | UNLOGGED | stream (nightly swap) |
//! | explorer / submit tables | migrations | LOGGED | explorer, `submit` |
//! | trajectory tables (6) | `tables/engine/trajectory_tables.sql` | UNLOGGED | `convert` (DROP + CREATE + COPY), stream (nightly swap) |
//!
//! # Migrated vs regenerated tables
//!
//! Tables whose rows outlive a load are **migrated**: [`migrate`] applies the
//! pending `sql/migrations/V{n}__{name}.sql` files, in version order, and
//! records each one in `refinery_schema_history`. A schema change of one of
//! them is a new `V{n+1}__{name}.sql`; an applied migration is frozen (its
//! checksum is checked at every run, see the crate README).
//!
//! The 6 trajectory tables are **regenerated** instead: every `convert`
//! drops and recreates them ([`recreate_trajectory_tables`]) and every
//! nightly sync builds a fresh generation in a staging schema
//! ([`crate::night_sync`]), both from `tables/engine/trajectory_tables.sql`.
//! A schema change of one of them is an edit of that script, never a
//! migration: a migration of the `public` copy would be overwritten by the
//! next load.
//!
//! # Schema-independent scripts
//!
//! The trajectory scripts never qualify table names: run with the default
//! `search_path` they target `public` (`convert`), run with
//! `search_path = fink_fat_staging, public` they build the staging copy the
//! nightly sync swaps in (see [`crate::night_sync`]). Index and constraint
//! names are therefore the same in both schemas.

use postgres::{Client, Transaction};
use refinery::{Report, Runner};

use crate::SqlError;

/// The refinery migrations of `sql/migrations/`, embedded at compile time
/// (generates `embedded::migrations::runner()`).
///
/// An edit of an existing file triggers a rebuild (the macro embeds each file
/// with [`include_str!`]); a **new** file is only picked up once the crate is
/// rebuilt for another reason — `cargo clean -p fink-fat-sql` (or touching
/// this file) forces it.
mod embedded {
    refinery::embed_migrations!("sql/migrations");
}

/// Key of the session-level Postgres advisory lock [`migrate`] holds while
/// it runs: `convert`, the streaming pipeline and `submit` may all start at
/// the same time against the same database, and refinery takes no lock of its
/// own — it reads `refinery_schema_history` and applies the pending
/// migrations in separate transactions, so two concurrent runs would both
/// apply the same migration and the second would fail on the history table's
/// primary key. The value is arbitrary but fixed: the ASCII bytes of
/// `"finkfat"` followed by a 1 (lock #1 of this crate).
pub const MIGRATION_LOCK_KEY: i64 = 0x66_69_6E_6B_66_61_74_01;

/// The 6 trajectory tables (`branches`, `kf_bank`, `branch_observations`,
/// `hypotheses`, `kf_state`, `branch_best_state`), their foreign keys and
/// indexes; schema-independent (see the [module doc](self)).
pub const CREATE_TRAJECTORY_TABLES_SQL: &str =
    include_str!("../sql/tables/engine/trajectory_tables.sql");

/// Drop of the 6 trajectory tables of the current `search_path`, before
/// [`CREATE_TRAJECTORY_TABLES_SQL`] recreates them (see
/// [`recreate_trajectory_tables`]).
pub const DROP_TRAJECTORY_TABLES_SQL: &str =
    include_str!("../sql/bulk_load/drop_trajectory_tables.sql");

/// Drop of the trajectory tables' secondary indexes and foreign keys before
/// their bulk `COPY`s; schema-independent.
pub const DROP_TRAJECTORY_CONSTRAINTS_SQL: &str =
    include_str!("../sql/bulk_load/drop_trajectory_constraints.sql");

/// Rebuild of what [`DROP_TRAJECTORY_CONSTRAINTS_SQL`] dropped.
pub const RESTORE_TRAJECTORY_CONSTRAINTS_SQL: &str =
    include_str!("../sql/bulk_load/restore_trajectory_constraints.sql");

/// Drop of `observations`' secondary index before `convert`'s full `COPY`.
pub const DROP_OBSERVATIONS_INDEXES_SQL: &str =
    include_str!("../sql/bulk_load/drop_observations_indexes.sql");

/// Rebuild of what [`DROP_OBSERVATIONS_INDEXES_SQL`] dropped.
pub const RESTORE_OBSERVATIONS_INDEXES_SQL: &str =
    include_str!("../sql/bulk_load/restore_observations_indexes.sql");

/// Creation (and cleanup) of the nightly sync's staging schema.
pub const PREPARE_STAGING_SCHEMA_SQL: &str = include_str!("../sql/night_sync/prepare_staging.sql");

/// The nightly sync's atomic swap of the trajectory tables.
pub const SWAP_TRAJECTORY_TABLES_SQL: &str =
    include_str!("../sql/night_sync/swap_trajectory_tables.sql");

/// Drop of the trajectory tables swapped out by [`SWAP_TRAJECTORY_TABLES_SQL`].
pub const DROP_RETIRED_TRAJECTORY_TABLES_SQL: &str =
    include_str!("../sql/night_sync/drop_retired_tables.sql");

/// The refinery runner of the embedded migrations, configured for this
/// crate:
///
/// * `grouped`: every pending migration runs in **one** transaction, so a
///   failure leaves the database at its previous version (Postgres DDL is
///   transactional) instead of half-way between two versions;
/// * `abort_divergent`: refuse to run when an applied migration's file
///   changed since (checksum mismatch) — the file must be restored, never
///   edited;
/// * `abort_missing`: refuse to run when an applied migration is absent from
///   this binary, i.e. the binary is older than the database.
///
/// These are refinery's defaults except `grouped`, all set explicitly so the
/// policy is visible here.
fn migration_runner() -> Runner {
    embedded::migrations::runner()
        .set_grouped(true)
        .set_abort_divergent(true)
        .set_abort_missing(true)
}

/// Brings the persistent tables to the latest schema version: applies every
/// pending `sql/migrations/V{n}__{name}.sql` (see the
/// [module doc](self#migrated-vs-regenerated-tables)), under the
/// [`MIGRATION_LOCK_KEY`] advisory lock.
///
/// A database created before refinery was introduced (tables present, no
/// `refinery_schema_history`) needs no manual step: `V1__baseline.sql` is
/// idempotent and upgrades it in place. With no pending migration this costs
/// two short queries, so it is safe to run at every `convert`, every
/// streaming start and every `submit`.
///
/// Runs its own transactions, hence the [`Client`] rather than a
/// [`Transaction`]: call it before opening the load transaction.
///
/// # Arguments
///
/// * `client` — a connection outside any transaction.
///
/// # Returns
///
/// refinery's [`Report`], listing the migrations this call applied (empty
/// when the database was already up to date).
///
/// # Errors
///
/// * [`SqlError::Migration`] — a migration failed (nothing of the batch is
///   applied), an applied migration diverges from or is missing in this
///   binary, or the history table could not be read.
/// * [`SqlError::Postgres`] — taking or releasing the advisory lock failed.
pub fn migrate(client: &mut Client) -> Result<Report, SqlError> {
    client.execute("SELECT pg_advisory_lock($1)", &[&MIGRATION_LOCK_KEY])?;
    let outcome = migration_runner().run(client);
    // Released whatever the outcome (a failed migration's transaction is
    // already rolled back, the session stays usable); the migration error,
    // if any, takes precedence over an unlock error. A lost connection
    // releases the lock on its own.
    let unlock = client.execute("SELECT pg_advisory_unlock($1)", &[&MIGRATION_LOCK_KEY]);
    let report = outcome?;
    unlock?;
    for migration in report.applied_migrations() {
        tracing::info!(%migration, "applied schema migration");
    }
    Ok(report)
}

/// Makes a database ready for any fink-fat process: [`migrate`]s the
/// persistent tables, then creates the trajectory tables if absent
/// ([`create_trajectory_tables`]), in one transaction of its own.
///
/// The entry point of every caller that only needs the tables to exist (the
/// streaming sink, the schema-docs example, tests). `convert`, which reloads
/// the trajectory tables anyway, calls [`migrate`] and
/// [`recreate_trajectory_tables`] instead.
///
/// Existing trajectory tables are left as they are, even if an older
/// fink-fat created them: the streaming pipeline's next nightly sync, or the
/// next `convert`, replaces them with the current shape.
///
/// # Arguments
///
/// * `client` — a connection outside any transaction.
///
/// # Errors
///
/// See [`migrate`]; [`SqlError::Postgres`] if the trajectory tables cannot
/// be created.
pub fn prepare_schema(client: &mut Client) -> Result<(), SqlError> {
    migrate(client)?;
    let mut transaction = client.transaction()?;
    create_trajectory_tables(&mut transaction)?;
    transaction.commit()?;
    Ok(())
}

/// Creates the 6 trajectory tables of the current `search_path` if absent,
/// with their foreign keys, indexes and comments
/// ([`CREATE_TRAJECTORY_TABLES_SQL`]). Idempotent.
///
/// `branch_observations` references `observations`: run [`migrate`] first.
///
/// # Arguments
///
/// * `transaction` — the transaction to run the script in.
///
/// # Errors
///
/// Any statement failing, as a [`postgres::Error`].
pub fn create_trajectory_tables(transaction: &mut Transaction<'_>) -> Result<(), postgres::Error> {
    transaction.batch_execute(CREATE_TRAJECTORY_TABLES_SQL)
}

/// Drops the 6 trajectory tables of the current `search_path` and creates
/// them again from [`CREATE_TRAJECTORY_TABLES_SQL`]: `convert`'s first step
/// of a full reload, which leaves them empty and in their current shape —
/// whatever version of fink-fat created the previous ones, so their schema
/// changes never need a migration (see the
/// [module doc](self#migrated-vs-regenerated-tables)).
///
/// Takes `ACCESS EXCLUSIVE` on the tables until the transaction ends, like
/// the `TRUNCATE` it replaces.
///
/// # Arguments
///
/// * `transaction` — the bulk-load transaction.
///
/// # Errors
///
/// Any statement failing, as a [`postgres::Error`].
pub fn recreate_trajectory_tables(
    transaction: &mut Transaction<'_>,
) -> Result<(), postgres::Error> {
    transaction.batch_execute(DROP_TRAJECTORY_TABLES_SQL)?;
    transaction.batch_execute(CREATE_TRAJECTORY_TABLES_SQL)
}

/// Drops the trajectory tables' secondary indexes and foreign keys, so their
/// `COPY`s insert into bare (primary-key-only) tables instead of paying for
/// b-tree maintenance and FK lookups on every row. Always run right before
/// the `COPY`s and undone by [`restore_trajectory_constraints`] right after,
/// in the same transaction — nothing reads the tables in between (the data
/// is invisible until the transaction commits), so being unconstrained for
/// that window costs nothing.
///
/// Acts on the tables of the current `search_path` (see the
/// [module doc](self#schema-independent-scripts)). `IF EXISTS` throughout,
/// so it is idempotent.
///
/// # Arguments
///
/// * `transaction` — the bulk-load transaction.
///
/// # Errors
///
/// Any statement failing, as a [`postgres::Error`].
pub fn drop_trajectory_constraints(
    transaction: &mut Transaction<'_>,
) -> Result<(), postgres::Error> {
    transaction.batch_execute(DROP_TRAJECTORY_CONSTRAINTS_SQL)
}

/// Rebuilds what [`drop_trajectory_constraints`] dropped, now that the
/// tables hold their final data: one index build and one FK validation scan
/// per constraint, instead of one check per row during the `COPY`s.
///
/// # Arguments
///
/// * `transaction` — the bulk-load transaction.
///
/// # Errors
///
/// Any statement failing (e.g. a `branch_observations.obs_id` absent from
/// `observations`), as a [`postgres::Error`].
pub fn restore_trajectory_constraints(
    transaction: &mut Transaction<'_>,
) -> Result<(), postgres::Error> {
    transaction.batch_execute(RESTORE_TRAJECTORY_CONSTRAINTS_SQL)
}

/// Drops `observations`' secondary index before `convert`'s full reload
/// (see [`drop_trajectory_constraints`] for the rationale). Never used by
/// the streaming pipeline, which appends to a live, indexed table.
///
/// # Arguments
///
/// * `transaction` — the bulk-load transaction.
///
/// # Errors
///
/// Any statement failing, as a [`postgres::Error`].
pub fn drop_observations_indexes(transaction: &mut Transaction<'_>) -> Result<(), postgres::Error> {
    transaction.batch_execute(DROP_OBSERVATIONS_INDEXES_SQL)
}

/// Rebuilds what [`drop_observations_indexes`] dropped.
///
/// # Arguments
///
/// * `transaction` — the bulk-load transaction.
///
/// # Errors
///
/// Any statement failing, as a [`postgres::Error`].
pub fn restore_observations_indexes(
    transaction: &mut Transaction<'_>,
) -> Result<(), postgres::Error> {
    transaction.batch_execute(RESTORE_OBSERVATIONS_INDEXES_SQL)
}

/// Refreshes the planner statistics of the trajectory tables of the current
/// `search_path`.
///
/// A fresh or `TRUNCATE`d table has no statistics, and autovacuum may take a
/// long time to get back to the largest ones (`kf_state` stayed at
/// `reltuples = -1` well after a convert): until then the planner works
/// blind, and the explorer's hypothesis count (read from `reltuples`) is
/// missing. `ANALYZE` only samples each table, so this costs seconds even on
/// `kf_state`, and it can run inside the load transaction.
///
/// # Arguments
///
/// * `transaction` — the bulk-load transaction.
///
/// # Errors
///
/// Any statement failing, as a [`postgres::Error`].
pub fn analyze_trajectory_tables(transaction: &mut Transaction<'_>) -> Result<(), postgres::Error> {
    transaction.batch_execute(
        "ANALYZE branches, kf_bank, branch_observations, hypotheses, kf_state, branch_best_state;",
    )
}

/// Refreshes the planner statistics of `observations` and
/// `archived_trajectories` (see [`analyze_trajectory_tables`]).
///
/// # Arguments
///
/// * `transaction` — the bulk-load transaction.
///
/// # Errors
///
/// Any statement failing, as a [`postgres::Error`].
pub fn analyze_observation_and_archive_tables(
    transaction: &mut Transaction<'_>,
) -> Result<(), postgres::Error> {
    transaction.batch_execute("ANALYZE observations, archived_trajectories;")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The embedded migrations are numbered `V1`, `V2`, … with no gap and no
    /// `U` (unversioned) file: refinery's `abort_missing` refuses a version
    /// lower than the last applied one that was never applied, so a gap
    /// filled later would brick every database already past it.
    #[test]
    fn migrations_are_contiguous_versioned_from_one() {
        let runner = migration_runner();
        let migrations = runner.get_migrations();
        assert!(
            !migrations.is_empty(),
            "no migration embedded from sql/migrations"
        );

        let mut versions: Vec<i32> = migrations.iter().map(|m| m.version()).collect();
        versions.sort_unstable();
        let expected: Vec<i32> = (1..).take(versions.len()).collect();
        assert_eq!(
            versions, expected,
            "migration versions must be 1, 2, 3, … without gaps"
        );

        for migration in migrations {
            // `refinery_core::Type` is not re-exported by `refinery`: compare
            // its `Display` (`V` / `U`) instead.
            assert_eq!(
                migration.prefix().to_string(),
                "V",
                "{migration}: only versioned (`V{{n}}__…`) migrations are allowed"
            );
        }
    }
}
