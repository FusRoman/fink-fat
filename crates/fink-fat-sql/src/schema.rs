//! Schema DDL for the tables [`crate::write_bulk_tables`] bulk-loads, plus the
//! drop/rebuild of secondary indexes and foreign keys around the bulk load
//! itself.
//!
//! The DDL lives in plain `.sql` files under the crate's `sql/` directory,
//! embedded at compile time with [`include_str!`] and exposed as the
//! `*_SQL` constants below — kept as `.sql` rather than Rust string literals
//! so the schema can be read, diffed, linted (SQLFluff, Squawk) and run
//! directly with `psql`. Each function is a thin wrapper running one (or a
//! few) of these scripts; other crates that need a script on its own (e.g.
//! `fink-fat submit`'s `mpc_submissions` table) use the constant instead of
//! embedding the file a second time.

use postgres::Transaction;

/// Target schema of the 8 bulk-loaded tables and their indexes.
pub const CREATE_TABLES_SQL: &str = include_str!("../sql/create_tables.sql");

/// Migration of a database created by an older `convert` (UNLOGGED switch, added columns).
pub const MIGRATE_LEGACY_COLUMNS_SQL: &str = include_str!("../sql/migrate_legacy_columns.sql");

/// `orbit_fits` / `orbit_fit_failures`, owned by `fink-fat-explorer`.
pub const CREATE_ORBIT_FITS_TABLES_SQL: &str = include_str!("../sql/create_orbit_fits_tables.sql");

/// `cnd_queries`, owned by `fink-fat-explorer`.
pub const CREATE_CND_TABLES_SQL: &str = include_str!("../sql/create_cnd_tables.sql");

/// `skybot_bulk_jobs` / `skybot_obs_status`, owned by `fink-fat-explorer`.
pub const CREATE_SKYBOT_BULK_TABLES_SQL: &str =
    include_str!("../sql/create_skybot_bulk_tables.sql");

/// `mpc_submissions`, owned by `fink-fat submit`.
pub const CREATE_MPC_SUBMISSION_TABLES_SQL: &str =
    include_str!("../sql/create_mpc_submission_tables.sql");

/// Drop of the secondary indexes / foreign keys before the bulk `COPY`s.
pub const DROP_BULK_LOAD_CONSTRAINTS_SQL: &str =
    include_str!("../sql/drop_bulk_load_constraints.sql");

/// Rebuild of what [`DROP_BULK_LOAD_CONSTRAINTS_SQL`] dropped.
pub const RESTORE_BULK_LOAD_CONSTRAINTS_SQL: &str =
    include_str!("../sql/restore_bulk_load_constraints.sql");

/// Creates every table [`crate::write_bulk_tables`] needs if absent, and
/// migrates any that already exist from an older schema version.
///
/// Three separate scripts, run in order:
/// 1. `create_tables.sql` — the target schema of the 8 bulk-loaded tables,
///    as it would be created fresh.
/// 2. `migrate_legacy_columns.sql` — fixes for a database created by an
///    older version of this binary, where step 1's
///    `CREATE TABLE IF NOT EXISTS` was a no-op against an already-existing,
///    outdated table.
/// 3. `create_orbit_fits_tables.sql` — `orbit_fits`/`orbit_fit_failures`,
///    owned by `fink-fat-explorer` rather than by this bulk load, but
///    bootstrapped here so a fresh database has every table set up after one
///    `convert` run.
/// 4. `create_cnd_tables.sql` — `cnd_queries`, same rationale as step 3.
/// 5. `create_skybot_bulk_tables.sql` — `skybot_bulk_jobs`/`skybot_obs_status`,
///    same rationale as step 3. `skybot_obs_status` is the single source of
///    truth for Skybot results, shared by the per-lineage search and the
///    bulk sweep alike (see `fink-fat-explorer`'s `skybot_search` module doc
///    comment) — there is deliberately no separate per-lineage table.
/// 6. `create_mpc_submission_tables.sql` — `mpc_submissions`, owned by
///    `fink-fat submit` (see `src/submit/`) rather than by this bulk load,
///    same rationale as step 3 — bootstrapped here so a fresh database has
///    it set up even before the first `fink-fat submit` run (which also
///    creates it defensively itself, since it can run independently of
///    `convert`).
///
/// # Arguments
///
/// * `transaction` — the transaction [`crate::write_bulk_tables`] runs the
///   whole conversion in.
///
/// # Errors
///
/// Any statement failing, as a [`postgres::Error`].
pub fn create_tables(transaction: &mut Transaction<'_>) -> Result<(), postgres::Error> {
    transaction.batch_execute(CREATE_TABLES_SQL)?;
    transaction.batch_execute(MIGRATE_LEGACY_COLUMNS_SQL)?;
    transaction.batch_execute(CREATE_ORBIT_FITS_TABLES_SQL)?;
    transaction.batch_execute(CREATE_CND_TABLES_SQL)?;
    transaction.batch_execute(CREATE_SKYBOT_BULK_TABLES_SQL)?;
    transaction.batch_execute(CREATE_MPC_SUBMISSION_TABLES_SQL)?;
    Ok(())
}

/// Drops the secondary indexes and foreign keys `create_tables.sql` declares
/// on the 8 bulk-loaded tables, so the COPYs in [`crate::write_bulk_tables`]
/// insert into bare (primary-key-only) tables instead of paying for b-tree
/// maintenance and FK lookups on every row. Always run immediately before
/// the COPYs, and undone by [`restore_bulk_load_constraints`] immediately
/// after — nothing reads these tables in between (the data is invalid until
/// the whole transaction commits anyway), so being unconstrained for that
/// window costs nothing.
///
/// `IF EXISTS` throughout (see `drop_bulk_load_constraints.sql`):
/// `create_tables` always created these moments earlier in the same
/// transaction, but staying idempotent costs nothing and guards against this
/// function ever being called out of order.
///
/// # Arguments
///
/// * `transaction` — the transaction [`crate::write_bulk_tables`] runs the
///   whole conversion in.
///
/// # Errors
///
/// Any statement failing, as a [`postgres::Error`].
pub fn drop_bulk_load_constraints(
    transaction: &mut Transaction<'_>,
) -> Result<(), postgres::Error> {
    transaction.batch_execute(DROP_BULK_LOAD_CONSTRAINTS_SQL)
}

/// Rebuilds what [`drop_bulk_load_constraints`] dropped, now that the tables
/// hold their final data for this run: one index build and one FK validation
/// scan per constraint, instead of one check per row during the COPYs.
///
/// # Arguments
///
/// * `transaction` — the transaction [`crate::write_bulk_tables`] runs the
///   whole conversion in.
///
/// # Errors
///
/// Any statement failing, as a [`postgres::Error`].
pub fn restore_bulk_load_constraints(
    transaction: &mut Transaction<'_>,
) -> Result<(), postgres::Error> {
    transaction.batch_execute(RESTORE_BULK_LOAD_CONSTRAINTS_SQL)
}

/// Refreshes the planner statistics of every bulk-loaded table.
///
/// `TRUNCATE` resets a table's statistics, and autovacuum may take a long
/// time to get back to the largest ones (`kf_state` stayed at
/// `reltuples = -1` well after a convert): until then the planner works
/// blind, and the explorer's hypothesis count (read from `reltuples`) is
/// missing. `ANALYZE` only samples each table, so this costs seconds even on
/// `kf_state`, and it can run inside the conversion transaction.
///
/// # Arguments
///
/// * `transaction` — the transaction [`crate::write_bulk_tables`] runs the
///   whole conversion in.
///
/// # Errors
///
/// Any statement failing, as a [`postgres::Error`].
pub fn analyze_bulk_loaded_tables(
    transaction: &mut Transaction<'_>,
) -> Result<(), postgres::Error> {
    transaction.batch_execute(
        "ANALYZE observations, branches, kf_bank, branch_observations, hypotheses, kf_state, \
         branch_best_state, archived_trajectories;",
    )
}
