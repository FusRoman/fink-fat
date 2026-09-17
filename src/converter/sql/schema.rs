//! Schema DDL for the tables [`super::write_sql_tables`] bulk-loads, plus the
//! drop/rebuild of secondary indexes and foreign keys around the bulk load
//! itself.
//!
//! Each function is a thin wrapper around one (or a few) `.sql` file next to
//! this one, loaded at compile time with [`include_str!`] — kept as plain
//! `.sql` rather than Rust string literals so the schema itself can be read,
//! diffed, and (in principle) run directly with `psql` without extracting it
//! from Rust source first.

use postgres::Transaction;

/// Creates every table [`super::write_sql_tables`] needs if absent, and
/// migrates any that already exist from an older schema version.
///
/// Three separate scripts, run in order:
/// 1. `create_tables.sql` — the target schema of the 7 bulk-loaded tables,
///    as it would be created fresh.
/// 2. `migrate_legacy_columns.sql` — fixes for a database created by an
///    older version of this binary, where step 1's
///    `CREATE TABLE IF NOT EXISTS` was a no-op against an already-existing,
///    outdated table.
/// 3. `create_orbit_fits_tables.sql` — `orbit_fits`/`orbit_fit_failures`,
///    owned by `fink-fat-explorer` rather than by this bulk load, but
///    bootstrapped here so a fresh database has every table set up after one
///    `convert` run.
///
/// # Arguments
///
/// * `transaction` — the transaction [`super::write_sql_tables`] runs the
///   whole conversion in.
///
/// # Errors
///
/// Any statement failing, as a [`postgres::Error`].
pub(super) fn create_tables(transaction: &mut Transaction<'_>) -> Result<(), postgres::Error> {
    transaction.batch_execute(include_str!("create_tables.sql"))?;
    transaction.batch_execute(include_str!("migrate_legacy_columns.sql"))?;
    transaction.batch_execute(include_str!("create_orbit_fits_tables.sql"))?;
    Ok(())
}

/// Drops the secondary indexes and foreign keys `create_tables.sql` declares
/// on the 7 bulk-loaded tables, so the COPYs in [`super::write_sql_tables`]
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
/// * `transaction` — the transaction [`super::write_sql_tables`] runs the
///   whole conversion in.
///
/// # Errors
///
/// Any statement failing, as a [`postgres::Error`].
pub(super) fn drop_bulk_load_constraints(
    transaction: &mut Transaction<'_>,
) -> Result<(), postgres::Error> {
    transaction.batch_execute(include_str!("drop_bulk_load_constraints.sql"))
}

/// Rebuilds what [`drop_bulk_load_constraints`] dropped, now that the tables
/// hold their final data for this run: one index build and one FK validation
/// scan per constraint, instead of one check per row during the COPYs.
///
/// # Arguments
///
/// * `transaction` — the transaction [`super::write_sql_tables`] runs the
///   whole conversion in.
///
/// # Errors
///
/// Any statement failing, as a [`postgres::Error`].
pub(super) fn restore_bulk_load_constraints(
    transaction: &mut Transaction<'_>,
) -> Result<(), postgres::Error> {
    transaction.batch_execute(include_str!("restore_bulk_load_constraints.sql"))
}
