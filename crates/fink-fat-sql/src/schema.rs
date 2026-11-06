//! Schema DDL of every table this crate writes, plus the drop/rebuild of
//! secondary indexes and foreign keys around bulk loads.
//!
//! The DDL lives in plain `.sql` files under the crate's `sql/` directory,
//! embedded at compile time with [`include_str!`] and exposed as the `*_SQL`
//! constants below — kept as `.sql` rather than Rust string literals so the
//! schema can be read, diffed, linted (SQLFluff, Squawk) and run directly
//! with `psql`. Each function is a thin wrapper running one (or a few) of
//! these scripts; other crates that need a script on its own (e.g.
//! `fink-fat submit`'s `mpc_submissions` table) use the constant instead of
//! embedding the file a second time.
//!
//! # Script layout
//!
//! ```text
//! sql/
//! ├── tables/            CREATE TABLE IF NOT EXISTS, one file per table group
//! │   ├── engine/        written by the engine: observations, trajectory tables,
//! │   │                  archived_trajectories, fink_fat_sync_state
//! │   ├── explorer/      owned by fink-fat-explorer: orbit_fits, cnd_queries, skybot_*
//! │   └── submit/        owned by `fink-fat submit`: mpc_submissions
//! ├── bulk_load/         drop / restore of secondary indexes and FKs around the COPYs
//! ├── night_sync/        staging, swap and cleanup of the streaming pipeline's nightly sync
//! └── migrations/        upgrades of databases created by older versions
//! ```
//!
//! # Table groups
//!
//! | Group | Script | Persistence | Written by |
//! |-------|--------|-------------|-----------|
//! | `observations` | `tables/engine/observations.sql` | LOGGED | `convert` (full reload), stream (live inserts) |
//! | trajectory tables (6) | `tables/engine/trajectory_tables.sql` | UNLOGGED | `convert` (full reload), stream (nightly swap) |
//! | `archived_trajectories` | `tables/engine/archived_trajectories.sql` | UNLOGGED | `convert` (full reload), stream (inserts) |
//! | `fink_fat_sync_state` | `tables/engine/sync_state.sql` | UNLOGGED | stream (nightly swap) |
//! | explorer / submit tables | `tables/explorer/*.sql`, `tables/submit/*.sql` | LOGGED | explorer, `submit` |
//!
//! # Schema-independent scripts
//!
//! The trajectory scripts never qualify table names: run with the default
//! `search_path` they target `public` (`convert`), run with
//! `search_path = fink_fat_staging, public` they build the staging copy the
//! nightly sync swaps in (see [`crate::night_sync`]). Index and constraint
//! names are therefore the same in both schemas.

use postgres::Transaction;

/// `observations` and its index.
pub const CREATE_OBSERVATIONS_TABLE_SQL: &str =
    include_str!("../sql/tables/engine/observations.sql");

/// The 6 trajectory tables (`branches`, `kf_bank`, `branch_observations`,
/// `hypotheses`, `kf_state`, `branch_best_state`), their foreign keys and
/// indexes; schema-independent (see the [module doc](self)).
pub const CREATE_TRAJECTORY_TABLES_SQL: &str =
    include_str!("../sql/tables/engine/trajectory_tables.sql");

/// `archived_trajectories`.
pub const CREATE_ARCHIVED_TABLE_SQL: &str =
    include_str!("../sql/tables/engine/archived_trajectories.sql");

/// `fink_fat_sync_state`, the streaming pipeline's record of the last
/// trajectory swap.
pub const CREATE_SYNC_STATE_TABLE_SQL: &str = include_str!("../sql/tables/engine/sync_state.sql");

/// Migration of a database created by an older `convert` (persistence
/// switches, added columns, backfilled foreign key).
pub const MIGRATE_LEGACY_COLUMNS_SQL: &str = include_str!("../sql/migrations/legacy_columns.sql");

/// `orbit_fits` / `orbit_fit_failures`, owned by `fink-fat-explorer`.
pub const CREATE_ORBIT_FITS_TABLES_SQL: &str =
    include_str!("../sql/tables/explorer/orbit_fits.sql");

/// `cnd_queries`, owned by `fink-fat-explorer`.
pub const CREATE_CND_TABLES_SQL: &str = include_str!("../sql/tables/explorer/cnd_queries.sql");

/// `skybot_bulk_jobs` / `skybot_obs_status`, owned by `fink-fat-explorer`.
pub const CREATE_SKYBOT_BULK_TABLES_SQL: &str =
    include_str!("../sql/tables/explorer/skybot_bulk.sql");

/// `mpc_submissions`, owned by `fink-fat submit`.
pub const CREATE_MPC_SUBMISSION_TABLES_SQL: &str =
    include_str!("../sql/tables/submit/mpc_submissions.sql");

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

/// Creates every table this crate and its users need if absent, and migrates
/// any that already exist from an older schema version.
///
/// Scripts run in order:
/// 1. `migrations/legacy_columns.sql` — upgrades of tables created by an
///    older version of this crate (persistence switches, added columns, a
///    backfilled foreign key), where a `CREATE TABLE IF NOT EXISTS` would be
///    a no-op against the outdated table. First, because step 2's
///    `COMMENT ON COLUMN` statements need every column to exist; a no-op on
///    a fresh database (`ALTER TABLE IF EXISTS`).
/// 2. `tables/engine/*.sql` — `observations`, the trajectory tables (in the
///    current `search_path`, i.e. `public`), `archived_trajectories`,
///    `fink_fat_sync_state`: created if absent, and documented
///    (`COMMENT ON`). Observations first: a trajectory table references it.
/// 3. `tables/explorer/*.sql` (`orbit_fits`, `orbit_fit_failures`,
///    `cnd_queries`, `skybot_bulk_jobs`, `skybot_obs_status`) and
///    `tables/submit/*.sql` (`mpc_submissions`), owned by
///    `fink-fat-explorer` and `fink-fat submit` but bootstrapped here so a
///    fresh database has every table after one `convert` or one streaming
///    start. `skybot_obs_status` is the single source of truth for Skybot
///    results (see `fink-fat-explorer`'s `skybot_search` module doc);
///    `submit` also creates `mpc_submissions` defensively itself, since it
///    can run independently.
///
/// Idempotent: safe to run at every `convert` and every streaming start.
///
/// # Arguments
///
/// * `transaction` — the transaction to run the scripts in.
///
/// # Errors
///
/// Any statement failing, as a [`postgres::Error`].
pub fn create_tables(transaction: &mut Transaction<'_>) -> Result<(), postgres::Error> {
    transaction.batch_execute(MIGRATE_LEGACY_COLUMNS_SQL)?;
    transaction.batch_execute(CREATE_OBSERVATIONS_TABLE_SQL)?;
    transaction.batch_execute(CREATE_TRAJECTORY_TABLES_SQL)?;
    transaction.batch_execute(CREATE_ARCHIVED_TABLE_SQL)?;
    transaction.batch_execute(CREATE_SYNC_STATE_TABLE_SQL)?;
    transaction.batch_execute(CREATE_ORBIT_FITS_TABLES_SQL)?;
    transaction.batch_execute(CREATE_CND_TABLES_SQL)?;
    transaction.batch_execute(CREATE_SKYBOT_BULK_TABLES_SQL)?;
    transaction.batch_execute(CREATE_MPC_SUBMISSION_TABLES_SQL)?;
    Ok(())
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
