//! Load a [`BranchCollection`] into Postgres, using the same 7-table schema
//! (`observations`/`branches`/`kf_bank`/`branch_observations`/`hypotheses`/
//! `kf_state`/`archived_trajectories`, joined by
//! `branch_id`/`hypothesis_id`/`id`) as [`crate::converter::parquet`]'s
//! 6-table schema, plus `observations` itself.
//!
//! Organised by concern rather than by table:
//! - [`rows`] — plain row structs, no logic.
//! - [`build`] — pure functions turning a [`BranchCollection`] into those
//!   rows; the part of this module worth unit-testing directly (see its own
//!   `tests` module).
//! - [`observations`] — reading the external observation Parquet.
//! - [`schema`] — DDL (table/index/constraint creation), backed by the
//!   `.sql` files in this directory rather than Rust string literals.
//! - [`copy`] — the `COPY ... FROM STDIN BINARY` bulk loaders, one per table.
//!
//! `u64`/`usize` fields are cast to `i64` throughout — Postgres has no
//! unsigned integer type. No TLS: `Client::connect` is called with
//! [`NoTls`], suitable for a local/trusted Postgres instance. Add TLS
//! support if this ever talks to a non-local database.
//!
//! The 7 bulk-loaded tables are `UNLOGGED` (no WAL, reset to empty by
//! Postgres after a crash/unclean shutdown) and have their secondary
//! indexes/foreign keys dropped before the COPYs and rebuilt after
//! ([`schema::drop_bulk_load_constraints`]/[`schema::restore_bulk_load_constraints`])
//! — both are safe specifically because [`write_sql_tables`] always fully
//! regenerates these tables from source data inside one transaction: there
//! is no window where a crash could lose anything these tables hold that
//! re-running `convert` wouldn't already recreate, and no query reads them
//! while they're temporarily unconstrained. `orbit_fits`/`orbit_fit_failures`
//! are unaffected — see `create_orbit_fits_tables.sql`.

mod build;
mod copy;
mod observations;
mod rows;
mod schema;

use camino::Utf8Path;
use fink_fat_engine::topocentric_kf::branching::BranchCollection;
use postgres::{Client, NoTls};

/// Loads `branch_collection` and the raw observations at
/// `observations_parquet_path` into the Postgres database at `database_url`:
/// creates the 7 bulk-loaded tables (and `orbit_fits`/`orbit_fit_failures`)
/// if absent, `TRUNCATE`s the 7 bulk-loaded ones, then bulk-loads via
/// `COPY ... FROM STDIN BINARY`.
///
/// Runs in a single transaction — a failure partway through leaves the
/// previous contents untouched. Observations are read and copied before
/// `branch_observations`, so that `branch_observations.obs_id`'s foreign key
/// into `observations(id)` is always satisfied once it's rebuilt at the end
/// (see the module-level doc comment), regardless of invocation order
/// relative to any previous run.
///
/// # Arguments
///
/// * `branch_collection` — the loaded snapshot to convert.
/// * `database_url` — a Postgres connection string
///   (`postgres://user:pass@host/db`); never logged, even at the generic
///   "connecting" progress message, since it may embed a plaintext password.
/// * `observations_parquet_path` — path to the raw observation parquet (see
///   [`observations::read_observation_rows`]) to load into `observations`.
///
/// # Errors
///
/// Reading the observation parquet, connecting to Postgres, or any statement
/// within the transaction failing. The transaction is never explicitly
/// rolled back on error: dropping it without a `commit()` is enough for
/// Postgres to discard every change made so far.
pub(crate) fn write_sql_tables(
    branch_collection: &BranchCollection,
    database_url: &str,
    observations_parquet_path: &Utf8Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let observation_rows = observations::read_observation_rows(observations_parquet_path)?;

    // Keyed by `ObservationRow::id` (== `BranchObservationRow::obs_id`), so
    // each branch's arc/night-count/inter-night-gap stats can be computed
    // once here in Rust, up front, instead of re-derived by Postgres on
    // every homepage sort — see `build::compute_obs_stats`.
    let obs_index: build::ObsIndex = observation_rows
        .iter()
        .map(|r| (r.id, (r.night_id, r.mjd_tt)))
        .collect();

    let (branch_rows, kf_bank_rows, branch_observation_rows, hypothesis_rows, kf_state_rows) =
        build::build_branch_rows(branch_collection, &obs_index);
    let archived_rows = build::build_archived_rows(branch_collection);

    let total_rows = observation_rows.len()
        + branch_rows.len()
        + kf_bank_rows.len()
        + branch_observation_rows.len()
        + hypothesis_rows.len()
        + kf_state_rows.len()
        + archived_rows.len();
    let pb = copy::build_progress_bar(total_rows);

    // `database_url` may embed a plaintext password (`postgres://user:pw@host/db`)
    // — never logged, even at this generic "connecting" granularity.
    pb.println("Connecting to Postgres...");
    let mut client = Client::connect(database_url, NoTls)?;
    let mut transaction = client.transaction()?;

    pb.println("Creating tables (CREATE TABLE IF NOT EXISTS)...");
    schema::create_tables(&mut transaction)?;

    // Dropped here and recreated after the COPYs below
    // (`schema::restore_bulk_load_constraints`): a secondary index/FK built
    // incrementally, row by row, as COPY streams in is what turns a bulk
    // load into thousands of individual b-tree insertions and FK lookups.
    // Built once against data already in place, an index is a single
    // sort-and-build pass and an FK is a single scan/join — both far cheaper
    // than paying for them per row. The primary keys are untouched: they're
    // what guarantees uniqueness during the load and what the FKs look up
    // against on the referenced side.
    pb.println("Dropping secondary indexes/constraints for the bulk load...");
    schema::drop_bulk_load_constraints(&mut transaction)?;

    pb.println("Truncating existing tables...");
    transaction.batch_execute(
        "TRUNCATE TABLE branches, kf_bank, branch_observations, hypotheses, kf_state, \
         archived_trajectories, observations CASCADE;",
    )?;

    copy::copy_observations(&mut transaction, &observation_rows, &pb)?;
    copy::copy_branches(&mut transaction, &branch_rows, &pb)?;
    copy::copy_kf_bank(&mut transaction, &kf_bank_rows, &pb)?;
    copy::copy_branch_observations(&mut transaction, &branch_observation_rows, &pb)?;
    copy::copy_hypotheses(&mut transaction, &hypothesis_rows, &pb)?;
    copy::copy_kf_state(&mut transaction, &kf_state_rows, &pb)?;
    copy::copy_archived_trajectories(&mut transaction, &archived_rows, &pb)?;

    pb.println("Rebuilding secondary indexes/constraints...");
    schema::restore_bulk_load_constraints(&mut transaction)?;

    pb.println("Committing transaction...");
    transaction.commit()?;
    pb.finish_with_message("SQL export complete");
    Ok(())
}
