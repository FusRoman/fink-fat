//! Full bulk load of a [`BranchCollection`] and its observations into
//! Postgres: the orchestration of [`crate::schema`], [`crate::build`] and
//! [`crate::copy`] in one transaction.

use fink_fat_engine::topocentric_kf::branching::BranchCollection;
use postgres::{Client, NoTls};

use crate::{SqlError, build, copy, rows::ObservationRow, schema};

/// Loads `branch_collection` and `observation_rows` into the Postgres
/// database at `database_url`: creates the 8 bulk-loaded tables (and the
/// explorer/submit-owned tables, see [`schema::create_tables`]) if absent,
/// `TRUNCATE`s the 8 bulk-loaded ones, then bulk-loads them via
/// `COPY ... FROM STDIN BINARY`.
///
/// Runs in a single transaction — a failure partway through leaves the
/// previous contents untouched. Observations are copied before
/// `branch_observations`, so that `branch_observations.obs_id`'s foreign key
/// into `observations(id)` is always satisfied once it's rebuilt at the end
/// (see the crate-level doc comment), regardless of invocation order
/// relative to any previous run.
///
/// # Arguments
///
/// * `branch_collection` — the loaded snapshot to convert (branches and
///   `archived`).
/// * `observation_rows` — every observation to load into `observations`;
///   also the source of each branch's [`build::ObsStats`].
/// * `database_url` — a Postgres connection string
///   (`postgres://user:pass@host/db`); never logged, even at the generic
///   "connecting" progress message, since it may embed a plaintext password.
///
/// # Errors
///
/// [`SqlError::Postgres`] when connecting to Postgres or any statement
/// within the transaction fails. The transaction is never explicitly rolled
/// back on error: dropping it without a `commit()` is enough for Postgres to
/// discard every change made so far.
pub fn write_bulk_tables(
    branch_collection: &BranchCollection,
    observation_rows: &[ObservationRow],
    database_url: &str,
) -> Result<(), SqlError> {
    // Keyed by `ObservationRow::id` (== `BranchObservationRow::obs_id`), so
    // each branch's arc/night-count/inter-night-gap stats can be computed
    // once here in Rust, up front, instead of re-derived by Postgres on
    // every homepage sort — see `build::compute_obs_stats`.
    let obs_index: build::ObsIndex = observation_rows
        .iter()
        .map(|r| (r.id, (r.night_id, r.mjd_tt)))
        .collect();

    let (
        branch_rows,
        kf_bank_rows,
        branch_observation_rows,
        hypothesis_rows,
        kf_state_rows,
        best_state_rows,
    ) = build::build_branch_rows(branch_collection, &obs_index);
    let archived_rows = build::build_archived_rows(branch_collection);

    let total_rows = observation_rows.len()
        + branch_rows.len()
        + kf_bank_rows.len()
        + branch_observation_rows.len()
        + hypothesis_rows.len()
        + kf_state_rows.len()
        + best_state_rows.len()
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
         branch_best_state, archived_trajectories, observations CASCADE;",
    )?;

    copy::copy_observations(&mut transaction, observation_rows, &pb)?;
    copy::copy_branches(&mut transaction, &branch_rows, &pb)?;
    copy::copy_kf_bank(&mut transaction, &kf_bank_rows, &pb)?;
    copy::copy_branch_observations(&mut transaction, &branch_observation_rows, &pb)?;
    copy::copy_hypotheses(&mut transaction, &hypothesis_rows, &pb)?;
    let classifications = copy::classify_kf_state_rows(&kf_state_rows);
    copy::copy_kf_state(&mut transaction, &kf_state_rows, &classifications, &pb)?;
    copy::copy_branch_best_state(
        &mut transaction,
        &best_state_rows,
        &kf_state_rows,
        &classifications,
        &pb,
    )?;
    drop(classifications);
    copy::copy_archived_trajectories(&mut transaction, &archived_rows, &pb)?;

    pb.println("Rebuilding secondary indexes/constraints...");
    schema::restore_bulk_load_constraints(&mut transaction)?;

    pb.println("Refreshing planner statistics (ANALYZE)...");
    schema::analyze_bulk_loaded_tables(&mut transaction)?;

    pb.println("Committing transaction...");
    transaction.commit()?;
    pb.finish_with_message("SQL export complete");
    Ok(())
}
