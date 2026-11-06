//! Bulk loads: the full reload of `convert` ([`write_bulk_tables`]) and the
//! loading of the 6 trajectory tables it shares with the nightly sync
//! ([`load_trajectory_tables`]).

use fink_fat_engine::topocentric_kf::branching::BranchCollection;
use indicatif::ProgressBar;
use postgres::{Client, NoTls, Transaction};

use crate::{
    SqlError,
    build::{self, TrajectoryRows},
    copy,
    rows::ObservationRow,
    schema,
};

/// Loads `rows` into the 6 (empty) trajectory tables of the current
/// `search_path`: `public` for `convert`, the staging schema for the nightly
/// sync (see [`crate::night_sync`]).
///
/// # Steps
/// 1. Drop the secondary indexes and foreign keys
///    ([`schema::drop_trajectory_constraints`]): a secondary index or FK
///    maintained row by row as `COPY` streams in turns a bulk load into
///    millions of b-tree insertions and FK lookups, while built once against
///    data already in place it is a single sort or scan. Primary keys stay:
///    they guarantee uniqueness during the load.
/// 2. `COPY` the 6 tables, parents first. Each `kf_state` row's dynamical
///    family is classified once (in parallel) and shared by `kf_state` and
///    `branch_best_state`.
/// 3. Rebuild the indexes and foreign keys
///    ([`schema::restore_trajectory_constraints`]). This validates
///    `branch_observations.obs_id` against `observations`, which must
///    therefore already hold every referenced observation.
/// 4. `ANALYZE` the 6 tables.
///
/// # Arguments
///
/// * `transaction` — the load transaction, its `search_path` set to the
///   target schema.
/// * `rows` — see [`TrajectoryRows::build`].
/// * `pb` — progress bar, ticked once per row (pass
///   [`ProgressBar::hidden`] for an unattended run).
///
/// # Errors
///
/// [`SqlError::Postgres`] on any statement failure — notably a branch
/// observation absent from `observations` (step 3).
pub fn load_trajectory_tables(
    transaction: &mut Transaction<'_>,
    rows: &TrajectoryRows,
    pb: &ProgressBar,
) -> Result<(), SqlError> {
    schema::drop_trajectory_constraints(transaction)?;

    copy::copy_branches(transaction, &rows.branches, pb)?;
    copy::copy_kf_bank(transaction, &rows.kf_bank, pb)?;
    copy::copy_branch_observations(transaction, &rows.branch_observations, pb)?;
    copy::copy_hypotheses(transaction, &rows.hypotheses, pb)?;
    let classifications = copy::classify_kf_state_rows(&rows.kf_state);
    copy::copy_kf_state(transaction, &rows.kf_state, &classifications, pb)?;
    copy::copy_branch_best_state(
        transaction,
        &rows.best_state,
        &rows.kf_state,
        &classifications,
        pb,
    )?;
    drop(classifications);

    pb.println("Rebuilding trajectory indexes/constraints...");
    schema::restore_trajectory_constraints(transaction)?;
    pb.println("Refreshing trajectory planner statistics (ANALYZE)...");
    schema::analyze_trajectory_tables(transaction)?;
    Ok(())
}

/// Full reload: loads `branch_collection` and `observation_rows` into the
/// Postgres database at `database_url`, replacing whatever the 8 loaded
/// tables held.
///
/// # Steps
/// 1. Create every table if absent and migrate legacy ones
///    ([`schema::create_tables`]).
/// 2. Drop `observations`' secondary index, then `TRUNCATE` the 8 tables.
/// 3. `COPY` `observations` — before the trajectory tables, so that
///    `branch_observations.obs_id`'s foreign key is satisfied when it is
///    rebuilt.
/// 4. Load the trajectory tables ([`load_trajectory_tables`]).
/// 5. `COPY` `archived_trajectories`.
/// 6. Rebuild `observations`' index and `ANALYZE` the two remaining tables.
///
/// Runs in a single transaction: a failure partway through leaves the
/// previous contents untouched. The explorer's reads of these tables wait
/// for the commit (the `TRUNCATE` holds `ACCESS EXCLUSIVE`).
///
/// # Arguments
///
/// * `branch_collection` — the loaded snapshot to convert (`branches`, and
///   `archived` holding the run's whole archive).
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

    let trajectory_rows = TrajectoryRows::build(branch_collection, &obs_index);
    let archived_rows = build::build_archived_rows(&branch_collection.archived);

    let total_rows = observation_rows.len() + trajectory_rows.len() + archived_rows.len();
    let pb = copy::build_progress_bar(total_rows);

    // `database_url` may embed a plaintext password (`postgres://user:pw@host/db`)
    // — never logged, even at this generic "connecting" granularity.
    pb.println("Connecting to Postgres...");
    let mut client = Client::connect(database_url, NoTls)?;
    let mut transaction = client.transaction()?;

    // 1.
    pb.println("Creating tables (CREATE TABLE IF NOT EXISTS)...");
    schema::create_tables(&mut transaction)?;

    // 2.
    pb.println("Truncating existing tables...");
    schema::drop_observations_indexes(&mut transaction)?;
    transaction.batch_execute(
        "TRUNCATE TABLE branches, kf_bank, branch_observations, hypotheses, kf_state, \
         branch_best_state, archived_trajectories, observations CASCADE;",
    )?;

    // 3. – 5.
    copy::copy_observations(&mut transaction, "observations", observation_rows, &pb)?;
    load_trajectory_tables(&mut transaction, &trajectory_rows, &pb)?;
    copy::copy_archived_trajectories(
        &mut transaction,
        "archived_trajectories",
        &archived_rows,
        &pb,
    )?;

    // 6.
    pb.println("Rebuilding observations index, refreshing statistics...");
    schema::restore_observations_indexes(&mut transaction)?;
    schema::analyze_observation_and_archive_tables(&mut transaction)?;

    pb.println("Committing transaction...");
    transaction.commit()?;
    pb.finish_with_message("SQL export complete");
    Ok(())
}
