//! Postgres sink of `fink-fat-stream` ([`fink_fat_stream::sink`]), end to
//! end against a real database.
//!
//! **Requires a running Postgres** and **drops the fink-fat tables** of the
//! database it points at — like `sql_convert.rs`, whose instance it shares
//! (`docker compose up -d db`, or `$DATABASE_URL`, see
//! `common::database_url`). The tests of this file take a lock so they never
//! run against the database concurrently.
//!
//! * [`a_replay_publishes_what_convert_would_load`]: a replay with a
//!   database leaves the 8 tables with exactly the content `convert` loads
//!   from the same snapshot, archive and observations, the swap's private
//!   schemas empty and the indexes / foreign keys under their usual names.
//! * [`a_database_outage_is_caught_up_at_restart`]: a replay against an
//!   unreachable server keeps everything in the outbox; a restart with the
//!   server back publishes all of it.
//! * [`incremental_inserts_are_idempotent`].

mod common;

use std::sync::Mutex;

use camino::Utf8Path;
use common::{
    database_url, load_fixture,
    stream::{engine_config, storage_dir, synthetic_alerts},
};
use fink_fat_engine::{
    engine_config::EngineConfig,
    topocentric_kf::branching::{BranchCollection, read_archived_log},
};
use fink_fat_sql::{incremental, night_sync, rows::ObservationRow, write_bulk_tables};
use fink_fat_stream::{
    AlertRecord, DatabaseConfig, Driver, DriverConfig, DriverStats, ReplaySource, StreamPaths,
};
use postgres::{Client, NoTls};

/// Serializes the tests of this file: they share one database.
static DATABASE: Mutex<()> = Mutex::new(());

/// Alerts per replay batch.
const BATCH_SIZE: usize = 20;

/// The 8 tables both writers load, in a fixed order.
const TABLES: [&str; 8] = [
    "observations",
    "branches",
    "kf_bank",
    "branch_observations",
    "hypotheses",
    "kf_state",
    "branch_best_state",
    "archived_trajectories",
];

fn connect() -> Client {
    Client::connect(&database_url(), NoTls).expect("connect to the test Postgres")
}

/// Drop every table and schema the sink or `convert` create, so each test
/// starts from an empty database.
fn reset_database(client: &mut Client) {
    client
        .batch_execute(
            "DROP SCHEMA IF EXISTS fink_fat_staging CASCADE;
             DROP SCHEMA IF EXISTS fink_fat_retired CASCADE;
             DROP TABLE IF EXISTS branch_best_state, kf_state, hypotheses, branch_observations,
                 kf_bank, branches, archived_trajectories, observations, fink_fat_sync_state
                 CASCADE;",
        )
        .expect("reset the test database");
}

/// Sink configuration pointing at `url`, retrying fast.
fn database(url: &str) -> DatabaseConfig {
    DatabaseConfig {
        url: Some(url.to_string()),
        connect_timeout: 2.0 / 86_400.0,
        flush_backoff_min: 3_600.0 / 86_400.0,
        flush_backoff_max: 3_600.0 / 86_400.0,
        swap_retries: 2,
        swap_retry_delay: 1.0 / 86_400.0,
    }
}

/// Run a whole replay of `alerts` with the sink of `database`.
fn replay(
    alerts: Vec<AlertRecord>,
    database: DatabaseConfig,
    config: &EngineConfig,
) -> DriverStats {
    let ctx = config.build_context();
    let driver_config = DriverConfig {
        database: Some(database),
        ..DriverConfig::default()
    };
    Driver::new(
        ReplaySource::from_records(alerts, BATCH_SIZE),
        driver_config,
        &ctx,
        config,
    )
    .expect("start driver")
    .run()
    .expect("replay")
}

/// Order-independent digest of every row of `table`.
fn digest(client: &mut Client, table: &str) -> (i64, String) {
    let row = client
        .query_one(
            &format!(
                "SELECT count(*), coalesce(md5(string_agg(t::text, '|' ORDER BY t::text)), '') \
                 FROM {table} t"
            ),
            &[],
        )
        .expect("digest table");
    (row.get(0), row.get(1))
}

fn count(client: &mut Client, sql: &str) -> i64 {
    client.query_one(sql, &[]).expect("count").get(0)
}

/// Every `observations` row, as `convert` would read it from a parquet.
fn observation_rows(client: &mut Client) -> Vec<ObservationRow> {
    client
        .query(
            "SELECT id, night_id, object_id, magnitude, mag_err, filter, mpc_code_obs, \
             ra, ra_err, dec, dec_err, mjd_tt FROM observations",
            &[],
        )
        .expect("read observations")
        .iter()
        .map(|r| ObservationRow {
            id: r.get(0),
            night_id: r.get(1),
            object_id: r.get(2),
            magnitude: r.get(3),
            mag_err: r.get(4),
            filter: r.get(5),
            mpc_code_obs: r.get(6),
            ra: r.get(7),
            ra_err: r.get(8),
            dec: r.get(9),
            dec_err: r.get(10),
            mjd_tt: r.get(11),
        })
        .collect()
}

/// The driver's final collection, with its whole archive (as `convert`
/// loads it).
fn final_collection<'a>(
    ctx: &'a fink_fat_engine::engine_config::kalman_context::KalmanContext,
    config: &'a EngineConfig,
) -> BranchCollection<'a, 'a> {
    let paths = StreamPaths::under(config);
    let mut collection = BranchCollection::load_snapshot_from_disk(&paths.snapshot, ctx, config)
        .expect("load snapshot");
    collection.archived = read_archived_log(&paths.archive).expect("read archive");
    collection
}

fn file_len(path: &Utf8Path) -> u64 {
    std::fs::metadata(path).map_or(0, |m| m.len())
}

#[test]
fn a_replay_publishes_what_convert_would_load() {
    let _database = DATABASE.lock().unwrap_or_else(|e| e.into_inner());
    let (tmp, obs_dataset, _) = load_fixture();
    let alerts = synthetic_alerts(&obs_dataset);
    let config = engine_config(&storage_dir(&tmp, "stream"));
    let mut client = connect();
    reset_database(&mut client);

    let stats = replay(alerts.clone(), database(&database_url()), &config);
    assert_eq!(stats.n_accepted, alerts.len());
    assert_eq!(stats.n_flush_failures, 0);
    assert_eq!(stats.n_sync_failures, 0);
    assert_eq!(stats.n_nights_synced, stats.n_nights_closed);
    assert_eq!(stats.n_observations_inserted, alerts.len() as u64);
    assert!(
        stats.n_nights_closed > 1,
        "the swap must have run on a non-empty public schema"
    );
    assert_eq!(file_len(&StreamPaths::under(&config).sql_outbox), 0);

    let ctx = config.build_context();
    let collection = final_collection(&ctx, &config);
    let last_synced = night_sync::last_synced_night(&mut client).expect("sync state");
    assert_eq!(
        last_synced,
        collection.last_closed_night_id.map(|n| i64::from(n.0))
    );
    assert_eq!(
        count(
            &mut client,
            "SELECT count(*) FROM pg_tables \
             WHERE schemaname IN ('fink_fat_staging', 'fink_fat_retired')"
        ),
        0,
        "the swap leaves no private table behind"
    );
    for (table, constraint) in [
        ("kf_bank", "kf_bank_branch_id_fkey"),
        ("branch_observations", "branch_observations_obs_id_fkey"),
        ("kf_state", "kf_state_hypothesis_id_fkey"),
    ] {
        assert_eq!(
            count(
                &mut client,
                &format!(
                    "SELECT count(*) FROM pg_constraint \
                     WHERE conname = '{constraint}' AND conrelid = 'public.{table}'::regclass"
                )
            ),
            1,
            "{constraint} on public.{table}"
        );
    }
    assert_eq!(
        count(
            &mut client,
            "SELECT count(*) FROM pg_indexes WHERE schemaname = 'public' AND indexname IN \
             ('idx_branches_lineage_designation', 'idx_branches_lineage_id', \
              'idx_hypotheses_branch_log_weight', 'idx_branch_observations_obs_id', \
              'idx_observations_object_id')"
        ),
        5
    );

    // The same state loaded by `convert` must give the same tables.
    let streamed: Vec<_> = TABLES.iter().map(|t| digest(&mut client, t)).collect();
    let observations = observation_rows(&mut client);
    write_bulk_tables(&collection, &observations, &database_url()).expect("convert");
    let converted: Vec<_> = TABLES.iter().map(|t| digest(&mut client, t)).collect();
    for ((table, streamed), converted) in TABLES.iter().zip(&streamed).zip(&converted) {
        assert_eq!(streamed, converted, "table {table}: stream vs convert");
    }
}

#[test]
fn a_database_outage_is_caught_up_at_restart() {
    let _database = DATABASE.lock().unwrap_or_else(|e| e.into_inner());
    let (tmp, obs_dataset, _) = load_fixture();
    let alerts = synthetic_alerts(&obs_dataset);
    let config = engine_config(&storage_dir(&tmp, "outage"));
    let mut client = connect();
    reset_database(&mut client);

    // Nothing listens on port 1: every connection is refused.
    let unreachable = "postgres://finkfat:finkfat@127.0.0.1:1/finkfat_test";
    let down = replay(alerts.clone(), database(unreachable), &config);
    assert!(down.n_flush_failures > 0);
    assert_eq!(down.n_nights_synced, 0);
    assert_eq!(down.n_sync_failures, down.n_nights_closed);
    let outbox = StreamPaths::under(&config).sql_outbox;
    assert!(file_len(&outbox) > 0, "everything waits in the outbox");

    // Restart with the server back and nothing new to replay: start-up
    // flushes the outbox and publishes the snapshot.
    let up = replay(Vec::new(), database(&database_url()), &config);
    assert_eq!(up.n_nights_synced, 1);
    assert_eq!(up.n_observations_inserted, alerts.len() as u64);
    assert_eq!(file_len(&outbox), 0);

    let ctx = config.build_context();
    let collection = final_collection(&ctx, &config);
    assert_eq!(
        count(&mut client, "SELECT count(*) FROM observations"),
        alerts.len() as i64
    );
    assert_eq!(
        count(&mut client, "SELECT count(*) FROM branches"),
        collection.branches.len() as i64
    );
    assert_eq!(
        count(&mut client, "SELECT count(*) FROM archived_trajectories"),
        collection.archived.len() as i64
    );
    assert_eq!(
        night_sync::last_synced_night(&mut client).expect("sync state"),
        collection.last_closed_night_id.map(|n| i64::from(n.0))
    );

    // Already up to date: a further restart publishes nothing.
    let again = replay(Vec::new(), database(&database_url()), &config);
    assert_eq!(again.n_nights_synced, 0);
}

#[test]
fn incremental_inserts_are_idempotent() {
    let _database = DATABASE.lock().unwrap_or_else(|e| e.into_inner());
    let mut client = connect();
    reset_database(&mut client);
    let mut transaction = client.transaction().expect("transaction");
    fink_fat_sql::schema::create_tables(&mut transaction).expect("create tables");

    let row = |id: i64| ObservationRow {
        id,
        night_id: 61_344,
        object_id: id.to_string(),
        magnitude: 20.0,
        mag_err: 0.1,
        filter: 2,
        mpc_code_obs: "X05".into(),
        ra: 1.0,
        ra_err: 1e-6,
        dec: -0.5,
        dec_err: 1e-6,
        mjd_tt: 61_345.1,
    };
    let first = incremental::insert_observations(&mut transaction, &[row(1), row(2), row(2)])
        .expect("first insert");
    let second = incremental::insert_observations(&mut transaction, &[row(2), row(3)])
        .expect("second insert");
    assert_eq!((first, second), (2, 1));
    let total: i64 = transaction
        .query_one("SELECT count(*) FROM observations", &[])
        .expect("count")
        .get(0);
    assert_eq!(total, 3);
}
