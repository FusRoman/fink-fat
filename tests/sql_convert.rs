//! Black-box integration test for `fink-fat convert --format sql`.
//!
//! Same pattern as `tests/reconstruction.rs`: `fink-fat` is a binary-only
//! crate (no `src/lib.rs`), so a test in `tests/` cannot call
//! `converter::sql::write_sql_tables` directly — it invokes the compiled
//! binary via [`assert_cmd`] instead, then inspects the result with its own,
//! independent Postgres connection.
//!
//! **Requires a running Postgres** and **destroys the contents of its 7
//! bulk-loaded tables** (`write_sql_tables` always `TRUNCATE`s them). CI
//! provides a throwaway instance for this (see the `sql-integration-test`
//! job in `.github/workflows/CI.yml`); locally, use the repo's
//! `docker-compose.yml`:
//!
//! ```text
//! docker compose up -d db
//! cargo test --test sql_convert
//! ```
//!
//! Override `DATABASE_URL` in the environment to point at a different
//! instance; defaults to the `docker-compose.yml` `db` service reached from
//! the host.

use std::path::PathBuf;

use assert_cmd::Command;
use postgres::{Client, NoTls};
use tempfile::TempDir;

const CONFIG_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/config.yaml");
const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");

/// The single night fixture this test tracks and then converts. Any one
/// `tests/data/night_id_<N>.parquet` file works as both the `track` input
/// and the `convert --path-observation` input: it's already shaped like the
/// candid-keyed alert parquet `read_observation_rows` expects (see
/// `src/converter/sql/observations.rs`).
fn first_night_path() -> PathBuf {
    let mut nights: Vec<PathBuf> = std::fs::read_dir(DATA_DIR)
        .expect("read tests/data dir")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("parquet"))
        .collect();
    nights.sort();
    nights
        .into_iter()
        .next()
        .expect("tests/data should contain at least one night fixture")
}

/// Connection string for the Postgres this test converts into. Defaults to
/// the `db` service in the repo's `docker-compose.yml`, reached from the
/// host (`docker compose up -d db` exposes it on `localhost:5432`).
fn database_url() -> String {
    std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://finkfat:finkfat@localhost:5432/finkfat_test".to_string())
}

/// Runs `fink-fat track --alerts <alerts_path> --config tests/config.yaml`,
/// asserting success — produces the snapshot + archive log `convert` reads.
fn run_track(alerts_path: &str) {
    Command::cargo_bin(assert_cmd::pkg_name!())
        .expect("binary should build")
        .args(["track", "--alerts", alerts_path, "--config", CONFIG_PATH])
        .assert()
        .success();
}

/// Runs `fink-fat convert --format sql ...` against `database_url`, loading
/// `observations_path` into the `observations` table, asserting success.
fn run_convert_sql(observations_path: &str, database_url: &str) {
    Command::cargo_bin(assert_cmd::pkg_name!())
        .expect("binary should build")
        .args([
            "convert",
            "--config",
            CONFIG_PATH,
            "--format",
            "sql",
            "--database-url",
            database_url,
            "--path-observation",
            observations_path,
        ])
        .assert()
        .success();
}

/// One table's row count and `UNLOGGED` status.
struct TableStatus {
    row_count: i64,
    is_unlogged: bool,
}

fn table_status(client: &mut Client, table: &str) -> TableStatus {
    let row_count: i64 = client
        .query_one(&format!("SELECT count(*) FROM {table}"), &[])
        .unwrap_or_else(|e| panic!("count rows in {table}: {e}"))
        .get(0);
    // `relpersistence`: 'u' = unlogged, 'p' = permanent (logged), 't' = temp.
    let relpersistence: i8 = client
        .query_one(
            "SELECT relpersistence FROM pg_class WHERE relname = $1",
            &[&table],
        )
        .unwrap_or_else(|e| panic!("look up pg_class for {table}: {e}"))
        .get(0);
    TableStatus {
        row_count,
        is_unlogged: relpersistence as u8 as char == 'u',
    }
}

fn constraint_exists(client: &mut Client, conname: &str) -> bool {
    let count: i64 = client
        .query_one(
            "SELECT count(*) FROM pg_constraint WHERE conname = $1",
            &[&conname],
        )
        .expect("query pg_constraint")
        .get(0);
    count > 0
}

#[test]
fn convert_sql_loads_unlogged_tables_with_constraints_restored() {
    let night = first_night_path();
    let night_str = night.to_str().expect("utf8 fixture path");

    let storage_dir = TempDir::new().expect("create storage tempdir");
    // SAFETY: this test is the only one in this binary and does not run
    // concurrently with other tests mutating this env var.
    unsafe {
        std::env::set_var("FINK_FAT__STORAGE_PATH", storage_dir.path());
    }

    run_track(night_str);

    let database_url = database_url();
    run_convert_sql(night_str, &database_url);

    let mut client = Client::connect(&database_url, NoTls).unwrap_or_else(|e| {
        panic!("connect to {database_url}: {e} (is `docker compose up -d db` running?)")
    });

    for table in [
        "observations",
        "branches",
        "kf_bank",
        "branch_observations",
        "hypotheses",
        "kf_state",
        "archived_trajectories",
    ] {
        let status = table_status(&mut client, table);
        assert!(
            status.row_count > 0,
            "{table} should hold at least one row after converting a real night fixture"
        );
        assert!(
            status.is_unlogged,
            "{table} should be UNLOGGED after `convert` (see src/converter/sql/create_tables.sql)"
        );
    }

    // Rebuilt by `restore_bulk_load_constraints.sql` after the COPYs — see
    // `src/converter/sql/schema.rs`.
    for conname in [
        "kf_bank_branch_id_fkey",
        "branch_observations_branch_id_fkey",
        "branch_observations_obs_id_fkey",
        "hypotheses_branch_id_fkey",
        "kf_state_hypothesis_id_fkey",
    ] {
        assert!(
            constraint_exists(&mut client, conname),
            "{conname} should exist after `convert` finishes"
        );
    }

    // `branch_observations`'s primary key (`branch_id, position`) is never
    // dropped during the bulk load (see `schema::drop_bulk_load_constraints`),
    // so it should already have rejected any duplicate had one been written;
    // this just double-checks the invariant end to end.
    let (n_rows, n_distinct): (i64, i64) = {
        let row = client
            .query_one(
                "SELECT count(*), count(DISTINCT (branch_id, position)) FROM branch_observations",
                &[],
            )
            .expect("query branch_observations");
        (row.get(0), row.get(1))
    };
    assert_eq!(
        n_rows, n_distinct,
        "branch_observations should have no duplicate (branch_id, position) pairs"
    );
}
