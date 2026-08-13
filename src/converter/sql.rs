//! Load a [`BranchCollection`] into Postgres, using the same 6-table schema
//! as [`crate::converter::parquet`] (`branches`/`kf_bank`/
//! `branch_observations`/`hypotheses`/`kf_state`/`archived_trajectories`,
//! joined by `branch_id`/`hypothesis_id`/`designation`).
//!
//! Deliberately re-walks `&BranchCollection` with the same shape of loop as
//! `parquet::to_dataframes` rather than reading back the already-built
//! Polars `DataFrame`s: extracting typed values out of `ChunkedArray`/
//! `ListChunked` just to re-serialize them for `COPY` would be a pointless
//! extra materialization pass for a one-shot batch export. `u64`/`usize`
//! fields are cast to `i64` throughout — Postgres has no unsigned integer
//! type.
//!
//! No TLS: `Client::connect` is called with [`NoTls`], suitable for a local/
//! trusted Postgres instance. Add TLS support if this ever talks to a
//! non-local database.

use fink_fat_engine::topocentric_kf::{
    branching::BranchCollection, single_kalman::KFStateSnapshot,
};
use indicatif::{ProgressBar, ProgressStyle};
use nalgebra::Vector3;
use postgres::{Client, NoTls, binary_copy::BinaryCopyInWriter, types::Type};

use crate::converter::family::{DynamicalFamily, classify_from_attributable_state};

/// Columns shared by the `kf_state` and `archived_trajectories` tables,
/// mirroring [`crate::converter::parquet::KfStateColumns`] but as a
/// per-row struct instead of per-column vectors, since `COPY` writes one
/// row at a time.
struct KfStateFields {
    ra: f64,
    dec: f64,
    ra_dot: f64,
    dec_dot: f64,
    rho: f64,
    rho_dot: f64,
    covariance: Vec<f64>,
    epoch: f64,
    r_obs_x: f64,
    r_obs_y: f64,
    r_obs_z: f64,
    v_obs_x: f64,
    v_obs_y: f64,
    v_obs_z: f64,
    universal_anomaly: Option<f64>,
    kalman_gain: Option<Vec<f64>>,
    nis_ema: Option<f64>,
}

impl KfStateFields {
    fn from_snapshot(kf: &KFStateSnapshot) -> Self {
        Self {
            ra: kf.state[0],
            dec: kf.state[1],
            ra_dot: kf.state[2],
            dec_dot: kf.state[3],
            rho: kf.state[4],
            rho_dot: kf.state[5],
            covariance: kf.covariance.to_vec(),
            epoch: kf.epoch,
            r_obs_x: kf.r_obs[0],
            r_obs_y: kf.r_obs[1],
            r_obs_z: kf.r_obs[2],
            v_obs_x: kf.v_obs[0],
            v_obs_y: kf.v_obs[1],
            v_obs_z: kf.v_obs[2],
            universal_anomaly: kf.universal_anomaly,
            kalman_gain: kf.kalman_gain.map(|g| g.to_vec()),
            nis_ema: kf.nis_ema,
        }
    }
}

const KF_STATE_COLUMN_NAMES: &str = "ra, dec, ra_dot, dec_dot, rho, rho_dot, covariance, epoch, \
     r_obs_x, r_obs_y, r_obs_z, v_obs_x, v_obs_y, v_obs_z, universal_anomaly, kalman_gain, nis_ema";

const KF_STATE_COLUMN_TYPES: &[Type] = &[
    Type::FLOAT8,
    Type::FLOAT8,
    Type::FLOAT8,
    Type::FLOAT8,
    Type::FLOAT8,
    Type::FLOAT8,
    Type::FLOAT8_ARRAY,
    Type::FLOAT8,
    Type::FLOAT8,
    Type::FLOAT8,
    Type::FLOAT8,
    Type::FLOAT8,
    Type::FLOAT8,
    Type::FLOAT8,
    Type::FLOAT8,
    Type::FLOAT8_ARRAY,
    Type::FLOAT8,
];

struct BranchRow {
    branch_id: i64,
    lineage_id: i64,
    parent_branch_id: i64,
    ancestor_at_scan_horizon: i64,
    ancestor_creation_step: i64,
    last_real_update_step: i64,
    n_real_updates: i64,
    cumulative_llr: f64,
    lineage_designation: String,
    designation: String,
}

struct KfBankRow {
    branch_id: i64,
    n_steps: i64,
    absolute_magnitude_estimate: Option<f64>,
    absolute_magnitude_sample_count: i32,
}

struct BranchObservationRow {
    branch_id: i64,
    position: i32,
    obs_id: i64,
}

struct HypothesisRow {
    hypothesis_id: i64,
    branch_id: i64,
    local_hyp_id: i64,
    log_weight: f64,
    recent_log_liks: Vec<f64>,
}

struct KfStateRow {
    hypothesis_id: i64,
    fields: KfStateFields,
}

struct ArchivedRow {
    designation: String,
    lineage_id: i64,
    track_ids: Vec<i64>,
    cumulative_llr: f64,
    n_real_updates: i64,
    last_real_update_step: i64,
    archived_at_step: i64,
    absolute_magnitude_estimate: Option<f64>,
    absolute_magnitude_sample_count: i32,
    kf_state: KfStateFields,
}

/// Build the progress bar shown while [`write_sql_tables`] runs: one tick
/// per row `COPY`'d, with `pb.println(...)` used by the caller for the
/// coarser stages (connect/DDL/truncate/commit) that aren't row-counted.
fn build_progress_bar(total_rows: usize) -> ProgressBar {
    let pb = ProgressBar::new(total_rows as u64);
    pb.set_style(
        ProgressStyle::with_template("{bar:40.cyan/blue} {pos}/{len} rows  {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    pb
}

/// Load `branch_collection` into the Postgres database at `database_url`:
/// create the 6 tables if absent, `TRUNCATE` them, then bulk-load via
/// `COPY ... FROM STDIN BINARY`. Runs in a single transaction — a failure
/// partway through leaves the previous contents untouched.
pub fn write_sql_tables(
    branch_collection: &BranchCollection,
    database_url: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let (branch_rows, kf_bank_rows, branch_observation_rows, hypothesis_rows, kf_state_rows) =
        build_branch_rows(branch_collection);
    let archived_rows = build_archived_rows(branch_collection);

    let total_rows = branch_rows.len()
        + kf_bank_rows.len()
        + branch_observation_rows.len()
        + hypothesis_rows.len()
        + kf_state_rows.len()
        + archived_rows.len();
    let pb = build_progress_bar(total_rows);

    // `database_url` may embed a plaintext password (`postgres://user:pw@host/db`)
    // — never logged, even at this generic "connecting" granularity.
    pb.println("Connecting to Postgres...");
    let mut client = Client::connect(database_url, NoTls)?;
    let mut transaction = client.transaction()?;

    pb.println("Creating tables (CREATE TABLE IF NOT EXISTS)...");
    create_tables(&mut transaction)?;

    pb.println("Truncating existing tables...");
    transaction.batch_execute(
        "TRUNCATE TABLE branches, kf_bank, branch_observations, hypotheses, kf_state, \
         archived_trajectories CASCADE;",
    )?;

    copy_branches(&mut transaction, &branch_rows, &pb)?;
    copy_kf_bank(&mut transaction, &kf_bank_rows, &pb)?;
    copy_branch_observations(&mut transaction, &branch_observation_rows, &pb)?;
    copy_hypotheses(&mut transaction, &hypothesis_rows, &pb)?;
    copy_kf_state(&mut transaction, &kf_state_rows, &pb)?;
    copy_archived_trajectories(&mut transaction, &archived_rows, &pb)?;

    pb.println("Committing transaction...");
    transaction.commit()?;
    pb.finish_with_message("SQL export complete");
    Ok(())
}

#[allow(clippy::type_complexity)]
fn build_branch_rows(
    branch_collection: &BranchCollection,
) -> (
    Vec<BranchRow>,
    Vec<KfBankRow>,
    Vec<BranchObservationRow>,
    Vec<HypothesisRow>,
    Vec<KfStateRow>,
) {
    let n_branches = branch_collection.branches.len();
    let n_track_id_rows: usize = branch_collection
        .branches
        .iter()
        .map(|b| b.bank.track_ids().len())
        .sum();
    let n_hypotheses: usize = branch_collection
        .branches
        .iter()
        .map(|b| b.bank.hypotheses().len())
        .sum();

    let mut branch_rows = Vec::with_capacity(n_branches);
    let mut kf_bank_rows = Vec::with_capacity(n_branches);
    let mut branch_observation_rows = Vec::with_capacity(n_track_id_rows);
    let mut hypothesis_rows = Vec::with_capacity(n_hypotheses);
    let mut kf_state_rows = Vec::with_capacity(n_hypotheses);

    let mut next_hypothesis_id: i64 = 0;

    for branch in &branch_collection.branches {
        let branch_id = branch.branch_id as i64;

        branch_rows.push(BranchRow {
            branch_id,
            lineage_id: branch.lineage_id as i64,
            parent_branch_id: branch.parent_branch_id as i64,
            ancestor_at_scan_horizon: branch.ancestor_at_scan_horizon as i64,
            ancestor_creation_step: branch.ancestor_creation_step as i64,
            last_real_update_step: branch.last_real_update_step as i64,
            n_real_updates: branch.n_real_updates as i64,
            cumulative_llr: branch.cumulative_llr,
            lineage_designation: branch.lineage_designation.to_string(),
            designation: branch.designation().to_string(),
        });

        let bank_snapshot = branch.bank.to_snapshot();

        kf_bank_rows.push(KfBankRow {
            branch_id,
            n_steps: bank_snapshot.n_steps as i64,
            absolute_magnitude_estimate: bank_snapshot.absolute_magnitude_estimate,
            absolute_magnitude_sample_count: bank_snapshot.absolute_magnitude_sample_count as i32,
        });

        for (position, id) in bank_snapshot.track_ids.iter().enumerate() {
            branch_observation_rows.push(BranchObservationRow {
                branch_id,
                position: position as i32,
                obs_id: *id as i64,
            });
        }

        for hyp_snapshot in bank_snapshot.hypotheses {
            let hypothesis_id = next_hypothesis_id;
            next_hypothesis_id += 1;

            hypothesis_rows.push(HypothesisRow {
                hypothesis_id,
                branch_id,
                local_hyp_id: hyp_snapshot.id as i64,
                log_weight: hyp_snapshot.log_weight,
                recent_log_liks: hyp_snapshot.recent_log_liks,
            });

            kf_state_rows.push(KfStateRow {
                hypothesis_id,
                fields: KfStateFields::from_snapshot(&hyp_snapshot.kf),
            });
        }
    }

    (
        branch_rows,
        kf_bank_rows,
        branch_observation_rows,
        hypothesis_rows,
        kf_state_rows,
    )
}

fn build_archived_rows(branch_collection: &BranchCollection) -> Vec<ArchivedRow> {
    branch_collection
        .archived
        .iter()
        .map(|trajectory| ArchivedRow {
            designation: trajectory.designation.to_string(),
            lineage_id: trajectory.lineage_id as i64,
            track_ids: trajectory.track_ids.iter().map(|id| *id as i64).collect(),
            cumulative_llr: trajectory.cumulative_llr,
            n_real_updates: trajectory.n_real_updates as i64,
            last_real_update_step: trajectory.last_real_update_step as i64,
            archived_at_step: trajectory.archived_at_step as i64,
            absolute_magnitude_estimate: trajectory.absolute_magnitude_estimate,
            absolute_magnitude_sample_count: trajectory.absolute_magnitude_sample_count as i32,
            kf_state: KfStateFields::from_snapshot(&trajectory.map_state),
        })
        .collect()
}

fn create_tables(transaction: &mut postgres::Transaction<'_>) -> Result<(), postgres::Error> {
    transaction.batch_execute(
        "
        CREATE TABLE IF NOT EXISTS branches (
            branch_id BIGINT PRIMARY KEY,
            lineage_id BIGINT NOT NULL,
            parent_branch_id BIGINT NOT NULL,
            ancestor_at_scan_horizon BIGINT NOT NULL,
            ancestor_creation_step BIGINT NOT NULL,
            last_real_update_step BIGINT NOT NULL,
            n_real_updates BIGINT NOT NULL,
            cumulative_llr DOUBLE PRECISION NOT NULL,
            lineage_designation TEXT NOT NULL,
            designation TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS kf_bank (
            branch_id BIGINT PRIMARY KEY REFERENCES branches(branch_id),
            n_steps BIGINT NOT NULL,
            absolute_magnitude_estimate DOUBLE PRECISION,
            absolute_magnitude_sample_count INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS branch_observations (
            branch_id BIGINT NOT NULL REFERENCES branches(branch_id),
            position INTEGER NOT NULL,
            obs_id BIGINT NOT NULL,
            PRIMARY KEY (branch_id, position)
        );

        CREATE TABLE IF NOT EXISTS hypotheses (
            hypothesis_id BIGINT PRIMARY KEY,
            branch_id BIGINT NOT NULL REFERENCES branches(branch_id),
            local_hyp_id BIGINT NOT NULL,
            log_weight DOUBLE PRECISION NOT NULL,
            recent_log_liks DOUBLE PRECISION[] NOT NULL
        );

        CREATE TABLE IF NOT EXISTS kf_state (
            hypothesis_id BIGINT PRIMARY KEY REFERENCES hypotheses(hypothesis_id),
            ra DOUBLE PRECISION NOT NULL,
            dec DOUBLE PRECISION NOT NULL,
            ra_dot DOUBLE PRECISION NOT NULL,
            dec_dot DOUBLE PRECISION NOT NULL,
            rho DOUBLE PRECISION NOT NULL,
            rho_dot DOUBLE PRECISION NOT NULL,
            covariance DOUBLE PRECISION[] NOT NULL,
            epoch DOUBLE PRECISION NOT NULL,
            r_obs_x DOUBLE PRECISION NOT NULL,
            r_obs_y DOUBLE PRECISION NOT NULL,
            r_obs_z DOUBLE PRECISION NOT NULL,
            v_obs_x DOUBLE PRECISION NOT NULL,
            v_obs_y DOUBLE PRECISION NOT NULL,
            v_obs_z DOUBLE PRECISION NOT NULL,
            universal_anomaly DOUBLE PRECISION,
            kalman_gain DOUBLE PRECISION[],
            nis_ema DOUBLE PRECISION,
            dynamic_family TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS archived_trajectories (
            designation TEXT PRIMARY KEY,
            lineage_id BIGINT NOT NULL,
            track_ids BIGINT[] NOT NULL,
            cumulative_llr DOUBLE PRECISION NOT NULL,
            n_real_updates BIGINT NOT NULL,
            last_real_update_step BIGINT NOT NULL,
            archived_at_step BIGINT NOT NULL,
            absolute_magnitude_estimate DOUBLE PRECISION,
            absolute_magnitude_sample_count INTEGER NOT NULL,
            ra DOUBLE PRECISION NOT NULL,
            dec DOUBLE PRECISION NOT NULL,
            ra_dot DOUBLE PRECISION NOT NULL,
            dec_dot DOUBLE PRECISION NOT NULL,
            rho DOUBLE PRECISION NOT NULL,
            rho_dot DOUBLE PRECISION NOT NULL,
            covariance DOUBLE PRECISION[] NOT NULL,
            epoch DOUBLE PRECISION NOT NULL,
            r_obs_x DOUBLE PRECISION NOT NULL,
            r_obs_y DOUBLE PRECISION NOT NULL,
            r_obs_z DOUBLE PRECISION NOT NULL,
            v_obs_x DOUBLE PRECISION NOT NULL,
            v_obs_y DOUBLE PRECISION NOT NULL,
            v_obs_z DOUBLE PRECISION NOT NULL,
            universal_anomaly DOUBLE PRECISION,
            kalman_gain DOUBLE PRECISION[],
            nis_ema DOUBLE PRECISION,
            dynamic_family TEXT NOT NULL,
            semi_major_axis DOUBLE PRECISION NOT NULL,
            eccentricity DOUBLE PRECISION NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_hypotheses_branch_log_weight
            ON hypotheses (branch_id, log_weight DESC);

        -- Columns added after the initial rollout: CREATE TABLE IF NOT EXISTS
        -- above is a no-op against a pre-existing table, so a DB created
        -- before these columns existed needs them backfilled explicitly.
        -- The DEFAULT satisfies NOT NULL on any existing rows; the table is
        -- TRUNCATEd right after create_tables() runs, so the default values
        -- never actually get read back out.
        ALTER TABLE kf_state ADD COLUMN IF NOT EXISTS dynamic_family TEXT NOT NULL DEFAULT 'Unknown';
        ALTER TABLE kf_state ADD COLUMN IF NOT EXISTS semi_major_axis DOUBLE PRECISION NOT NULL DEFAULT 0;
        ALTER TABLE kf_state ADD COLUMN IF NOT EXISTS eccentricity DOUBLE PRECISION NOT NULL DEFAULT 0;
        ALTER TABLE archived_trajectories ADD COLUMN IF NOT EXISTS dynamic_family TEXT NOT NULL DEFAULT 'Unknown';
        ALTER TABLE archived_trajectories ADD COLUMN IF NOT EXISTS semi_major_axis DOUBLE PRECISION NOT NULL DEFAULT 0;
        ALTER TABLE archived_trajectories ADD COLUMN IF NOT EXISTS eccentricity DOUBLE PRECISION NOT NULL DEFAULT 0;
        ",
    )
}

fn copy_branches(
    transaction: &mut postgres::Transaction<'_>,
    rows: &[BranchRow],
    pb: &ProgressBar,
) -> Result<(), postgres::Error> {
    pb.println(format!("Copying {} rows into branches...", rows.len()));
    let sink = transaction.copy_in(
        "COPY branches (branch_id, lineage_id, parent_branch_id, ancestor_at_scan_horizon, \
         ancestor_creation_step, last_real_update_step, n_real_updates, cumulative_llr, \
         lineage_designation, designation) FROM STDIN BINARY",
    )?;
    let mut writer = BinaryCopyInWriter::new(
        sink,
        &[
            Type::INT8,
            Type::INT8,
            Type::INT8,
            Type::INT8,
            Type::INT8,
            Type::INT8,
            Type::INT8,
            Type::FLOAT8,
            Type::TEXT,
            Type::TEXT,
        ],
    );
    for row in rows {
        writer.write(&[
            &row.branch_id,
            &row.lineage_id,
            &row.parent_branch_id,
            &row.ancestor_at_scan_horizon,
            &row.ancestor_creation_step,
            &row.last_real_update_step,
            &row.n_real_updates,
            &row.cumulative_llr,
            &row.lineage_designation,
            &row.designation,
        ])?;
        pb.inc(1);
    }
    writer.finish()?;
    Ok(())
}

fn copy_kf_bank(
    transaction: &mut postgres::Transaction<'_>,
    rows: &[KfBankRow],
    pb: &ProgressBar,
) -> Result<(), postgres::Error> {
    pb.println(format!("Copying {} rows into kf_bank...", rows.len()));
    let sink = transaction.copy_in(
        "COPY kf_bank (branch_id, n_steps, absolute_magnitude_estimate, \
         absolute_magnitude_sample_count) FROM STDIN BINARY",
    )?;
    let mut writer =
        BinaryCopyInWriter::new(sink, &[Type::INT8, Type::INT8, Type::FLOAT8, Type::INT4]);
    for row in rows {
        writer.write(&[
            &row.branch_id,
            &row.n_steps,
            &row.absolute_magnitude_estimate,
            &row.absolute_magnitude_sample_count,
        ])?;
        pb.inc(1);
    }
    writer.finish()?;
    Ok(())
}

fn copy_branch_observations(
    transaction: &mut postgres::Transaction<'_>,
    rows: &[BranchObservationRow],
    pb: &ProgressBar,
) -> Result<(), postgres::Error> {
    pb.println(format!(
        "Copying {} rows into branch_observations...",
        rows.len()
    ));
    let sink = transaction
        .copy_in("COPY branch_observations (branch_id, position, obs_id) FROM STDIN BINARY")?;
    let mut writer = BinaryCopyInWriter::new(sink, &[Type::INT8, Type::INT4, Type::INT8]);
    for row in rows {
        writer.write(&[&row.branch_id, &row.position, &row.obs_id])?;
        pb.inc(1);
    }
    writer.finish()?;
    Ok(())
}

fn copy_hypotheses(
    transaction: &mut postgres::Transaction<'_>,
    rows: &[HypothesisRow],
    pb: &ProgressBar,
) -> Result<(), postgres::Error> {
    pb.println(format!("Copying {} rows into hypotheses...", rows.len()));
    let sink = transaction.copy_in(
        "COPY hypotheses (hypothesis_id, branch_id, local_hyp_id, log_weight, recent_log_liks) \
         FROM STDIN BINARY",
    )?;
    let mut writer = BinaryCopyInWriter::new(
        sink,
        &[
            Type::INT8,
            Type::INT8,
            Type::INT8,
            Type::FLOAT8,
            Type::FLOAT8_ARRAY,
        ],
    );
    for row in rows {
        writer.write(&[
            &row.hypothesis_id,
            &row.branch_id,
            &row.local_hyp_id,
            &row.log_weight,
            &row.recent_log_liks,
        ])?;
        pb.inc(1);
    }
    writer.finish()?;
    Ok(())
}

fn copy_kf_state(
    transaction: &mut postgres::Transaction<'_>,
    rows: &[KfStateRow],
    pb: &ProgressBar,
) -> Result<(), postgres::Error> {
    pb.println(format!("Copying {} rows into kf_state...", rows.len()));
    let sink = transaction.copy_in(&format!(
        "COPY kf_state (hypothesis_id, {KF_STATE_COLUMN_NAMES}, dynamic_family, semi_major_axis, eccentricity) FROM STDIN BINARY"
    ))?;

    let mut types = vec![Type::INT8];
    types.extend_from_slice(KF_STATE_COLUMN_TYPES);
    types.push(Type::TEXT);
    types.push(Type::FLOAT8);
    types.push(Type::FLOAT8);

    let mut writer = BinaryCopyInWriter::new(sink, &types);
    for row in rows {
        let f = &row.fields;

        let (dyn_family, semi_major, eccentricity) = classify_from_attributable_state(
            f.ra,
            f.dec,
            f.ra_dot,
            f.dec_dot,
            f.rho,
            f.rho_dot,
            f.epoch,
            Vector3::new(f.r_obs_x, f.r_obs_y, f.r_obs_z),
            Vector3::new(f.v_obs_x, f.v_obs_y, f.v_obs_z),
        )
        .unwrap_or((DynamicalFamily::Unknown, 0., 0.));

        writer.write(&[
            &row.hypothesis_id,
            &f.ra,
            &f.dec,
            &f.ra_dot,
            &f.dec_dot,
            &f.rho,
            &f.rho_dot,
            &f.covariance,
            &f.epoch,
            &f.r_obs_x,
            &f.r_obs_y,
            &f.r_obs_z,
            &f.v_obs_x,
            &f.v_obs_y,
            &f.v_obs_z,
            &f.universal_anomaly,
            &f.kalman_gain,
            &f.nis_ema,
            &dyn_family.label(),
            &semi_major,
            &eccentricity,
        ])?;
        pb.inc(1);
    }
    writer.finish()?;
    Ok(())
}

fn copy_archived_trajectories(
    transaction: &mut postgres::Transaction<'_>,
    rows: &[ArchivedRow],
    pb: &ProgressBar,
) -> Result<(), postgres::Error> {
    pb.println(format!(
        "Copying {} rows into archived_trajectories...",
        rows.len()
    ));
    let sink = transaction.copy_in(&format!(
        "COPY archived_trajectories (designation, lineage_id, track_ids, cumulative_llr, \
         n_real_updates, last_real_update_step, archived_at_step, absolute_magnitude_estimate, \
         absolute_magnitude_sample_count, {KF_STATE_COLUMN_NAMES}, dynamic_family, \
         semi_major_axis, eccentricity) FROM STDIN BINARY"
    ))?;
    let mut types = vec![
        Type::TEXT,
        Type::INT8,
        Type::INT8_ARRAY,
        Type::FLOAT8,
        Type::INT8,
        Type::INT8,
        Type::INT8,
        Type::FLOAT8,
        Type::INT4,
    ];
    types.extend_from_slice(KF_STATE_COLUMN_TYPES);
    types.push(Type::TEXT);
    types.push(Type::FLOAT8);
    types.push(Type::FLOAT8);
    let mut writer = BinaryCopyInWriter::new(sink, &types);
    for row in rows {
        let f = &row.kf_state;

        let (dyn_family, semi_major, eccentricity) = classify_from_attributable_state(
            f.ra,
            f.dec,
            f.ra_dot,
            f.dec_dot,
            f.rho,
            f.rho_dot,
            f.epoch,
            Vector3::new(f.r_obs_x, f.r_obs_y, f.r_obs_z),
            Vector3::new(f.v_obs_x, f.v_obs_y, f.v_obs_z),
        )
        .unwrap_or((DynamicalFamily::Unknown, 0., 0.));

        writer.write(&[
            &row.designation,
            &row.lineage_id,
            &row.track_ids,
            &row.cumulative_llr,
            &row.n_real_updates,
            &row.last_real_update_step,
            &row.archived_at_step,
            &row.absolute_magnitude_estimate,
            &row.absolute_magnitude_sample_count,
            &f.ra,
            &f.dec,
            &f.ra_dot,
            &f.dec_dot,
            &f.rho,
            &f.rho_dot,
            &f.covariance,
            &f.epoch,
            &f.r_obs_x,
            &f.r_obs_y,
            &f.r_obs_z,
            &f.v_obs_x,
            &f.v_obs_y,
            &f.v_obs_z,
            &f.universal_anomaly,
            &f.kalman_gain,
            &f.nis_ema,
            &dyn_family.label(),
            &semi_major,
            &eccentricity,
        ])?;
        pb.inc(1);
    }
    writer.finish()?;
    Ok(())
}
