//! `COPY ... FROM STDIN BINARY` into every table [`super::write_sql_tables`]
//! bulk-loads, one function per table, plus the shared progress bar.
//!
//! Each function follows the same shape: open a `COPY` sink, wrap it in a
//! [`BinaryCopyInWriter`] with an explicit column-type list, write every row,
//! `finish()`. `postgres`'s binary `COPY` protocol requires the client to
//! declare each column's [`Type`] up front (it does not infer types from the
//! target table), which is why every function below pairs a literal column
//! list in its SQL text with a matching literal `Type` array — the two must
//! stay in the same order, there is no way to make the compiler check that.

use indicatif::{ProgressBar, ProgressStyle};
use postgres::{Transaction, binary_copy::BinaryCopyInWriter, types::Type};

use super::build::classify_rows_in_parallel;
use super::rows::{
    ArchivedRow, BranchObservationRow, BranchRow, HypothesisRow, KfBankRow, KfStateFields,
    KfStateRow, ObservationRow,
};

/// Columns shared by the `kf_state` and `archived_trajectories` `COPY`
/// statements — both tables carry a full [`KfStateFields`] plus their own
/// leading/trailing columns (see [`copy_kf_state`]/[`copy_archived_trajectories`]).
const KF_STATE_COLUMN_NAMES: &str = "ra, dec, ra_dot, dec_dot, rho, rho_dot, covariance, epoch, \
     r_obs_x, r_obs_y, r_obs_z, v_obs_x, v_obs_y, v_obs_z, universal_anomaly, kalman_gain, nis_ema";

/// [`Type`] list matching [`KF_STATE_COLUMN_NAMES`], in the same order.
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

/// Builds the progress bar shown while [`super::write_sql_tables`] runs: one
/// tick per row `COPY`'d, with `pb.println(...)` used by the caller for the
/// coarser stages (connect/DDL/truncate/commit) that aren't row-counted.
///
/// # Arguments
///
/// * `total_rows` — the sum of every table's row count for this run.
///
/// # Returns
///
/// A ready-to-use [`ProgressBar`] with its length set to `total_rows`.
pub(super) fn build_progress_bar(total_rows: usize) -> ProgressBar {
    let pb = ProgressBar::new(total_rows as u64);
    pb.set_style(
        ProgressStyle::with_template("{bar:40.cyan/blue} {pos}/{len} rows  {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    pb
}

/// Bulk-loads `rows` into `observations`.
///
/// Must run before [`copy_branch_observations`] within the same transaction,
/// so that `branch_observations.obs_id`'s foreign key into `observations(id)`
/// is satisfied once it's rebuilt after the load (see
/// [`super::schema::restore_bulk_load_constraints`]).
///
/// # Arguments
///
/// * `transaction` — the transaction [`super::write_sql_tables`] runs the
///   whole conversion in.
/// * `rows` — every row to insert.
/// * `pb` — ticked once per row.
///
/// # Errors
///
/// Any Postgres error while opening the `COPY` sink, writing a row, or
/// finishing the writer, as a [`postgres::Error`].
pub(super) fn copy_observations(
    transaction: &mut Transaction<'_>,
    rows: &[ObservationRow],
    pb: &ProgressBar,
) -> Result<(), postgres::Error> {
    pb.println(format!("Copying {} rows into observations...", rows.len()));
    let sink = transaction.copy_in(
        "COPY observations (id, night_id, object_id, magnitude, mag_err, filter, \
         mpc_code_obs, ra, ra_err, dec, dec_err, mjd_tt) FROM STDIN BINARY",
    )?;
    let mut writer = BinaryCopyInWriter::new(
        sink,
        &[
            Type::INT8,
            Type::INT8,
            Type::TEXT,
            Type::FLOAT8,
            Type::FLOAT8,
            Type::INT2,
            Type::TEXT,
            Type::FLOAT8,
            Type::FLOAT8,
            Type::FLOAT8,
            Type::FLOAT8,
            Type::FLOAT8,
        ],
    );
    for row in rows {
        writer.write(&[
            &row.id,
            &row.night_id,
            &row.object_id,
            &row.magnitude,
            &row.mag_err,
            &row.filter,
            &row.mpc_code_obs,
            &row.ra,
            &row.ra_err,
            &row.dec,
            &row.dec_err,
            &row.mjd_tt,
        ])?;
        pb.inc(1);
    }
    writer.finish()?;
    Ok(())
}

/// Bulk-loads `rows` into `branches`.
///
/// # Arguments
///
/// * `transaction` — the transaction [`super::write_sql_tables`] runs the
///   whole conversion in.
/// * `rows` — every row to insert.
/// * `pb` — ticked once per row.
///
/// # Errors
///
/// Any Postgres error while opening the `COPY` sink, writing a row, or
/// finishing the writer, as a [`postgres::Error`].
pub(super) fn copy_branches(
    transaction: &mut Transaction<'_>,
    rows: &[BranchRow],
    pb: &ProgressBar,
) -> Result<(), postgres::Error> {
    pb.println(format!("Copying {} rows into branches...", rows.len()));
    let sink = transaction.copy_in(
        "COPY branches (branch_id, lineage_id, parent_branch_id, ancestor_at_scan_horizon, \
         ancestor_creation_step, last_real_update_step, n_real_updates, cumulative_llr, \
         lineage_designation, designation, arc_length_days, n_nights, \
         median_inter_night_dt_days) FROM STDIN BINARY",
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
            Type::FLOAT8,
            Type::INT8,
            Type::FLOAT8,
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
            &row.arc_length_days,
            &row.n_nights,
            &row.median_inter_night_dt_days,
        ])?;
        pb.inc(1);
    }
    writer.finish()?;
    Ok(())
}

/// Bulk-loads `rows` into `kf_bank`.
///
/// # Arguments
///
/// * `transaction` — the transaction [`super::write_sql_tables`] runs the
///   whole conversion in.
/// * `rows` — every row to insert.
/// * `pb` — ticked once per row.
///
/// # Errors
///
/// Any Postgres error while opening the `COPY` sink, writing a row, or
/// finishing the writer, as a [`postgres::Error`].
pub(super) fn copy_kf_bank(
    transaction: &mut Transaction<'_>,
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

/// Bulk-loads `rows` into `branch_observations`.
///
/// # Arguments
///
/// * `transaction` — the transaction [`super::write_sql_tables`] runs the
///   whole conversion in.
/// * `rows` — every row to insert.
/// * `pb` — ticked once per row.
///
/// # Errors
///
/// Any Postgres error while opening the `COPY` sink, writing a row, or
/// finishing the writer, as a [`postgres::Error`].
pub(super) fn copy_branch_observations(
    transaction: &mut Transaction<'_>,
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

/// Bulk-loads `rows` into `hypotheses`.
///
/// # Arguments
///
/// * `transaction` — the transaction [`super::write_sql_tables`] runs the
///   whole conversion in.
/// * `rows` — every row to insert.
/// * `pb` — ticked once per row.
///
/// # Errors
///
/// Any Postgres error while opening the `COPY` sink, writing a row, or
/// finishing the writer, as a [`postgres::Error`].
pub(super) fn copy_hypotheses(
    transaction: &mut Transaction<'_>,
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

/// Bulk-loads `rows` into `kf_state`.
///
/// Classifies every row's dynamical family in parallel
/// ([`classify_rows_in_parallel`]) before the write loop, since the write
/// loop itself is necessarily sequential (one shared `Transaction`/`COPY`
/// stream).
///
/// # Arguments
///
/// * `transaction` — the transaction [`super::write_sql_tables`] runs the
///   whole conversion in.
/// * `rows` — every row to insert.
/// * `pb` — ticked once per row.
///
/// # Errors
///
/// Any Postgres error while opening the `COPY` sink, writing a row, or
/// finishing the writer, as a [`postgres::Error`].
pub(super) fn copy_kf_state(
    transaction: &mut Transaction<'_>,
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
    let field_refs: Vec<&KfStateFields> = rows.iter().map(|row| &row.fields).collect();
    let classifications = classify_rows_in_parallel(&field_refs);
    for (row, (dyn_family, semi_major, eccentricity)) in rows.iter().zip(classifications) {
        let f = &row.fields;

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

/// Bulk-loads `rows` into `archived_trajectories`.
///
/// Classifies every row's dynamical family in parallel — see
/// [`copy_kf_state`]'s doc comment, same rationale.
///
/// # Arguments
///
/// * `transaction` — the transaction [`super::write_sql_tables`] runs the
///   whole conversion in.
/// * `rows` — every row to insert.
/// * `pb` — ticked once per row.
///
/// # Errors
///
/// Any Postgres error while opening the `COPY` sink, writing a row, or
/// finishing the writer, as a [`postgres::Error`].
pub(super) fn copy_archived_trajectories(
    transaction: &mut Transaction<'_>,
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
    let field_refs: Vec<&KfStateFields> = rows.iter().map(|row| &row.kf_state).collect();
    let classifications = classify_rows_in_parallel(&field_refs);
    for (row, (dyn_family, semi_major, eccentricity)) in rows.iter().zip(classifications) {
        let f = &row.kf_state;

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
