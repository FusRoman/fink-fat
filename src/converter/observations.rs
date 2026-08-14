//! Read raw observation rows (candid-keyed ZTF/LSST alerts, e.g.
//! `sso_dataset_eval.parquet`, produced by `test_exp/prep_alert.py`) from
//! Parquet and bulk-load them into the `observations` table, using the same
//! `COPY ... FROM STDIN BINARY` pattern as the rest of
//! [`crate::converter::sql`].
//!
//! Kept in its own module (rather than folded into `sql.rs`) because it has
//! a distinct concern — reading an external Parquet file — instead of
//! walking a [`fink_fat_engine::topocentric_kf::branching::BranchCollection`].

use camino::Utf8Path;
use indicatif::ProgressBar;
use polars::prelude::*;
use postgres::{binary_copy::BinaryCopyInWriter, types::Type};

/// One row of the `observations` table: a single alert as produced by
/// `test_exp/prep_alert.py`. The ground-truth `traj_id` column present in
/// that parquet is deliberately not read here — it's an evaluation-only
/// artifact with no equivalent on real production data, and this table is
/// meant to also hold production observations.
pub struct ObservationRow {
    pub id: i64,
    pub night_id: i64,
    pub object_id: String,
    pub magnitude: f64,
    pub mag_err: f64,
    pub filter: i16,
    pub mpc_code_obs: String,
    pub ra: f64,
    pub ra_err: f64,
    pub dec: f64,
    pub dec_err: f64,
    pub mjd_tt: f64,
}

/// Read every row of `path` (a parquet file shaped like
/// `sso_dataset_eval.parquet`: `night_id`, `id`, `objectId`, `magnitude`,
/// `mag_err`, `filter`, `mpc_code_obs`, `ra`, `ra_err`, `dec`, `dec_err`,
/// `mjd_tt`, plus a `traj_id` column that is ignored) into a
/// `Vec<ObservationRow>`.
pub fn read_observation_rows(
    path: &Utf8Path,
) -> Result<Vec<ObservationRow>, Box<dyn std::error::Error>> {
    let mut df =
        LazyFrame::scan_parquet(path.as_str().into(), ScanArgsParquet::default())?.collect()?;
    df.rechunk_mut();

    let id = df
        .column("id")?
        .as_materialized_series()
        .u64()?
        .cont_slice()?;
    let night_id = df
        .column("night_id")?
        .as_materialized_series()
        .u32()?
        .cont_slice()?;
    let object_id = df.column("objectId")?.as_materialized_series().str()?;
    let magnitude = df
        .column("magnitude")?
        .as_materialized_series()
        .f64()?
        .cont_slice()?;
    let mag_err = df
        .column("mag_err")?
        .as_materialized_series()
        .f64()?
        .cont_slice()?;
    let filter = df
        .column("filter")?
        .as_materialized_series()
        .u8()?
        .cont_slice()?;
    let mpc_code_obs = df.column("mpc_code_obs")?.as_materialized_series().str()?;
    let ra = df
        .column("ra")?
        .as_materialized_series()
        .f64()?
        .cont_slice()?;
    let ra_err = df
        .column("ra_err")?
        .as_materialized_series()
        .f64()?
        .cont_slice()?;
    let dec = df
        .column("dec")?
        .as_materialized_series()
        .f64()?
        .cont_slice()?;
    let dec_err = df
        .column("dec_err")?
        .as_materialized_series()
        .f64()?
        .cont_slice()?;
    let mjd_tt = df
        .column("mjd_tt")?
        .as_materialized_series()
        .f64()?
        .cont_slice()?;

    let n = df.height();
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        rows.push(ObservationRow {
            id: id[i] as i64,
            night_id: night_id[i] as i64,
            object_id: object_id
                .get(i)
                .ok_or("null objectId in observation parquet")?
                .to_string(),
            magnitude: magnitude[i],
            mag_err: mag_err[i],
            filter: filter[i] as i16,
            mpc_code_obs: mpc_code_obs
                .get(i)
                .ok_or("null mpc_code_obs in observation parquet")?
                .to_string(),
            ra: ra[i],
            ra_err: ra_err[i],
            dec: dec[i],
            dec_err: dec_err[i],
            mjd_tt: mjd_tt[i],
        });
    }
    Ok(rows)
}

/// Bulk-load `rows` into `observations` via `COPY ... FROM STDIN BINARY`.
/// Must run before [`crate::converter::sql`]'s `copy_branch_observations`
/// within the same transaction, so that `branch_observations.obs_id`'s
/// foreign key into `observations(id)` is satisfied.
pub(crate) fn copy_observations(
    transaction: &mut postgres::Transaction<'_>,
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
