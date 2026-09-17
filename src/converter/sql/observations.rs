//! Read the raw observation parquet (candid-keyed ZTF/LSST alerts, e.g.
//! `sso_dataset_eval.parquet`, produced by `test_exp/prep_alert.py`) that
//! feeds the `observations` table.
//!
//! Kept separate from [`super::build`] because it has a distinct concern —
//! reading an external Parquet file — rather than walking a
//! [`fink_fat_engine::topocentric_kf::branching::BranchCollection`].

use camino::Utf8Path;
use polars::prelude::*;

use super::rows::ObservationRow;

/// Reads every row of `path` into a [`Vec<ObservationRow>`].
///
/// # Arguments
///
/// * `path` — a parquet file shaped like `sso_dataset_eval.parquet`:
///   `night_id`, `id`, `objectId`, `magnitude`, `mag_err`, `filter`,
///   `mpc_code_obs`, `ra`, `ra_err`, `dec`, `dec_err`, `mjd_tt`, plus a
///   `traj_id` column that is ignored (see [`ObservationRow`]'s doc comment).
///
/// # Returns
///
/// One [`ObservationRow`] per row of `path`, in file order.
///
/// # Errors
///
/// The file failing to open/parse as the expected schema (via [`polars`]),
/// or a null `objectId`/`mpc_code_obs` value (both are `NOT NULL` columns on
/// `observations` and have no sensible default).
pub(super) fn read_observation_rows(
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
