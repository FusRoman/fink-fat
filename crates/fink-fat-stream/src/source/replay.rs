//! [`ReplaySource`]: replays a Fink alert dump (parquet) as if it were
//! arriving live.
//!
//! A dump has no reception time, so each alert's
//! [`arrival_mjd`](AlertRecord::arrival_mjd) is simulated as its exposure
//! mid-point (converted TAI → UTC) plus a fixed latency — about three
//! minutes are observed between a Rubin exposure and its alerts reaching
//! Fink. Alerts are delivered in arrival order, in batches; the driver's
//! clock then follows their event time.
//!
//! # Column mapping
//!
//! The dump's column names live in one place, [`ColumnMap`] (defaults: the
//! Rubin `DiaSource` names), overridable from the configuration. A row with
//! a null in any mapped column is skipped and counted.
//!
//! # Restart semantics
//!
//! A replay has no offsets: [`AlertSource::ack`] is a no-op, and a restarted
//! replay delivers the whole dump again. That is enough for the driver,
//! which drops what it already processed — alerts of closed nights are
//! *late*, alerts of the open night already in its journal are *duplicates*
//! — so a replay exercises exactly the redelivery path of a live source.
//!
//! The whole dump is loaded in memory at start-up: the replay is a test and
//! validation harness, not the production source.

use std::vec::IntoIter;

use camino::{Utf8Path, Utf8PathBuf};
use fink_fat_engine::engine_config::units::de_time_days;
use polars::prelude::*;
use serde::Deserialize;

use crate::{
    error::Result,
    record::{AlertRecord, mjd_tai_to_utc},
    source::AlertSource,
};

/// Names of the dump's columns holding each [`AlertRecord`] field (the
/// `replay.columns` section of the configuration; unset fields keep the
/// Rubin name).
///
/// Only flat columns are supported: a dump nesting the `DiaSource` in a
/// struct column must be flattened first.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ColumnMap {
    /// `diaSourceId` (integer).
    pub dia_source_id: String,
    /// `visit` (integer).
    pub visit: String,
    /// `midpointMjdTai` (MJD TAI).
    pub midpoint_mjd_tai: String,
    /// `ra` (degrees).
    pub ra: String,
    /// `dec` (degrees).
    pub dec: String,
    /// `raErr` (on-sky, degrees).
    pub ra_err: String,
    /// `decErr` (degrees).
    pub dec_err: String,
    /// `psfFlux` (nJy).
    pub psf_flux: String,
    /// `psfFluxErr` (nJy).
    pub psf_flux_err: String,
    /// `band` (string).
    pub band: String,
}

impl Default for ColumnMap {
    /// The Rubin `DiaSource` column names.
    fn default() -> Self {
        Self {
            dia_source_id: "diaSourceId".into(),
            visit: "visit".into(),
            midpoint_mjd_tai: "midpointMjdTai".into(),
            ra: "ra".into(),
            dec: "dec".into(),
            ra_err: "raErr".into(),
            dec_err: "decErr".into(),
            psf_flux: "psfFlux".into(),
            psf_flux_err: "psfFluxErr".into(),
            band: "band".into(),
        }
    }
}

/// Internal names the mapped columns are renamed to before extraction, so
/// that [`read_records`] reads fixed names whatever the dump's.
mod canonical {
    pub const DIA_SOURCE_ID: &str = "dia_source_id";
    pub const VISIT: &str = "visit";
    pub const MIDPOINT: &str = "midpoint_mjd_tai";
    pub const RA: &str = "ra";
    pub const DEC: &str = "dec";
    pub const RA_ERR: &str = "ra_err";
    pub const DEC_ERR: &str = "dec_err";
    pub const PSF_FLUX: &str = "psf_flux";
    pub const PSF_FLUX_ERR: &str = "psf_flux_err";
    pub const BAND: &str = "band";
}

impl ColumnMap {
    /// The `select` expressions of [`read_records`]: each mapped column,
    /// cast to its extraction type and renamed to its [`canonical`] name.
    fn select_exprs(&self) -> Vec<Expr> {
        let typed = |name: &str, dtype: DataType, alias: &str| col(name).cast(dtype).alias(alias);
        vec![
            typed(
                &self.dia_source_id,
                DataType::Int64,
                canonical::DIA_SOURCE_ID,
            ),
            typed(&self.visit, DataType::Int64, canonical::VISIT),
            typed(
                &self.midpoint_mjd_tai,
                DataType::Float64,
                canonical::MIDPOINT,
            ),
            typed(&self.ra, DataType::Float64, canonical::RA),
            typed(&self.dec, DataType::Float64, canonical::DEC),
            typed(&self.ra_err, DataType::Float64, canonical::RA_ERR),
            typed(&self.dec_err, DataType::Float64, canonical::DEC_ERR),
            typed(&self.psf_flux, DataType::Float64, canonical::PSF_FLUX),
            typed(
                &self.psf_flux_err,
                DataType::Float64,
                canonical::PSF_FLUX_ERR,
            ),
            typed(&self.band, DataType::String, canonical::BAND),
        ]
    }
}

fn default_latency() -> f64 {
    3.0 / 1_440.0
}

fn default_batch_size() -> usize {
    1_000
}

/// `replay:` section of the stream configuration.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayConfig {
    /// Parquet dump to replay.
    pub path: Utf8PathBuf,
    /// Simulated delay between an exposure's mid-point and its alerts'
    /// arrival, days (YAML accepts units; default `"3 min"`).
    #[serde(default = "default_latency", deserialize_with = "de_time_days")]
    pub latency: f64,
    /// Alerts per batch (default 1000).
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// Column names of the dump (default: Rubin names).
    #[serde(default)]
    pub columns: ColumnMap,
}

/// Alerts extracted from a dump by [`read_records`].
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayRecords {
    /// One record per complete row, in file order.
    pub records: Vec<AlertRecord>,
    /// Rows skipped for a null in a mapped column.
    pub n_null_rows: usize,
}

/// Extract the alerts of `frame`, simulating their arrival times.
///
/// # Steps
/// 1. Select the [`ColumnMap`]'s columns, cast each to the type its
///    [`AlertRecord`] field is read from (`Int64`, `Float64` or `String`)
///    and rename it to an internal canonical name — so the rest of the
///    function never depends on the dump's naming.
/// 2. Build one record per row. A row with a null in any of these columns
///    is skipped and counted, not an error: a dump may hold partial rows,
///    and the stream must go on.
/// 3. Set each record's `arrival_mjd` to `midpointMjdTai` converted to UTC
///    plus `latency` (see the [module doc](self)).
///
/// Floats are cast to `Float64` then narrowed to `f32` for the APDB `float`
/// fields, so a dump storing them as `Float32` round-trips exactly.
///
/// # Arguments
/// * `frame` – The dump, lazily (a parquet scan, or an in-memory frame in
///   tests).
/// * `columns` – Where each field lives in the dump.
/// * `latency` – Simulated exposure-to-arrival delay, days.
///
/// # Returns
/// The records in file order (not sorted: [`ReplaySource::from_records`]
/// sorts them), and the number of skipped rows.
///
/// # Errors
/// A mapped column is missing or cannot be cast.
pub fn read_records(frame: LazyFrame, columns: &ColumnMap, latency: f64) -> Result<ReplayRecords> {
    // 1. Canonical, typed columns.
    let mut df = frame.select(columns.select_exprs()).collect()?;
    df.rechunk_mut();
    let series =
        |name: &str| -> Result<Series> { Ok(df.column(name)?.as_materialized_series().clone()) };
    let (dia_source_id, visit) = (series(canonical::DIA_SOURCE_ID)?, series(canonical::VISIT)?);
    let (midpoint, ra, dec) = (
        series(canonical::MIDPOINT)?,
        series(canonical::RA)?,
        series(canonical::DEC)?,
    );
    let (ra_err, dec_err) = (series(canonical::RA_ERR)?, series(canonical::DEC_ERR)?);
    let (psf_flux, psf_flux_err) = (
        series(canonical::PSF_FLUX)?,
        series(canonical::PSF_FLUX_ERR)?,
    );
    let band = series(canonical::BAND)?;
    let (dia_source_id, visit) = (dia_source_id.i64()?, visit.i64()?);
    let (midpoint, ra, dec) = (midpoint.f64()?, ra.f64()?, dec.f64()?);
    let (ra_err, dec_err) = (ra_err.f64()?, dec_err.f64()?);
    let (psf_flux, psf_flux_err) = (psf_flux.f64()?, psf_flux_err.f64()?);
    let band = band.str()?;

    // 2.–3. One record per complete row.
    let mut out = ReplayRecords {
        records: Vec::with_capacity(df.height()),
        n_null_rows: 0,
    };
    for row in 0..df.height() {
        // `None` as soon as any field of the row is null.
        let record = (|| {
            let midpoint_mjd_tai = midpoint.get(row)?;
            Some(AlertRecord {
                dia_source_id: dia_source_id.get(row)?,
                visit: visit.get(row)?,
                midpoint_mjd_tai,
                ra_deg: ra.get(row)?,
                dec_deg: dec.get(row)?,
                ra_err_deg: ra_err.get(row)? as f32,
                dec_err_deg: dec_err.get(row)? as f32,
                psf_flux_njy: psf_flux.get(row)? as f32,
                psf_flux_err_njy: psf_flux_err.get(row)? as f32,
                band: band.get(row)?.to_string(),
                arrival_mjd: mjd_tai_to_utc(midpoint_mjd_tai) + latency,
            })
        })();
        match record {
            Some(record) => out.records.push(record),
            None => out.n_null_rows += 1,
        }
    }
    Ok(out)
}

/// Replays a fixed set of alerts in arrival order (see the
/// [module doc](self)).
///
/// Event time only: [`AlertSource::wall_clock_mjd`] is `None`, and the
/// source reports itself caught up only once everything was delivered.
#[derive(Debug)]
pub struct ReplaySource {
    /// Alerts not delivered yet, in delivery order.
    remaining: IntoIter<AlertRecord>,
    /// Alerts per batch (≥ 1).
    batch_size: usize,
}

impl ReplaySource {
    /// Replay `records` in batches of `batch_size`.
    ///
    /// Records are sorted by arrival time, ties broken by `diaSourceId` so
    /// that two replays of the same dump deliver the same sequence (alerts
    /// of one exposure share their simulated arrival).
    ///
    /// # Arguments
    /// * `records` – Alerts to replay, any order.
    /// * `batch_size` – Alerts per batch; `0` is raised to `1`.
    pub fn from_records(mut records: Vec<AlertRecord>, batch_size: usize) -> Self {
        records.sort_by(|a, b| {
            a.arrival_mjd
                .total_cmp(&b.arrival_mjd)
                .then(a.dia_source_id.cmp(&b.dia_source_id))
        });
        Self {
            remaining: records.into_iter(),
            batch_size: batch_size.max(1),
        }
    }

    /// Load the parquet dump of `config` ([`read_records`]) and replay it
    /// ([`Self::from_records`]).
    ///
    /// # Returns
    /// The source and the number of rows skipped for nulls.
    ///
    /// # Errors
    /// The dump cannot be read, or lacks a mapped column.
    pub fn from_parquet(config: &ReplayConfig) -> Result<(Self, usize)> {
        let frame = scan(&config.path)?;
        let read = read_records(frame, &config.columns, config.latency)?;
        Ok((
            Self::from_records(read.records, config.batch_size),
            read.n_null_rows,
        ))
    }

    /// Alerts not delivered yet.
    pub fn n_remaining(&self) -> usize {
        self.remaining.len()
    }
}

/// Lazy scan of the parquet file at `path`.
fn scan(path: &Utf8Path) -> Result<LazyFrame> {
    Ok(LazyFrame::scan_parquet(
        path.as_str().into(),
        ScanArgsParquet::default(),
    )?)
}

impl AlertSource for ReplaySource {
    /// The next `batch_size` alerts (fewer at the end), `None` once all
    /// were delivered.
    fn next_batch(&mut self) -> Result<Option<Vec<AlertRecord>>> {
        let batch: Vec<AlertRecord> = self.remaining.by_ref().take(self.batch_size).collect();
        Ok((!batch.is_empty()).then_some(batch))
    }

    /// Nothing to acknowledge: a restarted replay starts over, and the
    /// driver skips what it already processed (closed nights are late,
    /// journaled alerts are duplicates).
    fn ack(&mut self) -> Result<()> {
        Ok(())
    }

    /// `true` once every alert was delivered: a replay has no "live edge"
    /// before its end.
    fn caught_up(&self) -> bool {
        self.remaining.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dump() -> LazyFrame {
        df![
            "diaSourceId" => [Some(2_i64), Some(1), None],
            "visit" => [2_026_103_100_002_i64, 2_026_103_100_001, 2_026_103_100_001],
            "midpointMjdTai" => [61_345.2, 61_345.1, 61_345.1],
            "ra" => [10.0, 11.0, 12.0],
            "dec" => [-5.0, -6.0, -7.0],
            "raErr" => [1e-5_f32, 2e-5, 3e-5],
            "decErr" => [1e-5_f32, 2e-5, 3e-5],
            "psfFlux" => [100.0_f32, 200.0, 300.0],
            "psfFluxErr" => [10.0_f32, 20.0, 30.0],
            "band" => ["g", "r", "i"],
        ]
        .unwrap()
        .lazy()
    }

    #[test]
    fn rows_become_records_and_null_rows_are_counted() {
        let latency = 3.0 / 1_440.0;
        let read = read_records(dump(), &ColumnMap::default(), latency).unwrap();
        assert_eq!(read.n_null_rows, 1);
        assert_eq!(read.records.len(), 2);
        let first = &read.records[0];
        assert_eq!(first.dia_source_id, 2);
        assert_eq!(first.visit, 2_026_103_100_002);
        assert_eq!(first.band, "g");
        assert_eq!(first.ra_err_deg, 1e-5);
        assert!((first.arrival_mjd - (mjd_tai_to_utc(61_345.2) + latency)).abs() < 1e-12);
    }

    #[test]
    fn the_column_map_renames_columns() {
        let frame = dump().rename(["ra"], ["ra_deg"], true);
        assert!(read_records(frame.clone(), &ColumnMap::default(), 0.0).is_err());
        let columns = ColumnMap {
            ra: "ra_deg".into(),
            ..ColumnMap::default()
        };
        assert_eq!(read_records(frame, &columns, 0.0).unwrap().records.len(), 2);
    }

    #[test]
    fn replay_delivers_in_arrival_order_and_in_batches() {
        let records = read_records(dump(), &ColumnMap::default(), 0.0)
            .unwrap()
            .records;
        let mut source = ReplaySource::from_records(records, 1);
        assert!(!source.caught_up());
        let first = source.next_batch().unwrap().unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].dia_source_id, 1, "earliest exposure first");
        assert_eq!(source.next_batch().unwrap().unwrap()[0].dia_source_id, 2);
        assert!(source.caught_up());
        assert_eq!(source.next_batch().unwrap(), None);
    }

    #[test]
    fn replay_config_parses_units_and_defaults() {
        let config: ReplayConfig =
            serde_yaml::from_str("path: dump.parquet\nlatency: \"90 sec\"\n").unwrap();
        assert_eq!(config.path, "dump.parquet");
        assert!((config.latency - 90.0 / 86_400.0).abs() < 1e-12);
        assert_eq!(config.batch_size, 1_000);
        assert_eq!(config.columns, ColumnMap::default());
    }
}
