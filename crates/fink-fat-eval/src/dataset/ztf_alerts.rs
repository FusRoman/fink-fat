//! ZTF-like alert Parquet loader (Polars).
//!
//! Overview
//! --------
//! This module is the ingestion bridge between **columnar Parquet datasets**
//! (ZTF-like alert exports) and the **row-oriented** [`AlertStore`] used by the
//! `fink-fat-engine`.
//!
//! The evaluation crate (`fink-fat-eval`) typically needs to:
//! - run repeated experiments on large alert datasets,
//! - tune seeding / graph / solver parameters,
//! - measure both *compute performance* and *physical reconstruction quality*.
//!
//! For those workloads, ingestion must be:
//! - **fast** (multi-million rows in sub-second to a few seconds),
//! - **deterministic** (same input → same in-memory store),
//! - **robust** to Parquet writer differences (dtypes, chunking, etc.).
//!
//! This file therefore provides two main entry points:
//! - [`scan_ztf_alerts`]: builds a normalized [`LazyFrame`] (predicate/projection pushdown).
//! - [`alert_store_from_lazyframe`]: materializes a [`LazyFrame`] into an engine [`AlertStore`]
//!   using a fast contiguous-slice path when possible, with a chunk-safe iterator fallback.
//!
//! Dataset schema
//! --------------
//! The expected ZTF-like schema is compatible with the following columns (names are fixed):
//! - `candid` (Int64): unique detection identifier (mapped to `Alert::dia_source_id`).
//! - `ra`, `dec` (Float64): sky coordinates, either degrees or radians (configurable).
//! - `jd` (Float64): timestamp in Julian Date (configurable conversion to MJD).
//! - `fid` (Int32): filter/band id (mapped to `Alert::band`).
//! - `magpsf`, `sigmapsf` (Float32): optional photometry proxy (stored as `flux`, `flux_err`).
//! - `nid` (Int32): night id (optional scan filter).
//! - `ssnamenr` (String): optional truth label (not used by the engine ingestion).
//! - `trajectory_id` (Int32): optional truth association (optional scan filter).
//!
//! Units & conversions
//! -------------------
//! The engine uses:
//! - `Alert::ra`, `Alert::dec` in **radians**,
//! - `Alert::mjd_tt` in **days** (MJD-like), with no strict TT conversion here.
//!
//! In practice for evaluation datasets:
//! - `ra/dec` are often provided in degrees → convert using `cfg.radec_in_degrees`.
//! - `jd` is often provided as JD → convert to MJD via `cfg.jd_to_mjd` (subtract 2_400_000.5).
//!
//! Performance model
//! -----------------
//! When Polars columns are backed by a single contiguous Arrow buffer, we can access them as
//! `&[T]` slices via `cont_slice()`, enabling a very fast tight loop.
//!
//! However, Parquet reads frequently produce **chunked** arrays (multiple buffers), especially
//! for Float32 columns (`magpsf`, `sigmapsf`). In that case, the module falls back to
//! `into_no_null_iter()` which iterates chunk-by-chunk efficiently and avoids per-row indexing
//! overhead (`get(i)`), while remaining allocation-free.
//!
//! See also
//! --------
//! - [`AlertIngestConfig`] for ingestion-time conversion options.
//! - `dataset::schema::cols` for canonical column names used across the crate.

use anyhow::{Context, Result};
use fink_fat_engine::{Alert, AlertId, alerts::AlertStore};
use polars::prelude::*;

use crate::dataset::{ParquetSource, ingest_config::AlertIngestConfig};

use super::schema::cols;

/// A convenient wrapper to configure a scan on the alert Parquet file.
///
/// Notes
/// -----
/// This type controls *lazy* operations only (filtering and projection).
/// It does not influence how the [`AlertStore`] is built (see [`AlertIngestConfig`]).
///
/// Fields
/// ------
/// - `nid`: optional filter by night id (`nid` column).
/// - `only_truth`: if `true`, keep only alerts with `trajectory_id > 0`.
/// - `minimal`: if `true`, project only the minimal subset of columns.
#[derive(Clone, Debug)]
pub struct ZtfAlertScan {
    /// Optional filter on `nid` (night id).
    pub nid: Option<i32>,
    /// Optional filter to keep only alerts with a non-zero trajectory id (truth-associated).
    pub only_truth: bool,
    /// Optional projection: keep only the minimal columns needed by the engine.
    pub minimal: bool,
}

impl Default for ZtfAlertScan {
    /// Default scan configuration used for most evaluation workloads.
    ///
    /// Defaults
    /// --------
    /// - No `nid` filter.
    /// - Keep all alerts regardless of truth association.
    /// - Enable minimal projection (reduces IO and memory).
    fn default() -> Self {
        Self {
            nid: None,
            only_truth: false,
            minimal: true,
        }
    }
}

/// Build a normalized [`LazyFrame`] for the ZTF alert dataset.
///
/// This function performs a *lazy scan* of the Parquet file. The returned [`LazyFrame`]
/// benefits from Polars query optimization, notably:
/// - **predicate pushdown** (filters applied at scan time),
/// - **projection pushdown** (read only required columns),
/// - lazy expression planning.
///
/// Parameters
/// ----------
/// path : &ParquetSource
///     Parquet dataset location.
/// scan : ZtfAlertScan
///     Scan configuration (filters and projection).
///
/// Returns
/// -------
/// LazyFrame
///     A lazy computation plan with explicit casting and optional filtering.
///
/// Errors
/// ------
/// Returns an error if:
/// - the file cannot be scanned by Polars,
/// - required columns cannot be resolved at planning time (rare; usually at collect time).
pub fn scan_ztf_alerts(path: &ParquetSource, scan: ZtfAlertScan) -> Result<LazyFrame> {
    let pl_path = PlPath::from_str(path.as_path().as_str());

    // Scan lazily to enable predicate/projection pushdown into Parquet IO.
    let mut lf = LazyFrame::scan_parquet(pl_path, ScanArgsParquet::default())
        .with_context(|| format!("Failed to scan parquet file: {path:?}"))?;

    // Normalize dtypes to protect against differences between parquet writers.
    // This also helps keep downstream extraction code predictable and fast.
    lf = lf.with_columns([
        col(cols::CANDID).cast(DataType::Int64),
        col(cols::RA).cast(DataType::Float64),
        col(cols::DEC).cast(DataType::Float64),
        col(cols::JD).cast(DataType::Float64),
        col(cols::MAGPSF).cast(DataType::Float32),
        col(cols::SIGMAPSF).cast(DataType::Float32),
        col(cols::FID).cast(DataType::Int32),
        col(cols::NID).cast(DataType::Int32),
        col(cols::SSNAMENR).cast(DataType::String),
        col(cols::TRAJECTORY_ID).cast(DataType::Int32),
    ]);

    // Optional filters (lazy predicates).
    if let Some(nid) = scan.nid {
        lf = lf.filter(col(cols::NID).eq(lit(nid)));
    }
    if scan.only_truth {
        // Note: `trajectory_id` is Int32, but comparing with i64 is fine (Polars casts).
        lf = lf.filter(col(cols::TRAJECTORY_ID).gt(lit(0i64)));
    }

    // Optional projection (reduce IO + memory footprint).
    if scan.minimal {
        lf = lf.select([
            col(cols::CANDID),
            col(cols::RA),
            col(cols::DEC),
            col(cols::JD),
            col(cols::MAGPSF),
            col(cols::SIGMAPSF),
            col(cols::FID),
            col(cols::NID),
            col(cols::SSNAMENR),
            col(cols::TRAJECTORY_ID),
        ]);
    }

    Ok(lf)
}

/// Attempt to obtain a contiguous slice for an Int64 column.
///
/// Parameters
/// ----------
/// ca : &Int64Chunked
///     Chunked array to view as a contiguous slice.
///
/// Returns
/// -------
/// Option<&[i64]>
///     `Some(slice)` if the array is backed by a single contiguous buffer,
///     `None` if it is chunked or otherwise not representable as a single slice.
#[inline]
fn try_i64(ca: &Int64Chunked) -> Option<&[i64]> {
    ca.cont_slice().ok()
}

/// Attempt to obtain a contiguous slice for an Int32 column.
///
/// Parameters
/// ----------
/// ca : &Int32Chunked
///     Chunked array to view as a contiguous slice.
///
/// Returns
/// -------
/// Option<&[i32]>
///     `Some(slice)` if contiguous, otherwise `None`.
#[inline]
fn try_i32(ca: &Int32Chunked) -> Option<&[i32]> {
    ca.cont_slice().ok()
}

/// Attempt to obtain a contiguous slice for a Float64 column.
///
/// Parameters
/// ----------
/// ca : &Float64Chunked
///     Chunked array to view as a contiguous slice.
///
/// Returns
/// -------
/// Option<&[f64]>
///     `Some(slice)` if contiguous, otherwise `None`.
#[inline]
fn try_f64(ca: &Float64Chunked) -> Option<&[f64]> {
    ca.cont_slice().ok()
}

/// Attempt to obtain a contiguous slice for a Float32 column.
///
/// Parameters
/// ----------
/// ca : &Float32Chunked
///     Chunked array to view as a contiguous slice.
///
/// Returns
/// -------
/// Option<&[f32]>
///     `Some(slice)` if contiguous, otherwise `None`.
#[inline]
fn try_f32(ca: &Float32Chunked) -> Option<&[f32]> {
    ca.cont_slice().ok()
}

/// Borrowed views of the *required* columns (typed Polars chunked arrays).
///
/// This struct is a lightweight "handle" that keeps the extraction code together and
/// avoids passing many references around.
///
/// Notes
/// -----
/// These are borrowed from a collected [`DataFrame`], so their lifetime is limited
/// to the function that owns `df`.
struct BaseCols<'a> {
    candid: &'a Int64Chunked,
    ra: &'a Float64Chunked,
    dec: &'a Float64Chunked,
    jd: &'a Float64Chunked,
    fid: &'a Int32Chunked,
}

/// Borrowed views of optional photometry-proxy columns.
///
/// Notes
/// -----
/// Only loaded when `cfg.store_mag_as_flux_proxy == true`.
struct MagCols<'a> {
    magpsf: &'a Float32Chunked,
    sigmapsf: &'a Float32Chunked,
}

/// Load required columns from a Polars [`DataFrame`] as typed chunked arrays.
///
/// Parameters
/// ----------
/// df : &DataFrame
///     Collected frame (typically from `LazyFrame::collect()`).
///
/// Returns
/// -------
/// BaseCols
///     Borrowed typed column views (no allocations).
///
/// Errors
/// ------
/// Returns an error if:
/// - required columns are missing,
/// - dtypes do not match expectations,
/// - any required column contains null values.
///
/// Notes
/// -----
/// The ingestion code assumes no-null data for high performance. If your upstream
/// can produce nulls, you can:
/// - pre-filter them in `scan_ztf_alerts`, or
/// - relax the checks and use `Option` handling in the hot loop (slower).
#[inline]
fn load_base_cols<'a>(df: &'a DataFrame) -> Result<BaseCols<'a>> {
    let candid = df
        .column("candid")?
        .as_materialized_series()
        .i64()
        .context("candid not Int64")?;
    let ra = df
        .column("ra")?
        .as_materialized_series()
        .f64()
        .context("ra not Float64")?;
    let dec = df
        .column("dec")?
        .as_materialized_series()
        .f64()
        .context("dec not Float64")?;
    let jd = df
        .column("jd")?
        .as_materialized_series()
        .f64()
        .context("jd not Float64")?;
    let fid = df
        .column("fid")?
        .as_materialized_series()
        .i32()
        .context("fid not Int32")?;

    // Null checks once: unlocks no-null iterators and eliminates per-row Option overhead.
    anyhow::ensure!(candid.null_count() == 0, "candid has nulls");
    anyhow::ensure!(ra.null_count() == 0, "ra has nulls");
    anyhow::ensure!(dec.null_count() == 0, "dec has nulls");
    anyhow::ensure!(jd.null_count() == 0, "jd has nulls");
    anyhow::ensure!(fid.null_count() == 0, "fid has nulls");

    Ok(BaseCols {
        candid,
        ra,
        dec,
        jd,
        fid,
    })
}

/// Load photometry-proxy columns from a Polars [`DataFrame`] as typed chunked arrays.
///
/// Parameters
/// ----------
/// df : &DataFrame
///     Collected frame (typically from `LazyFrame::collect()`).
///
/// Returns
/// -------
/// MagCols
///     Borrowed typed column views for `magpsf` and `sigmapsf`.
///
/// Errors
/// ------
/// Returns an error if:
/// - either `magpsf` or `sigmapsf` is missing,
/// - dtypes do not match Float32,
/// - either column contains null values.
#[inline]
fn load_mag_cols<'a>(df: &'a DataFrame) -> Result<MagCols<'a>> {
    let magpsf = df
        .column("magpsf")?
        .as_materialized_series()
        .f32()
        .context("magpsf not Float32")?;
    let sigmapsf = df
        .column("sigmapsf")?
        .as_materialized_series()
        .f32()
        .context("sigmapsf not Float32")?;

    anyhow::ensure!(magpsf.null_count() == 0, "magpsf has nulls");
    anyhow::ensure!(sigmapsf.null_count() == 0, "sigmapsf has nulls");

    Ok(MagCols { magpsf, sigmapsf })
}

/// Borrowed contiguous slices for the required columns.
///
/// Notes
/// -----
/// This struct is used by the *fast path* when all required columns are backed by
/// single contiguous buffers (`cont_slice()` succeeds).
struct BaseSlices<'a> {
    candid: &'a [i64],
    ra: &'a [f64],
    dec: &'a [f64],
    jd: &'a [f64],
    fid: &'a [i32],
}

/// Try to obtain contiguous slices for the required columns.
///
/// Parameters
/// ----------
/// b : &BaseCols
///     Borrowed chunked arrays for required columns.
///
/// Returns
/// -------
/// Option<BaseSlices>
///     `Some(BaseSlices)` if **all** required columns are contiguous,
///     `None` otherwise.
#[inline]
fn try_base_slices<'a>(b: &BaseCols<'a>) -> Option<BaseSlices<'a>> {
    Some(BaseSlices {
        candid: try_i64(b.candid)?,
        ra: try_f64(b.ra)?,
        dec: try_f64(b.dec)?,
        jd: try_f64(b.jd)?,
        fid: try_i32(b.fid)?,
    })
}

/// Try to obtain contiguous slices for photometry-proxy columns.
///
/// Parameters
/// ----------
/// m : &MagCols
///     Borrowed chunked arrays for `magpsf` and `sigmapsf`.
///
/// Returns
/// -------
/// Option<(&[f32], &[f32])>
///     `Some((mag, sig))` if both are contiguous, otherwise `None`.
#[inline]
fn try_mag_slices<'a>(m: &MagCols<'a>) -> Option<(&'a [f32], &'a [f32])> {
    Some((try_f32(m.magpsf)?, try_f32(m.sigmapsf)?))
}

/// Push a single engine [`Alert`] into an output vector.
///
/// Parameters
/// ----------
/// alerts : &mut Vec<Alert>
///     Destination buffer (pre-allocated by caller).
/// idx : usize
///     0-based dense index used to build [`AlertId`].
/// dia_source_id : u64
///     Detection identifier (from `candid`).
/// ra, dec : f64
///     Sky coordinates in radians.
/// mjd_tt : f64
///     Timestamp in days (MJD-like).
/// flux, flux_err : f32
///     Photometry proxy values (or zeros if disabled).
/// band_i32 : i32
///     Input band id (clamped into 0..=255).
/// sigma_rad : f64
///     Default astrometric uncertainty assigned to `ra_err`, `dec_err`.
///
/// Returns
/// -------
/// ()
///     Appends an `Alert` to `alerts`.
#[inline(always)]
fn push_alert(
    alerts: &mut Vec<Alert>,
    idx: usize,
    dia_source_id: u64,
    ra: f64,
    dec: f64,
    mjd_tt: f64,
    flux: f32,
    flux_err: f32,
    band_i32: i32,
    sigma_rad: f64,
) {
    alerts.push(Alert {
        id: AlertId::from(idx),
        dia_source_id,
        ra,
        ra_err: sigma_rad,
        dec,
        dec_err: sigma_rad,
        mjd_tt,
        flux,
        flux_err,
        band: band_i32.clamp(0, 255) as u8,
    });
}

/// Build alerts using contiguous slices (no photometry proxy).
///
/// This is the fastest path when the required columns are contiguous.
///
/// Parameters
/// ----------
/// n : usize
///     Number of rows.
/// sl : BaseSlices
///     Contiguous views of required columns.
/// angle_scale : f64
///     Multiply RA/DEC by this factor (1.0 if already in radians, deg2rad otherwise).
/// jd_offset : f64
///     Subtract from JD to obtain MJD (0.0 if already MJD-like).
/// sigma_rad : f64
///     Default astrometric uncertainty assigned to all alerts (radians).
/// alerts : &mut Vec<Alert>
///     Destination buffer (pre-allocated by caller).
///
/// Returns
/// -------
/// f64
///     Minimum `mjd_tt` observed while building alerts (used to compute `start_mjd`).
fn build_contiguous_no_mag(
    n: usize,
    sl: BaseSlices<'_>,
    angle_scale: f64,
    jd_offset: f64,
    sigma_rad: f64,
    alerts: &mut Vec<Alert>,
) -> f64 {
    let mut min_mjd = f64::INFINITY;

    // Tight loop: direct indexing on slices, no per-row Option overhead.
    for i in 0..n {
        let ra_val = sl.ra[i] * angle_scale;
        let dec_val = sl.dec[i] * angle_scale;
        let mjd_tt = sl.jd[i] - jd_offset;
        min_mjd = min_mjd.min(mjd_tt);

        push_alert(
            alerts,
            i,
            sl.candid[i] as u64,
            ra_val,
            dec_val,
            mjd_tt,
            0.0,
            0.0,
            sl.fid[i],
            sigma_rad,
        );
    }

    min_mjd
}

/// Build alerts using contiguous slices (with photometry proxy).
///
/// Parameters
/// ----------
/// n : usize
///     Number of rows.
/// sl : BaseSlices
///     Contiguous views of required columns.
/// mag, sig : &[f32]
///     Contiguous photometry-proxy slices (`magpsf`, `sigmapsf`).
/// angle_scale : f64
///     Multiply RA/DEC by this factor (1.0 if already in radians, deg2rad otherwise).
/// jd_offset : f64
///     Subtract from JD to obtain MJD (0.0 if already MJD-like).
/// sigma_rad : f64
///     Default astrometric uncertainty assigned to all alerts (radians).
/// alerts : &mut Vec<Alert>
///     Destination buffer (pre-allocated by caller).
///
/// Returns
/// -------
/// f64
///     Minimum `mjd_tt` observed while building alerts.
fn build_contiguous_with_mag(
    n: usize,
    sl: BaseSlices<'_>,
    mag: &[f32],
    sig: &[f32],
    angle_scale: f64,
    jd_offset: f64,
    sigma_rad: f64,
    alerts: &mut Vec<Alert>,
) -> f64 {
    let mut min_mjd = f64::INFINITY;

    for i in 0..n {
        let ra_val = sl.ra[i] * angle_scale;
        let dec_val = sl.dec[i] * angle_scale;
        let mjd_tt = sl.jd[i] - jd_offset;
        min_mjd = min_mjd.min(mjd_tt);

        push_alert(
            alerts,
            i,
            sl.candid[i] as u64,
            ra_val,
            dec_val,
            mjd_tt,
            mag[i],
            sig[i],
            sl.fid[i],
            sigma_rad,
        );
    }

    min_mjd
}

/// Build alerts using chunk-safe no-null iterators (no photometry proxy).
///
/// This path is used when one or more required columns are not contiguous.
/// It remains efficient by iterating chunk-by-chunk rather than indexing by row.
///
/// Parameters
/// ----------
/// base : &BaseCols
///     Borrowed typed column views.
/// angle_scale : f64
///     Multiply RA/DEC by this factor.
/// jd_offset : f64
///     Subtract from JD to obtain MJD.
/// sigma_rad : f64
///     Default astrometric uncertainty (radians).
/// alerts : &mut Vec<Alert>
///     Destination buffer (pre-allocated).
///
/// Returns
/// -------
/// f64
///     Minimum `mjd_tt` observed while building alerts.
fn build_iter_no_mag(
    base: &BaseCols<'_>,
    angle_scale: f64,
    jd_offset: f64,
    sigma_rad: f64,
    alerts: &mut Vec<Alert>,
) -> f64 {
    let mut min_mjd = f64::INFINITY;
    let mut i = 0usize;

    // Zipped iterators: chunk-friendly and avoids per-row `get(i)` overhead.
    for ((((dia, ra_v), dec_v), jd_v), fid_v) in base
        .candid
        .into_no_null_iter()
        .zip(base.ra.into_no_null_iter())
        .zip(base.dec.into_no_null_iter())
        .zip(base.jd.into_no_null_iter())
        .zip(base.fid.into_no_null_iter())
    {
        let ra_val = ra_v * angle_scale;
        let dec_val = dec_v * angle_scale;
        let mjd_tt = jd_v - jd_offset;
        min_mjd = min_mjd.min(mjd_tt);

        push_alert(
            alerts, i, dia as u64, ra_val, dec_val, mjd_tt, 0.0, 0.0, fid_v, sigma_rad,
        );
        i += 1;
    }

    min_mjd
}

/// Build alerts using chunk-safe no-null iterators (with photometry proxy).
///
/// Parameters
/// ----------
/// base : &BaseCols
///     Borrowed typed views of required columns.
/// mag : &MagCols
///     Borrowed typed views of photometry-proxy columns.
/// angle_scale : f64
///     Multiply RA/DEC by this factor.
/// jd_offset : f64
///     Subtract from JD to obtain MJD.
/// sigma_rad : f64
///     Default astrometric uncertainty (radians).
/// alerts : &mut Vec<Alert>
///     Destination buffer (pre-allocated).
///
/// Returns
/// -------
/// f64
///     Minimum `mjd_tt` observed while building alerts.
fn build_iter_with_mag(
    base: &BaseCols<'_>,
    mag: &MagCols<'_>,
    angle_scale: f64,
    jd_offset: f64,
    sigma_rad: f64,
    alerts: &mut Vec<Alert>,
) -> f64 {
    let mut min_mjd = f64::INFINITY;
    let mut i = 0usize;

    // Photometry is zipped as a tuple iterator to preserve tight loop structure.
    for (((((dia, ra_v), dec_v), jd_v), fid_v), (mag_v, sig_v)) in base
        .candid
        .into_no_null_iter()
        .zip(base.ra.into_no_null_iter())
        .zip(base.dec.into_no_null_iter())
        .zip(base.jd.into_no_null_iter())
        .zip(base.fid.into_no_null_iter())
        .zip(
            mag.magpsf
                .into_no_null_iter()
                .zip(mag.sigmapsf.into_no_null_iter()),
        )
    {
        let ra_val = ra_v * angle_scale;
        let dec_val = dec_v * angle_scale;
        let mjd_tt = jd_v - jd_offset;
        min_mjd = min_mjd.min(mjd_tt);

        push_alert(
            alerts, i, dia as u64, ra_val, dec_val, mjd_tt, mag_v, sig_v, fid_v, sigma_rad,
        );
        i += 1;
    }

    min_mjd
}

/// Build an engine [`AlertStore`] from a Polars [`LazyFrame`].
///
/// This is the main ingestion routine used by `fink-fat-eval` after a dataset scan.
///
/// Pipeline
/// --------
/// 1. Collect the `LazyFrame` into a `DataFrame`.
/// 2. Extract required typed columns (`candid`, `ra`, `dec`, `jd`, `fid`), with null checks.
/// 3. Optionally extract photometry-proxy columns (`magpsf`, `sigmapsf`), with null checks.
/// 4. Compute ingestion constants:
///    - `angle_scale` (degrees → radians, if requested),
///    - `jd_offset` (JD → MJD, if requested),
///    - `sigma_rad` default astrometric uncertainty (arcsec → radians).
/// 5. Build `Vec<Alert>` with:
///    - a contiguous-slice fast path when possible,
///    - a chunk-safe iterator fallback otherwise.
/// 6. Compute `start_mjd` as `floor(min(mjd_tt))` (computed on-the-fly during building).
///
/// Parameters
/// ----------
/// lf : LazyFrame
///     Polars lazy plan containing at least the required columns.
/// cfg : AlertIngestConfig
///     Conversion options and default uncertainties.
///
/// Returns
/// -------
/// AlertStore
///     A contiguous store of engine alerts suitable for seeding and linking.
///
/// Errors
/// ------
/// Returns an error if:
/// - the lazy plan cannot be collected,
/// - required columns are missing or have incompatible dtypes,
/// - required columns contain null values,
/// - photometry-proxy columns are requested but missing / invalid.
///
/// Notes
/// -----
/// - `Alert::dia_source_id` is populated from `candid` (ZTF-like identifier).
/// - `Alert::flux` / `Alert::flux_err` store `magpsf` / `sigmapsf` as a *proxy* when enabled.
/// - No TT conversion is applied; `mjd_tt` is treated as "MJD-like days" for evaluation.
/// - The output `AlertId` is dense and matches the row order of the collected frame.
pub fn alert_store_from_lazyframe(lf: LazyFrame, cfg: AlertIngestConfig) -> Result<AlertStore> {
    // Collect the lazy computation plan. This is the point where IO happens.
    let df = lf
        .collect()
        .context("Failed to collect LazyFrame into a DataFrame")?;

    // Extract required columns (typed) and validate that we have no nulls.
    let base = load_base_cols(&df)?;

    // Output: store alerts in a single contiguous Vec for cache-friendly downstream passes.
    let n = df.height();
    let mut alerts = Vec::with_capacity(n);

    // Precompute conversion constants once.
    let deg2rad = std::f64::consts::PI / 180.0;
    let sigma_rad = cfg.default_sigma_arcsec * deg2rad / 3600.0;
    let jd_offset = if cfg.jd_to_mjd { 2_400_000.5 } else { 0.0 };
    let angle_scale = if cfg.radec_in_degrees { deg2rad } else { 1.0 };

    // Choose the best building strategy depending on data contiguity and config.
    let min_mjd = if cfg.store_mag_as_flux_proxy {
        // Load photometry-proxy columns only if requested.
        let mag = load_mag_cols(&df)?;

        // Fast path if all required + optional columns are backed by contiguous buffers.
        if let (Some(sl), Some((mag_sl, sig_sl))) = (try_base_slices(&base), try_mag_slices(&mag)) {
            build_contiguous_with_mag(
                n,
                sl,
                mag_sl,
                sig_sl,
                angle_scale,
                jd_offset,
                sigma_rad,
                &mut alerts,
            )
        } else {
            // Fallback: chunk-safe no-null iterators (still fast, no per-row get()).
            build_iter_with_mag(&base, &mag, angle_scale, jd_offset, sigma_rad, &mut alerts)
        }
    } else {
        // No photometry requested: only required columns matter.
        if let Some(sl) = try_base_slices(&base) {
            build_contiguous_no_mag(n, sl, angle_scale, jd_offset, sigma_rad, &mut alerts)
        } else {
            build_iter_no_mag(&base, angle_scale, jd_offset, sigma_rad, &mut alerts)
        }
    };

    // Engine convention: start_mjd is the floor of the minimum mjd in the store.
    Ok(AlertStore::new(min_mjd.floor(), alerts))
}
