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
//! This file therefore provides two ingestion entry points:
//! - [`scan_ztf_alerts`]: builds a normalized [`LazyFrame`] (predicate/projection pushdown).
//! - [`alert_store_from_lazyframe`]: materializes a [`LazyFrame`] into an engine [`AlertStore`].
//! - [`alert_store_with_truth_from_lazyframe`]: same as above, but also extracts a
//!   truth-association sidecar aligned with [`AlertId`] (evaluation workloads).
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
//! - `trajectory_id` (Int32): optional truth association (scan filter + evaluation sidecar).
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
//! Truth association (evaluation)
//! -----------------------------
//! Many evaluation datasets provide a per-alert `trajectory_id` that encodes the
//! *ground-truth* object which generated the alert (e.g., simulated truth tracks
//! or cross-matched labels).
//!
//! The core engine [`Alert`] does not store this field. For evaluation, this
//! module offers [`AlertStoreWithTruth`], which pairs:
//! - an [`AlertStore`] (engine-ready alerts),
//! - a `Vec<i32>` of `trajectory_id` values aligned with dense [`AlertId`] order.
//!
//! This design keeps the engine types unchanged while enabling fast evaluation:
//! - O(1) truth lookup: `truth[alert_id.idx()]`.
//! - No extra allocations or per-row overhead beyond a single `Vec<i32>`.
//!
//! The [`fmt::Display`] implementation for [`AlertStoreWithTruth`] reuses the
//! underlying store display and adds summary statistics over truth trajectories
//! (by default, only `trajectory_id > 0` are considered truth-associated):
//! - number of alerts with truth and fraction of total,
//! - number of unique truth trajectories,
//! - min / mean / max trajectory length (in alerts).
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
//! The truth sidecar extraction follows the same strategy:
//! - fast path: `cont_slice()` → `to_vec()`,
//! - fallback: `into_no_null_iter().collect()`.
//!
//! See also
//! --------
//! - [`AlertIngestConfig`] for ingestion-time conversion options.
//! - `dataset::schema::cols` for canonical column names used across the crate.

use std::{collections::HashMap, fmt};

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
/// * path : &ParquetSource
///     Parquet dataset location.
/// * scan : ZtfAlertScan
///     Scan configuration (filters and projection).
///
/// Returns
/// -------
/// * LazyFrame :
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
/// * ca : &Int64Chunked
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

/// AlertStore augmented with per-alert truth association (`trajectory_id`).
///
/// Overview
/// --------
/// This wrapper is intended for evaluation workflows where each alert may carry
/// a ground-truth trajectory identifier (`trajectory_id`) that should remain
/// aligned with the engine [`AlertId`] after ingestion.
///
/// The core engine [`Alert`] does not store truth metadata. Instead, we keep a
/// **sidecar vector** aligned with dense row order:
/// `trajectory_id[alert_id.idx()]` is the truth id for that alert.
///
/// Notes
/// -----
/// - By convention in evaluation datasets, `trajectory_id <= 0` often means
///   "no truth association". The provided statistics helper treats those as
///   non-associated by default.
/// - Alignment relies on the ingestion guarantee that [`AlertId`] is dense and
///   matches the row order of the collected [`DataFrame`].
#[derive(Debug)]
pub struct AlertStoreWithTruth {
    /// Engine-ready alert store.
    pub store: AlertStore,
    /// Truth trajectory id aligned with dense `AlertId` order.
    pub trajectory_id: Vec<i32>,
}

impl AlertStoreWithTruth {
    /// Return the truth trajectory id for a given alert.
    ///
    /// Parameters
    /// ----------
    /// id : AlertId
    ///     Dense alert identifier into this store.
    ///
    /// Returns
    /// -------
    /// i32
    ///     Truth id associated with the alert (may be <= 0 if unassociated).
    ///
    /// Notes
    /// -----
    /// This is an O(1) lookup into the sidecar vector.
    #[inline]
    pub fn truth_for(&self, id: AlertId) -> i32 {
        self.trajectory_id[id.idx()]
    }

    /// Compute basic summary statistics over truth trajectories (`trajectory_id > 0`).
    ///
    /// This routine groups alerts by their `trajectory_id` and reports basic
    /// distribution statistics over per-trajectory lengths (number of alerts).
    ///
    /// Returns
    /// -------
    /// (usize, usize, usize, usize, f64, usize)
    ///     Tuple of:
    ///     - `n_alerts_total`: total number of alerts in the store,
    ///     - `n_alerts_with_truth`: number of alerts with `trajectory_id > 0`,
    ///     - `n_unique_trajectories`: number of distinct `trajectory_id > 0`,
    ///     - `min_len`: minimum trajectory length (alerts),
    ///     - `mean_len`: mean trajectory length (alerts),
    ///     - `max_len`: maximum trajectory length (alerts).
    ///
    /// Notes
    /// -----
    /// - Only `trajectory_id > 0` are considered truth-associated.
    /// - The computation allocates a hash map of size `n_unique_trajectories`.
    fn truth_stats(&self) -> (usize, usize, usize, usize, f64, usize) {
        let n_alerts_total = self.trajectory_id.len();

        // Count number of alerts per truth trajectory id.
        let mut counts: HashMap<i32, usize> = HashMap::new();
        let mut n_alerts_with_truth = 0usize;

        for &tid in &self.trajectory_id {
            if tid > 0 {
                n_alerts_with_truth += 1;
                *counts.entry(tid).or_insert(0) += 1;
            }
        }

        let n_unique = counts.len();
        if n_unique == 0 {
            return (n_alerts_total, 0, 0, 0, 0.0, 0);
        }

        let mut min_len = usize::MAX;
        let mut max_len = 0usize;
        let mut sum_len = 0usize;

        for &c in counts.values() {
            min_len = min_len.min(c);
            max_len = max_len.max(c);
            sum_len += c;
        }

        let mean_len = (sum_len as f64) / (n_unique as f64);
        (
            n_alerts_total,
            n_alerts_with_truth,
            n_unique,
            min_len,
            mean_len,
            max_len,
        )
    }

    /// Iterate over all alerts associated with a given `trajectory_id`.
    ///
    /// Parameters
    /// ----------
    /// * tid : i32
    ///     Truth trajectory identifier.
    ///
    /// Returns
    /// -------
    /// * impl Iterator<Item = &Alert>
    ///     Iterator over alerts whose `trajectory_id == tid`.
    ///
    /// Notes
    /// -----
    /// - Runs in O(n) time.
    /// - Zero allocation.
    /// - Preserves alert order (time / row order).
    /// - If `tid <= 0`, this will typically return an empty iterator.
    pub fn alerts_for_trajectory<'a>(&'a self, tid: i32) -> impl Iterator<Item = &'a Alert> + 'a {
        // Version A: si AlertStore expose un slice
        self.store
            .alerts
            .iter()
            .zip(self.trajectory_id.iter())
            .filter_map(move |(alert, &t)| if t == tid { Some(alert) } else { None })
    }

    /// Collect all alerts associated with a given truth trajectory.
    ///
    /// Overview
    /// --------
    /// This is a convenience wrapper around [`alerts_for_trajectory`] that
    /// materializes the result into a `Vec<&Alert>`. It is intended for
    /// evaluation or analysis code that needs to iterate multiple times over
    /// the same trajectory, or perform operations requiring a concrete
    /// collection (sorting, random access, statistics, etc.).
    ///
    /// Parameters
    /// ----------
    /// * tid : i32
    ///     Truth trajectory identifier to select. By convention,
    ///     `tid <= 0` usually indicates "no truth association" and will
    ///     typically return an empty vector.
    ///
    /// Returns
    /// -------
    /// * Vec<&Alert>
    ///     All alerts belonging to the given truth trajectory, in the same
    ///     order as stored in the underlying [`AlertStore`] (i.e. dense
    ///     `AlertId` / row order).
    ///
    /// Notes
    /// -----
    /// - This method allocates a new `Vec` to store references to the alerts.
    /// - Internally, it relies on [`alerts_for_trajectory`] and therefore
    ///   runs in **O(n)** time, where `n` is the total number of alerts.
    /// - For single-pass processing or performance-critical paths, prefer
    ///   using the iterator returned by [`alerts_for_trajectory`] directly
    ///   to avoid the allocation.
    ///
    /// See also
    /// --------
    /// - [`alerts_for_trajectory`] – Zero-allocation iterator over alerts of
    ///   a given truth trajectory.
    pub fn alerts_for_trajectory_vec(&self, tid: i32) -> Vec<&Alert> {
        self.alerts_for_trajectory(tid).collect()
    }
}

/// Human-readable summary for evaluation logs.
///
/// This display implementation reuses the underlying [`AlertStore`] display and
/// appends truth association statistics (see [`AlertStoreWithTruth::truth_stats`]).
///
/// Notes
/// -----
/// The statistics consider only `trajectory_id > 0` as truth-associated by default.
impl fmt::Display for AlertStoreWithTruth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Reuse the engine store Display.
        writeln!(f, "{}", self.store)?;

        // Truth stats.
        let (n_total, n_truth, n_traj, min_len, mean_len, max_len) = self.truth_stats();
        let frac_truth = if n_total == 0 {
            0.0
        } else {
            (n_truth as f64) / (n_total as f64)
        };

        writeln!(f, "Truth association")?;
        writeln!(f, "----------------")?;
        writeln!(f, "alerts total            : {}", n_total)?;
        writeln!(
            f,
            "alerts with truth       : {} ({:.2}%)",
            n_truth,
            100.0 * frac_truth
        )?;

        if n_traj == 0 {
            writeln!(f, "unique trajectories      : 0")?;
            writeln!(f, "trajectory length (alerts): n/a")?;
            return Ok(());
        }

        writeln!(f, "unique trajectories      : {}", n_traj)?;
        writeln!(f, "trajectory length (alerts)")?;
        writeln!(f, "  min                   : {}", min_len)?;
        writeln!(f, "  mean                  : {:.2}", mean_len)?;
        writeln!(f, "  max                   : {}", max_len)?;

        Ok(())
    }
}

/// Borrowed view of truth-association column.
struct TruthCols<'a> {
    trajectory_id: &'a Int32Chunked,
}

/// Load truth-association column from a Polars [`DataFrame`].
///
/// Parameters
/// ----------
/// * df : &DataFrame
///     Collected frame (typically from `LazyFrame::collect()`).
///
/// Returns
/// -------
/// TruthCols
///     Borrowed typed view of `trajectory_id` (no allocations).
///
/// Errors
/// ------
/// Returns an error if:
/// - the `trajectory_id` column is missing,
/// - dtype is not Int32,
/// - the column contains null values.
///
/// Notes
/// -----
/// The evaluation path assumes no-null truth ids for simplicity and speed.
/// If you need to support nulls, change the extraction to `into_iter()` and
/// store `Vec<Option<i32>>` instead.
#[inline]
fn load_truth_cols<'a>(df: &'a DataFrame) -> Result<TruthCols<'a>> {
    let trajectory_id = df
        .column(cols::TRAJECTORY_ID)?
        .as_materialized_series()
        .i32()
        .context("trajectory_id not Int32")?;

    anyhow::ensure!(trajectory_id.null_count() == 0, "trajectory_id has nulls");

    Ok(TruthCols { trajectory_id })
}

/// Extract an Int32 column into a `Vec<i32>` efficiently.
///
/// Notes
/// -----
/// - Fast path: a single contiguous Arrow buffer (`cont_slice()`) => `to_vec()`.
/// - Fallback: chunk-safe no-null iterator (`into_no_null_iter().collect()`).
#[inline]
fn extract_i32_vec(ca: &Int32Chunked) -> Vec<i32> {
    if let Some(sl) = try_i32(ca) {
        sl.to_vec()
    } else {
        ca.into_no_null_iter().collect()
    }
}

/// Build an engine [`AlertStore`] plus an aligned `trajectory_id` sidecar.
///
/// This is the evaluation-oriented ingestion routine. It performs the same alert
/// materialization as [`alert_store_from_lazyframe`], but also extracts
/// `trajectory_id` into a contiguous `Vec<i32>` aligned with dense [`AlertId`].
///
/// Pipeline
/// --------
/// 1. Collect the `LazyFrame` into a `DataFrame`.
/// 2. Extract required typed columns (see [`load_base_cols`]) with null checks.
/// 3. Extract truth-association column (see [`load_truth_cols`]) with null checks.
/// 4. Optionally extract photometry-proxy columns (`magpsf`, `sigmapsf`).
/// 5. Build `Vec<Alert>` with contiguous-slice fast path or iterator fallback.
/// 6. Extract `trajectory_id` with contiguous-slice fast path or iterator fallback.
/// 7. Compute `start_mjd` as `floor(min(mjd_tt))`.
///
/// Parameters
/// ----------
/// * lf : LazyFrame
///     Polars lazy plan containing at least the required columns.
/// * cfg : AlertIngestConfig
///     Conversion options and default uncertainties.
///
/// Returns
/// -------
/// * AlertStoreWithTruth : 
///     The engine store plus a truth sidecar aligned with dense [`AlertId`].
///
/// Errors
/// ------
/// Returns an error if:
/// - the lazy plan cannot be collected,
/// - required columns are missing / invalid / contain nulls,
/// - `trajectory_id` is missing / invalid / contains nulls,
/// - photometry-proxy columns are requested but missing / invalid.
///
/// Notes
/// -----
/// - Truth association is returned as a sidecar to avoid modifying engine types.
/// - Alignment is guaranteed by using row order consistently for both alerts and truth.
pub fn alert_store_with_truth_from_lazyframe(
    lf: LazyFrame,
    cfg: AlertIngestConfig,
) -> Result<AlertStoreWithTruth> {
    // Collect once.
    let df = lf
        .collect()
        .context("Failed to collect LazyFrame into a DataFrame")?;

    // Required columns.
    let base = load_base_cols(&df)?;

    // Optional truth column (required for evaluation here).
    let truth = load_truth_cols(&df)?;

    let n = df.height();
    let mut alerts = Vec::with_capacity(n);

    // Conversion constants.
    let deg2rad = std::f64::consts::PI / 180.0;
    let sigma_rad = cfg.default_sigma_arcsec * deg2rad / 3600.0;
    let jd_offset = if cfg.jd_to_mjd { 2_400_000.5 } else { 0.0 };
    let angle_scale = if cfg.radec_in_degrees { deg2rad } else { 1.0 };

    // Build alerts (your existing logic).
    let min_mjd = if cfg.store_mag_as_flux_proxy {
        let mag = load_mag_cols(&df)?;
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
            build_iter_with_mag(&base, &mag, angle_scale, jd_offset, sigma_rad, &mut alerts)
        }
    } else {
        if let Some(sl) = try_base_slices(&base) {
            build_contiguous_no_mag(n, sl, angle_scale, jd_offset, sigma_rad, &mut alerts)
        } else {
            build_iter_no_mag(&base, angle_scale, jd_offset, sigma_rad, &mut alerts)
        }
    };

    // Extract truth sidecar (aligned with row order / AlertId).
    let trajectory_id = extract_i32_vec(truth.trajectory_id);
    anyhow::ensure!(
        trajectory_id.len() == n,
        "trajectory_id length mismatch: got {}, expected {}",
        trajectory_id.len(),
        n
    );

    let store = AlertStore::new(min_mjd.floor(), alerts);

    Ok(AlertStoreWithTruth {
        store,
        trajectory_id,
    })
}

/// Backward-compatible API: keep returning only the engine store.
///
/// If you want to evaluate truth later, call [`alert_store_with_truth_from_lazyframe`].
pub fn alert_store_from_lazyframe(lf: LazyFrame, cfg: AlertIngestConfig) -> Result<AlertStore> {
    Ok(alert_store_with_truth_from_lazyframe(lf, cfg)?.store)
}
