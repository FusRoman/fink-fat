//! Shared helpers for `fink-fat-eval` binaries.
//!
//! Overview
//! --------
//! This module centralizes common boilerplate used by many evaluation binaries:
//! - parsing / discovering night ids (`nid`),
//! - resolving a final nid list with `--max-nights`,
//! - time formatting helpers,
//! - ingesting a single night into an `AlertStoreWithTruth`.
//!
//! Keeping this logic in one place avoids copy/paste and keeps semantics consistent.

use anyhow::{Context, Result};
use camino::Utf8PathBuf;
use fink_fat_engine::MjdTt;
use polars::prelude::*;

use crate::{FiniteOr, dataset::{
    ParquetSource,
    ingest_config::AlertIngestConfig,
    schema::cols,
    ztf_alerts::{
        AlertLoadMode, AlertStoreWithTruth, ZtfAlertScan, alert_store_with_truth_from_lazyframe,
        scan_ztf_alerts,
    },
}};

/// Format a [`std::time::Duration`] in milliseconds.
#[inline]
pub fn fmt_ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1.0e3
}

/// Parse a comma-separated list of `nid` values.
///
/// Examples
/// --------
/// - `"3122"`
/// - `"3122, 3145,3156"`
pub fn parse_nids_csv(s: &str) -> Result<Vec<i32>> {
    let mut out = Vec::new();
    for raw in s.split(',') {
        let t = raw.trim();
        if t.is_empty() {
            continue;
        }
        out.push(
            t.parse::<i32>()
                .with_context(|| format!("invalid nid value: '{t}'"))?,
        );
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// List distinct `nid` values from the parquet file (best-effort).
pub fn discover_nids(parquet_path: &Utf8PathBuf) -> Result<Vec<i32>> {
    let pl_path = PlPath::from_str(parquet_path.as_path().as_str());
    let lf = LazyFrame::scan_parquet(pl_path, ScanArgsParquet::default())
        .with_context(|| format!("failed to scan parquet: {}", parquet_path))?;

    let df = lf
        .select([col(cols::NID)])
        .unique(None, UniqueKeepStrategy::First)
        .sort(
            [cols::NID],
            SortMultipleOptions::default().with_maintain_order(true),
        )
        .collect()
        .context("failed to collect unique nid list")?;

    let s = df
        .column(cols::NID)
        .with_context(|| format!("missing column '{}' while discovering nids", cols::NID))?
        .cast(&DataType::Int32)
        .context("failed to cast nid to Int32")?;

    let ca = s.i32().context("nid is not Int32 after cast")?;
    Ok(ca.into_no_null_iter().collect())
}

/// Resolve the final list of nights:
/// - use `nids_csv` if provided, otherwise discover from parquet,
/// - apply `max_nights` if set,
/// - ensure non-empty.
pub fn resolve_nids(
    parquet_path: &Utf8PathBuf,
    nids_csv: Option<&str>,
    max_nights: Option<usize>,
) -> Result<Vec<i32>> {
    let mut nids = if let Some(s) = nids_csv {
        parse_nids_csv(s)?
    } else {
        discover_nids(parquet_path)?
    };

    if let Some(max_n) = max_nights {
        if nids.len() > max_n {
            nids.truncate(max_n);
        }
    }

    anyhow::ensure!(!nids.is_empty(), "no nights to process (empty nid list)");
    Ok(nids)
}

/// Ingest a single night into an [`AlertStoreWithTruth`].
pub fn ingest_one_night(
    source: &ParquetSource,
    nid: i32,
    mode: AlertLoadMode,
    minimal: bool,
    ingest_cfg: &AlertIngestConfig,
) -> Result<AlertStoreWithTruth> {
    let scan = ZtfAlertScan {
        nid: Some(nid),
        mode,
        minimal,
    };

    let lf = scan_ztf_alerts(source, scan)?;
    let store = alert_store_with_truth_from_lazyframe(lf, ingest_cfg.clone())
        .with_context(|| format!("failed to ingest nid={nid}"))?;

    Ok(store)
}

/// Infer a robust origin `t0` for uniform time bins from the dataset.
///
/// The uniform time binner computes bin indices as:
///
/// ```text
/// bin(t) = floor((t - t0) / dt)
/// ```
///
/// Choosing `t0 = min(mjd_tt)` makes the binning deterministic for a fixed
/// dataset, which is sufficient for post-hoc studies and diagnostic tools.
///
/// Arguments
/// ---------
/// * `store` – Ingested alerts and truth sidecar.
///
/// Return
/// ------
/// * `MjdTt` – The minimum `mjd_tt` found in the dataset.
///   If the store is empty or only contains non-finite times, returns `0.0`.
///
/// Notes
/// -----
/// * For production pipelines that compare results across datasets, you may
///   prefer a fixed global origin (e.g., a reference MJD) instead of `min(t)`.
pub fn infer_t0_mjd_tt(store: &AlertStoreWithTruth) -> MjdTt {
    store
        .store
        .alerts
        .iter()
        .map(|a| a.mjd_tt)
        .fold(f64::INFINITY, |acc, x| acc.min(x))
        .if_finite_or(0.0)
}