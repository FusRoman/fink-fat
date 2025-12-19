//! Example: pretty terminal display for seeding metrics (pairs & triplets).
//!
//! This example:
//! 1) scans a ZTF-like Parquet dataset (Polars lazy scan),
//! 2) materializes it into `AlertStoreWithTruth`,
//! 3) builds *oracle* seeds from truth (consecutive-by-time pairs/triplets),
//! 4) computes metrics and prints them as pretty terminal tables.
//!
//! Usage
//! -----
//! ```bash
//! cargo run --example metrics_terminal -- <path/to/ztf_alerts.parquet> [--only-truth]
//! ```
//!
//! Notes
//! -----
//! - "Oracle" seeds here are derived from truth and represent an upper bound.
//! - Replace the `build_oracle_*` calls with your real seeding generator
//!   once `seed_gen.rs` is wired (pairs/triplets from `fink-fat-engine`).

use anyhow::{Context, Result};
use camino::Utf8PathBuf;
use clap::Parser;

use fink_fat_engine::{
    AlertId,
    seeding::{
        pairs::{Pair, Pairs},
        triplets::{Triplet, Triplets},
    },
};

use fink_fat_eval::{
    dataset::{
        ParquetSource,
        ingest_config::AlertIngestConfig,
        ztf_alerts::{
            AlertStoreWithTruth, ZtfAlertScan, alert_store_with_truth_from_lazyframe,
            scan_ztf_alerts,
        },
    },
    seeding::metrics::{pair_metrics, triplet_metrics},
};

/// Pretty terminal display for oracle seeding metrics (pairs & triplets).
#[derive(Parser, Debug)]
#[command(
    name = "metrics-terminal",
    about = "Compute oracle (truth-consecutive) pair/triplet seeding metrics and print them to the terminal.",
    long_about = None
)]
struct Cli {
    /// Input ZTF-like alerts Parquet file.
    #[arg(value_name = "ALERTS.parquet")]
    parquet: Utf8PathBuf,

    /// Keep only alerts having a truth association (trajectory_id > 0).
    #[arg(long)]
    only_truth: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // ---------------------------------------------------------------------
    // Ingest Parquet -> AlertStoreWithTruth
    // ---------------------------------------------------------------------
    let parquet_path_str = cli.parquet.to_string();

    let parquet_source = ParquetSource::new(&cli.parquet)
        .with_context(|| format!("Failed to open parquet source: {parquet_path_str}"))?;

    let scan = ZtfAlertScan {
        only_truth: cli.only_truth,
        ..Default::default()
    };

    let lf = scan_ztf_alerts(&parquet_source, scan)?;
    let store: AlertStoreWithTruth =
        alert_store_with_truth_from_lazyframe(lf, AlertIngestConfig::default())?;

    println!("==============================");
    println!("Ingestion summary");
    println!("==============================");
    println!("{store}");

    // ---------------------------------------------------------------------
    // Build oracle seeds from truth (consecutive-by-time)
    // ---------------------------------------------------------------------
    let oracle_pairs = build_oracle_consecutive_pairs(&store);
    let oracle_triplets = build_oracle_consecutive_triplets(&store);

    // ---------------------------------------------------------------------
    // Compute + pretty-print metrics
    // ---------------------------------------------------------------------
    let pm = pair_metrics(&store, &oracle_pairs);
    let tm = triplet_metrics(&store, &oracle_triplets);

    println!("\n==============================");
    println!("Pair metrics (oracle truth-consecutive)");
    println!("==============================");
    println!("{pm}");

    println!("\n==============================");
    println!("Triplet metrics (oracle truth-consecutive)");
    println!("==============================");
    println!("{tm}");

    Ok(())
}

/// Build "oracle" pairs: consecutive truth neighbors by time, per trajectory.
///
/// Overview
/// --------
/// For each `trajectory_id > 0`:
/// - collect alert indices,
/// - sort by `mjd_tt`,
/// - emit pairs `(id[i], id[i+1])`.
///
/// Parameters
/// ----------
/// store : &AlertStoreWithTruth
///     Ingested alerts plus truth sidecar.
///
/// Returns
/// -------
/// Pairs
///     A list of consecutive-by-time truth pairs.
///     This is a useful upper-bound baseline (should yield high precision and recall).
fn build_oracle_consecutive_pairs(store: &AlertStoreWithTruth) -> Pairs {
    // Group dense alert indices by truth id.
    let mut by_tid: std::collections::HashMap<i32, Vec<usize>> = std::collections::HashMap::new();
    for (idx, &tid) in store.trajectory_id.iter().enumerate() {
        if tid > 0 {
            by_tid.entry(tid).or_default().push(idx);
        }
    }

    let mut out: Pairs = Vec::new();

    for ids in by_tid.values_mut() {
        if ids.len() < 2 {
            continue;
        }

        // Sort indices by time for robust ordering.
        ids.sort_by(|&i, &j| {
            let ti = store.store.alerts[i].mjd_tt;
            let tj = store.store.alerts[j].mjd_tt;
            ti.partial_cmp(&tj).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Consecutive windows of size 2.
        out.extend(ids.windows(2).map(|w| Pair {
            a: AlertId::from(w[0]),
            b: AlertId::from(w[1]),
        }));
    }

    // Determinism: sort + dedup in case of weird duplicates in truth.
    out.sort_unstable();
    out.dedup();
    out
}

/// Build "oracle" triplets: consecutive truth neighbors by time, per trajectory.
///
/// Overview
/// --------
/// For each `trajectory_id > 0`:
/// - collect alert indices,
/// - sort by `mjd_tt`,
/// - emit triplets `(id[i], id[i+1], id[i+2])`.
///
/// Parameters
/// ----------
/// store : &AlertStoreWithTruth
///     Ingested alerts plus truth sidecar.
///
/// Returns
/// -------
/// Triplets
///     A list of consecutive-by-time truth triplets.
fn build_oracle_consecutive_triplets(store: &AlertStoreWithTruth) -> Triplets {
    let mut by_tid: std::collections::HashMap<i32, Vec<usize>> = std::collections::HashMap::new();
    for (idx, &tid) in store.trajectory_id.iter().enumerate() {
        if tid > 0 {
            by_tid.entry(tid).or_default().push(idx);
        }
    }

    let mut out: Triplets = Vec::new();

    for ids in by_tid.values_mut() {
        if ids.len() < 3 {
            continue;
        }

        ids.sort_by(|&i, &j| {
            let ti = store.store.alerts[i].mjd_tt;
            let tj = store.store.alerts[j].mjd_tt;
            ti.partial_cmp(&tj).unwrap_or(std::cmp::Ordering::Equal)
        });

        out.extend(ids.windows(3).map(|w| Triplet {
            a: AlertId::from(w[0]),
            b: AlertId::from(w[1]),
            c: AlertId::from(w[2]),
        }));
    }

    out.sort_unstable();
    out.dedup();
    out
}
