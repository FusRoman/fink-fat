//! Example: ingest ZTF-like alerts with truth association.
//!
//! Overview
//! --------
//! This example demonstrates how to:
//! - scan a ZTF-like Parquet alert dataset using [`scan_ztf_alerts`],
//! - materialize it into an [`AlertStoreWithTruth`] for evaluation,
//! - inspect global ingestion and truth statistics,
//! - iterate over alerts belonging to a given truth trajectory.
//!
//! The example assumes a Parquet file containing a `trajectory_id` column
//! (ground-truth association), typically available in simulated or
//! cross-matched evaluation datasets.
//!
//! Usage
//! -----
//! ```bash
//! cargo run --example ztf_alerts_with_truth
//! ```
//!
//! Notes
//! -----
//! - Paths are relative to the crate root; adjust as needed.
//! - Only alerts with `trajectory_id > 0` are kept in this example
//!   (`only_truth = true`).

use fink_fat_eval::dataset::{
    ParquetSource,
    ztf_alerts::{
        AlertLoadMode, ZtfAlertScan, alert_store_with_truth_from_lazyframe, scan_ztf_alerts
    },
};

fn main() -> anyhow::Result<()> {
    // ---------------------------------------------------------------------
    // Dataset configuration
    // ---------------------------------------------------------------------

    // Parquet dataset containing ZTF-like alerts with a `trajectory_id` column.
    let parquet_source =
        ParquetSource::new("../../test_exp/ztf_alert.parquet")?;

    // Configure a lazy scan:
    // - keep only alerts associated with a truth trajectory,
    // - keep minimal columns for ingestion.
    let scan = ZtfAlertScan {
        mode: AlertLoadMode::Oracle,
        ..Default::default()
    };

    // ---------------------------------------------------------------------
    // Lazy scan + ingestion
    // ---------------------------------------------------------------------

    // Build a lazy Polars plan (predicate + projection pushdown).
    let lf = scan_ztf_alerts(&parquet_source, scan)?;

    // Materialize into an engine store + truth sidecar.
    let alert_store =
        alert_store_with_truth_from_lazyframe(lf, Default::default())?;

    // ---------------------------------------------------------------------
    // Global summary
    // ---------------------------------------------------------------------

    // Display includes:
    // - engine AlertStore summary,
    // - truth association statistics (min / mean / max trajectory length).
    println!("{}", alert_store);

    // ---------------------------------------------------------------------
    // Inspect a few alerts
    // ---------------------------------------------------------------------

    let few_alerts = 10;

    println!("First {} alerts:", few_alerts);
    for alert in alert_store.store.iter().take(few_alerts) {
        println!(
            "alert {:?}: {} (trajectory_id: {})",
            alert.id,
            alert,
            alert_store.trajectory_id[alert.id.idx()],
        );
    }

    // ---------------------------------------------------------------------
    // Iterate over a single truth trajectory
    // ---------------------------------------------------------------------

    let traj_id = 33803;
    println!("\n===============\nAlerts for trajectory_id = {}:", traj_id);

    for alert in alert_store.alerts_for_trajectory(traj_id) {
        println!(
            "trajectory {} alert {:?}: {}",
            traj_id,
            alert.id,
            alert,
        );
    }

    Ok(())
}
