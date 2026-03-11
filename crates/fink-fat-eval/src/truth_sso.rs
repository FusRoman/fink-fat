//! Ground-truth SSO (Solar System Object) identity map.
//!
//! This module provides the [`TruthSSOMap`] type and the [`load_truth_sso_map`]
//! loader used by all evaluation sub-commands to ground-truth alert identities.
//!
//! ## Data model
//!
//! The truth table is a Parquet file with (at minimum) two columns:
//!
//! | Column            | Type     | Description                                           |
//! |-------------------|----------|-------------------------------------------------------|
//! | `dia_source_id`   | `uint64` | Unique detection identifier (matches [`DiaSourceId`]) |
//! | `trajectory_id`   | `int32`  | Ground-truth trajectory / object identifier           |
//!
//! [`load_truth_sso_map`] reads only these two columns and builds an
//! [`AHashMap`](ahash::AHashMap) for O(1) lookups during post-processing.

use anyhow::{Context, Result};
use camino::Utf8Path;
use fink_fat_engine::{Alert, alerts::DiaSourceId};
use polars::prelude::*;

/// Ground-truth trajectory identifier.
pub type TrajId = u32;

/// Map from [`DiaSourceId`] to ground-truth [`TrajId`].
///
/// Built once at startup from the truth Parquet file and then passed
/// read-only to every evaluation post-processing function.
pub type TruthSSOMap = ahash::AHashMap<DiaSourceId, TrajId>;

/// Load a [`TruthSSOMap`] from a Parquet truth file.
///
/// Only the `dia_source_id` and `trajectory_id` columns are read; all other
/// columns are ignored.
///
/// Arguments
/// ---------
/// * `path` – Path to the Parquet file containing the truth table.
///
/// Return
/// ------
/// * `Ok(TruthSSOMap)` – Map populated with one entry per row.
/// * `Err(...)` – On I/O failure, missing columns, or type mismatch.
pub fn load_truth_sso_map(path: &Utf8Path) -> Result<TruthSSOMap> {
    let df = LazyFrame::scan_parquet(path.as_str(), ScanArgsParquet::default())
        .context("failed to open truth parquet file")?
        .select([col("dia_source_id"), col("trajectory_id")])
        .collect()
        .context("failed to collect truth parquet columns")?;

    let ids = df
        .column("dia_source_id")
        .context("missing column 'dia_source_id'")?
        .u64()
        .context("'dia_source_id' is not uint64")?;

    let trajs = df
        .column("trajectory_id")
        .context("missing column 'trajectory_id'")?
        .i32()
        .context("'trajectory_id' is not int32")?;

    let mut map = TruthSSOMap::with_capacity(ids.len());
    for (id, traj) in ids.iter().zip(trajs.iter()) {
        if let (Some(id), Some(traj)) = (id, traj) {
            map.insert(id, traj as TrajId);
        }
    }

    Ok(map)
}

pub fn get_truth_traj_id(truth_map: &TruthSSOMap, alert: &Alert) -> Option<TrajId> {
    truth_map.get(&alert.key.dia_source_id).copied()
}

/// Classification of a seed with respect to the ground-truth map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedClass {
    /// All member alerts belong to the same ground-truth trajectory.
    TruePositive,
    /// Member alerts belong to at least two distinct ground-truth trajectories.
    FalsePositive,
    /// At least one member alert is absent from the truth map.
    Unknown,
}

/// Classify a resolved seed slice against the ground-truth map.
///
/// Arguments
/// ---------
/// * `truth_map` – Ground-truth [`DiaSourceId`] → [`TrajId`] map.
/// * `alerts`    – Resolved member alerts (borrowed from the alert store).
///
/// Return
/// ------
/// [`SeedClass`] indicating whether the seed is a true positive, false
/// positive, or has at least one alert missing from the truth map.
pub fn classify_seed(truth_map: &TruthSSOMap, alerts: &[&Alert]) -> SeedClass {
    let mut first_id: Option<TrajId> = None;
    for alert in alerts {
        match truth_map.get(&alert.key.dia_source_id).copied() {
            None => return SeedClass::Unknown,
            Some(traj_id) => match first_id {
                None => first_id = Some(traj_id),
                Some(ref expected) if *expected != traj_id => return SeedClass::FalsePositive,
                _ => {}
            },
        }
    }
    if first_id.is_some() {
        SeedClass::TruePositive
    } else {
        // Empty slice — treat as unknown.
        SeedClass::Unknown
    }
}

/// Returns `true` iff all alerts in the slice share the same ground-truth
/// trajectory and that trajectory is present in the truth map.
///
/// Arguments
/// ---------
/// * `truth_map` – Ground-truth map.
/// * `alerts`    – Resolved member alerts.
pub fn is_true_positive(truth_map: &TruthSSOMap, alerts: &[&Alert]) -> bool {
    matches!(classify_seed(truth_map, alerts), SeedClass::TruePositive)
}
