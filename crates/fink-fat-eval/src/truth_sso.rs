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

use ahash::AHashMap;
use anyhow::{Context, Result};
use camino::Utf8Path;
use fink_fat_engine::{Alert, alerts::DiaSourceId, night_id::NightId};
use polars::prelude::*;

/// Ground-truth trajectory identifier.
pub type TrajId = u32;

/// Map from [`DiaSourceId`] to ground-truth [`TrajId`].
///
/// Built once at startup from the truth Parquet file and then passed
/// read-only to every evaluation post-processing function.
pub type TruthSSOMap = AHashMap<DiaSourceId, TrajId>;

/// For each trajectory ID, count the number of alerts per night.
pub type TrajCountMap = AHashMap<TrajId, AHashMap<NightId, usize>>;

/// In-memory representation of the truth SSO map and related pre-computed data.
pub struct TruthSSO {
    /// Map from `dia_source_id` to `trajectory_id`.
    map: TruthSSOMap,
    /// Pre-computed count of alerts per trajectory per night, used for computing recoverable trajectories.
    traj_count: TrajCountMap,
}

impl TruthSSO {
    /// Load the truth SSO map and pre-compute trajectory counts from a Parquet file.
    ///
    /// Arguments
    /// ---------
    /// * `path` – Path to the Parquet file containing the truth table.
    ///
    /// Returns
    /// -------
    /// * `Ok(TruthSSO)` – Loaded truth SSO map and trajectory counts.
    /// * `Err(...)` – On I/O failure, missing columns, or type mismatch
    pub fn load(path: &Utf8Path) -> Result<Self> {
        let df = load_dataframe(path)?;
        let alert_ids = get_dia_source_id(&df)?;
        let trajs = get_trajectory_id(&df)?;
        let night_ids = get_night_id(&df)?; // Not currently used, but could be useful for validation or future features.

        let nb_uniq_trajs = trajs
            .n_unique()
            .context("failed to compute number of unique trajectories")?;

        let mut map = TruthSSOMap::with_capacity(alert_ids.len());
        let mut traj_alert_count: TrajCountMap = AHashMap::with_capacity(nb_uniq_trajs);

        for ((alert_id, traj), night_id) in alert_ids.iter().zip(trajs.iter()).zip(night_ids.iter())
        {
            if let (Some(id), Some(traj), Some(night_id)) = (alert_id, traj, night_id) {
                map.insert(id, traj as TrajId);
                let traj_entry = traj_alert_count
                    .entry(traj as TrajId)
                    .or_insert_with(AHashMap::new);
                *traj_entry.entry(night_id.into()).or_insert(0) += 1;
            }
        }

        Ok(Self {
            map,
            traj_count: traj_alert_count,
        })
    }

    /// Look up the ground-truth trajectory ID for a given alert.
    ///
    /// Arguments
    /// ---------
    /// * `alert` – The alert for which to look up the trajectory ID.
    ///
    /// Returns
    /// -------
    /// * `Some(traj_id)` – If the alert's `dia_source_id` is present in the truth map, returns the corresponding trajectory ID.
    /// * `None` – If the alert's `dia_source_id` is not present in the truth map.
    pub fn get_truth_traj_id(&self, alert: &Alert) -> Option<TrajId> {
        self.map.get(&alert.key.dia_source_id).copied()
    }

    /// Classify a resolved seed slice against the ground-truth map.
    ///
    /// Arguments
    /// ---------
    /// * `alerts` – Resolved member alerts (borrowed from the alert store).
    ///
    /// Return
    /// ------
    /// [`TruthClass`] indicating whether the seed is a true positive, false
    /// positive, or has at least one alert missing from the truth map.
    pub fn classify(&self, alerts: &[&Alert]) -> TruthClass {
        let mut first_id: Option<TrajId> = None;
        for alert in alerts {
            match self.map.get(&alert.key.dia_source_id).copied() {
                None => return TruthClass::Unknown,
                Some(traj_id) => match first_id {
                    None => first_id = Some(traj_id),
                    Some(ref expected) if *expected != traj_id => return TruthClass::FalsePositive,
                    _ => {}
                },
            }
        }
        if first_id.is_some() {
            TruthClass::TruePositive
        } else {
            // Empty slice — treat as unknown.
            TruthClass::Unknown
        }
    }

    /// Returns `true` iff all alerts in the slice share the same ground-truth
    /// trajectory and that trajectory is present in the truth map.
    ///
    /// Arguments
    /// ---------
    /// * `alerts`    – Resolved member alerts.
    pub fn is_true_positive(&self, alerts: &[&Alert]) -> bool {
        matches!(self.classify(alerts), TruthClass::TruePositive)
    }

    /// Iterator over `(traj_id, night_id, count)` tuples from a [`TrajCountMap`].
    ///
    /// This is a convenience for iterating over all trajectory-night pairs without having to deal with the nested map structure.
    ///
    /// Returns
    /// -------
    /// An iterator yielding `(traj_id, night_id, count)` tuples for each trajectory and night where the trajectory has at least one alert.
    pub fn traj_count_iter(&self) -> impl Iterator<Item = (TrajId, NightId, usize)> {
        self.traj_count.iter().flat_map(|(&traj_id, night_counts)| {
            night_counts
                .iter()
                .map(move |(&night_id, &count)| (traj_id, night_id, count))
        })
    }

    /// Get the count of alerts for a given trajectory ID and night ID.
    ///
    /// Arguments
    /// ---------
    /// * `traj_id`  – The trajectory ID for which to get the count.
    /// * `night_id` – The night ID for which to get the count.
    ///
    /// Returns
    /// -------
    /// The number of alerts associated with the given trajectory ID on the given night ID,
    /// or 0 if the trajectory or night is not present in the map.
    pub fn traj_count_for_night(&self, traj_id: TrajId, night_id: NightId) -> usize {
        self.traj_count
            .get(&traj_id)
            .and_then(|night_counts| night_counts.get(&night_id))
            .copied()
            .unwrap_or(0)
    }

    /// Get the total count of alerts for a given trajectory ID across all nights.
    ///
    /// Arguments
    /// ---------
    /// * `traj_id`  – The trajectory ID for which to get the total count.
    ///
    /// Returns
    /// -------
    /// The total number of alerts associated with the given trajectory ID across all nights,
    /// or 0 if the trajectory is not present in the map.
    pub fn traj_count_for_traj(&self, traj_id: TrajId) -> usize {
        self.traj_count
            .get(&traj_id)
            .map(|night_counts| night_counts.values().sum())
            .unwrap_or(0)
    }

    /// Get an iterator over trajectory IDs that have at least `night_count` alerts on the specified night.
    ///
    /// Arguments
    /// ---------
    /// * `night_id`   – The night ID for which to filter trajectories.
    /// * `night_count` – The minimum number of alerts a trajectory must have on the specified night to be included in the output.
    ///
    /// Returns
    /// -------
    /// An iterator yielding trajectory IDs that have at least `night_count` alerts on the specified night.
    /// This is used to identify "recoverable" trajectories for seeding evaluation.
    pub fn recoverable_seeds(
        &self,
        night_id: NightId,
        night_count: usize,
    ) -> impl Iterator<Item = TrajId> {
        self.traj_count
            .iter()
            .filter_map(move |(&traj_id, night_counts)| {
                let count = night_counts.get(&night_id).copied().unwrap_or(0);
                (count >= night_count).then_some(traj_id)
            })
    }

    /// Iterate over all detectable edges for the given seeding parameters.
    ///
    /// An edge connects two consecutive seeded nights of the same trajectory.
    /// A night is *seeded* when it contains at least `night_count` alerts for
    /// that trajectory.  Two seeded nights form an edge when their gap is
    /// ≤ `max_gap` nights.  Each edge is uniquely identified by
    /// `(traj_id, night_from, night_to)`.
    ///
    /// Arguments
    /// ---------
    /// * `night_count` – Minimum number of alerts on a single night to form a seed.
    /// * `max_gap`     – Maximum allowed gap (in nights) between two consecutive
    ///                   seeds for an edge to exist.
    ///
    /// Return
    /// ------
    /// An iterator yielding one `(traj_id, night_from, night_to)` tuple per
    /// detectable edge, where `night_from < night_to` and
    /// `night_to - night_from ≤ max_gap`.
    pub fn recoverable_edges(
        &self,
        night_count: usize,
        max_gap: u8,
    ) -> impl Iterator<Item = (TrajId, NightId, NightId)> + '_ {
        self.traj_count
            .iter()
            .flat_map(move |(&traj_id, night_counts)| {
                // Collect nights that have enough alerts to form a seed, sorted ascending.
                let mut seeded_nights: Vec<NightId> = night_counts
                    .iter()
                    .filter(|(_, count)| **count >= night_count)
                    .map(|(&night_id, _)| night_id)
                    .collect();

                seeded_nights.sort();

                // Emit one edge per consecutive pair within max_gap.
                seeded_nights
                    .windows(2)
                    .filter(|w| (w[1].0 - w[0].0) <= max_gap as u32)
                    .map(|w| (traj_id, w[0], w[1]))
                    .collect::<Vec<_>>()
            })
    }
}

/// Load a DataFrame from a Parquet file.
///
/// The required columns are `dia_source_id` (uint64), `trajectory_id` (int32), and `night_id` (uint32).
/// Only these three columns are read; all other columns in the Parquet file are ignored.
///
/// Arguments
/// ---------
/// * `path` – Path to the Parquet file containing the truth table.
///
/// Returns
/// -------
/// * `Ok(DataFrame)` – DataFrame containing the data.
/// * `Err(...)` – On I/O failure, missing columns, or type mismatch.
pub fn load_dataframe(path: &Utf8Path) -> Result<DataFrame> {
    LazyFrame::scan_parquet(path.as_str(), ScanArgsParquet::default())
        .context("failed to open truth parquet file")?
        .select([col("dia_source_id"), col("trajectory_id"), col("night_id")])
        .collect()
        .context("failed to collect truth parquet columns")
}

/// Extract the `dia_source_id` column as a `UInt64Chunked`.
///
/// Arguments
/// ---------
/// * `df` – DataFrame containing the truth data.
///
/// Returns
/// -------
/// * `Ok(&UInt64Chunked)` – The `dia_source_id` column as a `UInt64Chunked`.
/// * `Err(...)` – If the column is missing or has the wrong type.
pub fn get_dia_source_id(df: &DataFrame) -> Result<&UInt64Chunked> {
    df.column("dia_source_id")
        .context("missing column 'dia_source_id'")?
        .u64()
        .context("'dia_source_id' is not uint64")
}

/// Extract the `trajectory_id` column as an `Int32Chunked`.
///
/// Arguments
/// ---------
/// * `df` – DataFrame containing the truth data.
///
/// Returns
/// -------
/// * `Ok(&Int32Chunked)` – The `trajectory_id` column as an `Int32Chunked`.
/// * `Err(...)` – If the column is missing or has the wrong type.
pub fn get_trajectory_id(df: &DataFrame) -> Result<&Int32Chunked> {
    df.column("trajectory_id")
        .context("missing column 'trajectory_id'")?
        .i32()
        .context("'trajectory_id' is not int32")
}

/// Extract the `night_id` column as a `UInt32Chunked`.
///
/// Arguments
/// ---------
/// * `df` – DataFrame containing the truth data.
///
/// Returns
/// -------
/// * `Ok(&UInt32Chunked)` – The `night_id` column as a `UInt32Chunked`.
/// * `Err(...)` – If the column is missing or has the wrong type.
pub fn get_night_id(df: &DataFrame) -> Result<&UInt32Chunked> {
    df.column("night_id")
        .context("missing column 'night_id'")?
        .u32()
        .context("'night_id' is not uint32")
}

/// Classification of a seed with respect to the ground-truth map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruthClass {
    /// All member alerts belong to the same ground-truth trajectory.
    TruePositive,
    /// Member alerts belong to at least two distinct ground-truth trajectories.
    FalsePositive,
    /// At least one member alert is absent from the truth map.
    Unknown,
}
