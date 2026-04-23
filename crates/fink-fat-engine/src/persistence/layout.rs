//! Persistence layout (directory structure + file naming conventions).
//!
//! This module centralizes all on-disk paths used by the persistence layer.
//! It provides a single source of truth for:
//! - directory structure,
//! - file naming,
//! - per-night partitioning,
//! - edge storage using snapshot + per-night deltas.
//!
//! Design
//! ------
//! - Alerts are stored per night: `alerts/nid=<NightId>.bin`
//! - Seeds are stored per night:  `seeds/nid=<NightId>.bin`
//! - Edges are stored as:
//!   - a compacted snapshot:       `graph/snapshot.bin`
//!   - per-night delta chunks:     `graph/delta-nid=<NightId>.bin`
//! - A top-level manifest tracks what exists: `manifest.bin`
//!
//! Notes
//! -----
//! - All manifest paths are stored relative to the layout root to keep the
//!   state relocatable.
//! - This layout deliberately avoids deep sharding to keep I/O predictable.

use camino::{Utf8Path, Utf8PathBuf};

use crate::night_id::NightId;

/// Helper owning the root persistence directory and generating file paths.
///
/// Notes
/// -----
/// All returned paths are inside `root`.
#[derive(Clone, Debug)]
pub struct PersistenceLayout {
    root: Utf8PathBuf,
}

impl PersistenceLayout {
    /// Create a new layout rooted at `root`.
    pub fn new(root: impl Into<Utf8PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Root directory for all persistence artifacts.
    pub fn root(&self) -> &Utf8Path {
        &self.root
    }

    /// Path to the top-level manifest file.
    pub fn manifest_path(&self) -> Utf8PathBuf {
        self.root.join("manifest.json")
    }

    /// Path to the global observation dataset file.
    ///
    /// This file contains all observations across the current sliding window,
    /// serialized as a single [`photom::observation_dataset::ObsDataset`] payload
    /// wrapped in a [`crate::persistence::envelope::DiskEnvelope`].
    ///
    /// Example
    /// -------
    /// `obs_dataset.bin`
    pub fn obs_dataset_path(&self) -> Utf8PathBuf {
        self.root.join("obs_dataset.bin")
    }

    /// Directory containing per-night alerts.
    pub fn alerts_dir(&self) -> Utf8PathBuf {
        self.root.join("alerts")
    }

    /// Directory containing per-night seeds.
    pub fn seeds_dir(&self) -> Utf8PathBuf {
        self.root.join("seeds")
    }

    /// Directory containing edges (snapshot + deltas).
    pub fn graph_dir(&self) -> Utf8PathBuf {
        self.root.join("graph")
    }

    /// File path for alerts of one night.
    ///
    /// Example
    /// -------
    /// `alerts/nid=<NightId>.bin`
    pub fn alerts_night_path(&self, night_id: NightId) -> Utf8PathBuf {
        self.alerts_dir().join(format!("nid={night_id}.bin"))
    }

    /// File path for seeds of one night.
    ///
    /// Example
    /// -------
    /// `seeds/nid=<NightId>.bin`
    pub fn seeds_night_path(&self, night_id: NightId) -> Utf8PathBuf {
        self.seeds_dir().join(format!("nid={night_id}.bin"))
    }

    // -------------------------------------------------------------------------
    // Edges: snapshot + delta (one delta file per night)
    // -------------------------------------------------------------------------

    /// File path to the compacted edge snapshot.
    ///
    /// Example
    /// -------
    /// `graph/snapshot.bin`
    pub fn graph_snapshot_path(&self) -> Utf8PathBuf {
        self.graph_dir().join("snapshot.bin")
    }

    /// File path to the per-night edge delta chunk.
    ///
    /// Example
    /// -------
    /// `graph/delta-nid=<NightId>.bin`
    pub fn graph_delta_night_path(&self, night_id: NightId) -> Utf8PathBuf {
        self.graph_dir().join(format!("delta-nid={night_id}.bin"))
    }

    // -------------------------------------------------------------------------
    // Orbit exports (Parquet files, one set per night)
    // -------------------------------------------------------------------------

    /// Directory containing orbit export Parquet files.
    pub fn orbits_dir(&self) -> Utf8PathBuf {
        self.root.join("orbits")
    }

    /// Path to the track-members Parquet file for one night.
    ///
    /// This file maps each trajectory (`track_id`) to its constituent alerts
    /// (`dia_source_id`), enabling downstream joins.
    ///
    /// Example
    /// -------
    /// `orbits/track_members-nid=<NightId>.parquet`
    pub fn track_members_night_path(&self, night_id: NightId) -> Utf8PathBuf {
        self.orbits_dir()
            .join(format!("track_members-nid={night_id}.parquet"))
    }

    /// Path to the orbital-parameters Parquet file for one night.
    ///
    /// This file contains one row per trajectory with Keplerian orbital elements,
    /// reference epoch, RMS, and orbit type.
    ///
    /// Example
    /// -------
    /// `orbits/orbital_params-nid=<NightId>.parquet`
    pub fn orbital_params_night_path(&self, night_id: NightId) -> Utf8PathBuf {
        self.orbits_dir()
            .join(format!("orbital_params-nid={night_id}.parquet"))
    }

    // -------------------------------------------------------------------------
    // Log files (one per pipeline run, named by session timestamp)
    // -------------------------------------------------------------------------

    /// Directory containing per-run log files.
    pub fn logs_dir(&self) -> Utf8PathBuf {
        self.root.join("logs")
    }

    /// Path to the log file for one pipeline run.
    ///
    /// `run_id` is typically a UTC datetime string formatted as
    /// `YYYY-MM-DDTHH-MM-SS`, produced by the CLI before the pipeline starts.
    ///
    /// Example
    /// -------
    /// `logs/run-2026-03-04T14-30-00.log`
    pub fn log_run_path(&self, run_id: &str) -> Utf8PathBuf {
        self.logs_dir().join(format!("run-{run_id}.log"))
    }

    // -------------------------------------------------------------------------
    // Relative path helpers (manifest storage)
    // -------------------------------------------------------------------------

    /// Convert an absolute path inside this layout to a root-relative utf8-path.
    ///
    /// Returns `None` if the provided path is not under `root`.
    pub fn to_relative(&self, path: &Utf8Path) -> Option<Utf8PathBuf> {
        path.strip_prefix(self.root()).ok().map(|p| p.to_path_buf())
    }

    /// Resolve a root-relative utf8-path into an absolute path under `root`.
    pub fn resolve_relative(&self, rel: &Utf8Path) -> Utf8PathBuf {
        self.root.join(rel)
    }
}
