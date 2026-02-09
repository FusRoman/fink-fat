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
        self.root.join("manifest.bin")
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
