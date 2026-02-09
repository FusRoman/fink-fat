//! Persistence manifest for Fink-FAT.
//!
//! The manifest is a small index file describing which persistence artifacts
//! exist on disk and how to load them efficiently.
//!
//! This version integrates the **edge journal** strategy:
//! - a compacted **snapshot** containing a full edge view at some checkpoint,
//! - a sequence of per-night **delta files** containing edge operations since
//!   the snapshot.
//!
//! Load strategy
//! ------------
//! - Load the snapshot (if present).
//! - Apply all delta files in chronological order.
//! - Optionally only apply deltas inside a sliding window.
//!
//! Compaction strategy (planned)
//! ----------------------------
//! Periodically:
//! - rebuild the current edge state from snapshot + deltas,
//! - write a new snapshot,
//! - drop/compact old deltas.

use serde::{Deserialize, Serialize};

use camino::{Utf8Path, Utf8PathBuf};

use crate::{
    engine_config::EngineConfig,
    night_id::NightId,
    persistence::{
        edge::edge_journal::NightWindow, error::PersistenceIoError, layout::PersistenceLayout,
    },
};

use super::{
    ALERT_STORE_SCHEMA_VERSION, EDGE_JOURNAL_SCHEMA_VERSION, GRAPH_SCHEMA_VERSION,
    SEED_STORE_SCHEMA_VERSION, STATE_SCHEMA_VERSION, envelope::DiskEnvelope,
};

/// Entry describing the on-disk artifacts for a single night.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NightManifestEntry {
    /// Night identifier.
    pub night_id: NightId,

    /// Relative path to the alerts file for this night.
    alerts_relpath: String,

    /// Relative path to the seeds file for this night.
    seeds_relpath: String,

    /// Optional counts for sanity checks.
    pub n_alerts: Option<u64>,
    pub n_seeds: Option<u64>,
}

impl NightManifestEntry {
    pub fn new(
        night_id: NightId,
        alerts_relpath: Utf8PathBuf,
        seeds_relpath: Utf8PathBuf,
        n_alerts: Option<u64>,
        n_seeds: Option<u64>,
    ) -> Self {
        Self {
            night_id,
            alerts_relpath: alerts_relpath.to_string(),
            seeds_relpath: seeds_relpath.to_string(),
            n_alerts,
            n_seeds,
        }
    }

    #[inline]
    pub fn alerts_rel_path(&self) -> &Utf8Path {
        Utf8Path::new(&self.alerts_relpath)
    }

    #[inline]
    pub fn seeds_rel_path(&self) -> &Utf8Path {
        Utf8Path::new(&self.seeds_relpath)
    }

    #[inline]
    /// Get the absolute path to the alerts file for this night.
    pub fn alerts_abs_path(&self, layout: &PersistenceLayout) -> Utf8PathBuf {
        layout.resolve_relative(self.alerts_rel_path())
    }

    #[inline]
    /// Get the absolute path to the seeds file for this night.
    pub fn seeds_abs_path(&self, layout: &PersistenceLayout) -> Utf8PathBuf {
        layout.resolve_relative(self.seeds_rel_path())
    }
}

/// Optional info about the ML model used to build edges (if relevant).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelManifestEntry {
    /// Relative or absolute path to the ONNX model used by the runtime.
    pub model_path: String,
    /// Optional sha256 checksum (hex) to detect mismatch.
    pub model_sha256: Option<String>,
}

/// One delta segment for the edge journal.
///
/// In the "one file per night" design, each entry maps to:
/// `edges/delta-nid=<NightId>.bin`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EdgeDeltaEntry {
    /// Night that produced this delta.
    pub night_id: NightId,

    /// Relative path to the delta file.
    delta_relpath: String,

    /// Optional number of operations stored in the delta.
    pub n_ops: Option<u64>,
}

impl EdgeDeltaEntry {
    pub fn new(night_id: NightId, delta_relpath: Utf8PathBuf, n_ops: Option<u64>) -> Self {
        Self {
            night_id,
            delta_relpath: delta_relpath.to_string(),
            n_ops,
        }
    }

    #[inline]
    /// Get the absolute path to the delta file for this entry.
    pub fn delta_abs_path(&self, layout: &PersistenceLayout) -> Utf8PathBuf {
        layout.resolve_relative(Utf8Path::new(&self.delta_relpath))
    }

    #[inline]
    pub fn delta_rel_path(&self) -> &Utf8Path {
        Utf8Path::new(&self.delta_relpath)
    }
}

/// Edge journal manifest (snapshot + deltas).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EdgeJournalManifest {
    /// Schema version for the edge journal payloads.
    pub schema_version: u32,

    /// Optional relative path to the compacted snapshot.
    ///
    /// If `None`, the journal starts from an empty baseline and deltas build
    /// the full state.
    snapshot_relpath: Option<String>,

    /// Night ID that the snapshot represents (checkpoint).
    ///
    /// If present, deltas should have `night_id > snapshot_night_id` (typically).
    pub snapshot_night_id: Option<NightId>,

    /// Ordered list of delta files (typically one per night).
    pub deltas: Vec<EdgeDeltaEntry>,
}

impl EdgeJournalManifest {
    /// Create an empty edge journal manifest.
    pub fn new() -> Self {
        Self {
            schema_version: EDGE_JOURNAL_SCHEMA_VERSION,
            snapshot_relpath: None,
            snapshot_night_id: None,
            deltas: Vec::new(),
        }
    }

    /// Check if the journal is empty (no snapshot and no deltas).
    pub fn is_empty(&self) -> bool {
        self.snapshot_relpath.is_none() && self.deltas.is_empty()
    }

    /// Get the absolute path to the snapshot file, if it exists.
    #[inline]
    pub fn snapshot_abs_path(&self, layout: &PersistenceLayout) -> Option<Utf8PathBuf> {
        self.snapshot_relpath
            .as_ref()
            .map(|rel| layout.resolve_relative(Utf8Path::new(rel)))
    }

    #[inline]
    pub fn snapshot_rel_path(&self) -> Option<&Utf8Path> {
        self.snapshot_relpath.as_ref().map(|rel| Utf8Path::new(rel))
    }

    /// Set or replace the snapshot metadata.
    pub fn set_snapshot(&mut self, relpath: Utf8PathBuf, snapshot_night_id: NightId) {
        self.snapshot_relpath = Some(relpath.to_string());
        self.snapshot_night_id = Some(snapshot_night_id);
    }

    /// Insert or replace a delta entry for one night.
    ///
    /// This keeps deltas sorted by `night_id`.
    pub fn upsert_delta(&mut self, entry: EdgeDeltaEntry) {
        if let Some(pos) = self
            .deltas
            .iter()
            .position(|e| e.night_id == entry.night_id)
        {
            self.deltas[pos] = entry;
        } else {
            self.deltas.push(entry);
        }
        self.deltas.sort_by_key(|e| e.night_id);
    }

    /// Drop all deltas up to and including `night_id` (useful after compaction).
    pub fn drop_deltas_leq(&mut self, night_id: NightId) {
        self.deltas.retain(|e| e.night_id > night_id);
    }
}

/// Top-level persistence manifest.
///
/// Notes
/// -----
/// - Paths are stored as strings relative to the persistence root.
/// - This keeps the state relocatable (moving the `state/` directory is safe).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    /// Unix timestamp (seconds) when the manifest was created/updated.
    pub created_unix_s: i64,

    /// Schema versions used when writing the state.
    pub alert_store_schema_version: u32,
    pub seed_store_schema_version: u32,
    pub graph_schema_version: u32,
    pub state_schema_version: u32,

    /// One entry per night available in the state.
    pub nights: Vec<NightManifestEntry>,

    /// Edge storage as a journal (snapshot + deltas).
    pub edge_journal: EdgeJournalManifest,

    /// Optional ML model metadata.
    pub model: Option<ModelManifestEntry>,
}

impl Manifest {
    /// Create a new manifest with current schema versions.
    pub fn new(created_unix_s: i64) -> Self {
        Self {
            created_unix_s,
            alert_store_schema_version: ALERT_STORE_SCHEMA_VERSION,
            seed_store_schema_version: SEED_STORE_SCHEMA_VERSION,
            graph_schema_version: GRAPH_SCHEMA_VERSION,
            state_schema_version: STATE_SCHEMA_VERSION,
            nights: Vec::new(),
            edge_journal: EdgeJournalManifest::new(),
            model: None,
        }
    }

    /// Return the maximum night present in the persisted state.
    ///
    /// Notes
    /// -----
    /// This uses `nights` (alerts/seeds presence) as the canonical timeline.
    /// If you prefer using edge deltas, switch to `edge_journal.deltas`.
    pub fn max_night_id(&self) -> Option<NightId> {
        self.nights.iter().map(|e| e.night_id).max()
    }

    /// Compute the sliding window implied by `engine_config.max_gap_nights`.
    ///
    /// Window definition
    /// -----------------
    /// Let `G = max_gap_nights`. For the current max night `Nmax`, we keep:
    /// - `min_night = Nmax - G`
    /// - `max_night = Nmax`
    ///
    /// This is sufficient to reconstruct and solve the inter-night graph under
    /// the assumption that no edge spans more than `G` nights.
    pub fn compute_edge_window_from_config(
        &self,
        engine_config: &EngineConfig,
    ) -> Option<NightWindow> {
        let max_night = self.max_night_id()?;
        let g = engine_config.max_gap_nights() as u32;

        let min_night = NightId(max_night.0.saturating_sub(g));

        Some(NightWindow {
            min_night,
            max_night,
        })
    }

    /// Add or replace the entry for one night.
    ///
    /// If an entry with the same `night_id` exists, it is replaced.
    pub fn upsert_night(&mut self, entry: NightManifestEntry) {
        if let Some(pos) = self
            .nights
            .iter()
            .position(|e| e.night_id == entry.night_id)
        {
            self.nights[pos] = entry;
        } else {
            self.nights.push(entry);
        }
        self.nights.sort_by_key(|e| e.night_id);
    }

    /// Persist the manifest to disk using a `DiskEnvelope<Manifest>`.
    pub fn save(&self, path: &Utf8Path) -> Result<(), PersistenceIoError> {
        let env = DiskEnvelope::new(self.clone(), STATE_SCHEMA_VERSION, self.created_unix_s);
        env.save_enveloped(path)
    }

    /// Load and validate the manifest from disk.
    pub fn load(path: &Utf8Path) -> Result<Self, PersistenceIoError> {
        DiskEnvelope::<Manifest>::load_enveloped(path, STATE_SCHEMA_VERSION)
    }
}
