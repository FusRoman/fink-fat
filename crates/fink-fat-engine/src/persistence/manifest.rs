//! Persistence manifest for Fink-FAT.
//!
//! The manifest is a small JSON index file (wrapped in a
//! [`DiskEnvelope`](crate::persistence::envelope::DiskEnvelope)) that
//! describes which persistence artifacts exist on disk and how to reassemble
//! the pipeline state efficiently at startup.
//!
//! ## Edge journal strategy
//!
//! Edge data is stored as a **journal** composed of two layers:
//!
//! - a compacted **snapshot** — a full serialized edge view taken at some
//!   past checkpoint,
//! - a sequence of per-night **delta files** — each containing edge
//!   operations added since the snapshot.
//!
//! ## Load strategy
//!
//! 1. Load the snapshot (if present).
//! 2. Apply all delta files in chronological order.
//! 3. Optionally restrict to deltas inside a sliding window (controlled by
//!    [`EngineConfig::max_gap_nights`](crate::engine_config::EngineConfig)).
//!
//! ## Compaction strategy (planned)
//!
//! Periodically:
//!
//! 1. Rebuild the current edge state from snapshot + deltas.
//! 2. Write a new snapshot.
//! 3. Drop or archive outdated delta files.
//!
//! ## Main types
//!
//! - [`Manifest`] — top-level index; holds references to all per-night
//!   artifacts and the edge journal.
//! - [`NightManifestEntry`] — per-night record pointing to the alerts and
//!   seeds files for that observation night.
//! - [`EdgeJournalManifest`] — tracks the snapshot and the ordered list of
//!   delta files that together represent the current edge state.
//! - [`EdgeDeltaEntry`] — one entry in the edge journal, pointing to a
//!   single-night delta file.
//! - [`ModelManifestEntry`] — optional metadata about the ML model used when
//!   building edges.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use camino::{Utf8Path, Utf8PathBuf};

use crate::{
    engine_config::EngineConfig,
    error::FinkFatError,
    night_id::{NightId, PairingMode},
    persistence::{error::PersistenceIoError, layout::PersistenceLayout},
};

use super::{
    ALERT_STORE_SCHEMA_VERSION, EDGE_JOURNAL_SCHEMA_VERSION, GRAPH_SCHEMA_VERSION,
    SEED_STORE_SCHEMA_VERSION, STATE_SCHEMA_VERSION, compression::Compression,
    envelope::DiskEnvelope,
};

/// Entry describing the on-disk artifacts for a single observation night.
///
/// Each entry in [`Manifest::nights`] corresponds to one night and records
/// the relative paths to the alerts and seeds files for that night, along
/// with optional record counts used for sanity checks at load time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NightManifestEntry {
    /// Night identifier.
    pub night_id: NightId,

    /// Relative path to the alerts file for this night.
    alerts_relpath: String,

    /// Relative path to the seeds file for this night.
    seeds_relpath: String,

    /// Optional count of alert records stored in the file (sanity check).
    pub n_alerts: Option<u64>,
    /// Optional count of seed records stored in the file (sanity check).
    pub n_seeds: Option<u64>,
}

impl NightManifestEntry {
    /// Create a new [`NightManifestEntry`].
    ///
    /// Arguments
    /// ---------
    /// * `night_id` — Identifier for the observation night.
    /// * `alerts_relpath` — Path to the alerts file, relative to the
    ///   persistence root.
    /// * `seeds_relpath` — Path to the seeds file, relative to the
    ///   persistence root.
    /// * `n_alerts` — Optional count of alert records (used for sanity checks
    ///   at load time; pass `None` if unknown).
    /// * `n_seeds` — Optional count of seed records (used for sanity checks at
    ///   load time; pass `None` if unknown).
    ///
    /// Return
    /// ------
    /// A new [`NightManifestEntry`].
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

    /// Return the relative path to the alerts file for this night.
    ///
    /// Return
    /// ------
    /// A [`camino::Utf8Path`] slice into the stored path string.
    #[inline]
    pub fn alerts_rel_path(&self) -> &Utf8Path {
        Utf8Path::new(&self.alerts_relpath)
    }

    /// Return the relative path to the seeds file for this night.
    ///
    /// Return
    /// ------
    /// A [`camino::Utf8Path`] slice into the stored path string.
    #[inline]
    pub fn seeds_rel_path(&self) -> &Utf8Path {
        Utf8Path::new(&self.seeds_relpath)
    }

    /// Resolve the absolute path to the alerts file for this night.
    ///
    /// Arguments
    /// ---------
    /// * `layout` — Persistence layout providing the root directory against
    ///   which the relative path is resolved.
    ///
    /// Return
    /// ------
    /// Absolute [`camino::Utf8PathBuf`] to the alerts file.
    #[inline]
    pub fn alerts_abs_path(&self, layout: &PersistenceLayout) -> Utf8PathBuf {
        layout.resolve_relative(self.alerts_rel_path())
    }

    /// Resolve the absolute path to the seeds file for this night.
    ///
    /// Arguments
    /// ---------
    /// * `layout` — Persistence layout providing the root directory against
    ///   which the relative path is resolved.
    ///
    /// Return
    /// ------
    /// Absolute [`camino::Utf8PathBuf`] to the seeds file.
    #[inline]
    pub fn seeds_abs_path(&self, layout: &PersistenceLayout) -> Utf8PathBuf {
        layout.resolve_relative(self.seeds_rel_path())
    }
}

/// Metadata about the ML model used to score and select edges.
///
/// Stored inside the [`Manifest`] when the pipeline runs in ML mode.
/// Allows detecting model mismatches between a persisted state and a freshly
/// loaded runtime by comparing the stored path and checksum against the
/// currently configured model.
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
    /// Create a new [`EdgeDeltaEntry`].
    ///
    /// Arguments
    /// ---------
    /// * `night_id` — Night that produced this delta segment.
    /// * `delta_relpath` — Path to the delta file, relative to the persistence
    ///   root.
    /// * `n_ops` — Optional count of edge operations stored in the delta file
    ///   (used for sanity checks; pass `None` if unknown).
    ///
    /// Return
    /// ------
    /// A new [`EdgeDeltaEntry`].
    pub fn new(night_id: NightId, delta_relpath: Utf8PathBuf, n_ops: Option<u64>) -> Self {
        Self {
            night_id,
            delta_relpath: delta_relpath.to_string(),
            n_ops,
        }
    }

    /// Resolve the absolute path to the delta file for this entry.
    ///
    /// Arguments
    /// ---------
    /// * `layout` — Persistence layout providing the root directory against
    ///   which the relative path is resolved.
    ///
    /// Return
    /// ------
    /// Absolute [`camino::Utf8PathBuf`] to the delta file.
    #[inline]
    pub fn delta_abs_path(&self, layout: &PersistenceLayout) -> Utf8PathBuf {
        layout.resolve_relative(Utf8Path::new(&self.delta_relpath))
    }

    /// Return the relative path to the delta file for this entry.
    ///
    /// Return
    /// ------
    /// A [`camino::Utf8Path`] slice into the stored path string.
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
    ///
    /// The schema version is set to [`EDGE_JOURNAL_SCHEMA_VERSION`]. No
    /// snapshot and no deltas are registered.
    ///
    /// Return
    /// ------
    /// A new, empty [`EdgeJournalManifest`].
    pub fn new() -> Self {
        Self {
            schema_version: EDGE_JOURNAL_SCHEMA_VERSION,
            snapshot_relpath: None,
            snapshot_night_id: None,
            deltas: Vec::new(),
        }
    }

    /// Return `true` if the journal contains no snapshot and no delta files.
    ///
    /// Return
    /// ------
    /// `true` when both `snapshot_relpath` is `None` and `deltas` is empty.
    pub fn is_empty(&self) -> bool {
        self.snapshot_relpath.is_none() && self.deltas.is_empty()
    }

    /// Resolve the absolute path to the snapshot file, if present.
    ///
    /// Arguments
    /// ---------
    /// * `layout` — Persistence layout providing the root directory against
    ///   which the relative path is resolved.
    ///
    /// Return
    /// ------
    /// `Some(Utf8PathBuf)` with the absolute path when a snapshot is
    /// registered; `None` if the journal starts from an empty baseline.
    #[inline]
    pub fn snapshot_abs_path(&self, layout: &PersistenceLayout) -> Option<Utf8PathBuf> {
        self.snapshot_relpath
            .as_ref()
            .map(|rel| layout.resolve_relative(Utf8Path::new(rel)))
    }

    /// Return the relative path to the snapshot file, if present.
    ///
    /// Return
    /// ------
    /// `Some(&Utf8Path)` if a snapshot has been set; `None` if the journal
    /// starts from an empty baseline.
    #[inline]
    pub fn snapshot_rel_path(&self) -> Option<&Utf8Path> {
        self.snapshot_relpath.as_ref().map(|rel| Utf8Path::new(rel))
    }

    /// Set or replace the snapshot metadata.
    ///
    /// Calling this method a second time overwrites the previously recorded
    /// snapshot path and checkpoint night. It does not delete the old snapshot
    /// file from disk.
    ///
    /// Arguments
    /// ---------
    /// * `relpath` — Path to the snapshot file, relative to the persistence
    ///   root.
    /// * `snapshot_night_id` — Night ID that the snapshot represents; deltas
    ///   with `night_id > snapshot_night_id` should be applied on top of it.
    ///
    /// Return
    /// ------
    /// Nothing (`()`).
    pub fn set_snapshot(&mut self, relpath: Utf8PathBuf, snapshot_night_id: NightId) {
        self.snapshot_relpath = Some(relpath.to_string());
        self.snapshot_night_id = Some(snapshot_night_id);
    }

    /// Insert or replace a delta entry for one night.
    ///
    /// If a delta with the same `night_id` already exists it is replaced;
    /// otherwise the entry is appended. The `deltas` list is kept sorted by
    /// `night_id` after every call.
    ///
    /// Arguments
    /// ---------
    /// * `entry` — Delta entry to insert or replace.
    ///
    /// Return
    /// ------
    /// Nothing (`()`).
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

    /// Drop all delta entries with `night_id <= night_id`.
    ///
    /// Typically called after a new snapshot has been written to remove the
    /// delta files that are now folded into the snapshot.
    ///
    /// Arguments
    /// ---------
    /// * `night_id` — Upper bound (inclusive) of the nights to discard.
    ///
    /// Return
    /// ------
    /// Nothing (`()`). Delta files are **not** deleted from disk by this call;
    /// removal of the physical files is the caller's responsibility.
    pub fn drop_deltas_leq(&mut self, night_id: NightId) {
        self.deltas.retain(|e| e.night_id > night_id);
    }
}

/// Top-level persistence manifest.
///
/// The manifest is the single entry point for locating all on-disk artifacts
/// belonging to a Fink-FAT state. It records:
///
/// - one [`NightManifestEntry`] per processed night (alerts + seeds),
/// - the [`EdgeJournalManifest`] tracking the edge snapshot and deltas,
/// - schema versions for each artifact type (used to detect
///   serialization-format mismatches at load time),
/// - optional metadata about the ML model used to produce the edges.
///
/// Notes
/// -----
/// - All paths are stored as strings relative to the persistence root,
///   keeping the state directory fully relocatable.
/// - The manifest is serialized as a **pretty-printed JSON file** for human
///   readability. [`Manifest::save`] and [`Manifest::load`] wrap the payload
///   in a [`crate::persistence::envelope::DiskEnvelope`] JSON envelope.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    /// Unix timestamp (seconds) when the manifest was last written.
    pub created_unix_s: i64,

    /// Schema version used when writing alert store files.
    pub alert_store_schema_version: u32,
    /// Schema version used when writing seed store files.
    pub seed_store_schema_version: u32,
    /// Schema version used when writing graph files.
    pub graph_schema_version: u32,
    /// Schema version used when writing the manifest state envelope.
    pub state_schema_version: u32,

    /// One entry per night available in the state.
    pub nights: Vec<NightManifestEntry>,

    /// Edge storage as a journal (snapshot + deltas).
    pub edge_journal: EdgeJournalManifest,

    /// Optional ML model metadata.
    pub model: Option<ModelManifestEntry>,
}

impl Manifest {
    /// Create a new, empty manifest with current schema versions.
    ///
    /// All schema version fields are initialised from the compile-time
    /// constants. `created_unix_s` is set to the current Unix timestamp.
    /// `nights`, `edge_journal`, and `model` start empty / `None`.
    ///
    /// Return
    /// ------
    /// A new [`Manifest`] ready to be populated.
    pub fn new() -> Self {
        let created_unix_s = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
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

    /// Return the maximum night ID present in the manifest.
    ///
    /// The `nights` list (alerts/seeds presence) is used as the canonical
    /// timeline. To derive the maximum from edge deltas instead, inspect
    /// `edge_journal.deltas` directly.
    ///
    /// Return
    /// ------
    /// `Some(NightId)` for the latest registered night; `None` if no nights
    /// have been added yet.
    pub fn max_night_id(&self) -> Option<NightId> {
        self.nights.iter().map(|e| e.night_id).max()
    }

    /// Derive the sliding edge-window from the engine configuration.
    ///
    /// Window definition
    /// -----------------
    /// Let $G$ = `engine_config.max_gap_nights()`. For the current maximum
    /// night $N\_\mathrm{max}$ the window is:
    ///
    /// $$\begin{align} N\_\mathrm{min} &= N\_\mathrm{max} - G \\ N\_\mathrm{max} &= N\_\mathrm{max} \end{align}$$
    ///
    /// This is the minimal range needed to reconstruct and solve the
    /// inter-night graph under the assumption that no edge spans more than
    /// $G$ nights.
    ///
    /// Arguments
    /// ---------
    /// * `engine_config` — Engine configuration from which `max_gap_nights`
    ///   is read.
    ///
    /// Return
    /// ------
    /// * `Ok(Some(PairingMode))` — A batch-range pairing covering
    ///   $[N\_\mathrm{min},\, N\_\mathrm{max}]$.
    /// * `Ok(None)` — No nights are registered in the manifest yet.
    /// * `Err(FinkFatError)` — If [`PairingMode::batch_range`] rejects the
    ///   computed range.
    pub fn compute_edge_window_from_config(
        &self,
        engine_config: &EngineConfig,
    ) -> Result<Option<PairingMode>, FinkFatError> {
        let max_night = match self.max_night_id() {
            Some(n) => n,
            None => return Ok(None),
        };

        let g = engine_config.max_gap_nights() as u32;
        let min_night = NightId(max_night.0.saturating_sub(g));

        PairingMode::batch_range(min_night, max_night).map(Some)
    }

    /// Insert or replace the entry for one night.
    ///
    /// If an entry with the same `night_id` already exists it is replaced;
    /// otherwise the entry is appended. The `nights` list is kept sorted by
    /// `night_id` after every call.
    ///
    /// Arguments
    /// ---------
    /// * `entry` — Night entry to insert or replace.
    ///
    /// Return
    /// ------
    /// Nothing (`()`).
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

    /// Serialize and write the manifest to disk as a pretty-printed JSON file.
    ///
    /// The manifest is wrapped in a JSON envelope (see
    /// [`DiskEnvelope::save_enveloped_json`](crate::persistence::envelope::DiskEnvelope::save_enveloped_json))
    /// that records the schema version and creation timestamp alongside the
    /// payload. The resulting file is human-readable and can be inspected with
    /// any text editor or standard JSON tool.
    ///
    /// Arguments
    /// ---------
    /// * `path` — Destination file path (should be absolute).
    ///
    /// Return
    /// ------
    /// * `Ok(())` on success.
    /// * `Err(PersistenceIoError)` if serialization or the write fails.
    pub fn save(&self, path: &Utf8Path) -> Result<(), PersistenceIoError> {
        let env = DiskEnvelope::new(self.clone(), STATE_SCHEMA_VERSION, self.created_unix_s, Compression::None);
        env.save_enveloped_json(path)
    }

    /// Deserialize and validate the manifest from a JSON file on disk.
    ///
    /// The file is expected to contain a JSON envelope produced by
    /// [`Manifest::save`] (see
    /// [`DiskEnvelope::load_enveloped_json`](crate::persistence::envelope::DiskEnvelope::load_enveloped_json)).
    /// The `magic` string and schema version in the envelope are checked
    /// against `STATE_SCHEMA_VERSION` before deserialization.
    ///
    /// After a successful load, `created_unix_s` is refreshed to the current
    /// Unix timestamp so that a subsequent [`Manifest::save`] records an
    /// up-to-date modification time.
    ///
    /// Arguments
    /// ---------
    /// * `path` — Source JSON file path.
    ///
    /// Return
    /// ------
    /// * `Ok(Manifest)` on success.
    /// * `Err(PersistenceIoError)` if the file cannot be read, parsed,
    ///   or if the magic string or schema version do not match.
    pub fn load(path: &Utf8Path) -> Result<Self, PersistenceIoError> {
        let mut manifest =
            DiskEnvelope::<Manifest>::load_enveloped_json(path, STATE_SCHEMA_VERSION)?;
        manifest.created_unix_s = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        Ok(manifest)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use camino::Utf8PathBuf;
    use tempfile::tempdir;

    use crate::{
        engine_config::EngineConfig,
        night_id::{NightId, PairingMode},
        persistence::{
            STATE_SCHEMA_VERSION,
            envelope::JSON_MAGIC,
            error::{EnvelopeError, PersistenceIoError},
            layout::PersistenceLayout,
        },
    };

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    fn layout_in(dir: &tempfile::TempDir) -> PersistenceLayout {
        let root = Utf8PathBuf::from_path_buf(dir.path().to_path_buf())
            .expect("temp dir must be valid UTF-8");
        PersistenceLayout::new(root)
    }

    fn nid(n: u32) -> NightId {
        NightId(n)
    }

    /// Build a minimal but populated `Manifest` useful as a fixture.
    fn sample_manifest() -> Manifest {
        let mut m = Manifest::new();

        let entry1 = NightManifestEntry::new(
            nid(100),
            Utf8PathBuf::from("alerts/nid=100.bin"),
            Utf8PathBuf::from("seeds/nid=100.bin"),
            Some(500),
            Some(42),
        );
        let entry2 = NightManifestEntry::new(
            nid(101),
            Utf8PathBuf::from("alerts/nid=101.bin"),
            Utf8PathBuf::from("seeds/nid=101.bin"),
            None,
            None,
        );
        m.upsert_night(entry1);
        m.upsert_night(entry2);

        let delta = EdgeDeltaEntry::new(
            nid(101),
            Utf8PathBuf::from("graph/delta-nid=101.bin"),
            Some(30),
        );
        m.edge_journal.upsert_delta(delta);
        m.edge_journal.set_snapshot(
            Utf8PathBuf::from("graph/snapshot.bin"),
            nid(100),
        );

        m.model = Some(ModelManifestEntry {
            model_path: "models/edge_model.onnx".to_string(),
            model_sha256: Some("deadbeef".to_string()),
        });

        m
    }

    // =========================================================================
    // NightManifestEntry
    // =========================================================================

    #[test]
    fn night_entry_new_stores_paths() {
        let entry = NightManifestEntry::new(
            nid(10),
            Utf8PathBuf::from("alerts/nid=10.bin"),
            Utf8PathBuf::from("seeds/nid=10.bin"),
            Some(7),
            Some(3),
        );

        assert_eq!(entry.night_id, nid(10));
        assert_eq!(entry.alerts_rel_path(), Utf8Path::new("alerts/nid=10.bin"));
        assert_eq!(entry.seeds_rel_path(), Utf8Path::new("seeds/nid=10.bin"));
        assert_eq!(entry.n_alerts, Some(7));
        assert_eq!(entry.n_seeds, Some(3));
    }

    #[test]
    fn night_entry_optional_counts_can_be_none() {
        let entry = NightManifestEntry::new(
            nid(5),
            Utf8PathBuf::from("alerts/nid=5.bin"),
            Utf8PathBuf::from("seeds/nid=5.bin"),
            None,
            None,
        );

        assert!(entry.n_alerts.is_none());
        assert!(entry.n_seeds.is_none());
    }

    #[test]
    fn night_entry_abs_path_resolves_against_layout() {
        let dir = tempdir().unwrap();
        let layout = layout_in(&dir);

        let entry = NightManifestEntry::new(
            nid(20),
            Utf8PathBuf::from("alerts/nid=20.bin"),
            Utf8PathBuf::from("seeds/nid=20.bin"),
            None,
            None,
        );

        let expected_alerts = layout.root().join("alerts/nid=20.bin");
        let expected_seeds = layout.root().join("seeds/nid=20.bin");

        assert_eq!(entry.alerts_abs_path(&layout), expected_alerts);
        assert_eq!(entry.seeds_abs_path(&layout), expected_seeds);
    }

    // =========================================================================
    // EdgeDeltaEntry
    // =========================================================================

    #[test]
    fn delta_entry_stores_relpath_and_ops() {
        let entry = EdgeDeltaEntry::new(
            nid(7),
            Utf8PathBuf::from("graph/delta-nid=7.bin"),
            Some(99),
        );

        assert_eq!(entry.night_id, nid(7));
        assert_eq!(
            entry.delta_rel_path(),
            Utf8Path::new("graph/delta-nid=7.bin")
        );
        assert_eq!(entry.n_ops, Some(99));
    }

    #[test]
    fn delta_entry_abs_path_resolves_against_layout() {
        let dir = tempdir().unwrap();
        let layout = layout_in(&dir);

        let entry = EdgeDeltaEntry::new(
            nid(99),
            Utf8PathBuf::from("graph/delta-nid=99.bin"),
            None,
        );

        let expected = layout.root().join("graph/delta-nid=99.bin");
        assert_eq!(entry.delta_abs_path(&layout), expected);
    }

    // =========================================================================
    // EdgeJournalManifest
    // =========================================================================

    #[test]
    fn journal_new_is_empty() {
        let j = EdgeJournalManifest::new();

        assert!(j.is_empty());
        assert!(j.snapshot_rel_path().is_none());
        assert!(j.snapshot_night_id.is_none());
        assert!(j.deltas.is_empty());
    }

    #[test]
    fn journal_is_empty_false_after_delta_inserted() {
        let mut j = EdgeJournalManifest::new();
        j.upsert_delta(EdgeDeltaEntry::new(
            nid(1),
            Utf8PathBuf::from("graph/delta-nid=1.bin"),
            None,
        ));

        assert!(!j.is_empty());
    }

    #[test]
    fn journal_is_empty_false_when_snapshot_set() {
        let mut j = EdgeJournalManifest::new();
        j.set_snapshot(Utf8PathBuf::from("graph/snapshot.bin"), nid(0));

        assert!(!j.is_empty());
    }

    #[test]
    fn journal_set_snapshot_stores_metadata() {
        let mut j = EdgeJournalManifest::new();
        j.set_snapshot(Utf8PathBuf::from("graph/snapshot.bin"), nid(42));

        assert_eq!(
            j.snapshot_rel_path(),
            Some(Utf8Path::new("graph/snapshot.bin"))
        );
        assert_eq!(j.snapshot_night_id, Some(nid(42)));
    }

    #[test]
    fn journal_set_snapshot_overwrites_previous() {
        let mut j = EdgeJournalManifest::new();
        j.set_snapshot(Utf8PathBuf::from("graph/snapshot_old.bin"), nid(10));
        j.set_snapshot(Utf8PathBuf::from("graph/snapshot_new.bin"), nid(20));

        assert_eq!(
            j.snapshot_rel_path(),
            Some(Utf8Path::new("graph/snapshot_new.bin"))
        );
        assert_eq!(j.snapshot_night_id, Some(nid(20)));
    }

    #[test]
    fn journal_snapshot_abs_path_resolves_against_layout() {
        let dir = tempdir().unwrap();
        let layout = layout_in(&dir);
        let mut j = EdgeJournalManifest::new();
        j.set_snapshot(Utf8PathBuf::from("graph/snapshot.bin"), nid(5));

        let expected = layout.root().join("graph/snapshot.bin");
        assert_eq!(j.snapshot_abs_path(&layout), Some(expected));
    }

    #[test]
    fn journal_snapshot_abs_path_none_when_not_set() {
        let dir = tempdir().unwrap();
        let layout = layout_in(&dir);
        let j = EdgeJournalManifest::new();

        assert!(j.snapshot_abs_path(&layout).is_none());
    }

    #[test]
    fn journal_upsert_delta_appends_and_keeps_sorted() {
        let mut j = EdgeJournalManifest::new();

        j.upsert_delta(EdgeDeltaEntry::new(nid(3), Utf8PathBuf::from("g/d3.bin"), None));
        j.upsert_delta(EdgeDeltaEntry::new(nid(1), Utf8PathBuf::from("g/d1.bin"), None));
        j.upsert_delta(EdgeDeltaEntry::new(nid(2), Utf8PathBuf::from("g/d2.bin"), None));

        let ids: Vec<NightId> = j.deltas.iter().map(|e| e.night_id).collect();
        assert_eq!(ids, vec![nid(1), nid(2), nid(3)]);
    }

    #[test]
    fn journal_upsert_delta_replaces_same_night() {
        let mut j = EdgeJournalManifest::new();

        j.upsert_delta(EdgeDeltaEntry::new(nid(5), Utf8PathBuf::from("old.bin"), Some(10)));
        j.upsert_delta(EdgeDeltaEntry::new(nid(5), Utf8PathBuf::from("new.bin"), Some(99)));

        assert_eq!(j.deltas.len(), 1);
        assert_eq!(j.deltas[0].delta_rel_path(), Utf8Path::new("new.bin"));
        assert_eq!(j.deltas[0].n_ops, Some(99));
    }

    #[test]
    fn journal_drop_deltas_leq_removes_entries_at_and_below() {
        let mut j = EdgeJournalManifest::new();
        for n in [1u32, 2, 3, 4, 5] {
            j.upsert_delta(EdgeDeltaEntry::new(
                nid(n),
                Utf8PathBuf::from(format!("g/d{n}.bin")),
                None,
            ));
        }

        j.drop_deltas_leq(nid(3));

        let remaining: Vec<u32> = j.deltas.iter().map(|e| e.night_id.0).collect();
        assert_eq!(remaining, vec![4, 5]);
    }

    #[test]
    fn journal_drop_deltas_leq_keeps_all_when_bound_below_min() {
        let mut j = EdgeJournalManifest::new();
        for n in [10u32, 20, 30] {
            j.upsert_delta(EdgeDeltaEntry::new(
                nid(n),
                Utf8PathBuf::from(format!("g/d{n}.bin")),
                None,
            ));
        }

        j.drop_deltas_leq(nid(5));

        assert_eq!(j.deltas.len(), 3);
    }

    #[test]
    fn journal_drop_deltas_leq_drops_all_when_bound_at_max() {
        let mut j = EdgeJournalManifest::new();
        for n in [1u32, 2, 3] {
            j.upsert_delta(EdgeDeltaEntry::new(
                nid(n),
                Utf8PathBuf::from(format!("g/d{n}.bin")),
                None,
            ));
        }

        j.drop_deltas_leq(nid(3));

        assert!(j.deltas.is_empty());
    }

    // =========================================================================
    // Manifest
    // =========================================================================

    #[test]
    fn manifest_new_has_correct_schema_versions() {
        let m = Manifest::new();

        assert_eq!(m.alert_store_schema_version, crate::persistence::ALERT_STORE_SCHEMA_VERSION);
        assert_eq!(m.seed_store_schema_version, crate::persistence::SEED_STORE_SCHEMA_VERSION);
        assert_eq!(m.graph_schema_version, crate::persistence::GRAPH_SCHEMA_VERSION);
        assert_eq!(m.state_schema_version, STATE_SCHEMA_VERSION);
    }

    #[test]
    fn manifest_new_has_empty_collections() {
        let m = Manifest::new();

        assert!(m.nights.is_empty());
        assert!(m.edge_journal.is_empty());
        assert!(m.model.is_none());
    }

    #[test]
    fn manifest_new_sets_created_unix_s() {
        let before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let m = Manifest::new();

        let after = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        assert!(m.created_unix_s >= before);
        assert!(m.created_unix_s <= after);
    }

    #[test]
    fn manifest_max_night_id_none_when_empty() {
        let m = Manifest::new();
        assert!(m.max_night_id().is_none());
    }

    #[test]
    fn manifest_max_night_id_returns_largest() {
        let mut m = Manifest::new();
        for n in [5u32, 1, 3, 9, 2] {
            m.upsert_night(NightManifestEntry::new(
                nid(n),
                Utf8PathBuf::from(format!("alerts/{n}.bin")),
                Utf8PathBuf::from(format!("seeds/{n}.bin")),
                None,
                None,
            ));
        }

        assert_eq!(m.max_night_id(), Some(nid(9)));
    }

    #[test]
    fn manifest_upsert_night_keeps_sorted_order() {
        let mut m = Manifest::new();
        for n in [30u32, 10, 20] {
            m.upsert_night(NightManifestEntry::new(
                nid(n),
                Utf8PathBuf::from(format!("alerts/{n}.bin")),
                Utf8PathBuf::from(format!("seeds/{n}.bin")),
                None,
                None,
            ));
        }

        let ids: Vec<u32> = m.nights.iter().map(|e| e.night_id.0).collect();
        assert_eq!(ids, vec![10, 20, 30]);
    }

    #[test]
    fn manifest_upsert_night_replaces_existing() {
        let mut m = Manifest::new();

        m.upsert_night(NightManifestEntry::new(
            nid(5),
            Utf8PathBuf::from("alerts/old.bin"),
            Utf8PathBuf::from("seeds/old.bin"),
            Some(10),
            Some(2),
        ));
        m.upsert_night(NightManifestEntry::new(
            nid(5),
            Utf8PathBuf::from("alerts/new.bin"),
            Utf8PathBuf::from("seeds/new.bin"),
            Some(20),
            Some(4),
        ));

        assert_eq!(m.nights.len(), 1);
        assert_eq!(
            m.nights[0].alerts_rel_path(),
            Utf8Path::new("alerts/new.bin")
        );
        assert_eq!(m.nights[0].n_alerts, Some(20));
    }

    // =========================================================================
    // compute_edge_window_from_config
    // =========================================================================

    #[test]
    fn compute_edge_window_none_when_no_nights() {
        let m = Manifest::new();
        let cfg = EngineConfig::default();

        let result = m.compute_edge_window_from_config(&cfg).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn compute_edge_window_returns_correct_range() {
        // Default EngineConfig has max_gap_nights = 3.
        // With nights [8, 9, 10] the window should be [10 - 3, 10] = [7, 10].
        let mut m = Manifest::new();
        for n in [8u32, 9, 10] {
            m.upsert_night(NightManifestEntry::new(
                nid(n),
                Utf8PathBuf::from(format!("alerts/{n}.bin")),
                Utf8PathBuf::from(format!("seeds/{n}.bin")),
                None,
                None,
            ));
        }

        let cfg = EngineConfig::default(); // max_gap_nights = 3
        let window = m.compute_edge_window_from_config(&cfg).unwrap().unwrap();

        assert!(window.is_batch());
        assert_eq!(
            window,
            PairingMode::BatchRange {
                start: nid(7),
                end: nid(10)
            }
        );
    }

    #[test]
    fn compute_edge_window_saturates_min_at_zero() {
        // max_gap_nights = 3, single night 1 -> min = 1.saturating_sub(3) = 0
        let mut m = Manifest::new();
        m.upsert_night(NightManifestEntry::new(
            nid(1),
            Utf8PathBuf::from("alerts/1.bin"),
            Utf8PathBuf::from("seeds/1.bin"),
            None,
            None,
        ));

        let cfg = EngineConfig::default(); // max_gap_nights = 3
        let window = m.compute_edge_window_from_config(&cfg).unwrap().unwrap();

        assert_eq!(
            window,
            PairingMode::BatchRange {
                start: nid(0),
                end: nid(1)
            }
        );
    }

    // =========================================================================
    // JSON I/O — save / load roundtrip
    // =========================================================================

    #[test]
    fn save_and_load_roundtrip_empty_manifest() {
        let dir = tempdir().unwrap();
        let path = Utf8PathBuf::from_path_buf(dir.path().join("manifest.json"))
            .expect("UTF-8 path");

        let m = Manifest::new();
        m.save(&path).unwrap();

        let loaded = Manifest::load(&path).unwrap();

        assert_eq!(loaded.nights.len(), 0);
        assert!(loaded.edge_journal.is_empty());
        assert!(loaded.model.is_none());
        assert_eq!(loaded.state_schema_version, m.state_schema_version);
        assert_eq!(
            loaded.alert_store_schema_version,
            m.alert_store_schema_version
        );
    }

    #[test]
    fn save_and_load_roundtrip_populated_manifest() {
        let dir = tempdir().unwrap();
        let path = Utf8PathBuf::from_path_buf(dir.path().join("manifest.json"))
            .expect("UTF-8 path");

        let m = sample_manifest();
        m.save(&path).unwrap();

        let loaded = Manifest::load(&path).unwrap();

        // nights
        assert_eq!(loaded.nights.len(), 2);
        assert_eq!(loaded.nights[0].night_id, nid(100));
        assert_eq!(loaded.nights[1].night_id, nid(101));
        assert_eq!(loaded.nights[0].n_alerts, Some(500));
        assert_eq!(loaded.nights[0].n_seeds, Some(42));
        assert_eq!(loaded.nights[1].n_alerts, None);

        // edge journal
        assert_eq!(loaded.edge_journal.deltas.len(), 1);
        assert_eq!(loaded.edge_journal.deltas[0].night_id, nid(101));
        assert_eq!(loaded.edge_journal.deltas[0].n_ops, Some(30));
        assert_eq!(
            loaded.edge_journal.snapshot_rel_path(),
            Some(Utf8Path::new("graph/snapshot.bin"))
        );
        assert_eq!(loaded.edge_journal.snapshot_night_id, Some(nid(100)));

        // model
        let model = loaded.model.unwrap();
        assert_eq!(model.model_path, "models/edge_model.onnx");
        assert_eq!(model.model_sha256.as_deref(), Some("deadbeef"));
    }

    #[test]
    fn save_produces_valid_json_file() {
        let dir = tempdir().unwrap();
        let path = Utf8PathBuf::from_path_buf(dir.path().join("manifest.json"))
            .expect("UTF-8 path");

        sample_manifest().save(&path).unwrap();

        let content = std::fs::read_to_string(path.as_std_path()).unwrap();
        // The file must parse as valid JSON and contain the magic string.
        let v: serde_json::Value = serde_json::from_str(&content)
            .expect("saved manifest must be valid JSON");
        assert_eq!(v["magic"].as_str().unwrap(), JSON_MAGIC);
        assert_eq!(
            v["schema_version"].as_u64().unwrap(),
            STATE_SCHEMA_VERSION as u64
        );
    }

    #[test]
    fn save_creates_parent_directories() {
        let dir = tempdir().unwrap();
        let path = Utf8PathBuf::from_path_buf(dir.path().join("a/b/c/manifest.json"))
            .expect("UTF-8 path");

        Manifest::new().save(&path).unwrap();

        assert!(path.exists());
    }

    #[test]
    fn save_is_atomic_no_tmp_file_remains() {
        let dir = tempdir().unwrap();
        let path = Utf8PathBuf::from_path_buf(dir.path().join("manifest.json"))
            .expect("UTF-8 path");
        let tmp =
            Utf8PathBuf::from_path_buf(dir.path().join("manifest.json.tmp")).expect("UTF-8 path");

        Manifest::new().save(&path).unwrap();

        assert!(path.exists(), "manifest file must exist after save");
        assert!(!tmp.exists(), "temporary file must have been renamed away");
    }

    #[test]
    fn save_overwrites_existing_file() {
        let dir = tempdir().unwrap();
        let path = Utf8PathBuf::from_path_buf(dir.path().join("manifest.json"))
            .expect("UTF-8 path");

        let mut m1 = Manifest::new();
        m1.upsert_night(NightManifestEntry::new(
            nid(1),
            Utf8PathBuf::from("a/1.bin"),
            Utf8PathBuf::from("s/1.bin"),
            None,
            None,
        ));
        m1.save(&path).unwrap();

        let m2 = Manifest::new(); // empty – no nights
        m2.save(&path).unwrap();

        let loaded = Manifest::load(&path).unwrap();
        assert!(loaded.nights.is_empty(), "overwritten file should be empty");
    }

    #[test]
    fn load_refreshes_created_unix_s() {
        let dir = tempdir().unwrap();
        let path = Utf8PathBuf::from_path_buf(dir.path().join("manifest.json"))
            .expect("UTF-8 path");

        // Force a very old created_unix_s on disk via the JSON directly.
        let mut m = Manifest::new();
        m.created_unix_s = 0; // epoch
        m.save(&path).unwrap();

        let loaded = Manifest::load(&path).unwrap();

        // loaded.created_unix_s must be close to now, not 0.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        // Allow a 5-second window for slow CI.
        assert!(
            loaded.created_unix_s > 0,
            "created_unix_s should be refreshed, not 0"
        );
        assert!(
            (loaded.created_unix_s - now).abs() <= 5,
            "created_unix_s should be close to now"
        );
    }

    #[test]
    fn load_fails_on_wrong_schema_version() {
        let dir = tempdir().unwrap();
        let path = Utf8PathBuf::from_path_buf(dir.path().join("manifest.json"))
            .expect("UTF-8 path");

        // Write a JSON envelope with an unexpected schema_version.
        let wrong_version = STATE_SCHEMA_VERSION + 1;
        let json = serde_json::json!({
            "magic": JSON_MAGIC,
            "schema_version": wrong_version,
            "created_unix_s": 0_i64,
            "payload": Manifest::new()
        });
        std::fs::write(path.as_std_path(), serde_json::to_string_pretty(&json).unwrap()).unwrap();

        let err = Manifest::load(&path).unwrap_err();
        assert!(
            matches!(
                err,
                PersistenceIoError::Envelope(EnvelopeError::UnsupportedSchemaVersion { .. })
            ),
            "expected UnsupportedSchemaVersion, got {err:?}"
        );
    }

    #[test]
    fn load_fails_on_wrong_magic() {
        let dir = tempdir().unwrap();
        let path = Utf8PathBuf::from_path_buf(dir.path().join("manifest.json"))
            .expect("UTF-8 path");

        let json = serde_json::json!({
            "magic": "WRONGMAGIC",
            "schema_version": STATE_SCHEMA_VERSION,
            "created_unix_s": 0_i64,
            "payload": Manifest::new()
        });
        std::fs::write(path.as_std_path(), serde_json::to_string_pretty(&json).unwrap()).unwrap();

        let err = Manifest::load(&path).unwrap_err();
        assert!(
            matches!(
                err,
                PersistenceIoError::Envelope(EnvelopeError::InvalidMagic)
            ),
            "expected InvalidMagic, got {err:?}"
        );
    }

    #[test]
    fn load_fails_on_corrupt_bytes() {
        let dir = tempdir().unwrap();
        let path = Utf8PathBuf::from_path_buf(dir.path().join("manifest.json"))
            .expect("UTF-8 path");

        std::fs::write(path.as_std_path(), b"this is not json at all").unwrap();

        let err = Manifest::load(&path).unwrap_err();
        assert!(
            matches!(err, PersistenceIoError::Json(_)),
            "expected Json error, got {err:?}"
        );
    }

    #[test]
    fn load_fails_on_missing_file() {
        let dir = tempdir().unwrap();
        let path = Utf8PathBuf::from_path_buf(dir.path().join("does_not_exist.json"))
            .expect("UTF-8 path");

        let err = Manifest::load(&path).unwrap_err();
        assert!(
            matches!(err, PersistenceIoError::Io(_)),
            "expected Io error, got {err:?}"
        );
    }
}
