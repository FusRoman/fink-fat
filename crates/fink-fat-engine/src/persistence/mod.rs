pub mod compression;
pub mod edge_journal;
pub mod envelope;
pub mod error;
pub mod layout;
pub mod manifest;
pub mod runtime_state;

use std::fs;

use camino::Utf8PathBuf;
use outfit::FullOrbitResult;

use crate::{
    Alert,
    alerts::{AlertSlice, store::AlertStore},
    engine_config::EngineConfig,
    error::{EngineError, FinkFatError},
    graph::{AlertLinkageDAG, edge::Edge},
    night_id::{NightId, PairingMode},
    persistence::{
        compression::Compression,
        edge_journal::{EdgeJournalStore, edge_op::EdgeOp},
        envelope::DiskEnvelope,
        error::{PersistenceError, PersistenceIoError},
        layout::PersistenceLayout,
        manifest::{Manifest, NightManifestEntry},
        runtime_state::RuntimeState,
    },
    pipeline::hooks::StageProgress,
    seeding::{SeedNode, SeedNodeSlice, store::SeedStore},
    solver::HypothesisSet,
};

/// Alert store schema version.
pub const ALERT_STORE_SCHEMA_VERSION: u32 = 1;

/// Seed store schema version.
pub const SEED_STORE_SCHEMA_VERSION: u32 = 1;

/// Edge journal schema version (snapshot/delta payloads).
pub const EDGE_JOURNAL_SCHEMA_VERSION: u32 = 1;

/// Inter-night graph schema version.
pub const GRAPH_SCHEMA_VERSION: u32 = 1;

/// Optional: top-level state/manifest schema version (if you persist one).
pub const STATE_SCHEMA_VERSION: u32 = 1;

/// High-level persistence orchestrator.
#[derive(Clone, Debug)]
pub struct PersistenceManager {
    layout: PersistenceLayout,
    edge_journal: EdgeJournalStore,
}

impl PersistenceManager {
    /// Access the persistence layout (directory structure / path helpers).
    pub fn layout(&self) -> &PersistenceLayout {
        &self.layout
    }

    /// Open an existing persistence root or create a new one if missing.
    ///
    /// Parameters
    /// ----------
    /// storage_root : impl Into<Utf8PathBuf>
    ///     Root directory for persisted state (e.g. `engine_config.storage_path()`).
    ///
    /// Returns
    /// -------
    /// Result<Self, PersistenceIoError>
    ///     Ready-to-use persistence manager.
    pub fn open_or_create(storage_root: impl Into<Utf8PathBuf>) -> Result<Self, PersistenceError> {
        let layout = PersistenceLayout::new(storage_root);
        let edge_journal = EdgeJournalStore::new(layout.clone());

        Ok(Self {
            layout,
            edge_journal,
        })
    }

    /// Load the manifest if present, otherwise create a fresh one.
    pub fn load_or_init_manifest(&self) -> Result<Manifest, PersistenceError> {
        let mpath = self.layout.manifest_path();
        match Manifest::load(&mpath) {
            Ok(m) => Ok(m),
            // Manifest not found yet → start fresh. Use `is_not_found()` rather
            // than matching on `Io(e)` directly because the error may now be
            // wrapped inside a `WithPath` annotation.
            Err(e) if e.is_not_found() => Ok(Manifest::new()),
            Err(e) => Err(PersistenceError::Io(e)),
        }
    }

    /// Save the manifest to disk.
    pub fn save_manifest(&self, manifest: &Manifest) -> Result<(), PersistenceError> {
        Ok(manifest.save(&self.layout.manifest_path())?)
    }

    /// Compute the edge/window policy from config and current manifest.
    ///
    /// Returns `None` if the manifest is empty.
    pub fn compute_window(
        &self,
        manifest: &Manifest,
        cfg: &EngineConfig,
    ) -> Result<Option<PairingMode>, FinkFatError> {
        manifest.compute_edge_window_from_config(cfg)
    }

    // -------------------------------------------------------------------------
    // Low-level per-night I/O (alerts/seeds)
    // -------------------------------------------------------------------------

    /// Save alerts for one night and update manifest night entry.
    pub fn save_night_manifest(
        &self,
        manifest: &mut Manifest,
        night_id: NightId,
        created_unix_s: i64,
        alerts: &[Alert],
        seeds: &[SeedNode],
        compression: Compression,
    ) -> Result<(), PersistenceIoError> {
        let abs_alert_path =
            alerts.save_alerts_night(&self.layout, manifest, night_id, compression)?;
        let abs_seed_path =
            seeds.save_seeds_night(&self.layout, manifest, night_id, compression)?;

        let entry = NightManifestEntry::new(
            night_id,
            self.layout
                .to_relative(&abs_alert_path)
                .unwrap_or_else(|| abs_alert_path.clone()),
            self.layout
                .to_relative(&abs_seed_path)
                .unwrap_or_else(|| abs_seed_path.clone()),
            Some(alerts.len() as u64),
            Some(seeds.len() as u64),
        );
        manifest.upsert_night(entry);
        manifest.created_unix_s = created_unix_s;

        Ok(())
    }

    /// Load alerts payload for one night.
    pub fn load_alerts_for_night(
        &self,
        relpath: &Utf8PathBuf,
    ) -> Result<Vec<Alert>, PersistenceIoError> {
        let path = self.layout.resolve_relative(relpath);
        DiskEnvelope::<Vec<Alert>>::load_enveloped(&path, ALERT_STORE_SCHEMA_VERSION)
            .map_err(|e| e.with_path(&path))
    }

    /// Load seeds payload for one night.
    pub fn load_seeds_for_night(
        &self,
        relpath: &Utf8PathBuf,
    ) -> Result<Vec<SeedNode>, PersistenceIoError> {
        let path = self.layout.resolve_relative(relpath);
        DiskEnvelope::<Vec<SeedNode>>::load_enveloped(&path, SEED_STORE_SCHEMA_VERSION)
            .map_err(|e| e.with_path(&path))
    }

    // -------------------------------------------------------------------------
    // High-level: load runtime state (alerts + seeds + edges)
    // -------------------------------------------------------------------------

    /// Load the runtime state for the current persisted window.
    ///
    /// This:
    /// - loads/initializes the manifest,
    /// - computes the sliding window from `max_gap_nights`,
    /// - loads alerts and seeds for nights in the window,
    /// - converts `SeedNodeOwned` -> `SeedNode<'alert>` using the `AlertStore`,
    /// - loads edges via snapshot + delta journal and converts them to borrowed
    ///   edges using the `SeedStore`.
    pub fn load_runtime_state(
        &self,
        cfg: &EngineConfig,
        stage_sink: &dyn StageProgress,
    ) -> Result<RuntimeState, EngineError> {
        let manifest = self.load_or_init_manifest()?;
        let window = self.compute_window(&manifest, cfg)?;

        stage_sink.inc(1);

        // Select nights to load
        let nights_to_load: Vec<NightManifestEntry> = match window {
            None => Vec::new(),
            Some(w) => manifest
                .nights
                .iter()
                .filter(|&e| e.night_id >= w.start() && e.night_id <= w.end())
                .cloned()
                .collect(),
        };

        stage_sink.inc(1);

        // 1) Load alerts into runtime AlertStore
        let mut alert_store = AlertStore::new();
        for entry in &nights_to_load {
            let payload = self.load_alerts_for_night(&entry.alerts_rel_path().to_path_buf())?;
            alert_store.insert(entry.night_id, payload);
        }

        stage_sink.inc(1);

        // 2) Load seeds owned then convert to borrowed into runtime SeedStore
        let mut seed_store: SeedStore = SeedStore::new();

        nights_to_load.iter().try_for_each(|entry| {
            self.load_seeds_for_night(&entry.seeds_rel_path().to_path_buf())
                .map(|payload| {
                    seed_store.insert_vec_seed(entry.night_id, payload);
                })
        })?;

        stage_sink.inc(1);

        // 3) Load edges owned via edge journal (snapshot + deltas), then convert to graph.
        let edges = self.edge_journal.load_edges(&manifest, window)?;
        stage_sink.inc(1);
        let graph = AlertLinkageDAG::from_edges(edges);
        stage_sink.inc(1);

        Ok(RuntimeState {
            manifest,
            window,
            alert_store,
            seed_store,
            graph,
            track_hypotheses: HypothesisSet::new(),
            orbit_results: FullOrbitResult::default(),
        })
    }

    // -------------------------------------------------------------------------
    // High-level: commit + maintenance
    // -------------------------------------------------------------------------

    /// Commit the current night (alerts + seeds + edge ops delta) to disk.
    ///
    /// This updates the manifest in-memory, but does not automatically compact.
    pub fn commit_night(
        &self,
        mut manifest: Manifest,
        cfg: &EngineConfig,
        night_id: NightId,
        created_unix_s: i64,
        edge_ops: Vec<EdgeOp>,
    ) -> Result<Manifest, EngineError> {
        // 1) write edge delta
        self.edge_journal.write_delta_for_night(
            &mut manifest,
            night_id,
            created_unix_s,
            edge_ops,
            cfg.binary_compression,
        )?;

        // 2) persist manifest
        self.save_manifest(&manifest)?;

        // 3) optional cleanup of old nights outside window (to avoid disk growth)
        let window = self.compute_window(&manifest, cfg)?;
        if let Some(w) = window {
            self.cleanup_old_nights(&manifest, w)?;
        }

        Ok(manifest)
    }

    /// Compact edges to a snapshot and cleanup unreferenced deltas on disk.
    ///
    /// The caller provides the current in-memory edge set so that no disk reload
    /// is needed. This is the primary optimisation over the old design, which
    /// re-read every delta file even though the full edge set was already live
    /// in `RuntimeState.graph`.
    ///
    /// This also handles `cleanup_old_nights` so that callers that skip
    /// `commit_night` (because compaction is preferred) still benefit from
    /// the sliding-window file cleanup.
    ///
    /// Recommended strategy: compact every `k` nights or when deltas count grows.
    pub fn compact_edges_and_cleanup(
        &self,
        manifest: &mut Manifest,
        cfg: &EngineConfig,
        checkpoint_night_id: NightId,
        created_unix_s: i64,
        edges: Vec<Edge>,
    ) -> Result<(), EngineError> {
        let window = self.compute_window(manifest, cfg)?;
        self.edge_journal.compact_to_snapshot(
            manifest,
            checkpoint_night_id,
            created_unix_s,
            edges,
            window,
            cfg.binary_compression,
        )?;

        // Mirror the cleanup that `commit_night` would have done.
        if let Some(w) = window {
            self.cleanup_old_nights(manifest, w)?;
        }

        self.save_manifest(manifest)?;
        Ok(())
    }

    /// Delete alerts/seeds files for nights outside the given window.
    ///
    /// Notes
    /// -----
    /// This keeps disk usage bounded when using a sliding window.
    pub fn cleanup_old_nights(
        &self,
        manifest: &Manifest,
        window: PairingMode,
    ) -> Result<u64, PersistenceIoError> {
        let mut deleted = 0u64;

        for e in &manifest.nights {
            if e.night_id >= window.start() && e.night_id <= window.end() {
                continue;
            }

            // Delete alerts file (best-effort)
            let ap = self.layout.resolve_relative(e.alerts_rel_path());
            match fs::remove_file(ap.as_std_path()) {
                Ok(()) => deleted += 1,
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                Err(err) => return Err(PersistenceIoError::Io(err)),
            }

            // Delete seeds file (best-effort)
            let sp = self.layout.resolve_relative(e.seeds_rel_path());
            match fs::remove_file(sp.as_std_path()) {
                Ok(()) => deleted += 1,
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                Err(err) => return Err(PersistenceIoError::Io(err)),
            }
        }

        Ok(deleted)
    }
}
