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
    /// storage_root : `impl Into<Utf8PathBuf>`
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
    /// - loads alerts, seeds, and edges **concurrently** using a multi-threaded
    ///   tokio runtime: one blocking task per night (alerts + seeds together) and
    ///   one blocking task for the edge journal (snapshot + deltas).
    /// - populates the runtime stores sequentially once all I/O tasks complete.
    ///
    /// Concurrency strategy
    /// --------------------
    /// All three I/O phases (alerts, seeds, edges) are independent on disk.
    /// `tokio::task::spawn_blocking` dispatches each blocking file read to the
    /// thread pool so the OS can issue parallel read-ahead for multiple files.
    /// The main thread joins all results via a `JoinSet` before building the
    /// in-memory `AlertLinkageDAG`.
    pub fn load_runtime_state(
        &self,
        cfg: &EngineConfig,
        stage_sink: &dyn StageProgress,
    ) -> Result<RuntimeState, EngineError> {
        // ---------------------------------------------------------------
        // 1. Load manifest (fast, sequential — JSON, single file).
        // ---------------------------------------------------------------
        let manifest = self.load_or_init_manifest()?;
        stage_sink.inc(1);

        // ---------------------------------------------------------------
        // 2. Compute the sliding window from the manifest + engine config.
        // ---------------------------------------------------------------
        let window = self.compute_window(&manifest, cfg)?;

        // Select nights that fall inside the window.
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

        // ---------------------------------------------------------------
        // 3–5. Parallel I/O: load all nights (alerts + seeds) and the
        //      edge journal concurrently via tokio blocking tasks.
        //
        //  • One task per night reads both the alert file and the seed
        //    file for that night (two reads, kept together for locality).
        //  • One independent task loads the full edge journal (snapshot +
        //    all delta files).
        //  • All tasks run on tokio's blocking thread pool, allowing the
        //    OS to overlap disk reads across different files.
        // ---------------------------------------------------------------
        let rt = tokio::runtime::Builder::new_multi_thread()
            .build()
            .expect("failed to build tokio runtime for parallel persistence I/O");

        // `self` is `Clone` — each blocking task gets its own handle so
        // there is no shared mutable state and no need for Arc/Mutex.
        let pm = self.clone();
        let nights_clone = nights_to_load.clone();
        let manifest_for_edges = manifest.clone();

        let (night_data, edges) = rt.block_on(async move {
            use tokio::task::JoinSet;

            type NightLoadResult = Result<(NightId, Vec<Alert>, Vec<SeedNode>), PersistenceIoError>;

            // --- Spawn one task per night (alerts + seeds together) ----------
            let mut night_handles: JoinSet<NightLoadResult> = JoinSet::new();

            for entry in nights_clone {
                let pm = pm.clone();
                night_handles.spawn_blocking(move || {
                    let alerts =
                        pm.load_alerts_for_night(&entry.alerts_rel_path().to_path_buf())?;
                    let seeds = pm.load_seeds_for_night(&entry.seeds_rel_path().to_path_buf())?;
                    Ok((entry.night_id, alerts, seeds))
                });
            }

            // --- Spawn edge-journal loading concurrently --------------------
            let pm_edges = pm.clone();
            let edge_task = tokio::task::spawn_blocking(move || {
                pm_edges
                    .edge_journal
                    .load_edges(&manifest_for_edges, window)
            });

            // --- Collect per-night results -----------------------------------
            let mut night_data: Vec<(NightId, Vec<Alert>, Vec<SeedNode>)> = Vec::new();
            while let Some(res) = night_handles.join_next().await {
                let item = res
                    .map_err(|e| {
                        PersistenceIoError::Io(std::io::Error::other(format!(
                            "night I/O task panicked: {e}",
                        )))
                    })?
                    .map_err(EngineError::from)?;
                night_data.push(item);
            }

            // --- Collect edge result ----------------------------------------
            let edges = edge_task
                .await
                .map_err(|e| {
                    PersistenceIoError::Io(std::io::Error::other(format!(
                        "edge I/O task panicked: {e}",
                    )))
                })?
                .map_err(EngineError::from)?;

            Ok::<_, EngineError>((night_data, edges))
        })?;

        // All I/O is done — report progress as a batch.
        stage_sink.inc(3); // alerts + seeds + edges

        // ---------------------------------------------------------------
        // 6. Populate stores sequentially (CPU, fast).
        // ---------------------------------------------------------------
        let mut alert_store = AlertStore::new();
        let mut seed_store: SeedStore = SeedStore::new();

        for (night_id, alerts, seeds) in night_data {
            alert_store.insert(night_id, alerts);
            seed_store.insert_vec_seed(night_id, seeds);
        }

        // ---------------------------------------------------------------
        // 7. Build inter-night graph from the loaded edge set (CPU).
        // ---------------------------------------------------------------
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
        // Use checkpoint_night_id as the window anchor so that edges built
        // *this* night (to.night_id == checkpoint_night_id) are included in
        // the snapshot.  The manifest does not yet contain the current night
        // at this point, so compute_window(manifest, cfg) would produce a
        // window ending at the *previous* max night and silently drop all
        // edges to the current night.
        let g = cfg.max_gap_nights() as u32;
        let start = NightId(checkpoint_night_id.0.saturating_sub(g));
        let window = PairingMode::batch_range(start, checkpoint_night_id).map(Some)?;
        self.edge_journal.compact_to_snapshot(
            manifest,
            checkpoint_night_id,
            created_unix_s,
            edges.as_slice(),
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

    // -------------------------------------------------------------------------
    // Edge journal helpers without intermediate manifest save
    //
    // These are the preferred call sites when the caller manages the final
    // manifest write itself (e.g. `save_data` Stage Phase E), avoiding one
    // redundant `save_manifest` round-trip.
    // -------------------------------------------------------------------------

    /// Write an incremental edge delta for `night_id` and apply the sliding-
    /// window cleanup, but do **not** save the manifest to disk.
    ///
    /// This is equivalent to [`PersistenceManager::commit_night`] minus the
    /// intermediate `save_manifest` call.  Callers that write the final
    /// manifest themselves (e.g. after merging concurrent stage results) should
    /// prefer this method to avoid the redundant write.
    pub fn write_edge_delta(
        &self,
        manifest: &mut Manifest,
        cfg: &EngineConfig,
        night_id: NightId,
        created_unix_s: i64,
        edge_ops: Vec<EdgeOp>,
    ) -> Result<(), EngineError> {
        self.edge_journal.write_delta_for_night(
            manifest,
            night_id,
            created_unix_s,
            edge_ops,
            cfg.binary_compression,
        )?;

        let window = self.compute_window(manifest, cfg)?;
        if let Some(w) = window {
            self.cleanup_old_nights(manifest, w)?;
        }

        Ok(())
    }

    /// Compact the full in-memory edge set into a new snapshot, apply the
    /// sliding-window cleanup, but do **not** save the manifest to disk.
    ///
    /// This is equivalent to [`PersistenceManager::compact_edges_and_cleanup`]
    /// minus the intermediate `save_manifest` call.  Callers that write the
    /// final manifest themselves should prefer this method.
    ///
    /// `edges` is borrowed (not moved) to avoid the ~4 GB clone that the
    /// equivalent `Vec<Edge>` ownership transfer would require.  The caller
    /// typically passes `&RuntimeState.graph.edges` directly.
    pub fn compact_edges(
        &self,
        manifest: &mut Manifest,
        cfg: &EngineConfig,
        checkpoint_night_id: NightId,
        created_unix_s: i64,
        edges: &[Edge],
    ) -> Result<(), EngineError> {
        // Use checkpoint_night_id as the window anchor so that edges built
        // *this* night (to.night_id == checkpoint_night_id) are included in
        // the snapshot.  The manifest does not yet contain the current night
        // at this point, so compute_window(manifest, cfg) would produce a
        // window ending at the *previous* max night and silently drop all
        // edges to the current night.
        let g = cfg.max_gap_nights() as u32;
        let start = NightId(checkpoint_night_id.0.saturating_sub(g));
        let window = PairingMode::batch_range(start, checkpoint_night_id).map(Some)?;
        self.edge_journal.compact_to_snapshot(
            manifest,
            checkpoint_night_id,
            created_unix_s,
            edges,
            window,
            cfg.binary_compression,
        )?;

        if let Some(w) = window {
            self.cleanup_old_nights(manifest, w)?;
        }

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
