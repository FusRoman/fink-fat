//! Edge persistence using a snapshot + per-night delta journal.
//!
//! Overview
//! --------
//! Edges are stored on disk using two layers:
//!
//! 1) A **snapshot**: a full, compact representation of the current edge set.
//! 2) A sequence of **deltas** (one file per night): incremental edge operations
//!    produced since the snapshot.
//!
//! To reconstruct the current state:
//! - load snapshot (if any),
//! - apply deltas in chronological order.
//!
//! This design allows:
//! - fast daily writes (append new delta file),
//! - bounded load times through periodic compaction,
//! - optional sliding-window policies (drop older nights during compaction).
//!
//! File layout (recommended)
//! -------------------------
//! - `edges/snapshot.bin`
//! - `edges/delta-nid=<NightId>.bin`
//!
//! The manifest tracks which files exist and in which order to apply them.

pub mod delta_chunk;
pub mod edge_op;

use std::fs;

use ahash::AHashMap;
use serde::{Deserialize, Serialize};

use crate::graph::edge::{Edge, EdgeKey};
use crate::night_id::{NightId, PairingMode};
use crate::persistence::EDGE_JOURNAL_SCHEMA_VERSION;
use crate::persistence::compression::Compression;
use crate::persistence::edge_journal::delta_chunk::EdgeDeltaChunk;
use crate::persistence::edge_journal::edge_op::EdgeOp;
use crate::persistence::envelope::DiskEnvelope;
use crate::persistence::error::PersistenceIoError;
use crate::persistence::layout::PersistenceLayout;
use crate::persistence::manifest::{EdgeDeltaEntry, Manifest};

/// Snapshot payload: full edge set at a checkpoint.
///
/// This is typically persisted as `edges/snapshot.bin`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EdgeSnapshot {
    /// Night ID representing the checkpoint (typically the last compacted night).
    pub checkpoint_night_id: NightId,
    /// Unix timestamp (seconds) for traceability.
    pub created_unix_s: i64,
    /// Full edge set at the checkpoint.
    pub edges: Vec<Edge>,
}

/// High-level helper to manage edge snapshot + delta journal.
///
/// This object is intentionally thin: it relies on [`PersistenceLayout`] for
/// file paths and on [`Manifest`] for indexing what exists.
#[derive(Clone, Debug)]
pub struct EdgeJournalStore {
    pub layout: PersistenceLayout,
}

impl EdgeJournalStore {
    /// Create a new journal store rooted at `layout`.
    pub fn new(layout: PersistenceLayout) -> Self {
        Self { layout }
    }

    /// Write the delta file for `night_id` and update the manifest.
    ///
    /// Parameters
    /// ----------
    /// manifest : &mut Manifest
    ///     Manifest to update (delta entry will be added/replaced).
    /// night_id : NightId
    ///     Night producing the delta.
    /// created_unix_s : i64
    ///     Unix timestamp stored in the delta envelope.
    /// ops : Vec<EdgeOp>
    ///     Edge operations to persist.
    ///
    /// Returns
    /// -------
    /// Result<(), PersistenceIoError>
    ///     Ok on success.
    ///
    /// Notes
    /// -----
    /// - This uses the "one file per night" approach (robust, atomic).
    /// - If a delta file for this night already exists, it is replaced.
    pub fn write_delta_for_night(
        &self,
        manifest: &mut Manifest,
        night_id: NightId,
        created_unix_s: i64,
        ops: Vec<EdgeOp>,
        compression: Compression,
    ) -> Result<(), PersistenceIoError> {
        let path = self.layout.graph_delta_night_path(night_id);
        let Some(rel) = self.layout.to_relative(&path) else {
            // This should not happen if layout is well-formed, but handle it gracefully.
            return Err(PersistenceIoError::Other(format!(
                "Delta path {:?} is outside of layout root {:?}",
                path,
                self.layout.root()
            )));
        };

        let len_ops = ops.len() as u64;
        EdgeDeltaChunk::write(&path, night_id, created_unix_s, ops, compression)?;

        // Update manifest
        manifest
            .edge_journal
            .upsert_delta(EdgeDeltaEntry::new(night_id, rel, Some(len_ops)));

        // Keep manifest "updated time" coherent (caller may override)
        manifest.created_unix_s = created_unix_s;
        Ok(())
    }

    /// Load the current edge set by replaying snapshot + deltas.
    ///
    /// Parameters
    /// ----------
    /// manifest : &Manifest
    ///     Manifest describing snapshot and delta files.
    /// window : Option<NightWindow>
    ///     Optional sliding window filter applied while replaying deltas.
    ///     If provided, only deltas within the window are applied, and edges
    ///     outside the window are dropped from the final output.
    ///
    /// Returns
    /// -------
    /// Result<Vec<EdgeOwned>, PersistenceIoError>
    ///     The reconstructed edge set.
    pub fn load_edges(
        &self,
        manifest: &Manifest,
        window: Option<PairingMode>,
    ) -> Result<Vec<Edge>, PersistenceIoError> {
        let mut map: AHashMap<EdgeKey, Edge> = AHashMap::new();

        // 1) Load snapshot if present.
        if let Some(rel) = manifest.edge_journal.snapshot_rel_path() {
            let snap_path = self.layout.resolve_relative(rel);
            let snapshot: EdgeSnapshot = DiskEnvelope::<EdgeSnapshot>::load_enveloped(
                &snap_path,
                EDGE_JOURNAL_SCHEMA_VERSION,
            )?;

            for e in snapshot.edges {
                map.insert(e.key(), e);
            }
        }

        // 2) Apply deltas in chronological order.
        //    (manifest.edge_journal.deltas is already sorted by night_id)
        for d in &manifest.edge_journal.deltas {
            if let Some(w) = window {
                if !w.contains(d.night_id) {
                    continue;
                }
            }

            let path = self.layout.resolve_relative(d.delta_rel_path());
            let delta: EdgeDeltaChunk =
                DiskEnvelope::<EdgeDeltaChunk>::load_enveloped(&path, EDGE_JOURNAL_SCHEMA_VERSION)?;

            self.apply_ops(&mut map, delta.ops)?;
        }

        // 3) Optionally enforce window on the final edge set.
        if let Some(w) = window {
            map.retain(|k, _| self.edge_key_in_window(*k, w));
        }

        Ok(map.into_values().collect())
    }

    /// Compact snapshot + deltas into a new snapshot at `checkpoint_night_id`.
    ///
    /// Parameters
    /// ----------
    /// manifest : &mut Manifest
    ///     Manifest to update (snapshot metadata + delta pruning).
    /// checkpoint_night_id : NightId
    ///     Night to record as the snapshot checkpoint.
    /// created_unix_s : i64
    ///     Unix timestamp stored in the snapshot envelope.
    /// window : Option<NightWindow>
    ///     Optional sliding window to apply during compaction.
    ///
    /// Returns
    /// -------
    /// Result<(), PersistenceIoError>
    ///     Ok on success.
    ///
    /// Behavior
    /// --------
    /// - Reconstruct current edges using [`load_edges`].
    /// - Write a new `EdgeSnapshot` to `edges/snapshot.bin`.
    /// - Update manifest snapshot metadata.
    /// - Drop all deltas with `night_id <= checkpoint_night_id` from the manifest.
    ///
    /// Notes
    /// -----
    /// - This does not delete files from disk (yet). It only updates the manifest.
    ///   You can add a cleanup step once you are comfortable with the workflow.
    pub fn compact_to_snapshot(
        &self,
        manifest: &mut Manifest,
        checkpoint_night_id: NightId,
        created_unix_s: i64,
        window: Option<PairingMode>,
        compression: Compression,
    ) -> Result<(), PersistenceIoError> {
        // Rebuild current edges (snapshot + deltas)
        let edges = self.load_edges(manifest, window)?;

        let snapshot = EdgeSnapshot {
            checkpoint_night_id,
            created_unix_s,
            edges,
        };

        let path = self.layout.graph_snapshot_path();
        let Some(rel) = self.layout.to_relative(&path) else {
            // This should not happen if layout is well-formed, but handle it gracefully.
            return Err(PersistenceIoError::Other(format!(
                "Snapshot path {:?} is outside of layout root {:?}",
                path,
                self.layout.root()
            )));
        };

        let env = DiskEnvelope::new(snapshot, EDGE_JOURNAL_SCHEMA_VERSION, created_unix_s, compression);
        env.save_enveloped(&path)?;

        // Update manifest snapshot metadata and drop old deltas.
        manifest.edge_journal.set_snapshot(rel, checkpoint_night_id);
        manifest.edge_journal.drop_deltas_leq(checkpoint_night_id);
        manifest.created_unix_s = created_unix_s;

        let _deleted = self.cleanup_unreferenced_deltas(manifest)?;

        Ok(())
    }

    /// Delete edge delta files on disk that are not referenced by the manifest.
    ///
    /// This is the safest cleanup strategy:
    /// - the manifest is treated as the source of truth,
    /// - any delta file not listed in `manifest.edge_journal.deltas` is considered
    ///   obsolete and can be deleted.
    ///
    /// Parameters
    /// ----------
    /// manifest : &Manifest
    ///     Current manifest after compaction / delta pruning.
    ///
    /// Returns
    /// -------
    /// Result<u64, PersistenceIoError>
    ///     Number of deleted delta files.
    ///
    /// Notes
    /// -----
    /// - Only files matching the `delta-nid=*.bin` naming scheme are considered.
    /// - Missing files are ignored (idempotent).
    /// - This does not remove snapshots.
    pub fn cleanup_unreferenced_deltas(
        &self,
        manifest: &Manifest,
    ) -> Result<u64, PersistenceIoError> {
        // Build the set of referenced delta relative paths (as strings).
        let mut keep: ahash::AHashSet<&str> = ahash::AHashSet::new();
        for d in &manifest.edge_journal.deltas {
            keep.insert(d.delta_rel_path().as_str());
        }

        let graph_dir = self.layout.graph_dir();

        // If graph dir does not exist yet, nothing to do.
        let read_dir = match fs::read_dir(graph_dir.as_std_path()) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(e) => return Err(PersistenceIoError::Io(e)),
        };

        let mut deleted = 0u64;

        for entry in read_dir {
            let entry = entry.map_err(PersistenceIoError::Io)?;
            let path = entry.path();

            // We only care about regular files.
            let ft = entry.file_type().map_err(PersistenceIoError::Io)?;
            if !ft.is_file() {
                continue;
            }

            // Convert to Utf8Path for naming / relpath operations.
            let utf8_path = match camino::Utf8Path::from_path(&path) {
                Some(p) => p,
                None => continue, // non-utf8 path; ignore
            };

            // Only consider delta files: `delta-nid=*.bin`
            let fname = match utf8_path.file_name() {
                Some(s) => s,
                None => continue,
            };
            if !fname.starts_with("delta-nid=") || !fname.ends_with(".bin") {
                continue;
            }

            // Compute relpath under root: `graph/<fname>`
            let Some(rel) = self.layout.to_relative(utf8_path) else {
                continue; // not under root (should not happen)
            };

            // If not referenced by manifest, delete it.
            if !keep.contains(rel.as_str()) {
                match fs::remove_file(path) {
                    Ok(()) => deleted += 1,
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                        // already deleted, fine
                    }
                    Err(e) => return Err(PersistenceIoError::Io(e)),
                }
            }
        }

        Ok(deleted)
    }

    /// Apply a list of operations to an in-memory edge map.
    ///
    /// Notes
    /// -----
    /// This performs a purely logical update (no validation beyond basic
    /// self-consistency). Validation (e.g. seed existence) happens later when
    /// converting to borrowed edges using `SeedStore`.
    fn apply_ops(
        &self,
        map: &mut AHashMap<EdgeKey, Edge>,
        ops: Vec<EdgeOp>,
    ) -> Result<(), PersistenceIoError> {
        for op in ops {
            match op {
                EdgeOp::Upsert { key, edge } => {
                    // Optional safety: ensure key matches edge endpoints
                    // (defensive; remove if you want max speed)
                    debug_assert!(key.from == edge.from && key.to == edge.to);
                    map.insert(key, edge);
                }
                EdgeOp::Remove { key } => {
                    map.remove(&key);
                }
            }
        }
        Ok(())
    }

    /// Decide whether an edge key is inside a window.
    #[inline]
    fn edge_key_in_window(&self, key: EdgeKey, w: PairingMode) -> bool {
        w.contains(key.from.night_id) && w.contains(key.to.night_id)
    }
}
