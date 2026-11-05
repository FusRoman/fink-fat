//! Night-to-night container for the whole set of live branches — the single
//! entry point for running fink-fat's per-night loop.
//!
//! # Integration point
//!
//! Night 0 and every subsequent night are the **same call**:
//! [`BranchCollection::advance_one_night`] skips the update phase when
//! `self` is empty (nothing to propagate yet) and runs it otherwise. A
//! caller never special-cases "first night" vs. "later night":
//!
//! ```ignore
//! let mut collection = BranchCollection::empty();
//! for (night_id, night_obs) in &nights_in_order {
//!     if collection.is_night_closed(*night_id) {
//!         continue; // already advanced by a previous run
//!     }
//!     let step = collection.current_step;
//!     collection = collection.advance_one_night(
//!         night_obs, &obs_dataset, &engine_config, &kalman_context, step, Some(*night_id),
//!     )?;
//! }
//! collection.save_snapshot_to_disk(&snapshot_path)?;
//! ```
//!
//! A streaming caller that receives a night visit by visit drives the same
//! pipeline through a [`NightSession`] instead.
//!
//! # Snapshot file format
//!
//! [`BranchCollection::save_snapshot_to_disk`] writes a
//! [`SNAPSHOT_HEADER_LEN`]-byte header — the [`SNAPSHOT_MAGIC`], the format
//! version ([`SNAPSHOT_VERSION`], little-endian `u32`) and zero padding —
//! followed by the rkyv-encoded [`BranchCollectionSnapshot`]. The header
//! length is a multiple of 16 so the payload keeps rkyv's alignment inside
//! the memory-mapped file. A file without the magic is a legacy, pre-header
//! snapshot and is still readable (see [`parse_snapshot_header`]).

use std::collections::HashSet;
use std::io::{BufWriter, Write};

use camino::{Utf8Path, Utf8PathBuf};
use photom::{
    NightId,
    observation_dataset::{ObsDataset, ObsId, observation::Observation},
};

use crate::{
    engine_config::{kalman_context::KalmanContext, main_config::EngineConfig},
    error::{EngineError, FinkFatError, SnapshotError},
    spacetime_bucket::healpix_binner::HealpixBinner,
    topocentric_kf::branching::{
        ArchivedTrajectory, Branch, BranchSnapshot,
        night_session::{IdCounters, NightSession},
        orchestrate::{GateRecord, OrchestrateEvent},
        visit::group_observations_into_visits,
    },
};

/// Filename the [`BranchCollectionSnapshot`] is written to under
/// [`EngineConfig::storage_path_buf`] — shared by `fink-fat` (the real
/// engine binary, which writes it after the last night of a run, and every
/// `--snapshot-every N` nights when that flag is set) and `fink-fat-eval`
/// (which only reads it, for `tracking_analysis --from-snapshot`).
pub const SNAPSHOT_FILENAME: &str = "branch_collection.rkyv";

/// First bytes of every headered snapshot file (see the
/// [module doc](self#snapshot-file-format)).
pub const SNAPSHOT_MAGIC: [u8; 8] = *b"FFATSNAP";

/// Snapshot format version written by this build.
///
/// * `0` — legacy, header-less file (implicit; never written any more).
/// * `1` — header, plus persisted [`IdCounters`] and last closed night.
pub const SNAPSHOT_VERSION: u32 = 1;

/// Byte length of the snapshot header: magic, version, padding.
pub const SNAPSHOT_HEADER_LEN: usize = 16;

/// Layout of a snapshot file, as told by its first bytes (see
/// [`parse_snapshot_header`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotFormat {
    /// No magic: the whole file is a legacy, header-less rkyv payload.
    LegacyV0,
    /// Headered file; the payload starts at [`SNAPSHOT_HEADER_LEN`].
    Versioned {
        /// Format version read from the header.
        version: u32,
    },
    /// The magic is there but the header is cut short.
    TruncatedHeader,
}

/// Header written in front of a snapshot payload of format `version`.
pub fn snapshot_header(version: u32) -> [u8; SNAPSHOT_HEADER_LEN] {
    let mut header = [0; SNAPSHOT_HEADER_LEN];
    header[..SNAPSHOT_MAGIC.len()].copy_from_slice(&SNAPSHOT_MAGIC);
    header[SNAPSHOT_MAGIC.len()..SNAPSHOT_MAGIC.len() + 4].copy_from_slice(&version.to_le_bytes());
    header
}

/// Tell a snapshot file's layout from its leading `bytes` (the whole file is
/// fine). Pure: no validation of the payload itself.
pub fn parse_snapshot_header(bytes: &[u8]) -> SnapshotFormat {
    if !bytes.starts_with(&SNAPSHOT_MAGIC) {
        return SnapshotFormat::LegacyV0;
    }
    match bytes.get(SNAPSHOT_MAGIC.len()..SNAPSHOT_HEADER_LEN) {
        Some(rest) => SnapshotFormat::Versioned {
            version: u32::from_le_bytes(rest[..4].try_into().expect("4-byte slice")),
        },
        None => SnapshotFormat::TruncatedHeader,
    }
}

/// The full set of live branches tracked across nights — one entry per
/// surviving candidate association history, possibly several per original
/// bank once ambiguity has forced branching.
#[derive(Default)]
pub struct BranchCollection<'state_lf, 'bank_config> {
    pub branches: Vec<Branch<'state_lf, 'bank_config>>,
    /// Exposed for observability, not just diagnostics: observations
    /// consumed by a candidate extension of an existing lineage during the
    /// night just advanced, where that candidate branch was itself pruned
    /// before surviving to `branches` — see
    /// [`NightSession::close`]. These were already given back to that same
    /// night's discovery
    /// step this same night (see that method's doc), so a caller does not
    /// need to do anything with this field to benefit from that — it's
    /// informational only at this point (e.g. for reporting/telemetry).
    /// Empty before the first `advance_one_night` call and on nights where
    /// `self` started empty (nothing to advance/prune).
    pub last_night_consumed_then_pruned_ids: HashSet<ObsId>,
    /// Per-lineage gate records from the night just advanced — passive
    /// diagnostics for gate-selectivity analysis (see
    /// [`GateRecord`]). Not persisted in the
    /// snapshot (ephemeral, one night's worth); empty before the first
    /// `advance_one_night` and on nights that started empty.
    pub last_night_gate_records: Vec<GateRecord>,
    /// Night index this collection was last advanced to — the `current_step`
    /// a caller should pass into the next [`Self::advance_one_night`] call.
    /// `0` for a fresh/empty collection.
    pub current_step: usize,
    /// Next free branch / lineage ids, persisted so that no id is ever
    /// handed out twice across nights or process restarts (see
    /// [`IdCounters`]).
    pub ids: IdCounters,
    /// Last night advanced with a known [`NightId`]; `None` for a fresh
    /// collection, a legacy snapshot, or when no night was ever named. A
    /// night at or before it is refused by [`NightSession::begin`] (see
    /// [`Self::is_night_closed`]).
    pub last_closed_night_id: Option<NightId>,
    /// Reconstructions of the lineages that stopped being propagated
    /// **during the night just advanced** — see [`ArchivedTrajectory`] and
    /// [`purge_stale_lineages`](super::pruning::purge_stale_lineages).
    ///
    /// **Not cumulative** during live tracking: [`Self::advance_one_night`]
    /// returns only this night's fresh batch here (like `last_night_*`), not
    /// the run's full archive — the caller (`fink-fat track`'s loop) is
    /// expected to append it to the on-disk log
    /// ([`super::write_archived_batch`]) and drop it before the next call,
    /// rather than let it accumulate in RAM for the whole run (that
    /// accumulation, plus a full clone of it every night, is exactly what
    /// this field's contract used to cost before archived trajectories
    /// moved to a dedicated append-only log — see
    /// [`super::read_archived_log`]).
    ///
    /// Offline consumers (`fink-fat convert`, `tracking_analysis`) that need
    /// the *full* run's archive should instead load it explicitly with
    /// [`super::read_archived_log`] and assign it to this field themselves
    /// after loading a snapshot — at that point it *is* the cumulative
    /// history, since nothing else populates it. This dual meaning
    /// (per-night delta during tracking vs. explicit cumulative load
    /// offline) is a deliberate trade-off to avoid a signature change
    /// rippling through every offline consumer of `.archived`.
    pub archived: Vec<ArchivedTrajectory>,
}

/// Owned, borrow-free snapshot of a [`BranchCollection`], for persisting the
/// pipeline's state to disk at the end of a night and reloading it at the
/// start of the next one (see [`BranchCollection::to_snapshot`]).
///
/// Neither the live [`KalmanContext`]
/// nor the [`EngineConfig`] are part of the snapshot: both are cheap for the
/// caller to hold onto (or rebuild) across a process restart, and are
/// re-supplied to [`BranchCollection::from_snapshot`].
///
/// On disk it is framed by a versioned header (see the
/// [module doc](self#snapshot-file-format)); this struct is the payload of
/// format [`SNAPSHOT_VERSION`].
#[derive(Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct BranchCollectionSnapshot {
    pub branches: Vec<BranchSnapshot>,
    pub last_night_consumed_then_pruned_ids: HashSet<ObsId>,
    pub current_step: usize,
    /// See [`BranchCollection::ids`].
    pub ids: IdCounters,
    /// See [`BranchCollection::last_closed_night_id`], as its raw `u32`.
    pub last_closed_night_id: Option<u32>,
}

/// Payload of a legacy, header-less snapshot (format `0`), kept only to read
/// files written before the format was versioned.
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct BranchCollectionSnapshotV0 {
    branches: Vec<BranchSnapshot>,
    last_night_consumed_then_pruned_ids: HashSet<ObsId>,
    current_step: usize,
}

impl From<BranchCollectionSnapshotV0> for BranchCollectionSnapshot {
    /// Upgrade a legacy payload. Its id counters were never persisted, so
    /// they are rebuilt one past the live ids — the best available bound, but
    /// ids of lineages archived before the upgrade may be issued again.
    fn from(legacy: BranchCollectionSnapshotV0) -> Self {
        let ids = IdCounters::above(legacy.branches.iter().map(|b| (b.branch_id, b.lineage_id)));
        Self {
            branches: legacy.branches,
            last_night_consumed_then_pruned_ids: legacy.last_night_consumed_then_pruned_ids,
            current_step: legacy.current_step,
            ids,
            last_closed_night_id: None,
        }
    }
}

/// Decode a whole snapshot file's `bytes`, whatever its format.
///
/// `bytes` must start on a 16-byte boundary (true of a memory map) so the
/// payload keeps rkyv's alignment.
///
/// # Errors
/// [`SnapshotError::TruncatedHeader`] / [`SnapshotError::UnsupportedVersion`]
/// for a bad header, [`SnapshotError::Decode`] when the payload fails rkyv
/// validation.
pub fn decode_snapshot(bytes: &[u8]) -> Result<BranchCollectionSnapshot, SnapshotError> {
    let decode_err = |e: rkyv::rancor::Error| SnapshotError::Decode(e.to_string());
    match parse_snapshot_header(bytes) {
        SnapshotFormat::Versioned {
            version: SNAPSHOT_VERSION,
        } => rkyv::from_bytes::<BranchCollectionSnapshot, rkyv::rancor::Error>(
            &bytes[SNAPSHOT_HEADER_LEN..],
        )
        .map_err(decode_err),
        SnapshotFormat::Versioned { version } => Err(SnapshotError::UnsupportedVersion(version)),
        SnapshotFormat::TruncatedHeader => Err(SnapshotError::TruncatedHeader),
        SnapshotFormat::LegacyV0 => {
            let legacy = rkyv::from_bytes::<BranchCollectionSnapshotV0, rkyv::rancor::Error>(bytes)
                .map_err(decode_err)?;
            tracing::warn!(
                "legacy snapshot without header: id counters rebuilt from live branches, \
                 last closed night unknown"
            );
            Ok(legacy.into())
        }
    }
}

/// Read and decode the snapshot file at `path` (see [`decode_snapshot`]).
///
/// The file is memory-mapped rather than read into a heap-allocated
/// `Vec<u8>`: at the scale this snapshot can reach, that would be an extra
/// full-size copy of the file sitting in memory for the duration of the
/// (already memory-heavy) deserialization step.
pub fn read_snapshot_file(path: &Utf8Path) -> Result<BranchCollectionSnapshot, EngineError> {
    let file = std::fs::File::open(path).map_err(FinkFatError::Io)?;
    // SAFETY: `path` is fink-fat's own snapshot file, not modified by any
    // other process while a run holds it open (it is only ever replaced by
    // an atomic rename), so the mapped bytes stay valid and stable for the
    // lifetime of this read-only mapping.
    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(FinkFatError::Io)?;
    Ok(decode_snapshot(&mmap)?)
}

/// Durably write `snapshot` to `path` in the current format
/// ([`SNAPSHOT_VERSION`]).
///
/// The payload is streamed into the file instead of first materializing the
/// whole serialized buffer on the heap (a multi-GB allocation at scale). It
/// goes to a `.tmp` sibling that is fsynced, then atomically renamed over
/// `path`, and the parent directory is fsynced so the rename itself survives
/// a crash: at any instant `path` holds either the previous snapshot or the
/// new one, complete.
pub fn write_snapshot_file(
    snapshot: &BranchCollectionSnapshot,
    path: &Utf8Path,
) -> Result<(), EngineError> {
    let io_err = |e: std::io::Error| EngineError::from(FinkFatError::Io(e));
    let tmp_path = Utf8PathBuf::from(format!("{path}.tmp"));

    let mut out = BufWriter::new(std::fs::File::create(&tmp_path).map_err(io_err)?);
    out.write_all(&snapshot_header(SNAPSHOT_VERSION))
        .map_err(io_err)?;
    let out = rkyv::api::high::to_bytes_in::<_, rkyv::rancor::Error>(
        snapshot,
        rkyv::ser::writer::IoWriter::new(out),
    )
    .map_err(|e| SnapshotError::Encode(e.to_string()))?
    .into_inner();
    let file = out.into_inner().map_err(|e| io_err(e.into_error()))?;
    file.sync_all().map_err(io_err)?;
    drop(file);

    std::fs::rename(&tmp_path, path).map_err(io_err)?;
    sync_parent_dir(path).map_err(io_err)
}

/// Fsync the directory holding `path`, making a rename into it durable.
#[cfg(unix)]
fn sync_parent_dir(path: &Utf8Path) -> std::io::Result<()> {
    let dir = match path.parent() {
        Some(parent) if !parent.as_str().is_empty() => parent,
        _ => Utf8Path::new("."),
    };
    std::fs::File::open(dir)?.sync_all()
}

/// Directories cannot be opened for fsync outside Unix; the rename is still
/// atomic, only its durability across a power loss is not guaranteed.
#[cfg(not(unix))]
fn sync_parent_dir(_path: &Utf8Path) -> std::io::Result<()> {
    Ok(())
}

use crate::logging::LogTarget;

/// Structured log events for the per-night `BranchCollection` pipeline
/// entry point. See [`crate::logging`] for the `.emit()` pattern.
pub enum CollectionEvent {
    Start,
    InputSummary {
        n_observations: usize,
        current_step: usize,
    },
    SkipPropagation,
    Advancing,
    SeedingNewLineages,
    End {
        n_branches: usize,
    },
}

crate::impl_log_target!(
    CollectionEvent,
    "collection",
    "Per-night BranchCollection pipeline entry point (advance_one_night)",
    [tracing::Level::INFO, tracing::Level::DEBUG]
);

impl CollectionEvent {
    pub fn emit(&self) {
        use CollectionEvent::*;
        match self {
            Start => {
                tracing::info!(target: CollectionEvent::TARGET, "Start of the advance one night pipeline")
            }
            InputSummary {
                n_observations,
                current_step,
            } => tracing::debug!(
                target: CollectionEvent::TARGET, n_observations, current_step, "Advance one night: input summary"
            ),
            SkipPropagation => tracing::info!(
                target: CollectionEvent::TARGET, "Branches is empty, skip the kalman propagation"
            ),
            Advancing => tracing::info!(
                target: CollectionEvent::TARGET, "Find previous branches, perform kalman one night advance"
            ),
            SeedingNewLineages => {
                tracing::info!(target: CollectionEvent::TARGET, "Start new seeds generation")
            }
            End { n_branches } => tracing::info!(
                target: CollectionEvent::TARGET, n_branches, "End of the advance one night pipeline"
            ),
        }
    }

    pub fn span() -> tracing::Span {
        tracing::info_span!(target: CollectionEvent::TARGET, "Advance one night")
    }
}

impl<'state_lf, 'bank_config> BranchCollection<'state_lf, 'bank_config> {
    /// Convert to an owned, borrow-free snapshot suitable for on-disk
    /// persistence (see [`BranchCollectionSnapshot`] and
    /// [`Self::save_snapshot_to_disk`]).
    pub fn to_snapshot(&self) -> BranchCollectionSnapshot {
        BranchCollectionSnapshot {
            branches: self.branches.iter().map(Branch::to_snapshot).collect(),
            last_night_consumed_then_pruned_ids: self.last_night_consumed_then_pruned_ids.clone(),
            current_step: self.current_step,
            ids: self.ids,
            last_closed_night_id: self.last_closed_night_id.map(u32::from),
        }
    }

    /// Rebuild a full [`BranchCollection`] from a snapshot previously
    /// produced by [`Self::to_snapshot`], reattaching the live
    /// `kalman_context`/`engine_config` the caller already holds (the same
    /// arguments it would pass to [`Self::advance_one_night`]).
    pub fn from_snapshot(
        snapshot: BranchCollectionSnapshot,
        kalman_context: &'state_lf KalmanContext,
        engine_config: &'bank_config EngineConfig,
    ) -> Self {
        Self {
            branches: snapshot
                .branches
                .into_iter()
                .map(|b| Branch::from_snapshot(b, kalman_context, &engine_config.kfbank_config))
                .collect(),
            last_night_consumed_then_pruned_ids: snapshot.last_night_consumed_then_pruned_ids,
            last_night_gate_records: Vec::new(),
            current_step: snapshot.current_step,
            ids: snapshot.ids,
            last_closed_night_id: snapshot.last_closed_night_id.map(NightId),
            // Archived trajectories no longer round-trip through the
            // snapshot (see `BranchCollection::archived`'s doc) — a run
            // resuming from this snapshot has no need for its prior
            // archive, which lives entirely in the separate append-only log
            // instead.
            archived: Vec::new(),
        }
    }

    /// Read+deserialize+reconstruct a [`BranchCollection`] from the
    /// snapshot file at `path` ([`read_snapshot_file`], then
    /// [`Self::from_snapshot`]). Legacy header-less files are upgraded on the
    /// fly (see [`decode_snapshot`]).
    ///
    /// # Errors
    /// The file is missing or unreadable, or its header or payload is invalid.
    pub fn load_snapshot_from_disk(
        path: &Utf8Path,
        kalman_context: &'state_lf KalmanContext,
        engine_config: &'bank_config EngineConfig,
    ) -> Result<Self, EngineError> {
        let snapshot = read_snapshot_file(path)?;
        Ok(Self::from_snapshot(snapshot, kalman_context, engine_config))
    }

    /// Durably persist this collection to `path` (see
    /// [`write_snapshot_file`] for the crash-safety guarantees).
    pub fn save_snapshot_to_disk(&self, path: &Utf8Path) -> Result<(), EngineError> {
        write_snapshot_file(&self.to_snapshot(), path)
    }

    /// Whether `night_id` was already advanced into this collection — it is
    /// at or before [`Self::last_closed_night_id`]. A resumed run skips such
    /// nights; [`NightSession::begin`] refuses them.
    pub fn is_night_closed(&self, night_id: NightId) -> bool {
        self.last_closed_night_id
            .is_some_and(|last_closed| night_id <= last_closed)
    }

    pub fn empty() -> Self {
        BranchCollection {
            branches: Vec::new(),
            last_night_consumed_then_pruned_ids: HashSet::new(),
            last_night_gate_records: Vec::new(),
            current_step: 0,
            ids: IdCounters::default(),
            last_closed_night_id: None,
            archived: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.branches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.branches.is_empty()
    }

    /// Advance the collection by one night — the batch driver of a
    /// [`NightSession`].
    ///
    /// Groups `night_obs` into visits
    /// ([`group_observations_into_visits`]), feeds them in chronological
    /// order to [`NightSession::advance_visit`], then closes the night
    /// ([`NightSession::close`]: N-scan pruning, stale purge, discovery of new
    /// lineages from the unclaimed observations). When `self` is empty (night
    /// 0, or no lineage survived) the visit loop is skipped entirely and every
    /// observation goes to discovery. A caller never special-cases "first
    /// night" vs. "later night".
    ///
    /// # Arguments
    /// * `night_obs` – This night's observations.
    /// * `obs_dataset`, `kalman_context` – Shared ephemeris/observation
    ///   context, threaded through to both phases.
    /// * `engine_config` – Supplies the branching/pruning parameters
    ///   (`advance_params`), the discovery parameters and the HEALPix depth.
    /// * `current_step` – Current night index, for N-scan bookkeeping.
    /// * `night_id` – The night being advanced, when known (see
    ///   [`NightSession::begin`]).
    ///
    /// # Returns
    /// The next `BranchCollection`.
    ///
    /// # Errors
    /// [`EngineError::NightAlreadyClosed`] when `night_id` was already
    /// advanced, or the discovery phase's bank-building error (see
    /// [`build_kf_bank_collection_from_observations`](crate::topocentric_kf::kalman_bank::from_seeds::build_kf_bank_collection_from_observations)).
    pub fn advance_one_night(
        &self,
        night_obs: &[&Observation],
        obs_dataset: &ObsDataset,
        engine_config: &'bank_config EngineConfig,
        kalman_context: &'state_lf KalmanContext,
        current_step: usize,
        night_id: Option<NightId>,
    ) -> Result<Self, EngineError> {
        let _enter = CollectionEvent::span().entered();

        CollectionEvent::Start.emit();
        CollectionEvent::InputSummary {
            n_observations: night_obs.len(),
            current_step,
        }
        .emit();

        let spatial_binner = HealpixBinner::new(engine_config.healpix_depth);
        let params = &engine_config.advance_params;
        let mut session = NightSession::begin(self, night_id, current_step)?;

        if self.branches.is_empty() {
            CollectionEvent::SkipPropagation.emit();
        } else {
            CollectionEvent::Advancing.emit();
            let _advance = OrchestrateEvent::span().entered();
            let visits =
                group_observations_into_visits(night_obs, params.visit_epoch_tolerance_days);
            let n_visits = visits.len();
            OrchestrateEvent::VisitSummary { n_visits }.emit();
            let visit_progress_modulo = (n_visits / 100).max(1);
            for (visit_index, visit) in visits.iter().enumerate() {
                if visit_index % visit_progress_modulo == 0 {
                    OrchestrateEvent::VisitProgress {
                        visit_index,
                        n_visits,
                    }
                    .emit();
                }
                session.advance_visit(visit, obs_dataset, kalman_context, params, &spatial_binner);
            }
        }

        CollectionEvent::SeedingNewLineages.emit();
        let next = session.close(
            night_obs,
            obs_dataset,
            kalman_context,
            engine_config,
            &spatial_binner,
        )?;

        CollectionEvent::End {
            n_branches: next.branches.len(),
        }
        .emit();
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Branch-free snapshot carrying distinctive values in every other field.
    fn sample_snapshot() -> BranchCollectionSnapshot {
        BranchCollectionSnapshot {
            branches: Vec::new(),
            last_night_consumed_then_pruned_ids: HashSet::from([3, 9]),
            current_step: 12,
            ids: IdCounters {
                next_branch_id: 1_000,
                next_lineage_id: 77,
            },
            last_closed_night_id: Some(60_312),
        }
    }

    fn assert_same(a: &BranchCollectionSnapshot, b: &BranchCollectionSnapshot) {
        assert_eq!(a.branches.len(), b.branches.len());
        assert_eq!(
            a.last_night_consumed_then_pruned_ids,
            b.last_night_consumed_then_pruned_ids
        );
        assert_eq!(a.current_step, b.current_step);
        assert_eq!(a.ids, b.ids);
        assert_eq!(a.last_closed_night_id, b.last_closed_night_id);
    }

    #[test]
    fn header_round_trips_through_the_parser() {
        let header = snapshot_header(SNAPSHOT_VERSION);
        assert_eq!(
            parse_snapshot_header(&header),
            SnapshotFormat::Versioned {
                version: SNAPSHOT_VERSION
            }
        );
        assert_eq!(&header[12..], &[0; 4], "padding must be zeroed");
    }

    #[test]
    fn bytes_without_magic_are_legacy() {
        assert_eq!(parse_snapshot_header(&[]), SnapshotFormat::LegacyV0);
        assert_eq!(parse_snapshot_header(&[0xAB; 64]), SnapshotFormat::LegacyV0);
    }

    #[test]
    fn magic_with_short_header_is_truncated() {
        let header = snapshot_header(SNAPSHOT_VERSION);
        for len in [SNAPSHOT_MAGIC.len(), SNAPSHOT_HEADER_LEN - 1] {
            assert_eq!(
                parse_snapshot_header(&header[..len]),
                SnapshotFormat::TruncatedHeader
            );
        }
    }

    #[test]
    fn unknown_version_is_refused() {
        let header = snapshot_header(SNAPSHOT_VERSION + 1);
        assert!(matches!(
            decode_snapshot(&header),
            Err(SnapshotError::UnsupportedVersion(v)) if v == SNAPSHOT_VERSION + 1
        ));
    }

    #[test]
    fn v1_snapshot_round_trips_through_disk() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = Utf8PathBuf::from_path_buf(dir.path().join(SNAPSHOT_FILENAME)).unwrap();
        let snapshot = sample_snapshot();

        write_snapshot_file(&snapshot, &path).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            parse_snapshot_header(&bytes),
            SnapshotFormat::Versioned {
                version: SNAPSHOT_VERSION
            }
        );
        assert!(
            !Utf8PathBuf::from(format!("{path}.tmp")).exists(),
            "the temporary file must be renamed away"
        );
        assert_same(&read_snapshot_file(&path).unwrap(), &snapshot);
    }

    #[test]
    fn rewriting_a_snapshot_replaces_it() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = Utf8PathBuf::from_path_buf(dir.path().join(SNAPSHOT_FILENAME)).unwrap();
        write_snapshot_file(&sample_snapshot(), &path).unwrap();
        let mut next = sample_snapshot();
        next.current_step += 1;
        next.last_closed_night_id = Some(60_313);
        write_snapshot_file(&next, &path).unwrap();
        assert_same(&read_snapshot_file(&path).unwrap(), &next);
    }

    #[test]
    fn legacy_v0_file_is_upgraded_on_read() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = Utf8PathBuf::from_path_buf(dir.path().join(SNAPSHOT_FILENAME)).unwrap();
        let legacy = BranchCollectionSnapshotV0 {
            branches: Vec::new(),
            last_night_consumed_then_pruned_ids: HashSet::from([4]),
            current_step: 5,
        };
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&legacy).unwrap();
        std::fs::write(&path, &bytes).unwrap();

        let upgraded = read_snapshot_file(&path).unwrap();
        assert_eq!(upgraded.current_step, 5);
        assert_eq!(
            upgraded.last_night_consumed_then_pruned_ids,
            HashSet::from([4])
        );
        assert_eq!(upgraded.ids, IdCounters::default());
        assert_eq!(upgraded.last_closed_night_id, None);
    }

    #[test]
    fn is_night_closed_compares_with_the_last_closed_night() {
        let mut collection = BranchCollection::empty();
        assert!(!collection.is_night_closed(NightId(0)));
        collection.last_closed_night_id = Some(NightId(10));
        assert!(collection.is_night_closed(NightId(9)));
        assert!(collection.is_night_closed(NightId(10)));
        assert!(!collection.is_night_closed(NightId(11)));
    }
}
