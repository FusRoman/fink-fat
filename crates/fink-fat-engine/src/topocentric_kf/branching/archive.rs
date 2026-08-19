//! Closed trajectories: the reconstructions of lineages that stopped being
//! propagated but whose association history is still a result.
//!
//! # Why this exists
//!
//! Until this module, the engine's only output was its set of *live*
//! branches. [`purge_stale_lineages`](super::pruning::purge_stale_lineages)
//! therefore did two things at once: it stopped propagating a stale lineage
//! (the point — a coasting zombie keeps widening its error box, mis-associating,
//! and burning a whole `KFBank` worth of propagation every visit) **and** it
//! destroyed that lineage's accumulated `track_ids`, which is the
//! reconstruction itself.
//!
//! Measured on a 200-night ZTF-cadence run, enabling the purge moved 37 561
//! trajectories from "reconstructed" to "not reconstructed at all" while
//! genuinely removing ~52 000 contaminated ones — a real gain paid for with a
//! loss that was never necessary. The two effects are separable: a lineage can
//! stop being propagated *and* keep its result.
//!
//! [`ArchivedTrajectory`] is that result. It is deliberately **not** a
//! [`Branch`](super::Branch): a `Branch` owns a
//! [`KFBank`](crate::topocentric_kf::kalman_bank::KFBank) holding dozens of
//! hypotheses (the run above averaged 87 per branch), which is exactly the
//! state we want to stop paying for. What downstream consumers actually read
//! off a finished trajectory is its observation list, so that — plus enough
//! provenance to rank, group and audit it — is all this keeps.

use std::io::{self, Read, Write};

use camino::Utf8Path;
use photom::observation_dataset::ObsId;

use crate::error::{EngineError, FinkFatError};
use crate::topocentric_kf::branching::branch_id::BranchId;
use crate::topocentric_kf::single_kalman::KFStateSnapshot;

/// Filename (sibling of [`super::SNAPSHOT_FILENAME`] under `storage_path`) of
/// the append-only log of archived trajectories — see
/// [`write_archived_batch`]/[`read_archived_log`].
pub const ARCHIVE_LOG_FILENAME: &str = "archived_trajectories.rkyvlog";

/// A lineage's final reconstruction, retained after the lineage itself
/// stopped being propagated.
///
/// Cheap by construction (a `Vec<ObsId>` plus scalars — roughly 400 bytes for
/// a typical arc, so a full survey run's archive is measured in megabytes,
/// not gigabytes), and `rkyv`-serializable so it survives the snapshot
/// round-trip alongside the live branches.
#[derive(Clone, Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct ArchivedTrajectory {
    /// Stable, human-readable identifier of the branch this was archived
    /// from — see
    /// [`branch_designation`](super::branch_id::branch_designation). Suitable
    /// as a `trajectory_id` when persisting `(trajectory_id, observation_id)`
    /// rows.
    pub designation: BranchId,
    /// Lineage this reconstruction belonged to, for grouping an archived arc
    /// with whatever else came out of the same seed.
    pub lineage_id: u64,
    /// The reconstruction: every observation the lineage associated, in
    /// chronological order. Always non-empty — a lineage carries at least its
    /// founding pair.
    pub track_ids: Vec<ObsId>,
    /// Score at the moment of archiving, on the same scale as
    /// [`Branch::cumulative_llr`](super::Branch::cumulative_llr).
    pub cumulative_llr: f64,
    /// How many *real* (non-null) observations the arc consumed — the
    /// evidence volume behind `cumulative_llr`, and the credibility gate used
    /// when deciding whether an arc was worth archiving at all.
    pub n_real_updates: usize,
    /// Night index of the last real update, i.e. where the arc actually ends
    /// (as opposed to `archived_at_step`, which is when we noticed).
    pub last_real_update_step: usize,
    /// Night index at which the lineage was archived — always
    /// `>= last_real_update_step` by at least the staleness budget.
    pub archived_at_step: usize,
    /// The MAP hypothesis's Kalman state at the moment of archiving.
    ///
    /// Deliberately **one** state and not the bank: a bank averaged ~87
    /// hypotheses on the reference run (~52 kB), which is the cost this type
    /// exists to shed, while a single snapshot is ~530 B. That one state is
    /// enough to make a closed arc *propagatable* —
    /// [`KFStateSnapshot::into_kf_state`] reattaches the live
    /// [`KalmanContext`](crate::engine_config::kalman_context::KalmanContext),
    /// after which [`KFState::predict`](crate::topocentric_kf::single_kalman::KFState::predict)
    /// and [`sky_covariance`](crate::topocentric_kf::single_kalman::KFState::sky_covariance)
    /// answer "could this arc explain those observations?".
    ///
    /// Without it an archived arc is a bare list of `ObsId`s: it cannot be
    /// propagated, and `last_real_update_step` is a *night index*, not even
    /// an epoch. This field is what makes fragment linkage possible.
    pub map_state: KFStateSnapshot,
    /// Running absolute-magnitude (`H`) estimate carried over from the bank,
    /// `None` if the lineage never consumed a real observation.
    ///
    /// `H` is the object's *intrinsic* brightness, unlike apparent magnitude
    /// which varies with heliocentric distance and topocentric range, so it
    /// is the invariant two arcs of the same object must agree on. Note it is
    /// computed without the H-G phase term (see
    /// [`implied_absolute_magnitude`](super::detection_probability::implied_absolute_magnitude)),
    /// so comparisons across arcs observed at different phase angles carry a
    /// systematic offset of a few tenths of a magnitude.
    pub absolute_magnitude_estimate: Option<f64>,
    /// How many observations were folded into `absolute_magnitude_estimate`.
    pub absolute_magnitude_sample_count: u32,
}

/// Append one night's freshly archived trajectories to an open writer, as a
/// single length-prefixed `rkyv` record (`u64` little-endian byte length,
/// then the serialized `batch`). A no-op if `batch` is empty, so a quiet
/// night never grows the file.
///
/// One record per *night* rather than per trajectory: `rkyv` archives carry a
/// small fixed overhead each, and a night's batch is typically a handful of
/// trajectories, so batching amortizes that cost and keeps write syscalls
/// down over a whole run.
///
/// The caller owns the writer's lifetime (see [`read_archived_log`]'s doc for
/// why this crate does not open the file itself) — typically a `BufWriter`
/// over a file opened with `.append(true)`, kept open for the whole run and
/// flushed by the caller after each call so an archived batch survives a
/// crash as soon as it's written, independently of the (much less frequent)
/// `BranchCollection` snapshot cadence.
pub fn write_archived_batch<W: Write>(
    writer: &mut W,
    batch: &[ArchivedTrajectory],
) -> Result<(), EngineError> {
    if batch.is_empty() {
        return Ok(());
    }
    let bytes = rkyv::api::high::to_bytes::<rkyv::rancor::Error>(&batch.to_vec())
        .map_err(|e| FinkFatError::Message(e.to_string()))?;
    writer
        .write_all(&(bytes.len() as u64).to_le_bytes())
        .map_err(FinkFatError::Io)?;
    writer.write_all(&bytes).map_err(FinkFatError::Io)?;
    Ok(())
}

/// Read every batch previously written by [`write_archived_batch`] from the
/// log file at `path`, in the order they were appended, and flatten them into
/// one `Vec`.
///
/// Missing file is treated as "no trajectory archived yet" (`Ok(vec![])`),
/// not an error — a run that hasn't archived anything yet (or hasn't reached
/// its first archival) simply has no log file.
///
/// Tolerant of a truncated trailing record: if the file ends mid-length-prefix
/// or mid-payload (the process was killed while `write_archived_batch` was
/// writing, e.g. an OOM kill — precisely the failure mode this format exists
/// to survive), everything read up to that point is returned rather than
/// failing the whole read. A truncated record can only ever be the *last*
/// one, since each write is a single `write_all` of the length prefix
/// followed by a single `write_all` of the payload.
///
/// This is an offline/batch reader (used by `fink-fat convert` and
/// `tracking_analysis`, not by the live `track` loop, which never reads this
/// file back), so materializing the whole result in memory is fine.
pub fn read_archived_log(path: &Utf8Path) -> Result<Vec<ArchivedTrajectory>, EngineError> {
    let mut file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(FinkFatError::Io(e).into()),
    };

    let mut out = Vec::new();
    let mut len_buf = [0u8; 8];
    loop {
        match file.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(FinkFatError::Io(e).into()),
        }
        let len = u64::from_le_bytes(len_buf) as usize;

        let mut payload = vec![0u8; len];
        match file.read_exact(&mut payload) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(FinkFatError::Io(e).into()),
        }

        let batch = rkyv::from_bytes::<Vec<ArchivedTrajectory>, rkyv::rancor::Error>(&payload)
            .map_err(|e| FinkFatError::Message(e.to_string()))?;
        out.extend(batch);
    }
    Ok(out)
}
