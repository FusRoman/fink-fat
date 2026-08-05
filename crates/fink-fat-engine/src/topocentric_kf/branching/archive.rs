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

use photom::observation_dataset::ObsId;

use crate::topocentric_kf::branching::branch_id::BranchId;

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
}
