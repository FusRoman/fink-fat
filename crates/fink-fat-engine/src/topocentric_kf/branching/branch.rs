//! A single candidate association history for one tracked object.

use photom::observation_dataset::{ObsId, observation::Observation};

use crate::topocentric_kf::branching::branch_id::{self, BranchId};
use crate::topocentric_kf::kalman_bank::KFBank;

/// One candidate history for a single object: a bank state plus the
/// cumulative log-likelihood-ratio (LLR) that justifies it against the
/// clutter/no-detection background.
///
/// A `Branch` is produced once per candidate observation (or the null
/// hypothesis) each time a bank's search region contains ambiguous matches.
/// See the module-level design note in `kalman_update_instruction.md`.
#[derive(Clone)]
pub struct Branch<'state_lf, 'bank_config> {
    /// Bank state for this branch (post `branch_with`/`branch_null` + the
    /// usual intra-bank `moment_match_merge` cleanup).
    pub bank: KFBank<'state_lf, 'bank_config>,
    /// Cumulative log-LR since this branch's root (night-0) bank.
    pub cumulative_llr: f64,
    /// Identity of the original (night-0) bank this branch ultimately
    /// traces back to — distinct from `branch_id` (this specific candidate
    /// history). Used to group sibling branches for top-B pruning.
    pub lineage_id: u64,
    /// `branch_id` of the branch this one was spawned from.
    pub parent_branch_id: u64,
    /// Stable id of this branch.
    pub branch_id: u64,
    /// `branch_id` of the ancestor branch that anchors the current N-scan
    /// window (see [`ancestor_creation_step`](Self::ancestor_creation_step)).
    pub ancestor_at_scan_horizon: u64,
    /// Night index at which `ancestor_at_scan_horizon` was last reset.
    ///
    /// Paired with `ancestor_at_scan_horizon`, this lets
    /// [`apply_n_scan_pruning`](super::pruning::apply_n_scan_pruning) decide
    /// *when* the N-scan window has elapsed and the horizon should roll
    /// forward, without retaining the full branch tree.
    pub ancestor_creation_step: usize,
    /// Human-readable, deterministic id shared by every branch in this
    /// lineage (`FF{YYYY}{suffix}` — see
    /// [`branch_id::lineage_designation`]). Unlike `lineage_id`/`branch_id`,
    /// this is stable across process runs: recomputing the same input data
    /// reproduces the same id. Groups sibling candidates for the same
    /// tracked object; use [`Self::designation`] for a per-branch unique id
    /// suitable as a `trajectory_id` when persisting to disk.
    pub lineage_designation: BranchId,
}

impl<'state_lf, 'bank_config> Branch<'state_lf, 'bank_config> {
    /// Create the initial branch for a freshly built (night-0) bank: its own
    /// lineage root, zero cumulative LLR, anchoring its own N-scan horizon.
    ///
    /// # Arguments
    /// * `bank` – Bank built by
    ///   [`build_kf_bank_collection`](crate::topocentric_kf::bank_collection::build_kf_bank_collection).
    /// * `lineage_id`, `branch_id` – Identity assigned by the caller (see the
    ///   per-night orchestrator's monotonic id counters).
    pub fn seed(bank: KFBank<'state_lf, 'bank_config>, lineage_id: u64, branch_id: u64) -> Self {
        let epoch = bank
            .best()
            .expect("a freshly built bank has at least one live hypothesis")
            .kf
            .epoch;
        let lineage_designation = branch_id::lineage_designation(bank.track_ids(), epoch);
        Self {
            bank,
            cumulative_llr: 0.0,
            lineage_id,
            parent_branch_id: branch_id,
            branch_id,
            ancestor_at_scan_horizon: branch_id,
            ancestor_creation_step: 0,
            lineage_designation,
        }
    }

    /// Spawn an "observation branch": `predicted_bank` (already
    /// [`KFBank::predict_to`]'d) associated with `obs`.
    ///
    /// # Arguments
    /// * `predicted_bank` – The parent branch's bank, propagated once to
    ///   `obs`'s epoch (shared across every candidate branch spawned this
    ///   night — see [`KFBank::predict_to`]).
    /// * `parent` – Branch this one is spawned from; supplies
    ///   `cumulative_llr`, lineage/ancestry bookkeeping.
    /// * `llr_delta` – This candidate's LLR contribution, from
    ///   [`observation_llr_delta`](super::llr_score::observation_llr_delta).
    /// * `branch_id` – Id assigned to the new branch.
    ///
    /// # Returns
    /// `None` if every hypothesis in `predicted_bank` was gated or failed
    /// against `obs` (see [`KFBank::branch_with`]) — this candidate history
    /// is not viable.
    pub fn from_observation(
        predicted_bank: &KFBank<'state_lf, 'bank_config>,
        parent: &Branch<'state_lf, 'bank_config>,
        obs: &Observation,
        llr_delta: f64,
        branch_id: u64,
    ) -> Option<Self> {
        let (branched_bank, _mixture_likelihood_z) = predicted_bank.branch_with(obs)?;
        Some(Self {
            bank: branched_bank,
            cumulative_llr: parent.cumulative_llr + llr_delta,
            lineage_id: parent.lineage_id,
            parent_branch_id: parent.branch_id,
            branch_id,
            ancestor_at_scan_horizon: parent.ancestor_at_scan_horizon,
            ancestor_creation_step: parent.ancestor_creation_step,
            lineage_designation: parent.lineage_designation.clone(),
        })
    }

    /// Spawn the "null" (missed-detection) branch: `predicted_bank` left
    /// unassociated this night.
    ///
    /// # Arguments
    /// * `predicted_bank` – The parent branch's bank, propagated to the
    ///   target epoch (see [`KFBank::predict_to`]).
    /// * `parent` – Branch this one is spawned from.
    /// * `llr_delta` – From
    ///   [`null_branch_llr_delta`](super::llr_score::null_branch_llr_delta).
    /// * `branch_id` – Id assigned to the new branch.
    pub fn from_null(
        predicted_bank: &KFBank<'state_lf, 'bank_config>,
        parent: &Branch<'state_lf, 'bank_config>,
        llr_delta: f64,
        branch_id: u64,
    ) -> Self {
        Self {
            bank: predicted_bank.branch_null(),
            cumulative_llr: parent.cumulative_llr + llr_delta,
            lineage_id: parent.lineage_id,
            parent_branch_id: parent.branch_id,
            branch_id,
            ancestor_at_scan_horizon: parent.ancestor_at_scan_horizon,
            ancestor_creation_step: parent.ancestor_creation_step,
            lineage_designation: parent.lineage_designation.clone(),
        }
    }

    /// Association history of this branch's bank, in chronological order.
    pub fn track_ids(&self) -> &[ObsId] {
        self.bank.track_ids()
    }

    /// Human-readable id unique to this branch's exact association history
    /// (`{lineage_designation}-{suffix}`), suitable as a `trajectory_id` when
    /// persisting `(trajectory_id, observation_id)` rows to disk: two
    /// branches that have consumed different observations always get
    /// different ids. A "null" branch shares its parent's id, since it
    /// covers exactly the same observation set.
    ///
    /// Computed on demand rather than stored, since it depends only on
    /// `lineage_designation` (fixed after `seed`) and this branch's own
    /// (also fixed) `track_ids`.
    pub fn designation(&self) -> BranchId {
        branch_id::branch_designation(&self.lineage_designation, self.track_ids())
    }
}
