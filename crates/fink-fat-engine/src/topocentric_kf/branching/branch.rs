//! A single candidate association history for one tracked object.

use photom::observation_dataset::{ObsId, observation::Observation};

use crate::engine_config::kalman_context::KalmanContext;
use crate::engine_config::kf_bank_config::KFBankConfig;
use crate::topocentric_kf::branching::branch_id::{self, BranchId};
use crate::topocentric_kf::kalman_bank::{KFBank, KFBankSnapshot};

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
    /// Night index at which this lineage's bank last consumed a real
    /// (non-null) observation — as opposed to `ancestor_creation_step`, which
    /// tracks the N-scan window anchor. Left unchanged by `from_null` (a
    /// "blank observation night" never counts as a real update), so
    /// `current_step - last_real_update_step` is exactly the lineage's
    /// staleness age, consumed by
    /// [`purge_stale_lineages`](super::pruning::purge_stale_lineages).
    pub last_real_update_step: usize,
    /// Count of real (non-null) observations this lineage has consumed since
    /// [`Self::seed`] (which starts it at 1, for the bootstrap pair) —
    /// incremented by [`Self::from_observation`], left unchanged by
    /// [`Self::from_null`]. Lets a diagnostic normalize `cumulative_llr` by
    /// evidence volume (`cumulative_llr / n_real_updates`) instead of reading
    /// the raw sum, which conflates a lineage's *coasting duration* (more
    /// null-branch penalties) with the *quality* of its evidence — see
    /// `stale_llr_floor` calibration in `fink-fat-eval`'s
    /// `print_coasting_llr_by_class`.
    pub n_real_updates: usize,
    /// Human-readable, deterministic id shared by every branch in this
    /// lineage (`FF{YYYY}{suffix}` — see
    /// [`branch_id::lineage_designation`]). Unlike `lineage_id`/`branch_id`,
    /// this is stable across process runs: recomputing the same input data
    /// reproduces the same id. Groups sibling candidates for the same
    /// tracked object; use [`Self::designation`] for a per-branch unique id
    /// suitable as a `trajectory_id` when persisting to disk.
    pub lineage_designation: BranchId,
}

/// Owned, borrow-free snapshot of a [`Branch`], for persisting a
/// [`BranchCollection`](super::BranchCollection) to disk across nights (see
/// [`Branch::to_snapshot`]).
#[derive(Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct BranchSnapshot {
    pub bank: KFBankSnapshot,
    pub cumulative_llr: f64,
    pub lineage_id: u64,
    pub parent_branch_id: u64,
    pub branch_id: u64,
    pub ancestor_at_scan_horizon: u64,
    pub ancestor_creation_step: usize,
    pub last_real_update_step: usize,
    pub n_real_updates: usize,
    pub lineage_designation: BranchId,
}

impl<'state_lf, 'bank_config> Branch<'state_lf, 'bank_config> {
    /// Convert to an owned, borrow-free snapshot suitable for on-disk
    /// persistence (see [`BranchSnapshot`]).
    pub fn to_snapshot(&self) -> BranchSnapshot {
        BranchSnapshot {
            bank: self.bank.to_snapshot(),
            cumulative_llr: self.cumulative_llr,
            lineage_id: self.lineage_id,
            parent_branch_id: self.parent_branch_id,
            branch_id: self.branch_id,
            ancestor_at_scan_horizon: self.ancestor_at_scan_horizon,
            ancestor_creation_step: self.ancestor_creation_step,
            last_real_update_step: self.last_real_update_step,
            n_real_updates: self.n_real_updates,
            lineage_designation: self.lineage_designation.clone(),
        }
    }

    /// Reattach `shared_ctx`/`config` (supplied by the caller) to rebuild a
    /// full [`Branch`].
    pub fn from_snapshot(
        snapshot: BranchSnapshot,
        shared_ctx: &'state_lf KalmanContext,
        config: &'bank_config KFBankConfig,
    ) -> Self {
        Self {
            bank: KFBank::from_snapshot(snapshot.bank, shared_ctx, config),
            cumulative_llr: snapshot.cumulative_llr,
            lineage_id: snapshot.lineage_id,
            parent_branch_id: snapshot.parent_branch_id,
            branch_id: snapshot.branch_id,
            ancestor_at_scan_horizon: snapshot.ancestor_at_scan_horizon,
            ancestor_creation_step: snapshot.ancestor_creation_step,
            last_real_update_step: snapshot.last_real_update_step,
            n_real_updates: snapshot.n_real_updates,
            lineage_designation: snapshot.lineage_designation,
        }
    }
}

impl<'state_lf, 'bank_config> Branch<'state_lf, 'bank_config> {
    /// Create the initial branch for a freshly built (night-0) bank: its own
    /// lineage root, zero cumulative LLR, anchoring its own N-scan horizon.
    ///
    /// # Arguments
    /// * `bank` – Bank built by
    ///   [`build_kf_bank_collection`](crate::topocentric_kf::kalman_bank::from_seeds::build_kf_bank_collection).
    /// * `lineage_id`, `branch_id` – Identity assigned by the caller (see the
    ///   per-night orchestrator's monotonic id counters).
    /// * `current_step` – Night index this lineage is born on; seeds
    ///   `last_real_update_step` so a freshly discovered lineage starts at
    ///   age zero, not "already stale".
    pub fn seed(
        bank: KFBank<'state_lf, 'bank_config>,
        lineage_id: u64,
        branch_id: u64,
        current_step: usize,
    ) -> Self {
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
            last_real_update_step: current_step,
            // The bootstrap pair is this lineage's first real evidence.
            n_real_updates: 1,
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
    /// * `current_step` – Current night index; recorded as
    ///   `last_real_update_step` since this branch just consumed a real
    ///   observation.
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
        current_step: usize,
    ) -> Option<Self> {
        Self::from_observation_diag(
            predicted_bank,
            parent,
            obs,
            llr_delta,
            branch_id,
            current_step,
        )
        .0
    }

    /// Like [`Self::from_observation`], but also reports how many of
    /// `predicted_bank`'s hypotheses were rejected by the chi-square gate
    /// vs. failed numerically (see [`KFBank::branch_with_diag`]) — lets
    /// diagnostic tooling (`mot_analysis`) classify *why* a real-observation
    /// update collapsed, not just *that* it did.
    ///
    /// Returns `(from_observation_result, n_gated, n_failed)`.
    pub fn from_observation_diag(
        predicted_bank: &KFBank<'state_lf, 'bank_config>,
        parent: &Branch<'state_lf, 'bank_config>,
        obs: &Observation,
        llr_delta: f64,
        branch_id: u64,
        current_step: usize,
    ) -> (Option<Self>, usize, usize) {
        let (result, n_gated, n_failed) = predicted_bank.branch_with_diag(obs);
        let branch = result.map(|(branched_bank, _mixture_likelihood_z)| Self {
            bank: branched_bank,
            cumulative_llr: parent.cumulative_llr + llr_delta,
            lineage_id: parent.lineage_id,
            parent_branch_id: parent.branch_id,
            branch_id,
            ancestor_at_scan_horizon: parent.ancestor_at_scan_horizon,
            ancestor_creation_step: parent.ancestor_creation_step,
            last_real_update_step: current_step,
            n_real_updates: parent.n_real_updates + 1,
            lineage_designation: parent.lineage_designation.clone(),
        });
        (branch, n_gated, n_failed)
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
            // A blank observation night never counts as a real update — kept
            // unchanged so `current_step - last_real_update_step` measures
            // this lineage's true staleness age (see `purge_stale_lineages`).
            last_real_update_step: parent.last_real_update_step,
            // Unchanged, same reasoning as `last_real_update_step` above.
            n_real_updates: parent.n_real_updates,
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
