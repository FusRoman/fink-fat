//! Cross-bank pruning: top-B cap and N-scan window.
//!
//! Applied every night, immediately after branching and scoring, so the
//! number of live branches never grows unbounded between nights (see
//! `kalman_update_instruction.md`). This is a **separate** pass over
//! `Vec<Branch>`, distinct from the existing intra-bank pruning in
//! [`KFBank`](crate::topocentric_kf::kalman_bank::KFBank) (which operates on
//! `Vec<Hypothesis>` within a single bank).

use std::collections::HashMap;

use crate::topocentric_kf::branching::branch::Branch;

use crate::logging::LogTarget;

/// Structured log events for cross-bank pruning (top-B cap, N-scan window).
/// See [`crate::logging`] for the `.emit()` pattern.
pub enum PruningEvent {
    TopBSummary {
        n_before: usize,
        n_after: usize,
        n_lineages: usize,
    },
    NScanSummary {
        n_before: usize,
        n_after: usize,
    },
}

crate::impl_log_target!(
    PruningEvent,
    "pruning",
    "Cross-bank pruning: top-B cap per lineage and N-scan window",
    [tracing::Level::DEBUG]
);

impl PruningEvent {
    pub fn emit(&self) {
        use PruningEvent::*;
        match self {
            TopBSummary {
                n_before,
                n_after,
                n_lineages,
            } => tracing::debug!(
                target: PruningEvent::TARGET, n_before, n_after, n_lineages, "Top-B cap per lineage"
            ),
            NScanSummary { n_before, n_after } => tracing::debug!(
                target: PruningEvent::TARGET, n_before, n_after, "N-scan pruning"
            ),
        }
    }
}

/// Keep at most `cap` branches per lineage, sorted by descending
/// `cumulative_llr`.
///
/// Mirrors `KFBank`'s intra-bank sort-and-truncate cap, but grouped by
/// `lineage_id` across banks instead of by hypothesis within one bank.
///
/// # Arguments
/// * `branches` – All branches spawned this night, across every lineage.
/// * `cap` – `B`, typically 3–5 (see design doc).
///
/// # Returns
/// The surviving branches.
pub fn cap_top_b_per_lineage<'state_lf, 'bank_config>(
    branches: Vec<Branch<'state_lf, 'bank_config>>,
    cap: usize,
) -> Vec<Branch<'state_lf, 'bank_config>> {
    let n_before = branches.len();
    let mut by_lineage: HashMap<u64, Vec<Branch<'state_lf, 'bank_config>>> = HashMap::new();
    for branch in branches {
        by_lineage
            .entry(branch.lineage_id)
            .or_default()
            .push(branch);
    }
    let n_lineages = by_lineage.len();

    let survivors: Vec<_> = by_lineage
        .into_values()
        .flat_map(|mut lineage_branches| {
            lineage_branches.sort_by(|a, b| b.cumulative_llr.total_cmp(&a.cumulative_llr));
            lineage_branches.truncate(cap.max(1));
            lineage_branches
        })
        .collect();

    PruningEvent::TopBSummary {
        n_before,
        n_after: survivors.len(),
        n_lineages,
    }
    .emit();

    survivors
}

/// N-scan pruning: for every branch-tree node exactly `n_scan` nights old,
/// keep only the descendant with the best `cumulative_llr` and discard its
/// siblings.
///
/// Implementation note: rather than walking a fully retained branch tree,
/// each [`Branch`] carries `ancestor_at_scan_horizon` (the id of the ancestor
/// anchoring the current window) and `ancestor_creation_step` (when that
/// anchor was set). Grouping by `(lineage_id, ancestor_at_scan_horizon)`
/// recovers "same node N nights ago" without retaining history; once a
/// group's anchor is `n_scan` nights old, the surviving branch becomes the
/// new anchor for the next window.
///
/// # Arguments
/// * `branches` – Live branches, already capped by
///   [`cap_top_b_per_lineage`].
/// * `n_scan` – Scan-back window in nights (design doc recommends 1,
///   occasionally 2).
/// * `current_step` – Current night index.
///
/// # Returns
/// Surviving branches: one per `(lineage_id, ancestor_at_scan_horizon)`
/// group, with the horizon rolled forward for groups that reached the
/// window.
pub fn apply_n_scan_pruning<'state_lf, 'bank_config>(
    branches: Vec<Branch<'state_lf, 'bank_config>>,
    n_scan: usize,
    current_step: usize,
) -> Vec<Branch<'state_lf, 'bank_config>> {
    let n_before = branches.len();
    let mut by_horizon_node: HashMap<(u64, u64), Vec<Branch<'state_lf, 'bank_config>>> =
        HashMap::new();
    for branch in branches {
        by_horizon_node
            .entry((branch.lineage_id, branch.ancestor_at_scan_horizon))
            .or_default()
            .push(branch);
    }

    let survivors: Vec<_> = by_horizon_node
        .into_values()
        .map(|node_branches| best_by_cumulative_llr(node_branches))
        .map(|mut survivor| {
            if current_step.saturating_sub(survivor.ancestor_creation_step) >= n_scan {
                survivor.ancestor_at_scan_horizon = survivor.branch_id;
                survivor.ancestor_creation_step = current_step;
            }
            survivor
        })
        .collect();

    PruningEvent::NScanSummary {
        n_before,
        n_after: survivors.len(),
    }
    .emit();

    survivors
}

/// Pick the branch with the highest `cumulative_llr` from a non-empty group.
fn best_by_cumulative_llr<'state_lf, 'bank_config>(
    branches: Vec<Branch<'state_lf, 'bank_config>>,
) -> Branch<'state_lf, 'bank_config> {
    branches
        .into_iter()
        .reduce(|best, candidate| {
            if candidate.cumulative_llr > best.cumulative_llr {
                candidate
            } else {
                best
            }
        })
        .expect("groups are built from a non-empty input Vec<Branch>, never empty")
}
