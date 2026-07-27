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
    StaleLineagePurgeSummary {
        n_before: usize,
        n_after: usize,
        n_lineages_purged: usize,
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
            StaleLineagePurgeSummary {
                n_before,
                n_after,
                n_lineages_purged,
            } => tracing::debug!(
                target: PruningEvent::TARGET, n_before, n_after, n_lineages_purged, "Stale lineage purge"
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
        .flat_map(|node_branches| {
            // All members of a group share the same `ancestor_creation_step`
            // by construction (`Branch::from_observation`/`from_null` copy
            // the parent's fields unchanged) — read it off any member.
            let anchor_age = node_branches
                .first()
                .map(|b| current_step.saturating_sub(b.ancestor_creation_step))
                .unwrap_or(0);

            if anchor_age >= n_scan {
                let mut survivor = best_by_cumulative_llr(node_branches);
                survivor.ancestor_at_scan_horizon = survivor.branch_id;
                survivor.ancestor_creation_step = current_step;
                vec![survivor]
            } else {
                node_branches
            }
        })
        .collect();

    PruningEvent::NScanSummary {
        n_before,
        n_after: survivors.len(),
    }
    .emit();

    survivors
}

/// Purge every lineage whose freshest branch hasn't consumed a real
/// observation in at least `max_lifetime_nights` nights **and** whose best
/// branch's `cumulative_llr` has fallen to or below `stale_llr_floor` — the
/// whole `lineage_id` is dropped, not just its weakest branches.
///
/// Unlike [`cap_top_b_per_lineage`]/[`apply_n_scan_pruning`], which trim the
/// width of a lineage's branch tree, this can remove an entire lineage.
/// Age alone is not a safe trigger: [`null_branch_llr_delta`](super::llr_score::null_branch_llr_delta)
/// barely penalizes (or, when the lineage falls outside a visit's footprint
/// entirely, doesn't touch at all) a lineage predicted too faint or
/// currently unobservable to expect a detection — exactly the case of a
/// real object going quiet for a long, legitimate stretch (weather, lunar
/// phase, orbital geometry). Requiring the LLR floor too means only a
/// lineage whose own scoring judges it *less plausible than the clutter
/// background* is dropped, regardless of how long it's been stale.
///
/// # Arguments
/// * `branches` – Live branches, already capped by [`cap_top_b_per_lineage`]
///   and [`apply_n_scan_pruning`].
/// * `max_lifetime_nights` – Staleness threshold, in nights. `0` disables
///   this pass entirely (every branch is kept, whatever its age) — the
///   historical behavior, and the default.
/// * `stale_llr_floor` – `cumulative_llr` ceiling a lineage's best branch
///   must be at or below to be purged, once stale. Has no effect while
///   `max_lifetime_nights == 0`.
/// * `current_step` – Current night index.
///
/// # Returns
/// The surviving branches: every branch belonging to a lineage still within
/// its lifetime budget or still LLR-plausible, unchanged.
/// Pure decision predicate for [`purge_stale_lineages`]: is this lineage a
/// stale zombie that should be dropped?
///
/// True iff staleness pruning is enabled (`max_lifetime_nights != 0`), the
/// lineage has coasted at least `max_lifetime_nights` nights without a real
/// update (`age`), **and** its best branch's `best_llr` is not strictly above
/// `stale_llr_floor`.
///
/// The LLR test is written via `partial_cmp` (`!= Some(Greater)`) rather than
/// `best_llr <= stale_llr_floor` on purpose: it treats a **non-finite** score
/// as prunable. A numerically collapsed lineage scores `-inf` (already `<=`
/// any floor) or, when a near-singular innovation covariance turns a `+inf`
/// likelihood delta against a `-inf` null delta, `NaN` — and `NaN <= floor`
/// is `false`, which would let the very worst zombies *escape* the purge.
/// `NaN.partial_cmp(&floor)` is `None`, which is `!= Some(Greater)`, so they
/// are pruned; `+inf` (a genuinely over-confident but finite-evidence
/// lineage) compares `Greater` and is kept.
pub(crate) fn lineage_is_stale(
    age: usize,
    best_llr: f64,
    max_lifetime_nights: usize,
    stale_llr_floor: f64,
) -> bool {
    max_lifetime_nights != 0
        && age >= max_lifetime_nights
        && best_llr.partial_cmp(&stale_llr_floor) != Some(std::cmp::Ordering::Greater)
}

pub fn purge_stale_lineages<'state_lf, 'bank_config>(
    branches: Vec<Branch<'state_lf, 'bank_config>>,
    max_lifetime_nights: usize,
    stale_llr_floor: f64,
    current_step: usize,
) -> Vec<Branch<'state_lf, 'bank_config>> {
    if max_lifetime_nights == 0 {
        return branches;
    }

    let n_before = branches.len();
    let mut by_lineage: HashMap<u64, Vec<Branch<'state_lf, 'bank_config>>> = HashMap::new();
    for branch in branches {
        by_lineage
            .entry(branch.lineage_id)
            .or_default()
            .push(branch);
    }

    let mut n_lineages_purged = 0;
    let survivors: Vec<_> = by_lineage
        .into_values()
        .flat_map(|lineage_branches| {
            let freshest_update = lineage_branches
                .iter()
                .map(|b| b.last_real_update_step)
                .max()
                .unwrap_or(0);
            let age = current_step.saturating_sub(freshest_update);

            let best_llr = lineage_branches
                .iter()
                .map(|b| b.cumulative_llr)
                .fold(f64::NEG_INFINITY, f64::max);

            if lineage_is_stale(age, best_llr, max_lifetime_nights, stale_llr_floor) {
                n_lineages_purged += 1;
                Vec::new()
            } else {
                lineage_branches
            }
        })
        .collect();

    PruningEvent::StaleLineagePurgeSummary {
        n_before,
        n_after: survivors.len(),
        n_lineages_purged,
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

#[cfg(test)]
mod stale_lineage_tests {
    use super::lineage_is_stale;
    use proptest::prelude::*;

    // ── Unit truth table ──────────────────────────────────────────────────
    #[test]
    fn old_low_llr_is_stale() {
        assert!(lineage_is_stale(5, -1.0, 3, 0.0));
    }

    #[test]
    fn age_equal_to_budget_is_stale() {
        // boundary: age == max_lifetime counts as stale.
        assert!(lineage_is_stale(3, -1.0, 3, 0.0));
    }

    #[test]
    fn young_lineage_is_kept_regardless_of_llr() {
        assert!(!lineage_is_stale(2, -1e9, 3, 0.0));
    }

    #[test]
    fn high_llr_is_kept_even_when_old() {
        assert!(!lineage_is_stale(50, 1.0, 3, 0.0));
    }

    #[test]
    fn llr_exactly_at_floor_is_stale() {
        // `!(floor > floor)` == true, so a lineage sitting exactly on the
        // floor is purged (the `<=` boundary).
        assert!(lineage_is_stale(5, 0.0, 3, 0.0));
    }

    #[test]
    fn nan_llr_is_pruned_when_old() {
        // The whole point of `!(best_llr > floor)`: a NaN score (near-singular
        // covariance) must NOT escape the purge the way `NaN <= floor` would.
        assert!(lineage_is_stale(5, f64::NAN, 3, 0.0));
    }

    #[test]
    fn neg_inf_llr_is_pruned_when_old() {
        assert!(lineage_is_stale(5, f64::NEG_INFINITY, 3, 0.0));
    }

    #[test]
    fn pos_inf_llr_is_kept_when_old() {
        // +inf is over-confident but finite-evidence — keep it.
        assert!(!lineage_is_stale(5, f64::INFINITY, 3, 0.0));
    }

    #[test]
    fn disabled_budget_never_prunes() {
        assert!(!lineage_is_stale(1_000, f64::NEG_INFINITY, 0, 0.0));
        assert!(!lineage_is_stale(1_000, f64::NAN, 0, 0.0));
    }

    // ── Property-based ────────────────────────────────────────────────────
    proptest! {
        #[test]
        fn young_lineage_is_never_stale(
            age in 0usize..10_000,
            llr in proptest::num::f64::ANY,
            max in 1usize..10_000,
            floor in -1e6f64..1e6,
        ) {
            prop_assume!(age < max);
            prop_assert!(!lineage_is_stale(age, llr, max, floor));
        }

        #[test]
        fn disabled_is_never_stale(
            age in 0usize..10_000,
            llr in proptest::num::f64::ANY,
            floor in -1e6f64..1e6,
        ) {
            prop_assert!(!lineage_is_stale(age, llr, 0, floor));
        }

        #[test]
        fn staleness_is_monotone_in_age(
            age in 0usize..10_000,
            extra in 0usize..10_000,
            llr in proptest::num::f64::ANY,
            max in 1usize..10_000,
            floor in -1e6f64..1e6,
        ) {
            // Once stale, staying stale as the coasting age only grows.
            if lineage_is_stale(age, llr, max, floor) {
                prop_assert!(lineage_is_stale(age.saturating_add(extra), llr, max, floor));
            }
        }

        #[test]
        fn raising_finite_llr_never_makes_it_more_stale(
            age in 0usize..10_000,
            llr in -1e12f64..1e12,
            bump in 0.0f64..1e12,
            max in 1usize..10_000,
            floor in -1e6f64..1e6,
        ) {
            // A better-supported lineage can only become *less* prunable.
            if !lineage_is_stale(age, llr, max, floor) {
                prop_assert!(!lineage_is_stale(age, llr + bump, max, floor));
            }
        }

        #[test]
        fn non_finite_llr_pruned_when_old_except_pos_inf(
            max in 1usize..1_000,
            floor in -1e6f64..1e6,
            extra in 0usize..1_000,
        ) {
            let age = max + extra; // definitely old
            prop_assert!(lineage_is_stale(age, f64::NAN, max, floor));
            prop_assert!(lineage_is_stale(age, f64::NEG_INFINITY, max, floor));
            prop_assert!(!lineage_is_stale(age, f64::INFINITY, max, floor));
        }
    }
}
