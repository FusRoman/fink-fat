//! K-best (approximate) global solutions for the min-cost-flow assignment.
//!
//! # Overview
//! The min-cost-flow reduction yields a *single* optimal global solution
//! (minimum-cost assignment with break penalties). In ambiguous components,
//! returning only the optimum can be too brittle for downstream orbit
//! determination (IOD), because several near-optimal path covers may be
//! plausible.
//!
//! This module implements a **simple, deterministic K-solution strategy**:
//! 1. Compute the best solution (K=1).
//! 2. For `k > 1`, generate alternatives by **forbidding exactly one selected arc**
//!    from the best solution ("one-edge deviation"), and re-solving the assignment.
//! 3. Deduplicate solutions using a stable signature and keep the best `k`.
//!
//! This is not a full K-best assignment algorithm (e.g. Murty), but it is:
//! - easy to reason about,
//! - deterministic,
//! - inexpensive when the best solution has a moderate number of arcs,
//! - well-suited to producing clean, disjoint-chain hypotheses for IOD.
//!
//! # Determinism guarantees
//! Alternative attempts are generated in a stable order by sorting arcs selected
//! in the best solution by:
//! - increasing arc cost,
//! - then `(local_from, local_to)`.
//!
//! If the underlying min-cost-flow solver is deterministic, the full output set
//! will be deterministic as well.
//!
//! # Tuning knobs
//! - `k` controls the number of *unique* solutions to return (clamped by feasibility).
//! - `max_alt_attempts` caps re-solves to avoid worst-case behavior when the best
//!   solution selects many arcs.
//!
//! See also
//! --------
//! - `assignment::solve_once` for the underlying single-solve routine.
//! - `solution::Solution` for the in-memory representation and signature semantics.

use std::cmp::Ordering;

use ahash::{AHashMap, AHashSet};

use crate::graph::edge_id::EdgeId;

use super::{assignment::solve_once, solution::Solution};

/// Compute up to `k` unique global solutions (best-first by objective).
///
/// Parameters
/// ----------
/// n_nodes : usize
///     Number of nodes in the component (dense local indexing).
/// candidate_arcs : &AHashMap<(usize, usize), (EdgeId, f64)>
///     Candidate directed arcs `(u, v)` with backing edge id and scalar cost.
///     Keys and indices are local in `0..n_nodes`.
/// break_penalty : f64
///     Penalty for missing predecessor/successor (break edges in the assignment).
/// k : usize
///     Requested number of solutions. Values `< 1` are treated as `1`.
/// max_alt_attempts : usize
///     Maximum number of alternative re-solves. Each attempt forbids one arc
///     from the best solution, so the theoretical maximum number of attempts is
///     the number of selected arcs in the best solution, but we clamp it here.
///
/// Returns
/// -------
/// Vec<Solution>
///     A list of unique solutions sorted by increasing `total_cost` and then by
///     `signature` for tie-breaking. The vector can contain fewer than `k`
///     elements if:
///     - alternative instances are infeasible,
///     - or all feasible alternatives collapse to already-seen signatures.
///
/// Strategy
/// --------
/// - Always compute the best solution.
/// - If `k > 1`, generate alternatives by forbidding one selected arc from the
///   best solution (one-edge deviation).
/// - Deduplicate by `Solution.signature`.
///
/// Notes
/// -----
/// - This is a pragmatic approximation of K-best assignment and does not explore
///   the full solution space.
/// - The quality of alternatives depends on the structure of the assignment; in
///   particular, forbidding one low-cost arc may force a large structural change
///   (useful) or may be absorbed by a cheap rerouting (also useful).
/// - If you later need a stronger guarantee (true K-best), consider implementing
///   Murty’s algorithm or Eppstein’s K-shortest augmenting paths on the residual.
///   The current interface is compatible with such upgrades.
pub(super) fn compute_k_solutions(
    n_nodes: usize,
    candidate_arcs: &AHashMap<(usize, usize), (EdgeId, f64)>,
    break_penalty: f64,
    k: usize,
    max_alt_attempts: usize,
) -> Vec<Solution> {
    // Always compute the best (unconstrained) solution.
    let Some(best_solution) = solve_once(n_nodes, candidate_arcs, break_penalty, &AHashSet::new())
    else {
        return Vec::new();
    };

    let mut solutions = vec![best_solution];

    // Clamp: K=1 => best solution only.
    if k <= 1 {
        return solutions;
    }

    // Deterministic attempt ordering:
    // - increasing arc cost, then
    // - (local_from, local_to).
    let mut forbidden_candidates = solutions[0].selected_arcs.clone();
    forbidden_candidates.sort_by(|lhs, rhs| {
        lhs.cost
            .partial_cmp(&rhs.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| lhs.local_from.cmp(&rhs.local_from))
            .then_with(|| lhs.local_to.cmp(&rhs.local_to))
    });

    // Re-solve by forbidding one selected arc at a time (bounded).
    for selected_arc in forbidden_candidates.into_iter().take(max_alt_attempts) {
        if solutions.len() >= k {
            break;
        }

        // Forbid exactly one arc from the best solution.
        let forbidden_arcs: AHashSet<(usize, usize)> =
            [(selected_arc.local_from, selected_arc.local_to)]
                .into_iter()
                .collect();

        if let Some(candidate_solution) =
            solve_once(n_nodes, candidate_arcs, break_penalty, &forbidden_arcs)
        {
            // Deduplicate by stable signature.
            let is_unique = !solutions
                .iter()
                .any(|existing| existing.signature == candidate_solution.signature);

            if is_unique {
                solutions.push(candidate_solution);
            }
        }
    }

    // Best-first ordering (ties broken deterministically by signature).
    solutions.sort_by(|lhs, rhs| {
        lhs.total_cost
            .partial_cmp(&rhs.total_cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| lhs.signature.cmp(&rhs.signature))
    });
    solutions.truncate(k);

    solutions
}
