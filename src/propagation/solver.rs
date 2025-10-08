//! Pluggable assignment solvers for bipartite linking.
//!
//! # Overview
//! This module defines a minimal but extensible interface to perform
//! **one-to-one assignments** between two disjoint sets of nodes (“left” and
//! “right”), given a **sparse** list of candidate edges scored by a **cost**
//! (lower is better). The primary entry point is the [`AssignmentSolver`] trait,
//! with a simple, production-ready baseline ([`GreedySolver`]) and a placeholder
//! for an optimal dense solver ([`HungarianSolver`]) you can wire later.
//!
//! ## When to use what
//! - **Greedy (`GreedySolver`)** — *Recommended baseline.*
//!   - ✅ Very fast on **sparse** graphs; no dense matrix required,
//!   - ✅ Time: `O(E log E)`, Memory: `O(E)`,
//!   - ⚠️ Not globally optimal; relies on prior **gating/calibration** of costs.
//! - **Hungarian/JV (`HungarianSolver` stub)** — *Exact on dense LAP.*
//!   - ✅ Produces an optimal solution for the dense Linear Assignment Problem,
//!   - ⚠️ Typically `O(n^3)` and **memory-heavy**; use only on small/medium tiles
//!     or after strong Top-K pruning / partitioning.
//! - **Min-cost flow (future)** — *Best for multi-night layering & constraints.*
//!   - Good for continuity constraints, flow conservation, gaps, etc.
//!
//! ## Units & assumptions
//! - Costs are **non-negative finite floats** (`f64`), “smaller is better”.
//! - Optional diagnostics (e.g., `dt_days`) use **days (TT)** in this project.
//! - Node identifiers are [`SeedId`]s (opaque; only equality/hashing/ordering matter).
//!
//! ## Determinism & tie-breaking
//! - [`GreedySolver`] sorts edges by `cost` using `total_cmp` (IEEE-754 total order).
//!   Equal-cost edges preserve their **input order** (Rust’s `sort_by` is *stable*).
//!   If you want fully deterministic results, ensure the **input edge order** is
//!   deterministic (e.g., build edges in a stable iteration order).
//!
//! ## Validation & panics
//! - Use [`BipartiteProblem::validate`] in debug builds to assert invariants:
//!   unique node lists, finite & non-negative costs. The greedy solver does **not**
//!   re-validate at runtime; feeding NaNs or negatives leads to unspecified behavior.
//!
//! ## See also
//! - [`crate::propagation::features`] — seed feature engineering (IDs, times).
//! - Future: min-cost flow layer for multi-night track building.

use crate::propagation::features::SeedId;

/* ----------------------------- Edge model ----------------------------- */

/// A **scored candidate** connecting a left node to a right node.
///
/// The solver backends consume a flat list of `Edge`s instead of a dense matrix,
/// enabling scalable, partition-friendly pipelines.
///
/// # Invariants
/// - `cost` **must be** finite and `>= 0.0`.
/// - `from` belongs to [`BipartiteProblem::left`], `to` to `right` (not enforced).
///
/// # Fields
/// - `from` — Source seed in the **left** partition.
/// - `to` — Target seed in the **right** partition.
/// - `cost` — Non-negative edge cost; **smaller is better**.
/// - `dt_days` — Optional diagnostic (e.g., time gap in **days, TT**).
#[derive(Clone, Debug)]
pub struct Edge {
    /// Source seed on the left partition.
    pub from: SeedId,
    /// Target seed on the right partition.
    pub to: SeedId,
    /// Non-negative cost; smaller is better.
    pub cost: f64,
    /// Optional diagnostic: time gap (days).
    pub dt_days: f64,
}

/* --------------------------- Assignment model ------------------------- */

/// A **one-to-one** match produced by a solver.
///
/// Every left/right node appears **at most once** across all [`Assignment`]s.
///
/// # Fields
/// - `from` — Matched **left** seed id.
/// - `to` — Matched **right** seed id.
/// - `cost` — Edge cost associated with this match (as used by the solver).
#[derive(Clone, Debug)]
pub struct Assignment {
    /// Left seed id.
    pub from: SeedId,
    /// Right seed id.
    pub to: SeedId,
    /// Edge cost used by the solver.
    pub cost: f64,
}

/* --------------------------- Problem container ------------------------ */

/// Input container for bipartite assignment.
///
/// Provide all unique node IDs for both partitions plus a **sparse** list of
/// candidate edges. Solvers may assume the node lists are unique (no duplicates).
///
/// # Example
/// See the **module-level** example for a full usage snippet.
///
/// # Notes
/// - This container does **not** build any adjacency matrix; memory scales with `E`.
#[derive(Clone, Debug)]
pub struct BipartiteProblem {
    /// All left node ids (unique).
    pub left: Vec<SeedId>,
    /// All right node ids (unique).
    pub right: Vec<SeedId>,
    /// Candidate edges (sparse).
    pub edges: Vec<Edge>,
}

impl BipartiteProblem {
    /// Debug-only sanity checks for common invariants.
    ///
    /// This helper:
    /// - asserts that `left` and `right` contain **unique** IDs,
    /// - asserts that **every** edge has a **finite, non-negative** `cost`.
    ///
    /// # Panics
    /// Panics in **debug builds** if any invariant is violated. No-op in release builds.
    pub fn validate(&self) {
        debug_assert!(is_unique(&self.left));
        debug_assert!(is_unique(&self.right));
        debug_assert!(self
            .edges
            .iter()
            .all(|e| e.cost.is_finite() && e.cost >= 0.0));
    }
}

/// Return `true` if all elements are **pairwise distinct** (by ordering and equality).
fn is_unique<T: Ord + Copy>(xs: &[T]) -> bool {
    let mut v = xs.to_vec();
    v.sort(); // stable sort; equal items remain adjacent
    v.windows(2).all(|w| w[0] != w[1])
}

/* ----------------------------- Solver trait --------------------------- */

/// Generic interface for **bipartite one-to-one** assignment solvers.
///
/// Implementors produce a set of **disjoint** pairs `(from, to)` intended to
/// minimize the total cost (exactly or approximately).
///
/// # Contract
/// - **Input**: a sparse set of edges; costs are finite and `>= 0.0`.
/// - **Output**: each `from` and `to` appears **at most once** across results.
/// - **Tie-breaking**: may be arbitrary unless specified otherwise.
/// - **Partial coverage**: solvers may leave nodes unmatched.
/// - **Complexity**: depends on the backend; see each implementation.
///
/// # Determinism
/// Determinism is backend-dependent. [`GreedySolver`] is deterministic *iff* the
/// input edge order is deterministic (equal-cost edges keep input order).
pub trait AssignmentSolver {
    /// Solve a one-to-one assignment on the given sparse candidate set.
    ///
    /// Implementations should return a set of **disjoint** pairs `(from, to)`
    /// minimizing the total cost (exactly or approximately). If multiple edges
    /// have the same cost, tie-breaking can be arbitrary.
    fn solve(&self, pb: &BipartiteProblem) -> Vec<Assignment>;
}

/* ------------------------- Greedy baseline ------------------------- */

/// Greedy solver: sort edges by increasing cost and pick if both endpoints are free.
///
/// # Algorithm
/// 1. Copy and **stable-sort** edges by `cost` (`total_cmp`),
/// 2. Maintain two hash sets for already-matched left/right nodes,
/// 3. Scan edges in order; **accept** an edge iff both endpoints are unused.
///
/// # Complexity
/// - Time: `O(E log E)` due to sorting.
/// - Memory: `O(E)` for the edge copy + `O(|L| + |R|)` for the hash sets.
///
/// # Optimality
/// - Not globally optimal: it may miss better global combinations.
/// - Works well when prior **gating** yields few **confusable** candidates.
///
/// # Determinism
/// - Deterministic given deterministic input edge order.
/// - Equal-cost edges preserve their input order (stable sort).
///
/// # See also
/// - [`HungarianSolver`] for an optimal dense LAP placeholder.
///
/// # Examples
/// See the module-level example.
#[derive(Default, Debug)]
pub struct GreedySolver;

impl AssignmentSolver for GreedySolver {
    fn solve(&self, pb: &BipartiteProblem) -> Vec<Assignment> {
        use ahash::AHashSet;

        // 1) Stable sort by increasing cost (IEEE-754 total order).
        let mut edges = pb.edges.clone();
        edges.sort_by(|a, b| a.cost.total_cmp(&b.cost));

        // 2) Track already-matched endpoints.
        let mut used_left: AHashSet<SeedId> = AHashSet::with_capacity(pb.left.len());
        let mut used_right: AHashSet<SeedId> = AHashSet::with_capacity(pb.right.len());

        // 3) Greedy scan.
        let mut out = Vec::with_capacity(edges.len().min(pb.left.len().min(pb.right.len())));

        for e in edges {
            if !used_left.contains(&e.from) && !used_right.contains(&e.to) {
                used_left.insert(e.from);
                used_right.insert(e.to);
                out.push(Assignment {
                    from: e.from,
                    to: e.to,
                    cost: e.cost,
                });
            }
        }
        out
    }
}

/* -------------------------- Hungarian stub ------------------------- */

/// Placeholder for an **optimal** dense LAP solver (Hungarian/JV).
///
/// # Notes
/// Real-world Hungarian/JV implementations operate on a **dense cost matrix**
/// with one cost per `(left, right)` pair. If your edge set is **sparse**, you
/// typically:
/// - **densify** with a large “no-edge” penalty (memory heavy),
/// - or use a solver that supports **sparse adjacency** (e.g., auction variants),
/// - or **tile/partition** the problem (e.g., spatially) so matrices remain small.
///
/// This stub currently **falls back** to [`GreedySolver`], providing a safe,
/// fast baseline until an optimal backend is integrated.
///
/// # Complexity
/// Currently the same as [`GreedySolver`], due to the fallback.
///
/// # TODO
/// - Integrate a dense LAP implementation (Hungarian, Jonker-Volgenant),
/// - Provide a sparse-friendly variant (auction, successive shortest augmenting path),
/// - Add configurable “no-edge” penalties and tiling utilities.
#[derive(Default, Debug)]
pub struct HungarianSolver;

impl AssignmentSolver for HungarianSolver {
    fn solve(&self, pb: &BipartiteProblem) -> Vec<Assignment> {
        // TODO: implement or bridge to a crate providing LAP/JV/Hungarian.
        // For now, defer to the greedy baseline as a safe fallback.
        GreedySolver.solve(pb)
    }
}
