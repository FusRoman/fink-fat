//! End-to-end **bipartite linking engine** (candidate retrieval → scoring → assignment).
//!
//! # Overview
//! This module orchestrates steps **(2) candidate retrieval** and **(3) bipartite
//! linking** for inter-night tracking. Given two nights `N_left → N_right`, it:
//!
//! 1. Builds a **spatial index** over right-night seeds,
//! 2. For each left seed `i`, predicts a **cone** at the target epoch,
//! 3. Retrieves **candidate** right seeds via cell-based cone coverage,
//! 4. Computes **scores** with **hard gates** and keeps **Top-K** per left seed,
//! 5. Solves a **one-to-one assignment** on the sparse bipartite graph (pluggable solver),
//! 6. Optionally **stitches** pairwise assignments across nights into multi-night tracks.
//!
//! The solver backend is abstracted by [`AssignmentSolver`], enabling a drop-in
//! switch from the fast greedy baseline to an optimal **Hungarian/JV** solver, or
//! later to a global **min-cost flow**—without changing pipeline code.
//!
//! ## Design goals
//! - **Sparse-first**: use geometric prediction (cones) + HEALPix-like coverage to
//!   generate **few** plausible candidates per left seed,
//! - **Interpretable scoring**: see [`crate::propagation::scoring`], all additive and gated,
//! - **Pluggable assignment**: see [`crate::propagation::solver`], choose Greedy/Hungarian,
//! - **Composable**: link by **night pairs** and then **stitch**.
//!
//! ## Scale & performance
//! - Candidate retrieval and scoring are **sparse**,
//! - Keep **Top-K** per left before solving to bound `E` (edges) and memory,
//! - Solve per **night pair** to keep problems small; then **stitch** chains,
//! - Complexity (typical):
//!   - Retrieval: ~O(L · log R) amortized with spatial index,
//!   - Scoring: ~O(E), after Top-K pruning,
//!   - Greedy assignment: O(E log E); Hungarian (dense LAP): ~O(n³).
//!
//! ## Units & conventions
//! - Angles: **radians**,
//! - Times: **days** (MJD TT),
//! - Costs: **dimensionless**, non-negative (lower is better).
//!
//! ## Determinism
//! Determinism holds if inputs (seed lists, spatial binner iteration order) and
//! configuration are deterministic. Greedy assignment preserves a stable ordering
//! under equal costs (stable sort). Logging via `println!` is for **debug** only.
//!
//! ## Failure modes
//! - Overly large cones or missing Top-K can blow up `E` (memory/latency),
//! - Too-lax gates in scoring will increase contamination and solver load,
//! - If `N_right - N_left < 1`, we clamp to `Δ=1` revisits for scoring purposes.
//!
//! ## See also
//! - [`crate::propagation::scoring`] — edge scoring, gates/weights/scales,
//! - [`crate::propagation::solver`]  — bipartite solvers and interfaces,
//! - [`crate::seeding::space_time_bucket`] — binner traits (HEALPix-backed).

use ahash::AHashMap;

use crate::{
    params::engine_params::InterNightLinkConfig,
    propagation::{
        features::{SeedId, SeedNode, SeedSpatialIndex},
        scoring::ScoredEdge,
        solver::{Assignment, AssignmentSolver, BipartiteProblem, Edge},
    },
    seeding::space_time_bucket::SpatialBinner,
    NightId,
};

/* -------------------------- Top-K Edge Generation ------------------------- */

/// Génère les **edges scorés Top-K** de `left` vers `right`, sans résoudre.
/// Respecte `cfg.limits` (Top-K, max_cost, max_total_edges).
pub fn generate_topk_edges_between<B: SpatialBinner>(
    left: &[SeedNode],
    right: &[SeedNode],
    cfg: &InterNightLinkConfig,
    binner: &B,
    right_id_to_index: &AHashMap<SeedId, usize>,
) -> Vec<Edge> {
    let night_left = left.first().map(|s| s.night_id).unwrap_or(0);
    let night_right = right.first().map(|s| s.night_id).unwrap_or(0);

    // Δ revisits (≥ 1)
    let delta_revisit: u32 = night_right.saturating_sub(night_left).max(1);

    // 0) index spatial côté "right"
    let index_right = SeedSpatialIndex::build(right, binner);

    // 1) génération/scoring Top-K
    let mut edges: Vec<Edge> = Vec::with_capacity(left.len() * cfg.limits.top_k_per_left);
    let t_right_med = median_epoch(right);

    for i in left {
        // (a) cône au temps médian (couverture) -> ids candidats
        let (ra_c, dec_c, r_c) = i.predict_cone(t_right_med, binner, &cfg.predict);
        let cand_iter = SeedSpatialIndex::cone_query(&index_right, binner, ra_c, dec_c, r_c);

        // (b) score fin seed-par-seed au vrai epoch de j
        let mut scored: Vec<ScoredEdge> = Vec::new();
        for j_id in cand_iter {
            if let Some(&j_idx) = right_id_to_index.get(&j_id) {
                let j = &right[j_idx];
                if let Some(se) = ScoredEdge::score(i, j, cfg, delta_revisit) {
                    if let Some(cmax) = cfg.limits.max_cost {
                        if se.cost > cmax {
                            continue;
                        }
                    }
                    scored.push(se);
                }
            }
        }

        // (c) Top-K par coût croissant
        scored.sort_by(|a, b| a.cost.total_cmp(&b.cost));
        scored.truncate(cfg.limits.top_k_per_left);

        // (d) conversion -> Edge nu pour MCF
        edges.extend(scored.into_iter().map(|se| Edge {
            from: se.from,
            to: se.to,
            cost: se.cost,
            dt_days: se.dt_days,
        }));
    }

    // 2) Cap global optionnel
    if let Some(max_e) = cfg.limits.max_total_edges {
        if edges.len() > max_e {
            edges.sort_by(|a, b| a.cost.total_cmp(&b.cost));
            edges.truncate(max_e);
        }
    }

    edges
}

/* ------------------------------- Results --------------------------------- */

/// Result of linking a single night pair (`left → right`).
///
/// The assignment is **one-to-one** (subset of nodes). Diagnostics expose the
/// final number of edges **kept** after Top-K and global truncation.
#[derive(Clone, Debug)]
pub struct LinkResult {
    /// Night ids (source → target).
    pub night_left: NightId,
    pub night_right: NightId,
    /// Final **one-to-one** assignments returned by the solver.
    pub matches: Vec<Assignment>,
    /// Diagnostics: number of edges **kept** in the final problem.
    pub edges_kept: usize,
}

/* ------------------------------- Linking --------------------------------- */

/// Link a pair of nights with an explicit **spatial binner** and a right-side
/// `id → index` lookup map.
///
/// This avoids globals/thread-locals and keeps the engine deterministic and testable.
///
/// # Pipeline
/// - Build a spatial index for the **right** seeds,
/// - For each **left** seed:
///   - predict a **cone** (center + radius at median right epoch),
///   - query candidate **ids** with the binner,
///   - **score** each candidate at its own epoch with hard gates,
///   - keep **Top-K** by cost (and drop by `max_cost` if configured),
/// - Concatenate edges, optionally **cap** the global edge list,
/// - Build the [`BipartiteProblem`] and **solve** with the provided solver.
///
/// # Complexity (typical)
/// - Spatial index build: ~O(R),
/// - Cone queries: ~O(log R) amortized per left seed,
/// - Scoring: O(E), with E controlled by `top_k_per_left` (≈ L·K),
/// - Greedy assign: O(E log E) (Hungarian ~O(n³) dense).
///
/// # Determinism
/// Deterministic given deterministic inputs and binner iteration order.
/// Equal-cost ties are stable under Rust’s stable sort.
///
/// # Returns
/// - A [`LinkResult`] with matches and edge diagnostics.
///
/// # Panics
/// - Debug-only: [`BipartiteProblem::validate`] may assert invariants during development.
pub fn link_pair_with_binner<S, B>(
    left: &[SeedNode],
    right: &[SeedNode],
    cfg: &InterNightLinkConfig,
    solver: &S,
    binner: &B,
    right_id_to_index: &AHashMap<SeedId, usize>,
) -> LinkResult
where
    S: AssignmentSolver,
    B: crate::seeding::space_time_bucket::SpatialBinner,
{
    let night_left = left.first().map(|s| s.night_id).unwrap_or(0);
    let night_right = right.first().map(|s| s.night_id).unwrap_or(0);
    let edges = generate_topk_edges_between(left, right, cfg, binner, right_id_to_index);

    // Build problem and solve
    let left_ids: Vec<SeedId> = left.iter().map(|s| s.seed_id).collect();
    let right_ids: Vec<SeedId> = right.iter().map(|s| s.seed_id).collect();
    let pb = BipartiteProblem {
        left: left_ids,
        right: right_ids,
        edges,
    };
    pb.validate();

    println!(
        "Solving assignment with {} left, {} right, {} edges",
        pb.left.len(),
        pb.right.len(),
        pb.edges.len()
    );

    let matches = solver.solve(&pb);

    println!("Found {} matches", matches.len());

    LinkResult {
        night_left,
        night_right,
        matches,
        edges_kept: pb.edges.len(),
    }
}

/* ------------------------------- Internals ------------------------------- */

/// Compute the **median** epoch of a slice of seeds (robust to outliers).
///
/// Returns `0.0` for an empty slice.
fn median_epoch(seeds: &[SeedNode]) -> f64 {
    if seeds.is_empty() {
        return 0.0;
    }
    let mut v: Vec<f64> = seeds.iter().map(|s| s.epoch_mid).collect();
    let mid = v.len() / 2;
    v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    v[mid]
}

/// Build a right-side lookup map (seed_id → index).
///
/// This is an O(R) pre-pass that makes candidate dereferencing O(1).
pub fn build_id_to_index(seeds: &[SeedNode]) -> AHashMap<SeedId, usize> {
    let mut m = AHashMap::with_capacity(seeds.len());
    for (k, s) in seeds.iter().enumerate() {
        m.insert(s.seed_id, k);
    }
    m
}
