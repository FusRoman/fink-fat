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

/// Generate **Top-K scored** edges from `left` to `right` using a prebuilt index.
/// Respects `cfg.limits` (Top-K, max_cost, max_total_edges).
pub fn generate_topk_edges<B: SpatialBinner>(
    left: &[SeedNode],
    right: &[SeedNode],
    cfg: &InterNightLinkConfig,
    binner: &B,
    index_right: &SeedSpatialIndex, // NEW: pass prebuilt index
    t_right_med: f64,               // NEW: pass precomputed median epoch
    right_id_to_index: &AHashMap<SeedId, usize>,
) -> Vec<Edge> {
    // Δ revisits (≥ 1)
    let night_left = left.first().map(|s| s.night_id).unwrap_or(0);
    let night_right = right.first().map(|s| s.night_id).unwrap_or(0);
    let delta_revisit: u32 = night_right.saturating_sub(night_left).max(1);

    // 1) generation/scoring Top-K
    let mut edges: Vec<Edge> = Vec::with_capacity(left.len() * cfg.limits.top_k_per_left);

    for i in left {
        // (a) coarse cone at median time → candidate ids
        let (ra_c, dec_c, r_c) = i.predict_cone(t_right_med, binner, &cfg.predict);
        let cand_iter = SeedSpatialIndex::cone_query(index_right, binner, ra_c, dec_c, r_c);

        // (b) fine score per candidate at its true epoch; early gate & cmax
        let mut scored: Vec<ScoredEdge> = Vec::with_capacity(32);

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

        // (c) Top-K via partial selection instead of full sort
        let k = cfg.limits.top_k_per_left.min(scored.len());
        if k > 0 {
            // partition so that the k best are in [0..k] in arbitrary order (O(n))
            let (_, _, _) = scored.select_nth_unstable_by(k - 1, |a, b| a.cost.total_cmp(&b.cost));
            // ensure deterministic ascending order within the Top-K bucket (small sort)
            scored[..k].sort_by(|a, b| a.cost.total_cmp(&b.cost));
            scored.truncate(k);

            // (d) convert to bare Edge
            edges.extend(scored.into_iter().map(|se| Edge {
                from: se.from,
                to: se.to,
                cost: se.cost,
                dt_days: se.dt_days,
            }));
        }
    }

    // 2) Optional global cap using partial selection instead of full sort
    if let Some(max_e) = cfg.limits.max_total_edges {
        if edges.len() > max_e {
            let (_, _, _) =
                edges.select_nth_unstable_by(max_e - 1, |a, b| a.cost.total_cmp(&b.cost));
            // sort only the kept prefix to stabilize order for determinism
            edges[..max_e].sort_by(|a, b| a.cost.total_cmp(&b.cost));
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

    // PRECOMPUTE once for the current right-night
    let index_right = SeedSpatialIndex::build(right, binner); // moved out of the loop
    let t_right_med = median_epoch(right);
    let edges = generate_topk_edges(
        left,
        right,
        cfg,
        binner,
        &index_right,
        t_right_med,
        right_id_to_index,
    );

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
pub fn median_epoch(seeds: &[SeedNode]) -> f64 {
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

// Place this in `src/propagation/engine.rs` (at the end of the file)
// or in a dedicated integration test (then adjust `use` paths accordingly).

#[cfg(test)]
mod engine_tests {
    use super::{build_id_to_index, generate_topk_edges};
    use ahash::AHashMap;

    use crate::{
        params::engine_params::InterNightLinkConfigBuilder,
        propagation::{
            engine::median_epoch,
            features::{SeedNode, SeedSpatialIndex},
        },
        seeding::healpix_binners::HealpixBinner,
        NightId,
    };

    // --- Test helpers -------------------------------------------------------

    /// Build a permissive InterNightLinkConfig so that scoring doesn't reject most edges.
    ///
    /// We relax position gates (`max_d2_pos` large), inflate cones a bit, and allow
    /// generous Top-K. Each individual test will override specific limits (Top-K, max_cost, global caps).
    fn permissive_cfg() -> crate::params::engine_params::InterNightLinkConfig {
        InterNightLinkConfigBuilder::new()
            // Predictor: conservative-ish cones, plus padding for HEALPix boundaries.
            .with_predict(|p| p.k_sigma(3.5).pad_cell_radius(true))
            // Scoring: very permissive positional gates (huge threshold),
            // weights/scales left as defaults (builder will validate).
            .with_scoring(|s| s.set_max_d2_pos(1.0e9))
            // Limits: overriden per-test; set generous defaults here.
            .with_limits(|l| l.top_k_per_left(64).max_cost(None).max_total_edges(None))
            .build()
            .expect("valid permissive InterNightLinkConfig")
    }

    /// Minimal deterministic seed fabric.
    ///
    /// This constructs a SeedNode whose tangent plane is centered at `(center_ra, center_dec)`,
    /// with a simple kinematics model. We keep epochs close so that the predicted cone
    /// overlaps the counterpart on the right night under permissive gating.
    #[allow(clippy::too_many_arguments)]
    fn make_seed(
        seed_id: u64,
        night_id: NightId,
        epoch_mid: f64,
        center_ra: f64,
        center_dec: f64,
        pos_xy: [f64; 2],
        vel_xy: [f64; 2],
        band: u8,
    ) -> SeedNode {
        SeedNode::new(
            seed_id,
            night_id,
            epoch_mid,
            pos_xy,
            vel_xy,
            // unit-ish covariances (diagonal) to avoid degenerate scoring
            [[1e-12, 0.0], [0.0, 1e-12]],
            [[1e-12, 0.0], [0.0, 1e-12]],
            None,
            1000.0,
            10.0,
            band,
            2,
            Vec::new(), // not used in these tests
            center_ra,
            center_dec,
            center_ra,
            center_dec,
        )
    }

    /// Build a compact left/right scenario near the equator, with small offsets so
    /// that cones overlap for the median epoch. Right seeds are **slightly shifted**
    /// to ensure strictly-positive residuals/costs (prevents zero-cost edges).
    fn build_toy_left_right(
        n_left: usize,
        n_right: usize,
        night_left: NightId,
        night_right: NightId,
        epoch_left: f64,
        epoch_right: f64,
    ) -> (Vec<SeedNode>, Vec<SeedNode>) {
        let mut left = Vec::with_capacity(n_left);
        let mut right = Vec::with_capacity(n_right);

        // Base sky center (radians), tiny perturbations per seed
        let base_ra = 0.10_f64;
        let base_dec = 0.00_f64;

        // Small fixed nudge applied only to the RIGHT seeds.
        // ~ few arcseconds in tangent plane units is more than enough, but we keep it small.
        let delta = 3e-5_f64;

        for k in 0..n_left {
            let eps = 1e-4 * (k as f64);
            left.push(make_seed(
                k as u64,
                night_left,
                epoch_left,
                base_ra,
                base_dec,
                [eps, 0.0],
                [0.0, 0.0],
                0,
            ));
        }

        for k in 0..n_right {
            // Keep ordering monotonic by index, but apply a tiny delta so no perfect match.
            let eps = 1e-4 * (k as f64) + delta;
            right.push(make_seed(
                10_000 + k as u64,
                night_right,
                epoch_right,
                base_ra,
                base_dec,
                [eps, 0.0],
                [0.0, 0.0],
                0,
            ));
        }

        (left, right)
    }

    // --- Unit tests ---------------------------------------------------------

    #[test]
    fn topk_empty_inputs_produce_no_edges() {
        let cfg = permissive_cfg();
        let binner = HealpixBinner::new(6);

        let left: Vec<SeedNode> = vec![];
        let right: Vec<SeedNode> = vec![];

        let idmap: AHashMap<_, _> = build_id_to_index(&right);

        let index_right = SeedSpatialIndex::build(&right, &binner); // moved out of the loop
        let t_right_med = median_epoch(&right);
        let edges = generate_topk_edges(
            &left,
            &right,
            &cfg,
            &binner,
            &index_right,
            t_right_med,
            &idmap,
        );
        assert!(edges.is_empty());

        // Also empty right only
        let (left, _right) = build_toy_left_right(5, 0, 1, 2, 60000.0, 60001.0);
        let idmap: AHashMap<_, _> = AHashMap::new();

        let index_right = SeedSpatialIndex::build(&_right, &binner); // moved out of the loop
        let t_right_med = median_epoch(&_right);

        let edges =
            generate_topk_edges(&left, &[], &cfg, &binner, &index_right, t_right_med, &idmap);
        assert!(edges.is_empty());
    }

    #[test]
    fn enforces_top_k_per_left() {
        // We want many candidates per left → set K=2 but with 1 left and 5 right → expect 2 edges.
        let cfg = InterNightLinkConfigBuilder::new()
            .with_predict(|p| p.k_sigma(3.5).pad_cell_radius(true))
            .with_scoring(|s| s.set_max_d2_pos(1.0e9))
            .with_limits(|l| l.top_k_per_left(2).max_cost(None).max_total_edges(None))
            .build()
            .unwrap();

        let binner = HealpixBinner::new(7);
        let (left, right) = build_toy_left_right(1, 5, 10, 11, 60000.0, 60001.0);
        let idmap = build_id_to_index(&right);

        let index_right = SeedSpatialIndex::build(&right, &binner); // moved out of the loop
        let t_right_med = median_epoch(&right);

        let edges = generate_topk_edges(
            &left,
            &right,
            &cfg,
            &binner,
            &index_right,
            t_right_med,
            &idmap,
        );

        // Not more than K * |left|
        assert!(edges.len() <= 2);
        // Exactly 2 if scoring admitted enough (permissive gates → yes)
        assert_eq!(edges.len(), 2);

        // All edges originate from the single left seed
        let froms: std::collections::BTreeSet<_> = edges.iter().map(|e| e.from).collect();
        assert_eq!(froms.len(), 1);
    }

    #[test]
    fn enforces_max_cost_cutoff() {
        // ----------------------------------------------------------------------
        // Step 1. Generate edges without any max_cost cutoff.
        // ----------------------------------------------------------------------
        // We first run the edge generation with a permissive configuration
        // (no max_cost) to inspect the natural cost distribution and detect
        // if zero-cost edges exist. This helps us check that a stricter cutoff
        // behaves as expected regardless of the actual scale of costs.
        let cfg_loose = InterNightLinkConfigBuilder::new()
            .with_predict(|p| p.k_sigma(3.5).pad_cell_radius(true))
            .with_scoring(|s| s.set_max_d2_pos(1.0e9))
            .with_limits(|l| l.top_k_per_left(64).max_cost(None).max_total_edges(None))
            .build()
            .unwrap();

        let binner = HealpixBinner::new(6);
        let (left, right) = build_toy_left_right(4, 4, 2, 3, 60000.0, 60000.5);
        let idmap = build_id_to_index(&right);

        let index_right = SeedSpatialIndex::build(&right, &binner); // moved out of the loop
        let t_right_med = median_epoch(&right);

        let edges_unbounded = generate_topk_edges(
            &left,
            &right,
            &cfg_loose,
            &binner,
            &index_right,
            t_right_med,
            &idmap,
        );

        // Count how many zero-cost edges exist and whether we had any strictly positive costs.
        let eps = f64::EPSILON;
        let zero_cost_count = edges_unbounded
            .iter()
            .filter(|e| e.cost.abs() <= eps)
            .count();
        let had_positive = edges_unbounded.iter().any(|e| e.cost > 0.0);

        // ----------------------------------------------------------------------
        // Step 2. Apply a strict cutoff: max_cost = 0.0
        // ----------------------------------------------------------------------
        // With this configuration, only edges whose cost == 0 (within numerical
        // precision) should remain. All edges with cost > 0 must be discarded.
        let cfg_cut = InterNightLinkConfigBuilder::new()
            .with_predict(|p| p.k_sigma(3.5).pad_cell_radius(true))
            .with_scoring(|s| s.set_max_d2_pos(1.0e9))
            .with_limits(|l| {
                l.top_k_per_left(64)
                    .max_cost(Some(0.0))
                    .max_total_edges(None)
            })
            .build()
            .unwrap();

        let index_right = SeedSpatialIndex::build(&right, &binner); // moved out of the loop
        let t_right_med = median_epoch(&right);

        let edges_cut = generate_topk_edges(
            &left,
            &right,
            &cfg_cut,
            &binner,
            &index_right,
            t_right_med,
            &idmap,
        );

        // ----------------------------------------------------------------------
        // Step 3. Assertions
        // ----------------------------------------------------------------------

        // (a) All remaining edges must have cost == 0 (within floating tolerance).
        assert!(
            edges_cut.iter().all(|e| e.cost.abs() <= eps),
            "With max_cost=0.0, all kept edges must have cost == 0 (±eps). Got: {:?}",
            edges_cut
        );

        // (b) If we originally had edges with cost > 0, some must have been filtered out.
        if had_positive {
            assert!(
                edges_cut.len() < edges_unbounded.len(),
                "max_cost=0.0 should filter out strictly positive-cost edges"
            );
        }

        // (c) If no zero-cost edges existed before, the cutoff should yield an empty set.
        if zero_cost_count == 0 {
            assert!(
                edges_cut.is_empty(),
                "No zero-cost edges existed, so max_cost=0.0 must yield an empty set"
            );
        }
    }

    #[test]
    fn enforces_global_cap_and_sorts_by_cost() {
        // With 3 left × TopK=3 we could get up to 9 edges; cap globally to 4
        // and ensure truncation happens on globally-lowest costs.
        let cfg = InterNightLinkConfigBuilder::new()
            .with_predict(|p| p.k_sigma(3.5).pad_cell_radius(true))
            .with_scoring(|s| s.set_max_d2_pos(1.0e9))
            .with_limits(|l| l.top_k_per_left(3).max_cost(None).max_total_edges(Some(4)))
            .build()
            .unwrap();

        let binner = HealpixBinner::new(7);
        let (left, right) = build_toy_left_right(3, 10, 12, 13, 60000.0, 60001.0);
        let idmap = build_id_to_index(&right);

        let index_right = SeedSpatialIndex::build(&right, &binner); // moved out of the loop
        let t_right_med = median_epoch(&right);

        let edges = generate_topk_edges(
            &left,
            &right,
            &cfg,
            &binner,
            &index_right,
            t_right_med,
            &idmap,
        );

        // Cap must be enforced strictly
        assert_eq!(edges.len(), 4);

        // Global sort by cost before truncation → resulting slice is non-decreasing by cost
        let costs = edges.iter().map(|e| e.cost).collect::<Vec<_>>();
        let mut sorted = costs.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        assert_eq!(
            costs, sorted,
            "Edges should be globally sorted by cost after cap."
        );
    }

    #[test]
    fn deterministic_results_for_same_inputs() {
        let cfg = permissive_cfg();
        let binner = HealpixBinner::new(6);
        let (left, right) = build_toy_left_right(5, 7, 100, 101, 60000.0, 60000.2);
        let idmap = build_id_to_index(&right);

        let index_right = SeedSpatialIndex::build(&right, &binner); // moved out of the loop
        let t_right_med = median_epoch(&right);

        let edges_a = generate_topk_edges(
            &left,
            &right,
            &cfg,
            &binner,
            &index_right,
            t_right_med,
            &idmap,
        );
        let edges_b = generate_topk_edges(
            &left,
            &right,
            &cfg,
            &binner,
            &index_right,
            t_right_med,
            &idmap,
        );

        assert_eq!(edges_a.len(), edges_b.len());
        for (ea, eb) in edges_a.iter().zip(edges_b.iter()) {
            assert_eq!(ea.from, eb.from);
            assert_eq!(ea.to, eb.to);
            assert!((ea.cost - eb.cost).abs() <= f64::EPSILON);
            assert!((ea.dt_days - eb.dt_days).abs() <= f64::EPSILON);
        }
    }

    #[test]
    fn each_left_has_at_most_topk_edges_before_global_cap() {
        // TopK=2, with no global cap → per-left constraint should hold.
        let cfg = InterNightLinkConfigBuilder::new()
            .with_predict(|p| p.k_sigma(3.5).pad_cell_radius(true))
            .with_scoring(|s| s.set_max_d2_pos(1.0e9))
            .with_limits(|l| l.top_k_per_left(2).max_cost(None).max_total_edges(None))
            .build()
            .unwrap();

        let binner = HealpixBinner::new(7);
        let (left, right) = build_toy_left_right(6, 10, 20, 21, 60000.0, 60001.0);
        let idmap = build_id_to_index(&right);

        let index_right = SeedSpatialIndex::build(&right, &binner); // moved out of the loop
        let t_right_med = median_epoch(&right);

        let edges = generate_topk_edges(
            &left,
            &right,
            &cfg,
            &binner,
            &index_right,
            t_right_med,
            &idmap,
        );

        // Count edges per left
        let mut count_by_left: AHashMap<u64, usize> = AHashMap::new();
        for e in &edges {
            *count_by_left.entry(e.from).or_default() += 1;
        }

        // Every left must have <= TopK edges
        for (_, c) in count_by_left {
            assert!(c <= 2, "Per-left TopK should be enforced");
        }
    }

    // --- Property-based tests ----------------------------------------------

    use proptest::prelude::*;

    proptest! {
        /// For randomized small problems, ensure:
        /// - total edges ≤ |left| × TopK and ≤ global cap if present,
        /// - no edges if right is empty,
        /// - all edges satisfy `cost ≤ max_cost` when provided.
        #[test]
        fn prop_caps_and_cutoffs_hold(
            // number of left/right seeds in a small, tractable range
            n_left in 0usize..8,
            n_right in 0usize..8,
            // nights separated by 0..3 to test Δrevisit semantics (internally clamped ≥1)
            delta_n in 0u32..3,
            // top-k per-left
            topk in 1usize..5,
            // optional global edge cap (0 disables → we map 0 -> None)
            global_cap_raw in 0usize..10,
            // max_cost cutoff (we draw exponent to allow both very small and large)
            max_cost_exp in -30i32..10i32,
        ) {
            let night_left: NightId  = 100;
            let night_right: NightId = night_left + delta_n;

            let epoch_left  = 60000.0;
            let epoch_right = 60000.5;

            let (left, right) = build_toy_left_right(
                n_left, n_right,
                night_left, night_right,
                epoch_left, epoch_right
            );

            // Build config with randomized limits; keep scoring permissive.
            let global_cap = if global_cap_raw == 0 { None } else { Some(global_cap_raw) };
            let max_cost   = Some(10f64.powi(max_cost_exp)); // can be tiny or huge

            let cfg = InterNightLinkConfigBuilder::new()
                .with_predict(|p| p.k_sigma(3.5).pad_cell_radius(true))
                .with_scoring(|s| s.set_max_d2_pos(1.0e9))
                .with_limits(|l| l.top_k_per_left(topk).max_total_edges(global_cap).max_cost(max_cost))
                .build()
                .unwrap();

            let binner = HealpixBinner::new(6);
            let idmap = build_id_to_index(&right);

            let index_right = SeedSpatialIndex::build(&right, &binner); // moved out of the loop
            let t_right_med = median_epoch(&right);

            let edges = generate_topk_edges(
                &left,
                &right,
                &cfg,
                &binner,
                &index_right,
                t_right_med,
                &idmap,
            );

            // 1) Structural bounds.
            let mut bound = n_left.saturating_mul(topk);
            if let Some(cap) = global_cap {
                bound = bound.min(cap);
            }
            // Also bounded by the maximum possible bipartite pairs.
            let theoretical_max = n_left.saturating_mul(n_right);
            bound = bound.min(theoretical_max);

            prop_assert!(edges.len() <= bound, "Edges must respect TopK and global cap.");

            // 2) If right is empty, no edges.
            if n_right == 0 {
                prop_assert!(edges.is_empty());
            }

            // 3) If max_cost is extremely small, likely zero edges (can't assert strict),
            //    but if edges exist, their costs must not exceed max_cost.
            if let Some(cmax) = max_cost {
                for e in &edges {
                    prop_assert!(e.cost <= cmax + f64::EPSILON);
                }
            }

            // 4) Per-left TopK bound always holds, even when global cap is None.
            let mut per_left = AHashMap::<u64, usize>::new();
            for e in &edges {
                *per_left.entry(e.from).or_default() += 1;
            }
            for (_from, cnt) in per_left {
                prop_assert!(cnt <= topk);
            }
        }
    }
}
