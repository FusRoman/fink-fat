use std::fmt::{self, Display, Formatter};

use ahash::{AHashMap, AHashSet};
use fink_fat_engine::{
    engine_config::edge_config::EdgeConfig,
    graph::{edge::Edge, score::ScoredEdge},
    seeding::{seed_id::SeedId, seed_node::SeedNode, seed_spatial_index::SeedSpatialIndex},
    spacetime_bucket::{spatial_binner::SpatialBinner, time_binner::TimeBinner},
};

use crate::night_seeds::NightSeeds;

use fink_fat_engine::{
    astro_math::{l2_norm, radec_to_tangent},
    engine_config::score_config::{
        NumericConfig, PositionScore, PredictConfig, ScoreConfig, VelocityScore,
    },
};

/// Display helper for edge collections: prints summary stats + Top-K best edges.
pub struct EdgesStatsDisplay<'a> {
    pub edges: &'a [Edge<'a>],
    /// Number of best (lowest-cost) edges to show.
    pub top_k: usize,
    /// Whether to show only active edges in stats + Top-K.
    pub only_active: bool,
}

impl<'a> EdgesStatsDisplay<'a> {
    pub fn new(edges: &'a [Edge<'a>]) -> Self {
        Self {
            edges,
            top_k: 10,
            only_active: false,
        }
    }

    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    pub fn only_active(mut self, v: bool) -> Self {
        self.only_active = v;
        self
    }
}

impl<'a> Display for EdgesStatsDisplay<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let edges_iter = self.edges.iter().filter(|e| !self.only_active || e.active);

        // First pass: aggregate stats.
        let mut n: usize = 0;
        let mut n_active: usize = 0;

        let mut cost_sum = 0.0f64;
        let mut cost_min = f64::INFINITY;
        let mut cost_max = f64::NEG_INFINITY;

        let mut dt_sum = 0.0f64;
        let mut dt_min = f64::INFINITY;
        let mut dt_max = f64::NEG_INFINITY;

        // We'll also keep Top-K best edges (lowest cost) without sorting everything.
        // This is O(n log k) and allocates O(k).
        let k = self.top_k;
        let mut top: Vec<&Edge<'a>> = Vec::with_capacity(k);

        for e in edges_iter {
            n += 1;
            if e.active {
                n_active += 1;
            }

            // cost stats
            let c = e.cost;
            if c.is_finite() {
                cost_sum += c;
                cost_min = cost_min.min(c);
                cost_max = cost_max.max(c);
            }

            // dt stats
            let dt = e.dt_days;
            if dt.is_finite() {
                dt_sum += dt;
                dt_min = dt_min.min(dt);
                dt_max = dt_max.max(dt);
            }

            // Maintain Top-K smallest costs (best edges).
            if k > 0 && c.is_finite() {
                if top.len() < k {
                    top.push(e);
                    // Keep `top` sorted descending by cost so worst-of-top is at [0].
                    // (small k => this is cheap)
                    top.sort_by(|a, b| b.cost.total_cmp(&a.cost));
                } else if let Some(worst) = top.first() {
                    if c < worst.cost {
                        top[0] = e;
                        top.sort_by(|a, b| b.cost.total_cmp(&a.cost));
                    }
                }
            }
        }

        // Header
        if self.only_active {
            writeln!(f, "Edges summary (active only):")?;
        } else {
            writeln!(f, "Edges summary:")?;
        }

        if n == 0 {
            writeln!(f, "  n = 0")?;
            return Ok(());
        }

        let cost_mean = cost_sum / (n as f64);
        let dt_mean = dt_sum / (n as f64);

        writeln!(f, "  n = {}", n)?;
        writeln!(
            f,
            "  active = {} ({:.2}%)",
            n_active,
            100.0 * (n_active as f64) / (n as f64)
        )?;
        writeln!(
            f,
            "  cost: min={:.6}  mean={:.6}  max={:.6}",
            cost_min, cost_mean, cost_max
        )?;
        writeln!(
            f,
            "  Δt_days: min={:.6}  mean={:.6}  max={:.6}",
            dt_min, dt_mean, dt_max
        )?;

        // Top-K best edges (lowest cost)
        if k > 0 && !top.is_empty() {
            // `top` is currently sorted descending (worst first), reverse for best-first display.
            top.sort_by(|a, b| a.cost.total_cmp(&b.cost));
            writeln!(f, "  top-{} best edges (lowest cost):", top.len())?;
            for (rank, e) in top.iter().enumerate() {
                writeln!(
                    f,
                    "    #{:<3} cost={:.6}  {} → {}  Δt={:.3} d  [{}]",
                    rank + 1,
                    e.cost,
                    e.from.seed_id,
                    e.to.seed_id,
                    e.dt_days,
                    if e.active { "active" } else { "inactive" }
                )?;
            }
        }

        Ok(())
    }
}

/// Edge truth diagnostics between two nights.
///
/// Returns:
/// - n_true_edges: number of edges in `edges` whose endpoints share the same truth id.
/// - n_false_edges: number of edges in `edges` whose endpoints do not share the same truth id
///   (including cases where one or both truths are missing).
/// - n_true_edges_possible: total number of true edges that *exist* between the two nights,
///   i.e. Σ_t (count_left[t] * count_right[t]) over truth ids present on both nights.
pub fn edge_truth_counts_between_nights<'a>(
    left: &'a NightSeeds,
    right: &'a NightSeeds,
    edges: &'a [Edge<'a>],
) -> (usize, usize, usize, f64) {
    assert_eq!(
        left.seeds.len(),
        left.truth.len(),
        "left NightSeeds: seeds/truth length mismatch"
    );
    assert_eq!(
        right.seeds.len(),
        right.truth.len(),
        "right NightSeeds: seeds/truth length mismatch"
    );

    // Map SeedId -> truth_id for fast lookup when scanning edges.
    let left_truth_by_seed: AHashMap<SeedId, Option<i32>> = left
        .seeds
        .iter()
        .zip(left.truth.iter())
        .map(|(s, t)| (s.seed_id, *t))
        .collect();

    let right_truth_by_seed: AHashMap<SeedId, Option<i32>> = right
        .seeds
        .iter()
        .zip(right.truth.iter())
        .map(|(s, t)| (s.seed_id, *t))
        .collect();

    // Count true/false among produced edges.
    let mut n_true_edges = 0usize;
    let mut n_false_edges = 0usize;

    for e in edges {
        let tl = left_truth_by_seed.get(&e.from.seed_id).copied().flatten();
        let tr = right_truth_by_seed.get(&e.to.seed_id).copied().flatten();

        match (tl, tr) {
            (Some(a), Some(b)) if a == b => n_true_edges += 1,
            _ => n_false_edges += 1,
        }
    }

    // Count how many true edges exist in principle between the nights:
    // for each truth_id t, all pairs left(t) x right(t) are "true edges".
    let mut left_counts: AHashMap<i32, usize> = AHashMap::new();
    for &t in &left.truth {
        if let Some(tid) = t {
            *left_counts.entry(tid).or_insert(0) += 1;
        }
    }

    let mut right_counts: AHashMap<i32, usize> = AHashMap::new();
    for &t in &right.truth {
        if let Some(tid) = t {
            *right_counts.entry(tid).or_insert(0) += 1;
        }
    }

    let mut n_true_edges_possible = 0usize;
    for (tid, &cl) in &left_counts {
        if let Some(&cr) = right_counts.get(tid) {
            n_true_edges_possible += cl * cr;
        }
    }

    let recall = if n_true_edges_possible > 0 {
        (n_true_edges as f64) / (n_true_edges_possible as f64)
    } else {
        0.0
    };

    (n_true_edges, n_false_edges, n_true_edges_possible, recall)
}

/// Résumé des arêtes manquées ou obtenues.
#[derive(Debug, Clone)]
pub struct EdgeTruthSummary {
    /// Arêtes vraies générées (présentes dans `edges` et reliant deux seeds du même astéroïde).
    pub true_edges_generated: usize,
    /// Arêtes fausses générées (autres arêtes dans `edges`).
    pub false_edges_generated: usize,
    /// Nombre total d'arêtes vraies possibles (toutes paires left/right partageant le même `truth_id`).
    pub true_edges_possible: usize,
    /// Arêtes vraies générées par le scoring mais supprimées par le top‑k.
    pub true_edges_missed_top_k: usize,
    /// Arêtes vraies manquées avant top‑k (cône grossier trop petit, binage temporel, scoring filtrant).
    pub true_edges_missed_generation: usize,
}

impl EdgeTruthSummary {
    /// Retourne le rappel (recall) des arêtes vraies = arêtes vraies générées / arêtes vraies possibles.
    pub fn recall(&self) -> f64 {
        if self.true_edges_possible == 0 {
            0.0
        } else {
            (self.true_edges_generated as f64) / (self.true_edges_possible as f64)
        }
    }
}

pub fn edge_truth_diagnostics<'a, B: SpatialBinner, T: TimeBinner>(
    left: &'a NightSeeds,
    right: &'a NightSeeds,
    edges: &[Edge<'a>],
    config: &EdgeConfig,
    spatial_binner: &B,
    time_binner: &T,
) -> EdgeTruthSummary {
    assert_eq!(
        left.seeds.len(),
        left.truth.len(),
        "NightSeeds.left: mismatch seeds/truth lengths"
    );
    assert_eq!(
        right.seeds.len(),
        right.truth.len(),
        "NightSeeds.right: mismatch seeds/truth lengths"
    );

    let left_truth_by_id: AHashMap<SeedId, Option<i32>> = left
        .seeds
        .iter()
        .zip(left.truth.iter())
        .map(|(s, t)| (s.seed_id, *t))
        .collect();

    let right_truth_by_id: AHashMap<SeedId, Option<i32>> = right
        .seeds
        .iter()
        .zip(right.truth.iter())
        .map(|(s, t)| (s.seed_id, *t))
        .collect();

    let mut true_edges_generated = 0usize;
    let mut false_edges_generated = 0usize;

    // Compute the revisit separation (Δ nights), enforced to be ≥ 1.
    let delta_revisit: u32 = right.nid.0.saturating_sub(left.nid.0).max(1);

    for e in edges {
        let tl = left_truth_by_id.get(&e.from.seed_id).copied().flatten();
        let tr = right_truth_by_id.get(&e.to.seed_id).copied().flatten();

        match (tl, tr) {
            (Some(a), Some(b)) if a == b => true_edges_generated += 1,
            _ => false_edges_generated += 1,
        }
    }

    let mut left_counts: AHashMap<i32, usize> = AHashMap::new();
    for &t in &left.truth {
        if let Some(tid) = t {
            *left_counts.entry(tid).or_insert(0) += 1;
        }
    }
    let mut right_counts: AHashMap<i32, usize> = AHashMap::new();
    for &t in &right.truth {
        if let Some(tid) = t {
            *right_counts.entry(tid).or_insert(0) += 1;
        }
    }
    let mut true_edges_possible = 0usize;
    for (tid, &cl) in &left_counts {
        if let Some(&cr) = right_counts.get(tid) {
            true_edges_possible += cl * cr;
        }
    }

    let top_k = config.top_k_per_left;
    let mut true_edges_missed_top_k = 0usize;

    for src in &left.seeds {
        let src_truth = left_truth_by_id.get(&src.seed_id).copied().flatten();

        if src_truth.is_none() {
            continue;
        }
        let tid = src_truth.unwrap();

        let mut scored = src.score_edge_candidates(
            &right.seeds,
            spatial_binner,
            time_binner,
            config,
            delta_revisit,
        );

        let k = top_k.min(scored.len());
        if k > 0 {
            let _ = scored.select_nth_unstable_by(k - 1, |a, b| a.cost.total_cmp(&b.cost));
            scored[..k].sort_by(|a, b| a.cost.total_cmp(&b.cost));
            scored.truncate(k);
        }

        let mut true_scored: Vec<SeedId> = Vec::new();
        for se in src.score_edge_candidates(
            &right.seeds,
            spatial_binner,
            time_binner,
            config,
            delta_revisit,
        ) {
            if let Some(truth_r) = right_truth_by_id.get(&se.to).and_then(|&t| t) {
                if truth_r == tid {
                    true_scored.push(se.to);
                }
            }
        }

        let mut true_in_top: Vec<SeedId> = Vec::new();
        for se in scored {
            if let Some(truth_r) = right_truth_by_id.get(&se.to).and_then(|&t| t) {
                if truth_r == tid {
                    true_in_top.push(se.to);
                }
            }
        }

        for id in true_scored {
            if !true_in_top.iter().any(|&x| x == id) {
                true_edges_missed_top_k += 1;
            }
        }
    }

    let true_edges_missed_generation =
        true_edges_possible.saturating_sub(true_edges_generated + true_edges_missed_top_k);

    EdgeTruthSummary {
        true_edges_generated,
        false_edges_generated,
        true_edges_possible,
        true_edges_missed_top_k,
        true_edges_missed_generation,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MissReason {
    /// The right seed is in the correct night but fell outside the time bin slice `[lo, hi)`.
    NotInTimeBin,
    /// The right seed is in the bin slice, but the spatial cone query did not return it.
    NotInSpatialCone,
    /// The right seed was a spatial candidate, but the exact scorer rejected it (returned None).
    RejectedByScorer,
}

#[derive(Debug, Clone)]
pub struct MissedTrueEdge {
    pub from: SeedId,
    pub to: SeedId,
    pub truth_id: i32,
    pub reason: MissReason,

    // Small, useful context for debugging:
    pub t0: f64,
    pub t1: f64,
    pub t_center: f64,
    pub dt_half: f64,
    pub cone_radius: f64,
    pub to_epoch: f64,
    pub scorer_reject: Option<ScoreRejectDetail>,
}

fn build_truth_maps(
    left: &NightSeeds,
    right: &NightSeeds,
) -> (AHashMap<SeedId, Option<i32>>, AHashMap<SeedId, Option<i32>>) {
    let left_truth: AHashMap<SeedId, Option<i32>> = left
        .seeds
        .iter()
        .zip(left.truth.iter())
        .map(|(s, t)| (s.seed_id, *t))
        .collect();

    let right_truth: AHashMap<SeedId, Option<i32>> = right
        .seeds
        .iter()
        .zip(right.truth.iter())
        .map(|(s, t)| (s.seed_id, *t))
        .collect();

    (left_truth, right_truth)
}

fn oracle_true_pairs(
    left: &NightSeeds,
    right: &NightSeeds,
    left_truth: &AHashMap<SeedId, Option<i32>>,
    right_truth: &AHashMap<SeedId, Option<i32>>,
) -> Vec<(SeedId, SeedId, i32)> {
    // Group right seeds by truth id
    let mut right_by_tid: AHashMap<i32, Vec<SeedId>> = AHashMap::new();
    for s in &right.seeds {
        if let Some(tid) = right_truth.get(&s.seed_id).copied().flatten() {
            right_by_tid.entry(tid).or_default().push(s.seed_id);
        }
    }

    // For each left seed with truth, pair with all right seeds of same truth
    let mut pairs = Vec::new();
    for s in &left.seeds {
        if let Some(tid) = left_truth.get(&s.seed_id).copied().flatten() {
            if let Some(v) = right_by_tid.get(&tid) {
                for &to in v {
                    pairs.push((s.seed_id, to, tid));
                }
            }
        }
    }
    pairs
}

fn hits_from_scored_edges(scored: &[ScoredEdge]) -> AHashSet<(SeedId, SeedId)> {
    scored.iter().map(|se| (se.from, se.to)).collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ScoreRejectKind {
    // Position gate
    PosNonFinite,
    PosGateExceeded,

    // Velocity gate / numeric issues
    VelEpsInvalid,
    VelPredictNonFinite,
    VelCosNonFinite,
    VelDirGate,
    VelDvNonFinite,
    VelSpeedGate,
}

#[derive(Debug, Clone, Copy)]
pub struct ScoreRejectDetail {
    pub kind: ScoreRejectKind,

    // Context fields (only some are meaningful depending on kind)
    pub d2_pos: f64,
    pub max_d2: f64,

    pub eps_days: f64,
    pub cosang: f64,
    pub cos_min: f64,

    pub dv: f64,
    pub max_speed_diff: f64,
}

/// “Mirror” of the hard-gating logic in `ScoredEdge::score`, but returning a reason.
///
/// Important: we intentionally DO NOT reproduce photometry/gap/band logic,
/// because those do not reject edges in the current scorer implementation.
fn explain_score_reject(
    i: &SeedNode,
    j: &SeedNode,
    cfg: &ScoreConfig,
    dt_days: f64,
) -> Option<ScoreRejectDetail> {
    // -------------------------
    // 1) Position term gate
    // -------------------------
    match compute_position_term_mirror(i, j, &cfg.predict, &cfg.position, &cfg.numeric) {
        Ok(d2_pos) => {
            if !d2_pos.is_finite() {
                return Some(ScoreRejectDetail {
                    kind: ScoreRejectKind::PosNonFinite,
                    d2_pos,
                    max_d2: cfg.position.max_d2,
                    eps_days: f64::NAN,
                    cosang: f64::NAN,
                    cos_min: f64::NAN,
                    dv: f64::NAN,
                    max_speed_diff: cfg.velocity.max_speed_diff,
                });
            }
            if d2_pos > cfg.position.max_d2 {
                return Some(ScoreRejectDetail {
                    kind: ScoreRejectKind::PosGateExceeded,
                    d2_pos,
                    max_d2: cfg.position.max_d2,
                    eps_days: f64::NAN,
                    cosang: f64::NAN,
                    cos_min: f64::NAN,
                    dv: f64::NAN,
                    max_speed_diff: cfg.velocity.max_speed_diff,
                });
            }
        }
        Err(detail) => return Some(detail),
    }

    // -------------------------
    // 2) Velocity gates
    // -------------------------
    if let Err(detail) = compute_velocity_terms_mirror(i, j, dt_days, &cfg.velocity) {
        return Some(detail);
    }

    None
}

fn compute_position_term_mirror(
    i: &SeedNode,
    j: &SeedNode,
    predict: &PredictConfig,
    cfg: &PositionScore,
    numeric: &NumericConfig,
) -> Result<f64, ScoreRejectDetail> {
    let t_j = j.plane.epoch_mid;

    // Predict i to t_j on its own plane
    let (p_hat, cov_i) = i.plane.predict_on_plane(t_j, &predict.noise);
    let (px, py) = (p_hat[0], p_hat[1]);

    // Project j to i's plane
    let p_j = radec_to_tangent(
        j.plane.ra_mid,
        j.plane.dec_mid,
        i.plane.center.ra0,
        i.plane.center.dec0,
    );
    let (dx, dy) = (p_j[0] - px, p_j[1] - py);

    // Diagonal covariance S = cov_i + cov_pos(j)
    let mut sxx = (cov_i[0][0] + j.plane.cov_pos[0][0]).max(0.0);
    let mut syy = (cov_i[1][1] + j.plane.cov_pos[1][1]).max(0.0);

    if numeric.min_variance > 0.0 {
        sxx = sxx.max(numeric.min_variance);
        syy = syy.max(numeric.min_variance);
    }

    let inv_sxx = 1.0 / sxx;
    let inv_syy = 1.0 / syy;

    let d2_pos = dx * dx * inv_sxx + dy * dy * inv_syy;

    if !d2_pos.is_finite() {
        return Err(ScoreRejectDetail {
            kind: ScoreRejectKind::PosNonFinite,
            d2_pos,
            max_d2: cfg.max_d2,
            eps_days: f64::NAN,
            cosang: f64::NAN,
            cos_min: f64::NAN,
            dv: f64::NAN,
            max_speed_diff: f64::NAN,
        });
    }
    if d2_pos > cfg.max_d2 {
        return Err(ScoreRejectDetail {
            kind: ScoreRejectKind::PosGateExceeded,
            d2_pos,
            max_d2: cfg.max_d2,
            eps_days: f64::NAN,
            cosang: f64::NAN,
            cos_min: f64::NAN,
            dv: f64::NAN,
            max_speed_diff: f64::NAN,
        });
    }

    Ok(d2_pos)
}

fn compute_velocity_terms_mirror(
    i: &SeedNode,
    j: &SeedNode,
    dt_days: f64,
    cfg: &VelocityScore,
) -> Result<(), ScoreRejectDetail> {
    // Predict velocity of i at t_j
    let vi = if let Some(a) = i.plane.acc_xy {
        [
            i.plane.vel_xy[0] + a[0] * dt_days,
            i.plane.vel_xy[1] + a[1] * dt_days,
        ]
    } else {
        i.plane.vel_xy
    };

    // Symmetric finite difference for j, in i's plane
    let t_j = j.plane.epoch_mid;
    let eps = cfg.vel_eps_days;

    if !eps.is_finite() || eps <= 0.0 {
        return Err(ScoreRejectDetail {
            kind: ScoreRejectKind::VelEpsInvalid,
            d2_pos: f64::NAN,
            max_d2: f64::NAN,
            eps_days: eps,
            cosang: f64::NAN,
            cos_min: cfg.cos_max_theta(),
            dv: f64::NAN,
            max_speed_diff: cfg.max_speed_diff,
        });
    }

    let (ra_p, dec_p) = j.predict_radec(t_j + eps);
    let (ra_m, dec_m) = j.predict_radec(t_j - eps);

    if !(ra_p.is_finite() && dec_p.is_finite() && ra_m.is_finite() && dec_m.is_finite()) {
        return Err(ScoreRejectDetail {
            kind: ScoreRejectKind::VelPredictNonFinite,
            d2_pos: f64::NAN,
            max_d2: f64::NAN,
            eps_days: eps,
            cosang: f64::NAN,
            cos_min: cfg.cos_max_theta(),
            dv: f64::NAN,
            max_speed_diff: cfg.max_speed_diff,
        });
    }

    let p_plus = i.plane.radec_to_tangent_precomp(ra_p, dec_p);
    let p_minus = i.plane.radec_to_tangent_precomp(ra_m, dec_m);
    if !(p_plus[0].is_finite()
        && p_plus[1].is_finite()
        && p_minus[0].is_finite()
        && p_minus[1].is_finite())
    {
        return Err(ScoreRejectDetail {
            kind: ScoreRejectKind::VelPredictNonFinite,
            d2_pos: f64::NAN,
            max_d2: f64::NAN,
            eps_days: eps,
            cosang: f64::NAN,
            cos_min: cfg.cos_max_theta(),
            dv: f64::NAN,
            max_speed_diff: cfg.max_speed_diff,
        });
    }

    let inv_2eps = 1.0 / (2.0 * eps);
    let vj = [
        (p_plus[0] - p_minus[0]) * inv_2eps,
        (p_plus[1] - p_minus[1]) * inv_2eps,
    ];

    let norm_vi = l2_norm(vi[0], vi[1]);
    let norm_vj = l2_norm(vj[0], vj[1]);

    // Degenerate => scorer does NOT reject, it returns Some((None,None))
    if norm_vi == 0.0 || norm_vj == 0.0 {
        return Ok(());
    }

    let inv_norms = 1.0 / (norm_vi * norm_vj);
    let cosang = ((vi[0] * vj[0] + vi[1] * vj[1]) * inv_norms).clamp(-1.0, 1.0);

    if !cosang.is_finite() {
        return Err(ScoreRejectDetail {
            kind: ScoreRejectKind::VelCosNonFinite,
            d2_pos: f64::NAN,
            max_d2: f64::NAN,
            eps_days: eps,
            cosang,
            cos_min: cfg.cos_max_theta(),
            dv: f64::NAN,
            max_speed_diff: cfg.max_speed_diff,
        });
    }

    // Direction gate
    let cos_min = cfg.cos_max_theta();
    if cosang < cos_min {
        return Err(ScoreRejectDetail {
            kind: ScoreRejectKind::VelDirGate,
            d2_pos: f64::NAN,
            max_d2: f64::NAN,
            eps_days: eps,
            cosang,
            cos_min,
            dv: f64::NAN,
            max_speed_diff: cfg.max_speed_diff,
        });
    }

    // Speed gate
    let dv = (norm_vi - norm_vj).abs();
    if !dv.is_finite() {
        return Err(ScoreRejectDetail {
            kind: ScoreRejectKind::VelDvNonFinite,
            d2_pos: f64::NAN,
            max_d2: f64::NAN,
            eps_days: eps,
            cosang,
            cos_min,
            dv,
            max_speed_diff: cfg.max_speed_diff,
        });
    }
    if dv > cfg.max_speed_diff {
        return Err(ScoreRejectDetail {
            kind: ScoreRejectKind::VelSpeedGate,
            d2_pos: f64::NAN,
            max_d2: f64::NAN,
            eps_days: eps,
            cosang,
            cos_min,
            dv,
            max_speed_diff: cfg.max_speed_diff,
        });
    }

    Ok(())
}

pub fn diagnose_missed_true_edges<B: SpatialBinner, T: TimeBinner>(
    left: &NightSeeds,
    right: &NightSeeds,
    edge_config: &EdgeConfig,
    spatial_binner: &B,
    time_binner: &T,
    delta_revisit: u32,
) -> (AHashMap<MissReason, usize>, Vec<MissedTrueEdge>) {
    let (left_truth, right_truth) = build_truth_maps(left, right);

    // Toutes les arêtes vraies oracle (potentielles)
    let oracle = oracle_true_pairs(left, right, &left_truth, &right_truth);

    // Pour savoir si une arête vraie est déjà produite, on la recalcule au même niveau que top-k:
    // => on génère tous les scored edges via score_edge_candidates pour chaque seed gauche,
    // puis on garde les top-k (exactement comme ton pipeline).
    //
    // Ici, on veut juste savoir "miss préfiltrage" => si elle n'est même pas scorée candidate.
    // On construit un ensemble des couples (from,to) effectivement scorés-candidats (avant top-k).
    let mut scored_candidates: AHashSet<(SeedId, SeedId)> = AHashSet::new();
    for src in &left.seeds {
        let scored = src.score_edge_candidates(
            &right.seeds,
            spatial_binner,
            time_binner,
            edge_config,
            delta_revisit,
        );
        scored_candidates.extend(scored.iter().map(|se| (se.from, se.to)));
    }

    // Construire un accès rapide SeedId -> &SeedNode pour right
    let right_by_id: AHashMap<SeedId, &SeedNode> =
        right.seeds.iter().map(|s| (s.seed_id, s)).collect();
    let left_by_id: AHashMap<SeedId, &SeedNode> =
        left.seeds.iter().map(|s| (s.seed_id, s)).collect();

    // Précompute bin list once (same for all src because it depends on right time span).
    let t_min_r = right
        .seeds
        .first()
        .map(|s| s.plane.epoch_mid)
        .unwrap_or(0.0);
    let t_max_r = right.seeds.last().map(|s| s.plane.epoch_mid).unwrap_or(0.0);
    let bins = time_binner.bins_in_range(t_min_r, t_max_r);

    let pred_cfg = &edge_config.predictor_config;
    let score_cfg = &edge_config.score_config;

    let mut counts: AHashMap<MissReason, usize> = AHashMap::new();
    let mut details: Vec<MissedTrueEdge> = Vec::new();

    // On ne diagnostique que les arêtes vraies oracle qui ne sont PAS dans les candidats scorés.
    // (puisque ton diagnostic a déjà montré que top-k n'en enlève aucune)
    for (from, to, tid) in oracle {
        if scored_candidates.contains(&(from, to)) {
            continue; // celle-ci n'est pas "préfiltrage manquée"
        }

        let Some(src) = left_by_id.get(&from).copied() else {
            continue;
        };
        let Some(dst) = right_by_id.get(&to).copied() else {
            continue;
        };

        // Trouver le bin qui contient dst.epoch_mid
        let t_to = dst.plane.epoch_mid;
        let mut found_bin = None;
        for bin in &bins {
            let t0 = time_binner.bin_start(bin.0);
            let t1 = time_binner.bin_end(bin.0);
            if t_to >= t0 && t_to < t1 {
                found_bin = Some((bin.0, t0, t1));
                break;
            }
        }

        // Si le time_binner ne couvre pas exactement la plage, ça peut arriver.
        let Some((_, t0, t1)) = found_bin else {
            *counts.entry(MissReason::NotInTimeBin).or_insert(0) += 1;
            details.push(MissedTrueEdge {
                from,
                to,
                truth_id: tid,
                reason: MissReason::NotInTimeBin,
                t0: f64::NAN,
                t1: f64::NAN,
                t_center: f64::NAN,
                dt_half: 0.0,
                cone_radius: f64::NAN,
                to_epoch: t_to,
                scorer_reject: None,
            });
            continue;
        };

        // Slice [lo,hi) pour ce bin
        // (version scan monotone possible, ici simple pour diagnostic)
        let lo = lower_bound_epoch(&right.seeds, t0);
        let hi = lower_bound_epoch(&right.seeds, t1);
        if !(lo < hi) {
            *counts.entry(MissReason::NotInTimeBin).or_insert(0) += 1;
            details.push(MissedTrueEdge {
                from,
                to,
                truth_id: tid,
                reason: MissReason::NotInTimeBin,
                t0,
                t1,
                t_center: 0.5 * (t0 + t1),
                dt_half: 0.5 * time_binner.bin_width().max(1e-12),
                cone_radius: f64::NAN,
                to_epoch: t_to,
                scorer_reject: None,
            });
            continue;
        }

        // Vérifie que dst est bien dans right[lo..hi)
        // (sinon c’est une incohérence de binning / tri)
        let in_slice = right.seeds[lo..hi].iter().any(|s| s.seed_id == to);
        if !in_slice {
            *counts.entry(MissReason::NotInTimeBin).or_insert(0) += 1;
            details.push(MissedTrueEdge {
                from,
                to,
                truth_id: tid,
                reason: MissReason::NotInTimeBin,
                t0,
                t1,
                t_center: 0.5 * (t0 + t1),
                dt_half: 0.5 * time_binner.bin_width().max(1e-12),
                cone_radius: f64::NAN,
                to_epoch: t_to,
                scorer_reject: None,
            });
            continue;
        }

        // Build bin-local spatial index and do cone query (exactly like algo)
        let index_bin = SeedSpatialIndex::build(&right.seeds[lo..hi], spatial_binner);

        let t_center = 0.5 * (t0 + t1);
        let v = src.plane.vel_xy;
        let speed = (v[0].mul_add(v[0], v[1] * v[1])).sqrt();
        let speed_eff = (speed + pred_cfg.v_slack).max(0.0);
        let dt_half = 0.5 * time_binner.bin_width().max(1e-12);

        let (ra_c, dec_c, mut r_c) = src.predict_cone(t_center, spatial_binner, pred_cfg);
        r_c += speed_eff * dt_half;

        let mut was_spatial_candidate = false;
        for cand in index_bin.cone_query(spatial_binner, ra_c, dec_c, r_c) {
            if cand.seed_id == to {
                was_spatial_candidate = true;
                // It was returned by spatial query; check scorer.
                let ok = ScoredEdge::score(src, cand, score_cfg, delta_revisit).is_some();
                if !ok {
                    *counts.entry(MissReason::RejectedByScorer).or_insert(0) += 1;
                    details.push(MissedTrueEdge {
                        from,
                        to,
                        truth_id: tid,
                        reason: MissReason::RejectedByScorer,
                        t0,
                        t1,
                        t_center,
                        dt_half,
                        cone_radius: r_c,
                        to_epoch: t_to,
                        scorer_reject: explain_score_reject(
                            src,
                            cand,
                            score_cfg,
                            t_to - src.plane.epoch_mid,
                        ),
                    });
                } else {
                    // Should not happen: if scorer returns Some, it should have been in candidates set.
                    // But we keep it as "spatial ok" anyway.
                    *counts.entry(MissReason::RejectedByScorer).or_insert(0) += 1;
                    details.push(MissedTrueEdge {
                        from,
                        to,
                        truth_id: tid,
                        reason: MissReason::RejectedByScorer,
                        t0,
                        t1,
                        t_center,
                        dt_half,
                        cone_radius: r_c,
                        to_epoch: t_to,
                        scorer_reject: None,
                    });
                }
                break;
            }
        }

        if !was_spatial_candidate {
            *counts.entry(MissReason::NotInSpatialCone).or_insert(0) += 1;
            details.push(MissedTrueEdge {
                from,
                to,
                truth_id: tid,
                reason: MissReason::NotInSpatialCone,
                t0,
                t1,
                t_center,
                dt_half,
                cone_radius: r_c,
                to_epoch: t_to,
                scorer_reject: None,
            });
        }
    }

    (counts, details)
}

pub fn print_miss_report(
    counts: &AHashMap<MissReason, usize>,
    details: &[MissedTrueEdge],
    max_show: usize,
) {
    let n = details.len();
    println!("Miss report (true edges missed by prefilter): n={}", n);

    let get = |r| counts.get(&r).copied().unwrap_or(0);
    println!("  NotInTimeBin     : {}", get(MissReason::NotInTimeBin));
    println!("  NotInSpatialCone : {}", get(MissReason::NotInSpatialCone));
    println!("  RejectedByScorer : {}", get(MissReason::RejectedByScorer));

    // -------------------------------------------------------------------------
    // Breakdown for scorer rejections
    // -------------------------------------------------------------------------
    let mut n_rej_with_detail: usize = 0;
    let mut n_rej_without_detail: usize = 0;

    // Use BTreeMap for stable, deterministic print order.
    let mut by_kind: std::collections::BTreeMap<ScoreRejectKind, usize> =
        std::collections::BTreeMap::new();

    for m in details
        .iter()
        .filter(|m| m.reason == MissReason::RejectedByScorer)
    {
        if let Some(r) = m.scorer_reject {
            n_rej_with_detail += 1;
            *by_kind.entry(r.kind).or_insert(0) += 1;
        } else {
            n_rej_without_detail += 1;
        }
    }

    if get(MissReason::RejectedByScorer) > 0 {
        println!("  RejectedByScorer breakdown:");
        if !by_kind.is_empty() {
            for (k, v) in &by_kind {
                println!("    {:?}: {}", k, v);
            }
        } else {
            println!("    (no detailed reasons collected)");
        }

        if n_rej_without_detail > 0 {
            println!(
                "    note: {} scorer rejections had no ScoreRejectDetail attached",
                n_rej_without_detail
            );
        }
    }

    // -------------------------------------------------------------------------
    // Show a small sample (first `max_show`)
    // -------------------------------------------------------------------------
    for (i, m) in details.iter().take(max_show).enumerate() {
        match m.reason {
            MissReason::RejectedByScorer => {
                if let Some(r) = m.scorer_reject {
                    // Print extra numeric context depending on rejection kind
                    let extra = match r.kind {
                        ScoreRejectKind::PosNonFinite => {
                            format!("d2_pos={:.6} (non-finite) max_d2={:.6}", r.d2_pos, r.max_d2)
                        }
                        ScoreRejectKind::PosGateExceeded => {
                            format!("d2_pos={:.6} > max_d2={:.6}", r.d2_pos, r.max_d2)
                        }
                        ScoreRejectKind::VelEpsInvalid => {
                            format!("vel_eps_days={:.6} (invalid)", r.eps_days)
                        }
                        ScoreRejectKind::VelPredictNonFinite => format!(
                            "predict_radec/projection produced non-finite values (eps={:.6})",
                            r.eps_days
                        ),
                        ScoreRejectKind::VelCosNonFinite => format!(
                            "cosang={:.6} (non-finite) cos_min={:.6} eps={:.6}",
                            r.cosang, r.cos_min, r.eps_days
                        ),
                        ScoreRejectKind::VelDirGate => format!(
                            "cosang={:.6} < cos_min={:.6} (dir gate) eps={:.6}",
                            r.cosang, r.cos_min, r.eps_days
                        ),
                        ScoreRejectKind::VelDvNonFinite => format!(
                            "dv={:.6} (non-finite) max_speed_diff={:.6} eps={:.6}",
                            r.dv, r.max_speed_diff, r.eps_days
                        ),
                        ScoreRejectKind::VelSpeedGate => format!(
                            "dv={:.6} > max_speed_diff={:.6} (speed gate) eps={:.6}",
                            r.dv, r.max_speed_diff, r.eps_days
                        ),
                    };

                    println!(
                        "  #{:<3} tid={}  {} -> {}  reason={:?}/{:?}  to_epoch={:.6}  bin=[{:.6},{:.6})  cone_r={:.6}  |  {}",
                        i + 1,
                        m.truth_id,
                        m.from,
                        m.to,
                        m.reason,
                        r.kind,
                        m.to_epoch,
                        m.t0,
                        m.t1,
                        m.cone_radius,
                        extra
                    );
                } else {
                    // Scorer rejection but no detail (should be rare if you always call explain_score_reject)
                    println!(
                        "  #{:<3} tid={}  {} -> {}  reason={:?}  to_epoch={:.6}  bin=[{:.6},{:.6})  cone_r={:.6}  |  (no ScoreRejectDetail)",
                        i + 1,
                        m.truth_id,
                        m.from,
                        m.to,
                        m.reason,
                        m.to_epoch,
                        m.t0,
                        m.t1,
                        m.cone_radius
                    );
                }
            }
            _ => {
                // Keep original compact formatting for non-scorer misses
                println!(
                    "  #{:<3} tid={}  {} -> {}  reason={:?}  to_epoch={:.6}  bin=[{:.6},{:.6})  cone_r={:.6}",
                    i + 1,
                    m.truth_id,
                    m.from,
                    m.to,
                    m.reason,
                    m.to_epoch,
                    m.t0,
                    m.t1,
                    m.cone_radius
                );
            }
        }
    }
}

#[inline]
fn lower_bound_epoch(right: &[SeedNode], t_target: f64) -> usize {
    let mut lower = 0usize;
    let mut upper = right.len();

    while lower < upper {
        let mid = lower + (upper - lower) / 2;
        if right[mid].plane.epoch_mid < t_target {
            lower = mid + 1;
        } else {
            upper = mid;
        }
    }
    lower
}
