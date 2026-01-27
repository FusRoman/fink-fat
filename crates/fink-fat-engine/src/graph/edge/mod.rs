pub mod score;
pub mod edge_id;
pub mod edge_prediction;


use std::fmt::{self, Display, Formatter};

use ahash::AHashMap;

use crate::{
    engine_config::edge_config::EdgeConfig,
    graph::edge::edge_id::EdgeId,
    night_id::NightId,
    seeding::{seed_id::SeedId, seed_node::SeedNode, seed_spatial_index::SeedSpatialIndex},
    spacetime_bucket::{spatial_binner::SpatialBinner, time_binner::TimeBinner},
};

/// Vecteur de caractéristiques dérivées d'une arête.
///
/// Chaque champ est documenté et représente une grandeur physique ou
/// photométrique déduite des deux `SeedNode` reliés.
#[derive(Clone, Debug)]
pub struct EdgeFeatures {
    pub dt_days: f64,
    pub dt_days_sq: f64,
    pub inv_dt_days: f64,
    pub d2_pos: f64,
    pub log_d2_pos: f64,
    pub resid_norm: f64,
    pub resid_dx: f64,
    pub resid_dy: f64,
    pub speed_from: f64,
    pub speed_to: f64,
    pub speed_diff: f64,
    pub distance_travelled: f64,
    pub n_obs_from: f64,
    pub n_obs_to: f64,
    pub trace_cov_pos_from: f64,
    pub trace_cov_pos_to: f64,
    pub trace_cov_vel_from: f64,
    pub trace_cov_vel_to: f64,
    pub flux_abs_diff: f64,
    pub z_flux: f64,
    pub flux_std_ratio: f64,
    pub band_shared: f64,
    pub has_acc: f64,
}

/// Directed link from an older node to a newer node (forward in time).
///
/// Cost
/// ----
/// `cost` is a **dimensionless** score already aggregated upstream:
/// lower is better. Enforce strictly **positive** costs to avoid
/// degeneracies in later solvers.
#[derive(Clone, Debug)]
pub struct Edge<'a> {
    pub id: EdgeId,
    pub from: &'a SeedNode,
    pub to: &'a SeedNode,
    pub cost: f64,
    /// Time gap in days (TT) between the two seeds (positive).
    pub dt_days: f64,
    /// Whether the edge is currently active (used by solvers / CC exact recompute).
    pub active: bool,
}

impl<'a> Display for Edge<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Edge {{ id: {}, from: {}, to: {}, Δt: {:.3} d, cost: {:.6}, active: {} }}",
            self.id, self.from.seed_id, self.to.seed_id, self.dt_days, self.cost, self.active,
        )
    }
}

impl<'a> Edge<'a> {
    pub fn new(id: EdgeId, from: &'a SeedNode, to: &'a SeedNode, cost: f64, dt_days: f64) -> Self {
        assert!(
            cost.is_finite() && cost > 0.0,
            "Edge cost must be finite and > 0."
        );
        assert!(
            dt_days.is_finite() && dt_days > 0.0,
            "dt_days must be finite and > 0."
        );
        Self {
            id,
            from,
            to,
            cost,
            dt_days,
            active: true,
        }
    }

    /* -------------------------- Top-K Edge Generation ------------------------- */

    /// Generate **Top-K scored directed edges** from a set of left-hand seeds to a
    /// set of right-hand seeds.
    ///
    /// Overview
    /// --------
    /// This routine builds the inter-night graph edges between two consecutive
    /// (or non-consecutive) nights by:
    ///
    /// 1. Scoring candidate edges **independently for each left-hand seed** using
    ///    [`SeedNode::score_edge_candidates`], which applies spatio-temporal
    ///    prefiltering followed by exact kinematic scoring.
    /// 2. Retaining only the **Top-K lowest-cost edges per left seed** using
    ///    partial selection (no full sort).
    /// 3. Converting scored edges into final [`Edge`] objects that directly store
    ///    references to the source and target [`SeedNode`]s.
    /// 4. Optionally enforcing a **global cap** on the total number of edges
    ///    produced across all left-hand seeds.
    ///
    /// The returned edges are immediately usable by downstream graph solvers
    /// (e.g. min-cost flow) without any additional `(night, seed) → node`
    /// resolution step.
    ///
    /// Assumptions
    /// -----------
    /// - All `left` seeds belong to the same night.
    /// - All `right` seeds belong to the same (later) night.
    /// - `right` contains **all seeds referenced by scored edges** (by `SeedId`).
    /// - Costs returned by [`ScoredEdge::score`] are **dimensionless** and
    ///   strictly comparable (lower is better).
    ///
    /// Parameters
    /// ----------
    /// * `id_start` – First [`EdgeId`] to assign; subsequent edges are assigned
    ///   monotonically increasing IDs.
    /// * `left` – Slice of source seeds (earlier night).
    /// * `right` – Slice of target seeds (later night).
    /// * `edge_config` – Configuration controlling:
    ///   - per-left Top-K (`top_k_per_left`),
    ///   - optional global edge cap (`max_total_edges`),
    ///   - scoring and predictor parameters.
    /// * `spatial_binner` – Spatial partitioner used during candidate generation
    ///   (e.g. HEALPix).
    /// * `time_binner` – Time binning strategy used during candidate generation
    ///   (uniform or custom).
    ///
    /// Returns
    /// -------
    /// * `Vec<Edge<'a>>` – Directed edges from `left` to `right`, each carrying:
    ///   - references to the source and target `SeedNode`,
    ///   - the aggregated cost,
    ///   - the inter-night time difference `dt_days`.
    ///
    /// The vector is sorted **only if** a global cap is applied; otherwise the
    /// order reflects per-left Top-K extraction.
    ///
    /// Complexity
    /// ----------
    /// Let:
    /// - `L = left.len()`
    /// - `K = edge_config.top_k_per_left`
    /// - `C_i` be the number of scored candidates generated for left seed `i`
    ///
    /// Per left seed:
    /// - Candidate generation: cost of `score_edge_candidates`
    /// - Top-K selection: `O(C_i)` via `select_nth_unstable`
    ///
    /// Total:
    /// - `O(Σ_i C_i + L · K)` for per-left processing
    /// - Optional global cap: `O(E)` where `E ≤ L · K`
    ///
    /// Memory
    /// ------
    /// - Stores at most `L · K` edges before optional global truncation.
    /// - Uses a temporary `HashMap<SeedId, &SeedNode>` for safe ID → node resolution.
    ///
    /// Notes
    /// -----
    /// - Partial selection is used instead of full sorting to keep runtime linear
    ///   in the number of scored candidates.
    /// - The `SeedId → &SeedNode` map avoids relying on any implicit relationship
    ///   between `SeedId` and slice indices in `right`.
    /// - This function is intentionally sequential; parallelism is expected to be
    ///   handled at a higher level (e.g. across night pairs).
    pub fn generate_topk_edges<B: SpatialBinner, T: TimeBinner>(
        id_start: EdgeId,
        left: &'a [SeedNode],
        right: &'a [SeedNode],
        edge_config: &EdgeConfig,
        spatial_binner: &B,
        time_binner: &T,
    ) -> Vec<Self> {
        // Compute the revisit separation (Δ nights), enforced to be ≥ 1.
        // This is passed downstream to scoring for time-dependent penalties.
        let night_left: NightId = left.first().map(|s| s.night_id).unwrap_or_default();
        let night_right: NightId = right.first().map(|s| s.night_id).unwrap_or_default();
        let delta_revisit: u32 = night_right.0.saturating_sub(night_left.0).max(1);

        let top_k = edge_config.top_k_per_left;
        let max_total = edge_config.max_total_edges;

        // Build once: mapping from SeedId to &SeedNode for the right-hand night.
        //
        // This avoids relying on any implicit `SeedId -> index` correspondence
        // and guarantees safe resolution when converting ScoredEdge → Edge.
        let right_by_id: AHashMap<SeedId, &SeedNode> =
            right.iter().map(|s| (s.seed_id, s)).collect();

        // Upper bound: at most `left.len() * top_k` edges before global truncation.
        let mut edges: Vec<Self> = Vec::with_capacity(left.len() * top_k);
        let mut next_id = id_start.0;

        let right_index = SeedSpatialIndex::build(right, spatial_binner, time_binner);

        // Process each left-hand seed independently.
        for src in left {
            // Generate and score all candidate edges from this source seed.
            let mut scored =
                src.score_edge_candidates(right, &right_index, edge_config, delta_revisit);

            // Retain only the Top-K lowest-cost edges for this source.
            let k = top_k.min(scored.len());
            if k == 0 {
                continue;
            }

            // Partial selection: ensures the K smallest-cost elements are in [..k).
            let _ = scored.select_nth_unstable_by(k - 1, |a, b| a.cost.total_cmp(&b.cost));
            scored[..k].sort_by(|a, b| a.cost.total_cmp(&b.cost));
            scored.truncate(k);

            // Convert scored edges into final graph edges.
            edges.extend(scored.into_iter().map(|se| {
                let id = EdgeId(next_id);
                next_id += 1;

                // Resolve the target seed safely via SeedId.
                let to = *right_by_id
                    .get(&se.to)
                    .expect("ScoredEdge::to must refer to a SeedNode in `right`");

                Edge::new(id, src, to, se.cost, se.dt_days)
            }));
        }

        // Optional global cap on total number of edges.
        // Applied after per-left Top-K extraction.
        if let Some(max_e) = max_total {
            if edges.len() > max_e {
                let _ = edges.select_nth_unstable_by(max_e - 1, |a, b| a.cost.total_cmp(&b.cost));
                edges[..max_e].sort_by(|a, b| a.cost.total_cmp(&b.cost));
                edges.truncate(max_e);
            }
        }

        edges
    }

    pub fn compute_features(&self) -> EdgeFeatures {
        let dt = self.dt_days;
        let dt_sq = if dt.is_finite() { dt * dt } else { 0.0 };
        let inv_dt = if dt.is_finite() && dt > 0.0 {
            1.0 / dt
        } else {
            0.0
        };

        // Propagation déterministe du seed `from` à l'époque de `to`.
        let mut px = self.from.plane.pos_xy[0] + self.from.plane.vel_xy[0] * dt;
        let mut py = self.from.plane.pos_xy[1] + self.from.plane.vel_xy[1] * dt;
        // Application de l’accélération si présente.
        let has_acc = if let Some(a) = self.from.plane.acc_xy {
            px += 0.5 * a[0] * dt_sq;
            py += 0.5 * a[1] * dt_sq;
            1.0
        } else {
            0.0
        };

        // Projection du seed cible sur le plan tangent du seed source.
        let p_j = self
            .from
            .plane
            .radec_to_tangent_precomp(self.to.plane.ra_mid, self.to.plane.dec_mid);
        let dx = p_j[0] - px;
        let dy = p_j[1] - py;
        let resid_norm = (dx * dx + dy * dy).sqrt();

        // Propagation diagonale des covariances : Σ̂_i + Σ_pos_j.
        let var_x_from = self.from.plane.cov_pos[0][0] + dt_sq * self.from.plane.cov_vel[0][0];
        let var_y_from = self.from.plane.cov_pos[1][1] + dt_sq * self.from.plane.cov_vel[1][1];
        let var_x = var_x_from + self.to.plane.cov_pos[0][0];
        let var_y = var_y_from + self.to.plane.cov_pos[1][1];
        let var_floor = 1e-20_f64;
        let vx = if var_x.is_finite() && var_x > var_floor {
            var_x
        } else {
            var_floor
        };
        let vy = if var_y.is_finite() && var_y > var_floor {
            var_y
        } else {
            var_floor
        };

        let d2_pos = (dx * dx) / vx + (dy * dy) / vy;
        let log_d2_pos = if d2_pos.is_finite() {
            let eps = 1e-16_f64;
            (d2_pos + eps).ln()
        } else {
            0.0
        };

        // Vitesses prédictives et différences.
        let mut vx_i = self.from.plane.vel_xy[0];
        let mut vy_i = self.from.plane.vel_xy[1];
        if let Some(a) = self.from.plane.acc_xy {
            vx_i += a[0] * dt;
            vy_i += a[1] * dt;
        }
        let speed_from = (vx_i * vx_i + vy_i * vy_i).sqrt();
        let speed_to = (self.to.plane.vel_xy[0].powi(2) + self.to.plane.vel_xy[1].powi(2)).sqrt();
        let speed_diff = (speed_from - speed_to).abs();
        let distance_travelled = speed_from * dt;

        // Covariances et nombre d’observations.
        let n_obs_from = self.from.n_obs as f64;
        let n_obs_to = self.to.n_obs as f64;
        let trace_cov_pos_from = self.from.plane.cov_pos[0][0] + self.from.plane.cov_pos[1][1];
        let trace_cov_pos_to = self.to.plane.cov_pos[0][0] + self.to.plane.cov_pos[1][1];
        let trace_cov_vel_from = self.from.plane.cov_vel[0][0] + self.from.plane.cov_vel[1][1];
        let trace_cov_vel_to = self.to.plane.cov_vel[0][0] + self.to.plane.cov_vel[1][1];

        // Photométrie : différence absolue et z‑score avec un plancher de variance.
        let flux_i = self.from.photom.flux_mean as f64;
        let flux_j = self.to.photom.flux_mean as f64;
        let flux_abs_diff = (flux_j - flux_i).abs();
        let sigma_i = self.from.photom.flux_std as f64;
        let sigma_j = self.to.photom.flux_std as f64;
        let sigma_floor = 1.0_f64;
        let pooled_var = sigma_i.powi(2) + sigma_j.powi(2) + sigma_floor.powi(2);
        let z_flux = if pooled_var > 0.0 {
            flux_abs_diff / pooled_var.sqrt()
        } else {
            0.0
        };
        let flux_std_ratio = if sigma_i > 0.0 {
            sigma_j / sigma_i
        } else {
            0.0
        };

        // Indicateur de partage de bandes.
        let band_shared = if self.from.photom.shares_any_band(&self.to.photom) {
            1.0
        } else {
            0.0
        };

        EdgeFeatures {
            dt_days: dt,
            dt_days_sq: dt_sq,
            inv_dt_days: inv_dt,
            d2_pos,
            log_d2_pos,
            resid_norm,
            resid_dx: dx,
            resid_dy: dy,
            speed_from,
            speed_to,
            speed_diff,
            distance_travelled,
            n_obs_from,
            n_obs_to,
            trace_cov_pos_from,
            trace_cov_pos_to,
            trace_cov_vel_from,
            trace_cov_vel_to,
            flux_abs_diff,
            z_flux,
            flux_std_ratio,
            band_shared,
            has_acc,
        }
    }
}
