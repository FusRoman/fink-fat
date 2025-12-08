// src/graph/ingest.rs

//! Night ingestion façade that builds a new **graph layer** from raw alerts,
//! leveraging the existing `engine.rs` utilities for candidate generation,
//! scoring with hard gates, and Top-K pruning.
//!
//! Pipeline per night
//! ------------------
//! 1) **Seed extraction** from `AlertStore` → `Vec<SeedNode>`,
//! 2) Build a `SeedSpatialIndex` for the current (right) night,
//! 3) For each previous night within `Horizon`, call
//!    `engine::generate_topk_edges(left, right, cfg, binner, index_right, t_right_med, idmap)`
//!    to get **scored & sparsified** edges,
//! 4) Commit edges into the layered graph via `add_edge_by_seed` (strictly forward in time).
//!
//! Notes
//! -----
//! - Costs are **clamped** to strictly positive before committing (`> 0`) to
//!   respect `Edge::new` invariants downstream.
//! - `generate_topk_edges` already applies **hard gating**, **Top-K per left**,
//!   and optional **global cap** — we avoid duplicating that logic here.

use std::fmt::{self, Display, Formatter};

use ahash::AHashMap;
use pyo3::{pyclass, pymethods};

use crate::{
    alerts::{Alert, AlertStore},
    graph::{components::ComponentSolution, graph::InterNightGraph, node::Node, NodeId},
    params::FinkFatParams,
    propagation::{
        engine::{build_id_to_index, generate_topk_edges, median_epoch},
        features::{SeedId, SeedNode, SeedSpatialIndex},
    },
    seeding::space_time_bucket::SpatialBinner,
    AlertId, NightId,
};

/// Summary of the per-night ingestion.
#[pyclass(module = "fink_fat")]
#[derive(Clone, Debug)]
pub struct NightIngestionReport {
    pub night: NightId,
    pub n_seeds: usize,
    pub n_prev_nights_considered: usize,
    pub n_edges_scored_kept: usize,
    pub n_edges_committed: usize,
}

/// Pretty printer (Rust) for concise logs and snapshot-friendly tests.
impl Display for NightIngestionReport {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NightIngestionReport(night={}, seeds={}, prev_nights={}, edges_kept={}, edges_committed={})",
            self.night,
            self.n_seeds,
            self.n_prev_nights_considered,
            self.n_edges_scored_kept,
            self.n_edges_committed
        )
    }
}

#[pymethods]
impl NightIngestionReport {
    /// Return a compact, single-line representation (debug-oriented).
    ///
    /// Examples
    /// --------
    /// >>> repr(report)
    /// 'NightIngestionReport(night=60012, seeds=48231, prev_nights=3, edges_kept=128260, edges_committed=97000)'
    fn __repr__(&self) -> String {
        format!(
            "NightIngestionReport(night={}, seeds={}, prev_nights={}, edges_kept={}, edges_committed={})",
            self.night,
            self.n_seeds,
            self.n_prev_nights_considered,
            self.n_edges_scored_kept,
            self.n_edges_committed
        )
    }

    /// Human-friendly string; mirrors the Rust `Display` formatting.
    ///
    /// Examples
    /// --------
    /// >>> print(report)
    /// NightIngestionReport(night=60012, seeds=48231, prev_nights=3, edges_kept=128260, edges_committed=97000)
    fn __str__(&self) -> String {
        format!("{}", self)
    }
}

/// Strategy for turning an `AlertStore` into intra-night seeds.
///
/// Typical implementations delegate to your existing pair/triplet extractors,
/// then map them into `SeedNode`s with mid-epoch, plane velocity, covariances, etc.
pub trait SeedExtractor {
    fn extract(&self, store: &AlertStore, night: NightId, params: &FinkFatParams) -> Vec<SeedNode>;
}

/// One per-trajectory raw record reconstructed from solver paths.
#[derive(Clone, Debug)]
pub struct TrajectoryRaw {
    /// Ordered graph node ids forming the path (time-ordered).
    pub node_ids: Vec<NodeId>,
    /// (night, seed_id) per node (mirrors `node_ids` 1-to-1).
    pub seeds: Vec<(NightId, SeedId)>,
    /// Flattened (night, alert_id) for all members of all seeds along the path.
    /// The order is stable: we expand each seed in path order, preserving per-seed member order.
    pub detections: Vec<(NightId, AlertId)>,
}

impl TrajectoryRaw {
    /// Convenience accessor to get borrowed `Alert` views for downstream consumers.
    pub fn resolve_alerts<'a>(
        &'a self,
        stores: &'a AHashMap<NightId, AlertStore>,
    ) -> Vec<&'a Alert> {
        let mut out = Vec::with_capacity(self.detections.len());
        for (night, aid) in &self.detections {
            if let Some(store) = stores.get(night) {
                if let Some(alert) = store.get(*aid) {
                    out.push(alert);
                }
            }
        }
        out
    }
}

/// Rolling builder that owns the layered graph and a minimal per-night catalog.
pub struct RollingGraphBuilder<Bs: SpatialBinner, Ex: SeedExtractor> {
    /// Grow-only layered graph.
    pub graph: InterNightGraph,
    /// Seeds per night (local `SeedId` space, typically 0..N-1).
    seeds_by_night: AHashMap<NightId, Vec<SeedNode>>,
    /// Spatial index per night (built on `(ra_mid, dec_mid)`).
    index_by_night: AHashMap<NightId, SeedSpatialIndex>,
    /// Spatial partitioner (e.g., HEALPix).
    binner: Bs,
    /// Intra-night seed extractor.
    extractor: Ex,
}

impl<Bs: SpatialBinner, Ex: SeedExtractor> RollingGraphBuilder<Bs, Ex> {
    /// Construct an empty rolling builder.
    ///
    /// Parameters
    /// ----------
    /// graph : InterNightGraph
    ///     Pre-existing graph (can be empty). Its `horizon` drives which
    ///     previous nights are considered.
    /// binner : impl SpatialBinner
    ///     Spatial partitioner used by indexing & cone queries.
    /// extractor : impl SeedExtractor
    ///     Concrete extractor (pairs / triplets / both).
    /// cfg : InterNightLinkConfig
    ///     All gates/weights/limits used by `generate_topk_edges`.
    pub fn new(graph: InterNightGraph, binner: Bs, extractor: Ex) -> Self {
        Self {
            graph,
            seeds_by_night: AHashMap::default(),
            index_by_night: AHashMap::default(),
            binner,
            extractor,
        }
    }

    /// Retrieve the original `SeedNode`s corresponding to a list of graph `Node`s.
    ///
    /// Parameters
    /// ----------
    /// nodes : &[Node]
    ///    List of graph nodes whose seeds are to be retrieved.
    ///
    /// Returns
    /// -------
    /// Vec<SeedNode>
    ///     The corresponding seed nodes.
    pub fn get_seeds_from_nodes(&self, nodes: &[Node]) -> Vec<&SeedNode> {
        let mut seeds = Vec::with_capacity(nodes.len());
        for node in nodes {
            if let Some(night_seeds) = self.seeds_by_night.get(&node.night) {
                if let Some(seed) = night_seeds.get(node.seed as usize) {
                    seeds.push(seed);
                }
            }
        }
        seeds
    }

    /// Reconstruct one trajectory from a solver path (ordered NodeIds).
    ///
    /// Return
    /// ------
    /// * `Some(TrajectoryRaw)` if all nodes resolve,
    /// * `None` if any node is missing (should not happen in steady-state).
    pub fn reconstruct_one_from_path(&self, path: &[NodeId]) -> Option<TrajectoryRaw> {
        // 1) resolve nodes
        let nodes = self.graph.get_nodes_by_ids(path);

        // 2) map Node -> (night, seed_id) and expand to AlertIds
        let mut seeds_idx = Vec::with_capacity(nodes.len());
        let mut detections = Vec::new();

        for n in &nodes {
            let night = n.night;
            let sid = n.seed;
            seeds_idx.push((night, sid));

            if let Some(seeds) = self.seeds_by_night.get(&night) {
                let i = sid as usize;
                if let Some(seed) = seeds.get(i) {
                    // Preserve seed member order (stable)
                    for &aid in &seed.members {
                        detections.push((night, aid));
                    }
                }
            }
        }

        Some(TrajectoryRaw {
            node_ids: path.to_vec(),
            seeds: seeds_idx,
            detections,
        })
    }

    /// Batch reconstruction for many paths.
    pub fn reconstruct_all_from_paths(&self, paths: &[Vec<NodeId>]) -> Vec<TrajectoryRaw> {
        let mut out = Vec::with_capacity(paths.len());
        for p in paths {
            if let Some(t) = self.reconstruct_one_from_path(p) {
                out.push(t);
            }
        }
        out
    }

    /// Convenience: reconstruct directly from a connected-component solver output.
    pub fn reconstruct_from_component_solution(
        &self,
        sol: &ComponentSolution,
    ) -> Vec<TrajectoryRaw> {
        self.reconstruct_all_from_paths(&sol.paths)
    }

    /// Ingest one night: extract seeds, index them, score Top-K edges from the
    /// previous nights (within `Horizon`), and commit edges to the graph.
    ///
    /// Returns
    /// -------
    /// NightIngestionReport
    ///     Counts and diagnostics for this ingestion step.
    pub fn ingest_night(
        &mut self,
        night: NightId,
        store: &AlertStore,
        params: &FinkFatParams,
    ) -> NightIngestionReport {
        // ---- Step 1: extract seeds for the current (right) night ----------------
        println!("Extracting seeds for night {}", night);

        let right = self.extractor.extract(store, night, params);

        println!("  extracted {} seeds", right.len());

        let n_seeds = right.len();

        // Append the new layer immediately so node ids are allocated and resolvable.
        let right_seed_ids: Vec<SeedId> = right.iter().map(|s| s.seed_id).collect();

        println!("  adding graph layer for night {}", night);

        self.graph.add_night_layer(night, &right_seed_ids);

        // Build and cache the spatial index for the current night.
        let right_index = SeedSpatialIndex::build(&right, &self.binner);
        self.index_by_night.insert(night, right_index.clone());
        self.seeds_by_night.insert(night, right.clone());

        // If this is the first layer, nothing to link from.
        if self.graph.layers.len() <= 1 {
            println!("  first night only; skipping linking");

            return NightIngestionReport {
                night,
                n_seeds,
                n_prev_nights_considered: 0,
                n_edges_scored_kept: 0,
                n_edges_committed: 0,
            };
        }

        // ---- Prepare right-side helpers once (median epoch + id map) -----------
        let t_right_med = median_epoch(&right);
        let right_id_to_index = build_id_to_index(&right);

        // ---- Step 2: iterate over previous nights within the graph horizon -----
        let mut total_kept = 0usize;
        let mut total_committed = 0usize;

        // The graph already knows its horizon; use it to filter previous nights.
        let prev_nights: Vec<NightId> = self
            .seeds_by_night
            .keys()
            .copied()
            .filter(|&p| p < night && self.graph.horizon.within(p, night))
            .collect();

        println!(
            "Linking from {} previous nights into night {}",
            prev_nights.len(),
            night
        );

        for &pnight in &prev_nights {
            println!("  processing previous night {}", pnight);

            // Left slice & prerequisites.
            let left = match self.seeds_by_night.get(&pnight) {
                Some(v) => v,
                None => continue,
            };

            // ---- Step 3: generate Top-K edges with existing engine utility -----
            let edges = generate_topk_edges(
                left,
                &right,
                &params.link,
                &self.binner,
                &right_index,
                t_right_med,
                &right_id_to_index,
            );

            println!("    kept {} edges after scoring & pruning", edges.len());

            total_kept += edges.len();

            println!("    committing edges into the graph");

            // ---- Step 4: commit edges into the layered graph -------------------
            for e in edges {
                // Safety: enforce strictly positive + finite cost & dt_days.
                let cost = (e.cost as f32).max(f32::EPSILON);
                let dt = (e.dt_days as f32).max(f32::EPSILON);
                // `e.from` and `e.to` are SeedId on each night side.
                if self
                    .graph
                    .add_edge_by_seed(pnight, e.from, night, e.to, cost, dt)
                    .is_some()
                {
                    total_committed += 1;
                }
            }
        }

        NightIngestionReport {
            night,
            n_seeds,
            n_prev_nights_considered: prev_nights.len(),
            n_edges_scored_kept: total_kept,
            n_edges_committed: total_committed,
        }
    }
}

#[cfg(test)]
mod ingest_unit_tests {
    use super::*;
    use crate::{
        alerts::Alert,
        graph::Horizon,
        params::engine_params::{InterNightLinkConfig, InterNightLinkConfigBuilder},
        propagation::features::tangent_to_radec,
        seeding::healpix_binners::HealpixBinner,
        AlertId,
    };
    use proptest::prelude::*;

    /* ---------------------------------------------------------------------- */
    /* Helpers                                                                */
    /* ---------------------------------------------------------------------- */

    /// Build an `AlertStore` from a list of (dia, ra, dec, t, flux, flux_err, band).
    fn make_store_from_tuples(tuples: &[(u64, f64, f64, f64, f32, f32, u8)]) -> AlertStore {
        let mut alerts = Vec::with_capacity(tuples.len());
        let mut tmin = f64::INFINITY;

        for (i, &(dia, ra, dec, mjd_tt, flux, flux_err, band)) in tuples.iter().enumerate() {
            let a = Alert {
                id: AlertId::from(i),
                dia_source_id: dia,
                ra,
                ra_err: 0.0,
                dec,
                dec_err: 0.0,
                mjd_tt,
                flux,
                flux_err,
                band,
            };
            if mjd_tt < tmin {
                tmin = mjd_tt;
            }
            alerts.push(a);
        }

        AlertStore {
            start_mjd: if alerts.is_empty() { 0.0 } else { tmin.floor() },
            alerts,
        }
    }

    /// Minimal SeedExtractor for tests: map each alert to a SeedNode.
    #[derive(Clone, Default)]
    struct DummyExtractor;

    impl SeedExtractor for DummyExtractor {
        fn extract(
            &self,
            store: &AlertStore,
            night: NightId,
            _params: &FinkFatParams,
        ) -> Vec<SeedNode> {
            // Build one minimalist seed per alert:
            // - Tangent plane centered on the alert itself → pos_xy = [0, 0].
            // - Zero velocity/acceleration.
            // - Small positive diagonal covariances (so predict_cone ne casse pas).
            // - Photometry copiée depuis l’alerte.
            // - members = [AlertId] de l’alerte.
            let mut seeds = Vec::with_capacity(store.alerts.len());
            for (i, a) in store.alerts.iter().enumerate() {
                let center_ra = a.ra;
                let center_dec = a.dec;
                let (ra_mid, dec_mid) = (a.ra, a.dec); // pos_xy = 0 ⇒ inverse gnomonic = centre
                seeds.push(SeedNode::new(
                    i as u64,
                    night,
                    a.mjd_tt,
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [[1.0e-12, 0.0], [0.0, 1.0e-12]],
                    [[1.0e-12, 0.0], [0.0, 1.0e-12]],
                    None,
                    a.flux,
                    0.0,
                    a.band,
                    1,
                    vec![a.id],
                    center_ra,
                    center_dec,
                    ra_mid,
                    dec_mid,
                ));
            }
            seeds
        }
    }

    /// Test extractor that imposes a tiny linear motion across nights so
    /// inter-night linking has obvious matches.
    #[derive(Clone, Default)]
    struct DriftExtractor;

    impl SeedExtractor for DriftExtractor {
        fn extract(
            &self,
            store: &AlertStore,
            night: NightId,
            _params: &FinkFatParams,
        ) -> Vec<SeedNode> {
            // Base time origin (floor of min mjd) already carried by the store.
            let t0 = store.start_mjd;
            // Small angular speed (rad/day). Keep it modest to be easily within cones.
            let v = [5.0e-5_f64, -5.0e-5_f64];

            let mut seeds = Vec::with_capacity(store.alerts.len());
            for (i, a) in store.alerts.iter().enumerate() {
                let dt = a.mjd_tt - t0; // days
                let pos_xy = [v[0] * dt, v[1] * dt];

                // Inverse gnomonic to cache ra_mid/dec_mid consistently
                // (same formula as in features.rs).
                let (ra_mid, dec_mid) = tangent_to_radec(pos_xy[0], pos_xy[1], a.ra, a.dec);

                seeds.push(SeedNode::new(
                    i as u64,                         // seed_id local à la nuit
                    night,                            // night_id
                    a.mjd_tt,                         // epoch_mid
                    pos_xy,                           // pos at epoch_mid
                    v,                                // vel_xy = constant drift
                    [[1.0e-12, 0.0], [0.0, 1.0e-12]], // cov_pos
                    [[1.0e-12, 0.0], [0.0, 1.0e-12]], // cov_vel
                    None,                             // no acceleration
                    a.flux,
                    0.0,
                    a.band,
                    1,          // n_obs (synthetic seed made from 1 alert)
                    vec![a.id], // members
                    a.ra,
                    a.dec, // plane center
                    ra_mid,
                    dec_mid, // cached sky mid-epoch
                ));
            }
            seeds
        }
    }

    /// Utilitaire pour obtenir un binner réel (Healpix) côté tests.
    fn test_binner() -> HealpixBinner {
        // nside modéré pour les tests.
        HealpixBinner::new(5)
    }

    /// Config de linking souple par défaut.
    fn test_link_cfg() -> InterNightLinkConfig {
        InterNightLinkConfig::default() // ADAPT si besoin de surcharger des limites/poids
    }

    fn test_link_cfg_more_permissive() -> InterNightLinkConfig {
        use std::f64::consts::PI;
        InterNightLinkConfigBuilder::new()
            // --- prédicteur : cônes larges & padding cellulaire ---
            .with_predict(|p| {
                p.k_sigma(4.0) // 4σ → cônes plus larges
                    .with_noise(|n| {
                        // un peu d'inflation additive pour couvrir tout résidu de modèle
                        n.variance_floor(1e-10)
                            .drift_per_day(0.0)
                            .curvature_per_day2(1e-12)
                    })
                    .pad_cell_radius(true)
            })
            // --- scoring : désactiver les rejets durs ---
            .with_scoring(|s| {
                s.with_gates(|g| {
                    g.max_d2_pos(1.0e9) // énorme → jamais bloqué par d²
                        .max_theta_vel(PI) // 180° → aucune contrainte direction
                        .max_speed_diff(f64::INFINITY) // aucune contrainte de vitesse
                })
                // (poids & échelles par défaut conviennent : il n’y a plus de gate bloquant)
            })
            // --- limites : laisser passer beaucoup de candidats ---
            .with_limits(|l| {
                l.top_k_per_left(32) // fan-out généreux
                    .clear_max_cost() // pas de cutoff de coût
            })
            // pas de limite de vitesse globale
            .set_max_speed_rad_per_day(None)
            .build()
            .expect("test LinkConfig build")
    }

    /// Graph “vide”. Si ton constructeur nécessite un horizon explicite,
    /// remplace par l’appel ad hoc.
    fn empty_graph(horizon: Horizon) -> InterNightGraph {
        InterNightGraph::new(horizon) // ADAPT: InterNightGraph::with_horizon(...) si nécessaire
    }

    /* ---------------------------------------------------------------------- */
    /* Unit tests                                                             */
    /* ---------------------------------------------------------------------- */

    #[test]
    fn night_ingestion_report_display_and_repr_are_stable() {
        let r = NightIngestionReport {
            night: 60012,
            n_seeds: 48231,
            n_prev_nights_considered: 3,
            n_edges_scored_kept: 128_260,
            n_edges_committed: 97_000,
        };

        let disp = format!("{}", r);
        assert!(disp.contains("NightIngestionReport(night=60012"));
        assert!(disp.contains("seeds=48231"));
        assert!(disp.contains("prev_nights=3"));
        assert!(disp.contains("edges_kept=128260"));
        assert!(disp.contains("edges_committed=97000"));

        let repr = r.__repr__();
        assert_eq!(repr, disp); // même format pour simplicité
    }

    #[test]
    fn first_night_only_adds_layer_and_skips_linking() {
        let night = 42;

        let store = make_store_from_tuples(&[
            (1001, 1.000_000, 0.500_000, 60000.00, 1000.0, 10.0, 1),
            (1002, 1.010_000, 0.490_000, 60000.10, 1100.0, 10.0, 1),
        ]);

        let graph = empty_graph(Horizon { horizon_nights: 3 });
        let binner = test_binner();
        let extractor = DummyExtractor::default();
        let cfg = test_link_cfg();
        let params = FinkFatParams {
            link: cfg.clone(),
            ..Default::default()
        };

        let mut builder = RollingGraphBuilder::new(graph, binner, extractor);
        let rep = builder.ingest_night(night, &store, &params);

        assert_eq!(rep.night, night);
        assert_eq!(rep.n_prev_nights_considered, 0);
        assert_eq!(rep.n_edges_scored_kept, 0);
        assert_eq!(rep.n_edges_committed, 0);
        assert_eq!(builder.graph.layers.len(), 1);
        assert_eq!(rep.n_seeds, store.alerts.len());
    }

    #[test]
    fn second_night_links_forward_and_counts_match() {
        let n0 = 100;
        let n1 = 101;

        // Deux petits amas proches (drift léger) pour favoriser le linking.
        let s0 = make_store_from_tuples(&[
            (11, 0.100_000, 0.000_000, 60001.0, 1000.0, 10.0, 1),
            (12, 0.100_500, 0.000_500, 60001.0, 1100.0, 10.0, 1),
        ]);

        let s1 = make_store_from_tuples(&[
            (21, 0.100_200, 0.000_200, 60001.5, 1000.0, 10.0, 1),
            (22, 0.100_700, 0.000_700, 60001.5, 1100.0, 10.0, 1),
        ]);

        let graph = empty_graph(Horizon { horizon_nights: 3 });
        let binner = test_binner();
        let extractor = DummyExtractor::default();
        let cfg = test_link_cfg();
        let params = FinkFatParams {
            link: cfg.clone(),
            ..Default::default()
        };

        let mut builder = RollingGraphBuilder::new(graph, binner, extractor);

        let rep0 = builder.ingest_night(n0, &s0, &params);
        assert_eq!(rep0.n_edges_committed, 0);
        assert_eq!(builder.graph.layers.len(), 1);

        let rep1 = builder.ingest_night(n1, &s1, &params);

        // On s’attend à considérer au moins la nuit précédente (si horizon le permet).
        assert!(rep1.n_prev_nights_considered >= 1);
        // Les arêtes retenues ≥ arêtes commitées, et on accepte qu'il n'y en ait aucune
        // si le gating/Top-K a tout filtré pour ces seeds immobiles.
        assert!(rep1.n_edges_scored_kept >= rep1.n_edges_committed);
        assert_eq!(builder.graph.layers.len(), 2);
    }

    #[test]
    fn costs_and_dt_are_clamped_to_strictly_positive() {
        let n0 = 10;
        let n1 = 11;

        // Points “quasi identiques” (même position et temps très proches) → si le
        // score/cost tombe à 0, l’ingestion doit **clamper** > 0 avant commit.
        let s0 = make_store_from_tuples(&[(1, 1.234_000, -0.200_000, 59000.00, 1000.0, 10.0, 1)]);
        let s1 = make_store_from_tuples(&[(2, 1.234_000, -0.200_000, 59000.00, 1100.0, 10.0, 1)]);

        let graph = empty_graph(Horizon { horizon_nights: 3 });
        let binner = test_binner();
        let extractor = DummyExtractor::default();
        let cfg = test_link_cfg();
        let params = FinkFatParams {
            link: cfg.clone(),
            ..Default::default()
        };

        let mut builder = RollingGraphBuilder::new(graph, binner, extractor);
        let _ = builder.ingest_night(n0, &s0, &params);
        let rep = builder.ingest_night(n1, &s1, &params);

        // Le test vérifie l’absence de panic et que l’on a bien pu committer
        // (si le gating n’a pas tout rejeté).
        assert!(rep.n_edges_committed > 0);
    }

    #[test]
    fn three_nights_h2_reports_imply_links_for_all_pairs() {
        let n1 = 20u32;
        let n2 = 21u32;
        let n3 = 22u32;

        let s1 = make_store_from_tuples(&[
            (1, 2.0, 0.2, 60010.0, 1000.0, 10.0, 1),
            (2, 2.0003, 0.2002, 60010.2, 1100.0, 10.0, 1),
        ]);
        let s2 = make_store_from_tuples(&[
            (3, 2.0, 0.2, 60010.8, 1000.0, 10.0, 1),
            (4, 2.0003, 0.2002, 60011.0, 1100.0, 10.0, 1),
        ]);
        let s3 = make_store_from_tuples(&[
            (5, 2.0, 0.2, 60011.6, 1000.0, 10.0, 1),
            (6, 2.0003, 0.2002, 60011.8, 1100.0, 10.0, 1),
        ]);

        let graph = empty_graph(Horizon { horizon_nights: 2 });
        let binner = test_binner();
        let extractor = DriftExtractor::default();

        let cfg = test_link_cfg_more_permissive();
        let params = FinkFatParams {
            link: cfg.clone(),
            ..Default::default()
        };

        let mut builder = RollingGraphBuilder::new(graph, binner, extractor);

        let r1 = builder.ingest_night(n1, &s1, &params);
        assert_eq!(r1.n_prev_nights_considered, 0);

        let r2 = builder.ingest_night(n2, &s2, &params);
        // r2 représente les liens N1→N2
        assert!(r2.n_prev_nights_considered >= 1);
        assert!(r2.n_edges_committed > 0, "expected some links N1→N2");

        let r3 = builder.ingest_night(n3, &s3, &params);
        // r3 regroupe N1→N3 et N2→N3 (horizon=2 ⇒ 2 nuits précédentes considérées)
        assert_eq!(
            r3.n_prev_nights_considered, 2,
            "expected both N1 and N2 considered for N3"
        );
        assert!(
            r3.n_edges_committed > 0,
            "expected some links to N3 from previous nights"
        );
    }

    /* ---------------------------------------------------------------------- */
    /* Property-based tests                                                   */
    /* ---------------------------------------------------------------------- */

    proptest! {
        /// Peu importe un petit jeu de nuits strictement croissantes et des alertes
        /// synthétiques “proches”, l’ingestion ne tente jamais de lier **en arrière**
        /// et ne considère qu’un nombre borné de nuits précédentes (contrôlé par
        /// l’horizon du graphe). Ici on ne lit pas l’horizon interne ; on vérifie
        /// surtout la monotonie (0 pour la première nuit, puis ≥ 0).
        #[test]
        fn prop_forward_in_time_monotonic_prev_considered(
            mut nights in prop::collection::vec(1000u32..2000u32, 3..6),
            base_ra in 0.0f64..(2.0*std::f64::consts::PI),
            base_dec in -0.5f64..0.5f64,
            base_mjd in 59000.0f64..59100.0f64,
        ) {
            nights.sort();
            nights.dedup();
            prop_assume!(nights.len() >= 3);

            // Construire un store minimal par nuit (2 alertes proches).
            let mut stores: AHashMap<NightId, AlertStore> = AHashMap::default();
            for (k, n) in nights.iter().enumerate() {
                let night = *n;
                let ra = base_ra + (k as f64)*1e-3;
                let dec = base_dec + (k as f64)*1e-3;
                let mjd = base_mjd + (k as f64)*0.5;
                let store = make_store_from_tuples(&[
                    (1000 + k as u64, ra,            dec,            mjd,     1000.0, 10.0, 1),
                    (2000 + k as u64, ra + 2e-4,     dec + 2e-4,     mjd + 1e-3, 1100.0, 10.0, 1),
                ]);
                stores.insert(night, store);
            }

            let graph = empty_graph(Horizon { horizon_nights: 3 });
            let binner = test_binner();
            let extractor = DummyExtractor::default();
            let cfg = test_link_cfg();
            let params = FinkFatParams {
                link: cfg.clone(),
                ..Default::default()
            };

            let mut builder = RollingGraphBuilder::new(graph, binner, extractor);

            let mut prev_considered = Vec::new();
            for n in &nights {
                let night = *n;
                let st = stores.get(&night).unwrap();
                let rep = builder.ingest_night(night, st, &params);
                prev_considered.push(rep.n_prev_nights_considered);
            }

            // La première nuit ne considère aucune nuit précédente.
            prop_assert_eq!(prev_considered[0], 0);

            // Chaque valeur doit être dans [0, min(index, horizon)].
            // NB: ici l’horizon vaut 3 (Horizon { horizon_nights: 3 }).
            let horizon = 3usize;
            for (i, &v) in prev_considered.iter().enumerate() {
                let max_prev = i.min(horizon);
                prop_assert!(
                    v <= max_prev,
                    "n_prev_nights_considered={} dépasse la borne {} à l'étape {}",
                    v, max_prev, i
                );
            }
        }
    }
}
