// src/propagation/flow.rs

//! Global multi-night linking via **Min-Cost Flow (MCF)**.
//!
//! # Overview
//! This module declares the **data model** and **public method signatures** for a
//! time-expanded **min-cost flow** optimizer that links intra-night seeds into
//! globally consistent multi-night trajectories. It complements the existing
//! bipartite solvers by providing *global optimality* across several nights.
//!
//! ## Integration points
//! - `features::{SeedId, SeedNode}`: seed state and meta.
//! - `solver::Edge`: sparse candidate links with costs (Top-K).
//! - `track_registry::TrackRegistry`: final trajectory materialization and
//!   conflict handling via the registry policy.
//! - `errors`: typed error plumbing (`FlowError` defined here, convertible
//!   to/from crate errors later).
//!
//! ## Scope
//! **No solver logic here**: only data structures + method signatures to build
//! a time-expanded graph and to push solutions into the `TrackRegistry`.

use ahash::AHashMap;
use pyo3::{pyclass, pymethods};

use crate::params::min_cost_flow_params::MinCostFlowConfig;
use crate::propagation::features::{SeedId, SeedNode};
use crate::propagation::solver::Edge;
use crate::track_registry::{TrackRegistry, TrajectoryId};
use crate::NightId;

/// Compact index for nodes in the internal flow graph.
pub type NodeId = u32;

/// Compact index for arcs in the internal flow graph.
pub type ArcId = u32;

/* -------------------------------------------------------------------------- */
/*  Errors                                                                     */
/* -------------------------------------------------------------------------- */

/// Min-Cost Flow specific error kind.
///
/// Notes
/// -----
/// Keep it local for now; you can later unify with `crate::errors` via `From`/`Into`.
#[derive(thiserror::Error, Debug)]
pub enum FlowError {
    /// The requested layer (night) is not present in the problem.
    #[error("unknown night layer: {0}")]
    UnknownNight(NightId),

    /// The `(night, seed)` address is unknown or not indexed.
    #[error("unknown seed key: night={night}, seed={seed}")]
    UnknownSeed { night: NightId, seed: SeedId },

    /// Graph structure is inconsistent (e.g., missing Source/Sink).
    #[error("invalid graph structure: {0}")]
    InvalidGraph(&'static str),

    /// Solver backend reported an error (string forwarded as-is).
    #[error("solver backend failure: {0}")]
    Backend(&'static str),

    /// Registry integration error (conflict, policy violation, etc.).
    #[error("registry update failed: {0}")]
    Registry(&'static str),
}

/* -------------------------------------------------------------------------- */
/*  Seed addressing across nights                                             */
/* -------------------------------------------------------------------------- */

/// Address of a seed in the time-expanded graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SeedKey {
    /// Night the seed belongs to.
    pub night: NightId,
    /// Per-night (or global) seed identifier.
    pub seed: SeedId,
}

/* -------------------------------------------------------------------------- */
/*  Public configuration                                                      */
/* -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- */
/*  Graph containers (problem side)                                           */
/* -------------------------------------------------------------------------- */

/// Type of an arc in the flow graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArcKind {
    /// Source → Seed
    Start,
    /// Seed → Seed (possibly Δ>1 nights)
    Link,
    /// Seed → Sink
    End,
}

/// Immutable arc metadata used by solvers and diagnostics.
#[derive(Clone, Debug)]
pub struct FlowArc {
    /// Unique arc index.
    pub id: ArcId,
    /// Origin node id.
    pub from: NodeId,
    /// Destination node id.
    pub to: NodeId,
    /// Non-negative cost (lower is better).
    pub cost: f64,
    /// Capacity (usually 1).
    pub capacity: u32,
    /// Optional time gap in **days** (diagnostics only).
    pub dt_days: f64,
    /// Structural type (start/link/end).
    pub kind: ArcKind,
    /// Optional seed linkage info for debugging/exports.
    pub left: Option<SeedKey>,
    pub right: Option<SeedKey>,
}

/// Immutable node metadata used by solvers and diagnostics.
#[derive(Clone, Debug)]
pub struct FlowNode {
    /// Unique node index.
    pub id: NodeId,
    /// If this node represents a seed, its address. `None` for Source/Sink.
    pub seed: Option<SeedKey>,
}

/// A time layer containing all nodes (seeds) for a given night.
#[derive(Clone, Debug)]
pub struct NightLayer {
    pub night: NightId,
    /// Node ids corresponding to the seeds of this night (stable order).
    pub node_ids: Vec<NodeId>,
}

/* -------------------------------------------------------------------------- */
/*  Problem assembly                                                          */
/* -------------------------------------------------------------------------- */

/// A complete MCF instance ready for solving.
#[derive(Clone, Debug)]
pub struct FlowProblem {
    /// Global configuration (penalties, limits).
    pub cfg: MinCostFlowConfig,
    /// Global super-source node id.
    pub source: NodeId,
    /// Global super-sink node id.
    pub sink: NodeId,
    /// All nodes, including Source/Sink and per-seed nodes.
    pub nodes: Vec<FlowNode>,
    /// All arcs (start/link/end).
    pub arcs: Vec<FlowArc>,
    /// Layers in chronological order (strictly increasing `night`).
    pub layers: Vec<NightLayer>,
    /// Mapping `(night, seed) → NodeId` for quick lookups.
    pub index_of: AHashMap<SeedKey, NodeId>,
}

impl Default for FlowProblem {
    fn default() -> Self {
        Self::new(MinCostFlowConfig::default())
    }
}

impl FlowProblem {
    /// Create an empty problem with Source/Sink nodes allocated.
    ///
    /// Notes
    /// -----
    /// Source has `NodeId=0`, Sink has `NodeId=1`. Seed nodes start at 2.
    pub fn new(cfg: MinCostFlowConfig) -> Self {
        // Source (id = 0)
        // Sink (id = 1)
        let nodes = vec![
            FlowNode { id: 0, seed: None },
            FlowNode { id: 1, seed: None },
        ];

        FlowProblem {
            cfg,
            source: 0,
            sink: 1,
            nodes,
            arcs: Vec::new(),
            layers: Vec::new(),
            index_of: AHashMap::new(),
        }
    }

    /// Add a **night layer** with one node per seed. Returns the `NightLayer` index.
    ///
    /// Parameters
    /// ----------
    /// night : NightId
    ///     Night identifier for this layer.
    /// seeds : &[SeedNode]
    ///     The seeds extracted for this night (only metadata used here).
    pub fn add_layer(&mut self, night: NightId, seeds: &[SeedNode]) -> Result<usize, FlowError> {
        // (Optional) Enforce strict monotonicity if desired:
        // if let Some(prev) = self.layers.last() {
        //     if night <= prev.night {
        //         return Err(FlowError::InvalidGraph("layers must be strictly time-ordered"));
        //     }
        // }

        let mut node_ids = Vec::with_capacity(seeds.len());

        for s in seeds {
            // Allocate a new node id.
            let nid = self
                .nodes
                .len()
                .try_into()
                .expect("number of nodes fits in u32");
            let seed_key = SeedKey {
                night,
                seed: s.seed_id, // SeedId is u64 in features.rs
            };

            // Register node & index
            self.nodes.push(FlowNode {
                id: nid,
                seed: Some(seed_key),
            });

            self.index_of.insert(seed_key, nid);
            node_ids.push(nid);
        }

        let layer = NightLayer { night, node_ids };
        self.layers.push(layer);
        Ok(self.layers.len() - 1)
    }

    /// Add **start arcs** from Source to *all* seeds of a given layer.
    /// Returns the number of arcs created.
    pub fn add_start_arcs(&mut self, layer_idx: usize) -> Result<usize, FlowError> {
        let layer = self
            .layers
            .get(layer_idx)
            .ok_or(FlowError::InvalidGraph("layer index out of bounds"))?;
        let mut created = 0usize;

        for &node_id in &layer.node_ids {
            let seed_key = self.nodes[node_id as usize]
                .seed
                .expect("seed nodes must carry a SeedKey");

            let aid = self
                .arcs
                .len()
                .try_into()
                .expect("number of arcs fits in u32");

            self.arcs.push(FlowArc {
                id: aid,
                from: self.source,
                to: node_id,
                cost: self.cfg.lambda_start,
                capacity: 1,
                dt_days: 0.0,
                kind: ArcKind::Start,
                left: None,
                right: Some(seed_key),
            });
            created += 1;
        }
        Ok(created)
    }

    /// Add **end arcs** from *all* seeds of a given layer to Sink.
    /// Returns the number of arcs created.
    pub fn add_end_arcs(&mut self, layer_idx: usize) -> Result<usize, FlowError> {
        let layer = self
            .layers
            .get(layer_idx)
            .ok_or(FlowError::InvalidGraph("layer index out of bounds"))?;
        let mut created = 0usize;

        for &node_id in &layer.node_ids {
            let seed_key = self.nodes[node_id as usize]
                .seed
                .expect("seed nodes must carry a SeedKey");

            let aid = self
                .arcs
                .len()
                .try_into()
                .expect("number of arcs fits in u32");

            self.arcs.push(FlowArc {
                id: aid,
                from: node_id,
                to: self.sink,
                cost: self.cfg.lambda_end,
                capacity: 1,
                dt_days: 0.0,
                kind: ArcKind::End,
                left: Some(seed_key),
                right: None,
            });
            created += 1;
        }
        Ok(created)
    }

    /// Add **link arcs** between two layers using scored candidate edges.
    ///
    /// Parameters
    /// ----------
    /// left_night : NightId
    ///     Source night.
    /// right_night : NightId
    ///     Target night.
    /// edges : &[Edge]
    ///     Sparse candidate set with costs (from scoring / engine Top-K).
    ///
    /// Returns
    /// -------
    /// usize
    ///     Number of arcs created.
    pub fn add_link_arcs(
        &mut self,
        left_night: NightId,
        right_night: NightId,
        edges: &[Edge],
    ) -> Result<usize, FlowError> {
        let mut created = 0usize;

        for e in edges {
            let lk = SeedKey {
                night: left_night,
                seed: e.from,
            };
            let rk = SeedKey {
                night: right_night,
                seed: e.to,
            };

            let &from_node = self.index_of.get(&lk).ok_or(FlowError::UnknownSeed {
                night: left_night,
                seed: e.from,
            })?;
            let &to_node = self.index_of.get(&rk).ok_or(FlowError::UnknownSeed {
                night: right_night,
                seed: e.to,
            })?;

            let aid = self
                .arcs
                .len()
                .try_into()
                .expect("number of arcs fits in u32");

            // NOTE: Cost is taken directly from Edge; any additional penalties
            // (gap weights, hop limits, etc.) should be applied upstream or
            // encoded directly in `Edge::cost` by the scorer/engine.
            self.arcs.push(FlowArc {
                id: aid,
                from: from_node,
                to: to_node,
                cost: e.cost,
                capacity: 1,
                dt_days: e.dt_days,
                kind: ArcKind::Link,
                left: Some(lk),
                right: Some(rk),
            });
            created += 1;
        }

        Ok(created)
    }
}

/* -------------------------------------------------------------------------- */
/*  Solver interface (pluggable backends)                                     */
/* -------------------------------------------------------------------------- */

/// Result of a min-cost flow solve on the time-expanded graph.
#[derive(Clone, Debug)]
pub struct FlowSolution {
    /// Total flow pushed from Source to Sink (number of trajectories).
    pub total_flow: u32,
    /// Set of **active arcs** carrying unit flow (subset of `FlowProblem::arcs`).
    pub active_arcs: Vec<ArcId>,
    /// Convenience mapping: for each `SeedKey`, its chosen **successor** (if any).
    pub succ_of: AHashMap<SeedKey, SeedKey>,
    /// Convenience mapping: for each `SeedKey`, its chosen **predecessor** (if any).
    pub pred_of: AHashMap<SeedKey, SeedKey>,
}

impl FlowSolution {
    /// Extract **disjoint source→sink paths** as ordered lists of `SeedKey`s.
    ///
    /// Each path corresponds to a single trajectory across nights.
    pub fn paths(&self) -> Vec<Vec<SeedKey>> {
        use ahash::AHashSet;

        // 1) Collect all nodes that appear either as keys or values in succ/pred.
        let mut all: AHashSet<SeedKey> = AHashSet::new();
        for (&k, &v) in &self.succ_of {
            all.insert(k);
            all.insert(v);
        }
        for (&k, &v) in &self.pred_of {
            all.insert(k);
            all.insert(v);
        }

        // 2) Heads = nodes that are NOT a key in pred_of (i.e., no predecessor).
        //    (We also handle isolated nodes later.)
        let mut heads: Vec<SeedKey> = Vec::new();
        for &node in &all {
            if !self.pred_of.contains_key(&node) {
                heads.push(node);
            }
        }

        // 3) Reconstruct paths by walking succ from each head.
        let mut visited: AHashSet<SeedKey> = AHashSet::new();
        let mut paths: Vec<Vec<SeedKey>> = Vec::new();

        for head in heads {
            if visited.contains(&head) {
                continue;
            }
            let mut path: Vec<SeedKey> = Vec::new();
            let mut cur = head;

            // walk forward
            while !visited.contains(&cur) {
                visited.insert(cur);
                path.push(cur);

                if let Some(&nxt) = self.succ_of.get(&cur) {
                    // Stop if we detect a cycle (should not happen in a valid time-expanded flow)
                    if visited.contains(&nxt) {
                        // Break to avoid infinite loop and keep the partial path.
                        break;
                    }
                    cur = nxt;
                } else {
                    break; // end of chain
                }
            }

            if !path.is_empty() {
                paths.push(path);
            }
        }

        // 4) Add singletons that have no pred and no succ (not discovered above),
        //    or any remaining nodes not visited for some reason.
        for &node in &all {
            if !visited.contains(&node) {
                // Node not part of any chain we walked.
                // Treat it as a singleton path.
                paths.push(vec![node]);
                visited.insert(node);
            }
        }

        // Optional: stable ordering by (night, seed) of the head (helps determinism for exports)
        paths.sort_by(|a, b| {
            let la = a.first().unwrap();
            let lb = b.first().unwrap();
            (la.night, la.seed).cmp(&(lb.night, lb.seed))
        });

        paths
    }

    /// Apply the chosen policy and update the `TrackRegistry`.
    ///
    /// Notes
    /// -----
    /// - This routine **only** ensures/merge trajectories per **seed key** along each path
    ///   using the DSU/union in `TrackRegistry`.
    /// - It does **not** assign member detections, because that requires `SeedNode` and
    ///   `AlertStore` (available in nightly snapshots). Do that in a higher level step
    ///   when you have the `NightSnapshot`s at hand.
    pub fn into_registry(
        &self,
        registry: &mut TrackRegistry,
        // You can add a `policy` parameter here if required by your registry
    ) -> Result<Vec<TrajectoryId>, FlowError> {
        let paths = self.paths();
        let mut reps: Vec<TrajectoryId> = Vec::with_capacity(paths.len());

        for path in paths {
            if path.is_empty() {
                continue;
            }

            // Ensure a trajectory for the first seed of the path.
            let mut keep = registry.ensure_seed_traj(crate::track_registry::SeedKey {
                night_id: path[0].night,
                seed_id: path[0].seed,
            });

            // Merge along the path (keep = union(keep, next)).
            for sk in path.iter().skip(1) {
                let tid = registry.ensure_seed_traj(crate::track_registry::SeedKey {
                    night_id: sk.night,
                    seed_id: sk.seed,
                });
                keep = registry.merge_traj(keep, tid);
            }

            reps.push(keep);
        }

        Ok(reps)
    }
}

/* -------------------------------------------------------------------------- */
/*  Builder / orchestration                                                   */
/* -------------------------------------------------------------------------- */

/// Minimal summary of a layer added to the problem; useful for orchestration.
#[derive(Clone, Debug)]
pub struct LayerSummary {
    pub night: NightId,
    pub n_seeds: usize,
    pub n_start_arcs: usize,
    pub n_end_arcs: usize,
}

/// Facade to build/update a `FlowProblem` as nights arrive.
/// This is the MCF counterpart of the pairwise engine wiring.
#[derive(Clone, Debug, Default)]
pub struct FlowBuilder {
    pub pb: FlowProblem,
}

impl FlowBuilder {
    /// Create a new builder around an empty `FlowProblem`.
    ///
    /// Notes
    /// -----
    /// The underlying `FlowProblem::new` allocates Source (id=0) and Sink (id=1).
    /// Seed nodes will start at id=2 as layers are ingested.
    pub fn new(cfg: MinCostFlowConfig) -> Self {
        Self {
            pb: FlowProblem::new(cfg),
        }
    }

    /// Convenience: ingest a **pre-extracted** slice of seeds for a given night.
    ///
    /// This will:
    /// - append a new layer for `night`,
    /// - add one **Start** arc per seed (Source → seed),
    /// - add one **End** arc per seed (seed → Sink),
    ///   and return counts only (no solver call here).
    ///
    /// Parameters
    /// ----------
    /// night : NightId
    ///     The night identifier for this layer (must be strictly increasing
    ///     if you enforce monotonic layers in `FlowProblem::add_layer`).
    /// seeds : &[SeedNode]
    ///     The seeds for that night (only their ids & meta are used here).
    pub fn ingest_night(
        &mut self,
        night: NightId,
        seeds: &[SeedNode],
    ) -> Result<LayerSummary, FlowError> {
        let layer_idx = self.pb.add_layer(night, seeds)?;
        let n_start = self.pb.add_start_arcs(layer_idx)?;
        let n_end = self.pb.add_end_arcs(layer_idx)?;

        Ok(LayerSummary {
            night,
            n_seeds: seeds.len(),
            n_start_arcs: n_start,
            n_end_arcs: n_end,
        })
    }

    /// Convenience: add link arcs for a given night pair using Top-K edges.
    ///
    /// This is a thin wrapper over `FlowProblem::add_link_arcs`. It assumes that
    /// both layers `(left_night, right_night)` have already been ingested via
    /// [`ingest_night`](crate::propagation::flow::FlowBuilder::ingest_night).
    /// If a seed is unknown, you will get `FlowError::UnknownSeed`.
    pub fn add_links_between(
        &mut self,
        left_night: NightId,
        right_night: NightId,
        edges: &[Edge],
    ) -> Result<usize, FlowError> {
        self.pb.add_link_arcs(left_night, right_night, edges)
    }
}

/* -------------------------------------------------------------------------- */
/*  High-level orchestration signatures                                       */
/* -------------------------------------------------------------------------- */

/// Result of running an incremental MCF update after ingesting a new night.
#[pyclass(module = "fink_fat")]
#[derive(Clone, Debug)]
pub struct FlowUpdate {
    pub night: NightId,
    pub n_layers: usize,
    pub total_nodes: usize,
    pub total_arcs: usize,
    pub total_flow: u32,
    /// Extracted trajectories as `SeedKey` paths (order preserved).
    pub trajectories: Vec<Vec<SeedKey>>,
}

fn format_number_underscore<T: ToString>(n: T, sep: &str) -> String {
    let s = n.to_string();
    // coupe en blocs de 3 en partant de la fin, puis rejoint avec "_"
    s.as_bytes()
        .rchunks(3)
        .rev()
        .map(|c| std::str::from_utf8(c).unwrap())
        .collect::<Vec<_>>()
        .join(sep)
}

#[pymethods]
impl FlowUpdate {
    /// Number of trajectories extracted in this update.
    #[getter]
    pub fn n_trajectories(&self) -> usize {
        self.trajectories.len()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "FlowUpdate(night={}, n_layers={}, total_nodes={}, total_arcs={}, total_flow={}, n_trajectories={})",
            self.night,
            format_number_underscore(self.n_layers, "_"),
            format_number_underscore(self.total_nodes, "_"),
            format_number_underscore(self.total_arcs, "_"),
            format_number_underscore(self.total_flow, "_"),
            format_number_underscore(self.n_trajectories(), "_"),
        )
    }
}

/// Generic interface for **min-cost flow** backends.
pub trait MinCostFlowSolver {
    /// Solve the given flow problem and return the set of active arcs and path maps.
    ///
    /// Implementations should:
    /// - respect capacities (unit flow on arcs),
    /// - honor `cfg.max_total_flow` if provided,
    /// - guarantee **conservation** at internal nodes,
    /// - minimize the **total cost**.
    fn solve(&self, pb: &FlowProblem) -> Result<FlowSolution, FlowError>;
}

/// A trivial Min-Cost Flow solver that performs **no optimization**.
///
/// This backend is useful to **exercise graph construction** at scale (layers,
/// nodes, link arcs), collect builder/runtime metrics, and validate the
/// end-to-end plumbing without paying any solving cost.
///
/// Behavior
/// --------
/// * Returns a `FlowSolution` with:
///   - `total_flow = 0`,
///   - `active_arcs = []`,
///   - empty successor/predecessor maps (`succ_of`, `pred_of`).
/// * Never fails on a well-formed `FlowProblem`.
///
/// Notes
/// -----
/// Downstream extractors (`solve_and_extract`) should handle this gracefully,
/// typically yielding **zero trajectories** while still exposing the **graph
/// counts** (layers/nodes/arcs) that you want to benchmark.
///
/// See also
/// --------
/// * [`MinCostFlowSolver`] – trait implemented by real backends (SSP, cost-scaling, …).
/// * [`FlowProblem`] / [`FlowSolution`] – graph model and solution container.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullFlowSolver;

impl MinCostFlowSolver for NullFlowSolver {
    fn solve(&self, _pb: &FlowProblem) -> Result<FlowSolution, FlowError> {
        // We deliberately **do not** activate any arc and we do not build paths.
        // This keeps the solution consistent but with zero flow.
        Ok(FlowSolution {
            total_flow: 0,
            active_arcs: Vec::new(),
            succ_of: ahash::AHashMap::new(),
            pred_of: ahash::AHashMap::new(),
        })
    }
}

/// High-level entry-point to **solve** the current flow and extract trajectories.
///
/// This is the MCF analogue to a “stitch” step, but driven by global optimality.
/// The function is pure w.r.t. the builder: it does not mutate the `FlowBuilder`
/// nor its underlying `FlowProblem`.
pub fn solve_and_extract<S: MinCostFlowSolver>(
    builder: &FlowBuilder,
    solver: &S,
) -> Result<FlowUpdate, FlowError> {
    // 1) Solve the current time-expanded min-cost flow.
    let sol = solver.solve(&builder.pb)?;

    // 2) Reconstruct disjoint Source→Sink paths as ordered SeedKey sequences.
    let trajectories = sol.paths();

    // 3) Populate a compact update summary for orchestration/telemetry.
    let night = builder.pb.layers.last().map(|ly| ly.night).unwrap_or(0);

    let upd = FlowUpdate {
        night,
        n_layers: builder.pb.layers.len(),
        total_nodes: builder.pb.nodes.len(),
        total_arcs: builder.pb.arcs.len(),
        total_flow: sol.total_flow,
        trajectories,
    };

    Ok(upd)
}
