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

    /// Internal error (should not happen).
    #[error("internal error: {0}")]
    Internal(&'static str),
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

    /* ------------------- NEW: zero-copy CSR adjacencies ------------------- */
    /// CSR offsets for outgoing arcs; length = nodes.len() + 1.
    /// Outgoing arcs of node `u` are in `out_adj[out_off[u] .. out_off[u+1]]`.
    pub out_off: Vec<u32>,
    /// Flat array of arc ids for outgoing adjacency.
    pub out_adj: Vec<ArcId>,

    /// CSR offsets for incoming arcs; length = nodes.len() + 1.
    /// Incoming arcs of node `v` are in `in_adj[in_off[v] .. in_off[v+1]]`.
    pub in_off: Vec<u32>,
    /// Flat array of arc ids for incoming adjacency.
    pub in_adj: Vec<ArcId>,

    /// Internal flag: CSR indices are up-to-date w.r.t. `arcs` and `nodes`.
    csr_ready: bool,
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
            out_off: Vec::new(),
            out_adj: Vec::new(),
            in_off: Vec::new(),
            in_adj: Vec::new(),
            csr_ready: false,
        }
    }

    /// Mark CSR indices invalid after any graph mutation.
    #[inline]
    fn invalidate_csr(&mut self) {
        self.csr_ready = false;
    }

    /// Build CSR indices from current `arcs`. O(E) time, O(E) memory.
    pub fn rebuild_csr(&mut self) {
        let n = self.nodes.len();
        let e = self.arcs.len();

        self.out_off.clear();
        self.out_adj.clear();
        self.in_off.clear();
        self.in_adj.clear();

        self.out_off.resize(n + 1, 0);
        self.in_off.resize(n + 1, 0);
        self.out_adj.resize(e, 0);
        self.in_adj.resize(e, 0);

        // 1) Degree counts
        for a in &self.arcs {
            self.out_off[a.from as usize] += 1;
            self.in_off[a.to as usize] += 1;
        }
        // 2) Prefix sums → offsets
        let mut acc = 0u32;
        for x in self.out_off.iter_mut() {
            let c = *x;
            *x = acc;
            acc += c;
        }
        let mut acc2 = 0u32;
        for x in self.in_off.iter_mut() {
            let c = *x;
            *x = acc2;
            acc2 += c;
        }

        // 3) Fill adjacency using cursors
        let mut cur_out = self.out_off.clone();
        let mut cur_in = self.in_off.clone();
        for a in &self.arcs {
            let i = cur_out[a.from as usize] as usize;
            self.out_adj[i] = a.id;
            cur_out[a.from as usize] += 1;

            let j = cur_in[a.to as usize] as usize;
            self.in_adj[j] = a.id;
            cur_in[a.to as usize] += 1;
        }

        self.csr_ready = true;
    }

    /// Ensure CSR indices exist; rebuild lazily if not ready.
    #[inline]
    pub fn ensure_csr(&mut self) {
        if !self.csr_ready {
            self.rebuild_csr();
        }
    }

    /// Slice of outgoing arcs for node `u`.
    #[inline]
    pub fn out_arcs(&self, u: NodeId) -> &[ArcId] {
        debug_assert!(self.csr_ready, "call ensure_csr() before using CSR slices");
        let u = u as usize;
        &self.out_adj[self.out_off[u] as usize..self.out_off[u + 1] as usize]
    }

    /// Slice of incoming arcs for node `v`.
    #[inline]
    pub fn in_arcs(&self, v: NodeId) -> &[ArcId] {
        debug_assert!(self.csr_ready, "call ensure_csr() before using CSR slices");
        let v = v as usize;
        &self.in_adj[self.in_off[v] as usize..self.in_off[v + 1] as usize]
    }

    /// Remove all **Start** arcs that target any node in `layer_idx`.
    pub fn remove_start_arcs_in_layer(&mut self, layer_idx: usize) -> Result<usize, FlowError> {
        let layer = self
            .layers
            .get(layer_idx)
            .ok_or(FlowError::InvalidGraph("layer index out of bounds"))?;
        let node_mask: ahash::AHashSet<NodeId> = layer.node_ids.iter().copied().collect();

        let mut kept = Vec::with_capacity(self.arcs.len());
        let mut removed = 0usize;
        for a in self.arcs.drain(..) {
            let is_start_here = a.kind == ArcKind::Start && node_mask.contains(&a.to);
            if is_start_here {
                removed += 1;
            } else {
                kept.push(a);
            }
        }
        // Reassign arc ids for stability.
        for (i, a) in kept.iter_mut().enumerate() {
            a.id = i as u32;
        }
        self.arcs = kept;
        self.invalidate_csr();
        Ok(removed)
    }

    /// Remove all **End** arcs that leave any node in `layer_idx`.
    pub fn remove_end_arcs_in_layer(&mut self, layer_idx: usize) -> Result<usize, FlowError> {
        let layer = self
            .layers
            .get(layer_idx)
            .ok_or(FlowError::InvalidGraph("layer index out of bounds"))?;
        let node_mask: ahash::AHashSet<NodeId> = layer.node_ids.iter().copied().collect();

        let mut kept = Vec::with_capacity(self.arcs.len());
        let mut removed = 0usize;
        for a in self.arcs.drain(..) {
            let is_end_here = a.kind == ArcKind::End && node_mask.contains(&a.from);
            if is_end_here {
                removed += 1;
            } else {
                kept.push(a);
            }
        }
        // Reassign arc ids for stability.
        for (i, a) in kept.iter_mut().enumerate() {
            a.id = i as u32;
        }
        self.arcs = kept;
        self.invalidate_csr();
        Ok(removed)
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
        self.invalidate_csr(); // NEW
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

        self.invalidate_csr(); // NEW
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

        self.invalidate_csr(); // NEW
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

        self.invalidate_csr(); // NEW
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
        // 1) Add the layer.
        let new_idx = self.pb.add_layer(night, seeds)?;

        // 2) Compute indices.
        let n_layers = self.pb.layers.len();
        let last_idx = new_idx;
        let prev_last_idx = last_idx.checked_sub(1);

        // 3) We will count only the arcs created as part of this update (for telemetry).
        let mut n_start_created = 0usize;
        let mut n_end_created = 0usize;

        // 4) If there was a previous last layer, make it **startable** now,
        //    and ensure it no longer ends directly into Sink.
        if let Some(prev_idx) = prev_last_idx {
            // Remove old End arcs from the previous last (N_{k} → Sink).
            let _removed = self.pb.remove_end_arcs_in_layer(prev_idx)?;
            // Add Start arcs from Source to N_{k} (if not already there).
            // (Idempotent: add_start_arcs simply adds one per seed.)
            n_start_created += self.pb.add_start_arcs(prev_idx)?;
        }

        // 5) On the current last layer:
        //    - do NOT add Start arcs (forbid Source → N_last → Sink),
        //    - add End arcs only if we have at least two layers (N_{k-1} exists).
        if n_layers >= 2 {
            n_end_created += self.pb.add_end_arcs(last_idx)?;
        } else {
            // With a single layer, ensure no stray Start/End arcs exist.
            // (If you never added any before on layer 0, this is a no-op.)
            let _ = self.pb.remove_start_arcs_in_layer(last_idx)?;
            let _ = self.pb.remove_end_arcs_in_layer(last_idx)?;
        }

        Ok(LayerSummary {
            night,
            n_seeds: seeds.len(),
            n_start_arcs: n_start_created,
            n_end_arcs: n_end_created,
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
    builder: &mut FlowBuilder,
    solver: &S,
) -> Result<FlowUpdate, FlowError> {
    // Ensure CSR indices exist (lazy build)
    builder.pb.ensure_csr(); // NEW

    // 1) Solve the current time-expanded min-cost flow.
    let sol = solver.solve(&builder.pb)?;

    println!(
        "[MCF] Solved flow: total_flow = {}, active_arcs = {}",
        sol.total_flow,
        sol.active_arcs.len()
    );

    println!("Making paths...");

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

#[cfg(test)]
mod flow_graph_tests {
    use super::*;
    use crate::propagation::features::{SeedId, SeedNode};
    use crate::propagation::solver::Edge;
    use proptest::prelude::*;

    /* ------------------------------- Utilities ------------------------------- */

    /// Minimal deterministic seed fabric.
    ///
    /// This constructs a SeedNode whose tangent plane is centered at `(center_ra, center_dec)`,
    /// with a simple kinematics model. We keep epochs close so that the predicted cone
    /// overlaps the counterpart on the right night under permissive gating.
    #[allow(clippy::too_many_arguments)]
    fn mk_seed(seed_id: u64) -> SeedNode {
        SeedNode::new(
            seed_id,
            0,
            0.0,
            [0.0, 0.0],
            [0.0, 0.0],
            // unit-ish covariances (diagonal) to avoid degenerate scoring
            [[1e-12, 0.0], [0.0, 1e-12]],
            [[1e-12, 0.0], [0.0, 1e-12]],
            None,
            1000.0,
            10.0,
            0,
            2,
            Vec::new(), // not used in these tests
            0.0,
            0.0,
            0.0,
            0.0,
        )
    }

    fn mk_seeds(base: SeedId, n: usize) -> Vec<SeedNode> {
        (0..n).map(|i| mk_seed(base + i as SeedId)).collect()
    }

    fn arcs_of_kind(pb: &FlowProblem, kind: ArcKind) -> Vec<&FlowArc> {
        pb.arcs.iter().filter(|a| a.kind == kind).collect()
    }

    fn count_starts(pb: &FlowProblem) -> usize {
        pb.arcs.iter().filter(|a| a.kind == ArcKind::Start).count()
    }
    fn count_ends(pb: &FlowProblem) -> usize {
        pb.arcs.iter().filter(|a| a.kind == ArcKind::End).count()
    }
    fn is_node_in_layer(pb: &FlowProblem, layer_idx: usize, node_id: NodeId) -> bool {
        pb.layers[layer_idx]
            .node_ids
            .iter()
            .any(|&nid| nid == node_id)
    }

    /* ------------------------------ Unit tests ------------------------------ */

    #[test]
    fn source_and_sink_initialized_and_seed_nodes_start_after_them() {
        let pb = FlowProblem::new(MinCostFlowConfig::default());
        assert_eq!(pb.source, 0);
        assert_eq!(pb.sink, 1);
        assert_eq!(pb.nodes.len(), 2);
        assert!(pb.nodes[0].seed.is_none());
        assert!(pb.nodes[1].seed.is_none());
    }

    #[test]
    fn add_layer_populates_nodes_layers_and_index() {
        let mut pb = FlowProblem::new(MinCostFlowConfig::default());
        let seeds = mk_seeds(100, 3);
        let night: NightId = 42;

        let idx = pb.add_layer(night, &seeds).expect("add_layer ok");
        assert_eq!(idx, 0);
        assert_eq!(pb.layers.len(), 1);
        assert_eq!(pb.layers[0].night, night);
        assert_eq!(pb.layers[0].node_ids.len(), seeds.len());
        assert_eq!(pb.nodes.len(), 2 + seeds.len());

        // index_of doit refléter seed<->node
        for (k, nid) in &pb.index_of {
            assert_eq!(pb.nodes[*nid as usize].seed.unwrap(), *k);
        }
    }

    #[test]
    fn start_and_end_arcs_have_expected_shape_and_costs() {
        let cfg = MinCostFlowConfig::default();
        let mut pb = FlowProblem::new(cfg.clone());
        let seeds = mk_seeds(10, 4);
        let layer = pb.add_layer(1, &seeds).unwrap();

        let n_start = pb.add_start_arcs(layer).unwrap();
        let n_end = pb.add_end_arcs(layer).unwrap();
        assert_eq!(n_start, seeds.len());
        assert_eq!(n_end, seeds.len());

        let starts = arcs_of_kind(&pb, ArcKind::Start);
        let ends = arcs_of_kind(&pb, ArcKind::End);
        assert_eq!(starts.len(), seeds.len());
        assert_eq!(ends.len(), seeds.len());

        for a in starts {
            assert_eq!(a.from, pb.source);
            assert!(pb.nodes[a.to as usize].seed.is_some());
            assert!((a.cost - cfg.lambda_start).abs() < 1e-12);
            assert_eq!(a.capacity, 1);
            assert!(matches!(a.kind, ArcKind::Start));
            assert!(a.left.is_none());
            assert!(a.right.is_some());
        }

        for a in ends {
            assert!(pb.nodes[a.from as usize].seed.is_some());
            assert_eq!(a.to, pb.sink);
            assert!((a.cost - cfg.lambda_end).abs() < 1e-12);
            assert_eq!(a.capacity, 1);
            assert!(matches!(a.kind, ArcKind::End));
            assert!(a.left.is_some());
            assert!(a.right.is_none());
        }
    }

    #[test]
    fn link_arcs_go_forward_in_time_and_match_index() {
        let mut pb = FlowProblem::new(MinCostFlowConfig::default());
        let left_night: NightId = 5;
        let right_night: NightId = 6;

        let left_seeds = mk_seeds(1_000, 3);
        let right_seeds = mk_seeds(2_000, 2);

        pb.add_layer(left_night, &left_seeds).unwrap();
        pb.add_layer(right_night, &right_seeds).unwrap();

        let mut edges = Vec::new();
        for (i, l) in left_seeds.iter().enumerate() {
            for (j, r) in right_seeds.iter().enumerate() {
                edges.push(Edge {
                    from: l.seed_id,
                    to: r.seed_id,
                    cost: (i + j) as f64,
                    dt_days: 0.5,
                });
            }
        }

        let created = pb
            .add_link_arcs(left_night, right_night, &edges)
            .expect("add_link_arcs ok");
        assert_eq!(created, edges.len());

        for a in pb.arcs.iter().filter(|a| a.kind == ArcKind::Link) {
            let lk = a.left.expect("left seed present");
            let rk = a.right.expect("right seed present");
            assert_eq!(lk.night, left_night);
            assert_eq!(rk.night, right_night);

            let from_seed = pb.nodes[a.from as usize].seed.unwrap();
            let to_seed = pb.nodes[a.to as usize].seed.unwrap();
            assert_eq!(from_seed, lk);
            assert_eq!(to_seed, rk);

            // invariant temporel (le module construit « vers l’avant »)
            assert!(lk.night < rk.night, "link must go forward in time");
            assert_eq!(a.capacity, 1);
            assert!(a.cost.is_finite());
        }
    }

    #[test]
    fn solve_and_extract_with_null_solver_reports_zero_flow() {
        let cfg = MinCostFlowConfig::default();
        let mut builder = FlowBuilder::new(cfg);

        let l1 = mk_seeds(10, 2);
        let l2 = mk_seeds(20, 3);
        builder.ingest_night(100, &l1).unwrap();
        builder.ingest_night(101, &l2).unwrap();

        let edges = vec![
            Edge {
                from: 10,
                to: 20,
                cost: 1.0,
                dt_days: 0.5,
            },
            Edge {
                from: 11,
                to: 21,
                cost: 2.0,
                dt_days: 0.5,
            },
        ];
        builder.add_links_between(100, 101, &edges).unwrap();

        let upd = solve_and_extract(&mut builder, &NullFlowSolver).expect("solve ok");
        assert_eq!(upd.night, 101);
        assert_eq!(upd.n_layers, 2);
        assert_eq!(upd.total_flow, 0);
        assert!(upd.trajectories.is_empty());
        assert_eq!(upd.total_nodes, builder.pb.nodes.len());
        assert_eq!(upd.total_arcs, builder.pb.arcs.len());
    }

    #[test]
    fn flowsolution_paths_extracts_chains() {
        // Build a synthetic solution with two chains (no standalone singleton).
        // Path A: (n=1,s=10) -> (n=2,s=20) -> (n=3,s=30)
        // Path B: (n=2,s=21) -> (n=3,s=31)
        let a1 = SeedKey { night: 1, seed: 10 };
        let a2 = SeedKey { night: 2, seed: 20 };
        let a3 = SeedKey { night: 3, seed: 30 };
        let b1 = SeedKey { night: 2, seed: 21 };
        let b2 = SeedKey { night: 3, seed: 31 };

        let mut succ = ahash::AHashMap::new();
        let mut pred = ahash::AHashMap::new();

        succ.insert(a1, a2);
        pred.insert(a2, a1);
        succ.insert(a2, a3);
        pred.insert(a3, a2);
        succ.insert(b1, b2);
        pred.insert(b2, b1);

        let sol = FlowSolution {
            total_flow: 2,
            active_arcs: vec![],
            succ_of: succ,
            pred_of: pred,
        };

        let paths = sol.paths();
        // Sorted by head (night, seed): heads are a1 (1,10) then b1 (2,21).
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0], vec![a1, a2, a3]);
        assert_eq!(paths[1], vec![b1, b2]);
    }

    // ------------------ [NEW] : Terminal-arc policy tests ------------------

    #[test]
    fn builder_policy_no_start_on_last_and_end_only_on_last() {
        let mut builder = FlowBuilder::new(MinCostFlowConfig::default());

        // N1 arrives: no Start/End arcs anywhere.
        let n1 = mk_seeds(10_000, 3);
        let l1 = builder.ingest_night(100, &n1).unwrap();
        assert_eq!(l1.n_start_arcs, 0);
        assert_eq!(l1.n_end_arcs, 0);
        assert_eq!(count_starts(&builder.pb), 0);
        assert_eq!(count_ends(&builder.pb), 0);

        // N2 arrives:
        // - N1 becomes "startable" => add Start arcs on N1,
        // - N2 becomes "endable"   => add End arcs on N2,
        // - never add Start arcs on the last layer (N2).
        let n2 = mk_seeds(20_000, 2);
        let l2 = builder.ingest_night(101, &n2).unwrap();
        assert_eq!(l2.n_start_arcs, n1.len()); // N1 Start arcs
        assert_eq!(l2.n_end_arcs, n2.len()); // N2 End arcs
                                             // Global counts after N2:
        assert_eq!(count_starts(&builder.pb), n1.len());
        assert_eq!(count_ends(&builder.pb), n2.len());

        // N3 arrives:
        // - remove End arcs from N2 (to forbid Source→N2→Sink),
        // - add Start arcs on N2 (so Source→N2→N3→Sink becomes possible),
        // - add End arcs on N3,
        // - never add Start arcs on the last layer (N3).
        let n3 = mk_seeds(30_000, 4);
        let l3 = builder.ingest_night(102, &n3).unwrap();
        assert_eq!(l3.n_start_arcs, n2.len()); // N2 Start arcs
        assert_eq!(l3.n_end_arcs, n3.len()); // N3 End arcs

        // Global post-N3:
        // - Start arcs exist on N1 and N2, not on N3
        assert_eq!(count_starts(&builder.pb), n1.len() + n2.len());
        // - End arcs exist only on N3
        assert_eq!(count_ends(&builder.pb), n3.len());

        // Structural checks: no Start into last layer; End only from last layer.
        let last_idx = builder.pb.layers.len() - 1;
        for a in builder.pb.arcs.iter().filter(|a| a.kind == ArcKind::Start) {
            assert!(
                !is_node_in_layer(&builder.pb, last_idx, a.to),
                "no Start on last layer"
            );
            assert_eq!(a.from, builder.pb.source);
        }
        for a in builder.pb.arcs.iter().filter(|a| a.kind == ArcKind::End) {
            assert!(
                is_node_in_layer(&builder.pb, last_idx, a.from),
                "End only from last layer"
            );
            assert_eq!(a.to, builder.pb.sink);
        }
    }

    #[test]
    fn builder_policy_forbids_two_hop_paths() {
        // Asserts the policy forbids Source→seed→Sink by:
        // - ensuring there are no Start arcs into the last layer,
        // - ensuring there are End arcs only from the last layer.
        let mut builder = FlowBuilder::new(MinCostFlowConfig::default());
        let n1 = mk_seeds(1_000, 1);
        let n2 = mk_seeds(2_000, 1);
        let n3 = mk_seeds(3_000, 1);
        builder.ingest_night(10, &n1).unwrap();
        builder.ingest_night(11, &n2).unwrap();
        builder.ingest_night(12, &n3).unwrap();

        let last_idx = builder.pb.layers.len() - 1;

        // No Start arcs into the last layer
        for a in builder.pb.arcs.iter().filter(|a| a.kind == ArcKind::Start) {
            assert!(!is_node_in_layer(&builder.pb, last_idx, a.to));
        }
        // End arcs only from the last layer
        for a in builder.pb.arcs.iter().filter(|a| a.kind == ArcKind::End) {
            assert!(is_node_in_layer(&builder.pb, last_idx, a.from));
        }
    }

    /* -------------------------- Property-based tests ------------------------- */

    proptest! {

        #[test]
        fn prop_link_arcs_forward_in_time_and_consistent_with_index(
            left_night in 1u32..100u32,
            gap in 1u32..5u32,
            n_left in 1usize..=5,
            n_right in 1usize..=5,
            costs in prop::collection::vec(0f64..100f64, 1..=32),
        ) {
            let right_night = left_night + gap;

            let mut pb = FlowProblem::new(MinCostFlowConfig::default());
            let left = mk_seeds(10_000, n_left);
            let right = mk_seeds(20_000, n_right);
            pb.add_layer(left_night, &left).unwrap();
            pb.add_layer(right_night, &right).unwrap();

            let mut edges = Vec::new();
            let mut k = 0usize;
            'outer: for l in &left {
                for r in &right {
                    if k >= costs.len() { break 'outer; }
                    edges.push(Edge {
                        from: l.seed_id,
                        to: r.seed_id,
                        cost: costs[k].abs(),
                        dt_days: gap as f64
                    });
                    k += 1;
                }
            }
            let created = pb.add_link_arcs(left_night, right_night, &edges).unwrap();
            assert_eq!(created, edges.len());

            for a in pb.arcs.iter().filter(|a| a.kind == ArcKind::Link) {
                let lk = a.left.unwrap();
                let rk = a.right.unwrap();
                assert_eq!(lk.night, left_night);
                assert_eq!(rk.night, right_night);
                assert!(lk.night < rk.night);

                let from_idx = *pb.index_of.get(&lk).expect("from indexed");
                let to_idx = *pb.index_of.get(&rk).expect("to indexed");
                assert_eq!(from_idx, a.from);
                assert_eq!(to_idx, a.to);

                assert_eq!(a.capacity, 1);
                assert!(a.cost.is_finite());
                assert!(a.dt_days >= 0.0);
            }
        }

        /// Property test for the final invariants enforced by the "no Source→seed→Sink" policy.
        ///
        /// Rationale
        /// ---------
        /// The builder policy guarantees a *final shape* for terminal arcs after all nights
        /// have been ingested:
        ///   - Start arcs (Source→seed) exist on **all layers except the last one**, with
        ///     exactly **one Start per seed** on those non-last layers; **no Start** is allowed
        ///     into the last layer.
        ///   - End arcs (seed→Sink) exist **only on the last layer**, with exactly **one End
        ///     per seed** on the last layer. In the special case where there is only a single
        ///     layer (L == 1), there must be **zero End arcs overall**.
        ///
        /// Why we check *final* invariants only
        /// ------------------------------------
        /// During ingestion, per-step increments (the counts returned by `ingest_night`) can
        /// legitimately be zero in edge cases (e.g., first layer, deduplicated nights, or when
        /// a previous last layer had zero seeds so no terminals are added/removed). Therefore,
        /// asserting the *final* graph shape is both simpler and robust to such edge cases.
        ///
        /// What we assert
        /// --------------
        /// After all nights are ingested:
        ///   1) Global counts:
        ///        - total Start arcs  == sum of seeds in all layers **except** the last;
        ///        - total End arcs    == number of seeds in the last layer (except if L == 1, then 0).
        ///   2) Per-layer structure:
        ///        - each seed in a non-last layer has exactly one **incoming Start** and **zero** outgoing End;
        ///        - each seed in the last layer has **zero** incoming Start and exactly one **outgoing End**
        ///          (except when L == 1 → zero End total).
        ///   3) Arc metadata sanity:
        ///        - Start: from Source, correct cost and capacity, never target the last layer;
        ///        - End: to Sink, correct cost and capacity, originate only from the last layer
        ///          (unless L == 1, in which case there must be none).
        ///   4) Indexing sanity:
        ///        - `index_of` maps every seed key to its node id.
        #[test]
        fn prop_terminal_policy_final_invariants_hold(
            nights in prop::collection::vec(1u32..200u32, 1..=6),
            seeds_per in prop::collection::vec(0usize..=5, 1..=6),
            lambda_start in 0f64..10_000f64,
            lambda_end   in 0f64..10_000f64,
        ) {
            // Enforce strictly increasing unique nights: the policy is defined on chronologically
            // ordered layers. Duplicates can appear in the generator; we deduplicate here.
            let mut nights_sorted = nights.clone();
            nights_sorted.sort_unstable();
            nights_sorted.dedup();
            prop_assume!(!nights_sorted.is_empty());

            // Align the seeds_per vector to the deduplicated nights.
            let mut seeds_per_clean = Vec::with_capacity(nights_sorted.len());
            for i in 0..nights_sorted.len() {
                seeds_per_clean.push(seeds_per.get(i).copied().unwrap_or(0));
            }

            let cfg = MinCostFlowConfig {
                lambda_start: lambda_start.abs(),
                lambda_end:   lambda_end.abs(),
                ..Default::default()
            };

            let mut builder = FlowBuilder::new(cfg.clone());

            // Ingest all nights. We do NOT assert per-step increments here; we will only
            // check the final shape (global + per-layer invariants).
            let mut total_seeds = 0usize;
            for (i, &night) in nights_sorted.iter().enumerate() {
                let nseeds = seeds_per_clean[i];
                let base = (i as SeedId) * 10_000;
                let seeds = mk_seeds(base, nseeds);
                let _ = builder.ingest_night(night, &seeds).unwrap();
                total_seeds += nseeds;
            }

            // Build CSR once before using in_arcs()/out_arcs() slices.
            builder.pb.ensure_csr();

            let pb = &builder.pb;
            let layer_len = pb.layers.len();
            prop_assume!(layer_len > 0); // guaranteed by previous assumption

            // Collect per-layer sizes from the graph (authoritative).
            let last_idx = layer_len - 1;
            let mut size_per_layer: Vec<usize> = Vec::with_capacity(layer_len);
            for li in 0..layer_len {
                size_per_layer.push(pb.layers[li].node_ids.len());
            }

            // Expected global counts:
            //  - total Starts: sum of seeds on all non-last layers,
            //  - total Ends:   #seeds on the last layer, except if layer_len == 1 (then 0).
            let expected_starts: usize = size_per_layer.iter().take(layer_len.saturating_sub(1)).sum();
            let expected_ends: usize = if layer_len == 1 { 0 } else { size_per_layer[last_idx] };

            // Actual global counts pulled from the arc list.
            let starts = pb.arcs.iter().filter(|a| a.kind == ArcKind::Start).collect::<Vec<_>>();
            let ends   = pb.arcs.iter().filter(|a| a.kind == ArcKind::End).collect::<Vec<_>>();

            prop_assert_eq!(starts.len(), expected_starts, "total Start arcs mismatch");
            prop_assert_eq!(ends.len(),   expected_ends,   "total End arcs mismatch");

            // ---------- Per-layer structure: Start arcs ----------
            // Non-last layers: exactly one incoming Start per seed.
            // Last layer:      strictly zero incoming Start per seed.
            for li in 0..layer_len {
                for &nid in &pb.layers[li].node_ids {
                    let in_slice = pb.in_arcs(nid);
                    let n_start_here = in_slice
                        .iter()
                        .filter(|&&aid| pb.arcs[aid as usize].kind == ArcKind::Start)
                        .count();
                    if li == last_idx {
                        prop_assert_eq!(n_start_here, 0, "no Start into the last layer");
                    } else {
                        prop_assert_eq!(n_start_here, 1, "exactly one Start into non-last layers");
                    }
                }
            }

            // ---------- Per-layer structure: End arcs ----------
            // If L == 1 → zero End arcs anywhere (single-layer graphs cannot end).
            // Else:
            //   - Last layer:      exactly one outgoing End per seed,
            //   - Non-last layers: zero outgoing End per seed.
            for li in 0..layer_len {
                for &nid in &pb.layers[li].node_ids {
                    let out_slice = pb.out_arcs(nid);
                    let n_end_here = out_slice
                        .iter()
                        .filter(|&&aid| pb.arcs[aid as usize].kind == ArcKind::End)
                        .count();

                    if layer_len == 1 {
                        prop_assert_eq!(n_end_here, 0, "single-layer graph must have zero End arcs");
                    } else if li == last_idx {
                        prop_assert_eq!(n_end_here, 1, "exactly one End from the last layer");
                    } else {
                        prop_assert_eq!(n_end_here, 0, "no End from non-last layers");
                    }
                }
            }

            // ---------- Arc metadata sanity checks ----------
            // Start arcs:
            for a in &starts {
                prop_assert!((a.cost - cfg.lambda_start).abs() < 1e-9, "Start cost mismatch");
                prop_assert_eq!(a.capacity, 1, "Start capacity must be 1");
                prop_assert_eq!(a.from, pb.source, "Start must originate at Source");
                prop_assert!(a.right.is_some(), "Start should carry the right SeedKey");
                // Must not target a node in the last layer.
                let targets_last = pb.layers[last_idx].node_ids.contains(&a.to);
                prop_assert!(!targets_last, "no Start allowed into last layer");
            }

            // End arcs:
            for a in &ends {
                prop_assert!((a.cost - cfg.lambda_end).abs() < 1e-9, "End cost mismatch");
                prop_assert_eq!(a.capacity, 1, "End capacity must be 1");
                prop_assert_eq!(a.to, pb.sink, "End must terminate at Sink");
                prop_assert!(a.left.is_some(), "End should carry the left SeedKey");
                if layer_len == 1 {
                    // With a single layer, the set `ends` should already be empty (global check above),
                    // so we should not even enter this branch. Keep the assertion for clarity.
                    prop_assert!(false, "no End arcs expected when layer_len == 1");
                } else {
                    // Must originate from a node in the last layer only.
                    let from_is_last = pb.layers[last_idx].node_ids.contains(&a.from);
                    prop_assert!(from_is_last, "End must originate from the last layer only");
                }
            }

            // ---------- Indexing sanity ----------
            // Every seed node (all layers) must have an entry in `index_of` that maps
            // to the node id that carries it.
            let mut counted = 0usize;
            for n in pb.nodes.iter().skip(2) { // skip Source/Sink
                let sk = n.seed.expect("seed node must have SeedKey");
                let got = pb.index_of.get(&sk).copied().expect("indexed");
                prop_assert_eq!(got, n.id, "index_of must map SeedKey to the right node id");
                counted += 1;
            }
            prop_assert_eq!(counted, total_seeds, "index_of coverage mismatch");
        }
    }

    mod flow_csr_tests {
        use super::*;

        /* ------------------------------ CSR tests -------------------------------- */

        /// Naive degree counters for validation against CSR.
        fn naive_degrees(pb: &FlowProblem) -> (Vec<usize>, Vec<usize>) {
            let n = pb.nodes.len();
            let mut dout = vec![0usize; n];
            let mut din = vec![0usize; n];
            for a in &pb.arcs {
                dout[a.from as usize] += 1;
                din[a.to as usize] += 1;
            }
            (dout, din)
        }

        /// Check internal CSR invariants against the raw arc list.
        fn assert_csr_consistency(pb: &FlowProblem) {
            let n = pb.nodes.len();
            let e = pb.arcs.len();

            // Basic sizes
            assert_eq!(pb.out_off.len(), n + 1, "out_off must have length n+1");
            assert_eq!(pb.in_off.len(), n + 1, "in_off must have length n+1");
            assert_eq!(pb.out_adj.len(), e, "out_adj must have length |E|");
            assert_eq!(pb.in_adj.len(), e, "in_adj must have length |E|");

            // Last offsets are cumulative degree sums and equal to |E|
            assert_eq!(pb.out_off[n] as usize, e, "last out_off must be |E|");
            assert_eq!(pb.in_off[n] as usize, e, "last in_off must be |E|");

            // Prefix-sum monotonicity
            for w in pb.out_off.windows(2) {
                assert!(w[0] <= w[1], "out_off must be non-decreasing");
            }
            for w in pb.in_off.windows(2) {
                assert!(w[0] <= w[1], "in_off must be non-decreasing");
            }

            // Degrees per node from CSR == naive scan
            let (dout, din) = naive_degrees(pb);
            for u in 0..n {
                let csr_dout = (pb.out_off[u + 1] - pb.out_off[u]) as usize;
                let csr_din = (pb.in_off[u + 1] - pb.in_off[u]) as usize;
                assert_eq!(csr_dout, dout[u], "out-degree mismatch at node {}", u);
                assert_eq!(csr_din, din[u], "in-degree mismatch at node {}", u);
            }

            // Every arc id appears exactly once in out_adj and in_adj
            {
                let mut seen_out = vec![0u8; e];
                for &aid in &pb.out_adj {
                    let idx = aid as usize;
                    assert!(idx < e, "arc id out of range in out_adj");
                    seen_out[idx] += 1;
                }
                for (i, c) in seen_out.iter().enumerate() {
                    assert_eq!(*c, 1, "arc {} must appear exactly once in out_adj", i);
                }
            }
            {
                let mut seen_in = vec![0u8; e];
                for &aid in &pb.in_adj {
                    let idx = aid as usize;
                    assert!(idx < e, "arc id out of range in in_adj");
                    seen_in[idx] += 1;
                }
                for (i, c) in seen_in.iter().enumerate() {
                    assert_eq!(*c, 1, "arc {} must appear exactly once in in_adj", i);
                }
            }

            // Slices returned by helpers match arc endpoints
            for u in 0..n as u32 {
                for &aid in pb.out_arcs(u) {
                    let a = &pb.arcs[aid as usize];
                    assert_eq!(a.from, u, "out_arcs slice must only contain arcs leaving u");
                }
            }
            for v in 0..n as u32 {
                for &aid in pb.in_arcs(v) {
                    let a = &pb.arcs[aid as usize];
                    assert_eq!(a.to, v, "in_arcs slice must only contain arcs entering v");
                }
            }
        }

        #[test]
        fn csr_on_empty_graph_and_slices_are_empty() {
            let mut pb = FlowProblem::new(MinCostFlowConfig::default());
            // No layers/arcs yet; only Source/Sink nodes exist.
            pb.ensure_csr();
            assert!(pb.csr_ready, "CSR must be marked ready after ensure_csr()");
            assert_csr_consistency(&pb);

            // Source and Sink have zero degrees, hence empty slices.
            assert!(pb.out_arcs(pb.source).is_empty());
            assert!(pb.in_arcs(pb.source).is_empty());
            assert!(pb.out_arcs(pb.sink).is_empty());
            assert!(pb.in_arcs(pb.sink).is_empty());
        }

        #[test]
        fn csr_includes_start_and_end_arcs_in_expected_nodes() {
            let cfg = MinCostFlowConfig::default();
            let mut pb = FlowProblem::new(cfg);
            let seeds = mk_seeds(42, 3);
            let layer_idx = pb.add_layer(7, &seeds).unwrap();

            // Add start & end arcs then build CSR.
            pb.add_start_arcs(layer_idx).unwrap();
            pb.add_end_arcs(layer_idx).unwrap();

            // CSR not yet built -> ensure_csr must rebuild.
            assert!(!pb.csr_ready);
            pb.ensure_csr();
            assert!(pb.csr_ready);

            // All Start arcs must appear in out_arcs(Source) and in_arcs(seed).
            let start_ids: Vec<ArcId> = pb
                .arcs
                .iter()
                .filter(|a| a.kind == ArcKind::Start)
                .map(|a| a.id)
                .collect();

            for &aid in pb.out_arcs(pb.source) {
                assert_eq!(pb.arcs[aid as usize].kind, ArcKind::Start);
            }
            // Each seed must have exactly one incoming Start arc.
            for &nid in &pb.layers[0].node_ids {
                let in_slice = pb.in_arcs(nid);
                let n_start_here = in_slice
                    .iter()
                    .filter(|&&aid| pb.arcs[aid as usize].kind == ArcKind::Start)
                    .count();
                assert_eq!(
                    n_start_here, 1,
                    "each seed must have one Start arc incoming"
                );
            }

            // All End arcs must appear in in_arcs(Sink) and out_arcs(seed).
            for &aid in pb.in_arcs(pb.sink) {
                assert_eq!(pb.arcs[aid as usize].kind, ArcKind::End);
            }
            for &nid in &pb.layers[0].node_ids {
                let out_slice = pb.out_arcs(nid);
                let n_end_here = out_slice
                    .iter()
                    .filter(|&&aid| pb.arcs[aid as usize].kind == ArcKind::End)
                    .count();
                assert_eq!(n_end_here, 1, "each seed must have one End arc outgoing");
            }

            // Global CSR consistency
            assert_csr_consistency(&pb);

            // And all Start/End arc ids should be referenced somewhere in CSR.
            for aid in start_ids {
                let found = pb.out_arcs(pb.source).iter().any(|&x| x == aid);
                assert!(found, "every Start arc must be in out_arcs(Source)");
            }
            let end_ids: Vec<ArcId> = pb
                .arcs
                .iter()
                .filter(|a| a.kind == ArcKind::End)
                .map(|a| a.id)
                .collect();
            for aid in end_ids {
                let found = pb.in_arcs(pb.sink).iter().any(|&x| x == aid);
                assert!(found, "every End arc must be in in_arcs(Sink)");
            }
        }

        #[test]
        fn csr_invalidates_on_mutation_and_rebuilds_lazily() {
            let mut pb = FlowProblem::new(MinCostFlowConfig::default());
            let left = mk_seeds(1_000, 2);
            let right = mk_seeds(2_000, 2);
            pb.add_layer(10, &left).unwrap();
            pb.add_layer(11, &right).unwrap();

            // Build some arcs then build CSR.
            pb.add_start_arcs(0).unwrap();
            pb.add_end_arcs(0).unwrap();
            pb.ensure_csr();
            assert!(pb.csr_ready);

            // Mutate (add links) -> csr_ready must be false.
            let mut edges = Vec::new();
            for l in &left {
                for r in &right {
                    edges.push(Edge {
                        from: l.seed_id,
                        to: r.seed_id,
                        cost: 1.0,
                        dt_days: 1.0,
                    });
                }
            }
            pb.add_link_arcs(10, 11, &edges).unwrap();
            assert!(!pb.csr_ready, "mutations must invalidate CSR");

            // Lazy rebuild
            pb.ensure_csr();
            assert!(pb.csr_ready, "ensure_csr() must rebuild after invalidation");
            assert_csr_consistency(&pb);
        }

        proptest! {
            /// Build small random layered graphs, connect **all** pairs between consecutive
            /// nights, and check CSR invariants systematically.
            #[test]
            fn prop_csr_consistent_on_random_layered_graphs(
                nights in prop::collection::vec(1u32..150, 1..=4),
                per_layer in prop::collection::vec(0usize..=5, 1..=4),
            ) {
                // Ensure strictly increasing unique nights.
                let mut nights = nights;
                nights.sort_unstable();
                nights.dedup();
                prop_assume!(!nights.is_empty());

                // Align per-layer sizes
                let mut per = Vec::with_capacity(nights.len());
                for i in 0..nights.len() {
                    per.push(*per_layer.get(i).unwrap_or(&0usize));
                }

                let mut pb = FlowProblem::new(MinCostFlowConfig::default());

                // Ingest layers and Start/End arcs
                let mut layer_seeds: Vec<Vec<SeedNode>> = Vec::new();
                for (i, &n) in nights.iter().enumerate() {
                    let seeds = mk_seeds((i as SeedId) * 10_000, per[i]);
                    pb.add_layer(n, &seeds).unwrap();
                    pb.add_start_arcs(i).unwrap();
                    pb.add_end_arcs(i).unwrap();
                    layer_seeds.push(seeds);
                }

                // Fully connect consecutive nights with Link arcs (dense bipartite).
                for i in 0..nights.len().saturating_sub(1) {
                    let left_n = nights[i];
                    let right_n = nights[i+1];
                    let mut edges = Vec::new();
                    for l in &layer_seeds[i] {
                        for r in &layer_seeds[i+1] {
                            edges.push(Edge {
                                from: l.seed_id,
                                to: r.seed_id,
                                cost: ((l.seed_id as i64 - r.seed_id as i64).abs() as f64).sqrt() + 0.1,
                                dt_days: (right_n - left_n) as f64
                            });
                        }
                    }
                    pb.add_link_arcs(left_n, right_n, &edges).unwrap();
                }

                // Build CSR & validate
                pb.ensure_csr();
                assert!(pb.csr_ready);
                assert_csr_consistency(&pb);

                // Spot-check a few nodes: out_arcs/in_arcs slices contain proper arc ends.
                let n = pb.nodes.len() as u32;
                for u in 0..n {
                    for &aid in pb.out_arcs(u) {
                        let a = &pb.arcs[aid as usize];
                        prop_assert_eq!(a.from, u);
                    }
                    for &aid in pb.in_arcs(u) {
                        let a = &pb.arcs[aid as usize];
                        prop_assert_eq!(a.to, u);
                    }
                }
            }
        }
    }
}
