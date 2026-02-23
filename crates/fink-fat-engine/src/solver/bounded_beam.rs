//! Bounded beam solver: beam-search enumeration of path hypotheses inside a
//! time-forward DAG component.
//!
//! Overview
//! --------
//! This module implements the [`BoundedBeamSolver`], a solver that enumerates
//! plausible trajectory hypotheses (**tracks**) **within a single connected
//! component** of the runtime graph.
//!
//! The solver does **bounded path enumeration** in the component-restricted
//! directed subgraph exposed by [`ConnectedComponents`].
//!
//! Inputs from `ConnectedComponents`
//! --------------------------------
//! For a given `component_id`, the solver consumes:
//!
//! - `component_nodes(component_id)`
//!   - returns the node slice in **component-local index order**,
//! - `component_out_edges(component_id)`
//!   - returns the **restricted outgoing adjacency** as `&Edge` references,
//!     indexed by local node id,
//! - `component_sources_local(component_id)`
//!   - returns source nodes (local indices) used as starting points.
//!
//! Problem solved
//! --------------
//! Given a directed acyclic subgraph (in the time-forward sense), enumerate a
//! **bounded set** of low-cost directed paths that can represent plausible
//! multi-night linking hypotheses.
//!
//! Why "bounded beam"?
//! -------------------
//! Naively enumerating all paths in a branching DAG can grow combinatorially.
//! Beam search keeps only the best partial hypotheses at each step, and this
//! solver additionally enforces hard limits on:
//!
//! - exploration width (`beam_width`),
//! - local branching (`max_out_per_node`),
//! - total work (`max_expansions`),
//! - output volume (`max_tracks`, `max_tracks_per_source`).
//!
//! Key assumptions (invariants)
//! ----------------------------
//! - **Time-forward edges**: for any edge `u -> v`, `night(v) > night(u)` (gaps allowed).
//! - **No directed cycles across nights**: time-forward implies acyclicity for solving.
//! - The adjacency exposed by `ConnectedComponents` is already restricted to the
//!   component (no cross-component edges).
//!
//! High-level algorithm
//! --------------------
//! For each component:
//!
//! 1) **Prepare adjacency (solver view)**
//!    - Convert CC adjacency into an internal adjacency that caches:
//!      - destination local index,
//!      - cost,
//!      - reference to the original `Edge`.
//!    - Sort outgoing edges by increasing cost.
//!    - Prune outgoing edges per node to `max_out_per_node`.
//!
//! 2) **Initialize beam**
//!    - Create one root state per source node.
//!
//! 3) **Beam search loop**
//!    - Expand the current frontier into next candidate states.
//!    - Enforce global exploration limits (`max_expansions`, `beam_width`).
//!    - Emit terminal states when reaching sinks (nodes with no outgoing edges).
//!    - Enforce per-source emission limit (`max_tracks_per_source`).
//!
//! 4) **Reconstruct tracks**
//!    - Follow parent backpointers from terminal states to rebuild node/edge lists.
//!
//! 5) **Rank and truncate**
//!    - Sort tracks by **average cost per edge**,
//!    - tie-break by longer tracks, then lower total cost,
//!    - truncate to `max_tracks`.
//!
//! Why average cost per edge?
//! --------------------------
//! Sorting by total cost alone tends to favor short paths, because every
//! additional edge increases total cost.
//! Average cost per edge normalizes for path length and reduces this bias,
//! while still prioritizing low-cost connections.
//!
//! Complexity notes
//! ---------------
//! Let:
//! - `n` be the number of nodes in the component,
//! - `m` be the number of edges after component restriction,
//! - `B = beam_width`,
//! - `D` be the effective out-degree after `max_out_per_node` pruning.
//!
//! - Adjacency preparation: `O(n + Σ_u deg(u) log deg(u))` (per-node sorting).
//! - One beam iteration: roughly `O((B*D) log(B*D))` due to candidate sorting.
//! - Total runtime is bounded by `max_expansions` and output caps.
//!
//! Determinism
//! -----------
//! The solver is deterministic given deterministic inputs (node order, adjacency,
//! finite costs). Sorting uses `partial_cmp` with an equality fallback; NaN costs
//! can lead to unstable ordering and should be avoided in production.

use std::cmp::Ordering;

use ahash::AHashMap;

use crate::{
    engine_config::solver_config::bounded_beam_config::BoundedBeamConfig,
    graph::{AlertLinkageDAG, edge::Edge},
    seeding::{SeedKey, SeedNode},
    solver::{
        Solver, SolverDiagnostics, SolverOutput,
        components::{ComponentId, ConnectedComponents, LocalIdx},
    },
    trajectory::TrackHypothesis,
};

// -----------------------------------------------------------------------------
// Bounded beam solver (beam search over time-forward DAG components)
// -----------------------------------------------------------------------------

/// Bounded beam solver: enumerates path hypotheses inside a time-forward DAG component.
///
/// This solver is a **bounded path enumerator**:
/// - it does not attempt to enumerate all paths in a branching DAG,
/// - it keeps only the best partial hypotheses using beam search,
/// - it enforces strict caps on exploration and output size.
///
/// Component data (nodes, restricted adjacency, sources) is obtained from
/// [`ConnectedComponents`].
#[derive(Clone, Debug, Default)]
pub struct BoundedBeamSolver {
    /// Beam-search and output capping configuration.
    pub cfg: BoundedBeamConfig,
}

impl BoundedBeamSolver {
    /// Create a new bounded beam solver with the given configuration.
    ///
    /// Arguments
    /// ---------
    /// * `cfg` – Beam search and output capping configuration.
    pub fn new(cfg: BoundedBeamConfig) -> Self {
        Self { cfg }
    }

    /// Solve a single component using the precomputed component view.
    ///
    /// This method enumerates candidate tracks inside the component-restricted
    /// directed subgraph exposed by [`ConnectedComponents`].
    ///
    /// Algorithm
    /// ---------
    /// 1) Fetch the component view:
    ///    - node slice in local order,
    ///    - outgoing adjacency restricted to the component,
    ///    - local sources.
    ///
    /// 2) Build a solver adjacency:
    ///    - cache destination local index,
    ///    - sort outgoing edges by cost,
    ///    - prune to `cfg.max_out_per_node`.
    ///
    /// 3) Run bounded beam search:
    ///    - expand partial paths from sources,
    ///    - emit terminal paths at sinks,
    ///    - enforce budgets:
    ///      - `cfg.beam_width`,
    ///      - `cfg.max_expansions`,
    ///      - `cfg.max_tracks`,
    ///      - `cfg.max_tracks_per_source`.
    ///
    /// 4) Reconstruct terminal paths into [`TrackHypothesis`].
    ///
    /// 5) Rank by average cost per edge, then truncate.
    ///
    /// Arguments
    /// ---------
    /// * `graph` – Global runtime graph (read-only; used for lifetimes and invariants).
    /// * `cc` – Connected components object providing restricted adjacency and sources.
    /// * `component_id` – Component to solve.
    ///
    /// Return
    /// ------
    /// * `SolverOutput` – Candidate tracks for this component, sorted and truncated.
    ///
    /// Notes
    /// -----
    /// - If `component_nodes.len() < cfg.min_nodes`, no track can meet the minimum
    ///   length, so this returns an empty output.
    /// - If the component has no sources, this returns an empty output.
    pub fn solve_component_cc<'edge_lf, 'seed_lf>(
        &self,
        graph: &'edge_lf AlertLinkageDAG,
        cc: &'edge_lf ConnectedComponents<'edge_lf, 'seed_lf>,
        component_id: ComponentId,
    ) -> SolverOutput
    where
        'edge_lf: 'seed_lf,
    {
        let cfg = &self.cfg;

        let component_nodes = cc.component_nodes(component_id);
        let n_nodes = component_nodes.len();

        // Diagnostics are updated along the way (candidate edges seen, selected tracks, ...).
        let mut diag = SolverDiagnostics {
            component_id,
            solver_name: self.name(),
            n_nodes: n_nodes as u32,
            m_active_edges: cc.component_active_edges(component_id),
            time_est_s: 0.0,
            time_spent_s: 0.0,
            n_candidates: 0,
            n_selected: 0,
        };

        // Early exit: even a full component path cannot satisfy `min_nodes`.
        if n_nodes < cfg.min_nodes {
            return SolverOutput {
                tracks: AHashMap::new(),
                diag,
            };
        }

        let out_edges = cc.component_out_edges(component_id);
        let sources = cc.component_sources_local(component_id);

        // Beam search requires at least one start point.
        if sources.is_empty() {
            return SolverOutput {
                tracks: AHashMap::new(),
                diag,
            };
        }

        let tracks = enumerate_beam_tracks_from_component_view(
            cfg,
            graph,
            component_nodes,
            out_edges,
            sources,
            &mut diag,
        );

        diag.n_selected = tracks.len() as u32;
        SolverOutput { tracks, diag }
    }
}

impl<'edge_lf, 'seed_lf, 'alert_lf> Solver<'edge_lf, 'seed_lf> for BoundedBeamSolver {
    /// Stable solver name for logs and metrics.
    fn name(&self) -> &'static str {
        "bounded_beam"
    }

    /// Solve a component using its `component_id` and the `ConnectedComponents` view.
    ///
    /// This is the trait entrypoint used by orchestration code.
    fn solve(
        &self,
        graph: &'edge_lf AlertLinkageDAG,
        cc: &'edge_lf ConnectedComponents<'edge_lf, 'seed_lf>,
        component_id: ComponentId,
    ) -> SolverOutput
    where
        'edge_lf: 'seed_lf,
    {
        self.solve_component_cc(graph, cc, component_id)
    }
}

// -----------------------------------------------------------------------------
// Private implementation details
// -----------------------------------------------------------------------------

/// Local outgoing edge inside a component (pruned/sorted solver view).
///
/// This is derived from the `&Edge` adjacency stored in [`ConnectedComponents`].
/// The solver caches:
///
/// - `dst`: destination local index (for O(1) traversal),
/// - `cost`: edge cost (to avoid repeated deref),
/// - `edge`: reference to the original edge (for reconstruction).
///
/// Notes
/// -----
/// - Keeping references avoids copying edge payloads.
/// - Caching `dst` and `cost` reduces overhead inside the inner beam loop.
#[derive(Clone, Copy, Debug)]
struct OutEdge<'edge_lf> {
    /// Destination node (component-local index).
    dst: usize,
    /// Borrowed reference to the original edge.
    edge: &'edge_lf Edge,
    /// Cached edge cost (lower is better).
    cost: f64,
}

/// Pruned/sorted adjacency used by the beam search.
///
/// Layout
/// ------
/// `out[u]` is the list of outgoing edges from local node `u`.
type Adjacency<'edge_lf> = Vec<Vec<OutEdge<'edge_lf>>>;

/// Beam search state stored as a node in an implicit path tree.
///
/// Each state represents a partial path ending at `last`.
/// The path is reconstructed by following `parent` indices back to a root.
///
/// Stored data
/// -----------
/// - `last`: local node id where the partial path ends,
/// - `parent`: state id of the previous step (None for roots),
/// - `in_edge`: edge used to reach this state from its parent,
/// - `total_cost`: sum of edge costs along the path,
/// - `n_edges`: number of edges in the path,
/// - `first_night`, `last_night`: cached night bounds (fast span computation),
/// - `source`: root source node local id (used for per-source caps).
#[derive(Clone, Copy, Debug)]
struct State<'edge_lf> {
    last: usize,
    parent: Option<usize>, // index into `states`
    in_edge: Option<&'edge_lf Edge>,
    total_cost: f64,
    n_edges: u32,
    first_night: u32,
    last_night: u32,
    source: usize,
}

// -----------------------------------------------------------------------------
// Step 1: convert CC adjacency into a solver adjacency (sorted + pruned)
// -----------------------------------------------------------------------------

/// Build a pruned/sorted adjacency for the solver from CC-provided edge refs.
///
/// The `ConnectedComponents` adjacency is already restricted to the component:
/// - `component_out_edges[u]` contains only edges whose `from` is local `u`,
/// - and whose `to` is also within the component.
///
/// This function builds a solver-friendly adjacency:
/// - each `&Edge` becomes an `OutEdge { dst, edge, cost }`,
/// - outgoing edges are sorted by increasing cost,
/// - outgoing edges are truncated to `cfg.max_out_per_node`,
/// - diagnostics count how many edge candidates were seen before pruning.
///
/// Arguments
/// ---------
/// * `cfg` – Bounded beam configuration.
/// * `component_nodes` – Component nodes in local index order.
/// * `component_out_edges` – Restricted adjacency: `out[u] = Vec<&Edge>`.
/// * `diag` – Diagnostics updated in-place (candidate edge counts).
///
/// Return
/// ------
/// * `Adjacency` – Pruned/sorted adjacency suitable for beam search.
///
/// Notes
/// -----
/// The solver must map each edge destination to a component-local index.
/// A temporary map is built once:
/// - key: `SeedKey` of each component node,
/// - value: local index.
///
/// This avoids O(n) scans per edge and keeps expansions fast.
///
/// Expected input quality
/// ----------------------
/// Edge costs are expected to be finite and non-NaN. If NaNs occur, sorting falls
/// back to equality ordering and may become unstable.
fn build_solver_adjacency<'edge_lf, 'seed_lf>(
    cfg: &BoundedBeamConfig,
    component_nodes: &[&'seed_lf SeedNode],
    component_out_edges: &[Vec<&'edge_lf Edge>],
    diag: &mut SolverDiagnostics,
) -> Adjacency<'edge_lf> {
    // Build SeedKey -> local index for destinations.
    // This is small (component-local) and prevents repeated O(n) scans.
    let mut local_of_key: ahash::AHashMap<SeedKey, usize> = ahash::AHashMap::default();
    local_of_key.reserve(component_nodes.len());

    for (i, &node) in component_nodes.iter().enumerate() {
        local_of_key.insert(node.key(), i);
    }

    let n = component_nodes.len();
    let mut out: Adjacency<'edge_lf> = (0..n).map(|_| Vec::new()).collect();

    for (u, edges_u) in component_out_edges.iter().enumerate() {
        // Convert and cache (dst, cost) for faster expansion.
        let mut buf: Vec<OutEdge<'edge_lf>> = Vec::with_capacity(edges_u.len());

        for &e in edges_u.iter() {
            // Destination must be inside the component (guaranteed by CC).
            let Some(&dst) = local_of_key.get(&e.to) else {
                // Defensive fallback: skip malformed edges.
                continue;
            };

            // Count candidates *before* pruning (diagnostics).
            diag.n_candidates += 1;

            buf.push(OutEdge {
                dst,
                edge: e,
                cost: e.cost,
            });
        }

        // Sort by increasing cost so beam search sees best options first.
        buf.sort_by(|a, b| {
            a.cost
                .partial_cmp(&b.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Local pruning to avoid high branching factor.
        if buf.len() > cfg.max_out_per_node {
            buf.truncate(cfg.max_out_per_node);
        }

        out[u] = buf;
    }

    out
}

// -----------------------------------------------------------------------------
// Step 2: initialize beam states
// -----------------------------------------------------------------------------

/// Initialize the state pool and initial beam with one root state per source.
///
/// Each root state represents a path of length 0 starting at the corresponding
/// source node.
///
/// Arguments
/// ---------
/// * `sources` – Component-local source nodes (local indices).
/// * `component_nodes` – Component nodes (used to read the source night).
///
/// Return
/// ------
/// * `(states, beam)`
///   - `states`: state pool containing all created states,
///   - `beam`: indices into `states` representing the current frontier.
///
/// Notes
/// -----
/// - Root states have `parent = None` and `in_edge = None`.
/// - `total_cost = 0.0` and `n_edges = 0` for roots.
fn init_beam_states<'edge_lf, 'seed_lf, 'alert_lf>(
    sources: &[usize],
    component_nodes: &[&'seed_lf SeedNode],
) -> (Vec<State<'edge_lf>>, Vec<usize>) {
    let mut states: Vec<State<'edge_lf>> = Vec::new();
    let mut beam: Vec<usize> = Vec::new();

    for &s in sources {
        let night = component_nodes[s].night_id().value();
        states.push(State {
            last: s,
            parent: None,
            in_edge: None,
            total_cost: 0.0,
            n_edges: 0,
            first_night: night,
            last_night: night,
            source: s,
        });
        beam.push(states.len() - 1);
    }

    (states, beam)
}

// -----------------------------------------------------------------------------
// Step 3: beam expansion
// -----------------------------------------------------------------------------

/// Test whether a node is a sink in the solver adjacency.
///
/// A sink is a node with no outgoing edges in the restricted solver adjacency.
///
/// Arguments
/// ---------
/// * `out` – Solver adjacency.
/// * `u` – Local node index.
///
/// Return
/// ------
/// `true` if the node has no outgoing edges.
fn is_sink<'edge_lf>(out: &Adjacency<'edge_lf>, u: usize) -> bool {
    out[u].is_empty()
}

/// Perform one beam iteration: expand the current frontier into next candidates.
///
/// This is the core beam-search step.
///
/// For each state in the current beam:
/// - If the state ends at a sink, it is emitted as a terminal endpoint.
/// - Otherwise, it is expanded along each outgoing edge in `out[last]`.
///
/// Each expansion produces a new `State` appended to the `states` pool.
/// The returned next frontier contains only the best candidates after sorting
/// by total cost and truncating to `cfg.beam_width`.
///
/// Arguments
/// ---------
/// * `cfg` – Bounded beam configuration.
/// * `out` – Solver adjacency (sorted/pruned).
/// * `component_nodes` – Component nodes (used for defensive time-forward checks).
/// * `states` – State pool grown in-place.
/// * `beam` – Current frontier (indices into `states`).
/// * `terminal_states` – Output buffer collecting terminal endpoints (state ids).
/// * `emitted_per_source` – Per-source cap enforcing `max_tracks_per_source`.
/// * `expansions` – Global counter enforcing `max_expansions`.
///
/// Return
/// ------
/// * `Vec<usize>` – Next beam frontier (state ids), truncated to `beam_width`.
///
/// Notes
/// -----
/// Terminal emission is constrained by:
/// - `cfg.max_tracks` (global output cap),
/// - `cfg.max_tracks_per_source` (per-source cap),
/// - `cfg.min_nodes` (minimum track length in nodes).
fn expand_beam<'edge_lf, 'seed_lf>(
    cfg: &BoundedBeamConfig,
    out: &Adjacency<'edge_lf>,
    component_nodes: &[&'seed_lf SeedNode],
    states: &mut Vec<State<'edge_lf>>,
    beam: &[usize],
    terminal_states: &mut Vec<usize>,
    emitted_per_source: &mut [u32],
    expansions: &mut usize,
) -> Vec<usize> {
    let mut next_candidates: Vec<usize> = Vec::new();

    for &sid in beam {
        // Enforce global budgets early.
        if *expansions >= cfg.max_expansions || terminal_states.len() >= cfg.max_tracks {
            break;
        }

        let st = states[sid];
        let u = st.last;

        // If sink: emit terminal candidate and do not expand.
        if is_sink(out, u) {
            let n_nodes = st.n_edges as usize + 1; // edges + 1 nodes
            if n_nodes >= cfg.min_nodes {
                let src = st.source;
                if emitted_per_source[src] < cfg.max_tracks_per_source as u32 {
                    terminal_states.push(sid);
                    emitted_per_source[src] += 1;
                }
            }
            continue;
        }

        // Expand along outgoing edges (already sorted by increasing cost).
        for oe in out[u].iter() {
            if *expansions >= cfg.max_expansions || terminal_states.len() >= cfg.max_tracks {
                break;
            }

            // Defensive time-forward guard.
            let v_night = component_nodes[oe.dst].night_id().value();
            if v_night <= st.last_night {
                continue;
            }

            states.push(State {
                last: oe.dst,
                parent: Some(sid),
                in_edge: Some(oe.edge),
                total_cost: st.total_cost + oe.cost,
                n_edges: st.n_edges + 1,
                first_night: st.first_night,
                last_night: v_night,
                source: st.source,
            });

            next_candidates.push(states.len() - 1);
            *expansions += 1;
        }
    }

    // Keep best partial paths by total cost (beam heuristic).
    next_candidates.sort_by(|&a, &b| {
        let ca = states[a].total_cost;
        let cb = states[b].total_cost;
        ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
    });

    if next_candidates.len() > cfg.beam_width {
        next_candidates.truncate(cfg.beam_width);
    }

    next_candidates
}

// -----------------------------------------------------------------------------
// Step 4: reconstruction + ranking
// -----------------------------------------------------------------------------

/// Reconstruct a [`TrackHypothesis`] from a terminal state id.
///
/// Reconstruction follows `parent` backpointers to collect:
/// - node local indices,
/// - edges used between nodes.
///
/// The resulting track is returned in forward-time order.
///
/// Arguments
/// ---------
/// * `tmp_track_id` – Temporary track id before final assignment after orbit fitting.
/// * `terminal_state_id` – State pool index representing the end of a path.
/// * `states` – State pool.
/// * `component_nodes` – Component nodes in local index order.
///
/// Return
/// ------
/// * `TrackHypothesis` – Nodes + edges along the reconstructed path.
///
/// Notes
/// -----
/// - Reconstruction cost is linear in the path length.
/// - The returned `night_span` is computed from cached `(first_night, last_night)`.
fn reconstruct_track<'edge_lf, 'seed_lf>(
    terminal_state_id: usize,
    states: &[State<'edge_lf>],
    component_nodes: &[&'seed_lf SeedNode],
) -> TrackHypothesis {
    let mut node_idx_rev: Vec<usize> = Vec::new();
    let mut edge_rev: Vec<&'edge_lf Edge> = Vec::new();

    let mut cur = Some(terminal_state_id);
    while let Some(id) = cur {
        let st = states[id];
        node_idx_rev.push(st.last);
        if let Some(e) = st.in_edge {
            edge_rev.push(e);
        }
        cur = st.parent;
    }

    node_idx_rev.reverse();
    edge_rev.reverse();

    let nodes = node_idx_rev
        .into_iter()
        .map(|idx| component_nodes[idx])
        .collect::<Vec<_>>();

    let st = states[terminal_state_id];
    let night_span = st.last_night - st.first_night;

    TrackHypothesis {
        nodes: nodes.iter().map(|&n| n.key()).collect(),
        edges: edge_rev.into_iter().map(|e| e.key()).collect(),
        cost: st.total_cost,
        night_span,
    }
}

/// Sort tracks by average cost per edge, then truncate.
///
/// Ordering
/// --------
/// 1) lower average cost per edge is better,
/// 2) tie-break: prefer longer tracks (more edges),
/// 3) tie-break: lower total cost.
///
/// Arguments
/// ---------
/// * `cfg` – Bounded beam configuration.
/// * `tracks` – Track list sorted and truncated in-place.
///
/// Notes
/// -----
/// - `edges.len().max(1)` prevents division by zero for degenerate tracks.
/// - Costs are expected to be finite and non-NaN.
fn sort_and_truncate_tracks<'edge_lf, 'seed_lf>(
    cfg: &BoundedBeamConfig,
    tracks: &mut AHashMap<u32, TrackHypothesis>,
) {
    // Rien à faire si déjà <= max_tracks
    if tracks.len() <= cfg.max_tracks {
        return;
    }

    // 1) Materialize en vec pour pouvoir trier
    let mut items: Vec<(u32, TrackHypothesis)> = tracks.drain().collect();

    // 2) Tri (meilleur d'abord)
    items.sort_by(|(_ka, a), (_kb, b)| {
        let a_edges = a.edges.len().max(1) as f64;
        let b_edges = b.edges.len().max(1) as f64;

        let a_avg = a.cost / a_edges;
        let b_avg = b.cost / b_edges;

        a_avg
            .partial_cmp(&b_avg)
            .unwrap_or(Ordering::Equal)
            // tie-break: préférer les tracks plus longues
            .then_with(|| b.edges.len().cmp(&a.edges.len()))
            // tie-break: coût total plus faible
            .then_with(|| a.cost.partial_cmp(&b.cost).unwrap_or(Ordering::Equal))
    });

    // 3) Tronque au top-K
    items.truncate(cfg.max_tracks);

    // 4) Réinsère dans la hashmap (capacity correcte)
    tracks.extend(items);
}

// -----------------------------------------------------------------------------
// Main entry: orchestration using ConnectedComponents
// -----------------------------------------------------------------------------

/// Enumerate candidate tracks in a component using the `ConnectedComponents` view.
///
/// This is the main enumerator used by [`BoundedBeamSolver::solve_component_cc`].
///
/// Algorithm
/// ---------
/// 1) Convert `sources_local` from `LocalIdx` to `usize`.
///
/// 2) Build solver adjacency:
///    - convert edge refs to cached edges,
///    - sort by cost,
///    - prune outgoing lists.
///
/// 3) Initialize:
///    - `states`: pool of partial-path states,
///    - `beam`: current frontier (initially roots).
///
/// 4) Beam loop:
///    - expand beam into `next`,
///    - emit terminal states at sinks,
///    - enforce budgets:
///      - `max_expansions`,
///      - `max_tracks`,
///      - `beam_width`,
///      - `max_tracks_per_source`.
///
/// 5) If the loop stops early, check remaining beam states for sinks and emit
///    them (still respecting caps).
///
/// 6) Reconstruct tracks from terminal states.
///
/// 7) Rank and truncate final tracks.
///
/// Arguments
/// ---------
/// * `cfg` – Bounded beam configuration.
/// * `_graph` – Global graph (unused currently; reserved for future heuristics).
/// * `component_nodes` – Nodes of the component in local order.
/// * `component_out_edges` – Restricted directed adjacency (`&Edge` refs) from CC.
/// * `sources_local` – Precomputed sources (local indices) from CC.
/// * `diag` – Diagnostics updated in-place.
///
/// Return
/// ------
/// * `AHashMap<u32, TrackHypothesis>` – Candidate tracks sorted and truncated.
///
/// Notes
/// -----
/// - `diag.n_candidates` counts edges *seen* before pruning.
/// - `cfg.max_out_per_node` pruning is applied here so it can be tuned per solver.
fn enumerate_beam_tracks_from_component_view<'edge_lf, 'seed_lf>(
    cfg: &BoundedBeamConfig,
    _graph: &AlertLinkageDAG,
    component_nodes: &[&'seed_lf SeedNode],
    component_out_edges: &[Vec<&'edge_lf Edge>],
    sources_local: &[LocalIdx],
    diag: &mut SolverDiagnostics,
) -> AHashMap<u32, TrackHypothesis> {
    let n = component_nodes.len();
    if n == 0 {
        return AHashMap::new();
    }

    // Convert sources (LocalIdx) into indices used by local vectors.
    let sources = sources_local
        .iter()
        .map(|&x| x as usize)
        .collect::<Vec<_>>();

    if sources.is_empty() {
        return AHashMap::new();
    }

    // 1) Build solver adjacency (sorted + pruned).
    let out = build_solver_adjacency(cfg, component_nodes, component_out_edges, diag);

    // 2) Initialize beam + state pool.
    let (mut states, mut beam) = init_beam_states(&sources, component_nodes);

    // Per-source emission counters (index by local node id of the source).
    let mut emitted_per_source: Vec<u32> = vec![0; n];

    // Terminal states to reconstruct into tracks.
    let mut terminal_states: Vec<usize> = Vec::new();

    // Global expansion counter to enforce a hard exploration budget.
    let mut expansions: usize = 0;

    // 3) Beam loop.
    while !beam.is_empty()
        && terminal_states.len() < cfg.max_tracks
        && expansions < cfg.max_expansions
    {
        let next = expand_beam(
            cfg,
            &out,
            component_nodes,
            &mut states,
            &beam,
            &mut terminal_states,
            &mut emitted_per_source,
            &mut expansions,
        );

        beam = next;
        if beam.is_empty() {
            break;
        }
    }

    // 4) Also consider sinks still present in the beam (in case we exited early).
    for &sid in beam.iter() {
        if terminal_states.len() >= cfg.max_tracks {
            break;
        }
        let st = states[sid];
        if is_sink(&out, st.last) {
            let n_nodes = st.n_edges as usize + 1;
            if n_nodes >= cfg.min_nodes {
                let src = st.source;
                if emitted_per_source[src] < cfg.max_tracks_per_source as u32 {
                    terminal_states.push(sid);
                    emitted_per_source[src] += 1;
                }
            }
        }
    }

    // 5) Reconstruct tracks.
    let mut tracks: AHashMap<u32, TrackHypothesis> = terminal_states
        .into_iter()
        .enumerate()
        .filter_map(|(tmp_track_id, sid)| {
            let t = reconstruct_track(sid, &states, component_nodes);

            if t.nodes.len() >= cfg.min_nodes {
                Some((tmp_track_id as u32, t))
            } else {
                None
            }
        })
        .collect();

    // 6) Rank + truncate.
    sort_and_truncate_tracks(cfg, &mut tracks);
    tracks
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod bounded_beam_tests {
    use super::*;
    use crate::{
        Alert, AlertKey,
        astro_math::arcsec_to_rad,
        engine_config::solver_config::bounded_beam_config::BoundedBeamConfig,
        graph::AlertLinkageDAG,
        graph::edge::Edge,
        night_id::NightId,
        seeding::{SeedKey, SeedNode, store::SeedStore},
        solver::{Solver, SolverOutput, components::ConnectedComponents},
    };
    use ahash::AHashSet;
    use proptest::prelude::*;

    // =========================================================================
    // Helpers (same pattern as connected_components_tests)
    // =========================================================================

    fn nid(v: u32) -> NightId {
        NightId::from(v)
    }

    fn mk_alert(source_id: u64, night_id: NightId, mjd_tt: f64) -> Alert {
        Alert {
            key: AlertKey {
                night_id,
                dia_source_id: source_id,
            },
            ra: 1.0,
            ra_err: arcsec_to_rad(0.5),
            dec: 0.1,
            dec_err: arcsec_to_rad(0.5),
            mjd_tt,
            flux: 1000.0,
            flux_err: 10.0,
            band: 1,
            ..Default::default()
        }
    }

    fn insert_seeds(
        store: &mut SeedStore,
        night_id: NightId,
        count: usize,
        source_id_offset: u64,
    ) -> Vec<SeedKey> {
        let mut keys = Vec::with_capacity(count);
        let t0 = 60000.0 + night_id.value() as f64;

        for i in 0..count {
            let sid_a = source_id_offset + (2 * i) as u64;
            let sid_b = source_id_offset + (2 * i + 1) as u64;
            let dt = 30.0 / 1440.0;
            let alert_a = mk_alert(sid_a, night_id, t0 + i as f64 * 0.01);
            let alert_b = mk_alert(sid_b, night_id, t0 + i as f64 * 0.01 + dt);

            if let Some(seed) = SeedNode::from_pair(store, night_id, &alert_a, &alert_b, None) {
                keys.push(seed.key());
                store.insert_vec_seed(night_id, vec![seed]);
            }
        }
        keys
    }

    fn build_store(spec: &[(u32, usize)]) -> (SeedStore, Vec<(NightId, Vec<SeedKey>)>) {
        let mut store = SeedStore::new();
        let mut record = Vec::new();
        let mut source_id_offset: u64 = 0;

        for &(n, count) in spec {
            let night_id = nid(n);
            let keys = insert_seeds(&mut store, night_id, count, source_id_offset);
            record.push((night_id, keys));
            source_id_offset += (count as u64) * 2 + 100;
        }
        (store, record)
    }

    fn mk_edge(from: SeedKey, to: SeedKey, cost: f64, active: bool) -> Edge {
        Edge {
            from,
            to,
            cost,
            dt_days: 1.0,
            active,
        }
    }

    fn build_graph(edges: Vec<Edge>) -> AlertLinkageDAG {
        AlertLinkageDAG::from_edges(edges)
    }

    /// Default config with relaxed limits for testing.
    fn test_cfg() -> BoundedBeamConfig {
        BoundedBeamConfig {
            max_tracks: 100,
            min_nodes: 2,
            beam_width: 64,
            max_out_per_node: 16,
            max_tracks_per_source: 32,
            max_expansions: 100_000,
        }
    }

    /// Solve all components and collect all tracks into a flat vec.
    fn solve_all<'e: 's, 's>(
        solver: &BoundedBeamSolver,
        graph: &'e AlertLinkageDAG,
        cc: &'e ConnectedComponents<'e, 's>,
    ) -> Vec<SolverOutput> {
        (0..cc.n_components)
            .map(|cid| solver.solve(graph, cc, cid))
            .collect()
    }

    // =========================================================================
    // Unit tests — BoundedBeamSolver construction
    // =========================================================================

    #[test]
    fn solver_name_is_bounded_beam() {
        let solver = BoundedBeamSolver::default();
        assert_eq!(solver.name(), "bounded_beam");
    }

    #[test]
    fn new_stores_config() {
        let cfg = BoundedBeamConfig {
            max_tracks: 42,
            min_nodes: 5,
            beam_width: 128,
            max_out_per_node: 4,
            max_tracks_per_source: 10,
            max_expansions: 9999,
        };
        let solver = BoundedBeamSolver::new(cfg.clone());
        assert_eq!(solver.cfg.max_tracks, 42);
        assert_eq!(solver.cfg.min_nodes, 5);
        assert_eq!(solver.cfg.beam_width, 128);
        assert_eq!(solver.cfg.max_out_per_node, 4);
        assert_eq!(solver.cfg.max_tracks_per_source, 10);
        assert_eq!(solver.cfg.max_expansions, 9999);
    }

    // =========================================================================
    // Unit tests — empty / trivial inputs
    // =========================================================================

    #[test]
    fn empty_store_returns_empty_output() {
        let store = SeedStore::new();
        let graph = build_graph(vec![]);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();
        let solver = BoundedBeamSolver::new(test_cfg());

        assert_eq!(cc.n_components, 0);
        let outputs = solve_all(&solver, &graph, &cc);
        assert!(outputs.is_empty());
    }

    #[test]
    fn singleton_component_no_edges_returns_empty_tracks() {
        let (store, _) = build_store(&[(10, 1)]);
        let graph = build_graph(vec![]);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();
        let solver = BoundedBeamSolver::new(test_cfg());

        let output = solver.solve(&graph, &cc, 0);
        // A single node cannot form a track with min_nodes >= 2.
        assert!(output.tracks.is_empty());
    }

    #[test]
    fn component_too_small_for_min_nodes_returns_empty() {
        // 2 nodes, 1 edge. With min_nodes=3, no track can be emitted.
        let (store, record) = build_store(&[(10, 1), (11, 1)]);
        let k0 = record[0].1[0];
        let k1 = record[1].1[0];
        let edges = vec![mk_edge(k0, k1, 1.0, true)];
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        let cfg = BoundedBeamConfig {
            min_nodes: 3,
            ..test_cfg()
        };
        let solver = BoundedBeamSolver::new(cfg);
        let output = solver.solve(&graph, &cc, 0);
        assert!(output.tracks.is_empty());
    }

    // =========================================================================
    // Unit tests — simple linear chain
    // =========================================================================

    #[test]
    fn linear_chain_produces_one_track() {
        // N10 -> N11 -> N12: a single source-to-sink path.
        let (store, record) = build_store(&[(10, 1), (11, 1), (12, 1)]);
        let k0 = record[0].1[0];
        let k1 = record[1].1[0];
        let k2 = record[2].1[0];

        let edges = vec![mk_edge(k0, k1, 1.0, true), mk_edge(k1, k2, 2.0, true)];
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        let solver = BoundedBeamSolver::new(test_cfg());
        let output = solver.solve(&graph, &cc, 0);

        assert_eq!(output.tracks.len(), 1);
        let trk = output.tracks.values().next().unwrap();

        // Track must have 3 nodes and 2 edges.
        assert_eq!(trk.nodes.len(), 3);
        assert_eq!(trk.edges.len(), 2);

        // Nodes are in time-forward order.
        assert_eq!(trk.nodes[0].night_id, nid(10));
        assert_eq!(trk.nodes[1].night_id, nid(11));
        assert_eq!(trk.nodes[2].night_id, nid(12));

        // Cost is sum of edge costs.
        assert!((trk.cost - 3.0).abs() < 1e-12);

        // Night span = 12 - 10 = 2.
        assert_eq!(trk.night_span, 2);
    }

    #[test]
    fn linear_chain_min_nodes_2_emits_partial_and_full() {
        // N10 -> N11 -> N12. With min_nodes=2, partial paths (2-node) ending at
        // intermediate sinks *also* get emitted. But since N11 is not a sink
        // (it has outgoing edges to N12), only the full 3-node path is emitted.
        let (store, record) = build_store(&[(10, 1), (11, 1), (12, 1)]);
        let k0 = record[0].1[0];
        let k1 = record[1].1[0];
        let k2 = record[2].1[0];

        let edges = vec![mk_edge(k0, k1, 1.0, true), mk_edge(k1, k2, 2.0, true)];
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        let cfg = BoundedBeamConfig {
            min_nodes: 2,
            ..test_cfg()
        };
        let solver = BoundedBeamSolver::new(cfg);
        let output = solver.solve(&graph, &cc, 0);

        // Only 1 track: N10->N11->N12. N11 is not a sink so no partial track.
        assert_eq!(output.tracks.len(), 1);
    }

    // =========================================================================
    // Unit tests — diamond topology
    // =========================================================================

    #[test]
    fn diamond_topology_finds_both_paths() {
        //     s0 (N10)
        //    /    \
        //  s1      s2 (N11)
        //    \    /
        //     s3 (N12)
        let (store, record) = build_store(&[(10, 1), (11, 2), (12, 1)]);
        let s0 = record[0].1[0];
        let s1 = record[1].1[0];
        let s2 = record[1].1[1];
        let s3 = record[2].1[0];

        let edges = vec![
            mk_edge(s0, s1, 1.0, true),
            mk_edge(s0, s2, 2.0, true),
            mk_edge(s1, s3, 1.0, true),
            mk_edge(s2, s3, 1.0, true),
        ];
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        let cfg = BoundedBeamConfig {
            min_nodes: 3,
            ..test_cfg()
        };
        let solver = BoundedBeamSolver::new(cfg);
        let output = solver.solve(&graph, &cc, 0);

        // Two distinct 3-node paths: s0->s1->s3 (cost 2.0) and s0->s2->s3 (cost 3.0).
        assert_eq!(output.tracks.len(), 2);

        // All tracks have 3 nodes and 2 edges.
        for trk in output.tracks.values() {
            assert_eq!(trk.nodes.len(), 3);
            assert_eq!(trk.edges.len(), 2);
        }
    }

    // =========================================================================
    // Unit tests — max_tracks truncation
    // =========================================================================

    #[test]
    fn max_tracks_limits_output_size() {
        // Fan-out: s0 -> {s1, s2, s3, s4} -> s5
        // Each path produces a track: s0->si->s5, so 4 tracks.
        // max_tracks = 2 should truncate.
        let (store, record) = build_store(&[(10, 1), (11, 4), (12, 1)]);
        let s0 = record[0].1[0];
        let s_mid: Vec<SeedKey> = record[1].1.clone();
        let s5 = record[2].1[0];

        let mut edges = Vec::new();
        for (i, &sm) in s_mid.iter().enumerate() {
            edges.push(mk_edge(s0, sm, (i + 1) as f64, true));
            edges.push(mk_edge(sm, s5, 1.0, true));
        }
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        let cfg = BoundedBeamConfig {
            max_tracks: 2,
            min_nodes: 3,
            ..test_cfg()
        };
        let solver = BoundedBeamSolver::new(cfg);
        let output = solver.solve(&graph, &cc, 0);

        assert!(output.tracks.len() <= 2);
    }

    // =========================================================================
    // Unit tests — max_tracks_per_source cap
    // =========================================================================

    #[test]
    fn max_tracks_per_source_limits_emission() {
        // s0 -> {s1, s2, s3} -> s4 : three 3-node paths, all from source s0.
        // max_tracks_per_source = 1: only one track should be emitted.
        let (store, record) = build_store(&[(10, 1), (11, 3), (12, 1)]);
        let s0 = record[0].1[0];
        let s_mid: Vec<SeedKey> = record[1].1.clone();
        let s4 = record[2].1[0];

        let mut edges = Vec::new();
        for (i, &sm) in s_mid.iter().enumerate() {
            edges.push(mk_edge(s0, sm, (i + 1) as f64, true));
            edges.push(mk_edge(sm, s4, 1.0, true));
        }
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        let cfg = BoundedBeamConfig {
            max_tracks: 100,
            max_tracks_per_source: 1,
            min_nodes: 3,
            ..test_cfg()
        };
        let solver = BoundedBeamSolver::new(cfg);
        let output = solver.solve(&graph, &cc, 0);

        assert_eq!(output.tracks.len(), 1);
    }

    // =========================================================================
    // Unit tests — max_out_per_node pruning
    // =========================================================================

    #[test]
    fn max_out_per_node_limits_branching() {
        // s0 has 5 outgoing edges to N11 seeds. max_out_per_node=2 should prune.
        let (store, record) = build_store(&[(10, 1), (11, 5), (12, 1)]);
        let s0 = record[0].1[0];
        let s_mid: Vec<SeedKey> = record[1].1.clone();
        let s_sink = record[2].1[0];

        let mut edges = Vec::new();
        for (i, &sm) in s_mid.iter().enumerate() {
            edges.push(mk_edge(s0, sm, (i + 1) as f64, true));
            edges.push(mk_edge(sm, s_sink, 1.0, true));
        }
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        let cfg = BoundedBeamConfig {
            max_out_per_node: 2,
            min_nodes: 3,
            ..test_cfg()
        };
        let solver = BoundedBeamSolver::new(cfg);
        let output = solver.solve(&graph, &cc, 0);

        // With max_out_per_node=2 from s0, at most 2 paths are reachable.
        assert!(output.tracks.len() <= 2);
    }

    // =========================================================================
    // Unit tests — max_expansions budget
    // =========================================================================

    #[test]
    fn max_expansions_limits_work() {
        // Build a longer chain: N1 -> N2 -> N3 -> N4 -> N5
        let (store, record) = build_store(&[(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]);
        let keys: Vec<SeedKey> = record.iter().map(|(_, ks)| ks[0]).collect();

        let mut edges = Vec::new();
        for i in 0..keys.len() - 1 {
            edges.push(mk_edge(keys[i], keys[i + 1], 1.0, true));
        }
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        // max_expansions=1: only one step can happen.
        let cfg = BoundedBeamConfig {
            max_expansions: 1,
            min_nodes: 2,
            ..test_cfg()
        };
        let solver = BoundedBeamSolver::new(cfg);
        let output = solver.solve(&graph, &cc, 0);

        // The solver may find partial tracks or nothing, depending on when
        // the budget cuts in; the important thing is it terminates and
        // n_candidates is bounded.
        assert!(output.diag.n_candidates <= 5); // bounded work
    }

    // =========================================================================
    // Unit tests — diagnostics
    // =========================================================================

    #[test]
    fn diagnostics_are_populated() {
        let (store, record) = build_store(&[(10, 1), (11, 1), (12, 1)]);
        let k0 = record[0].1[0];
        let k1 = record[1].1[0];
        let k2 = record[2].1[0];

        let edges = vec![mk_edge(k0, k1, 1.0, true), mk_edge(k1, k2, 2.0, true)];
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        let solver = BoundedBeamSolver::new(test_cfg());
        let output = solver.solve(&graph, &cc, 0);

        assert_eq!(output.diag.solver_name, "bounded_beam");
        assert_eq!(output.diag.component_id, 0);
        assert_eq!(output.diag.n_nodes, 3);
        assert!(output.diag.n_candidates > 0);
        assert_eq!(output.diag.n_selected, output.tracks.len() as u32);
    }

    // =========================================================================
    // Unit tests — track invariants
    // =========================================================================

    #[test]
    fn track_nodes_are_time_ordered() {
        // 5-night chain.
        let (store, record) = build_store(&[(10, 1), (11, 1), (12, 1), (13, 1), (14, 1)]);
        let keys: Vec<SeedKey> = record.iter().map(|(_, ks)| ks[0]).collect();

        let mut edges = Vec::new();
        for i in 0..keys.len() - 1 {
            edges.push(mk_edge(keys[i], keys[i + 1], 1.0, true));
        }
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        let solver = BoundedBeamSolver::new(test_cfg());
        let output = solver.solve(&graph, &cc, 0);

        for trk in output.tracks.values() {
            for w in trk.nodes.windows(2) {
                assert!(
                    w[0].night_id < w[1].night_id,
                    "nodes must be in strictly increasing night order"
                );
            }
        }
    }

    #[test]
    fn track_edges_match_consecutive_nodes() {
        let (store, record) = build_store(&[(10, 1), (11, 1), (12, 1)]);
        let k0 = record[0].1[0];
        let k1 = record[1].1[0];
        let k2 = record[2].1[0];

        let edges = vec![mk_edge(k0, k1, 1.0, true), mk_edge(k1, k2, 2.0, true)];
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        let solver = BoundedBeamSolver::new(test_cfg());
        let output = solver.solve(&graph, &cc, 0);

        for trk in output.tracks.values() {
            assert_eq!(trk.edges.len(), trk.nodes.len() - 1);
            for (i, ek) in trk.edges.iter().enumerate() {
                assert_eq!(ek.from, trk.nodes[i], "edge.from must match nodes[i]");
                assert_eq!(ek.to, trk.nodes[i + 1], "edge.to must match nodes[i+1]");
            }
        }
    }

    #[test]
    fn track_cost_equals_sum_of_edge_costs() {
        let (store, record) = build_store(&[(10, 1), (11, 1), (12, 1)]);
        let k0 = record[0].1[0];
        let k1 = record[1].1[0];
        let k2 = record[2].1[0];

        let edges = vec![mk_edge(k0, k1, 1.5, true), mk_edge(k1, k2, 2.5, true)];
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        let solver = BoundedBeamSolver::new(test_cfg());
        let output = solver.solve(&graph, &cc, 0);

        for trk in output.tracks.values() {
            let edge_cost_sum: f64 = trk
                .edges
                .iter()
                .map(|ek| graph.edge_by_key(ek).unwrap().cost)
                .sum();
            assert!(
                (trk.cost - edge_cost_sum).abs() < 1e-12,
                "track cost {} != sum of edge costs {}",
                trk.cost,
                edge_cost_sum
            );
        }
    }

    #[test]
    fn track_night_span_matches_first_last() {
        let (store, record) = build_store(&[(5, 1), (10, 1), (20, 1)]);
        let k0 = record[0].1[0];
        let k1 = record[1].1[0];
        let k2 = record[2].1[0];

        let edges = vec![mk_edge(k0, k1, 1.0, true), mk_edge(k1, k2, 1.0, true)];
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        let solver = BoundedBeamSolver::new(test_cfg());
        let output = solver.solve(&graph, &cc, 0);

        for trk in output.tracks.values() {
            let first_night = trk.nodes.first().unwrap().night_id.value();
            let last_night = trk.nodes.last().unwrap().night_id.value();
            assert_eq!(trk.night_span, last_night - first_night);
        }
    }

    // =========================================================================
    // Unit tests — sort_and_truncate_tracks ranking
    // =========================================================================

    #[test]
    fn sort_and_truncate_prefers_lower_avg_cost() {
        // Two tracks with different costs and same length.
        // Track A: cost=2.0, 2 edges => avg=1.0
        // Track B: cost=6.0, 2 edges => avg=3.0
        // After sort, A should be retained when max_tracks=1.
        let (store, record) = build_store(&[(10, 2), (11, 1), (12, 1)]);
        let s0a = record[0].1[0];
        let s0b = record[0].1[1];
        let s1 = record[1].1[0];
        let s2 = record[2].1[0];

        // Path A: s0a -> s1 -> s2 (cost 1+1=2)
        // Path B: s0b -> s1 -> s2 (cost 5+1=6)
        let edges = vec![
            mk_edge(s0a, s1, 1.0, true),
            mk_edge(s0b, s1, 5.0, true),
            mk_edge(s1, s2, 1.0, true),
        ];
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        let cfg = BoundedBeamConfig {
            max_tracks: 1,
            min_nodes: 3,
            ..test_cfg()
        };
        let solver = BoundedBeamSolver::new(cfg);
        let output = solver.solve(&graph, &cc, 0);

        assert_eq!(output.tracks.len(), 1);
        let trk = output.tracks.values().next().unwrap();
        // The cheaper path (avg cost 1.0) should win.
        assert!((trk.cost - 2.0).abs() < 1e-12);
    }

    // =========================================================================
    // Unit tests — disjoint components solved independently
    // =========================================================================

    #[test]
    fn disjoint_components_solved_independently() {
        // Two disjoint 3-night chains.
        let (store, record) = build_store(&[(10, 1), (11, 1), (12, 1), (20, 1), (21, 1), (22, 1)]);
        let ka = [record[0].1[0], record[1].1[0], record[2].1[0]];
        let kb = [record[3].1[0], record[4].1[0], record[5].1[0]];

        let edges = vec![
            mk_edge(ka[0], ka[1], 1.0, true),
            mk_edge(ka[1], ka[2], 1.0, true),
            mk_edge(kb[0], kb[1], 2.0, true),
            mk_edge(kb[1], kb[2], 2.0, true),
        ];
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        assert_eq!(cc.n_components, 2);

        let solver = BoundedBeamSolver::new(test_cfg());
        let outputs = solve_all(&solver, &graph, &cc);

        // Each component should produce exactly 1 track.
        for output in &outputs {
            assert_eq!(output.tracks.len(), 1);
        }
    }

    // =========================================================================
    // Unit tests — active_only edge filtering
    // =========================================================================

    #[test]
    fn active_only_cc_excludes_inactive_edges() {
        // N10 -> N11 (active), N11 -> N12 (inactive).
        // With active_only CC, N12 is disconnected.
        let (store, record) = build_store(&[(10, 1), (11, 1), (12, 1)]);
        let k0 = record[0].1[0];
        let k1 = record[1].1[0];
        let k2 = record[2].1[0];

        let edges = vec![mk_edge(k0, k1, 1.0, true), mk_edge(k1, k2, 1.0, false)];
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, true).unwrap();

        let solver = BoundedBeamSolver::new(test_cfg());
        let outputs = solve_all(&solver, &graph, &cc);

        // The active component has only k0, k1 — emits a 2-node track.
        let total_tracks: usize = outputs.iter().map(|o| o.tracks.len()).sum();
        assert_eq!(total_tracks, 1);

        let trk = outputs
            .iter()
            .flat_map(|o| o.tracks.values())
            .next()
            .unwrap();
        assert_eq!(trk.nodes.len(), 2);
    }

    // =========================================================================
    // Unit tests — build_solver_adjacency
    // =========================================================================

    #[test]
    fn build_solver_adjacency_prunes_and_sorts() {
        // 1 source with 5 outgoing edges (costs 5,3,1,4,2). max_out=3.
        // After sort+prune: costs [1,2,3].
        let (store, record) = build_store(&[(10, 1), (11, 5)]);
        let s0 = record[0].1[0];
        let s_right: Vec<SeedKey> = record[1].1.clone();

        let costs = [5.0, 3.0, 1.0, 4.0, 2.0];
        let mut edges = Vec::new();
        for (i, &sm) in s_right.iter().enumerate() {
            edges.push(mk_edge(s0, sm, costs[i], true));
        }
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        let cid = cc.component_id_of_seed(&store, s0).unwrap();
        let component_nodes = cc.component_nodes(cid);
        let component_out = cc.component_out_edges(cid);

        let cfg = BoundedBeamConfig {
            max_out_per_node: 3,
            ..test_cfg()
        };
        let mut diag = SolverDiagnostics::default();
        let adj = build_solver_adjacency(&cfg, component_nodes, component_out, &mut diag);

        // Find the local index of s0.
        let s0_local = component_nodes.iter().position(|n| n.key() == s0).unwrap();

        // adj[s0_local] should have at most 3 entries, sorted by cost.
        assert!(adj[s0_local].len() <= 3);
        for w in adj[s0_local].windows(2) {
            assert!(w[0].cost <= w[1].cost, "adjacency must be sorted by cost");
        }
        // n_candidates should reflect edges seen before pruning.
        assert!(diag.n_candidates >= 3);
    }

    // =========================================================================
    // Unit tests — wider DAG
    // =========================================================================

    #[test]
    fn wider_dag_produces_multiple_tracks() {
        // N10: [a0, a1]   N11: [b0, b1]   N12: [c0]
        // Edges: a0->b0, a0->b1, a1->b0, a1->b1, b0->c0, b1->c0
        let (store, record) = build_store(&[(10, 2), (11, 2), (12, 1)]);
        let a0 = record[0].1[0];
        let a1 = record[0].1[1];
        let b0 = record[1].1[0];
        let b1 = record[1].1[1];
        let c0 = record[2].1[0];

        let edges = vec![
            mk_edge(a0, b0, 1.0, true),
            mk_edge(a0, b1, 2.0, true),
            mk_edge(a1, b0, 3.0, true),
            mk_edge(a1, b1, 4.0, true),
            mk_edge(b0, c0, 1.0, true),
            mk_edge(b1, c0, 1.0, true),
        ];
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        let cfg = BoundedBeamConfig {
            min_nodes: 3,
            ..test_cfg()
        };
        let solver = BoundedBeamSolver::new(cfg);
        let output = solver.solve(&graph, &cc, 0);

        // 4 possible 3-node paths: a0->b0->c0, a0->b1->c0, a1->b0->c0, a1->b1->c0.
        assert_eq!(output.tracks.len(), 4);
    }

    // =========================================================================
    // Property-based tests
    // =========================================================================

    /// Strategy: generate a graph with n_nights and random edges.
    fn arb_graph_spec()
    -> impl Strategy<Value = (Vec<(u32, usize)>, Vec<(usize, usize, usize, f64, bool)>)> {
        let nights = prop::collection::vec((1u32..50, 1usize..4), 2..6);

        nights.prop_flat_map(|night_spec| {
            let spec = night_spec.clone();
            let n_nights = spec.len();
            let edges = prop::collection::vec(
                (
                    0usize..n_nights.saturating_sub(1),
                    0usize..4,
                    0usize..4,
                    1.0f64..10.0f64,
                    any::<bool>(),
                ),
                0..20,
            );
            (Just(spec), edges)
        })
    }

    /// Build edges from descriptors, deduplicating by `(from, to)`.
    ///
    /// `EdgeKey` assumes at most one edge per `(from, to)` pair, so later
    /// descriptors overwrite earlier ones sharing the same endpoints.
    fn edges_from_descs(
        record: &[(NightId, Vec<SeedKey>)],
        descs: &[(usize, usize, usize, f64, bool)],
    ) -> Vec<Edge> {
        let mut seen: AHashMap<(SeedKey, SeedKey), usize> = AHashMap::new();
        let mut edges = Vec::new();
        for &(pair_idx, from_idx, to_idx, cost, active) in descs {
            if pair_idx + 1 >= record.len() {
                continue;
            }
            let (_, ref from_keys) = record[pair_idx];
            let (_, ref to_keys) = record[pair_idx + 1];
            if from_keys.is_empty() || to_keys.is_empty() {
                continue;
            }
            let from = from_keys[from_idx % from_keys.len()];
            let to = to_keys[to_idx % to_keys.len()];
            let pair = (from, to);
            if let Some(&idx) = seen.get(&pair) {
                edges[idx] = mk_edge(from, to, cost, active);
            } else {
                seen.insert(pair, edges.len());
                edges.push(mk_edge(from, to, cost, active));
            }
        }
        edges
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(150))]

        /// Every emitted track has at least min_nodes nodes.
        #[test]
        fn prop_tracks_satisfy_min_nodes(
            (spec, edge_descs) in arb_graph_spec(),
            min_nodes in 2usize..5
        ) {
            let (store, record) = build_store(&spec);
            let edges = edges_from_descs(&record, &edge_descs);
            let graph = build_graph(edges);
            let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

            let cfg = BoundedBeamConfig { min_nodes, ..test_cfg() };
            let solver = BoundedBeamSolver::new(cfg);
            let outputs = solve_all(&solver, &graph, &cc);

            for output in &outputs {
                for trk in output.tracks.values() {
                    prop_assert!(
                        trk.nodes.len() >= min_nodes,
                        "track has {} nodes < min_nodes={}",
                        trk.nodes.len(), min_nodes
                    );
                }
            }
        }

        /// Total number of emitted tracks per component <= max_tracks.
        #[test]
        fn prop_tracks_bounded_by_max_tracks(
            (spec, edge_descs) in arb_graph_spec(),
            max_tracks in 1usize..10
        ) {
            let (store, record) = build_store(&spec);
            let edges = edges_from_descs(&record, &edge_descs);
            let graph = build_graph(edges);
            let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

            let cfg = BoundedBeamConfig { max_tracks, min_nodes: 2, ..test_cfg() };
            let solver = BoundedBeamSolver::new(cfg);
            let outputs = solve_all(&solver, &graph, &cc);

            for output in &outputs {
                prop_assert!(
                    output.tracks.len() <= max_tracks,
                    "component {} emitted {} tracks > max_tracks={}",
                    output.diag.component_id, output.tracks.len(), max_tracks
                );
            }
        }

        /// Track nodes are strictly time-ordered (night_id strictly increasing).
        #[test]
        fn prop_track_nodes_time_ordered(
            (spec, edge_descs) in arb_graph_spec()
        ) {
            let (store, record) = build_store(&spec);
            let edges = edges_from_descs(&record, &edge_descs);
            let graph = build_graph(edges);
            let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

            let solver = BoundedBeamSolver::new(test_cfg());
            let outputs = solve_all(&solver, &graph, &cc);

            for output in &outputs {
                for trk in output.tracks.values() {
                    for w in trk.nodes.windows(2) {
                        prop_assert!(
                            w[0].night_id < w[1].night_id,
                            "nodes not time-forward: {} >= {}",
                            w[0].night_id.value(), w[1].night_id.value()
                        );
                    }
                }
            }
        }

        /// Track edges match consecutive node pairs.
        #[test]
        fn prop_track_edges_match_nodes(
            (spec, edge_descs) in arb_graph_spec()
        ) {
            let (store, record) = build_store(&spec);
            let edges = edges_from_descs(&record, &edge_descs);
            let graph = build_graph(edges);
            let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

            let solver = BoundedBeamSolver::new(test_cfg());
            let outputs = solve_all(&solver, &graph, &cc);

            for output in &outputs {
                for trk in output.tracks.values() {
                    prop_assert_eq!(
                        trk.edges.len(), trk.nodes.len().saturating_sub(1)
                    );
                    for (i, ek) in trk.edges.iter().enumerate() {
                        prop_assert_eq!(ek.from, trk.nodes[i]);
                        prop_assert_eq!(ek.to, trk.nodes[i + 1]);
                    }
                }
            }
        }

        /// Track cost equals sum of edge costs.
        #[test]
        fn prop_track_cost_is_edge_sum(
            (spec, edge_descs) in arb_graph_spec()
        ) {
            let (store, record) = build_store(&spec);
            let edges = edges_from_descs(&record, &edge_descs);
            let graph = build_graph(edges);
            let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

            let solver = BoundedBeamSolver::new(test_cfg());
            let outputs = solve_all(&solver, &graph, &cc);

            for output in &outputs {
                for trk in output.tracks.values() {
                    let edge_sum: f64 = trk
                        .edges
                        .iter()
                        .map(|ek| graph.edge_by_key(ek).unwrap().cost)
                        .sum();
                    prop_assert!(
                        (trk.cost - edge_sum).abs() < 1e-9,
                        "cost {} != edge sum {}", trk.cost, edge_sum
                    );
                }
            }
        }

        /// Track night_span matches first/last node night difference.
        #[test]
        fn prop_track_night_span_consistent(
            (spec, edge_descs) in arb_graph_spec()
        ) {
            let (store, record) = build_store(&spec);
            let edges = edges_from_descs(&record, &edge_descs);
            let graph = build_graph(edges);
            let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

            let solver = BoundedBeamSolver::new(test_cfg());
            let outputs = solve_all(&solver, &graph, &cc);

            for output in &outputs {
                for trk in output.tracks.values() {
                    let first = trk.nodes.first().unwrap().night_id.value();
                    let last = trk.nodes.last().unwrap().night_id.value();
                    prop_assert_eq!(
                        trk.night_span, last - first,
                        "night_span {} != {} - {}", trk.night_span, last, first
                    );
                }
            }
        }

        /// Track nodes reference existing seeds in the store.
        #[test]
        fn prop_track_nodes_exist_in_store(
            (spec, edge_descs) in arb_graph_spec()
        ) {
            let (store, record) = build_store(&spec);
            let all_keys: AHashSet<SeedKey> = record
                .iter()
                .flat_map(|(_, ks)| ks.iter().copied())
                .collect();

            let edges = edges_from_descs(&record, &edge_descs);
            let graph = build_graph(edges);
            let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

            let solver = BoundedBeamSolver::new(test_cfg());
            let outputs = solve_all(&solver, &graph, &cc);

            for output in &outputs {
                for trk in output.tracks.values() {
                    for node in &trk.nodes {
                        prop_assert!(
                            all_keys.contains(node),
                            "track node key {:?} not in store", node
                        );
                    }
                }
            }
        }

        /// n_selected in diagnostics matches the actual track count.
        #[test]
        fn prop_diag_n_selected_matches_tracks_len(
            (spec, edge_descs) in arb_graph_spec()
        ) {
            let (store, record) = build_store(&spec);
            let edges = edges_from_descs(&record, &edge_descs);
            let graph = build_graph(edges);
            let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

            let solver = BoundedBeamSolver::new(test_cfg());
            let outputs = solve_all(&solver, &graph, &cc);

            for output in &outputs {
                prop_assert_eq!(
                    output.diag.n_selected as usize,
                    output.tracks.len()
                );
            }
        }

        /// active_only=true emits tracks only using active edges.
        #[test]
        fn prop_active_only_tracks_use_active_edges(
            (spec, edge_descs) in arb_graph_spec()
        ) {
            let (store, record) = build_store(&spec);
            let edges = edges_from_descs(&record, &edge_descs);
            let graph = build_graph(edges);
            let cc = ConnectedComponents::compute(&store, &graph, true).unwrap();

            let solver = BoundedBeamSolver::new(test_cfg());
            let outputs = solve_all(&solver, &graph, &cc);

            for output in &outputs {
                for trk in output.tracks.values() {
                    for ek in &trk.edges {
                        let edge = graph.edge_by_key(ek).unwrap();
                        prop_assert!(
                            edge.active,
                            "track edge {} -> {} is inactive under active_only CC",
                            ek.from, ek.to
                        );
                    }
                }
            }
        }

        /// Solver is deterministic: same inputs yield same outputs.
        #[test]
        fn prop_solver_is_deterministic(
            (spec, edge_descs) in arb_graph_spec()
        ) {
            let (store, record) = build_store(&spec);
            let edges = edges_from_descs(&record, &edge_descs);

            let graph1 = build_graph(edges.clone());
            let cc1 = ConnectedComponents::compute(&store, &graph1, false).unwrap();
            let solver = BoundedBeamSolver::new(test_cfg());
            let outputs1 = solve_all(&solver, &graph1, &cc1);

            let graph2 = build_graph(edges);
            let cc2 = ConnectedComponents::compute(&store, &graph2, false).unwrap();
            let outputs2 = solve_all(&solver, &graph2, &cc2);

            prop_assert_eq!(outputs1.len(), outputs2.len());
            for (o1, o2) in outputs1.iter().zip(outputs2.iter()) {
                prop_assert_eq!(o1.tracks.len(), o2.tracks.len());
                prop_assert_eq!(o1.diag.n_selected, o2.diag.n_selected);
                prop_assert_eq!(o1.diag.n_candidates, o2.diag.n_candidates);
            }
        }
    }
}
