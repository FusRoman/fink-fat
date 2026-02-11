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

pub mod bounded_beam_config;

use crate::{
    graph::{RuntimeGraph, edge::Edge},
    seeding::seed_node::SeedNode,
    solver::{
        Solver, SolverDiagnostics, SolverOutput,
        bounded_beam::bounded_beam_config::BoundedBeamConfig,
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
    pub fn solve_component_cc<'edge_lf, 'seed_lf, 'alert_lf>(
        &self,
        graph: &'edge_lf RuntimeGraph<'seed_lf, 'alert_lf>,
        cc: &'edge_lf ConnectedComponents<'edge_lf, 'seed_lf, 'alert_lf>,
        component_id: ComponentId,
    ) -> SolverOutput<'edge_lf, 'seed_lf, 'alert_lf>
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
                tracks: Vec::new(),
                diag,
            };
        }

        let out_edges = cc.component_out_edges(component_id);
        let sources = cc.component_sources_local(component_id);

        // Beam search requires at least one start point.
        if sources.is_empty() {
            return SolverOutput {
                tracks: Vec::new(),
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

impl<'edge_lf, 'seed_lf, 'alert_lf> Solver<'edge_lf, 'seed_lf, 'alert_lf> for BoundedBeamSolver {
    /// Stable solver name for logs and metrics.
    fn name(&self) -> &'static str {
        "bounded_beam"
    }

    /// Solve a component using its `component_id` and the `ConnectedComponents` view.
    ///
    /// This is the trait entrypoint used by orchestration code.
    fn solve(
        &self,
        graph: &'edge_lf RuntimeGraph<'seed_lf, 'alert_lf>,
        cc: &'edge_lf ConnectedComponents<'edge_lf, 'seed_lf, 'alert_lf>,
        component_id: ComponentId,
    ) -> SolverOutput<'edge_lf, 'seed_lf, 'alert_lf>
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
struct OutEdge<'edge_lf, 'seed_lf, 'alert_lf> {
    /// Destination node (component-local index).
    dst: usize,
    /// Borrowed reference to the original edge.
    edge: &'edge_lf Edge<'seed_lf, 'alert_lf>,
    /// Cached edge cost (lower is better).
    cost: f64,
}

/// Pruned/sorted adjacency used by the beam search.
///
/// Layout
/// ------
/// `out[u]` is the list of outgoing edges from local node `u`.
type Adjacency<'edge_lf, 'seed_lf, 'alert_lf> = Vec<Vec<OutEdge<'edge_lf, 'seed_lf, 'alert_lf>>>;

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
struct State<'edge_lf, 'seed_lf, 'alert_lf> {
    last: usize,
    parent: Option<usize>, // index into `states`
    in_edge: Option<&'edge_lf Edge<'seed_lf, 'alert_lf>>,
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
fn build_solver_adjacency<'edge_lf, 'seed_lf, 'alert_lf>(
    cfg: &BoundedBeamConfig,
    component_nodes: &[&'seed_lf SeedNode<'alert_lf>],
    component_out_edges: &[Vec<&'edge_lf Edge<'seed_lf, 'alert_lf>>],
    diag: &mut SolverDiagnostics,
) -> Adjacency<'edge_lf, 'seed_lf, 'alert_lf> {
    // Build SeedKey -> local index for destinations.
    // This is small (component-local) and prevents repeated O(n) scans.
    let mut local_of_key: ahash::AHashMap<crate::persistence::seed_node::SeedKey, usize> =
        ahash::AHashMap::default();
    local_of_key.reserve(component_nodes.len());

    for (i, &node) in component_nodes.iter().enumerate() {
        local_of_key.insert(node.core.key, i);
    }

    let n = component_nodes.len();
    let mut out: Adjacency<'edge_lf, 'seed_lf, 'alert_lf> = (0..n).map(|_| Vec::new()).collect();

    for (u, edges_u) in component_out_edges.iter().enumerate() {
        // Convert and cache (dst, cost) for faster expansion.
        let mut buf: Vec<OutEdge<'edge_lf, 'seed_lf, 'alert_lf>> =
            Vec::with_capacity(edges_u.len());

        for &e in edges_u.iter() {
            // Destination must be inside the component (guaranteed by CC).
            let Some(&dst) = local_of_key.get(&e.to.core.key) else {
                // Defensive fallback: skip malformed edges.
                continue;
            };

            // Count candidates *before* pruning (diagnostics).
            diag.n_candidates += 1;

            buf.push(OutEdge {
                dst,
                edge: e,
                cost: e.core.cost,
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
    component_nodes: &[&'seed_lf SeedNode<'alert_lf>],
) -> (Vec<State<'edge_lf, 'seed_lf, 'alert_lf>>, Vec<usize>) {
    let mut states: Vec<State<'edge_lf, 'seed_lf, 'alert_lf>> = Vec::new();
    let mut beam: Vec<usize> = Vec::new();

    for &s in sources {
        let night = component_nodes[s].core.night_id().value();
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
fn is_sink<'edge_lf, 'seed_lf, 'alert_lf>(
    out: &Adjacency<'edge_lf, 'seed_lf, 'alert_lf>,
    u: usize,
) -> bool {
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
fn expand_beam<'edge_lf, 'seed_lf, 'alert_lf>(
    cfg: &BoundedBeamConfig,
    out: &Adjacency<'edge_lf, 'seed_lf, 'alert_lf>,
    component_nodes: &[&'seed_lf SeedNode<'alert_lf>],
    states: &mut Vec<State<'edge_lf, 'seed_lf, 'alert_lf>>,
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
            let v_night = component_nodes[oe.dst].core.night_id().value();
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
fn reconstruct_track<'edge_lf, 'seed_lf, 'alert_lf>(
    terminal_state_id: usize,
    states: &[State<'edge_lf, 'seed_lf, 'alert_lf>],
    component_nodes: &[&'seed_lf SeedNode<'alert_lf>],
) -> TrackHypothesis<'edge_lf, 'seed_lf, 'alert_lf> {
    let mut node_idx_rev: Vec<usize> = Vec::new();
    let mut edge_rev: Vec<&'edge_lf Edge<'seed_lf, 'alert_lf>> = Vec::new();

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
        nodes,
        edges: edge_rev,
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
fn sort_and_truncate_tracks<'edge_lf, 'seed_lf, 'alert_lf>(
    cfg: &BoundedBeamConfig,
    tracks: &mut Vec<TrackHypothesis<'edge_lf, 'seed_lf, 'alert_lf>>,
) {
    tracks.sort_by(|a, b| {
        let a_edges = a.edges.len().max(1) as f64;
        let b_edges = b.edges.len().max(1) as f64;

        let a_avg = a.cost / a_edges;
        let b_avg = b.cost / b_edges;

        a_avg
            .partial_cmp(&b_avg)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.edges.len().cmp(&a.edges.len()))
            .then_with(|| {
                a.cost
                    .partial_cmp(&b.cost)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    if tracks.len() > cfg.max_tracks {
        tracks.truncate(cfg.max_tracks);
    }
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
/// * `Vec<TrackHypothesis>` – Candidate tracks sorted and truncated.
///
/// Notes
/// -----
/// - `diag.n_candidates` counts edges *seen* before pruning.
/// - `cfg.max_out_per_node` pruning is applied here so it can be tuned per solver.
fn enumerate_beam_tracks_from_component_view<'edge_lf, 'seed_lf, 'alert_lf>(
    cfg: &BoundedBeamConfig,
    _graph: &RuntimeGraph<'seed_lf, 'alert_lf>,
    component_nodes: &[&'seed_lf SeedNode<'alert_lf>],
    component_out_edges: &[Vec<&'edge_lf Edge<'seed_lf, 'alert_lf>>],
    sources_local: &[LocalIdx],
    diag: &mut SolverDiagnostics,
) -> Vec<TrackHypothesis<'edge_lf, 'seed_lf, 'alert_lf>> {
    let n = component_nodes.len();
    if n == 0 {
        return Vec::new();
    }

    // Convert sources (LocalIdx) into indices used by local vectors.
    let sources = sources_local
        .iter()
        .map(|&x| x as usize)
        .collect::<Vec<_>>();

    if sources.is_empty() {
        return Vec::new();
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

        if next.is_empty() {
            break;
        }
        beam = next;
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
    let mut tracks: Vec<TrackHypothesis<'edge_lf, 'seed_lf, 'alert_lf>> = terminal_states
        .into_iter()
        .map(|sid| reconstruct_track(sid, &states, component_nodes))
        .filter(|t| t.nodes.len() >= cfg.min_nodes)
        .collect();

    // 6) Rank + truncate.
    sort_and_truncate_tracks(cfg, &mut tracks);
    tracks
}
