//! Trivial solver: beam-search enumeration of path hypotheses inside a small
//! time-forward DAG component.
//!
//! This refactor exploits the new `ConnectedComponents` API:
//! - adjacency (`component_out_edges`) is pre-built during CC construction,
//! - sources/sinks are precomputed (`component_sources_local`, `component_sinks_local`),
//! - local degrees are available (optional sanity / debug / heuristics).
//!
//! Therefore the trivial solver no longer rescans `graph.edges` nor rebuilds
//! membership maps per component.

pub mod trivial_config;

use crate::{
    graph::{RuntimeGraph, edge::Edge},
    seeding::seed_node::SeedNode,
    solver::{
        Solver, SolverDiagnostics, SolverOutput,
        components::{ComponentId, ConnectedComponents, LocalIdx},
        trivial_solver::trivial_config::TrivialSolverConfig,
    },
    trajectory::TrackHypothesis,
};

// -----------------------------------------------------------------------------
// Trivial solver (beam search over small DAG components)
// -----------------------------------------------------------------------------

/// Trivial solver: enumerates path hypotheses in a small DAG component.
///
/// Assumptions (expected invariants)
/// --------------------------------
/// - Directed acyclic graph (DAG) in the solver-facing sense:
///   edges always go forward in time (`night(to) > night(from)`; gaps allowed).
/// - Components are computed on the undirected view of the graph.
/// - `ConnectedComponents` exposes a *restricted* directed subgraph per component:
///   adjacency + local degrees + sources/sinks.
///
/// Strategy
/// --------
/// 1) Read pre-built restricted adjacency for the component from `ConnectedComponents`.
/// 2) Sort outgoing edges by increasing cost and locally prune (`max_out_per_node`).
/// 3) Beam-search path hypotheses from precomputed sources.
/// 4) Emit terminal paths that end at sinks (nodes with no outgoing edges).
/// 5) Rank tracks by average cost per edge (Option A) to avoid preferring tiny tracks.
#[derive(Clone, Debug, Default)]
pub struct TrivialSolver {
    pub cfg: TrivialSolverConfig,
}

impl TrivialSolver {
    /// Create a new trivial solver with the given configuration.
    ///
    /// Arguments
    /// ---------
    /// * `cfg` – Beam search and output capping configuration.
    pub fn new(cfg: TrivialSolverConfig) -> Self {
        Self { cfg }
    }

    /// Solve a single component using the precomputed component view.
    ///
    /// This is the preferred entrypoint once `ConnectedComponents` has been computed.
    ///
    /// Arguments
    /// ---------
    /// * `graph` – Global runtime graph (only used for edge refs lifetimes and invariants).
    /// * `cc` – Connected components object containing prebuilt restricted adjacency.
    /// * `component_id` – Component to solve.
    ///
    /// Return
    /// ------
    /// * `SolverOutput` – Candidate tracks for this component, sorted and truncated.
    ///
    /// Notes
    /// -----
    /// - This method does **not** rescan `graph.edges`.
    /// - The adjacency stored in `cc` is already restricted to the component and already
    ///   filtered by `active_only` used during CC construction.
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

        if n_nodes < cfg.min_nodes {
            return SolverOutput {
                tracks: Vec::new(),
                diag,
            };
        }

        let out_edges = cc.component_out_edges(component_id);
        let sources = cc.component_sources_local(component_id);

        // The CC fallback guarantees: if there is any outgoing edge in the component,
        // sources is non-empty. Still: keep defensive behavior.
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

impl<'edge_lf, 'seed_lf, 'alert_lf> Solver<'edge_lf, 'seed_lf, 'alert_lf> for TrivialSolver {
    /// Stable solver name for logs and metrics.
    fn name(&self) -> &'static str {
        "trivial"
    }

    /// Legacy trait entrypoint: solve a component when only node refs are provided.
    ///
    /// Arguments
    /// ---------
    /// * `graph` – Global runtime graph (read-only).
    /// * `component_nodes` – Node refs of the component (borrowed).
    ///
    /// Return
    /// ------
    /// * `SolverOutput` – Candidate tracks.
    ///
    /// Notes
    /// -----
    /// - With the new CC API, prefer calling `solve_component_cc()` instead.
    /// - This method is kept to satisfy the trait and for backwards compatibility.
    /// - If you want to route through `ConnectedComponents`, do it in `SolverManager`
    ///   (where you already have `component_id`).
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

/// Local outgoing edge inside a component (pruned/sorted view).
///
/// This is derived from the `&Edge` adjacency stored in `ConnectedComponents`,
/// and adds a cached destination local index + cost for fast beam expansion.
///
/// Notes
/// -----
/// - We keep `edge` as a reference to avoid copying edge data.
/// - We cache `dst` and `cost` to avoid repeated lookup / deref.
#[derive(Clone, Copy, Debug)]
struct OutEdge<'edge_lf, 'seed_lf, 'alert_lf> {
    dst: usize,
    edge: &'edge_lf Edge<'seed_lf, 'alert_lf>,
    cost: f64,
}

/// Pruned/sorted adjacency used by the beam search.
/// Layout: `out[u]` are outgoing edges from local node `u`.
type Adjacency<'edge_lf, 'seed_lf, 'alert_lf> = Vec<Vec<OutEdge<'edge_lf, 'seed_lf, 'alert_lf>>>;

/// Beam search state stored as a node in an implicit path tree.
///
/// Each state represents a partial path ending at `last`.
/// Paths are reconstructed through `parent` backpointers.
///
/// Notes
/// -----
/// - We store `in_edge` used to reach this state from its parent.
/// - We cache `total_cost` and `n_edges` so ranking is O(1) per state.
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
/// The CC adjacency is `&[Vec<&Edge>]`, i.e. it already contains only
/// intra-component edges and respects `active_only` used in CC computation.
///
/// This function:
/// - converts edge refs to `OutEdge { dst, edge, cost }`,
/// - sorts outgoing edges by `cost` ascending,
/// - truncates each outgoing list to `cfg.max_out_per_node`,
/// - counts candidates for diagnostics.
///
/// Arguments
/// ---------
/// * `cfg` – Trivial solver configuration.
/// * `component_nodes` – Component nodes in local index order.
/// * `component_out_edges` – CC restricted adjacency: `out[u] = Vec<&Edge>`.
/// * `diag` – Diagnostics updated in-place (candidate edge counts).
///
/// Return
/// ------
/// * `Adjacency` – Pruned/sorted adjacency suitable for beam search.
///
/// Notes
/// -----
/// - Destination local index is derived by reading `edge.to.core.key` and mapping it
///   through the fact that CC already guarantees it belongs to the component.
/// - We do not use a hash map here: the CC adjacency is already local-indexed by source,
///   and destination local index is computed by searching `component_nodes` via a linear scan
///   would be O(n). Instead we rely on the invariant:
///   `edge.to` points to a `SeedNode` reference that also appears in `component_nodes`,
///   and we recover the local index using a temporary map built once per component.
fn build_solver_adjacency<'edge_lf, 'seed_lf, 'alert_lf>(
    cfg: &TrivialSolverConfig,
    component_nodes: &[&'seed_lf SeedNode<'alert_lf>],
    component_out_edges: &[Vec<&'edge_lf Edge<'seed_lf, 'alert_lf>>],
    diag: &mut SolverDiagnostics,
) -> Adjacency<'edge_lf, 'seed_lf, 'alert_lf> {
    // Build SeedKey -> local index for destinations.
    // (This is per-component, small, and avoids O(n) scans during edge conversion.)
    let mut local_of_key: ahash::AHashMap<crate::persistence::seed_node::SeedKey, usize> =
        ahash::AHashMap::default();
    local_of_key.reserve(component_nodes.len());
    for (i, &node) in component_nodes.iter().enumerate() {
        local_of_key.insert(node.core.key, i);
    }

    let n = component_nodes.len();
    let mut out: Adjacency<'edge_lf, 'seed_lf, 'alert_lf> = (0..n).map(|_| Vec::new()).collect();

    for (u, edges_u) in component_out_edges.iter().enumerate() {
        let mut buf: Vec<OutEdge<'edge_lf, 'seed_lf, 'alert_lf>> =
            Vec::with_capacity(edges_u.len());

        for &e in edges_u.iter() {
            // Destination must be inside the component (guaranteed by CC).
            let Some(&dst) = local_of_key.get(&e.to.core.key) else {
                continue;
            };

            diag.n_candidates += 1;

            buf.push(OutEdge {
                dst,
                edge: e,
                cost: e.core.cost,
            });
        }

        buf.sort_by(|a, b| {
            a.cost
                .partial_cmp(&b.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
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
/// Arguments
/// ---------
/// * `sources` – Component-local source nodes (local indices).
/// * `component_nodes` – Component nodes to extract the starting night for each source.
///
/// Return
/// ------
/// * `(states, beam)`
///   - `states` is the state pool containing all created states,
///   - `beam` contains indices into `states` representing the current frontier.
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
/// Arguments
/// ---------
/// * `out` – Solver adjacency.
/// * `u` – Local node index.
///
/// Return
/// ------
/// `true` if the node has no outgoing edges in the restricted subgraph.
fn is_sink<'edge_lf, 'seed_lf, 'alert_lf>(
    out: &Adjacency<'edge_lf, 'seed_lf, 'alert_lf>,
    u: usize,
) -> bool {
    out[u].is_empty()
}

/// Perform one beam iteration: expand the current beam into next candidates.
///
/// Arguments
/// ---------
/// * `cfg` – Trivial solver configuration.
/// * `out` – Solver adjacency (sorted/pruned).
/// * `component_nodes` – Component nodes, used for a defensive time-forward guard.
/// * `states` – State pool grown in-place as we create new partial paths.
/// * `beam` – Current frontier (indices into `states`).
/// * `terminal_states` – Output buffer collecting terminal path endpoints (state ids).
/// * `emitted_per_source` – Per-source output counter enforcing `max_tracks_per_source`.
/// * `expansions` – Global expansion counter enforcing `max_expansions`.
///
/// Return
/// ------
/// * `Vec<usize>` – Next beam frontier (state ids), already truncated to `beam_width`.
///
/// Notes
/// -----
/// - If a beam state ends at a sink, it is considered terminal and is not expanded.
/// - A defensive check ensures that the destination night is strictly increasing.
fn expand_beam<'edge_lf, 'seed_lf, 'alert_lf>(
    cfg: &TrivialSolverConfig,
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
        if *expansions >= cfg.max_expansions || terminal_states.len() >= cfg.max_tracks {
            break;
        }

        let st = states[sid];
        let u = st.last;

        // If sink: emit terminal candidate and do not expand.
        if is_sink(out, u) {
            let n_nodes = st.n_edges as usize + 1;
            if n_nodes >= cfg.min_nodes {
                let src = st.source;
                if emitted_per_source[src] < cfg.max_tracks_per_source as u32 {
                    terminal_states.push(sid);
                    emitted_per_source[src] += 1;
                }
            }
            continue;
        }

        // Expand along outgoing edges.
        for oe in out[u].iter() {
            if *expansions >= cfg.max_expansions || terminal_states.len() >= cfg.max_tracks {
                break;
            }

            // Defensive time-forward guard (should always hold if graph invariants hold).
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

    // Beam heuristic: keep best partial paths by total cost.
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

/// Reconstruct a `TrackHypothesis` from a terminal state id.
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
/// - Reconstruction follows `parent` backpointers.
/// - The returned nodes are in forward time order.
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

/// Sort tracks by Option A (average cost per edge), then truncate.
///
/// Ordering
/// --------
/// 1) lower average cost per edge is better,
/// 2) tie-break: prefer longer tracks (more edges),
/// 3) tie-break: lower total cost.
///
/// Arguments
/// ---------
/// * `cfg` – Trivial solver configuration.
/// * `tracks` – Track list sorted and truncated in-place.
fn sort_and_truncate_tracks<'edge_lf, 'seed_lf, 'alert_lf>(
    cfg: &TrivialSolverConfig,
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

/// Enumerate candidate tracks in a component using the CC restricted view.
///
/// This is the main refactored enumerator used by `solve_component_cc()`.
///
/// Arguments
/// ---------
/// * `cfg` – Trivial solver configuration.
/// * `graph` – Global graph (unused currently, but kept for invariants / future heuristics).
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
/// - `diag.n_candidates` counts edges *seen* in the restricted adjacency before pruning.
/// - `cfg.max_out_per_node` pruning is applied here (not during CC build) so it can
///   be tuned per solver configuration.
fn enumerate_beam_tracks_from_component_view<'edge_lf, 'seed_lf, 'alert_lf>(
    cfg: &TrivialSolverConfig,
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

    // Convert sources to usize.
    let sources = sources_local
        .iter()
        .map(|&x| x as usize)
        .collect::<Vec<_>>();
    if sources.is_empty() {
        return Vec::new();
    }

    // 1) Build solver adjacency (sorted + pruned).
    let out = build_solver_adjacency(cfg, component_nodes, component_out_edges, diag);

    // 2) Init beam + state pool.
    let (mut states, mut beam) = init_beam_states(&sources, component_nodes);

    // Per-source emission counters (index by local node id).
    let mut emitted_per_source: Vec<u32> = vec![0; n];
    let mut terminal_states: Vec<usize> = Vec::new();
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

    // 6) Final ranking + truncate.
    sort_and_truncate_tracks(cfg, &mut tracks);
    tracks
}
