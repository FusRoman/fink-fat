pub mod trivial_config;

use ahash::AHashMap;

use crate::{
    graph::{RuntimeGraph, edge::Edge},
    persistence::seed_node::SeedKey,
    seeding::seed_node::SeedNode,
    solver::{
        Solver, SolverDiagnostics, SolverOutput,
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
/// - Directed acyclic graph (DAG).
/// - Edges go forward in time: `night(to) > night(from)` (gaps allowed).
///
/// Strategy
/// --------
/// 1) Build adjacency restricted to the component.
/// 2) Sort outgoing edges by increasing cost (lower is better) and locally prune (`max_out_per_node`).
/// 3) Find sources (nodes with no incoming edges in the restricted adjacency).
/// 4) Beam-search path hypotheses from sources.
/// 5) Emit terminal paths that end on sinks (nodes with no outgoing edges in the restricted adjacency).
/// 6) Rank tracks by average cost per edge to avoid preferring tiny tracks.
#[derive(Clone, Debug, Default)]
pub struct TrivialSolver {
    pub cfg: TrivialSolverConfig,
}

impl TrivialSolver {
    pub fn new(cfg: TrivialSolverConfig) -> Self {
        Self { cfg }
    }

    /// Solve one connected component using an explicit edge slice.
    ///
    /// Notes
    /// -----
    /// - Only uses edges whose endpoints are both in `component_nodes`.
    /// - If `active_only` is true, ignores inactive edges.
    pub fn solve_component<'edge_lf, 'seed_lf, 'alert_lf>(
        &self,
        edges: &'edge_lf [Edge<'seed_lf, 'alert_lf>],
        component_nodes: &[&'seed_lf SeedNode<'alert_lf>],
        active_only: bool,
        component_id: u32,
    ) -> SolverOutput<'edge_lf, 'seed_lf, 'alert_lf> {
        let cfg = &self.cfg;

        let mut diag = SolverDiagnostics {
            component_id,
            solver_name: self.name(),
            n_nodes: component_nodes.len() as u32,
            m_active_edges: 0,
            time_est_s: 0.0,
            time_spent_s: 0.0,
            n_candidates: 0,
            n_selected: 0,
        };

        if component_nodes.len() < cfg.min_nodes {
            return SolverOutput {
                tracks: Vec::new(),
                diag,
            };
        }

        let tracks = enumerate_beam_tracks(cfg, edges, component_nodes, active_only, &mut diag);

        diag.n_selected = tracks.len() as u32;
        SolverOutput { tracks, diag }
    }
}

/// Trait implementation: only `name()` and `solve()` belong here.
impl<'edge_lf, 'seed_lf, 'alert_lf> Solver<'edge_lf, 'seed_lf, 'alert_lf> for TrivialSolver {
    fn name(&self) -> &'static str {
        "trivial"
    }

    fn solve(
        &self,
        graph: &'edge_lf RuntimeGraph<'seed_lf, 'alert_lf>,
        component_nodes: &[&'seed_lf SeedNode<'alert_lf>],
    ) -> SolverOutput<'edge_lf, 'seed_lf, 'alert_lf> {
        let active_only = true;
        self.solve_component(&graph.edges, component_nodes, active_only, 0)
    }
}

// -----------------------------------------------------------------------------
// Private implementation details
// -----------------------------------------------------------------------------

type LocalIdx = usize;

#[derive(Clone, Copy, Debug)]
struct OutEdge<'edge_lf, 'seed_lf, 'alert_lf> {
    dst: LocalIdx,
    edge: &'edge_lf Edge<'seed_lf, 'alert_lf>,
    cost: f64,
}

type Adjacency<'edge_lf, 'seed_lf, 'alert_lf> = Vec<Vec<OutEdge<'edge_lf, 'seed_lf, 'alert_lf>>>;

#[derive(Clone, Copy, Debug)]
struct State<'edge_lf, 'seed_lf, 'alert_lf> {
    last: LocalIdx,
    parent: Option<usize>, // index into `states`
    in_edge: Option<&'edge_lf Edge<'seed_lf, 'alert_lf>>,
    total_cost: f64,
    n_edges: u32,
    first_night: u32,
    last_night: u32,
    source: LocalIdx,
}

// -----------------------------------------------------------------------------
// Step 1: membership
// -----------------------------------------------------------------------------

fn build_membership<'seed_lf, 'alert_lf>(
    component_nodes: &[&'seed_lf SeedNode<'alert_lf>],
) -> AHashMap<SeedKey, LocalIdx> {
    let mut in_comp: AHashMap<_, LocalIdx> = AHashMap::default();
    in_comp.reserve(component_nodes.len());

    for (i, &node) in component_nodes.iter().enumerate() {
        in_comp.insert(node.core.key, i);
    }
    in_comp
}

// -----------------------------------------------------------------------------
// Step 2: adjacency + indegree (restricted to component)
// -----------------------------------------------------------------------------

fn build_adjacency<'edge_lf, 'seed_lf, 'alert_lf>(
    cfg: &TrivialSolverConfig,
    edges: &'edge_lf [Edge<'seed_lf, 'alert_lf>],
    component_nodes: &[&'seed_lf SeedNode<'alert_lf>],
    in_comp: &AHashMap<SeedKey, LocalIdx>,
    active_only: bool,
    diag: &mut SolverDiagnostics,
) -> (Adjacency<'edge_lf, 'seed_lf, 'alert_lf>, Vec<u32>) {
    let n = component_nodes.len();
    let mut out: Adjacency<'edge_lf, 'seed_lf, 'alert_lf> = vec![Vec::new(); n];
    let mut indeg: Vec<u32> = vec![0; n];

    for e in edges.iter() {
        if active_only && !e.core.active {
            continue;
        }

        let Some(&src) = in_comp.get(&e.from.core.key) else {
            continue;
        };
        let Some(&dst) = in_comp.get(&e.to.core.key) else {
            continue;
        };
        if src == dst {
            continue;
        }

        // forward-time invariant (gaps allowed)
        let n0 = e.from.core.night_id().value();
        let n1 = e.to.core.night_id().value();
        if n1 <= n0 {
            continue;
        }

        diag.n_candidates += 1;
        if e.core.active {
            diag.m_active_edges += 1;
        }

        out[src].push(OutEdge {
            dst,
            edge: e,
            cost: e.core.cost,
        });
        indeg[dst] += 1;
    }

    // Sort outgoing by cost and locally prune.
    for outs in out.iter_mut() {
        outs.sort_by(|a, b| {
            a.cost
                .partial_cmp(&b.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if outs.len() > cfg.max_out_per_node {
            outs.truncate(cfg.max_out_per_node);
        }
    }

    (out, indeg)
}

// -----------------------------------------------------------------------------
// Step 3: sources
// -----------------------------------------------------------------------------

fn find_sources(out: &Adjacency<'_, '_, '_>, indeg: &[u32]) -> Vec<LocalIdx> {
    let n = out.len();

    // Preferred sources: indeg=0 and has outgoing edges.
    let mut sources: Vec<LocalIdx> = (0..n)
        .filter(|&i| indeg[i] == 0 && !out[i].is_empty())
        .collect();

    // Fallback: if no sources, start anywhere with outgoing edges.
    if sources.is_empty() {
        sources = (0..n).filter(|&i| !out[i].is_empty()).collect();
    }

    sources
}

// -----------------------------------------------------------------------------
// Step 4: beam search
// -----------------------------------------------------------------------------

fn init_beam_states<'edge_lf, 'seed_lf, 'alert_lf>(
    sources: &[LocalIdx],
    component_nodes: &[&'seed_lf SeedNode<'alert_lf>],
) -> (Vec<State<'edge_lf, 'seed_lf, 'alert_lf>>, Vec<usize>) {
    let mut states: Vec<State<'edge_lf, 'seed_lf, 'alert_lf>> = Vec::new();
    let mut beam: Vec<usize> = Vec::new();

    for &s in sources.iter() {
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

fn is_sink<'edge_lf, 'seed_lf, 'alert_lf>(
    out: &Adjacency<'edge_lf, 'seed_lf, 'alert_lf>,
    u: LocalIdx,
) -> bool {
    out[u].is_empty()
}

/// One beam iteration: expand current beam to next candidates.
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

    for &sid in beam.iter() {
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

            // Defensive time-forward guard (should always hold if adjacency was built correctly)
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
// Step 5: reconstruction + final ranking (avg cost)
// -----------------------------------------------------------------------------

fn reconstruct_track<'edge_lf, 'seed_lf, 'alert_lf>(
    terminal_state_id: usize,
    states: &[State<'edge_lf, 'seed_lf, 'alert_lf>],
    component_nodes: &[&'seed_lf SeedNode<'alert_lf>],
) -> TrackHypothesis<'edge_lf, 'seed_lf, 'alert_lf> {
    let mut node_idx_rev: Vec<LocalIdx> = Vec::new();
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

    let nodes: Vec<&'seed_lf SeedNode<'alert_lf>> = node_idx_rev
        .into_iter()
        .map(|idx| component_nodes[idx])
        .collect();

    let st = states[terminal_state_id];
    let night_span = st.last_night - st.first_night;

    TrackHypothesis {
        nodes,
        edges: edge_rev,
        cost: st.total_cost,
        night_span,
    }
}

fn sort_and_truncate_tracks<'edge_lf, 'seed_lf, 'alert_lf>(
    cfg: &TrivialSolverConfig,
    tracks: &mut Vec<TrackHypothesis<'edge_lf, 'seed_lf, 'alert_lf>>,
) {
    // Option A: average cost per edge, then prefer longer, then lower total cost.
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
// Main entry: clear step-by-step orchestration
// -----------------------------------------------------------------------------

/// Beam-search enumeration of path hypotheses in a time-forward DAG (gaps allowed).
///
/// Final ranking uses average cost per edge: `avg_cost = total_cost / n_edges`
/// to avoid preferring very short tracks.
pub fn enumerate_beam_tracks<'edge_lf, 'seed_lf, 'alert_lf>(
    cfg: &TrivialSolverConfig,
    edges: &'edge_lf [Edge<'seed_lf, 'alert_lf>],
    component_nodes: &[&'seed_lf SeedNode<'alert_lf>],
    active_only: bool,
    diag: &mut SolverDiagnostics,
) -> Vec<TrackHypothesis<'edge_lf, 'seed_lf, 'alert_lf>> {
    let n = component_nodes.len();
    if n == 0 {
        return Vec::new();
    }

    // 1) membership
    let in_comp = build_membership(component_nodes);

    // 2) restricted adjacency + indegree
    let (out, indeg) = build_adjacency(cfg, edges, component_nodes, &in_comp, active_only, diag);

    // 3) sources
    let sources = find_sources(&out, &indeg);
    if sources.is_empty() {
        return Vec::new();
    }

    // 4) init beam + state pool
    let (mut states, mut beam) = init_beam_states(&sources, component_nodes);

    let mut emitted_per_source: Vec<u32> = vec![0; n];
    let mut terminal_states: Vec<usize> = Vec::new();
    let mut expansions: usize = 0;

    // 5) beam loop
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

    // Also consider sinks still present in beam (in case we exited early).
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

    // 6) reconstruct tracks
    let mut tracks: Vec<TrackHypothesis<'edge_lf, 'seed_lf, 'alert_lf>> = terminal_states
        .into_iter()
        .map(|sid| reconstruct_track(sid, &states, component_nodes))
        .filter(|t| t.nodes.len() >= cfg.min_nodes)
        .collect();

    // 7) final ranking + truncate
    sort_and_truncate_tracks(cfg, &mut tracks);
    tracks
}
