//! Connected component construction for the Fink-FAT inter-night graph.
//!
//! Overview
//! --------
//! This module computes **connected components** over the *undirected view* of
//! the inter-night graph, then builds a **restricted directed subgraph per component**
//! (adjacency + local degrees + sources/sinks) that can be consumed efficiently
//! by solvers such as the *trivial beam-search solver*.
//!
//! Why two views of the graph?
//! ---------------------------
//! - For component detection we want *undirected* connectivity:
//!   if there exists *any* directed edge between two seeds (either direction),
//!   we consider them in the same "blob" for later solving.
//! - For solving, we need the *directed* structure (time-forward edges) to
//!   enumerate paths and build trajectory hypotheses.
//!
//! ID spaces and mappings
//! ----------------------
//! The engine uses **three levels of identifiers** for seeds:
//!
//! 1) `SeedKey` (stable key)
//!    - Identifies a seed uniquely across nights.
//!    - `(night_id, idx_in_night)` where `idx_in_night` is the seed’s index in its night.
//!
//! 2) `global_idx` (dense index 0..N_total-1)
//!    - Built by `SeedGlobalIndex` from the `SeedStore`.
//!    - Used by Union-Find to compute components efficiently.
//!
//! 3) `local_idx` (dense index inside a component)
//!    - Index into `component_ref[component_id]`.
//!    - Used by solvers operating on *one* component at a time.
//!
//! The relationships are:
//!
//! ```text
//! SeedKey  <->  global_idx  --(comp_of_node)-->  component_id
//!    \                                         /
//!     \-> (local_of_key) -> (component_id, local_idx)
//! ```
//!
//! Example
//! -------
//! Suppose we have 2 components (C0 and C1). Global seeds are indexed 0..7.
//!
//! ```text
//! global_idx:  0   1   2   3   4   5   6   7
//! comp_of_node:0   0   0   0   1   1   1   1
//!
//! component_ref[0] (C0): local_idx -> SeedKey
//!   0 -> (N10,0)
//!   1 -> (N10,3)
//!   2 -> (N11,1)
//!   3 -> (N12,0)
//!
//! component_ref[1] (C1):
//!   0 -> (N10,2)
//!   1 -> (N11,0)
//!   2 -> (N11,4)
//!   3 -> (N12,2)
//! ```
//!
//! Then the restricted directed adjacency for C0 is stored as:
//!
//! ```text
//! component_out[0][local_u] = list of outgoing edges (&Edge) within C0
//! component_in_deg_local[0][local_u] = local in-degree in C0
//! component_out_deg_local[0][local_u] = local out-degree in C0
//! ```
//!
//! Directed adjacency layout
//! -------------------------
//! `component_out` is a 3-level nested vector with the following meaning:
//!
//! ```text
//! component_out[component_id][local_u] = Vec<EdgeRef>
//! ```
//!
//! where `EdgeRef` is a reference to the *global* edge stored in `RuntimeGraph.edges`.
//!
//! Lifetime model
//! --------------
//! This module stores `&Edge` references inside `ConnectedComponents`.
//! This is safe as long as `ConnectedComponents` lives no longer than the
//! `RuntimeGraph` it was built from (same `'edge_lf` lifetime).
//!
//! Active-only mode
//! ----------------
//! If `active_only == true`:
//! - only active edges are used for Union-Find connectivity,
//! - only active edges are inserted into the restricted directed adjacency.
//!
//! This matches the typical “solve on active edges” behavior.
//!
//! Notes
//! -----
//! - The restricted directed adjacency enforces *time-forward edges*
//!   (`night(to) > night(from)`). Back-edges are ignored.
//! - The adjacency lists are **not** globally deduplicated.
//!   (They should not contain duplicates if the graph builder does not emit any.)
//! - `graph.core.out_deg` is used as a best-effort capacity hint to reduce
//!   adjacency list reallocations.
//!
//! See also
//! --------
//! - `seed_index::SeedGlobalIndex` for the dense global indexing.
//! - `union_find::UnionFind` for connectivity.
//! - `solver_manager::SolverPolicy` for solver selection heuristics.

pub mod seed_index;
pub mod union_find;

use ahash::AHashMap;

use crate::{
    graph::{RuntimeGraph, edge::Edge},
    night_id::NightId,
    persistence::seed_node::SeedKey,
    pipeline::seed_store::SeedStore,
    seeding::seed_node::SeedNode,
    solver::{
        components::{seed_index::SeedGlobalIndex, union_find::UnionFind},
        solver_manager::{SolverChoice, SolverPolicy},
    },
};

/// Dense identifier for a connected component.
///
/// Components are assigned dense ids `[0, n_components)` by `dense_components()`.
pub type ComponentId = u32;

/// Borrowed node references grouped per component.
///
/// Layout
/// ------
/// `component_ref[cid]` is a list of node references belonging to the component `cid`.
pub type ComponentRef<'seed_lf, 'alert_lf> = Vec<Vec<&'seed_lf SeedNode<'alert_lf>>>;

/// Night bounds per component.
///
/// Layout
/// ------
/// `component_night_bound[cid] = Some((min_night, max_night))` if the component is non-empty.
pub type ComponentNightBound = Vec<Option<(NightId, NightId)>>;

/// Local node index inside one component.
///
/// `LocalIdx` indexes into `component_ref[component_id]`.
pub type LocalIdx = u32;

/// Reference to a global edge stored in `RuntimeGraph.edges`.
///
/// This module stores `&Edge` references for fast solver access without
/// rebuilding adjacency lists.
///
/// Lifetime
/// --------
/// - `'edge_lf` ties the reference to the lifetime of the `RuntimeGraph`.
pub type EdgeRef<'edge_lf, 'seed_lf, 'alert_lf> = &'edge_lf Edge<'seed_lf, 'alert_lf>;

/// Per-component directed adjacency (restricted to the component nodes).
///
/// Layout
/// ------
/// `component_out[cid][local_u] = Vec<EdgeRef>` lists outgoing edges from `local_u`
/// *that stay inside the same component*.
pub type ComponentOut<'edge_lf, 'seed_lf, 'alert_lf> =
    Vec<Vec<Vec<EdgeRef<'edge_lf, 'seed_lf, 'alert_lf>>>>;

/// Local degrees in the restricted directed subgraph.
///
/// Layout
/// ------
/// `component_in_deg_local[cid][local_u]` = in-degree of `local_u` within the restricted subgraph.
/// `component_out_deg_local[cid][local_u]` = out-degree of `local_u` within the restricted subgraph.
pub type ComponentDegLocal = Vec<Vec<u32>>;

/// Connected components plus per-component restricted directed adjacency.
///
/// This structure is the "bridge" between:
/// - the global inter-night graph (`RuntimeGraph`) and
/// - per-component solvers that need *local* directed structure.
///
/// Stored information
/// ------------------
/// - **Undirected connectivity** (via `comp_of_node`) computed by Union-Find.
/// - **Component membership** as borrowed seed references (`component_ref`).
/// - **Night span bounds** (`component_night_bound`).
/// - **Active intra-component edge counts** (`component_active_edge`).
/// - **Restricted directed subgraph** per component:
///   - outgoing adjacency (`component_out`),
///   - local degrees (`component_in_deg_local`, `component_out_deg_local`),
///   - local sources and sinks (`component_sources_local`, `component_sinks_local`).
///
/// Lifetime model
/// --------------
/// The directed adjacency stores `&Edge` references. Therefore:
/// - the `RuntimeGraph` used to build this object must outlive it (`'edge_lf`),
/// - the `SeedStore` / `SeedNode` references must outlive it (`'seed_lf`, `'alert_lf`).
#[derive(Debug, Clone)]
pub struct ConnectedComponents<'edge_lf, 'seed_lf, 'alert_lf> {
    /// Dense global seed index (SeedKey <-> global_idx).
    index: SeedGlobalIndex,

    /// Component id for each global node index.
    ///
    /// Layout
    /// ------
    /// `comp_of_node[global_idx] = component_id`.
    comp_of_node: Vec<ComponentId>,

    /// Number of components (dense ids 0..n_components-1).
    pub n_components: u32,

    /// Nodes of each component as borrowed references.
    component_ref: ComponentRef<'seed_lf, 'alert_lf>,

    /// Night bounds per component (min,max).
    component_night_bound: ComponentNightBound,

    /// Number of *active* edges that stay within each component.
    component_active_edge: Vec<u32>,

    // ---------------------------------------------------------------------
    // Directed intra-component graph (restricted adjacency + degrees)
    // ---------------------------------------------------------------------
    /// Map each seed key to its component-local coordinates.
    ///
    /// Layout
    /// ------
    /// `local_of_key[SeedKey] = (component_id, local_idx)`
    /// where `local_idx` indexes into `component_ref[component_id]`.
    local_of_key: AHashMap<SeedKey, (ComponentId, LocalIdx)>,

    /// Outgoing adjacency lists (restricted to each component).
    ///
    /// Layout
    /// ------
    /// `component_out[cid][local_u] = Vec<EdgeRef>` contains edges:
    /// - whose `from` key corresponds to `local_u`,
    /// - whose `to` key is also in the same component `cid`,
    /// - optionally restricted to active edges only (depending on `active_only` in `compute()`),
    /// - and always respecting `night(to) > night(from)` (back-edges ignored).
    component_out: ComponentOut<'edge_lf, 'seed_lf, 'alert_lf>,

    /// Local in-degree per node in the restricted directed subgraph.
    component_in_deg_local: ComponentDegLocal,

    /// Local out-degree per node in the restricted directed subgraph.
    component_out_deg_local: ComponentDegLocal,

    /// Precomputed local sources per component.
    ///
    /// Definition
    /// ----------
    /// A node is a **source** if:
    /// - local in-degree == 0
    /// - local out-degree > 0
    ///
    /// Notes
    /// -----
    /// - If a component has no such node (degenerate case), a fallback list is built:
    ///   all nodes with local out-degree > 0.
    component_sources_local: Vec<Vec<LocalIdx>>,

    /// Precomputed local sinks per component.
    ///
    /// Definition
    /// ----------
    /// A node is a **sink** if local out-degree == 0.
    component_sinks_local: Vec<Vec<LocalIdx>>,
}

impl<'edge_lf, 'seed_lf, 'alert_lf> ConnectedComponents<'edge_lf, 'seed_lf, 'alert_lf> {
    /// Compute connected components and build per-component directed adjacency.
    ///
    /// Behavior
    /// --------
    /// 1) Build a dense `global_idx` mapping for all seeds (`SeedGlobalIndex`).
    /// 2) Run Union-Find on the **undirected** view of the graph:
    ///    each directed edge `(u -> v)` unions nodes `u` and `v`.
    /// 3) Convert Union-Find roots to dense component ids (`0..C-1`) stored in `comp_of_node`.
    /// 4) Count active intra-component edges for diagnostics / policy decisions.
    /// 5) Build:
    ///    - `component_ref`: borrowed nodes per component,
    ///    - `component_night_bound`: min/max nights per component,
    ///    - `local_of_key`: `SeedKey -> (component_id, local_idx)`.
    /// 6) Build the restricted **directed** subgraph per component:
    ///    adjacency + degrees + local sources/sinks.
    ///
    /// Arguments
    /// ---------
    /// * `seed_store` – Seed storage grouped by night; used to build the global index and
    ///   to populate `component_ref`.
    /// * `graph` – Runtime graph containing:
    ///   - `graph.edges`: global directed edges storing references to `SeedNode`s,
    ///   - `graph.core`: global degree maps used as capacity hints.
    /// * `active_only` – If `true`:
    ///   - Union-Find only considers active edges,
    ///   - directed adjacency only includes active edges.
    ///
    /// Return
    /// ------
    /// `ConnectedComponents` containing:
    /// - component membership for each node,
    /// - borrowed node lists per component,
    /// - restricted directed adjacency and local degrees.
    ///
    /// Notes
    /// -----
    /// - This function assumes the `RuntimeGraph` edges are time-forward.
    ///   It still enforces `night(to) > night(from)` defensively when building adjacency.
    pub fn compute(
        seed_store: &'seed_lf SeedStore<'alert_lf>,
        graph: &'edge_lf RuntimeGraph<'seed_lf, 'alert_lf>,
        active_only: bool,
    ) -> Self {
        let index = SeedGlobalIndex::build(seed_store);
        let n_total = index.n_total();

        // 1) Union-Find on nodes (undirected view).
        let mut uf = UnionFind::new(n_total);
        Self::union_edges(&index, &mut uf, &graph.edges, active_only);

        // 2) Dense component ids + comp_of_node.
        let (comp_of_node, n_components) = Self::dense_components(&mut uf, n_total);

        // 3) Count active intra-component edges (always active ones).
        let component_active_edge =
            Self::count_active_intra_edges(&index, &graph.edges, &comp_of_node, n_components);

        // 4) Build component -> node refs, night bounds, and local_of_key.
        let (component_ref, component_night_bound, local_of_key) =
            Self::build_refs_bounds_and_local_map(seed_store, &index, &comp_of_node, n_components);

        // 5) Build restricted directed adjacency + local degrees + sources/sinks.
        let (
            component_out,
            component_in_deg_local,
            component_out_deg_local,
            component_sources_local,
            component_sinks_local,
        ) = Self::build_local_digraph(
            graph,
            active_only,
            n_components,
            &component_ref,
            &local_of_key,
            &comp_of_node,
            &index,
        );

        Self {
            index,
            comp_of_node,
            n_components: n_components as u32,
            component_ref,
            component_night_bound,
            component_active_edge,

            local_of_key,
            component_out,
            component_in_deg_local,
            component_out_deg_local,
            component_sources_local,
            component_sinks_local,
        }
    }

    // -------------------------------------------------------------------------
    // Internal helpers (undirected connectivity)
    // -------------------------------------------------------------------------

    /// Union all graph edges in Union-Find (undirected view).
    ///
    /// Each directed edge `(from -> to)` is treated as an undirected connection,
    /// and unions the two corresponding nodes.
    ///
    /// Arguments
    /// ---------
    /// * `index` – Global dense index mapping `SeedKey -> global_idx`.
    /// * `uf` – Union-Find data structure updated in-place.
    /// * `edges` – Global directed edges.
    /// * `active_only` – If `true`, unions only edges with `edge.core.active == true`.
    ///
    /// Notes
    /// -----
    /// - This step groups together all nodes that are connected by at least one edge,
    ///   regardless of direction.
    fn union_edges(index: &SeedGlobalIndex, uf: &mut UnionFind, edges: &[Edge], active_only: bool) {
        for e in edges {
            if active_only && !e.core.active {
                continue;
            }
            let u = index.idx_of_key(e.from.core.key);
            let v = index.idx_of_key(e.to.core.key);
            uf.union(u, v);
        }
    }

    /// Assign dense component ids to Union-Find roots.
    ///
    /// After Union-Find is built, each node `i` has a root `r = uf.find(i)`.
    /// This function assigns a dense `ComponentId` to each unique root and returns:
    /// - `comp_of_node[i] = component_id`
    /// - `n_components`
    ///
    /// Arguments
    /// ---------
    /// * `uf` – Union-Find containing final connectivity.
    /// * `n_total` – Number of nodes in the global index space.
    ///
    /// Return
    /// ------
    /// * `(comp_of_node, n_components)` – Dense component ids `[0..n_components)`.
    ///
    /// Notes
    /// -----
    /// - Component ids are assigned in first-seen order of roots (iteration over nodes).
    fn dense_components(uf: &mut UnionFind, n_total: usize) -> (Vec<ComponentId>, usize) {
        let mut root_to_cid: AHashMap<usize, ComponentId> = AHashMap::default();
        let mut comp_of_node = vec![0u32; n_total];
        let mut n_components: ComponentId = 0;

        for i in 0..n_total {
            let r = uf.find(i);
            let cid = *root_to_cid.entry(r).or_insert_with(|| {
                let c = n_components;
                n_components += 1;
                c
            });
            comp_of_node[i] = cid;
        }

        (comp_of_node, n_components as usize)
    }

    /// Count the number of active edges whose endpoints lie in the same component.
    ///
    /// This is used primarily for solver policy decisions / diagnostics.
    ///
    /// Arguments
    /// ---------
    /// * `index` – Global dense index mapping.
    /// * `edges` – Global directed edges.
    /// * `comp_of_node` – Component id for each global node index.
    /// * `n_components` – Number of components.
    ///
    /// Return
    /// ------
    /// `Vec<u32>` of length `n_components` where `counts[cid]` is the number of
    /// **active** edges fully contained inside component `cid`.
    ///
    /// Notes
    /// -----
    /// - This function *always* counts only active edges, regardless of the
    ///   `active_only` flag used when computing connectivity.
    fn count_active_intra_edges(
        index: &SeedGlobalIndex,
        edges: &[Edge],
        comp_of_node: &[ComponentId],
        n_components: usize,
    ) -> Vec<u32> {
        let mut counts = vec![0u32; n_components];

        for e in edges {
            if !e.core.active {
                continue;
            }
            let u = index.idx_of_key(e.from.core.key);
            let v = index.idx_of_key(e.to.core.key);

            let cu = comp_of_node[u];
            let cv = comp_of_node[v];
            if cu == cv {
                counts[cu as usize] += 1;
            }
        }

        counts
    }

    // -------------------------------------------------------------------------
    // Internal helpers (component membership lists + bounds)
    // -------------------------------------------------------------------------

    /// Build:
    /// - `component_ref`: borrowed nodes per component,
    /// - `component_night_bound`: min/max nights per component,
    /// - `local_of_key`: `SeedKey -> (component_id, local_idx)`.
    ///
    /// Arguments
    /// ---------
    /// * `seed_store` – Seed storage grouped by night.
    /// * `index` – Global dense index mapping.
    /// * `comp_of_node` – Component id for each global node index.
    /// * `n_components` – Number of components.
    ///
    /// Return
    /// ------
    /// * `(component_ref, component_night_bound, local_of_key)`
    ///
    /// Notes
    /// -----
    /// - The order of `component_ref[cid]` is the iteration order of `seed_store.iter()`.
    ///   It is deterministic if `SeedStore::iter()` is deterministic.
    /// - `local_of_key` is the bridge needed later to convert an edge endpoint (`SeedKey`)
    ///   into a component-local index.
    fn build_refs_bounds_and_local_map(
        seed_store: &'seed_lf SeedStore<'alert_lf>,
        index: &SeedGlobalIndex,
        comp_of_node: &[ComponentId],
        n_components: usize,
    ) -> (
        ComponentRef<'seed_lf, 'alert_lf>,
        ComponentNightBound,
        AHashMap<SeedKey, (ComponentId, LocalIdx)>,
    ) {
        // Pre-compute component sizes to reserve capacity.
        let mut sizes = vec![0u32; n_components];
        for &cid in comp_of_node {
            sizes[cid as usize] += 1;
        }

        let mut component_ref: ComponentRef<'seed_lf, 'alert_lf> = (0..n_components)
            .map(|cid| Vec::with_capacity(sizes[cid] as usize))
            .collect();

        let mut local_of_key: AHashMap<SeedKey, (ComponentId, LocalIdx)> = AHashMap::default();
        local_of_key.reserve(index.n_total());

        let mut min_night: Vec<Option<NightId>> = vec![None; n_components];
        let mut max_night: Vec<Option<NightId>> = vec![None; n_components];

        for (night_id, seeds) in seed_store.iter() {
            for seed in seeds {
                let gid = index.idx_of_key(seed.core.key);
                let cid = comp_of_node[gid] as usize;

                let local_idx = component_ref[cid].len() as u32;
                component_ref[cid].push(seed);
                local_of_key.insert(seed.core.key, (cid as u32, local_idx));

                min_night[cid] = Some(match min_night[cid] {
                    None => *night_id,
                    Some(x) => x.min(*night_id),
                });
                max_night[cid] = Some(match max_night[cid] {
                    None => *night_id,
                    Some(x) => x.max(*night_id),
                });
            }
        }

        let component_night_bound: ComponentNightBound = (0..n_components)
            .map(|cid| match (min_night[cid], max_night[cid]) {
                (Some(a), Some(b)) => Some((a, b)),
                _ => None,
            })
            .collect();

        (component_ref, component_night_bound, local_of_key)
    }

    // -------------------------------------------------------------------------
    // Internal helpers (restricted directed subgraph)
    // -------------------------------------------------------------------------

    /// Build the restricted directed subgraph per component.
    ///
    /// This step constructs:
    /// - `component_out[cid][local_u]` outgoing adjacency (as `&Edge` references),
    /// - `in_deg_local` and `out_deg_local` inside the restricted subgraph,
    /// - `sources_local` and `sinks_local` (precomputed solver entry/exit points).
    ///
    /// Arguments
    /// ---------
    /// * `graph` – Global runtime graph (`edges` + global degrees in `graph.core`).
    /// * `active_only` – If `true`, only active edges are included.
    /// * `n_components` – Number of components.
    /// * `component_ref` – Borrowed node lists per component, used for sizing.
    /// * `local_of_key` – `SeedKey -> (component_id, local_idx)` mapping.
    /// * `comp_of_node` – `global_idx -> component_id` mapping.
    /// * `index` – Global dense index mapping.
    ///
    /// Return
    /// ------
    /// * `(component_out, in_deg_local, out_deg_local, sources_local, sinks_local)`
    ///
    /// Notes
    /// -----
    /// - Enforces time-forward directed edges (`night(to) > night(from)`); back-edges are ignored.
    /// - Uses `graph.core.out_deg` as a best-effort capacity hint for adjacency allocation.
    fn build_local_digraph(
        graph: &'edge_lf RuntimeGraph<'seed_lf, 'alert_lf>,
        active_only: bool,
        n_components: usize,
        component_ref: &ComponentRef<'seed_lf, 'alert_lf>,
        local_of_key: &AHashMap<SeedKey, (ComponentId, LocalIdx)>,
        comp_of_node: &[ComponentId],
        index: &SeedGlobalIndex,
    ) -> (
        ComponentOut<'edge_lf, 'seed_lf, 'alert_lf>,
        ComponentDegLocal,
        ComponentDegLocal,
        Vec<Vec<LocalIdx>>,
        Vec<Vec<LocalIdx>>,
    ) {
        // Allocate per-component node arrays.
        let mut component_out: ComponentOut<'edge_lf, 'seed_lf, 'alert_lf> = (0..n_components)
            .map(|cid| {
                let n = component_ref[cid].len();
                (0..n).map(|_| Vec::new()).collect::<Vec<_>>()
            })
            .collect();

        let mut in_deg_local: ComponentDegLocal = (0..n_components)
            .map(|cid| vec![0u32; component_ref[cid].len()])
            .collect();

        let mut out_deg_local: ComponentDegLocal = (0..n_components)
            .map(|cid| vec![0u32; component_ref[cid].len()])
            .collect();

        // Exploit graph.core degrees to reserve outgoing capacities (best-effort).
        //
        // We do *not* assume global out-degree equals intra-component out-degree;
        // it is only an upper bound / hint.
        for cid in 0..n_components {
            for (lu, node) in component_ref[cid].iter().enumerate() {
                let key = node.core.key;
                let cap = graph.core.out_deg.get(&key).copied().unwrap_or(0);
                component_out[cid][lu].reserve(cap.min(64));
            }
        }

        // Single pass over edges: keep only intra-component directed edges.
        for e in graph.edges.iter() {
            if active_only && !e.core.active {
                continue;
            }

            // Enforce time-forward directed edges (gaps allowed).
            let n0 = e.from.core.night_id().value();
            let n1 = e.to.core.night_id().value();
            if n1 <= n0 {
                continue;
            }

            // Fast component id check using global index.
            let u_gid = index.idx_of_key(e.from.core.key);
            let v_gid = index.idx_of_key(e.to.core.key);
            let cu = comp_of_node[u_gid];
            let cv = comp_of_node[v_gid];
            if cu != cv {
                continue;
            }

            // Convert endpoints to component-local indices.
            let Some(&(cid_u, lu)) = local_of_key.get(&e.from.core.key) else {
                continue;
            };
            let Some(&(_, lv)) = local_of_key.get(&e.to.core.key) else {
                continue;
            };
            debug_assert_eq!(cid_u, cu);

            let cid = cu as usize;
            let lu_usize = lu as usize;
            let lv_usize = lv as usize;

            component_out[cid][lu_usize].push(e);
            out_deg_local[cid][lu_usize] += 1;
            in_deg_local[cid][lv_usize] += 1;
        }

        // Precompute sources/sinks in the restricted digraph.
        //
        // A source is a node with (in_deg==0, out_deg>0).
        // A sink is a node with (out_deg==0).
        //
        // If a component has no sources (degenerate case), we fall back to
        // "any node with out_deg>0" as allowed starting points.
        let mut sources_local: Vec<Vec<LocalIdx>> = Vec::with_capacity(n_components);
        let mut sinks_local: Vec<Vec<LocalIdx>> = Vec::with_capacity(n_components);

        for cid in 0..n_components {
            let n = component_ref[cid].len();

            let mut srcs = Vec::new();
            let mut snks = Vec::new();

            for u in 0..n {
                let indeg = in_deg_local[cid][u];
                let outdeg = out_deg_local[cid][u];

                if outdeg == 0 {
                    snks.push(u as u32);
                }
                if indeg == 0 && outdeg > 0 {
                    srcs.push(u as u32);
                }
            }

            if srcs.is_empty() {
                for u in 0..n {
                    if out_deg_local[cid][u] > 0 {
                        srcs.push(u as u32);
                    }
                }
            }

            sources_local.push(srcs);
            sinks_local.push(snks);
        }

        (
            component_out,
            in_deg_local,
            out_deg_local,
            sources_local,
            sinks_local,
        )
    }

    // -------------------------------------------------------------------------
    // Unified getters / accessors
    // -------------------------------------------------------------------------

    /// Return the component id containing a given seed key.
    ///
    /// Arguments
    /// ---------
    /// * `seed_key` – Stable seed identifier.
    ///
    /// Return
    /// ------
    /// `ComponentId` for the connected component that contains this seed.
    ///
    /// Notes
    /// -----
    /// - This lookup uses the global dense index (`SeedGlobalIndex`) + `comp_of_node`.
    pub fn component_id_of_seed(&self, seed_key: SeedKey) -> ComponentId {
        let gid = self.index.idx_of_key(seed_key);
        self.comp_of_node[gid]
    }

    /// Return the nodes of a component as borrowed references.
    ///
    /// Arguments
    /// ---------
    /// * `component_id` – Dense component id.
    ///
    /// Return
    /// ------
    /// `&[&SeedNode]` slice of nodes in this component, in component-local order.
    pub fn component_nodes(&self, component_id: ComponentId) -> &[&SeedNode<'alert_lf>] {
        &self.component_ref[component_id as usize]
    }

    /// Return the number of nodes in a component.
    ///
    /// Arguments
    /// ---------
    /// * `component_id` – Dense component id.
    ///
    /// Return
    /// ------
    /// Component size in nodes.
    pub fn component_size(&self, component_id: ComponentId) -> usize {
        self.component_ref[component_id as usize].len()
    }

    /// Return the number of *active* edges whose endpoints are inside the component.
    ///
    /// Arguments
    /// ---------
    /// * `component_id` – Dense component id.
    ///
    /// Return
    /// ------
    /// Count of active intra-component edges.
    ///
    /// Notes
    /// -----
    /// - This is a per-component diagnostic / policy input, not necessarily the number
    ///   of edges present in `component_out` (which depends on `active_only` during compute).
    pub fn component_active_edges(&self, component_id: ComponentId) -> u32 {
        self.component_active_edge[component_id as usize]
    }

    /// Return the (min_night, max_night) bounds for the component, if any.
    ///
    /// Arguments
    /// ---------
    /// * `component_id` – Dense component id.
    ///
    /// Return
    /// ------
    /// `Some((min_night, max_night))` if the component is non-empty, otherwise `None`.
    pub fn component_night_bounds(&self, component_id: ComponentId) -> Option<(NightId, NightId)> {
        self.component_night_bound[component_id as usize]
    }

    /// Return the night span (integer days) for the component.
    ///
    /// This is `max_night - min_night` with saturating subtraction.
    ///
    /// Arguments
    /// ---------
    /// * `component_id` – Dense component id.
    ///
    /// Return
    /// ------
    /// Night span in integer days (0 if bounds are missing).
    pub fn component_night_span(&self, component_id: ComponentId) -> u32 {
        match self.component_night_bounds(component_id) {
            Some((min_n, max_n)) => {
                let min: u32 = min_n.into();
                let max: u32 = max_n.into();
                max.saturating_sub(min)
            }
            None => 0,
        }
    }

    /// Return the restricted outgoing adjacency for a component.
    ///
    /// Arguments
    /// ---------
    /// * `component_id` – Dense component id.
    ///
    /// Return
    /// ------
    /// Slice where `out[local_u]` is the list of `&Edge` outgoing from `local_u`
    /// within the component.
    ///
    /// Notes
    /// -----
    /// - The adjacency is built at `compute()` time, so solvers do not need to rescan
    ///   `graph.edges` to build local structure.
    pub fn component_out_edges(
        &self,
        component_id: ComponentId,
    ) -> &[Vec<EdgeRef<'edge_lf, 'seed_lf, 'alert_lf>>] {
        &self.component_out[component_id as usize]
    }

    /// Return the restricted local in-degrees for a component.
    ///
    /// Arguments
    /// ---------
    /// * `component_id` – Dense component id.
    ///
    /// Return
    /// ------
    /// Slice where `in_deg[local_u]` is the number of incoming edges to `local_u`
    /// within the restricted directed subgraph.
    pub fn component_in_deg_local(&self, component_id: ComponentId) -> &[u32] {
        &self.component_in_deg_local[component_id as usize]
    }

    /// Return the restricted local out-degrees for a component.
    ///
    /// Arguments
    /// ---------
    /// * `component_id` – Dense component id.
    ///
    /// Return
    /// ------
    /// Slice where `out_deg[local_u]` is the number of outgoing edges from `local_u`
    /// within the restricted directed subgraph.
    pub fn component_out_deg_local(&self, component_id: ComponentId) -> &[u32] {
        &self.component_out_deg_local[component_id as usize]
    }

    /// Return the precomputed local sources for a component.
    ///
    /// Arguments
    /// ---------
    /// * `component_id` – Dense component id.
    ///
    /// Return
    /// ------
    /// Slice of local indices `u` such that:
    /// - `in_deg_local[u] == 0`
    /// - `out_deg_local[u] > 0`
    ///
    /// Notes
    /// -----
    /// - If the component has no such node, this returns a fallback list:
    ///   all nodes with `out_deg_local[u] > 0`.
    pub fn component_sources_local(&self, component_id: ComponentId) -> &[LocalIdx] {
        &self.component_sources_local[component_id as usize]
    }

    /// Return the precomputed local sinks for a component.
    ///
    /// Arguments
    /// ---------
    /// * `component_id` – Dense component id.
    ///
    /// Return
    /// ------
    /// Slice of local indices `u` such that `out_deg_local[u] == 0`.
    pub fn component_sinks_local(&self, component_id: ComponentId) -> &[LocalIdx] {
        &self.component_sinks_local[component_id as usize]
    }

    /// Classify a component into a solver choice using the current policy.
    ///
    /// Strategy
    /// --------
    /// - Use a trivial solver for small components.
    /// - Reject very wide night spans from Min-Cost Flow (MCF).
    /// - Otherwise estimate MCF time and choose between MCF and BlobBreaker.
    ///
    /// Arguments
    /// ---------
    /// * `component_id` – Dense component id.
    /// * `solver_policy` – Policy containing thresholds and time budget model.
    ///
    /// Return
    /// ------
    /// `SolverChoice` selected for this component.
    ///
    /// Notes
    /// -----
    /// - This classification uses:
    ///   - component size,
    ///   - night span,
    ///   - number of active intra-component edges.
    pub fn classify(
        &self,
        component_id: ComponentId,
        solver_policy: &SolverPolicy,
    ) -> SolverChoice {
        let n = self.component_size(component_id);

        if n <= solver_policy.trivial_max_nodes as usize {
            return SolverChoice::Trivial;
        }

        if self.component_night_span(component_id) > solver_policy.max_night_span_for_mcf {
            return SolverChoice::BlobBreaker;
        }

        let t_est =
            solver_policy.estimate_mcf_time_s(n as u32, self.component_active_edges(component_id));

        if t_est <= solver_policy.mcf_budget_s {
            SolverChoice::MinCostFlow
        } else {
            SolverChoice::BlobBreaker
        }
    }
}
