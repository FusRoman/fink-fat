//! Connected component construction for the inter-night graph.
//!
//! Overview
//! --------
//! This module computes **connected components** over the *undirected view* of the
//! inter-night graph and materializes, for each component, a **restricted directed
//! subgraph** that can be consumed efficiently by solvers.
//!
//! The primary goal is to provide solvers with a component-local representation:
//!
//! - the list of nodes in the component (borrowed seed references),
//! - a directed outgoing adjacency restricted to those nodes (borrowed edge references),
//! - local in/out degrees inside that restricted directed subgraph,
//! - precomputed local sources and sinks.
//!
//! This keeps solvers focused on the local combinatorial problem and avoids repeated
//! global scans or per-solver reconstruction of adjacency.
//!
//! Connectivity vs directed structure
//! ---------------------------------
//! The component partition is computed using **undirected connectivity**:
//! two nodes belong to the same component if there exists a path between them when
//! ignoring edge directions.
//!
//! Solvers, however, operate on a **directed time-forward** structure:
//! edges are intended to connect older to newer nights. This module therefore
//! builds a directed adjacency per component and enforces a time-forward guard
//! (`night(to) > night(from)`) when inserting edges into that adjacency.
//!
//! Identifier spaces and mappings
//! ------------------------------
//! The engine uses three related identifier spaces for seeds:
//!
//! 1) `SeedKey` (stable key)
//!    - Unique identifier for a seed across runs and persistence.
//!    - Typically `(night_id, idx_in_night)`.
//!
//! 2) `global_idx` (dense index in `0..N_total`)
//!    - Built by `SeedGlobalIndex` from the `SeedStore`.
//!    - Used by Union-Find to compute connectivity efficiently.
//!
//! 3) `local_idx` (dense index inside one component)
//!    - Index into `component_ref[component_id]`.
//!    - Used by solvers to index component-local arrays.
//!
//! The mapping relationships are:
//!
//! ```text
//! SeedKey <-> global_idx --(comp_of_node)--> component_id
//!    \                                      /
//!     \-> local_of_key: SeedKey -> (component_id, local_idx)
//! ```
//!
//! Directed adjacency layout
//! -------------------------
//! The restricted directed adjacency is stored as:
//!
//! ```text
//! component_out[component_id][local_u] = Vec<&Edge>
//! ```
//!
//! Each adjacency list contains references to the **global** edges stored in
//! `RuntimeGraph.edges`.
//!
//! Lifetime model
//! --------------
//! `ConnectedComponents` stores borrowed references:
//!
//! - `&SeedNode` references borrowed from the `SeedStore` (via `component_ref`),
//! - `&Edge` references borrowed from the `RuntimeGraph.edges` (via `component_out`).
//!
//! Therefore, `ConnectedComponents` must not outlive:
//! - the `RuntimeGraph` used to build it (edge references),
//! - the `SeedStore` / seeds used to build it (seed references).
//!
//! Active-only mode
//! ----------------
//! The `compute()` method supports an `active_only` flag:
//!
//! - if `active_only == true`:
//!   - Union-Find connectivity uses only edges with `edge.core.active == true`,
//!   - the restricted directed adjacency also includes only active edges.
//!
//! This matches the typical solver behavior that operates on active edges only.
//!
//! Notes
//! -----
//! - The restricted directed adjacency ignores "back-edges" where `night(to) <= night(from)`.
//! - The adjacency lists are not globally deduplicated.
//! - `graph.core.out_deg` is used as a best-effort capacity hint to reduce reallocations.
//!
//! See also
//! --------
//! - `seed_index::SeedGlobalIndex` for dense global indexing.
//! - `union_find::UnionFind` for connectivity.
//! - `solver_manager::SolverPolicy` for solver routing heuristics.

pub mod seed_index;
pub mod union_find;

use ahash::AHashMap;

use crate::{
    engine_config::solver_config::solver_policy::{SolverChoice, SolverPolicy, SolverRoutingMode},
    graph::{RuntimeGraph, edge::Edge},
    night_id::NightId,
    persistence::seed_node::SeedKey,
    pipeline::seed_store::SeedStore,
    seeding::seed_node::SeedNode,
    solver::components::{seed_index::SeedGlobalIndex, union_find::UnionFind},
};

/// Dense identifier for a connected component.
///
/// Components are assigned dense ids `[0, n_components)` during component
/// materialization.
pub type ComponentId = u32;

/// Borrowed node references grouped per component.
///
/// Layout
/// ------
/// `component_ref[cid]` is the list of nodes belonging to component `cid`.
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
/// Lifetime
/// --------
/// `'edge_lf` ties this reference to the lifetime of the `RuntimeGraph`.
pub type EdgeRef<'edge_lf, 'seed_lf, 'alert_lf> = &'edge_lf Edge<'seed_lf, 'alert_lf>;

/// Per-component directed adjacency (restricted to the component nodes).
///
/// Layout
/// ------
/// `component_out[cid][local_u] = Vec<EdgeRef>` lists outgoing edges from `local_u`
/// that stay inside the same component.
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
/// This structure provides a component-local graph view suitable for solvers:
/// - component membership (`component_ref`),
/// - night bounds (`component_night_bound`),
/// - restricted directed adjacency (`component_out`) using borrowed `&Edge` refs,
/// - local degrees and precomputed sources/sinks.
///
/// Stored information
/// ------------------
/// Connectivity:
/// - `index`: `SeedKey <-> global_idx` mapping.
/// - `comp_of_node[global_idx] = component_id`.
///
/// Component materialization:
/// - `component_ref[component_id]`: borrowed nodes.
/// - `component_night_bound[component_id]`: `(min_night, max_night)` bounds.
/// - `component_active_edge[component_id]`: count of active intra-component edges.
///
/// Restricted directed subgraph per component:
/// - `component_out[component_id][local_u]`: outgoing adjacency as `&Edge`.
/// - `component_in_deg_local`, `component_out_deg_local`.
/// - `component_sources_local`, `component_sinks_local`.
///
/// Lifetime model
/// --------------
/// The adjacency stores `&Edge` references. Therefore:
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

    /// Number of components (dense ids `0..n_components-1`).
    pub n_components: u32,

    /// Nodes of each component as borrowed references.
    component_ref: ComponentRef<'seed_lf, 'alert_lf>,

    /// Night bounds per component (min,max).
    component_night_bound: ComponentNightBound,

    /// Number of active edges that stay within each component.
    component_active_edge: Vec<u32>,

    // ---------------------------------------------------------------------
    // Directed intra-component graph (restricted adjacency + degrees)
    // ---------------------------------------------------------------------
    /// Outgoing adjacency lists (restricted to each component).
    ///
    /// Layout
    /// ------
    /// `component_out[cid][local_u] = Vec<EdgeRef>` contains edges:
    /// - whose `from` corresponds to `local_u`,
    /// - whose `to` is also in the same component `cid`,
    /// - included only if allowed by `active_only` at `compute()` time,
    /// - and satisfying the time-forward guard `night(to) > night(from)`.
    component_out: ComponentOut<'edge_lf, 'seed_lf, 'alert_lf>,

    /// Local in-degree per node inside the restricted directed subgraph.
    component_in_deg_local: ComponentDegLocal,

    /// Local out-degree per node inside the restricted directed subgraph.
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
    /// If a component has no sources (degenerate case), a fallback list is used:
    /// all nodes with local out-degree > 0.
    component_sources_local: Vec<Vec<LocalIdx>>,

    /// Precomputed local sinks per component.
    ///
    /// Definition
    /// ----------
    /// A node is a **sink** if local out-degree == 0.
    component_sinks_local: Vec<Vec<LocalIdx>>,
}

impl<'edge_lf, 'seed_lf, 'alert_lf> ConnectedComponents<'edge_lf, 'seed_lf, 'alert_lf> {
    /// Compute connected components and build per-component restricted directed adjacency.
    ///
    /// This is the main entrypoint that materializes both:
    /// - the undirected component partition,
    /// - the solver-facing directed subgraph per component.
    ///
    /// Algorithm
    /// ---------
    /// 1) Build the dense global index (`SeedGlobalIndex`) for all seeds.
    ///
    /// 2) Run Union-Find on undirected connectivity:
    ///    - for each directed edge `(u -> v)`, union `(u, v)`,
    ///    - optionally restricted to active edges only (`active_only`).
    ///
    /// 3) Convert Union-Find roots to dense component ids (`0..C-1`):
    ///    - store `comp_of_node[global_idx] = component_id`.
    ///
    /// 4) Count active intra-component edges for each component:
    ///    - used for diagnostics and solver routing policies.
    ///
    /// 5) Materialize component membership and indexing:
    ///    - build `component_ref[cid] = Vec<&SeedNode>`,
    ///    - compute `component_night_bound[cid] = (min_night, max_night)`,
    ///    - build `local_of_key: SeedKey -> (component_id, local_idx)`.
    ///
    /// 6) Build the restricted directed subgraph per component:
    ///    - `component_out[cid][local_u] = Vec<&Edge>` inside the component,
    ///    - local in/out degrees,
    ///    - sources and sinks.
    ///
    /// Arguments
    /// ---------
    /// * `seed_store` – Seed storage grouped by night; used to build the global index and
    ///   materialize node references per component.
    /// * `graph` – Runtime graph containing:
    ///   - `graph.edges`: global directed edges storing references to `SeedNode`s,
    ///   - `graph.core`: global degree maps used as capacity hints.
    /// * `active_only` – If `true`:
    ///   - Union-Find considers only `edge.core.active == true`,
    ///   - adjacency includes only active edges.
    ///
    /// Return
    /// ------
    /// `ConnectedComponents` containing:
    /// - membership and night bounds,
    /// - restricted directed adjacency and local degrees,
    /// - sources/sinks for solver initialization.
    ///
    /// Notes
    /// -----
    /// - The directed adjacency enforces `night(to) > night(from)` defensively.
    /// - Component ids are dense and stable for the duration of this object.
    pub fn compute(
        seed_store: &'seed_lf SeedStore<'alert_lf>,
        graph: &'edge_lf RuntimeGraph<'seed_lf, 'alert_lf>,
        active_only: bool,
    ) -> Self {
        let index = SeedGlobalIndex::build(seed_store);
        let n_total = index.n_total();

        // 1) Union-Find over the undirected view of the graph.
        let mut uf = UnionFind::new(n_total);
        Self::union_edges(&index, &mut uf, &graph.edges, active_only);

        // 2) Assign dense component ids.
        let (comp_of_node, n_components) = Self::dense_components(&mut uf, n_total);

        // 3) Count active intra-component edges for diagnostics and policies.
        let component_active_edge =
            Self::count_active_intra_edges(&index, &graph.edges, &comp_of_node, n_components);

        // 4) Materialize component membership lists and per-component bounds.
        let (component_ref, component_night_bound, local_of_key) =
            Self::build_refs_bounds_and_local_map(seed_store, &index, &comp_of_node, n_components);

        // 5) Build the restricted directed subgraph per component.
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

    /// Union all graph edges in Union-Find using an undirected interpretation.
    ///
    /// Each directed edge `(from -> to)` is treated as an undirected connection
    /// and unions the two corresponding nodes.
    ///
    /// Arguments
    /// ---------
    /// * `index` – Dense global index mapping `SeedKey -> global_idx`.
    /// * `uf` – Union-Find updated in-place.
    /// * `edges` – Global directed edges.
    /// * `active_only` – If `true`, union only edges with `edge.core.active == true`.
    ///
    /// Notes
    /// -----
    /// - This step groups together nodes that are connected by at least one edge,
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
    /// This is used as a component-level signal for diagnostics and routing policies.
    ///
    /// Arguments
    /// ---------
    /// * `index` – Dense global index mapping.
    /// * `edges` – Global directed edges.
    /// * `comp_of_node` – Component id for each global node index.
    /// * `n_components` – Number of components.
    ///
    /// Return
    /// ------
    /// `Vec<u32>` of length `n_components` where `counts[cid]` is the number of
    /// active edges fully contained inside component `cid`.
    ///
    /// Notes
    /// -----
    /// - This function counts active edges only, independent of `active_only`.
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

    /// Build the component membership lists, night bounds, and local indexing map.
    ///
    /// This step materializes:
    /// - `component_ref[cid]`: borrowed seed references,
    /// - `component_night_bound[cid]`: `(min_night, max_night)` span,
    /// - `local_of_key`: `SeedKey -> (component_id, local_idx)`.
    ///
    /// Arguments
    /// ---------
    /// * `seed_store` – Seed storage grouped by night.
    /// * `index` – Dense global index mapping.
    /// * `comp_of_node` – Component id for each global node index.
    /// * `n_components` – Number of components.
    ///
    /// Return
    /// ------
    /// * `(component_ref, component_night_bound, local_of_key)`
    ///
    /// Notes
    /// -----
    /// - `local_of_key` is used later to map edge endpoints to component-local indices.
    /// - The order of nodes in each component is the iteration order of `seed_store.iter()`.
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
        // Pre-compute component sizes to reserve capacity in `component_ref`.
        let mut sizes = vec![0u32; n_components];
        for &cid in comp_of_node {
            sizes[cid as usize] += 1;
        }

        let mut component_ref: ComponentRef<'seed_lf, 'alert_lf> = (0..n_components)
            .map(|cid| Vec::with_capacity(sizes[cid] as usize))
            .collect();

        let mut local_of_key: AHashMap<SeedKey, (ComponentId, LocalIdx)> = AHashMap::default();
        local_of_key.reserve(index.n_total());

        // Track per-component min/max nights during materialization.
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
    /// - `component_out[cid][local_u]`: outgoing adjacency inside the component,
    /// - `component_in_deg_local`, `component_out_deg_local`: local degrees,
    /// - `component_sources_local`, `component_sinks_local`: solver entry/exit points.
    ///
    /// Arguments
    /// ---------
    /// * `graph` – Global runtime graph (edges + global degrees in `graph.core`).
    /// * `active_only` – If `true`, include only active edges.
    /// * `n_components` – Number of components.
    /// * `component_ref` – Borrowed node lists per component (for sizing).
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
    /// - Enforces time-forward edges (`night(to) > night(from)`); back-edges are ignored.
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
        // Allocate per-component adjacency and degree arrays in component-local space.
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

        // Best-effort capacity reservation:
        // global out-degree is an upper bound for intra-component out-degree.
        for cid in 0..n_components {
            for (lu, node) in component_ref[cid].iter().enumerate() {
                let key = node.core.key;
                let cap = graph.core.out_deg.get(&key).copied().unwrap_or(0);
                component_out[cid][lu].reserve(cap.min(64));
            }
        }

        // Single pass over global edges:
        // keep only edges fully inside the same component and time-forward.
        for e in graph.edges.iter() {
            if active_only && !e.core.active {
                continue;
            }

            // Enforce time-forward structure.
            let n0 = e.from.core.night_id().value();
            let n1 = e.to.core.night_id().value();
            if n1 <= n0 {
                continue;
            }

            // Fast component check in global index space.
            let u_gid = index.idx_of_key(e.from.core.key);
            let v_gid = index.idx_of_key(e.to.core.key);
            let cu = comp_of_node[u_gid];
            let cv = comp_of_node[v_gid];
            if cu != cv {
                continue;
            }

            // Convert endpoints to component-local indices using `local_of_key`.
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

            // Insert adjacency and update local degrees.
            component_out[cid][lu_usize].push(e);
            out_deg_local[cid][lu_usize] += 1;
            in_deg_local[cid][lv_usize] += 1;
        }

        // Precompute sources and sinks in each restricted digraph.
        //
        // - source: in_deg == 0 and out_deg > 0
        // - sink: out_deg == 0
        //
        // If no sources exist, fall back to "any node with out_deg > 0".
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
    // Accessors
    // -------------------------------------------------------------------------

    /// Return the component id containing a given seed key.
    ///
    /// Arguments
    /// ---------
    /// * `seed_key` – Stable seed identifier.
    ///
    /// Return
    /// ------
    /// Component id containing this seed.
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
    /// Slice of nodes in this component in component-local order.
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

    /// Return the number of active edges whose endpoints are inside the component.
    ///
    /// Arguments
    /// ---------
    /// * `component_id` – Dense component id.
    ///
    /// Return
    /// ------
    /// Active intra-component edge count.
    pub fn component_active_edges(&self, component_id: ComponentId) -> u32 {
        self.component_active_edge[component_id as usize]
    }

    /// Return the `(min_night, max_night)` bounds for the component, if any.
    ///
    /// Arguments
    /// ---------
    /// * `component_id` – Dense component id.
    ///
    /// Return
    /// ------
    /// `Some((min_night, max_night))` if the component is non-empty, else `None`.
    pub fn component_night_bounds(&self, component_id: ComponentId) -> Option<(NightId, NightId)> {
        self.component_night_bound[component_id as usize]
    }

    /// Return the night span (integer day difference) for the component.
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
    /// Slice where `out[local_u]` is the list of outgoing `&Edge` from `local_u`
    /// restricted to the component.
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
    /// Slice where `in_deg[local_u]` is the local in-degree of `local_u`.
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
    /// Slice where `out_deg[local_u]` is the local out-degree of `local_u`.
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
    /// - If no such node exists, this returns a fallback list of nodes with `out_deg > 0`.
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

    /// Classify a component into a solver choice using the given policy.
    ///
    /// This method selects a `SolverChoice` for a given connected component
    /// based on:
    /// - global routing mode (`Force` vs `Heuristics`),
    /// - component size (number of nodes),
    /// - number of active intra-component edges,
    /// - night span,
    /// - estimated min-cost flow runtime vs time budget.
    ///
    /// Decision logic
    /// --------------
    /// The classification proceeds in the following order:
    ///
    /// 0) Global override
    ///    If `solver_policy.routing == Force(choice)`, the provided `choice`
    ///    is returned immediately for all components.
    ///
    /// 1) Tiny component fast path (Trivial solver)
    ///    If both:
    ///    - `n_nodes <= trivial_max_nodes`, and
    ///    - `m_active_edges <= trivial_max_active_edges`,
    ///    the component is routed to `SolverChoice::Trivial`.
    ///
    ///    Rationale:
    ///    Small and sparse components are cheap to solve with a lightweight,
    ///    bounded enumeration strategy.
    ///
    /// 2) Excessive night span (BlobBreaker)
    ///    If:
    ///    - `night_span > max_night_span_for_mcf`,
    ///    the component is routed to `SolverChoice::BlobBreaker`.
    ///
    ///    Rationale:
    ///    Very wide temporal spans tend to induce complex combinatorics.
    ///    Even if node/edge counts are moderate, the structure may be
    ///    unfavorable for global optimization (MCF).
    ///
    /// 3) Budgeted MCF vs BlobBreaker
    ///    Otherwise, estimate the MCF runtime:
    ///      `t_est = k_mcf_s_per_edge_logn * m_active_edges * log2(n_nodes + 1)`
    /// 
    ///     If `t_est <= mcf_budget_s`, route to `SolverChoice::MinCostFlow`,
    ///       else route to `SolverChoice::BlobBreaker`.
    ///
    ///    Rationale:
    ///    Use MCF when the predicted runtime fits within the allowed budget,
    ///    otherwise fall back to a decomposition-based strategy.
    ///
    /// Arguments
    /// ---------
    /// * `component_id` – Dense component id.
    /// * `solver_policy` – Policy containing routing mode, thresholds, and time model.
    ///
    /// Return
    /// ------
    /// `SolverChoice` selected for this component.
    ///
    /// Notes
    /// -----
    /// - The routing is deterministic for fixed component statistics and policy.
    /// - The time model is intentionally simple and should be calibrated empirically.
    /// - This method does not execute any solver; it only selects one.
    pub fn classify(
        &self,
        component_id: ComponentId,
        solver_policy: &SolverPolicy,
    ) -> SolverChoice {
        // 0) Optional global override
        match solver_policy.routing {
            SolverRoutingMode::Force(choice) => return choice,
            SolverRoutingMode::Heuristics => {}
        }

        let n = self.component_size(component_id) as u32;
        let m_active = self.component_active_edges(component_id);

        // 1) Tiny components: trivial fast path (bounded by nodes and active edges)
        if n <= solver_policy.trivial_max_nodes
            && m_active <= solver_policy.trivial_max_active_edges
        {
            return SolverChoice::BoundedBeam;
        }

        // 2) Large night span: avoid MCF
        if self.component_night_span(component_id) > solver_policy.max_night_span_for_mcf {
            return SolverChoice::BlobBreaker;
        }

        // 3) Budgeted MCF vs blob-breaker
        let t_est = solver_policy.estimate_mcf_time_s(n, m_active);
        if t_est <= solver_policy.mcf_budget_s {
            SolverChoice::MinCostFlow
        } else {
            SolverChoice::BlobBreaker
        }
    }
}
