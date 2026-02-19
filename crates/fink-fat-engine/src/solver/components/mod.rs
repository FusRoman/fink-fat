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

pub mod error;
pub mod seed_index;
pub mod union_find;

use ahash::AHashMap;

use crate::{
    engine_config::solver_config::solver_policy::{SolverChoice, SolverPolicy, SolverRoutingMode},
    graph::{AlertLinkageDAG, edge::Edge},
    night_id::NightId,
    seeding::{SeedKey, SeedNode, store::SeedStore},
    solver::components::{
        error::{ComponentError, SeedOrigin},
        seed_index::SeedGlobalIndex,
        union_find::UnionFind,
    },
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
pub type ComponentRef<'seed_lf> = Vec<Vec<&'seed_lf SeedNode>>;

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

/// Reference to a global edge stored in `AlertLinkageDAG.edges`.
///
/// Lifetime
/// --------
/// `'edge_lf` ties this reference to the lifetime of the `AlertLinkageDAG`.
pub type EdgeRef<'edge_lf, 'seed_lf> = &'edge_lf Edge;

/// Per-component directed adjacency (restricted to the component nodes).
///
/// Layout
/// ------
/// `component_out[cid][local_u] = Vec<EdgeRef>` lists outgoing edges from `local_u`
/// that stay inside the same component.
pub type ComponentOut<'edge_lf, 'seed_lf> = Vec<Vec<Vec<EdgeRef<'edge_lf, 'seed_lf>>>>;

/// Local degrees in the restricted directed subgraph.
///
/// Layout
/// ------
/// `component_in_deg_local[cid][local_u]` = in-degree of `local_u` within the restricted subgraph.
/// `component_out_deg_local[cid][local_u]` = out-degree of `local_u` within the restricted subgraph.
pub type ComponentDegLocal = Vec<Vec<u32>>;

/// Precomputed local sources and sinks per component.
type ComponentData<'seed_lf> = (
    ComponentRef<'seed_lf>,
    ComponentNightBound,
    AHashMap<SeedKey, (ComponentId, LocalIdx)>,
);

/// Return type of `build_local_digraph()`.
type ComponentGraphData<'edge_lf, 'seed_lf> = (
    ComponentOut<'edge_lf, 'seed_lf>,
    ComponentDegLocal,
    ComponentDegLocal,
    Vec<Vec<LocalIdx>>,
    Vec<Vec<LocalIdx>>,
);

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
pub struct ConnectedComponents<'edge_lf, 'seed_lf> {
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
    component_ref: ComponentRef<'seed_lf>,

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
    component_out: ComponentOut<'edge_lf, 'seed_lf>,

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

impl<'edge_lf, 'seed_lf, 'alert_lf> ConnectedComponents<'edge_lf, 'seed_lf> {
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
        seed_store: &'seed_lf SeedStore,
        graph: &'edge_lf AlertLinkageDAG,
        active_only: bool,
    ) -> Result<Self, ComponentError> {
        let index = SeedGlobalIndex::build(seed_store)?;
        let n_total = index.n_total();

        // 1) Union-Find over the undirected view of the graph.
        let mut uf = UnionFind::new(n_total);
        Self::union_edges(seed_store, &index, &mut uf, &graph.edges, active_only)?;

        // 2) Assign dense component ids.
        let (comp_of_node, n_components) = Self::dense_components(&mut uf, n_total);

        // 3) Count active intra-component edges for diagnostics and policies.
        let component_active_edge = Self::count_active_intra_edges(
            seed_store,
            &index,
            &graph.edges,
            &comp_of_node,
            n_components,
        )?;

        // 4) Materialize component membership lists and per-component bounds.
        let (component_ref, component_night_bound, local_of_key) =
            Self::build_refs_bounds_and_local_map(seed_store, &index, &comp_of_node, n_components)?;

        // 5) Build the restricted directed subgraph per component.
        let (
            component_out,
            component_in_deg_local,
            component_out_deg_local,
            component_sources_local,
            component_sinks_local,
        ) = Self::build_local_digraph(
            seed_store,
            graph,
            active_only,
            n_components,
            &component_ref,
            &local_of_key,
            &comp_of_node,
            &index,
        )?;

        Ok(Self {
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
        })
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
    fn union_edges(
        seed_store: &'seed_lf SeedStore,
        index: &SeedGlobalIndex,
        uf: &mut UnionFind,
        edges: &[Edge],
        active_only: bool,
    ) -> Result<(), ComponentError> {
        for e in edges {
            if active_only && !e.active {
                continue;
            }
            let u = index.idx_of_key(seed_store, e.from)?;

            let v = index.idx_of_key(seed_store, e.to)?;
            uf.union(u, v);
        }
        Ok(())
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
        seed_store: &'seed_lf SeedStore,
        index: &SeedGlobalIndex,
        edges: &[Edge],
        comp_of_node: &[ComponentId],
        n_components: usize,
    ) -> Result<Vec<u32>, ComponentError> {
        let mut counts = vec![0u32; n_components];

        for e in edges {
            if !e.active {
                continue;
            }
            let u = index.idx_of_key(seed_store, e.from)?;
            let v = index.idx_of_key(seed_store, e.to)?;

            let cu = comp_of_node[u];
            let cv = comp_of_node[v];
            if cu == cv {
                counts[cu as usize] += 1;
            }
        }

        Ok(counts)
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
        seed_store: &'seed_lf SeedStore,
        index: &SeedGlobalIndex,
        comp_of_node: &[ComponentId],
        n_components: usize,
    ) -> Result<ComponentData<'seed_lf>, ComponentError> {
        // Pre-compute component sizes to reserve capacity in `component_ref`.
        let mut sizes = vec![0u32; n_components];
        for &cid in comp_of_node {
            sizes[cid as usize] += 1;
        }

        let mut component_ref: ComponentRef<'seed_lf> = (0..n_components)
            .map(|cid| Vec::with_capacity(sizes[cid] as usize))
            .collect();

        let mut local_of_key: AHashMap<SeedKey, (ComponentId, LocalIdx)> = AHashMap::default();
        local_of_key.reserve(index.n_total());

        // Track per-component min/max nights during materialization.
        let mut min_night: Vec<Option<NightId>> = vec![None; n_components];
        let mut max_night: Vec<Option<NightId>> = vec![None; n_components];

        for (night_id, seeds) in seed_store.iter() {
            for seed in seeds {
                let gid = index.idx_of_key(seed_store, seed.key())?;
                let cid = comp_of_node[gid] as usize;

                let local_idx = component_ref[cid].len() as u32;
                component_ref[cid].push(seed);
                local_of_key.insert(seed.key(), (cid as u32, local_idx));

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

        Ok((component_ref, component_night_bound, local_of_key))
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
        seed_store: &'seed_lf SeedStore,
        graph: &'edge_lf AlertLinkageDAG,
        active_only: bool,
        n_components: usize,
        component_ref: &ComponentRef<'seed_lf>,
        local_of_key: &AHashMap<SeedKey, (ComponentId, LocalIdx)>,
        comp_of_node: &[ComponentId],
        index: &SeedGlobalIndex,
    ) -> Result<ComponentGraphData<'edge_lf, 'seed_lf>, ComponentError> {
        // Allocate per-component adjacency and degree arrays in component-local space.
        let mut component_out: ComponentOut<'edge_lf, 'seed_lf> = (0..n_components)
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
                let key = node.key();
                let cap = graph.out_deg.get(&key).copied().unwrap_or(0);
                component_out[cid][lu].reserve(cap.min(64));
            }
        }

        // Single pass over global edges:
        // keep only edges fully inside the same component and time-forward.
        for e in graph.edges.iter() {
            if active_only && !e.active {
                continue;
            }

            let get_seed = |key| {
                seed_store
                    .try_get_seed(key)
                    .ok_or_else(|| ComponentError::SeedKeyNotFound {
                        key,
                        origin: SeedOrigin::Store,
                    })
            };

            let seed1 = get_seed(e.from)?;
            let seed2 = get_seed(e.to)?;

            // Enforce time-forward structure.
            let n0 = seed1.key().night_id.value();
            let n1 = seed2.key().night_id.value();
            if n1 <= n0 {
                continue;
            }

            // Fast component check in global index space.
            let u_gid = index.idx_of_key(seed_store, e.from)?;
            let v_gid = index.idx_of_key(seed_store, e.to)?;
            let cu = comp_of_node[u_gid];
            let cv = comp_of_node[v_gid];
            if cu != cv {
                continue;
            }

            // Convert endpoints to component-local indices using `local_of_key`.
            let Some(&(cid_u, lu)) = local_of_key.get(&e.from) else {
                continue;
            };
            let Some(&(_, lv)) = local_of_key.get(&e.to) else {
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

        Ok((
            component_out,
            in_deg_local,
            out_deg_local,
            sources_local,
            sinks_local,
        ))
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
    pub fn component_id_of_seed(
        &self,
        seed_store: &SeedStore,
        seed_key: SeedKey,
    ) -> Result<ComponentId, ComponentError> {
        let gid = self.index.idx_of_key(seed_store, seed_key)?;
        Ok(self.comp_of_node[gid])
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
    pub fn component_nodes(&self, component_id: ComponentId) -> &[&SeedNode] {
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
    ) -> &[Vec<EdgeRef<'edge_lf, 'seed_lf>>] {
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

#[cfg(test)]
mod connected_components_tests {
    use super::*;
    use crate::{
        Alert, AlertKey,
        astro_math::arcsec_to_rad,
        engine_config::solver_config::solver_policy::{
            SolverChoice, SolverPolicy, SolverRoutingMode,
        },
        graph::AlertLinkageDAG,
        graph::edge::Edge,
        night_id::NightId,
        seeding::{SeedKey, SeedNode, store::SeedStore},
    };
    use ahash::AHashSet;
    use proptest::prelude::*;

    // =========================================================================
    // Helpers
    // =========================================================================

    fn nid(v: u32) -> NightId {
        NightId::from(v)
    }

    /// Build a minimal `Alert` with given parameters.
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
        }
    }

    /// Insert `count` seeds into `store` for `night_id`, built from alert pairs.
    /// Returns the `SeedKey`s in insertion order.
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
            let dt = 30.0 / 1440.0; // 30 min
            let alert_a = mk_alert(sid_a, night_id, t0 + i as f64 * 0.01);
            let alert_b = mk_alert(sid_b, night_id, t0 + i as f64 * 0.01 + dt);

            if let Some(seed) = SeedNode::from_pair(store, night_id, &alert_a, &alert_b, None) {
                keys.push(seed.key());
                store.insert_vec_seed(night_id, vec![seed]);
            }
        }
        keys
    }

    /// Build a store + record from a spec: `(night_id_u32, seed_count)`.
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

    /// Create a directed edge between two seed keys.
    fn mk_edge(from: SeedKey, to: SeedKey, cost: f64, active: bool) -> Edge {
        Edge {
            from,
            to,
            cost,
            dt_days: 1.0,
            active,
        }
    }

    /// Build an `AlertLinkageDAG` from a list of edges.
    fn build_graph(edges: Vec<Edge>) -> AlertLinkageDAG {
        AlertLinkageDAG::from_edges(edges)
    }

    // =========================================================================
    // Unit tests — empty / trivial cases
    // =========================================================================

    #[test]
    fn empty_store_empty_graph_gives_zero_components() {
        let store = SeedStore::new();
        let graph = build_graph(vec![]);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        assert_eq!(cc.n_components, 0);
    }

    #[test]
    fn single_seed_no_edges_produces_one_singleton_component() {
        let (store, record) = build_store(&[(10, 1)]);
        let graph = build_graph(vec![]);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        assert_eq!(cc.n_components, 1);
        assert_eq!(cc.component_size(0), 1);
        assert_eq!(cc.component_active_edges(0), 0);

        // Night bounds should be the single night.
        let bounds = cc.component_night_bounds(0).unwrap();
        assert_eq!(bounds, (nid(10), nid(10)));

        // Night span is 0 for a single-night component.
        assert_eq!(cc.component_night_span(0), 0);

        // Sources and sinks: no edges, so no sources; all nodes are sinks.
        assert!(cc.component_sources_local(0).is_empty());
        assert_eq!(cc.component_sinks_local(0).len(), 1);

        // Accessor: component_id_of_seed
        let key = record[0].1[0];
        assert_eq!(cc.component_id_of_seed(&store, key).unwrap(), 0);
    }

    #[test]
    fn isolated_seeds_across_nights_produce_singleton_components() {
        let (store, _) = build_store(&[(10, 1), (11, 1), (12, 1)]);
        let graph = build_graph(vec![]);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        assert_eq!(cc.n_components, 3);
        for cid in 0..3 {
            assert_eq!(cc.component_size(cid), 1);
        }
    }

    // =========================================================================
    // Unit tests — linear chain (N0 -> N1 -> N2)
    // =========================================================================

    #[test]
    fn linear_chain_forms_single_component() {
        let (store, record) = build_store(&[(10, 1), (11, 1), (12, 1)]);
        let k0 = record[0].1[0];
        let k1 = record[1].1[0];
        let k2 = record[2].1[0];

        let edges = vec![mk_edge(k0, k1, 1.0, true), mk_edge(k1, k2, 1.0, true)];
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        assert_eq!(cc.n_components, 1);
        assert_eq!(cc.component_size(0), 3);
        assert_eq!(cc.component_active_edges(0), 2);
        assert_eq!(cc.component_night_span(0), 2); // 12 - 10

        // The node on night 10 should be a source (in_deg=0, out_deg>0).
        let sources = cc.component_sources_local(0);
        assert!(!sources.is_empty());

        // The node on night 12 should be a sink (out_deg=0).
        let sinks = cc.component_sinks_local(0);
        assert!(!sinks.is_empty());
    }

    // =========================================================================
    // Unit tests — two disjoint components
    // =========================================================================

    #[test]
    fn two_disjoint_chains_form_two_components() {
        // Component A: night 10 -> night 11
        // Component B: night 20 -> night 21
        let (store, record) = build_store(&[(10, 1), (11, 1), (20, 1), (21, 1)]);
        let ka0 = record[0].1[0]; // night 10
        let ka1 = record[1].1[0]; // night 11
        let kb0 = record[2].1[0]; // night 20
        let kb1 = record[3].1[0]; // night 21

        let edges = vec![mk_edge(ka0, ka1, 1.0, true), mk_edge(kb0, kb1, 2.0, true)];
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        assert_eq!(cc.n_components, 2);

        // Each component should have 2 nodes.
        let sizes: AHashSet<usize> = (0..2).map(|cid| cc.component_size(cid)).collect();
        assert_eq!(sizes, AHashSet::from_iter([2]));

        // Both endpoints of edge A should be in the same component.
        let ca0 = cc.component_id_of_seed(&store, ka0).unwrap();
        let ca1 = cc.component_id_of_seed(&store, ka1).unwrap();
        assert_eq!(ca0, ca1);

        // Both endpoints of edge B should be in the same component.
        let cb0 = cc.component_id_of_seed(&store, kb0).unwrap();
        let cb1 = cc.component_id_of_seed(&store, kb1).unwrap();
        assert_eq!(cb0, cb1);

        // The two chains should be in different components.
        assert_ne!(ca0, cb0);
    }

    // =========================================================================
    // Unit tests — active_only filtering
    // =========================================================================

    #[test]
    fn active_only_ignores_inactive_edges_for_connectivity() {
        let (store, record) = build_store(&[(10, 1), (11, 1), (12, 1)]);
        let k0 = record[0].1[0];
        let k1 = record[1].1[0];
        let k2 = record[2].1[0];

        // k0 -> k1 is active, k1 -> k2 is INACTIVE
        let edges = vec![mk_edge(k0, k1, 1.0, true), mk_edge(k1, k2, 1.0, false)];
        let graph = build_graph(edges);

        // Without active_only: single component (all edges used for connectivity).
        let cc_all = ConnectedComponents::compute(&store, &graph, false).unwrap();
        assert_eq!(cc_all.n_components, 1);

        // With active_only: the inactive edge is ignored, so k2 is isolated.
        let cc_active = ConnectedComponents::compute(&store, &graph, true).unwrap();
        assert_eq!(cc_active.n_components, 2);
    }

    #[test]
    fn active_only_excludes_inactive_edges_from_adjacency() {
        let (store, record) = build_store(&[(10, 2), (11, 1)]);
        let k0a = record[0].1[0]; // night 10, seed 0
        let k0b = record[0].1[1]; // night 10, seed 1
        let k1 = record[1].1[0]; // night 11

        // Both edges point forward, but only first is active.
        let edges = vec![
            mk_edge(k0a, k1, 1.0, true),
            mk_edge(k0b, k1, 1.0, false),
        ];
        let graph = build_graph(edges);

        let cc = ConnectedComponents::compute(&store, &graph, true).unwrap();

        // k0b is disconnected (active_only), so we expect 2 components.
        assert_eq!(cc.n_components, 2);

        // Find the component with k0a and k1.
        let cid = cc.component_id_of_seed(&store, k0a).unwrap();
        assert_eq!(cc.component_id_of_seed(&store, k1).unwrap(), cid);

        // In that component, the adjacency should have exactly 1 edge.
        let out = cc.component_out_edges(cid);
        let total_edges: usize = out.iter().map(|adj| adj.len()).sum();
        assert_eq!(total_edges, 1);
    }

    // =========================================================================
    // Unit tests — directed adjacency and degree invariants
    // =========================================================================

    #[test]
    fn adjacency_degrees_match_edge_count() {
        //   night 10: [s0, s1]
        //   night 11: [s2]
        //   edges: s0 -> s2, s1 -> s2  (both time-forward and active)
        let (store, record) = build_store(&[(10, 2), (11, 1)]);
        let s0 = record[0].1[0];
        let s1 = record[0].1[1];
        let s2 = record[1].1[0];

        let edges = vec![mk_edge(s0, s2, 1.0, true), mk_edge(s1, s2, 2.0, true)];
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        assert_eq!(cc.n_components, 1);
        let cid: ComponentId = 0;
        let n = cc.component_size(cid);
        assert_eq!(n, 3);

        let in_deg = cc.component_in_deg_local(cid);
        let out_deg = cc.component_out_deg_local(cid);

        // Sum of in-degrees == sum of out-degrees == number of directed edges.
        let sum_in: u32 = in_deg.iter().sum();
        let sum_out: u32 = out_deg.iter().sum();
        assert_eq!(sum_in, sum_out);
        assert_eq!(sum_in, 2); // 2 time-forward edges

        // component_out adjacency total edge count must match.
        let out_edges = cc.component_out_edges(cid);
        let adj_total: usize = out_edges.iter().map(|adj| adj.len()).sum();
        assert_eq!(adj_total, 2);
    }

    #[test]
    fn back_edges_are_excluded_from_adjacency() {
        // night 11 -> night 10 is a back-edge (not time-forward).
        let (store, record) = build_store(&[(10, 1), (11, 1)]);
        let k10 = record[0].1[0];
        let k11 = record[1].1[0];

        // Back-edge: night 11 to night 10 (wrong direction).
        let edges = vec![Edge {
            from: k11,
            to: k10,
            cost: 1.0,
            dt_days: 1.0,
            active: true,
        }];
        let graph = build_graph(edges);

        // active_only=false: back-edge still participates in connectivity
        // (Union-Find is undirected) but not in the directed adjacency.
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();
        assert_eq!(cc.n_components, 1);
        assert_eq!(cc.component_size(0), 2);

        // The directed adjacency should have zero edges (back-edge filtered out).
        let out = cc.component_out_edges(0);
        let total: usize = out.iter().map(|adj| adj.len()).sum();
        assert_eq!(total, 0);

        // Since there are no directed edges, all nodes are sinks.
        assert_eq!(cc.component_sinks_local(0).len(), 2);
    }

    // =========================================================================
    // Unit tests — sources and sinks definitions
    // =========================================================================

    #[test]
    fn sources_have_zero_in_degree_and_positive_out_degree() {
        // Chain: s0 -> s1 -> s2
        let (store, record) = build_store(&[(10, 1), (11, 1), (12, 1)]);
        let k0 = record[0].1[0];
        let k1 = record[1].1[0];
        let k2 = record[2].1[0];

        let edges = vec![mk_edge(k0, k1, 1.0, true), mk_edge(k1, k2, 1.5, true)];
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        let cid: ComponentId = 0;
        let sources = cc.component_sources_local(cid);
        let in_deg = cc.component_in_deg_local(cid);
        let out_deg = cc.component_out_deg_local(cid);

        for &src in sources {
            assert_eq!(in_deg[src as usize], 0, "source must have in_deg == 0");
            assert!(
                out_deg[src as usize] > 0,
                "source must have out_deg > 0"
            );
        }
    }

    #[test]
    fn sinks_have_zero_out_degree() {
        let (store, record) = build_store(&[(10, 1), (11, 1), (12, 1)]);
        let k0 = record[0].1[0];
        let k1 = record[1].1[0];
        let k2 = record[2].1[0];

        let edges = vec![mk_edge(k0, k1, 1.0, true), mk_edge(k1, k2, 1.5, true)];
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        let cid: ComponentId = 0;
        let sinks = cc.component_sinks_local(cid);
        let out_deg = cc.component_out_deg_local(cid);

        for &snk in sinks {
            assert_eq!(out_deg[snk as usize], 0, "sink must have out_deg == 0");
        }
    }

    // =========================================================================
    // Unit tests — night bounds
    // =========================================================================

    #[test]
    fn night_bounds_reflect_min_max_nights_in_component() {
        let (store, record) = build_store(&[(5, 1), (10, 1), (20, 1)]);
        let k5 = record[0].1[0];
        let k10 = record[1].1[0];
        let k20 = record[2].1[0];

        // All connected: single component spanning nights 5..20.
        let edges = vec![mk_edge(k5, k10, 1.0, true), mk_edge(k10, k20, 1.0, true)];
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        assert_eq!(cc.n_components, 1);
        let bounds = cc.component_night_bounds(0).unwrap();
        assert_eq!(bounds.0, nid(5));
        assert_eq!(bounds.1, nid(20));
        assert_eq!(cc.component_night_span(0), 15);
    }

    // =========================================================================
    // Unit tests — classify (solver routing)
    // =========================================================================

    #[test]
    fn classify_force_mode_returns_forced_choice() {
        let (store, record) = build_store(&[(10, 2), (11, 2)]);
        let k0 = record[0].1[0];
        let k1 = record[1].1[0];

        let edges = vec![mk_edge(k0, k1, 1.0, true)];
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        let policy = SolverPolicy {
            routing: SolverRoutingMode::Force(SolverChoice::MinCostFlow),
            ..Default::default()
        };

        for cid in 0..cc.n_components {
            assert_eq!(cc.classify(cid, &policy), SolverChoice::MinCostFlow);
        }
    }

    #[test]
    fn classify_tiny_component_returns_bounded_beam() {
        let (store, record) = build_store(&[(10, 1), (11, 1)]);
        let k0 = record[0].1[0];
        let k1 = record[1].1[0];

        let edges = vec![mk_edge(k0, k1, 1.0, true)];
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        let policy = SolverPolicy {
            routing: SolverRoutingMode::Heuristics,
            trivial_max_nodes: 10,
            trivial_max_active_edges: 20,
            ..Default::default()
        };

        let cid = cc.component_id_of_seed(&store, k0).unwrap();
        assert_eq!(cc.classify(cid, &policy), SolverChoice::BoundedBeam);
    }

    #[test]
    fn classify_large_night_span_returns_blob_breaker() {
        // Component spanning many nights.
        let (store, record) = build_store(&[(1, 1), (100, 1)]);
        let k0 = record[0].1[0];
        let k1 = record[1].1[0];

        let edges = vec![mk_edge(k0, k1, 1.0, true)];
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        let policy = SolverPolicy {
            routing: SolverRoutingMode::Heuristics,
            trivial_max_nodes: 1,        // too small to be trivial
            trivial_max_active_edges: 0,  // too small to be trivial
            max_night_span_for_mcf: 4,
            ..Default::default()
        };

        let cid = cc.component_id_of_seed(&store, k0).unwrap();
        assert_eq!(cc.classify(cid, &policy), SolverChoice::BlobBreaker);
    }

    #[test]
    fn classify_within_mcf_budget_returns_min_cost_flow() {
        let (store, record) = build_store(&[(10, 3), (11, 3)]);
        let k0 = record[0].1[0];
        let k1 = record[0].1[1];
        let k2 = record[0].1[2];
        let k3 = record[1].1[0];
        let k4 = record[1].1[1];
        let k5 = record[1].1[2];

        // Fully connected bipartite: 9 edges.
        let mut edges = Vec::new();
        for &from in &[k0, k1, k2] {
            for &to in &[k3, k4, k5] {
                edges.push(mk_edge(from, to, 1.0, true));
            }
        }
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        let policy = SolverPolicy {
            routing: SolverRoutingMode::Heuristics,
            trivial_max_nodes: 2,         // 6 nodes > 2, not trivial
            trivial_max_active_edges: 2,  // 9 edges > 2
            max_night_span_for_mcf: 10,
            mcf_budget_s: 1000.0,         // generous budget
            k_mcf_s_per_edge_logn: 1e-8,
            ..Default::default()
        };

        let cid = cc.component_id_of_seed(&store, k0).unwrap();
        assert_eq!(cc.classify(cid, &policy), SolverChoice::MinCostFlow);
    }

    // =========================================================================
    // Unit tests — component_active_edges counts only active intra-component
    // =========================================================================

    #[test]
    fn component_active_edges_counts_only_active() {
        let (store, record) = build_store(&[(10, 2), (11, 2)]);
        let s0 = record[0].1[0];
        let s1 = record[0].1[1];
        let s2 = record[1].1[0];
        let s3 = record[1].1[1];

        // 3 active edges + 1 inactive edge
        let edges = vec![
            mk_edge(s0, s2, 1.0, true),
            mk_edge(s0, s3, 1.0, true),
            mk_edge(s1, s2, 1.0, true),
            mk_edge(s1, s3, 1.0, false), // inactive
        ];
        let graph = build_graph(edges);

        // active_only=false: all 4 edges join, single component.
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();
        assert_eq!(cc.n_components, 1);
        // component_active_edges always counts active only (independent of active_only flag).
        assert_eq!(cc.component_active_edges(0), 3);
    }

    // =========================================================================
    // Unit tests — multiple seeds per night, diamond topology
    // =========================================================================

    #[test]
    fn diamond_topology_degrees_are_correct() {
        //     s0 (night 10)
        //    /    \
        //  s1      s2 (night 11)
        //    \    /
        //     s3 (night 12)
        let (store, record) = build_store(&[(10, 1), (11, 2), (12, 1)]);
        let s0 = record[0].1[0]; // night 10
        let s1 = record[1].1[0]; // night 11
        let s2 = record[1].1[1]; // night 11
        let s3 = record[2].1[0]; // night 12

        let edges = vec![
            mk_edge(s0, s1, 1.0, true),
            mk_edge(s0, s2, 1.0, true),
            mk_edge(s1, s3, 1.0, true),
            mk_edge(s2, s3, 1.0, true),
        ];
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        assert_eq!(cc.n_components, 1);
        let cid: ComponentId = 0;
        assert_eq!(cc.component_size(cid), 4);
        assert_eq!(cc.component_active_edges(cid), 4);

        // Sum of in_deg == sum of out_deg == 4.
        let in_deg = cc.component_in_deg_local(cid);
        let out_deg = cc.component_out_deg_local(cid);
        let sum_in: u32 = in_deg.iter().sum();
        let sum_out: u32 = out_deg.iter().sum();
        assert_eq!(sum_in, 4);
        assert_eq!(sum_out, 4);

        // There should be exactly 1 source (s0) and 1 sink (s3).
        assert_eq!(cc.component_sources_local(cid).len(), 1);
        assert_eq!(cc.component_sinks_local(cid).len(), 1);
    }

    // =========================================================================
    // Unit tests — component_nodes returns correct nodes
    // =========================================================================

    #[test]
    fn component_nodes_returns_all_seeds_in_component() {
        let (store, record) = build_store(&[(10, 2), (11, 1)]);
        let s0 = record[0].1[0];
        let s1 = record[0].1[1];
        let s2 = record[1].1[0];

        let edges = vec![mk_edge(s0, s2, 1.0, true)];
        let graph = build_graph(edges);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        // s0 and s2 are connected; s1 is isolated.
        let cid_s0 = cc.component_id_of_seed(&store, s0).unwrap();
        let cid_s1 = cc.component_id_of_seed(&store, s1).unwrap();
        assert_ne!(cid_s0, cid_s1);

        let nodes_s0: AHashSet<SeedKey> = cc
            .component_nodes(cid_s0)
            .iter()
            .map(|n| n.key())
            .collect();
        assert!(nodes_s0.contains(&s0));
        assert!(nodes_s0.contains(&s2));
        assert_eq!(nodes_s0.len(), 2);
    }

    // =========================================================================
    // Unit tests — component_id_of_seed error on unknown key
    // =========================================================================

    #[test]
    fn component_id_of_seed_unknown_key_returns_error() {
        let (store, _) = build_store(&[(10, 1)]);
        let graph = build_graph(vec![]);
        let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

        let bad_key = SeedKey {
            night_id: nid(999),
            unique_id: 12345,
        };
        assert!(cc.component_id_of_seed(&store, bad_key).is_err());
    }

    // =========================================================================
    // Property-based tests (proptest)
    // =========================================================================

    /// Strategy: generate a graph with `n_nights` nights, `seeds_per_night` seeds each,
    /// and random edges between consecutive nights.
    fn arb_graph_spec() -> impl Strategy<Value = (Vec<(u32, usize)>, Vec<(usize, usize, usize, usize, bool)>)>
    {
        // 2..6 nights, 1..5 seeds each.
        let nights = prop::collection::vec((1u32..50, 1usize..5), 2..6);

        nights.prop_flat_map(|night_spec| {
            let spec = night_spec.clone();
            let n_nights = spec.len();
            // Generate up to 15 random edge descriptors between consecutive night pairs.
            let edges = prop::collection::vec(
                (
                    0usize..n_nights.saturating_sub(1), // night pair index
                    0usize..5,                           // from seed idx (clamped later)
                    0usize..5,                           // to seed idx (clamped later)
                    0usize..5,                           // dummy (unused in uniform case)
                    any::<bool>(),                       // active
                ),
                0..15,
            );
            (Just(spec), edges)
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        /// Every seed key must map to exactly one component id.
        #[test]
        fn prop_every_seed_belongs_to_exactly_one_component(
            (spec, edge_descs) in arb_graph_spec()
        ) {
            let (store, record) = build_store(&spec);

            // Build edges from descriptors.
            let mut edges = Vec::new();
            for (pair_idx, from_idx, to_idx, _, active) in &edge_descs {
                let pair_idx = *pair_idx;
                if pair_idx + 1 >= record.len() { continue; }

                let (_, ref from_keys) = record[pair_idx];
                let (_, ref to_keys) = record[pair_idx + 1];
                if from_keys.is_empty() || to_keys.is_empty() { continue; }

                let from = from_keys[from_idx % from_keys.len()];
                let to = to_keys[to_idx % to_keys.len()];
                edges.push(mk_edge(from, to, 1.0, *active));
            }
            let graph = build_graph(edges);
            let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

            // Every key must be assigned a valid component id.
            for (_, keys) in &record {
                for &key in keys {
                    let cid = cc.component_id_of_seed(&store, key).unwrap();
                    prop_assert!(cid < cc.n_components);
                }
            }
        }

        /// The total number of nodes across all components equals the total seed count.
        #[test]
        fn prop_total_nodes_across_components_equals_store_size(
            (spec, edge_descs) in arb_graph_spec()
        ) {
            let (store, record) = build_store(&spec);
            let total_seeds: usize = record.iter().map(|(_, keys)| keys.len()).sum();

            let mut edges = Vec::new();
            for (pair_idx, from_idx, to_idx, _, active) in &edge_descs {
                let pair_idx = *pair_idx;
                if pair_idx + 1 >= record.len() { continue; }
                let (_, ref from_keys) = record[pair_idx];
                let (_, ref to_keys) = record[pair_idx + 1];
                if from_keys.is_empty() || to_keys.is_empty() { continue; }
                let from = from_keys[from_idx % from_keys.len()];
                let to = to_keys[to_idx % to_keys.len()];
                edges.push(mk_edge(from, to, 1.0, *active));
            }
            let graph = build_graph(edges);
            let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

            let total_in_components: usize = (0..cc.n_components)
                .map(|cid| cc.component_size(cid))
                .sum();
            prop_assert_eq!(total_in_components, total_seeds);
        }

        /// Component ids are dense: `0..n_components`.
        #[test]
        fn prop_component_ids_are_dense(
            (spec, edge_descs) in arb_graph_spec()
        ) {
            let (store, record) = build_store(&spec);

            let mut edges = Vec::new();
            for (pair_idx, from_idx, to_idx, _, active) in &edge_descs {
                let pair_idx = *pair_idx;
                if pair_idx + 1 >= record.len() { continue; }
                let (_, ref from_keys) = record[pair_idx];
                let (_, ref to_keys) = record[pair_idx + 1];
                if from_keys.is_empty() || to_keys.is_empty() { continue; }
                let from = from_keys[from_idx % from_keys.len()];
                let to = to_keys[to_idx % to_keys.len()];
                edges.push(mk_edge(from, to, 1.0, *active));
            }
            let graph = build_graph(edges);
            let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

            let mut seen_ids: AHashSet<ComponentId> = AHashSet::default();
            for (_, keys) in &record {
                for &key in keys {
                    let cid = cc.component_id_of_seed(&store, key).unwrap();
                    seen_ids.insert(cid);
                }
            }
            prop_assert_eq!(seen_ids.len(), cc.n_components as usize);
            for cid in 0..cc.n_components {
                prop_assert!(seen_ids.contains(&cid));
            }
        }

        /// Sum of local in-degrees == sum of local out-degrees in each component.
        #[test]
        fn prop_in_degree_sum_equals_out_degree_sum_per_component(
            (spec, edge_descs) in arb_graph_spec()
        ) {
            let (store, record) = build_store(&spec);

            let mut edges = Vec::new();
            for (pair_idx, from_idx, to_idx, _, active) in &edge_descs {
                let pair_idx = *pair_idx;
                if pair_idx + 1 >= record.len() { continue; }
                let (_, ref from_keys) = record[pair_idx];
                let (_, ref to_keys) = record[pair_idx + 1];
                if from_keys.is_empty() || to_keys.is_empty() { continue; }
                let from = from_keys[from_idx % from_keys.len()];
                let to = to_keys[to_idx % to_keys.len()];
                edges.push(mk_edge(from, to, 1.0, *active));
            }
            let graph = build_graph(edges);
            let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

            for cid in 0..cc.n_components {
                let in_deg = cc.component_in_deg_local(cid);
                let out_deg = cc.component_out_deg_local(cid);
                let sum_in: u32 = in_deg.iter().sum();
                let sum_out: u32 = out_deg.iter().sum();
                prop_assert_eq!(
                    sum_in, sum_out,
                    "component {}: sum_in={} != sum_out={}", cid, sum_in, sum_out
                );
            }
        }

        /// Out-degree of local node matches number of adjacency entries.
        #[test]
        fn prop_out_degree_matches_adjacency_len(
            (spec, edge_descs) in arb_graph_spec()
        ) {
            let (store, record) = build_store(&spec);

            let mut edges = Vec::new();
            for (pair_idx, from_idx, to_idx, _, active) in &edge_descs {
                let pair_idx = *pair_idx;
                if pair_idx + 1 >= record.len() { continue; }
                let (_, ref from_keys) = record[pair_idx];
                let (_, ref to_keys) = record[pair_idx + 1];
                if from_keys.is_empty() || to_keys.is_empty() { continue; }
                let from = from_keys[from_idx % from_keys.len()];
                let to = to_keys[to_idx % to_keys.len()];
                edges.push(mk_edge(from, to, 1.0, *active));
            }
            let graph = build_graph(edges);
            let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

            for cid in 0..cc.n_components {
                let out_deg = cc.component_out_deg_local(cid);
                let out_edges = cc.component_out_edges(cid);
                for (lu, adj) in out_edges.iter().enumerate() {
                    prop_assert_eq!(
                        adj.len() as u32,
                        out_deg[lu],
                        "component {}, local node {}: adj.len()={} != out_deg={}",
                        cid, lu, adj.len(), out_deg[lu]
                    );
                }
            }
        }

        /// All sinks have out-degree 0.
        #[test]
        fn prop_sinks_have_zero_out_degree(
            (spec, edge_descs) in arb_graph_spec()
        ) {
            let (store, record) = build_store(&spec);

            let mut edges = Vec::new();
            for (pair_idx, from_idx, to_idx, _, active) in &edge_descs {
                let pair_idx = *pair_idx;
                if pair_idx + 1 >= record.len() { continue; }
                let (_, ref from_keys) = record[pair_idx];
                let (_, ref to_keys) = record[pair_idx + 1];
                if from_keys.is_empty() || to_keys.is_empty() { continue; }
                let from = from_keys[from_idx % from_keys.len()];
                let to = to_keys[to_idx % to_keys.len()];
                edges.push(mk_edge(from, to, 1.0, *active));
            }
            let graph = build_graph(edges);
            let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

            for cid in 0..cc.n_components {
                let out_deg = cc.component_out_deg_local(cid);
                for &snk in cc.component_sinks_local(cid) {
                    prop_assert_eq!(
                        out_deg[snk as usize], 0,
                        "component {}: sink local_idx={} has out_deg={}",
                        cid, snk, out_deg[snk as usize]
                    );
                }
            }
        }

        /// Sources either have (in_deg==0 && out_deg>0) or the fallback applies.
        #[test]
        fn prop_sources_invariant(
            (spec, edge_descs) in arb_graph_spec()
        ) {
            let (store, record) = build_store(&spec);

            let mut edges = Vec::new();
            for (pair_idx, from_idx, to_idx, _, active) in &edge_descs {
                let pair_idx = *pair_idx;
                if pair_idx + 1 >= record.len() { continue; }
                let (_, ref from_keys) = record[pair_idx];
                let (_, ref to_keys) = record[pair_idx + 1];
                if from_keys.is_empty() || to_keys.is_empty() { continue; }
                let from = from_keys[from_idx % from_keys.len()];
                let to = to_keys[to_idx % to_keys.len()];
                edges.push(mk_edge(from, to, 1.0, *active));
            }
            let graph = build_graph(edges);
            let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

            for cid in 0..cc.n_components {
                let in_deg = cc.component_in_deg_local(cid);
                let out_deg = cc.component_out_deg_local(cid);
                let sources = cc.component_sources_local(cid);

                // Check if there exist any "true sources" (in_deg==0 && out_deg>0).
                let n = cc.component_size(cid);
                let true_sources: Vec<u32> = (0..n as u32)
                    .filter(|&u| in_deg[u as usize] == 0 && out_deg[u as usize] > 0)
                    .collect();

                if !true_sources.is_empty() {
                    // Normal case: sources must exactly match true sources.
                    let source_set: AHashSet<u32> = sources.iter().copied().collect();
                    let true_set: AHashSet<u32> = true_sources.iter().copied().collect();
                    prop_assert_eq!(
                        source_set, true_set,
                        "component {}: source sets differ", cid
                    );
                } else {
                    // Fallback: sources are all nodes with out_deg > 0.
                    let fallback: Vec<u32> = (0..n as u32)
                        .filter(|&u| out_deg[u as usize] > 0)
                        .collect();
                    let source_set: AHashSet<u32> = sources.iter().copied().collect();
                    let fallback_set: AHashSet<u32> = fallback.iter().copied().collect();
                    prop_assert_eq!(
                        source_set, fallback_set,
                        "component {}: fallback source sets differ", cid
                    );
                }
            }
        }

        /// Night bounds are consistent with the nodes in the component.
        #[test]
        fn prop_night_bounds_are_consistent(
            (spec, edge_descs) in arb_graph_spec()
        ) {
            let (store, record) = build_store(&spec);

            let mut edges = Vec::new();
            for (pair_idx, from_idx, to_idx, _, active) in &edge_descs {
                let pair_idx = *pair_idx;
                if pair_idx + 1 >= record.len() { continue; }
                let (_, ref from_keys) = record[pair_idx];
                let (_, ref to_keys) = record[pair_idx + 1];
                if from_keys.is_empty() || to_keys.is_empty() { continue; }
                let from = from_keys[from_idx % from_keys.len()];
                let to = to_keys[to_idx % to_keys.len()];
                edges.push(mk_edge(from, to, 1.0, *active));
            }
            let graph = build_graph(edges);
            let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

            for cid in 0..cc.n_components {
                let nodes = cc.component_nodes(cid);
                if nodes.is_empty() { continue; }

                let actual_min = nodes.iter().map(|n| n.night_id()).min().unwrap();
                let actual_max = nodes.iter().map(|n| n.night_id()).max().unwrap();

                let bounds = cc.component_night_bounds(cid);
                prop_assert!(bounds.is_some(), "non-empty component {} has None bounds", cid);
                let (bmin, bmax) = bounds.unwrap();
                prop_assert_eq!(bmin, actual_min, "component {}: min night mismatch", cid);
                prop_assert_eq!(bmax, actual_max, "component {}: max night mismatch", cid);
            }
        }

        /// Edges in the adjacency only connect nodes within the same component.
        #[test]
        fn prop_adjacency_edges_are_intra_component(
            (spec, edge_descs) in arb_graph_spec()
        ) {
            let (store, record) = build_store(&spec);

            let mut edges = Vec::new();
            for (pair_idx, from_idx, to_idx, _, active) in &edge_descs {
                let pair_idx = *pair_idx;
                if pair_idx + 1 >= record.len() { continue; }
                let (_, ref from_keys) = record[pair_idx];
                let (_, ref to_keys) = record[pair_idx + 1];
                if from_keys.is_empty() || to_keys.is_empty() { continue; }
                let from = from_keys[from_idx % from_keys.len()];
                let to = to_keys[to_idx % to_keys.len()];
                edges.push(mk_edge(from, to, 1.0, *active));
            }
            let graph = build_graph(edges);
            let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

            for cid in 0..cc.n_components {
                let nodes = cc.component_nodes(cid);
                let node_keys: AHashSet<SeedKey> = nodes.iter().map(|n| n.key()).collect();
                let out = cc.component_out_edges(cid);

                for adj in out {
                    for edge in adj {
                        prop_assert!(
                            node_keys.contains(&edge.from),
                            "component {}: edge.from {} not in component",
                            cid, edge.from
                        );
                        prop_assert!(
                            node_keys.contains(&edge.to),
                            "component {}: edge.to {} not in component",
                            cid, edge.to
                        );
                    }
                }
            }
        }

        /// Edges in the directed adjacency are time-forward: night(to) > night(from).
        #[test]
        fn prop_adjacency_edges_are_time_forward(
            (spec, edge_descs) in arb_graph_spec()
        ) {
            let (store, record) = build_store(&spec);

            let mut edges = Vec::new();
            for (pair_idx, from_idx, to_idx, _, active) in &edge_descs {
                let pair_idx = *pair_idx;
                if pair_idx + 1 >= record.len() { continue; }
                let (_, ref from_keys) = record[pair_idx];
                let (_, ref to_keys) = record[pair_idx + 1];
                if from_keys.is_empty() || to_keys.is_empty() { continue; }
                let from = from_keys[from_idx % from_keys.len()];
                let to = to_keys[to_idx % to_keys.len()];
                edges.push(mk_edge(from, to, 1.0, *active));
            }
            let graph = build_graph(edges);
            let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

            for cid in 0..cc.n_components {
                let out = cc.component_out_edges(cid);
                for adj in out {
                    for edge in adj {
                        prop_assert!(
                            edge.to.night_id > edge.from.night_id,
                            "component {}: edge not time-forward: from_nid={} to_nid={}",
                            cid, edge.from.night_id.value(), edge.to.night_id.value()
                        );
                    }
                }
            }
        }

        /// active_only=true produces at least as many components as active_only=false
        /// (more restrictive connectivity → more or equal components).
        #[test]
        fn prop_active_only_produces_at_least_as_many_components(
            (spec, edge_descs) in arb_graph_spec()
        ) {
            let (store, record) = build_store(&spec);

            let mut edges = Vec::new();
            for (pair_idx, from_idx, to_idx, _, active) in &edge_descs {
                let pair_idx = *pair_idx;
                if pair_idx + 1 >= record.len() { continue; }
                let (_, ref from_keys) = record[pair_idx];
                let (_, ref to_keys) = record[pair_idx + 1];
                if from_keys.is_empty() || to_keys.is_empty() { continue; }
                let from = from_keys[from_idx % from_keys.len()];
                let to = to_keys[to_idx % to_keys.len()];
                edges.push(mk_edge(from, to, 1.0, *active));
            }
            let graph = build_graph(edges);

            let cc_all = ConnectedComponents::compute(&store, &graph, false).unwrap();
            let cc_active = ConnectedComponents::compute(&store, &graph, true).unwrap();

            prop_assert!(
                cc_active.n_components >= cc_all.n_components,
                "active_only should produce >= components: {} < {}",
                cc_active.n_components,
                cc_all.n_components
            );
        }

        /// classify with Force routing always returns the forced choice.
        #[test]
        fn prop_classify_force_returns_forced(
            (spec, edge_descs) in arb_graph_spec(),
            choice in prop_oneof![
                Just(SolverChoice::BoundedBeam),
                Just(SolverChoice::MinCostFlow),
                Just(SolverChoice::BlobBreaker),
            ]
        ) {
            let (store, record) = build_store(&spec);

            let mut edges = Vec::new();
            for (pair_idx, from_idx, to_idx, _, active) in &edge_descs {
                let pair_idx = *pair_idx;
                if pair_idx + 1 >= record.len() { continue; }
                let (_, ref from_keys) = record[pair_idx];
                let (_, ref to_keys) = record[pair_idx + 1];
                if from_keys.is_empty() || to_keys.is_empty() { continue; }
                let from = from_keys[from_idx % from_keys.len()];
                let to = to_keys[to_idx % to_keys.len()];
                edges.push(mk_edge(from, to, 1.0, *active));
            }
            let graph = build_graph(edges);
            let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

            let policy = SolverPolicy {
                routing: SolverRoutingMode::Force(choice),
                ..Default::default()
            };

            for cid in 0..cc.n_components {
                prop_assert_eq!(cc.classify(cid, &policy), choice);
            }
        }

        /// component_night_span == max_night - min_night for each component.
        #[test]
        fn prop_night_span_matches_bounds(
            (spec, edge_descs) in arb_graph_spec()
        ) {
            let (store, record) = build_store(&spec);

            let mut edges = Vec::new();
            for (pair_idx, from_idx, to_idx, _, active) in &edge_descs {
                let pair_idx = *pair_idx;
                if pair_idx + 1 >= record.len() { continue; }
                let (_, ref from_keys) = record[pair_idx];
                let (_, ref to_keys) = record[pair_idx + 1];
                if from_keys.is_empty() || to_keys.is_empty() { continue; }
                let from = from_keys[from_idx % from_keys.len()];
                let to = to_keys[to_idx % to_keys.len()];
                edges.push(mk_edge(from, to, 1.0, *active));
            }
            let graph = build_graph(edges);
            let cc = ConnectedComponents::compute(&store, &graph, false).unwrap();

            for cid in 0..cc.n_components {
                let span = cc.component_night_span(cid);
                match cc.component_night_bounds(cid) {
                    Some((lo, hi)) => {
                        let expected = u32::from(hi).saturating_sub(u32::from(lo));
                        prop_assert_eq!(span, expected, "component {}: span mismatch", cid);
                    }
                    None => {
                        prop_assert_eq!(span, 0, "component {}: missing bounds but span != 0", cid);
                    }
                }
            }
        }
    }
}
