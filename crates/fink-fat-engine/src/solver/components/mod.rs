pub mod seed_index;
pub mod union_find;

use ahash::AHashMap;

use crate::{
    graph::edge::Edge,
    night_id::NightId,
    persistence::seed_node::SeedKey,
    pipeline::seed_store::SeedStore,
    seeding::seed_node::SeedNode,
    solver::{
        components::{seed_index::SeedGlobalIndex, union_find::UnionFind},
        solver_manager::{SolverChoice, SolverPolicy},
    },
};

pub type ComponentId = u32;
pub type ComponentRef<'seed_lf, 'alert_lf> = Vec<Vec<&'seed_lf SeedNode<'alert_lf>>>;
pub type ComponentNightBound = Vec<Option<(NightId, NightId)>>;

#[derive(Debug, Clone)]
pub struct ConnectedComponents<'seed_lf, 'alert_lf> {
    index: SeedGlobalIndex,
    /// Component id for each global node index (0..N_total).
    comp_of_node: Vec<ComponentId>,
    /// Number of components (dense component ids 0..n_components-1).
    pub n_components: u32,
    component_ref: ComponentRef<'seed_lf, 'alert_lf>,
    component_night_bound: ComponentNightBound,
    /// Count of *active* intra-component edges per component.
    component_active_edge: Vec<u32>,
}

impl<'seed_lf, 'alert_lf> ConnectedComponents<'seed_lf, 'alert_lf> {
    /// Compute connected components on the *undirected* view of the graph.
    ///
    /// If `active_only` is true, only edges with `edge.core.active == true` are used
    /// to build components. Edge counts (`component_active_edge`) always count only
    /// active edges.
    pub fn compute(
        seed_store: &'seed_lf SeedStore<'alert_lf>,
        edges: &[Edge],
        active_only: bool,
    ) -> Self {
        let index = SeedGlobalIndex::build(seed_store);
        let n_total = index.n_total();

        // 1) Union-Find on nodes (undirected view).
        let mut uf = UnionFind::new(n_total);
        Self::union_edges(&index, &mut uf, edges, active_only);

        // 2) Dense component ids + comp_of_node.
        let (comp_of_node, n_components) = Self::dense_components(&mut uf, n_total);

        // 3) Count active intra-component edges (always active ones).
        let component_active_edge =
            Self::count_active_intra_edges(&index, edges, &comp_of_node, n_components);

        // 4) Build component -> node refs, and night bounds.
        let (component_ref, component_night_bound) =
            Self::build_refs_and_bounds(seed_store, &index, &comp_of_node, n_components);

        Self {
            index,
            comp_of_node,
            n_components: n_components as u32,
            component_ref,
            component_night_bound,
            component_active_edge,
        }
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

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

    /// Return (comp_of_node, n_components) with dense component ids 0..C-1.
    fn dense_components(uf: &mut UnionFind, n_total: usize) -> (Vec<ComponentId>, usize) {
        // root -> dense cid
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

    fn build_refs_and_bounds(
        seed_store: &'seed_lf SeedStore<'alert_lf>,
        index: &SeedGlobalIndex,
        comp_of_node: &[ComponentId],
        n_components: usize,
    ) -> (ComponentRef<'seed_lf, 'alert_lf>, ComponentNightBound) {
        // Pre-compute component sizes to reserve exact-ish capacity.
        let mut sizes = vec![0u32; n_components];
        for &cid in comp_of_node {
            sizes[cid as usize] += 1;
        }

        let mut component_ref: ComponentRef<'seed_lf, 'alert_lf> = (0..n_components)
            .map(|cid| Vec::with_capacity(sizes[cid] as usize))
            .collect();

        let mut min_night: Vec<Option<NightId>> = vec![None; n_components];
        let mut max_night: Vec<Option<NightId>> = vec![None; n_components];

        for (night_id, seeds) in seed_store.iter() {
            for seed in seeds {
                let gid = index.idx_of_key(seed.core.key);
                let cid = comp_of_node[gid] as usize;

                component_ref[cid].push(seed);

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

        (component_ref, component_night_bound)
    }

    // -------------------------------------------------------------------------
    // Unified getters / accessors
    // -------------------------------------------------------------------------

    /// Component id for a given seed key.
    pub fn component_id_of_seed(&self, seed_key: SeedKey) -> ComponentId {
        let gid = self.index.idx_of_key(seed_key);
        self.comp_of_node[gid]
    }

    /// Nodes of a component (borrowed seed refs).
    pub fn component_nodes(&self, component_id: ComponentId) -> &[&SeedNode<'alert_lf>] {
        &self.component_ref[component_id as usize]
    }

    /// Size (#nodes) of a component.
    pub fn component_size(&self, component_id: ComponentId) -> usize {
        self.component_ref[component_id as usize].len()
    }

    /// Size (#nodes) of the component containing `seed_key`.
    pub fn component_size_of_seed(&self, seed_key: SeedKey) -> usize {
        let cid = self.component_id_of_seed(seed_key);
        self.component_size(cid)
    }

    /// Number of *active* intra-component edges.
    pub fn component_active_edges(&self, component_id: ComponentId) -> u32 {
        self.component_active_edge[component_id as usize]
    }

    /// Number of *active* intra-component edges for the component containing `seed_key`.
    pub fn component_active_edges_of_seed(&self, seed_key: SeedKey) -> u32 {
        let cid = self.component_id_of_seed(seed_key);
        self.component_active_edges(cid)
    }

    /// (min_night, max_night) bounds for the component (if any node exists).
    pub fn component_night_bounds(&self, component_id: ComponentId) -> Option<(NightId, NightId)> {
        self.component_night_bound[component_id as usize]
    }

    /// Night bounds for the component containing `seed_key`.
    pub fn component_night_bounds_of_seed(&self, seed_key: SeedKey) -> Option<(NightId, NightId)> {
        let cid = self.component_id_of_seed(seed_key);
        self.component_night_bounds(cid)
    }

    /// Night span in integer days (max - min, saturating).
    #[inline]
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

    /// Classify a component into a solver choice using the current policy.
    pub fn classify(
        &self,
        component_id: ComponentId,
        solver_policy: &SolverPolicy,
    ) -> SolverChoice {
        let n = self.component_size(component_id);

        // Trivial fast path.
        if n <= solver_policy.trivial_max_nodes as usize {
            return SolverChoice::Trivial;
        }

        // Night span guardrail.
        if self.component_night_span(component_id) > solver_policy.max_night_span_for_mcf {
            return SolverChoice::BlobBreaker;
        }

        // Budget-aware MCF vs BlobBreaker.
        let t_est =
            solver_policy.estimate_mcf_time_s(n as u32, self.component_active_edges(component_id));

        if t_est <= solver_policy.mcf_budget_s {
            SolverChoice::MinCostFlow
        } else {
            SolverChoice::BlobBreaker
        }
    }
}
