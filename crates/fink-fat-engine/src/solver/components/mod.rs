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
    /// Number of components.
    pub n_components: u32,
    component_ref: ComponentRef<'seed_lf, 'alert_lf>,
    component_night_bound: ComponentNightBound,
    component_active_edge: Vec<u32>,
}

impl<'seed_lf, 'alert_lf> ConnectedComponents<'seed_lf, 'alert_lf> {
    /// Compute CCs on the undirected view of the graph.
    ///
    /// If `active_only` is true, only edges with `edge.core.active == true` are used.
    pub fn compute(
        seed_store: &'seed_lf SeedStore<'alert_lf>,
        edges: &[Edge],
        active_only: bool,
    ) -> Self {
        let index = SeedGlobalIndex::build(seed_store);
        let n = index.n_total();
        let mut uf = UnionFind::new(n);

        for e in edges {
            if active_only && !e.core.active {
                continue;
            }
            let u = index.idx_of_key(e.from.core.key);
            let v = index.idx_of_key(e.to.core.key);
            uf.union(u, v);
        }

        // root -> component id (dense)
        let mut root_to_cid: AHashMap<usize, ComponentId> = AHashMap::default();
        let mut comp_of_node = vec![0u32; n];
        let mut n_components: ComponentId = 0;

        for i in 0..n {
            let r = uf.find(i);
            let cid = *root_to_cid.entry(r).or_insert_with(|| {
                let c = n_components;
                n_components += 1;
                c
            });
            comp_of_node[i] = cid;
        }

        let mut comp_active_edge: Vec<u32> = Vec::new();
        for e in edges {
            if !e.core.active {
                continue;
            }
            let u = index.idx_of_key(e.from.core.key);
            let cu: ComponentId = comp_of_node[u as usize];

            let v = index.idx_of_key(e.to.core.key);
            let cv: ComponentId = comp_of_node[v as usize];

            if cu == cv {
                comp_active_edge[cu as usize] += 1;
            }
        }

        let mut sizes = vec![0u32; n_components as usize];
        for &cid in &comp_of_node {
            sizes[cid as usize] += 1;
        }

        let ncomp = n_components as usize;
        let mut components_ref: ComponentRef<'seed_lf, 'alert_lf> = (0..ncomp)
            .map(|cid| Vec::with_capacity(sizes[cid] as usize))
            .collect();

        let mut min: Vec<Option<NightId>> = vec![None; ncomp];
        let mut max: Vec<Option<NightId>> = vec![None; ncomp];

        for (night_id, seeds) in seed_store.iter() {
            for seed in seeds {
                let node_gid = index.idx_of_key(seed.core.key);
                let cid = comp_of_node[node_gid as usize] as usize;

                components_ref[cid].push(seed);

                min[cid] = Some(match min[cid] {
                    None => *night_id,
                    Some(x) => x.min(*night_id),
                });
                max[cid] = Some(match max[cid] {
                    None => *night_id,
                    Some(x) => x.max(*night_id),
                });
            }
        }

        let night_bounds = (0..ncomp)
            .map(|cid| match (min[cid], max[cid]) {
                (Some(a), Some(b)) => Some((a, b)),
                _ => None,
            })
            .collect();

        Self {
            index,
            comp_of_node,
            n_components: n_components,
            component_ref: components_ref,
            component_night_bound: night_bounds,
            component_active_edge: comp_active_edge,
        }
    }

    pub fn comp_nodes(&self, component_id: ComponentId) -> &[&SeedNode<'alert_lf>] {
        &self.component_ref[component_id as usize]
    }

    pub fn size_component_id(&self, component_id: ComponentId) -> usize {
        self.component_ref[component_id as usize].len()
    }

    pub fn size_component_seed(&self, seed_key: SeedKey) -> usize {
        let cid = self.compid_of_node(seed_key);
        self.size_component_id(cid)
    }

    pub fn nb_edges_component_id(&self, component_id: ComponentId) -> u32 {
        self.component_active_edge[component_id as usize]
    }

    pub fn nb_edges_component_seed(&self, seed_key: SeedKey) -> u32 {
        let cid = self.compid_of_node(seed_key);
        self.nb_edges_component_id(cid)
    }

    pub fn night_bounds_component_seed(&self, seed_key: SeedKey) -> Option<(NightId, NightId)> {
        let cid = self.compid_of_node(seed_key);
        self.component_night_bound[cid as usize]
    }

    pub fn compid_of_node(&self, seed_key: SeedKey) -> ComponentId {
        let node_gid = self.index.idx_of_key(seed_key);
        self.comp_of_node[node_gid as usize]
    }

    #[inline]
    pub fn night_span(&self, component_id: ComponentId) -> u32 {
        let night_bounds = self.component_night_bound[component_id as usize];
        match night_bounds {
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
        let comp_size = self.size_component_id(component_id);
        // Trivial fast path.
        if comp_size <= solver_policy.trivial_max_nodes as usize {
            return SolverChoice::Trivial;
        }

        // Night span guardrail.
        if self.night_span(component_id) > solver_policy.max_night_span_for_mcf {
            return SolverChoice::BlobBreaker;
        }

        // Budget-aware MCF vs BlobBreaker.
        let t_est = solver_policy
            .estimate_mcf_time_s(comp_size as u32, self.nb_edges_component_id(component_id));
        if t_est <= solver_policy.mcf_budget_s {
            SolverChoice::MinCostFlow
        } else {
            SolverChoice::BlobBreaker
        }
    }
}
