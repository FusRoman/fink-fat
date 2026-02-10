pub mod component_stats;
pub mod seed_index;
pub mod union_find;

use ahash::AHashMap;

use crate::{
    graph::edge::Edge,
    night_id::NightId,
    persistence::seed_node::SeedKey,
    pipeline::seed_store::SeedStore,
    seeding::seed_node::SeedNode,
    solver::components::{seed_index::SeedGlobalIndex, union_find::UnionFind},
};

pub type ComponentId = u32;

#[derive(Debug, Clone)]
pub struct ConnectedComponents {
    index: SeedGlobalIndex,
    /// Component id for each global node index (0..N_total).
    comp_of_node: Vec<ComponentId>,
    /// Number of components.
    pub n_components: u32,
}

impl ConnectedComponents {
    /// Compute CCs on the undirected view of the graph.
    ///
    /// If `active_only` is true, only edges with `edge.core.active == true` are used.
    pub fn compute(seed_store: &SeedStore, edges: &[Edge], active_only: bool) -> Self {
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
        let mut next: ComponentId = 0;

        for i in 0..n {
            let r = uf.find(i);
            let cid = *root_to_cid.entry(r).or_insert_with(|| {
                let c = next;
                next += 1;
                c
            });
            comp_of_node[i] = cid;
        }

        Self {
            index,
            comp_of_node,
            n_components: next,
        }
    }

    /// Return component sizes (number of nodes per component).
    pub fn sizes(&self) -> Vec<u32> {
        let mut sizes = vec![0u32; self.n_components as usize];
        for &cid in &self.comp_of_node {
            sizes[cid as usize] += 1;
        }
        sizes
    }

    pub fn to_component_ref<'seed_lf, 'alert_lf>(
        &self,
        seed_store: &'seed_lf SeedStore<'alert_lf>,
        index: &SeedGlobalIndex,
    ) -> Vec<Vec<&'seed_lf SeedNode<'alert_lf>>> {
        let init = std::iter::repeat_with(Vec::new)
            .take(self.n_components as usize)
            .collect::<Vec<_>>();

        seed_store
            .iter()
            .map(|(_, seeds)| seeds.iter())
            // aplatit en un flux de &SeedNode
            .flatten()
            .fold(init, |mut components, seed| {
                let gid = index.idx_of_key(seed.core.key);
                let cid = self.comp_of_node[gid] as usize;
                components[cid].push(seed);
                components
            })
    }

    pub fn component_night_bounds<'seed_lf, 'alert_lf>(
        &self,
        seed_store: &'seed_lf SeedStore<'alert_lf>,
        index: &SeedGlobalIndex,
    ) -> Vec<Option<(NightId, NightId)>> {
        let mut min: Vec<Option<NightId>> = vec![None; self.n_components as usize];
        let mut max: Vec<Option<NightId>> = vec![None; self.n_components as usize];

        for (night_id, seeds) in seed_store.iter() {
            for seed in seeds {
                let gid = index.idx_of_key(seed.core.key);
                let cid = self.comp_of_node[gid] as usize;

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

        (0..self.n_components as usize)
            .map(|cid| match (min[cid], max[cid]) {
                (Some(a), Some(b)) => Some((a, b)),
                _ => None,
            })
            .collect()
    }

    pub fn materialize_components_with_bounds<'seed_lf, 'alert_lf>(
        &self,
        seed_store: &'seed_lf SeedStore<'alert_lf>
    ) -> (
        Vec<Vec<&'seed_lf SeedNode<'alert_lf>>>,
        Vec<Option<(NightId, NightId)>>,
    ) {
        let ncomp = self.n_components as usize;

        let sizes = self.sizes();
        let mut components: Vec<Vec<&'seed_lf SeedNode<'alert_lf>>> = (0..ncomp)
            .map(|cid| Vec::with_capacity(sizes[cid] as usize))
            .collect();

        let mut min: Vec<Option<NightId>> = vec![None; ncomp];
        let mut max: Vec<Option<NightId>> = vec![None; ncomp];

        for (night_id, seeds) in seed_store.iter() {
            for seed in seeds {
                let cid = self.compid_of_node(seed.core.key) as usize;

                components[cid].push(seed);

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

        (components, night_bounds)
    }

    pub fn compid_of_node(&self, seed_key: SeedKey) -> ComponentId {
        let node_gid = self.index.idx_of_key(seed_key);
        self.comp_of_node[node_gid as usize]
    }
}
