use ahash::AHashMap;

use crate::{
    graph::{GraphCore, RuntimeGraph},
    persistence::{edge::EdgeOwned, error::BorrowError},
    pipeline::seed_store::SeedStore,
};

pub struct GraphOwned {
    pub core: GraphCore,
    pub edges: Vec<EdgeOwned>,
}

impl GraphOwned {
    pub fn from_edges(edges: Vec<EdgeOwned>) -> Self {
        let mut core = GraphCore {
            in_deg: AHashMap::new(),
            out_deg: AHashMap::new(),
        };

        for edge in &edges {
            let from = edge.from;
            let to = edge.to;

            *core.out_deg.entry(from).or_insert(0) += 1;
            *core.in_deg.entry(to).or_insert(0) += 1;
        }

        Self { core, edges }
    }

    pub fn to_borrowed<'seed, 'alert>(
        &self,
        seeds: &'seed SeedStore<'alert>,
    ) -> Result<RuntimeGraph<'seed, 'alert>, BorrowError> {
        let mut edges = Vec::with_capacity(self.edges.len());
        for e in &self.edges {
            edges.push(e.to_borrowed(seeds)?);
        }
        Ok(RuntimeGraph {
            core: self.core.clone(),
            edges,
        })
    }
}
