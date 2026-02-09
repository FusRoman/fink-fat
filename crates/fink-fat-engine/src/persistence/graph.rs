use crate::{
    graph::InterNightGraph,
    persistence::{edge::EdgeOwned, error::BorrowError},
    pipeline::seed_store::SeedStore,
};

pub struct GraphOwned {
    pub edges: Vec<EdgeOwned>,
}

impl GraphOwned {
    pub fn to_borrowed<'seed, 'alert>(
        &self,
        seeds: &'seed SeedStore<'alert>,
    ) -> Result<InterNightGraph<'seed, 'alert>, BorrowError> {
        let mut edges = Vec::with_capacity(self.edges.len());
        for e in &self.edges {
            edges.push(e.to_borrowed(seeds)?);
        }
        Ok(InterNightGraph { edges })
    }
}
