use serde::{Deserialize, Serialize};

use crate::{
    graph::edge::{Edge, EdgeCore},
    persistence::seed_node::SeedKey,
    pipeline::seed_store::SeedStore,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EdgeOwned {
    pub core: EdgeCore,
    pub from: SeedKey,
    pub to: SeedKey,
}

impl EdgeOwned {
    pub fn to_borrowed<'seed, 'alert>(
        &self,
        seeds: &'seed SeedStore<'alert>,
    ) -> Result<Edge<'seed, 'alert>, String> {
        let from = seeds
            .get_by_key(self.from)
            .ok_or_else(|| format!("Missing seed for key {:?}", self.from))?;
        let to = seeds
            .get_by_key(self.to)
            .ok_or_else(|| format!("Missing seed for key {:?}", self.to))?;

        Ok(Edge {
            core: self.core.clone(),
            from,
            to,
        })
    }
}
