//! Seeding-level policy configuration.
//!
//! This module contains options that control how intra-night seeds are emitted
//! by the [`BuildSeeds`](crate::pipeline::stages::PipelineStage::BuildSeeds)
//! stage.

use serde::{Deserialize, Serialize};

/// Configuration controlling how seeds are emitted during `BuildSeeds`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SeedingConfig {
    /// If `true`, keep only triplet-derived seeds and drop pair-derived seeds.
    pub triplet_only: bool,
}
