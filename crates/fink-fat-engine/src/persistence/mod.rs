pub mod alert;
pub mod edge;
pub mod envelope;
pub mod graph;
pub mod seed_node;
pub mod seed_store;

/// Alert store schema version.
pub const ALERT_STORE_SCHEMA_VERSION: u32 = 1;

/// Seed store schema version.
pub const SEED_STORE_SCHEMA_VERSION: u32 = 1;

/// Inter-night graph schema version.
pub const GRAPH_SCHEMA_VERSION: u32 = 1;

/// Optional: top-level state/manifest schema version (if you persist one).
pub const STATE_SCHEMA_VERSION: u32 = 1;
