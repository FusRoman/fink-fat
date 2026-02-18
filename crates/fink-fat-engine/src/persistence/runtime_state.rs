use crate::{
    alerts::store::AlertStore,
    night_id::PairingMode,
    persistence::{graph::GraphOwned, manifest::Manifest},
    seeding::store::SeedStore,
};

/// Loaded runtime state built from persisted artifacts.
///
/// This matches the runtime needs:
/// - `AlertStore` owns the alert vectors (per night).
/// - `SeedStoreOwned` owns `SeedNodeOwned`.
/// - `InterNightGraph<'seed,'alert>` owns `Edge<'seed,'alert>` that borrow seeds.
///
/// Notes
/// -----
/// This struct is meant to be created and then moved into your runtime engine
/// (or into a `FinkFat` instance).
pub struct RuntimeState {
    pub manifest: Manifest,
    pub window: Option<PairingMode>,
    pub alert_store: AlertStore,
    pub seed_store: SeedStore,
    pub graph: GraphOwned,
}
