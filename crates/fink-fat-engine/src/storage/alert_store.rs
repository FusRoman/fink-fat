use ahash::AHashMap;

use crate::{Alert, night_id::NightId};

/// Contiguous store of alerts for (typically) a single night.
///
/// The vector `alerts` is indexed by [`AlertId`] (0-based) and provides
/// cache-friendly iteration for the seeding and linking pipeline.
///
/// Design
/// ------
/// - `start_mjd` is the **floor** of the minimum `mjd_tt` in the store and
///   can serve as origin for uniform time binning in time-based indexing
///   structures.
/// - Alerts are treated as immutable after construction to simplify sharing
///   across threads.
#[derive(Debug, Clone)]
pub struct AlertStore(AHashMap<NightId, Vec<Alert>>);

impl AlertStore {
    /// Create a new empty `AlertStore`.
    pub fn new() -> Self {
        Self(AHashMap::new())
    }

    /// Insert a vector of alerts for a given night.
    pub fn insert(&mut self, night_id: NightId, alerts: Vec<Alert>) {
        self.0.insert(night_id, alerts);
    }

    /// Get the vector of alerts for a given night, if it exists.
    pub fn get(&self, night_id: &NightId) -> Option<&Vec<Alert>> {
        self.0.get(night_id)
    }

    /// Get an iterator over all alerts in the store, across all nights.
    pub fn iter(&self) -> impl Iterator<Item = &Alert> {
        self.0.values().flatten()
    }

    /// Get an iterator over all alerts for a specific night, if it exists.
    pub fn iter_night(&self, night_id: &NightId) -> Option<impl Iterator<Item = &Alert>> {
        self.0.get(night_id).map(|alerts| alerts.iter())
    }
}
