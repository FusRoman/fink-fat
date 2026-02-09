use ahash::AHashMap;
use camino::Utf8PathBuf;

use crate::{
    night_id::NightId,
    persistence::{
        alert::{Alert, AlertKey, AlertSlice},
        error::PersistenceIoError,
        layout::PersistenceLayout,
        manifest::Manifest,
    },
};

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

    /// Create an `AlertStore` from a pre-existing map of night IDs to alert vectors.
    pub fn from_map(map: AHashMap<NightId, Vec<Alert>>) -> Self {
        Self(map)
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

    /// Get an alert by its key (night ID + index within night).
    pub fn get_by_key(&self, key: AlertKey) -> Option<&Alert> {
        let vec = self.0.get(&key.night_id)?;
        vec.get(key.idx_in_night as usize)
    }

    /// Iterate over (night_id, alerts) pairs.
    pub fn as_map_iter(&self) -> impl Iterator<Item = (&NightId, &Vec<Alert>)> {
        self.0.iter()
    }

    /// Get the internal map size (#nights).
    pub fn n_nights(&self) -> usize {
        self.0.len()
    }

    /// Persist all nights currently present in an `AlertStore`.
    ///
    /// This is a convenience function that:
    /// - iterates over the store,
    /// - writes each night file,
    /// - updates the manifest entries.
    ///
    /// Notes
    /// -----
    /// - This does **not** remove old manifest entries. If you want a "sync" behavior
    ///   (drop nights not present in the store), do it at call site.
    pub fn save_alert_night(
        &self,
        night_id: NightId,
        layout: &PersistenceLayout,
        manifest: &mut Manifest,
    ) -> Result<Utf8PathBuf, PersistenceIoError> {
        let alerts = self.0.get(&night_id).ok_or_else(|| {
            PersistenceIoError::Other(format!("No alerts for night_id {}", night_id))
        })?;
        let path = alerts
            .as_slice()
            .save_alerts_night(layout, manifest, night_id)?;
        return Ok(path);
    }
}
