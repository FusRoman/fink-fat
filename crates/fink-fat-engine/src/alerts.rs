//! Alert data model for the Fink-FAT engine.
//!
//! This module defines the core alert types used throughout the engine.
//! It is intentionally free of any Python bindings or seeding logic so it
//! can be reused from pure Rust crates (engine, evaluation, CLI).
//!
//! Units & Conventions
//! -------------------
//! - `ra`, `dec` are in **radians** (ICRS, J2000).
//! - `mjd_tt` is **Modified Julian Date in TT (Terrestrial Time)**.
//! - `flux` is PSF **difference** flux (e.g. nJy), and `flux_err` its error.
//! - [`AlertId`] is a **0-based** dense integer index into an `AlertStore`.

use std::fmt::{Display, Formatter, Result};

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

/// Dense identifier for an alert within a contiguous store.
///
/// This is a 0-based index into a [`Vec<Alert>`] or [`AlertStore::alerts`].
///
/// Invariants
/// ----------
/// - `idx()` must always be `< alerts.len()` when used for indexing.
/// - IDs are assumed to be dense (no gaps) within a given store.
#[derive(
    Copy,
    Clone,
    Debug,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Hash,
    Default,
    Serialize,
    Deserialize,
    Encode,
    Decode,
)]
pub struct AlertId(u32);

impl Display for AlertId {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "AlertId({})", self.0)
    }
}

impl AlertId {
    /// Create a new `AlertId` from a 0-based index.
    #[inline]
    pub fn new(idx: u32) -> Self {
        Self(idx)
    }

    /// Return the underlying 0-based index as `usize`.
    #[inline]
    pub fn idx(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for AlertId {
    #[inline]
    fn from(value: u32) -> Self {
        AlertId::new(value)
    }
}

impl From<AlertId> for u32 {
    #[inline]
    fn from(id: AlertId) -> Self {
        id.0
    }
}

impl From<usize> for AlertId {
    #[inline]
    fn from(value: usize) -> Self {
        AlertId::new(value as u32)
    }
}

/// Return the **first index** `i` such that `times_by_id[ids[i]] > key_time`.
///
/// This is a standard lower-bound search on a **time-sorted** `ids` slice,
/// implemented to avoid allocations and keep the inner loop branch-light.
///
/// Arguments
/// ---------
/// * `ids` – Slice of `AlertId` **sorted by time** (ascending).
/// * `key_time` – Threshold time (days).
/// * `times_by_id` – Dense table `AlertId → mjd_tt`.
#[inline]
pub(crate) fn lower_bound_gt_ids(ids: &[AlertId], key_time: f64, times_by_id: &[f64]) -> usize {
    let (mut lo, mut hi) = (0usize, ids.len());
    while lo < hi {
        let mid = (lo + hi) / 2;
        let t = times_by_id[ids[mid].idx()];
        if t > key_time { hi = mid } else { lo = mid + 1 }
    }
    lo
}

/// Single detection in the alert stream.
///
/// This record is intentionally compact and cloneable so it can be moved across
/// threads and used as a building block for seeding, graph construction and
/// trajectory reconstruction.
///
/// Fields
/// ------
/// - `id` – [`AlertId`] assigned on ingestion; indexes `alerts[id.idx()]`.
/// - `dia_source_id` – LSST `diaSourceId` (stable, 64-bit).
/// - `ra`, `dec` – ICRS coordinates in **radians**.
/// - `ra_err`, `dec_err` – 1-sigma uncertainties on `ra` and `dec` in **radians**.
/// - `mjd_tt` – **MJD (TT)** timestamp of the detection.
/// - `flux`, `flux_err` – PSF **difference** flux and its uncertainty
///   (units depend on upstream).
/// - `band` – integer photometric band code.
#[derive(Clone, Debug, Default)]
pub struct Alert {
    pub id: AlertId,
    pub dia_source_id: u64, // from LSST
    pub ra: f64,            // rad
    pub ra_err: f64,        // rad
    pub dec: f64,           // rad
    pub dec_err: f64,       // rad
    pub mjd_tt: f64,        // days (TT)
    pub flux: f32,          // psf flux (difference image, e.g. nJy)
    pub flux_err: f32,      // psf flux error
    pub band: u8,           // photometric band
}

/* ------------------------ Display / Debug ------------------------- */

impl Display for Alert {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        // Compact single-line rendering for logs.
        write!(
            f,
            "Alert(id={}, dia_source_id={}, ra={:.6} rad, dec={:.6} rad, mjd_tt={:.5}, \
             flux={:.3}±{:.3} nJy, band={})",
            self.id.idx(),
            self.dia_source_id,
            self.ra,
            self.dec,
            self.mjd_tt,
            self.flux,
            self.flux_err,
            self.band
        )
    }
}

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
pub struct AlertStore {
    /// Night anchor (TT): floor of the minimum `mjd_tt` in `alerts`.
    pub start_mjd: f64,
    /// All alerts, densely indexed by [`AlertId`].
    pub alerts: Vec<Alert>,
}

impl AlertStore {
    /// Construct a new `AlertStore` from a start MJD anchor and a vector of alerts.
    ///
    /// The caller is responsible for ensuring that:
    /// - `start_mjd` is consistent with the minimum `mjd_tt` in `alerts`,
    /// - alert IDs are dense and compatible with their position in the vector.
    pub fn new(start_mjd: f64, alerts: Vec<Alert>) -> Self {
        Self { start_mjd, alerts }
    }

    /// Number of alerts in the store.
    #[inline]
    pub fn len(&self) -> usize {
        self.alerts.len()
    }

    /// Return `true` if the store contains no alerts.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.alerts.is_empty()
    }

    /// Borrow one alert by id (checked).
    ///
    /// Return
    /// ------
    /// * `Some(&Alert)` if the id is within bounds,
    /// * `None` otherwise.
    #[inline]
    pub fn get(&self, id: AlertId) -> Option<&Alert> {
        self.alerts.get(id.idx())
    }

    /// Iterate over all alerts in the store.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Alert> {
        self.alerts.iter()
    }

    /// Iterate over all alert ids in the store (0-based, dense).
    #[inline]
    pub fn ids(&self) -> impl Iterator<Item = AlertId> {
        (0..self.alerts.len()).map(|i| AlertId::new(i as u32))
    }

    /// Resolve an arbitrary list of `AlertId`s into borrowed `&Alert`s (checked).
    ///
    /// The iterator short-circuits to `None` if any id is out-of-bounds.
    pub fn get_many<'a>(
        &self,
        ids: impl IntoIterator<Item = &'a AlertId>,
    ) -> Option<impl Iterator<Item = &Alert>> {
        let mut v = Vec::new();
        for id in ids {
            v.push(self.alerts.get(id.idx())?);
        }
        Some(v.into_iter())
    }
}

impl Display for AlertStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        // Lightweight summary without iterating the alerts.
        write!(
            f,
            "AlertStore(start_mjd={:.5}, n_alerts={})",
            self.start_mjd,
            self.alerts.len()
        )
    }
}
