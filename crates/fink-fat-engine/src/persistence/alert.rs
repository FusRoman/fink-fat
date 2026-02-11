//! Alert data model for the Fink-FAT engine.
//!
//! This module defines the core detection record (`Alert`) used throughout
//! the Fink-FAT engine pipeline (pairing, seeding, graph construction,
//! ML ranking, trajectory reconstruction).
//!
//! That separation keeps the `Alert` type reusable across crates (engine,
//! evaluation, CLI) and makes it easy to benchmark and test.
//!
//! Units & conventions
//! -------------------
//! - `ra`, `dec` are in **radians** (ICRS/J2000 conventions as provided upstream).
//! - `ra_err`, `dec_err` are **1σ** uncertainties in **radians**.
//! - `mjd_tt` is **Modified Julian Date** in **TT** (Terrestrial Time), in days.
//! - `flux` is PSF **difference** flux (units depend on upstream; often nJy),
//!   and `flux_err` is the corresponding 1σ uncertainty.
//! - `band` is a compact integer photometric band code.
//!
//! Ordering, hashing, and determinism
//! ----------------------------------
//! Many parts of the pipeline rely on deterministic iteration order:
//! - bucket members are sorted by time,
//! - candidate enumeration is reproducible,
//! - benchmarks and tests do not depend on hash-map iteration order.
//!
//! To support that, `Alert` implements:
//! - [`Ord`] / [`PartialOrd`] with a total ordering (primary key: `mjd_tt`),
//! - [`Hash`], [`Eq`], [`PartialEq`] using stable bitwise representations for floats.
//!
//! Important: float equality & hashing
//! -----------------------------------
//! Floating-point fields (`f64`, `f32`) are compared / hashed using their raw bit
//! patterns (`to_bits()`), not epsilon-based approximate equality. This choice:
//! - makes `Eq`/`Hash` **sound** and deterministic,
//! - allows using alerts as keys in hash sets/maps,
//! - avoids surprising behavior due to floating rounding tolerance.
//!
//! Consequences:
//! - Values that are numerically “close” but not bit-identical are **not equal**.
//! - Different NaN payloads are treated as **different** values.
//!
//! See also
//! --------
//! - `spacetime_bucket::bucket` – bucket index relies on `Ord` to sort members.
//! - `seeding::pairs` – deduplicates pairs using alert pointer identity.
//! - `seeding::seed_node` – seeds borrow `&Alert` references.

use std::{
    cmp::Ordering,
    fmt::{Display, Formatter, Result as FmtResult},
    hash::{Hash, Hasher},
};

use camino::Utf8PathBuf;
use serde::{Deserialize, Serialize};

use crate::{
    MJDTT, Radian,
    night_id::NightId,
    persistence::{
        ALERT_STORE_SCHEMA_VERSION, envelope::DiskEnvelope, error::PersistenceIoError,
        layout::PersistenceLayout, manifest::Manifest,
    },
};

/// Single detection in the alert stream.
///
/// This record is intentionally compact and cloneable so it can be moved across
/// threads and used as a building block for seeding, graph construction, and
/// trajectory reconstruction.
///
/// Fields
/// ------
/// - `dia_source_id` – LSST `diaSourceId` (stable 64-bit identifier).
/// - `ra`, `dec` – ICRS sky coordinates in **radians**.
/// - `ra_err`, `dec_err` – 1σ uncertainties on `ra` and `dec` in **radians**.
/// - `mjd_tt` – detection epoch in **MJD (TT)**, in days.
/// - `flux`, `flux_err` – PSF difference flux and its 1σ uncertainty
///   (units depend on upstream).
/// - `band` – integer photometric band code.
///
/// Notes
/// -----
/// - The struct does not encode provenance (visit, detector, etc.) by design.
///   Those may exist upstream but are not required for the core linking logic.
/// - The engine frequently borrows `&Alert` references in indices and seeds
///   rather than copying these fields repeatedly.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Alert {
    /// Unique identifier for the alert, used for disk persistance.
    pub key: AlertKey,
    /// LSST diaSourceId (stable, 64-bit).
    pub dia_source_id: u64,
    /// Right ascension (radians).
    pub ra: Radian,
    /// 1σ uncertainty on RA (radians).
    pub ra_err: Radian,
    /// Declination (radians).
    pub dec: Radian,
    /// 1σ uncertainty on Dec (radians).
    pub dec_err: Radian,
    /// Detection epoch (MJD TT, days).
    pub mjd_tt: MJDTT,
    /// PSF difference flux (units depend on upstream, e.g. nJy).
    pub flux: f32,
    /// 1σ uncertainty on flux (same units as `flux`).
    pub flux_err: f32,
    /// Photometric band code.
    pub band: u8,
}

/* ------------------------ Equality / Ordering ------------------------- */

impl PartialEq for Alert {
    fn eq(&self, other: &Self) -> bool {
        // We use bitwise float equality to make Eq/Hash sound and deterministic.
        self.dia_source_id == other.dia_source_id
            && self.band == other.band
            && self.mjd_tt.to_bits() == other.mjd_tt.to_bits()
            && self.ra.to_bits() == other.ra.to_bits()
            && self.dec.to_bits() == other.dec.to_bits()
            && self.ra_err.to_bits() == other.ra_err.to_bits()
            && self.dec_err.to_bits() == other.dec_err.to_bits()
            && self.flux.to_bits() == other.flux.to_bits()
            && self.flux_err.to_bits() == other.flux_err.to_bits()
    }
}

impl Eq for Alert {}

impl PartialOrd for Alert {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Alert {
    fn cmp(&self, other: &Self) -> Ordering {
        // Primary sort key: observation time.
        self.mjd_tt
            .total_cmp(&other.mjd_tt)
            // Deterministic tie-breakers.
            .then_with(|| self.dia_source_id.cmp(&other.dia_source_id))
            .then_with(|| self.band.cmp(&other.band))
            .then_with(|| self.ra.total_cmp(&other.ra))
            .then_with(|| self.dec.total_cmp(&other.dec))
            .then_with(|| self.ra_err.total_cmp(&other.ra_err))
            .then_with(|| self.dec_err.total_cmp(&other.dec_err))
            .then_with(|| self.flux.total_cmp(&other.flux))
            .then_with(|| self.flux_err.total_cmp(&other.flux_err))
    }
}

/* ----------------------------- Hash ---------------------------------- */

impl Hash for Alert {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dia_source_id.hash(state);
        self.band.hash(state);

        // Hash float fields by raw bits to match Eq and ensure determinism.
        self.mjd_tt.to_bits().hash(state);
        self.ra.to_bits().hash(state);
        self.dec.to_bits().hash(state);
        self.ra_err.to_bits().hash(state);
        self.dec_err.to_bits().hash(state);

        self.flux.to_bits().hash(state);
        self.flux_err.to_bits().hash(state);
    }
}

/* ------------------------ Display ------------------------------------ */

impl Display for Alert {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        // Compact single-line rendering for logs and debugging output.
        write!(
            f,
            "Alert(dia_source_id={}, ra={:.6} rad, dec={:.6} rad, mjd_tt={:.5}, \
             flux={:.3}±{:.3}, band={})",
            self.dia_source_id, self.ra, self.dec, self.mjd_tt, self.flux, self.flux_err, self.band
        )
    }
}

/* ------------------------ Alert Slice trait ------------------------------------ */

pub trait AlertSlice {
    fn save_alerts_night(
        &self,
        layout: &PersistenceLayout,
        manifest: &Manifest,
        night_id: NightId,
    ) -> Result<Utf8PathBuf, PersistenceIoError>;
}

impl AlertSlice for &[Alert] {
    /// Write the alerts of a single night to disk and upsert the manifest entry.
    ///
    /// Behavior
    /// --------
    /// - Writes `DiskEnvelope<Vec<Alert>>` to `layout.alerts_night_path(night_id)`.
    /// - Upserts `manifest.nights` for `night_id`:
    ///   - sets `alerts_rel_path`,
    ///   - sets `n_alerts`,
    ///   - preserves existing `seeds_rel_path` if present; otherwise fills it with
    ///     the default `layout.seeds_night_path(night_id)` relative path.
    ///   - preserves `n_seeds` if present.
    ///
    /// Returns
    /// -------
    /// Ok(()) on success, or `PersistenceIoError` on I/O / decode / envelope failure.
    fn save_alerts_night(
        &self,
        layout: &PersistenceLayout,
        manifest: &Manifest,
        night_id: NightId,
    ) -> Result<Utf8PathBuf, PersistenceIoError> {
        let abs_path = layout.alerts_night_path(night_id);

        // Write payload (enveloped).
        let env = DiskEnvelope::new(
            self.to_vec(),
            ALERT_STORE_SCHEMA_VERSION,
            manifest.created_unix_s,
        );
        env.save_enveloped(&abs_path)?;
        Ok(abs_path)
    }
}

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct AlertKey {
    pub night_id: NightId,
    pub idx_in_night: u32,
}
