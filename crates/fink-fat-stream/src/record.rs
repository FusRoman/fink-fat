//! [`AlertRecord`]: one Rubin alert as the driver receives, journals and
//! replays it.
//!
//! photom's [`DiaSourceRecord`] borrows its `band`, which suits a one-shot
//! conversion but not a value that is buffered, written to the night journal
//! and read back after a crash. `AlertRecord` is its owned counterpart, plus
//! the instant the alert reached the driver ([`AlertRecord::arrival_mjd`]).
//! It is `rkyv`-serializable: it is what the
//! [night journal](crate::journal) stores.
//!
//! # Three time scales
//!
//! | Quantity | Scale | Used by |
//! |----------|-------|---------|
//! | [`AlertRecord::midpoint_mjd_tai`] | TAI (as published by Rubin) | photom's conversion, and the night consistency check |
//! | the converted observation's epoch | TT (`TAI + 32.184 s`, set by photom) | the tracker (Kalman propagation, visits) |
//! | [`AlertRecord::arrival_mjd`] | **UTC** | the driver's clock: visit idle timeout, twilight, `dayObs` rollover |
//!
//! The clock is on UTC because twilight and the 12:00 UTC rollover are civil
//! times; [`mjd_tai_to_utc`] bridges a TAI mid-point to it (the replay uses it
//! to simulate arrivals).

use photom::NightId;
use photom::io::lsst_alert::{
    DiaSourceRecord, LsstAlertError,
    night::{VISIT_SEQ_FACTOR, day_obs_from_visit, day_obs_to_night_id},
};

/// `TAI − UTC`, in days: 37 s, constant since 2017-01-01 (no leap second
/// has been announced since).
pub const TAI_MINUS_UTC_DAYS: f64 = 37.0 / 86_400.0;

/// Convert an MJD on the TAI scale to UTC (see [`TAI_MINUS_UTC_DAYS`]).
///
/// Valid for instants after 2017-01-01; earlier ones would need the leap
/// second table, which no Rubin alert does.
pub fn mjd_tai_to_utc(mjd_tai: f64) -> f64 {
    mjd_tai - TAI_MINUS_UTC_DAYS
}

/// One alert's `DiaSource` fields, owned, in Rubin units (see
/// [`DiaSourceRecord`] for each field's meaning), plus its arrival time.
///
/// Only the fields tracking needs are kept: the rest of the alert (cutouts,
/// history, object fields) never leaves the source. Float fields keep the
/// APDB's precision (`f32` for errors and fluxes) so a record round-trips
/// exactly through the journal.
#[derive(Debug, Clone, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct AlertRecord {
    /// `diaSourceId`.
    pub dia_source_id: i64,
    /// `visit` = `dayObs × 100000 + seqNum`.
    pub visit: i64,
    /// `midpointMjdTai`, MJD TAI.
    pub midpoint_mjd_tai: f64,
    /// `ra`, degrees.
    pub ra_deg: f64,
    /// `dec`, degrees.
    pub dec_deg: f64,
    /// `raErr` (on sky), degrees.
    pub ra_err_deg: f32,
    /// `decErr`, degrees.
    pub dec_err_deg: f32,
    /// `psfFlux`, nJy.
    pub psf_flux_njy: f32,
    /// `psfFluxErr`, nJy.
    pub psf_flux_err_njy: f32,
    /// `band`, one of `u g r i z y`.
    pub band: String,
    /// When the alert reached the driver, MJD **UTC** — the driver's clock
    /// (see [`crate::driver`]). For a live source this is the reception
    /// time; a replay simulates it (see
    /// [`ReplaySource`](crate::source::ReplaySource)).
    pub arrival_mjd: f64,
}

impl AlertRecord {
    /// Borrowed view in photom's input type, for its quality cuts and its
    /// conversion (see [`crate::ingest`]). Every field but
    /// [`Self::arrival_mjd`] is passed through unchanged.
    pub fn dia_source(&self) -> DiaSourceRecord<'_> {
        DiaSourceRecord {
            dia_source_id: self.dia_source_id,
            visit: self.visit,
            midpoint_mjd_tai: self.midpoint_mjd_tai,
            ra_deg: self.ra_deg,
            dec_deg: self.dec_deg,
            ra_err_deg: self.ra_err_deg,
            dec_err_deg: self.dec_err_deg,
            psf_flux_njy: self.psf_flux_njy,
            psf_flux_err_njy: self.psf_flux_err_njy,
            band: &self.band,
        }
    }

    /// Observing night encoded in [`Self::visit`]: the MJD of its `dayObs`
    /// date (Rubin's night label, rolling over at 12:00 UTC so a night is
    /// never split).
    ///
    /// Read from `visit` alone, without the full conversion: the driver
    /// uses it to find the night of a journal before replaying it.
    ///
    /// # Errors
    /// [`LsstAlertError`] when `visit` does not encode a valid date.
    pub fn night_id(&self) -> Result<NightId, LsstAlertError> {
        day_obs_to_night_id(day_obs_from_visit(self.visit)?)
    }

    /// `seqNum`: the exposure's rank within its night (the low five
    /// digits of `visit`). Visits of a night ordered by `visit` are ordered
    /// by `seqNum`, hence chronologically.
    pub fn seq_num(&self) -> i64 {
        self.visit % VISIT_SEQ_FACTOR
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A valid alert of visit `2026103100042` (night 2026-10-31, MJD 61344).
    pub(crate) fn sample_record() -> AlertRecord {
        AlertRecord {
            dia_source_id: 1_234,
            visit: 2_026_103_100_042,
            midpoint_mjd_tai: 61_345.125,
            ra_deg: 120.0,
            dec_deg: -30.0,
            ra_err_deg: 1e-5,
            dec_err_deg: 1e-5,
            psf_flux_njy: 5_000.0,
            psf_flux_err_njy: 100.0,
            band: "r".to_string(),
            arrival_mjd: 61_345.127,
        }
    }

    #[test]
    fn night_and_seq_num_come_from_the_visit() {
        let record = sample_record();
        assert_eq!(record.night_id().unwrap(), NightId(61_344));
        assert_eq!(record.seq_num(), 42);
    }

    #[test]
    fn dia_source_view_carries_every_field() {
        let record = sample_record();
        let view = record.dia_source();
        assert_eq!(view.dia_source_id, record.dia_source_id);
        assert_eq!(view.visit, record.visit);
        assert_eq!(view.band, "r");
        assert_eq!(view.psf_flux_njy, record.psf_flux_njy);
    }

    #[test]
    fn records_round_trip_through_rkyv() {
        let record = sample_record();
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&record).unwrap();
        let back = rkyv::from_bytes::<AlertRecord, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(back, record);
    }

    #[test]
    fn utc_is_37_seconds_behind_tai() {
        assert!((mjd_tai_to_utc(1.0) - (1.0 - 37.0 / 86_400.0)).abs() < 1e-15);
    }
}
