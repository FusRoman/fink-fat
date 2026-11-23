//! Alert ingestion: quality cuts, conversion to a photom observation, and
//! deduplication — the pure step between the source and the night.
//!
//! # Two stages
//!
//! | Stage | Function | Needs | Rejection |
//! |-------|----------|-------|-----------|
//! | cuts + conversion | [`convert`] / [`Ingestor::convert`] | the alert alone | [`Rejection::Cut`], [`Rejection::Unconvertible`] |
//! | deduplication | [`Ingestor::accept`] | the set of ids accepted tonight | [`Rejection::Duplicate`] |
//!
//! The conversion itself is photom's ([`dia_source_to_observation`]: units,
//! `1/cos δ` RA error, TAI → TT, `visit` → night); this module only chains it
//! with the configured [`LsstQualityCuts`] (a *policy*: a cut alert is
//! perfectly convertible, we just choose not to track it).
//!
//! # Why deduplicate
//!
//! The alert stream is delivered **at least once**: a consumer restart
//! re-reads every alert it had not acknowledged, some of which it had
//! already processed. Inserting the same `diaSourceId` twice into a night
//! would be an error in the night's dataset, and counting it twice in the
//! tracker would be wrong anyway.
//!
//! The set is **per night**, which keeps it bounded: an alert can only be
//! redelivered into the night it belongs to (alerts of closed nights are
//! dropped as late before reaching it).
//!
//! # Why two stages
//!
//! The caller needs the alert's night — known only after conversion — to
//! decide whether the open night closes first. Accepting the id before that
//! decision would put the first alert of a new night into the *old* night's
//! set, which the close then clears: a redelivered copy of that alert would
//! slip through. Hence `convert`, then route, then `accept`.

use std::collections::HashSet;

use photom::io::lsst_alert::{
    CutReason, LsstAlertError, LsstConversionParams, LsstObservation, LsstQualityCuts,
    dia_source_to_observation,
};

use crate::record::AlertRecord;

/// Why an alert was not handed to the tracker. Not an error: rejections are
/// counted and logged, and the stream goes on.
#[derive(Debug, Clone, PartialEq)]
pub enum Rejection {
    /// The alert failed a science-quality cut.
    Cut(CutReason),
    /// The alert cannot be converted (bad flux, coordinates, band, visit…).
    Unconvertible(LsstAlertError),
    /// Tonight already accepted an alert with this `diaSourceId`.
    Duplicate,
}

/// Apply `cuts`, then convert `record` with `params` — the stateless
/// first stage (see the [module doc](self#two-stages)).
///
/// Cuts come first: they are cheaper, and a cut alert need not be
/// convertible.
///
/// # Arguments
/// * `record` – The alert.
/// * `cuts` – Quality thresholds (SNR, astrometric errors), each optional.
/// * `params` – Conversion parameters (astrometric floor).
///
/// # Returns
/// The photom observation, with its night and visit.
///
/// # Errors
/// [`Rejection::Cut`] or [`Rejection::Unconvertible`]; never
/// [`Rejection::Duplicate`] (see [`Ingestor`]).
pub fn convert(
    record: &AlertRecord,
    cuts: &LsstQualityCuts,
    params: &LsstConversionParams,
) -> Result<LsstObservation, Rejection> {
    let source = record.dia_source();
    cuts.check(&source).map_err(Rejection::Cut)?;
    dia_source_to_observation(&source, params).map_err(Rejection::Unconvertible)
}

/// Both ingestion stages, with the configuration and the per-night
/// deduplication state (see the [module doc](self)).
///
/// The driver keeps one `Ingestor` for its whole life and calls
/// [`Self::reset`] at every night close.
#[derive(Debug, Clone)]
pub struct Ingestor {
    /// Quality thresholds.
    cuts: LsstQualityCuts,
    /// Conversion parameters.
    params: LsstConversionParams,
    /// `diaSourceId`s accepted since the last [`Self::reset`].
    accepted: HashSet<i64>,
}

impl Ingestor {
    /// Ingestor applying `cuts` and converting with `params`, with an empty
    /// deduplication set.
    pub fn new(cuts: LsstQualityCuts, params: LsstConversionParams) -> Self {
        Self {
            cuts,
            params,
            accepted: HashSet::new(),
        }
    }

    /// First stage: apply the cuts and convert `record` (see [`convert`]).
    /// Does not touch the deduplication set, so it can be called before the
    /// alert's night is known to be the open one.
    ///
    /// # Errors
    /// [`Rejection::Cut`] or [`Rejection::Unconvertible`].
    pub fn convert(&self, record: &AlertRecord) -> Result<LsstObservation, Rejection> {
        convert(record, &self.cuts, &self.params)
    }

    /// Second stage: record `dia_source_id` as accepted tonight.
    ///
    /// Call it only once the alert is known to belong to the open night (see
    /// the [module doc](self#why-two-stages)) and was converted
    /// successfully, so that a rejected alert is never remembered.
    ///
    /// # Errors
    /// [`Rejection::Duplicate`] when tonight already accepted it.
    pub fn accept(&mut self, dia_source_id: i64) -> Result<(), Rejection> {
        if self.accepted.insert(dia_source_id) {
            Ok(())
        } else {
            Err(Rejection::Duplicate)
        }
    }

    /// [`Self::convert`] then [`Self::accept`], for an alert known to belong
    /// to tonight. A rejected alert is not remembered, so a redelivered copy
    /// is rejected again for the same reason rather than as a duplicate.
    ///
    /// # Errors
    /// The [`Rejection`] of the alert.
    pub fn ingest(&mut self, record: &AlertRecord) -> Result<LsstObservation, Rejection> {
        let observation = self.convert(record)?;
        self.accept(record.dia_source_id)?;
        Ok(observation)
    }

    /// Forget the accepted ids: the night closed, the next one starts with
    /// an empty set.
    pub fn reset(&mut self) {
        self.accepted.clear();
    }

    /// Number of alerts accepted since the last [`Self::reset`].
    pub fn n_accepted(&self) -> usize {
        self.accepted.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::tests::sample_record;

    fn ingestor(cuts: LsstQualityCuts) -> Ingestor {
        Ingestor::new(cuts, LsstConversionParams::default())
    }

    #[test]
    fn a_valid_alert_is_converted_with_its_night_and_visit() {
        let record = sample_record();
        let obs = ingestor(LsstQualityCuts::default())
            .ingest(&record)
            .unwrap();
        assert_eq!(obs.input.id, 1_234);
        assert_eq!(obs.visit, record.visit);
        assert_eq!(obs.night_id, record.night_id().unwrap());
    }

    #[test]
    fn a_low_snr_alert_is_cut() {
        let cuts = LsstQualityCuts {
            min_snr: Some(100.0),
            ..LsstQualityCuts::default()
        };
        // SNR = 5000 / 100 = 50.
        assert!(matches!(
            ingestor(cuts).ingest(&sample_record()),
            Err(Rejection::Cut(CutReason::LowSnr { .. }))
        ));
    }

    #[test]
    fn an_unconvertible_alert_is_rejected() {
        let record = AlertRecord {
            psf_flux_njy: -1.0,
            ..sample_record()
        };
        assert!(matches!(
            ingestor(LsstQualityCuts::default()).ingest(&record),
            Err(Rejection::Unconvertible(LsstAlertError::NonPositiveFlux(_)))
        ));
    }

    #[test]
    fn a_redelivered_alert_is_a_duplicate_until_reset() {
        let mut ingestor = ingestor(LsstQualityCuts::default());
        let record = sample_record();
        assert!(ingestor.ingest(&record).is_ok());
        assert_eq!(ingestor.ingest(&record).unwrap_err(), Rejection::Duplicate);
        assert_eq!(ingestor.n_accepted(), 1);

        ingestor.reset();
        assert!(ingestor.ingest(&record).is_ok());
    }

    #[test]
    fn a_rejected_alert_is_not_remembered() {
        let cuts = LsstQualityCuts {
            min_snr: Some(100.0),
            ..LsstQualityCuts::default()
        };
        let mut ingestor = ingestor(cuts);
        let record = sample_record();
        for _ in 0..2 {
            assert!(matches!(ingestor.ingest(&record), Err(Rejection::Cut(_))));
        }
        assert_eq!(ingestor.n_accepted(), 0);
    }
}
