//! Pure conversion of an accepted alert into its `observations` row.

use fink_fat_sql::rows::ObservationRow;
use photom::{io::lsst_alert::LsstObservation, observer::dataset::ObserverId, photometry::Filter};

/// The `observations` row of an accepted alert.
///
/// | Column | Value |
/// |--------|-------|
/// | `id` | `diaSourceId` |
/// | `night_id` | the alert's night: MJD of its `dayObs` (the batch pipeline stores a relative night index instead — do not mix both in one database) |
/// | `object_id` | `diaSourceId` as text, the convention of `test_exp/prep_lsst_eval_data.py` (Rubin has no persistent per-object id for unknown moving objects) |
/// | `magnitude`, `mag_err` | AB magnitude and its error, from photom's conversion |
/// | `filter` | photom's band code (`u`…`y` → `0`…`5`); `-1` for a label filter, which LSST alerts never have |
/// | `mpc_code_obs` | the observer's MPC code (`X05` for Rubin); empty if none |
/// | `ra`, `dec`, `ra_err`, `dec_err` | radians; `ra_err` is the error on the α coordinate (on-sky error / cos δ) |
/// | `mjd_tt` | exposure mid-point, MJD TT |
///
/// # Arguments
/// * `observation` – The alert as converted by ingestion.
pub fn observation_row(observation: &LsstObservation) -> ObservationRow {
    let input = &observation.input;
    let equ = &input.equ_coord;
    ObservationRow {
        id: input.id as i64,
        night_id: i64::from(observation.night_id.0),
        object_id: input.id.to_string(),
        magnitude: input.photometry.magnitude,
        mag_err: input.photometry.error,
        filter: match input.photometry.filter {
            Filter::Int(code) => i16::try_from(code).unwrap_or(-1),
            Filter::String(_) => -1,
        },
        mpc_code_obs: match &input.observer {
            Some(ObserverId::MpcCode(code)) => String::from_utf8_lossy(code).into_owned(),
            _ => String::new(),
        },
        ra: equ.ra,
        ra_err: equ.ra_error,
        dec: equ.dec,
        dec_err: equ.dec_error,
        mjd_tt: input.mjd_tt,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ingest::convert, record::tests::sample_record};
    use photom::io::lsst_alert::{LsstConversionParams, LsstQualityCuts};

    #[test]
    fn an_accepted_alert_maps_to_its_observation_row() {
        let record = sample_record();
        let observation = convert(
            &record,
            &LsstQualityCuts::default(),
            &LsstConversionParams::default(),
        )
        .unwrap();
        let row = observation_row(&observation);

        assert_eq!(row.id, record.dia_source_id);
        assert_eq!(row.object_id, record.dia_source_id.to_string());
        assert_eq!(row.night_id, i64::from(record.night_id().unwrap().0));
        assert_eq!(row.mpc_code_obs, "X05");
        assert_eq!(row.filter, 2, "band r");
        assert!((row.dec - record.dec_deg.to_radians()).abs() < 1e-12);
        assert!((row.ra - record.ra_deg.to_radians()).abs() < 1e-12);
        assert_eq!(row.mjd_tt, observation.input.mjd_tt);
        assert_eq!(row.magnitude, observation.input.photometry.magnitude);
    }
}
