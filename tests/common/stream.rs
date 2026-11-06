//! Synthetic Rubin alerts built from the `tests/data` nights, and engine
//! configurations storing in a temporary directory — shared by the tests
//! that drive `fink-fat-stream` end to end (`stream_resume.rs`,
//! `stream_sql.rs`).

use std::f64::consts::LN_10;

use camino::{Utf8Path, Utf8PathBuf};
use fink_fat_engine::engine_config::EngineConfig;
use fink_fat_stream::{AlertRecord, record::mjd_tai_to_utc};
use photom::{
    io::lsst_alert::{night::day_obs_to_night_id, units::TT_MINUS_TAI_DAYS},
    observation_dataset::{ObsDataset, observation::Observation},
    photometry::Filter,
};
use tempfile::TempDir;

use super::CONFIG_PATH;

/// Simulated delay between an exposure and its alerts' arrival.
pub const LATENCY_DAYS: f64 = 3.0 / 1_440.0;
/// `YYYYMMDD` of the civil date whose MJD is `mjd` (Hinnant's
/// `civil_from_days`).
pub fn day_obs_of(mjd: i64) -> i64 {
    let z = mjd - 40_587 + 719_468; // days since 0000-03-01
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    year * 10_000 + month * 100 + day
}

/// One synthetic Rubin alert per fixture observation (see the module doc).
pub fn synthetic_alerts(obs_dataset: &ObsDataset) -> Vec<AlertRecord> {
    let mut observations: Vec<&Observation> = obs_dataset.iter_observations().collect();
    observations.sort();

    let mut records = Vec::with_capacity(observations.len());
    let mut visit_of_epoch: Option<(u64, i64)> = None; // (epoch bits, visit)
    let mut seq_num = 0;
    for observation in observations {
        let midpoint_mjd_tai = observation.mjd_tt() - TT_MINUS_TAI_DAYS;
        let night = (midpoint_mjd_tai - 0.5).floor() as i64;
        let day_obs = day_obs_of(night);
        assert_eq!(
            i64::from(day_obs_to_night_id(day_obs as u32).unwrap().0),
            night,
            "day_obs_of must invert photom's night id"
        );

        let epoch_bits = observation.mjd_tt().to_bits();
        let visit = match visit_of_epoch {
            Some((bits, visit)) if bits == epoch_bits => visit,
            _ => {
                let previous_day_obs = visit_of_epoch.map(|(_, v)| v / 100_000);
                seq_num = if previous_day_obs == Some(day_obs) {
                    seq_num + 1
                } else {
                    1
                };
                day_obs * 100_000 + seq_num
            }
        };
        visit_of_epoch = Some((epoch_bits, visit));

        let equ = observation.equ_coord();
        let photometry = observation.photometry();
        let flux = 10f64.powf((31.4 - photometry.magnitude) / 2.5);
        records.push(AlertRecord {
            dia_source_id: *observation.id() as i64,
            visit,
            midpoint_mjd_tai,
            ra_deg: equ.ra.to_degrees(),
            dec_deg: equ.dec.to_degrees(),
            // The fixture's RA error is on the α coordinate; Rubin's is on sky.
            ra_err_deg: (equ.ra_error * equ.dec.cos()).to_degrees() as f32,
            dec_err_deg: equ.dec_error.to_degrees() as f32,
            psf_flux_njy: flux as f32,
            psf_flux_err_njy: (photometry.error * flux * LN_10 / 2.5) as f32,
            band: match photometry.filter {
                Filter::Int(1) => "g",
                _ => "r",
            }
            .to_string(),
            arrival_mjd: mjd_tai_to_utc(midpoint_mjd_tai) + LATENCY_DAYS,
        });
    }
    records
}

/// The test engine configuration, storing under `storage`.
pub fn engine_config(storage: &Utf8Path) -> EngineConfig {
    let yaml = std::fs::read_to_string(CONFIG_PATH).expect("read test config");
    assert!(yaml.contains("storage_path: ./storage\n"));
    let yaml = yaml.replace(
        "storage_path: ./storage\n",
        &format!("storage_path: {storage}\n"),
    );
    let path = storage.join("engine_config.yaml");
    std::fs::create_dir_all(storage).expect("create storage");
    std::fs::write(&path, yaml).expect("write engine config");
    EngineConfig::load_engine_config_validated(&path).expect("load engine config")
}

pub fn storage_dir(tmp: &TempDir, name: &str) -> Utf8PathBuf {
    Utf8PathBuf::from_path_buf(tmp.path().join(name)).expect("utf8 tempdir")
}
