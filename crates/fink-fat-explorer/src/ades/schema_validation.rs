//! Local, pure re-implementation of the `submit.xsd` constraints relevant to
//! an ADES optical-observation submission, read directly from
//! <https://raw.githubusercontent.com/IAU-ADES/ADES-Master/master/xsd/submit.xsd>.
//!
//! MPC's live `submit_xml_test` endpoint (see `mpc_submission.rs`) is
//! asynchronous — it only ever acknowledges a submission and emails the real
//! validation report later, so it cannot back a synchronous, blocking
//! green/red indicator. This module is that indicator instead: it is the
//! sole gate controlling whether the ADES download button is enabled.

use crate::ades::model::{
    band_index_to_ades_band, normalize_trk_sub, AdesHeaderInput, NightObservation,
};

/// `submit.xsd`'s `StationType`: 3–4 alphanumeric characters.
const STATION_LEN_RANGE: std::ops::RangeInclusive<usize> = 3..=4;
/// `submit.xsd`'s `CatType`: at most 8 characters, `[.A-Za-z0-9_]*`.
const AST_CAT_MAX_LEN: usize = 8;
/// `submit.xsd`'s `ModeType`: at most 3 alphanumeric characters.
const MODE_MAX_LEN: usize = 3;
/// `submit.xsd`'s `MagType` bounds.
const MAG_RANGE: std::ops::RangeInclusive<f64> = -5.0..=35.0;
/// `submit.xsd`'s `DeclinationType` bounds.
const DEC_RANGE: std::ops::RangeInclusive<f64> = -90.0..=90.0;

fn is_ast_cat_char(c: char) -> bool {
    c == '.' || c.is_ascii_alphanumeric() || c == '_'
}

/// Check every observation and header field against the `submit.xsd`
/// constraints relevant to an ADES optical-observation submission. Returns a
/// list of human-readable violations; an empty list means the document that
/// would be built from these inputs is locally conformant.
///
/// This is the sole gate controlling the download button: MPC's
/// `submit_xml_test` endpoint is only ever consulted (see
/// `mpc_submission.rs`) once this function returns no violations, and its
/// own outcome never re-opens or re-closes that gate.
pub fn check_local_schema_violations(
    lineage_designation: &str,
    observations: &[NightObservation],
    header: &AdesHeaderInput,
) -> Vec<String> {
    let mut violations = Vec::new();

    if let Err(err) = normalize_trk_sub(lineage_designation) {
        violations.push(format!("invalid trkSub: {err}"));
    }

    if observations.is_empty() {
        violations.push(
            "no observations remain to export (either the lineage has none, or every \
             observation was removed as belonging to a singleton night)"
                .to_string(),
        );
    }

    if header.submitter_name.trim().is_empty() {
        violations.push("submitter name is required".to_string());
    }
    if header.measurers.iter().all(|m| m.trim().is_empty()) {
        violations.push(
            "at least one measurer name is required (ADES obsContext.measurers is mandatory)"
                .to_string(),
        );
    }
    if header.telescope_design.trim().is_empty() {
        violations.push("telescope design is required".to_string());
    }
    if header.telescope_detector.trim().is_empty() {
        violations.push("telescope detector is required".to_string());
    }
    match header.telescope_aperture.trim().parse::<f64>() {
        Ok(v) if v > 0.0 => {}
        _ => violations.push(format!(
            "telescope aperture '{}' must be a positive number",
            header.telescope_aperture
        )),
    }
    if header.ack_message.trim().is_empty() {
        violations.push("an acknowledgment message (ack) is required by MPC".to_string());
    }
    if !header.ac2_email.contains('@') || !header.ac2_email.trim_end().ends_with(|c: char| c != '@')
    {
        violations.push(format!(
            "acknowledgment email address '{}' does not look valid",
            header.ac2_email
        ));
    }
    if header.mode.is_empty()
        || header.mode.len() > MODE_MAX_LEN
        || !header.mode.chars().all(|c| c.is_ascii_alphanumeric())
    {
        violations.push(format!(
            "mode '{}' must be 1-{MODE_MAX_LEN} alphanumeric characters",
            header.mode
        ));
    }
    if header.ast_cat.is_empty()
        || header.ast_cat.len() > AST_CAT_MAX_LEN
        || !header.ast_cat.chars().all(is_ast_cat_char)
    {
        violations.push(format!(
            "astCat '{}' must be 1-{AST_CAT_MAX_LEN} characters matching [.A-Za-z0-9_]",
            header.ast_cat
        ));
    }

    for night_obs in observations {
        let obs = &night_obs.observation;
        let ra_deg = obs.ra.to_degrees();
        let dec_deg = obs.dec.to_degrees();

        if !(0.0..360.0).contains(&ra_deg) {
            violations.push(format!(
                "observation {}: ra {ra_deg:.6} deg is out of the [0, 360) range",
                obs.id
            ));
        }
        if !DEC_RANGE.contains(&dec_deg) {
            violations.push(format!(
                "observation {}: dec {dec_deg:.6} deg is out of the [-90, 90] range",
                obs.id
            ));
        }
        if !MAG_RANGE.contains(&obs.magnitude) {
            violations.push(format!(
                "observation {}: magnitude {} is out of the [-5, 35] range",
                obs.id, obs.magnitude
            ));
        }
        if !STATION_LEN_RANGE.contains(&obs.mpc_code_obs.len())
            || !obs.mpc_code_obs.chars().all(|c| c.is_ascii_alphanumeric())
        {
            violations.push(format!(
                "observation {}: station code '{}' must be 3-4 alphanumeric characters",
                obs.id, obs.mpc_code_obs
            ));
        }
        if let Err(err) = band_index_to_ades_band(obs.filter) {
            violations.push(format!("observation {}: {err}", obs.id));
        }
    }

    violations
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lineage_page::observations_table::ObservationRow;

    fn valid_header() -> AdesHeaderInput {
        AdesHeaderInput {
            submitter_name: "Jane Doe".to_string(),
            submitter_institution: None,
            observers: vec!["Jane Doe".to_string()],
            measurers: vec!["Jane Doe".to_string()],
            telescope_design: "Reflector".to_string(),
            telescope_aperture: "1.2".to_string(),
            telescope_detector: "CCD".to_string(),
            ast_cat: "Gaia2".to_string(),
            mode: "CCD".to_string(),
            funding_source: None,
            ack_message: "fink-fat export".to_string(),
            ac2_email: "user@example.com".to_string(),
        }
    }

    fn valid_observation() -> NightObservation {
        NightObservation {
            night_id: 1,
            observation: ObservationRow {
                id: 1,
                object_id: "obj1".to_string(),
                position: 0,
                mjd_tt: 60000.0,
                ra: 1.5_f64.to_radians(),
                ra_err: 0.0,
                dec: -12.0_f64.to_radians(),
                dec_err: 0.0,
                magnitude: 19.5,
                mag_err: 0.1,
                filter: 2,
                mpc_code_obs: "I41".to_string(),
                night_id: 1,
            },
        }
    }

    #[test]
    fn valid_input_has_no_violations() {
        let violations =
            check_local_schema_violations("FF2024AB", &[valid_observation()], &valid_header());
        assert!(
            violations.is_empty(),
            "unexpected violations: {violations:?}"
        );
    }

    #[test]
    fn empty_observations_is_a_violation() {
        let violations = check_local_schema_violations("FF2024AB", &[], &valid_header());
        assert!(violations.iter().any(|v| v.contains("no observations")));
    }

    #[test]
    fn out_of_range_dec_is_a_violation() {
        let mut obs = valid_observation();
        obs.observation.dec = (-95.0_f64).to_radians();
        let violations = check_local_schema_violations("FF2024AB", &[obs], &valid_header());
        assert!(violations.iter().any(|v| v.contains("dec")));
    }

    #[test]
    fn out_of_range_magnitude_is_a_violation() {
        let mut obs = valid_observation();
        obs.observation.magnitude = 100.0;
        let violations = check_local_schema_violations("FF2024AB", &[obs], &valid_header());
        assert!(violations.iter().any(|v| v.contains("magnitude")));
    }

    #[test]
    fn short_station_code_is_a_violation() {
        let mut obs = valid_observation();
        obs.observation.mpc_code_obs = "AB".to_string();
        let violations = check_local_schema_violations("FF2024AB", &[obs], &valid_header());
        assert!(violations.iter().any(|v| v.contains("station code")));
    }

    #[test]
    fn oversized_ast_cat_is_a_violation() {
        let mut header = valid_header();
        header.ast_cat = "TooLongCatalogName".to_string();
        let violations = check_local_schema_violations("FF2024AB", &[valid_observation()], &header);
        assert!(violations.iter().any(|v| v.contains("astCat")));
    }

    #[test]
    fn empty_measurers_is_a_violation() {
        let mut header = valid_header();
        header.measurers = vec![];
        let violations = check_local_schema_violations("FF2024AB", &[valid_observation()], &header);
        assert!(violations.iter().any(|v| v.contains("measurer")));
    }

    #[test]
    fn empty_ack_or_ac2_is_a_violation() {
        let mut header = valid_header();
        header.ack_message = String::new();
        header.ac2_email = "not-an-email".to_string();
        let violations = check_local_schema_violations("FF2024AB", &[valid_observation()], &header);
        assert!(violations
            .iter()
            .any(|v| v.contains("acknowledgment message")));
        assert!(violations.iter().any(|v| v.contains("email")));
    }
}
