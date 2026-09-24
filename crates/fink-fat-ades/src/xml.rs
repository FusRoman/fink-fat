//! `quick-xml`/`serde` struct hierarchy mirroring MPC's `submit.xsd` (ADES
//! 2022), and the pure `AdesDocument` builder from a lineage's observations
//! and its user-confirmed header form.

use serde::Serialize;

use crate::error::AdesError;
use crate::model::{
    AdesHeaderInput, NightObservation, band_index_to_ades_band, mjd_tt_to_ades_obs_time,
    normalize_trk_sub,
};

/// Root `<ades version="2022">` element (`submit.xsd`'s `ADESType`). A
/// fink-fat export always contains exactly one `obsBlock`, since one lineage
/// is one submission batch.
#[derive(Debug, Serialize, PartialEq)]
#[serde(rename = "ades")]
pub struct AdesDocument {
    #[serde(rename = "@version")]
    pub version: String,
    #[serde(rename = "obsBlock")]
    pub obs_block: ObsBlock,
}

/// `submit.xsd`'s `ObsBlockType`.
#[derive(Debug, Serialize, PartialEq)]
pub struct ObsBlock {
    #[serde(rename = "obsContext")]
    pub obs_context: ObsContext,
    #[serde(rename = "obsData")]
    pub obs_data: ObsData,
}

/// `submit.xsd`'s `ObsContextType` (an `xsd:all`, so element order isn't
/// significant to MPC; kept in schema order here for readability).
#[derive(Debug, Serialize, PartialEq)]
pub struct ObsContext {
    pub observatory: Observatory,
    pub submitter: Submitter,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observers: Option<Names>,
    /// Required by the schema (`ObsContextType` has no `minOccurs="0"` on
    /// `measurers`, unlike `observers`).
    pub measurers: Names,
    pub telescope: Telescope,
}

/// `submit.xsd`'s `NamesType`: a wrapper element around repeated `<name>`
/// children, not a bare list of strings directly under the parent element.
#[derive(Debug, Serialize, PartialEq)]
pub struct Names {
    pub name: Vec<String>,
}

/// `submit.xsd`'s `ObservatoryType`.
#[derive(Debug, Serialize, PartialEq)]
pub struct Observatory {
    #[serde(rename = "mpcCode")]
    pub mpc_code: String,
}

/// `submit.xsd`'s `SubmitterType`.
#[derive(Debug, Serialize, PartialEq)]
pub struct Submitter {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub institution: Option<String>,
}

/// `submit.xsd`'s `TelescopeType`. `design`/`aperture`/`detector` are
/// required; `aperture` is numeric (`PosDecimalTypeW6`) but modeled as a
/// pre-formatted `String`, same rationale as `ra`/`dec` on
/// [`OpticalObservation`].
#[derive(Debug, Serialize, PartialEq)]
pub struct Telescope {
    pub design: String,
    pub aperture: String,
    pub detector: String,
}

/// `submit.xsd`'s `ObsDataType`, restricted to the `optical` observation
/// kind (the only one fink-fat produces).
#[derive(Debug, Serialize, PartialEq)]
pub struct ObsData {
    pub optical: Vec<OpticalObservation>,
}

/// `submit.xsd`'s `OpticalType`. `mode`/`stn`/`astCat` are elements of this
/// type (repeated per observation), not of `ObsContextType`.
#[derive(Debug, Serialize, PartialEq)]
pub struct OpticalObservation {
    #[serde(rename = "trkSub")]
    pub trk_sub: String,
    pub mode: String,
    pub stn: String,
    #[serde(rename = "obsTime")]
    pub obs_time: String,
    pub ra: String,
    pub dec: String,
    #[serde(rename = "astCat")]
    pub ast_cat: String,
    pub mag: String,
    pub band: String,
}

/// Format a right ascension in degrees as the decimal string `submit.xsd`'s
/// `RAType` pattern expects (`([1-3][0-9]{2}|[1-9]?[0-9])?(\.[0-9]{0,9})?`,
/// i.e. no leading zero on the integer part). `format!("{:.6}", _)` never
/// zero-pads the integer part in Rust, so this is a thin, separately-tested
/// wrapper rather than a raw `format!` at every call site.
fn format_ra_deg(ra_deg: f64) -> String {
    format!("{ra_deg:.6}")
}

/// Format a declination in degrees as the decimal string `submit.xsd`'s
/// `DeclinationType` pattern expects (`[+\-]?([1-9]?[0-9])?(\.[0123456789]{0,9})?`).
fn format_dec_deg(dec_deg: f64) -> String {
    format!("{dec_deg:.6}")
}

/// Build an `AdesDocument` (one `obsBlock`, one lineage per call) from a
/// lineage's surviving observations (after singleton-night removal — see
/// [`crate::model::remove_singleton_nights`]) and the user-confirmed header
/// form. Pure: no I/O, no XML text produced yet — see
/// [`ades_document_to_xml`].
///
/// # Arguments
/// * `lineage_designation` — the lineage this document is for.
/// * `observations` — the observations to include (post singleton-night
///   removal).
/// * `header` — the submitter/telescope header fields.
///
/// # Return
/// The built [`AdesDocument`].
///
/// # Errors
/// Returns [`AdesError::NoObservations`] if `observations` is empty, and any
/// [`AdesError`] `normalize_trk_sub`/`band_index_to_ades_band`/
/// `mjd_tt_to_ades_obs_time` produce for an individual observation.
pub fn build_ades_document(
    lineage_designation: &str,
    observations: &[NightObservation],
    header: &AdesHeaderInput,
) -> Result<AdesDocument, AdesError> {
    if observations.is_empty() {
        return Err(AdesError::NoObservations {
            lineage_designation: lineage_designation.to_string(),
        });
    }

    let trk_sub = normalize_trk_sub(lineage_designation)?.as_str().to_string();

    let optical = observations
        .iter()
        .map(|night_obs| {
            let obs = &night_obs.observation;
            let band = band_index_to_ades_band(obs.filter)?;
            let obs_time = mjd_tt_to_ades_obs_time(obs.mjd_tt)?;
            Ok(OpticalObservation {
                trk_sub: trk_sub.clone(),
                mode: header.mode.clone(),
                stn: obs.mpc_code_obs.clone(),
                obs_time,
                ra: format_ra_deg(obs.ra.to_degrees()),
                dec: format_dec_deg(obs.dec.to_degrees()),
                ast_cat: header.ast_cat.clone(),
                mag: format!("{:.2}", obs.magnitude),
                band: band.to_string(),
            })
        })
        .collect::<Result<Vec<_>, AdesError>>()?;

    let observers = if header.observers.iter().any(|o| !o.trim().is_empty()) {
        Some(Names {
            name: header
                .observers
                .iter()
                .filter(|o| !o.trim().is_empty())
                .cloned()
                .collect(),
        })
    } else {
        None
    };

    Ok(AdesDocument {
        version: "2022".to_string(),
        obs_block: ObsBlock {
            obs_context: ObsContext {
                observatory: Observatory {
                    mpc_code: optical.first().map(|o| o.stn.clone()).unwrap_or_default(),
                },
                submitter: Submitter {
                    name: header.submitter_name.clone(),
                    institution: header.submitter_institution.clone(),
                },
                observers,
                measurers: Names {
                    name: header
                        .measurers
                        .iter()
                        .filter(|m| !m.trim().is_empty())
                        .cloned()
                        .collect(),
                },
                telescope: Telescope {
                    design: header.telescope_design.clone(),
                    aperture: header.telescope_aperture.clone(),
                    detector: header.telescope_detector.clone(),
                },
            },
            obs_data: ObsData { optical },
        },
    })
}

/// Serialize an `AdesDocument` to a UTF-8 XML string (with XML declaration).
///
/// # Arguments
/// * `doc` — the document to serialize.
///
/// # Return
/// The XML text, including the `<?xml ...?>` declaration.
///
/// # Errors
/// Returns [`AdesError::XmlSerialize`] if `quick_xml` itself fails (should
/// not happen for a well-formed `AdesDocument` produced by
/// [`build_ades_document`], but is not `unwrap`-safe per repo convention).
pub fn ades_document_to_xml(doc: &AdesDocument) -> Result<String, AdesError> {
    let body = quick_xml::se::to_string(doc)?;
    Ok(format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n{body}\n"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ObservationRow;

    fn header() -> AdesHeaderInput {
        AdesHeaderInput {
            submitter_name: "Jane Doe".to_string(),
            submitter_institution: None,
            observers: vec![],
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

    fn night_obs(id: i64, ra_deg: f64, dec_deg: f64) -> NightObservation {
        NightObservation {
            night_id: 1,
            observation: ObservationRow {
                id,
                object_id: format!("obj{id}"),
                position: 0,
                mjd_tt: 60000.0,
                ra: ra_deg.to_radians(),
                ra_err: 0.0,
                dec: dec_deg.to_radians(),
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
    fn build_ades_document_rejects_empty_observations() {
        let err = build_ades_document("FF2024AB", &[], &header()).unwrap_err();
        assert!(matches!(err, AdesError::NoObservations { .. }));
    }

    #[test]
    fn build_ades_document_propagates_unknown_band() {
        let mut obs = night_obs(1, 1.0, 1.0);
        obs.observation.filter = 42;
        let err = build_ades_document("FF2024AB", &[obs], &header()).unwrap_err();
        assert!(matches!(err, AdesError::UnknownBand { filter: 42 }));
    }

    #[test]
    fn build_ades_document_produces_one_optical_record_per_observation() {
        let doc = build_ades_document(
            "FF2024AB",
            &[night_obs(1, 1.0, 1.0), night_obs(2, 2.0, 2.0)],
            &header(),
        )
        .unwrap();
        assert_eq!(doc.obs_block.obs_data.optical.len(), 2);
        assert_eq!(doc.obs_block.obs_data.optical[0].trk_sub, "FF2024AB");
        assert_eq!(doc.obs_block.obs_context.measurers.name, vec!["Jane Doe"]);
        assert!(doc.obs_block.obs_context.observers.is_none());
    }

    #[test]
    fn ades_document_to_xml_round_trips_expected_shape() {
        let doc = build_ades_document(
            "FF2024AB",
            &[night_obs(1, 123.456789, -12.345678)],
            &header(),
        )
        .unwrap();
        let xml = ades_document_to_xml(&doc).unwrap();
        assert!(xml.starts_with("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"));
        assert!(xml.contains("<ades version=\"2022\">"));
        assert!(xml.contains("<trkSub>FF2024AB</trkSub>"));
        assert!(xml.contains("<stn>I41</stn>"));
        assert!(xml.contains("<ra>123.456789</ra>"));
        assert!(xml.contains("<dec>-12.345678</dec>"));
        assert!(xml.contains("<measurers><name>Jane Doe</name></measurers>"));
        assert!(!xml.contains("<observers>"));
    }

    #[test]
    fn format_ra_deg_has_no_leading_zero_on_boundary_values() {
        assert_eq!(format_ra_deg(0.0), "0.000000");
        assert_eq!(format_ra_deg(359.999999), "359.999999");
    }

    #[test]
    fn format_dec_deg_handles_signed_boundary_values() {
        assert_eq!(format_dec_deg(90.0), "90.000000");
        assert_eq!(format_dec_deg(-90.0), "-90.000000");
    }
}
