//! Pure request/response handling for one Skybot conesearch call: building
//! the request URL and turning a response body into [`super::SkybotHit`]s.
//! Nothing here touches the network — only [`super::run`] does — so all of
//! it is covered directly by the unit tests below.

use serde::Deserialize;

use super::sexagesimal::{parse_dec_dms_to_deg, parse_ra_hms_to_deg};
use super::SkybotHit;

const SKYBOT_CONESEARCH_URL: &str = "https://ssp.imcce.fr/webservices/skybot/api/conesearch.php";

/// Converts a Modified Julian Date (TT scale, as stored on
/// `lineage_page::observations_table::ObservationRow::mjd_tt`) to the Julian
/// Day Skybot's `-ep` parameter expects.
///
/// This treats the epoch as if it were UTC, ignoring the ~70 second TT/UTC
/// offset — negligible next to the arcsecond-scale search radii this
/// feature uses ([`super::MIN_RADIUS_ARCSEC`]..=[`super::MAX_RADIUS_ARCSEC`]).
pub fn mjd_tt_to_jd(mjd_tt: f64) -> f64 {
    mjd_tt + 2_400_000.5
}

/// Builds the Skybot conesearch request URL for one query point.
///
/// `-output=all` asks for every field Skybot can return (magnitude, class,
/// distances, SSODNet links); `-objFilter=110` restricts results to
/// asteroids and planets (comets are a separate, much rarer case not worth
/// the extra noise here); `-observer=500` is the geocenter, matching the
/// epoch precision this feature needs (no per-station light-time
/// correction).
pub fn conesearch_url(point: &super::SkybotQueryPoint, radius_arcsec: f64) -> String {
    format!(
        "{SKYBOT_CONESEARCH_URL}?-ep={:.6}&-ra={:.8}&-dec={:.8}&-rs={:.2}&-mime=json&-output=all&-observer=500&-objFilter=110&-refsys=EQJ2000&-from=fink-fat",
        mjd_tt_to_jd(point.mjd_tt),
        point.ra_deg,
        point.dec_deg,
        radius_arcsec,
    )
}

#[derive(Deserialize)]
struct RawSsodnetLinks {
    #[serde(default)]
    quaero: Option<String>,
    #[serde(default)]
    ssocard: Option<String>,
}

/// One row of Skybot's `-mime=json&-output=all` conesearch response.
#[derive(Deserialize)]
struct RawSkybotRow {
    #[serde(rename = "Name")]
    name: String,
    #[serde(rename = "Class")]
    class: String,
    #[serde(rename = "RA (hour)")]
    ra_hms: String,
    #[serde(rename = "DEC (deg)")]
    dec_dms: String,
    #[serde(rename = "VMag (mag)", default)]
    vmag: Option<f64>,
    #[serde(rename = "Err (arcsec)", default)]
    err_arcsec: Option<f64>,
    #[serde(rename = "dg (ua)", default)]
    dg_au: Option<f64>,
    #[serde(rename = "dh (ua)", default)]
    dh_au: Option<f64>,
    #[serde(default)]
    ssodnet: Option<RawSsodnetLinks>,
}

/// Skybot's JSON conesearch response, as either a bare array or (some
/// deployments/versions) an array wrapped in `{"data": [...]}`. Accepting
/// both shapes here, rather than assuming one, keeps a response-format
/// change from turning "no objects found" into a hard parse error.
#[derive(Deserialize)]
#[serde(untagged)]
enum RawConesearchResponse {
    Rows(Vec<RawSkybotRow>),
    Wrapped { data: Vec<RawSkybotRow> },
}

impl RawConesearchResponse {
    fn into_rows(self) -> Vec<RawSkybotRow> {
        match self {
            RawConesearchResponse::Rows(rows) => rows,
            RawConesearchResponse::Wrapped { data } => data,
        }
    }
}

/// Converts one raw response row into a [`SkybotHit`], tagging it with
/// `source_index`. Returns `None` if the row's RA/Dec can't be parsed —
/// dropped rather than failing the whole response, since one malformed row
/// shouldn't discard every other hit found at the same point.
fn raw_row_to_hit(row: RawSkybotRow, source_index: usize) -> Option<SkybotHit> {
    let ra_deg = parse_ra_hms_to_deg(&row.ra_hms)?;
    let dec_deg = parse_dec_dms_to_deg(&row.dec_dms)?;
    let ssodnet_url = row.ssodnet.and_then(|links| links.ssocard.or(links.quaero));

    Some(SkybotHit {
        source_index,
        name: row.name,
        class: row.class,
        ra_deg,
        dec_deg,
        vmag: row.vmag,
        err_arcsec: row.err_arcsec,
        geocentric_distance_au: row.dg_au,
        heliocentric_distance_au: row.dh_au,
        ssodnet_url,
    })
}

/// Parses one Skybot conesearch JSON response body into hits, tagging each
/// with `source_index` (the query point it came from).
///
/// Skybot answers with an empty (204) body rather than `[]` when nothing is
/// found near a point, so an empty/whitespace-only body is zero hits, not a
/// parse error — same as an empty array. Only a non-empty body that isn't
/// valid JSON in either accepted shape returns `Err`.
pub fn parse_conesearch_response(
    body: &str,
    source_index: usize,
) -> Result<Vec<SkybotHit>, String> {
    if body.trim().is_empty() {
        return Ok(Vec::new());
    }

    let response: RawConesearchResponse =
        serde_json::from_str(body).map_err(|e| format!("failed to parse Skybot response: {e}"))?;
    Ok(response
        .into_rows()
        .into_iter()
        .filter_map(|row| raw_row_to_hit(row, source_index))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skybot_search::SkybotQueryPoint;

    #[test]
    fn mjd_tt_to_jd_adds_the_standard_offset() {
        assert!((mjd_tt_to_jd(60000.0) - 2_460_000.5).abs() < 1e-9);
    }

    #[test]
    fn conesearch_url_embeds_point_and_radius() {
        let point = SkybotQueryPoint {
            source_index: 0,
            ra_deg: 148.67,
            dec_deg: 16.3838,
            mjd_tt: 60000.0,
        };
        let url = conesearch_url(&point, 10.0);
        assert!(url.starts_with(SKYBOT_CONESEARCH_URL));
        assert!(url.contains("-ra=148.67000000"));
        assert!(url.contains("-dec=16.38380000"));
        assert!(url.contains("-rs=10.00"));
        assert!(url.contains("-ep=2460000.500000"));
        assert!(url.contains("-mime=json"));
    }

    const SAMPLE_ROW: &str = r#"{
        "Name": "(1) Ceres",
        "Class": "MB>Middle",
        "RA (hour)": "09:52:12.34",
        "DEC (deg)": "+16:23:01.2",
        "VMag (mag)": 8.7,
        "Err (arcsec)": 0.1,
        "dg (ua)": 2.5,
        "dh (ua)": 2.9,
        "ssodnet": {"quaero": "https://example.org/quaero/Ceres", "ssocard": "https://example.org/ssocard/Ceres"}
    }"#;

    #[test]
    fn parses_bare_array_response() {
        let body = format!("[{SAMPLE_ROW}]");
        let hits = parse_conesearch_response(&body, 3).unwrap();
        assert_eq!(hits.len(), 1);
        let hit = &hits[0];
        assert_eq!(hit.source_index, 3);
        assert_eq!(hit.name, "(1) Ceres");
        assert_eq!(hit.vmag, Some(8.7));
        assert_eq!(
            hit.ssodnet_url.as_deref(),
            Some("https://example.org/ssocard/Ceres")
        );
        assert!((hit.ra_deg - 148.051_416_666_666_66).abs() < 1e-6);
    }

    #[test]
    fn parses_wrapped_data_response() {
        let body = format!(r#"{{"data": [{SAMPLE_ROW}]}}"#);
        let hits = parse_conesearch_response(&body, 0).unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn falls_back_to_quaero_link_when_ssocard_is_absent() {
        let body = r#"[{
            "Name": "(4) Vesta",
            "Class": "MB>Middle",
            "RA (hour)": "09:52:12.34",
            "DEC (deg)": "+16:23:01.2",
            "ssodnet": {"quaero": "https://example.org/quaero/Vesta"}
        }]"#;
        let hits = parse_conesearch_response(body, 0).unwrap();
        assert_eq!(
            hits[0].ssodnet_url.as_deref(),
            Some("https://example.org/quaero/Vesta")
        );
    }

    #[test]
    fn empty_array_is_zero_hits_not_an_error() {
        let hits = parse_conesearch_response("[]", 0).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn empty_body_is_zero_hits_not_an_error() {
        // Skybot answers 204 (no body) rather than `[]` when nothing matches.
        let hits = parse_conesearch_response("", 0).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn drops_rows_with_unparseable_coordinates_instead_of_failing() {
        let body = r#"[{
            "Name": "Bad row",
            "Class": "MB>Middle",
            "RA (hour)": "not-a-time",
            "DEC (deg)": "+16:23:01.2"
        }]"#;
        let hits = parse_conesearch_response(body, 0).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn invalid_json_is_an_error() {
        assert!(parse_conesearch_response("not json", 0).is_err());
    }
}
