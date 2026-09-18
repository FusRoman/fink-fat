//! Pure request/response handling for one Skybot conesearch call: building
//! the request URL and turning a response body into [`super::SkybotHit`]s.
//! Nothing here touches the network — only [`super::run`] does — so all of
//! it is covered directly by the unit tests below.

use photom::coordinates::equatorial::EquCoord;
use serde::Deserialize;

use super::sexagesimal::{parse_dec_dms_to_deg, parse_ra_hms_to_deg};
use super::{SkybotHit, SkybotQueryPoint};

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
pub fn conesearch_url(point: &SkybotQueryPoint, radius_arcsec: f64) -> String {
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
///
/// Field names/formats here were confirmed against a live response rather
/// than Skybot's (sparse) published docs, e.g.:
/// `{"Name":"2015 DJ284","RA (hms)":"09 57 19.2502","DEC (dms)":"+00 56
/// 6.372", ...}`.
#[derive(Deserialize)]
struct RawSkybotRow {
    #[serde(rename = "Name")]
    name: String,
    #[serde(rename = "Class")]
    class: String,
    #[serde(rename = "RA (hms)")]
    ra_hms: String,
    #[serde(rename = "DEC (dms)")]
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

/// Great-circle separation between two sky positions, in arcseconds —
/// wraps `photom`'s Vincenty [`EquCoord::angular_separation`] (numerically
/// stable at both very small and near-antipodal separations, unlike a naive
/// haversine), converting its radians result to the arcsecond scale this
/// feature's search radii use. Astrometric errors are irrelevant to a plain
/// separation, so both positions are built with zero error.
fn angular_separation_arcsec(a_ra_deg: f64, a_dec_deg: f64, b_ra_deg: f64, b_dec_deg: f64) -> f64 {
    let a = EquCoord::from_degrees(a_ra_deg, 0.0, a_dec_deg, 0.0);
    let b = EquCoord::from_degrees(b_ra_deg, 0.0, b_dec_deg, 0.0);
    a.angular_separation(&b).to_degrees() * 3600.0
}

/// Converts one raw response row into a [`SkybotHit`], tagging it with
/// `query_point.source_index` and the separation between the hit's own
/// position and the real observation `query_point` was built from. Returns
/// `None` if the row's RA/Dec can't be parsed — dropped rather than failing
/// the whole response, since one malformed row shouldn't discard every other
/// hit found at the same point.
fn raw_row_to_hit(row: RawSkybotRow, query_point: &SkybotQueryPoint) -> Option<SkybotHit> {
    let ra_deg = parse_ra_hms_to_deg(&row.ra_hms)?;
    let dec_deg = parse_dec_dms_to_deg(&row.dec_dms)?;
    let ssodnet_url = row.ssodnet.and_then(|links| links.ssocard.or(links.quaero));
    let separation_arcsec =
        angular_separation_arcsec(query_point.ra_deg, query_point.dec_deg, ra_deg, dec_deg);

    Some(SkybotHit {
        source_index: query_point.source_index,
        name: row.name,
        class: row.class,
        ra_deg,
        dec_deg,
        vmag: row.vmag,
        err_arcsec: row.err_arcsec,
        geocentric_distance_au: row.dg_au,
        heliocentric_distance_au: row.dh_au,
        ssodnet_url,
        separation_arcsec,
    })
}

/// Parses one Skybot conesearch JSON response body into hits queried around
/// `query_point`, tagging each with its `source_index` and its separation
/// from that point (see [`raw_row_to_hit`]).
///
/// Skybot answers with an empty (204) body rather than `[]` when nothing is
/// found near a point, so an empty/whitespace-only body is zero hits, not a
/// parse error — same as an empty array. Only a non-empty body that isn't
/// valid JSON in either accepted shape returns `Err`.
pub fn parse_conesearch_response(
    body: &str,
    query_point: &SkybotQueryPoint,
) -> Result<Vec<SkybotHit>, String> {
    if body.trim().is_empty() {
        return Ok(Vec::new());
    }

    let response: RawConesearchResponse =
        serde_json::from_str(body).map_err(|e| format!("failed to parse Skybot response: {e}"))?;
    Ok(response
        .into_rows()
        .into_iter()
        .filter_map(|row| raw_row_to_hit(row, query_point))
        .collect())
}

/// Queries Skybot for one point and parses its response — the only network
/// call this feature makes. Takes a plain `&reqwest::Client` rather than
/// reaching for `crate::get_http_client()` itself, so it has no dependency
/// on the job registry, `tokio::spawn`, or dioxus: [`super::run`] is the only
/// caller in the app, but the live tests below call it directly too.
#[cfg(feature = "server")]
pub async fn fetch_conesearch_hits(
    client: &reqwest::Client,
    point: &SkybotQueryPoint,
    radius_arcsec: f64,
    timeout: std::time::Duration,
) -> Result<Vec<SkybotHit>, String> {
    let url = conesearch_url(point, radius_arcsec);

    let response = client
        .get(&url)
        .timeout(timeout)
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| format!("Skybot request failed: {e}"))?;

    let body = response
        .text()
        .await
        .map_err(|e| format!("failed to read Skybot response: {e}"))?;

    parse_conesearch_response(&body, point)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skybot_search::SkybotQueryPoint;

    /// A query point at `(ra_deg, dec_deg)`, epoch/index unused by the
    /// tests that only care about parsing/separation.
    fn point_at(ra_deg: f64, dec_deg: f64) -> SkybotQueryPoint {
        SkybotQueryPoint {
            source_index: 3,
            ra_deg,
            dec_deg,
            mjd_tt: 60000.0,
        }
    }

    #[test]
    fn mjd_tt_to_jd_adds_the_standard_offset() {
        assert!((mjd_tt_to_jd(60000.0) - 2_460_000.5).abs() < 1e-9);
    }

    #[test]
    fn angular_separation_arcsec_is_zero_for_the_same_point() {
        assert_eq!(
            angular_separation_arcsec(148.67, 16.3838, 148.67, 16.3838),
            0.0
        );
    }

    #[test]
    fn angular_separation_arcsec_matches_a_known_one_arcsecond_offset() {
        // 1 arcsecond of declination, at dec = 0 (where a degree of RA and a
        // degree of arc coincide), is exactly 1 arcsecond of great-circle
        // separation.
        let one_arcsec_deg = 1.0 / 3600.0;
        let sep = angular_separation_arcsec(10.0, 0.0, 10.0, one_arcsec_deg);
        assert!((sep - 1.0).abs() < 1e-6);
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

    /// A real response row, captured live from the conesearch endpoint for
    /// `2015 DJ284` — used verbatim as a regression fixture so a future
    /// field-name/format assumption drifting from the actual API is caught
    /// by a test rather than silently dropping every hit again (see the
    /// history of this file: the first implementation guessed `"RA
    /// (hour)"`/`"DEC (deg)"` with `:`-separated values from Skybot's
    /// sparse docs, which don't match the live API at all).
    const SAMPLE_ROW: &str = r#"{
        "Num": 628454,
        "Name": "2015 DJ284",
        "RA (hms)": "09 57 19.2502",
        "DEC (dms)": "+00 56 6.372",
        "Class": "MB>Outer",
        "VMag (mag)": 22.7,
        "Err (arcsec)": 0.021,
        "d (arcsec)": 2.298,
        "dRA (arcsec/h)": -2.1095,
        "dDEC (arcsec/h)": -6.034,
        "dg (ua)": 2.63312747642,
        "dh (ua)": 3.24631620478,
        "Phase (deg)": 15.12,
        "SunElong (deg)": 59.39,
        "position (au)": {"x": -2.280735361, "y": 2.269259072, "z": 0.442724195},
        "velocity (au/d)": {"x": -0.006408015, "y": -0.006491177, "z": -0.002347841},
        "ref_epoch": 2461030,
        "ssodnet": {
            "quaero": "https://api.ssodnet.imcce.fr/quaero/1/sso/search?q=\"2015 DJ284\" AND type:(Asteroid OR Dwarf Planet)",
            "ssocard": "https://ssp.imcce.fr/webservices/ssodnet/api/ssocard/2015_DJ284"
        }
    }"#;

    #[test]
    fn parses_bare_array_response() {
        let body = format!("[{SAMPLE_ROW}]");
        // The query point *is* the row's real position, so the separation
        // should come out at (near) zero.
        let query_point = point_at(149.330_209_166_666_67, 0.935_103_333_333_33);
        let hits = parse_conesearch_response(&body, &query_point).unwrap();
        assert_eq!(hits.len(), 1);
        let hit = &hits[0];
        assert_eq!(hit.source_index, 3);
        assert_eq!(hit.name, "2015 DJ284");
        assert_eq!(hit.vmag, Some(22.7));
        assert_eq!(
            hit.ssodnet_url.as_deref(),
            Some("https://ssp.imcce.fr/webservices/ssodnet/api/ssocard/2015_DJ284")
        );
        // 09h57m19.2502s -> (9 + 57/60 + 19.2502/3600) * 15
        assert!((hit.ra_deg - 149.330_209_166_666_67).abs() < 1e-6);
        // +00d56m6.372s
        assert!((hit.dec_deg - 0.935_103_333_333_33).abs() < 1e-6);
        assert!(hit.separation_arcsec < 1e-3);
    }

    #[test]
    fn parses_wrapped_data_response() {
        let body = format!(r#"{{"data": [{SAMPLE_ROW}]}}"#);
        let query_point = point_at(149.330_209_166_666_67, 0.935_103_333_333_33);
        let hits = parse_conesearch_response(&body, &query_point).unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn falls_back_to_quaero_link_when_ssocard_is_absent() {
        let body = r#"[{
            "Name": "(4) Vesta",
            "Class": "MB>Middle",
            "RA (hms)": "09 52 12.34",
            "DEC (dms)": "+16 23 01.2",
            "ssodnet": {"quaero": "https://example.org/quaero/Vesta"}
        }]"#;
        let hits = parse_conesearch_response(body, &point_at(0.0, 0.0)).unwrap();
        assert_eq!(
            hits[0].ssodnet_url.as_deref(),
            Some("https://example.org/quaero/Vesta")
        );
    }

    #[test]
    fn empty_array_is_zero_hits_not_an_error() {
        let hits = parse_conesearch_response("[]", &point_at(0.0, 0.0)).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn empty_body_is_zero_hits_not_an_error() {
        // Skybot answers 204 (no body) rather than `[]` when nothing matches.
        let hits = parse_conesearch_response("", &point_at(0.0, 0.0)).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn drops_rows_with_unparseable_coordinates_instead_of_failing() {
        let body = r#"[{
            "Name": "Bad row",
            "Class": "MB>Middle",
            "RA (hms)": "not-a-time",
            "DEC (dms)": "+16 23 01.2"
        }]"#;
        let hits = parse_conesearch_response(body, &point_at(0.0, 0.0)).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn invalid_json_is_an_error() {
        assert!(parse_conesearch_response("not json", &point_at(0.0, 0.0)).is_err());
    }
}

/// Live tests against the *real* Skybot service — the only thing standing
/// between fink-fat and a repeat of the bug that shipped this file: the
/// first implementation guessed field names/format from Skybot's docs
/// (`"RA (hour)"`/`"DEC (deg)"`, `:`-separated) instead of a live response,
/// which don't match the actual API at all (`"RA (hms)"`/`"DEC (dms)"`,
/// space-separated) — every response silently failed to parse, so the
/// feature always found zero hits and nothing above this layer (the job
/// orchestration, the UI) ever noticed. Mocking the HTTP response would
/// have made the exact same wrong assumption and passed anyway; only an
/// actual round trip through [`fetch_conesearch_hits`] against
/// `ssp.imcce.fr` catches that class of bug.
///
/// These need the `server` feature (for `reqwest`/`tokio`) and network
/// access, and are `#[ignore]`d so a normal `cargo test` (or CI without
/// network egress) doesn't depend on IMCCE's service being reachable.
/// Run them explicitly with:
///
/// ```text
/// cargo test -p fink-fat-explorer --features server --no-default-features \
///     -- --ignored skybot_search::parsing::live_tests
/// ```
#[cfg(all(test, feature = "server"))]
mod live_tests {
    use std::time::Duration;

    use super::*;
    use crate::skybot_search::SkybotQueryPoint;

    const LIVE_REQUEST_TIMEOUT: Duration = Duration::from_secs(20);

    /// `(1) 2015 DJ284`'s exact position/epoch, captured from a real
    /// conesearch response while investigating the parsing bug above (see
    /// `SAMPLE_ROW`'s doc comment) — a fixed *past* epoch, so Skybot's
    /// ephemeris for it is a deterministic calculation, not a "where is it
    /// now" query. As long as this well-catalogued asteroid's orbital
    /// elements aren't revised enough to move it outside a 10″ cone (not
    /// expected), this stays reproducible indefinitely.
    fn known_object_point() -> SkybotQueryPoint {
        SkybotQueryPoint {
            source_index: 0,
            ra_deg: 149.330_606_28,
            dec_deg: 0.935_603_11,
            mjd_tt: 61_033.278_741,
        }
    }

    #[tokio::test]
    #[ignore = "hits the live ssp.imcce.fr service — run explicitly, see module docs"]
    async fn live_conesearch_finds_a_known_object() {
        let client = reqwest::Client::new();
        let hits =
            fetch_conesearch_hits(&client, &known_object_point(), 10.0, LIVE_REQUEST_TIMEOUT)
                .await
                .expect("Skybot request/parse failed");

        let hit = hits
            .iter()
            .find(|h| h.name == "2015 DJ284")
            .unwrap_or_else(|| panic!("expected to find 2015 DJ284, got: {hits:?}"));

        // The query point *is* this object's real reported position, so the
        // parsed hit should land almost exactly on it.
        assert!((hit.ra_deg - 149.330_606_28).abs() < 1e-3);
        assert!((hit.dec_deg - 0.935_603_11).abs() < 1e-3);
        assert!(hit.vmag.is_some());
        assert!(hit.ssodnet_url.is_some());
        // Same reasoning: the two positions coincide, so the Vincenty
        // separation should be a fraction of an arcsecond.
        assert!(
            hit.separation_arcsec < 0.5,
            "expected a near-zero separation, got {}\"",
            hit.separation_arcsec
        );
    }

    #[tokio::test]
    #[ignore = "hits the live ssp.imcce.fr service — run explicitly, see module docs"]
    async fn live_conesearch_finds_nothing_near_the_celestial_pole() {
        // Minor planets orbit close to the ecliptic; a tiny cone right at
        // the celestial pole, at the same epoch as the known-object test
        // above, should come back empty — exercising the real "no hits"
        // (204/empty-body) response path end to end.
        let point = SkybotQueryPoint {
            source_index: 0,
            ra_deg: 0.0,
            dec_deg: 89.9,
            mjd_tt: 61_033.278_741,
        };
        let client = reqwest::Client::new();
        let hits = fetch_conesearch_hits(&client, &point, 1.0, LIVE_REQUEST_TIMEOUT)
            .await
            .expect("Skybot request/parse failed");

        assert!(
            hits.is_empty(),
            "expected no hits near the pole, got: {hits:?}"
        );
    }
}
