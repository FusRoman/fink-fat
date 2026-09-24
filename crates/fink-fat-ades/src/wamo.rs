//! Request-shaping and response-parsing for MPC's **WAMO** API — the
//! fine-grained, per-observation status lookup, complementary to
//! [`crate::submission_status_api`]'s coarse submission-level ingest check.
//!
//! Live-probed (read-only `GET`, no submission involved) while implementing
//! this module, since WAMO's documented shape ("a JSON array of
//! identifiers, up to ~50,000 per call") turned out to need verification:
//! the request is a `GET` whose body is a bare JSON array of identifier
//! strings (not an object), and the response has three top-level keys —
//! `found` (array of single-key `{identifier: [observation, ...]}` objects),
//! `malformed` (array of `[identifier, reason]` pairs), and `not_found`
//! (array of bare identifier strings). All three fixtures below are real,
//! captured live: an obs ID from MPC's own docs example (`found`), a
//! well-formed-looking but nonexistent submission block ID (`not_found`),
//! and a plain designation string, which WAMO's grammar doesn't accept as
//! an identifier at all (`malformed`).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::AdesError;

/// MPC's WAMO API endpoint.
pub const MPC_WAMO_URL: &str = "https://data.minorplanetcenter.net/api/wamo";

/// WAMO's documented per-call identifier limit ("up to ~50,000").
pub const WAMO_MAX_IDENTIFIERS: usize = 50_000;

/// One way to identify an observation/submission to WAMO — the four forms
/// its docs list. Typed rather than a bare `String` so a caller can't
/// accidentally send a malformed identifier string built by hand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WamoIdentifier {
    /// A tracklet submission ID and station code, e.g. `"5T0D452 703"`.
    TrackletStation {
        tracklet_submission_id: String,
        station_code: String,
    },
    /// An MPC observation ID, e.g. `"L4eBVG000000CfiO010000A9a"`.
    ObservationId(String),
    /// A full 80/160-column MPC 1992-format observation string, exactly as
    /// originally submitted.
    Mpc80(String),
    /// A submission block ID, e.g.
    /// `"2024-05-02T21:03:35.001_0000FzZw_01"`.
    SubmissionBlockId(String),
}

impl WamoIdentifier {
    /// The exact string WAMO expects for this identifier in its request
    /// array.
    pub fn as_query_string(&self) -> String {
        match self {
            Self::TrackletStation {
                tracklet_submission_id,
                station_code,
            } => format!("{tracklet_submission_id} {station_code}"),
            Self::ObservationId(id) => id.clone(),
            Self::Mpc80(line) => line.clone(),
            Self::SubmissionBlockId(id) => id.clone(),
        }
    }
}

/// Build the JSON request body for a WAMO call: a bare array of identifier
/// query strings, ready to send as the body of a `GET`
/// (`reqwest::RequestBuilder::json(&body)` works on a `GET` request builder
/// the same way as on `POST`).
///
/// # Arguments
/// * `identifiers` — the identifiers to query, in any of WAMO's four forms.
///
/// # Return
/// The identifiers' query strings, in the same order.
///
/// # Errors
/// Returns [`AdesError::WamoTooManyIdentifiers`] if `identifiers` exceeds
/// [`WAMO_MAX_IDENTIFIERS`].
pub fn build_wamo_request_body(identifiers: &[WamoIdentifier]) -> Result<Vec<String>, AdesError> {
    if identifiers.len() > WAMO_MAX_IDENTIFIERS {
        return Err(AdesError::WamoTooManyIdentifiers {
            count: identifiers.len(),
            limit: WAMO_MAX_IDENTIFIERS,
        });
    }
    Ok(identifiers
        .iter()
        .map(WamoIdentifier::as_query_string)
        .collect())
}

/// One observation WAMO found for a queried identifier. Fields beyond
/// `input_type`/`obsid`/`status`/`status_decoded` are `Option` defensively:
/// WAMO's docs note astrometry is suppressed for unpublished observations,
/// so a not-yet-published hit may carry fewer populated fields than the
/// published example this type was modeled from.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WamoObservation {
    pub iau_desig: Option<String>,
    pub input_type: String,
    pub obs80: Option<String>,
    pub obsid: String,
    pub obssubid: Option<String>,
    #[serde(rename = "ref")]
    pub reference: Option<String>,
    pub status: String,
    pub status_decoded: String,
    pub submission_block_id: Option<String>,
    pub submission_id: Option<String>,
}

/// The full parsed shape of a WAMO response.
#[derive(Debug, Clone, PartialEq, Default, Deserialize, Serialize)]
pub struct WamoResponse {
    /// One entry per queried identifier that matched at least one
    /// observation, each a single-key map from that identifier's query
    /// string to its matching observations.
    #[serde(default)]
    pub found: Vec<HashMap<String, Vec<WamoObservation>>>,
    /// `(identifier, reason)` pairs for query strings WAMO's grammar could
    /// not parse as any of its four identifier forms at all.
    #[serde(default)]
    pub malformed: Vec<(String, String)>,
    /// Query strings that parsed as a valid identifier form but matched no
    /// observation.
    #[serde(default)]
    pub not_found: Vec<String>,
}

impl WamoResponse {
    /// Every observation WAMO returned for `identifier`, across every
    /// matching entry in [`Self::found`] (normally at most one, since WAMO
    /// echoes back exactly the query string it was given as the key).
    ///
    /// # Arguments
    /// * `identifier` — the exact query string originally sent (see
    ///   [`WamoIdentifier::as_query_string`]).
    ///
    /// # Return
    /// The matching observations, in response order.
    pub fn observations_for(&self, identifier: &str) -> Vec<&WamoObservation> {
        self.found
            .iter()
            .filter_map(|entry| entry.get(identifier))
            .flatten()
            .collect()
    }
}

/// Parse a `200 OK` response body from [`MPC_WAMO_URL`].
///
/// # Arguments
/// * `body` — the raw JSON response body text.
///
/// # Return
/// The parsed [`WamoResponse`].
///
/// # Errors
/// Returns [`AdesError::WamoResponseParse`] if `body` isn't valid JSON
/// matching the documented (and live-verified) response shape.
pub fn parse_wamo_response(body: &str) -> Result<WamoResponse, AdesError> {
    serde_json::from_str(body).map_err(|e| AdesError::WamoResponseParse(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracklet_station_identifier_formats_with_a_space() {
        let id = WamoIdentifier::TrackletStation {
            tracklet_submission_id: "5T0D452".to_string(),
            station_code: "703".to_string(),
        };
        assert_eq!(id.as_query_string(), "5T0D452 703");
    }

    #[test]
    fn build_wamo_request_body_preserves_order() {
        let identifiers = vec![
            WamoIdentifier::ObservationId("L4eBVG000000CfiO010000A9a".to_string()),
            WamoIdentifier::SubmissionBlockId("2024-05-02T21:03:35.001_0000FzZw_01".to_string()),
        ];
        let body = build_wamo_request_body(&identifiers).unwrap();
        assert_eq!(
            body,
            vec![
                "L4eBVG000000CfiO010000A9a".to_string(),
                "2024-05-02T21:03:35.001_0000FzZw_01".to_string(),
            ]
        );
    }

    #[test]
    fn build_wamo_request_body_rejects_too_many_identifiers() {
        let identifiers: Vec<WamoIdentifier> = (0..WAMO_MAX_IDENTIFIERS + 1)
            .map(|i| WamoIdentifier::ObservationId(i.to_string()))
            .collect();
        assert!(matches!(
            build_wamo_request_body(&identifiers),
            Err(AdesError::WamoTooManyIdentifiers { .. })
        ));
    }

    // Real fixtures captured live from `data.minorplanetcenter.net/api/wamo`
    // while implementing this module (see the module docs).

    const FOUND_FIXTURE: &str = r#"{
      "found": [
        {
          "L4eBVG000000CfiO010000A9a": [
            {
              "iau_desig": "380635",
              "input_type": "obsid",
              "obs80": "c0635         C2017 10 10.35217 01 24 30.94 +26 50 20.4          18.3 GU~2NGN703",
              "obsid": "L4eBVG000000CfiO010000A9a",
              "obssubid": null,
              "ref": "MPS   826083",
              "status": "P",
              "status_decoded": "c0635 has been identified as (380635) and published in MPS   826083.",
              "submission_block_id": "2017-10-10T12:17:02.000_0000CfiO_01",
              "submission_id": "2017-10-10T12:17:02.000_0000CfiO"
            }
          ]
        }
      ],
      "malformed": [],
      "not_found": []
    }"#;

    const NOT_FOUND_FIXTURE: &str = r#"{
      "found": [],
      "malformed": [],
      "not_found": ["2020-01-01T00:00:00.000_00000000_01"]
    }"#;

    const MALFORMED_FIXTURE: &str = r#"{
      "found": [],
      "malformed": [
        ["433", "Note that one or more of your lines including \"433\" could not be parsed as an obsid, trksub, submission_id, or submission_block_id."]
      ],
      "not_found": []
    }"#;

    #[test]
    fn parse_wamo_response_parses_a_found_observation() {
        let response = parse_wamo_response(FOUND_FIXTURE).unwrap();
        assert_eq!(response.found.len(), 1);
        assert!(response.malformed.is_empty());
        assert!(response.not_found.is_empty());

        let obs = response.observations_for("L4eBVG000000CfiO010000A9a");
        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0].iau_desig.as_deref(), Some("380635"));
        assert_eq!(obs[0].status, "P");
    }

    #[test]
    fn observations_for_returns_empty_for_an_unqueried_identifier() {
        let response = parse_wamo_response(FOUND_FIXTURE).unwrap();
        assert!(response.observations_for("not-in-the-response").is_empty());
    }

    #[test]
    fn parse_wamo_response_parses_not_found() {
        let response = parse_wamo_response(NOT_FOUND_FIXTURE).unwrap();
        assert!(response.found.is_empty());
        assert_eq!(
            response.not_found,
            vec!["2020-01-01T00:00:00.000_00000000_01".to_string()]
        );
    }

    #[test]
    fn parse_wamo_response_parses_malformed() {
        let response = parse_wamo_response(MALFORMED_FIXTURE).unwrap();
        assert_eq!(response.malformed.len(), 1);
        assert_eq!(response.malformed[0].0, "433");
        assert!(response.malformed[0].1.contains("could not be parsed"));
    }

    #[test]
    fn parse_wamo_response_rejects_malformed_json() {
        assert!(matches!(
            parse_wamo_response("not json"),
            Err(AdesError::WamoResponseParse(_))
        ));
    }
}
