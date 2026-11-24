# fink-fat-ades

ADES (Astrometry Data Exchange Standard) and Minor Planet Center (MPC) logic
for Fink-FAT: turning a tracked lineage into an ADES submission, checking it,
and interpreting what the MPC answers. It is shared by the `fink-fat submit`
CLI and by [`fink-fat-explorer`](../fink-fat-explorer/README.md) (ADES export
of the lineage page, submission dashboard), so both always agree on what a
valid document and an eligible lineage are.

The crate does no I/O: no Postgres, no HTTP, no filesystem, no async runtime
(the only side effect is reading the clock for today's date in
`format_epoch`). Each consumer fetches the data and performs the HTTP calls
itself, and calls this crate to shape requests, parse responses and decide
eligibility. Everything is therefore unit tested without a database or a
network.

## Workflow

From the observations of a lineage's best branch to an MPC verdict:

1. **Eligibility**: `quality_tier::assign_quality_tier` classifies the lineage
   (`WellSampledDiscovery`, `Discovery`, `WellSampledIdentified`,
   `Identified`, `Unconstrained`, …). Only the discovery tiers are submitted.
2. **Singleton nights**: `model::remove_singleton_nights` drops every
   observation alone in its night. The MPC rejects a whole batch that contains
   one.
3. **Local schema check**: `schema_validation::check_local_schema_violations`
   re-implements the constraints of the ADES `submit.xsd` and returns every
   violation, before anything is sent.
4. **Document**: `xml::build_ades_document` builds the ADES document (version
   2022, `trkSub` derived from the lineage designation), and
   `xml::ades_document_to_xml` serialises it.
   `model::check_submission_recommendations` adds non-blocking advisories
   (e.g. too few observations per night).
5. **Submission**: the caller POSTs the XML to
   `mpc_submission::SubmitEndpoint::submit_url()` and parses the answer with
   `mpc_submission::parse_mpc_submission_response`.
6. **Verdict**: submission is asynchronous; the MPC first returns a
   submission ID only. The outcome comes later from the status endpoints,
   parsed by `submission_status_api` (acceptance of the batch) and `wamo`
   (per-observation status).

## Endpoints

| Purpose | Test | Production |
| --- | --- | --- |
| ADES XML submission | `https://www.minorplanetcenter.net/submit_xml_test` | `https://minorplanetcenter.net/submit_xml` |
| Submission status | `https://submit-test.minorplanetcenter.net/submission_status/query/` (HTML page) | `https://data.minorplanetcenter.net/api/submission-status` (JSON API) |
| Per-observation status (WAMO) | — | `https://data.minorplanetcenter.net/api/wamo` |

`SubmitEndpoint::Test` is the default everywhere: a production submission is
irreversible and must be chosen explicitly (`fink-fat submit --endpoint
production`).

## Example

```rust
use fink_fat_ades::{
    model::{remove_singleton_nights, NightObservation},
    schema_validation::check_local_schema_violations,
    submitter_config::SubmitterConfig,
    xml::{ades_document_to_xml, build_ades_document},
};

fn to_ades_xml(
    lineage: &str,
    observations: &[NightObservation],
    submitter: SubmitterConfig,
    today: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let (kept, _singletons) = remove_singleton_nights(observations);
    let header = submitter.into_header(lineage, today);

    let violations = check_local_schema_violations(lineage, &kept, &header);
    if !violations.is_empty() {
        return Err(violations.join("; ").into());
    }

    let document = build_ades_document(lineage, &kept, &header)?;
    Ok(ades_document_to_xml(&document)?)
}
```

## Modules

| Module | Role |
| --- | --- |
| `model` | Input types (`NightObservation`, `AdesHeaderInput`), `trkSub` normalisation, band mapping, epoch conversion, singleton-night filtering, submission advisories. |
| `xml` | `quick-xml` / `serde` structs mirroring `submit.xsd`, the document builder and the serialiser. |
| `xml_format` | Pretty-printing of ADES XML for display. |
| `schema_validation` | Hand-written re-implementation of the `submit.xsd` constraints (the XSD itself is not vendored). |
| `mpc_submission` | `SubmitEndpoint`, request shaping and response parsing for `submit_xml` / `submit_xml_test`, and the test status page. |
| `submission_status_api` | Response parsing for the JSON Submission Status API. |
| `wamo` | Request body and response parsing for the WAMO per-observation status API. |
| `quality_tier` | The `QualityTier` cascade and the SQL query text it needs, shared by the CLI eligibility gate and the explorer homepage badge. |
| `submitter_config` | `SubmitterConfig`: the submitter and telescope identity, read from `fink-fat submit --submitter-config` and produced by the explorer form. |
| `format_epoch`, `lsst_band` | Formatting helpers used by `model`. |
| `error` | `AdesError`, returned by every fallible function. |

## References

- ADES `submit.xsd`:
  <https://raw.githubusercontent.com/IAU-ADES/ADES-Master/master/xsd/submit.xsd>
- MPC documentation: <https://docs.minorplanetcenter.net>

## Tests

The tests live next to the code (`#[cfg(test)]` modules), with recorded MPC
responses as fixtures:

```bash
cargo test -p fink-fat-ades
```
