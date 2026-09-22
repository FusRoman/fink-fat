//! Export of a lineage's observations as an ADES (Astrometry Data Exchange
//! Standard) XML pre-submission file for the Minor Planet Center (MPC).
//!
//! The pipeline is: fetch the lineage's best branch's observations (with
//! their `night_id`) → remove observations belonging to a singleton night
//! (`model::remove_singleton_nights` — the MPC guide to astrometry states a
//! batch containing one is rejected in its entirety) → build the ADES
//! document (`xml::build_ades_document`) → validate it locally against
//! `submit.xsd`'s constraints (`schema_validation::check_local_schema_violations`,
//! the sole gate on the download button) → if locally valid, submit it to
//! MPC's `submit_xml_test` endpoint for an informational acknowledgement
//! (`mpc_submission` — **not** a validity verdict; that endpoint is
//! asynchronous and emails its real report).
//!
//! Module layout: `model.rs`, `schema_validation.rs`, and `mpc_submission.rs`
//! are pure and always compiled (no XML or network dependency — even
//! `mpc_submission.rs`, which only shapes a request and parses a response,
//! with the actual HTTP call left to `server_fns.rs`). `xml.rs` is
//! server-only, since it depends on the `quick-xml` crate, itself only
//! pulled in under the `server` feature. `server_fns.rs` (the `#[server]`
//! orchestration, the only module that performs I/O) is always compiled too
//! — like every other `#[server]` fn in this crate, its macro-generated
//! client stub must exist in wasm builds; only its function *body*
//! (server-only imports scoped locally inside it, per this crate's existing
//! convention — see `lineage_page/alert_cutouts.rs`) is conditionally real.

pub mod error;
pub mod model;
pub mod mpc_submission;
pub mod schema_validation;
pub mod server_fns;

#[cfg(feature = "server")]
pub mod xml;
