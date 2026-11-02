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
//! asynchronous).
//!
//! `model`, `schema_validation`, `xml`, `mpc_submission`, and `error` are
//! pure and live in the shared [`fink_fat_ades`] crate (also depended on by
//! the `fink-fat submit` CLI — see that crate's docs) — re-exported here
//! under their original module paths so every existing `crate::ades::model`,
//! `crate::ades::error`, etc. call site in this crate needed no changes.
//! [`server_fns`] (the `#[server]` orchestration, the only module that
//! performs I/O) is the one part of this feature that stays here, since it's
//! specific to this app's Postgres schema and Dioxus fullstack wiring.

pub use fink_fat_ades::{error, model, mpc_submission, schema_validation, xml};

pub mod server_fns;
