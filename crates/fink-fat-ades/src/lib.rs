//! Pure ADES (Astrometry Data Exchange Standard) / Minor Planet Center (MPC)
//! domain logic, shared between `fink-fat-explorer` (the Dioxus web app,
//! reached from the lineage page's export modal) and the root `fink-fat`
//! binary's `submit` CLI subcommand.
//!
//! Every item in this crate is a pure function or a plain data type: no
//! Postgres, no `reqwest` HTTP calls, no filesystem access, no async runtime.
//! Both consumers do their own I/O (`fink-fat-explorer` via async `sqlx` +
//! `reqwest`, `fink-fat`'s CLI via the synchronous `postgres` crate +
//! `reqwest::blocking`) and call into this crate only for request shaping,
//! response parsing, and the submission-quality-tier decision — so the two
//! can never silently drift on what an ADES document, a valid submission, or
//! an "eligible for submission" lineage actually means.
//!
//! # Module layout
//!
//! - [`model`] — `trkSub` normalization, band-code mapping, epoch
//!   conversion, singleton-night filtering, non-blocking submission-quality
//!   advisories.
//! - [`schema_validation`] — local, pure re-implementation of `submit.xsd`'s
//!   constraints (the fast, synchronous validity pre-check).
//! - [`xml`] — the `quick-xml`/`serde` struct hierarchy mirroring
//!   `submit.xsd`, and the pure `AdesDocument` builder.
//! - [`mpc_submission`] — request-shaping and response-parsing for MPC's
//!   `submit_xml`/`submit_xml_test` endpoints and their status pages.
//! - [`submission_status_api`] — response parsing for MPC's documented,
//!   production, JSON Submission Status API.
//! - [`wamo`] — request-body building and response parsing for MPC's WAMO
//!   per-observation status API.
//! - [`quality_tier`] — the [`quality_tier::QualityTier`] submission-quality
//!   cascade and its `well_sampled_nights`/cross-match query text, shared so
//!   the CLI's eligibility gate and the explorer's homepage badge can never
//!   disagree.
//! - [`submitter_config`] — [`submitter_config::SubmitterConfig`], the
//!   submitter/telescope identity shape shared by `fink-fat submit`'s
//!   `--submitter-config` YAML file and the explorer's form that generates
//!   it.
//! - [`format_epoch`] / [`lsst_band`] — small formatting helpers `model`
//!   depends on.
//! - [`error`] — [`error::AdesError`], the typed error enum every fallible
//!   function in this crate returns.

pub mod error;
pub mod format_epoch;
pub mod lsst_band;
pub mod model;
pub mod mpc_submission;
pub mod quality_tier;
pub mod schema_validation;
pub mod submission_status_api;
pub mod submitter_config;
pub mod wamo;
pub mod xml;
