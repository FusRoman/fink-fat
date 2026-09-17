//! The orbit-fitting pipeline both fit paths share.
//!
//! `crate::orbit_fit` (one branch, on demand, from the lineage page) and
//! `crate::bulk_orbit_fit` (every eligible branch, as a background job) differ
//! only in *what* they fit and how they report progress. Everything about
//! *how* a fit runs lives here:
//!
//! - [`params`] — the form's parameters and their translation into `outfit`'s
//!   configuration,
//! - [`dataset`] — observations to a corrected `ObsDataset` + geometry cache,
//! - [`fit`] — running the fit and the [`fit::FitProduct`] it yields,
//! - [`store`] — writing that product to `orbit_fits`.
//!
//! Keeping them together is the point. Each time a step existed twice, the two
//! paths eventually disagreed on the same branch: a missing `traj_id` column
//! silently disabled the batch-RMS correction on one side, and a stray RNG
//! draw changed which Gauss solution seeded the correction on the other.

/// Server-only: builds on `photom`/`polars`, which the wasm bundle doesn't
/// carry. [`fit`] and [`params`] stay available to both builds, since the form
/// and the result pages need their types.
#[cfg(feature = "server")]
pub mod dataset;
pub mod fit;
pub mod params;
#[cfg(feature = "server")]
pub mod store;
