//! Automatic calibration of the KF-bank's search-region/gating/process-noise
//! parameters against a labelled trajectory population (see
//! `bin/kf_calibrate.rs`).
//!
//! # Why this exists
//!
//! [`crate::kalman_traj::study_kalman_asteroid`] already runs one known
//! trajectory through the exact production propagate →
//! `predict_search_region` → update loop, and
//! [`crate::trajectory_processing::TrajSummary`] already reduces that into
//! per-trajectory recall-adjacent metrics
//! (`completion_fraction`/`pct_within_search_radius`/
//! `mean_search_radius_arcsec`). This module adds the missing piece: a
//! search over the config values that feed that loop
//! ([`params::CalibrationParams`]), driven by a recall-vs-cost objective
//! ([`objective`]) and a progressive-sampling coordinate-descent strategy
//! ([`search::calibrate`]) so it scales to a population of tens of
//! thousands of trajectories.
//!
//! # Never rebuild the ephemeris per candidate
//!
//! [`params::CalibrationParams::build_context`] clones the caller's
//! `KalmanContext` (its ephemeris is `Arc`-shared, so this is O(1)) and only
//! overrides `q0`/`dt_ref` on the copy. Calling
//! `EngineConfig::build_context`/`KalmanContextConfig::build` instead — which
//! reloads the ephemeris from disk/network — inside a per-candidate hot loop
//! would make calibration prohibitively slow; every function in this module
//! is written to avoid it.

pub mod objective;
pub mod params;
pub mod report;
pub mod search;
