//! Night-after-night multi-hypothesis-tracking evaluation: drives
//! [`BranchCollection::advance_one_night`](fink_fat_engine::topocentric_kf::branching::BranchCollection::advance_one_night)
//! across a whole dataset (unlike [`crate::seed_bank_report`], which only
//! evaluates the night-0 seeding step in isolation) and reports timing,
//! branching volume, gold-trajectory recall/purity/completeness, LLR
//! confidence, and Kalman error-box sizing — per night and aggregated.
//!
//! Entry points: [`night_stats::compute_night_tracking_stats`] once per
//! night, collected into a [`report::TrackingReport`] and rendered with
//! [`report::AggregatedTrackingStats::print_summary`] and the [`plots`]
//! functions. See `bin/tracking_analysis.rs` for the orchestrating loop.

pub mod error_box;
pub mod gate_selectivity;
pub mod gold_trajectory;
pub mod lineage_lifecycle;
pub mod merge_shadow;
pub mod night_stats;
pub mod object_outcome;
pub mod plots;
pub mod report;
pub mod seeding_gate_diagnosis;
