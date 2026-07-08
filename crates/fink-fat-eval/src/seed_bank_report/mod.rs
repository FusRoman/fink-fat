//! Full-dataset evaluation of intra-night Kalman-bank seeding: recall
//! (did every real multi-detection object get a seed?), purity (was the
//! seed uncontaminated by another object?), and volume (how many banks and
//! hypotheses did seeding produce?), aggregated across every night in a
//! dataset.
//!
//! Entry points: [`ground_truth::ObsTrajLookup::build`] once per run, then
//! [`night_stats::compute_night_stats`] once per night, collected into a
//! [`report::SeedingReport`] and rendered with [`report::SeedingReport::print_summary`]
//! and the [`plots`] functions. See `bin/seed_to_bank_analysis.rs` for the
//! orchestrating loop.

pub mod ground_truth;
pub mod night_stats;
pub mod plots;
pub mod report;
