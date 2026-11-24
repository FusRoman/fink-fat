//! Intra-night observation pairing → Kalman-filter bank seeding.
//!
//! This module turns a [`photom::observation_dataset::ObsDataset`] into a
//! collection of [`crate::topocentric_kf::kalman_bank::KFBank`] hypotheses banks:
//!
//! 1. Observations are scoped to a single night (seeds never mix nights).
//! 2. Candidate observation pairs within that night are produced by
//!    [`tracklet_linker::link_tracklets`] — a time-ordered sweep that links
//!    each object's intra-night detections into a track and emits one
//!    maximal-baseline [`pairs::Pair`] per multi-detection object, instead
//!    of the `O(n²)` all-pairs enumeration a plain pairwise gate would
//!    produce on a night with many revisits per object. The linker only
//!    shares the [`pairs::Pair`] type with [`pairs::generate_pairs`]: it
//!    applies the same `max_angular_speed` criterion through its own code
//!    and its own neighborhood index, and does not call it.
//!    `generate_pairs` (the pairwise all-pairs enumeration) is no longer on
//!    the production path; it remains available for simpler pairing needs.
//! 3. Each surviving pair seeds one [`crate::topocentric_kf::kalman_bank::KFBank`]
//!    via `KFBank::from_grid`.
//!
//! See [`crate::topocentric_kf::kalman_bank::from_seeds::build_kf_bank_collection`] for the top-level entry
//! point.

pub mod error;
pub mod pairs;
pub mod tracklet_linker;
pub mod triplets;
