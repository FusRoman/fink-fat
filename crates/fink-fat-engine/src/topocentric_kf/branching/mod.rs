//! Track-oriented multi-hypothesis-tracking (MHT) across nights.
//!
//! [`kalman_bank`](crate::topocentric_kf::kalman_bank) resolves the `(ρ, ρ̇)`
//! ambiguity **within** a bank, against a single, already-decided observation.
//! It says nothing about *which* next-night observation a bank should
//! consume when several fall inside its predicted search ellipse — an
//! everyday occurrence at LSST cadence. This module adds that missing layer:
//! branching, log-likelihood-ratio (LLR) scoring against a clutter
//! background, and cross-bank pruning, following the strategy documented in
//! `kalman_update_instruction.md` (MHT branched at the *bank* level, top-B +
//! N-scan pruning).
//!
//! # Why branch at the bank level, not the hypothesis level
//!
//! A branch is a full [`KFBank`](crate::topocentric_kf::kalman_bank::KFBank)
//! clone (every `(ρ, ρ̇)` mode) plus one extended association history
//! (`track_ids`, carried by the bank itself). Branching below that level
//! would let two hypotheses in the same bank disagree about which
//! observations they've seen, which the bank's own invariant forbids (see
//! [`hypothesis`](crate::topocentric_kf::kalman_bank::hypothesis) module
//! docs).
//!
//! # Scope of this module
//!
//! Implemented here: candidate search (§ [`candidate_search`]), detection
//! probability from a running absolute-magnitude estimate (§
//! [`detection_probability`]), LLR scoring (§ [`llr_score`]), the [`Branch`]
//! bookkeeping type, cross-bank pruning (§ [`pruning`]), grouping a night's
//! observations into distinct exposure epochs (§ [`visit`] — LSST cadence
//! means a night is hundreds of epochs, not one), the per-visit branch/
//! score/prune step for *existing* lineages (§ [`orchestrate`]), seeding
//! *brand-new* lineages from unclaimed observations (§ [`discovery`]), and
//! [`BranchCollection`] (§ [`collection`]), the single entry point that ties
//! both into fink-fat's night-after-night loop.
//!
//! Deliberately **not** implemented here (see `kalman_update_instruction.md`
//! for the rationale): inter-bank deduplication ("promotion"), orbital-fit
//! arbitration, and persistence of banks across process runs — the whole
//! module operates as an in-memory, single-process, multi-night simulation.
//!
//! # Test coverage note
//!
//! Every [`KFBank`](crate::topocentric_kf::kalman_bank::KFBank) value borrows
//! a live `KalmanContext` (network-loaded JPL ephemeris + UT1 data), so
//! nothing that holds one — including [`Branch`] — can be built in a fast,
//! offline unit test (a pre-existing limitation of the crate, not introduced
//! here). Unit tests in this module are therefore concentrated on the pure,
//! `KFState`-free functions: LLR scoring, clutter density, detection
//! probability, and candidate search over hand-built `SearchRegion` values.

pub mod branch;
pub mod branch_id;
pub mod candidate_search;
pub mod collection;
pub mod detection_probability;
pub mod discovery;
pub mod llr_score;
pub mod orchestrate;
pub mod pruning;
pub mod visit;

pub use branch::{Branch, BranchSnapshot};
pub use branch_id::BranchId;
pub use collection::{BranchCollection, BranchCollectionSnapshot, SNAPSHOT_FILENAME};
