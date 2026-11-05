//! Diagnoses *why* a trackable-but-`NeverTouched` ground-truth object was
//! never linked into a seed pair.
//!
//! [`crate::tracking_report::object_outcome`] tells us an object was never
//! touched by any branch; it doesn't say why. For a trackable object (one
//! that did have >= 2 observations within a single night at some point —
//! see [`GoldTrajectoryTracker::is_trackable`]), the diagnosis proceeds in
//! three stages, each explaining a distinct failure mechanism found in
//! `fink-fat-engine`:
//!
//! 1. **Pairwise gate check** — does the object's own same-night pair
//!    violate one of [`PairConfig`]'s thresholds
//!    (`max_dt`/`max_angular_speed`/`max_mag_difference`), mirroring
//!    `fink_fat_engine::seeding::tracklet_linker::evaluate_candidate`'s
//!    "no fit yet" gate?
//! 2. **Claimed-then-pruned check** — was one of the object's own
//!    observations consumed by an *existing* lineage's speculative
//!    extension that was itself pruned before the end of that night (see
//!    `fink_fat_engine::topocentric_kf::branching::orchestrate::NightAdvanceOutcome::consumed_then_pruned_ids`)?
//!    If so, the observation was excluded from that night's seeding pool
//!    without ever appearing in any surviving branch — never a `PairConfig`
//!    issue at all.
//! 3. **Greedy-matching check** — replay the real
//!    `fink_fat_engine::seeding::tracklet_linker::link_tracklets` on that
//!    night's observations and check whether it actually emits the
//!    object's pair. `link_tracklets` assigns each observation to a single
//!    best-matching track in one time-ordered sweep, so a gate-valid pair
//!    can still lose to a competing candidate — this stage catches that.
//!
//! If none of the three explain it, the object is still unexplained and
//! worth a manual look.

use ahash::AHashSet;
use photom::{
    NightId, TrajId,
    observation_dataset::{ObsDataset, ObsId, observation::Observation},
};

use fink_fat_engine::{
    engine_config::pair_config::PairConfig, seeding::tracklet_linker::link_tracklets,
    spacetime_bucket::healpix_binner::HealpixBinner,
};

use crate::{
    seed_bank_report::ground_truth::ObsTrajLookup,
    tracking_report::gold_trajectory::GoldTrajectoryTracker,
};

/// Why a trackable object's same-night pair never became a seed —
/// mutually exclusive, checked in this order (the first that applies wins,
/// since each successive stage assumes the previous ones didn't already
/// explain it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateFailure {
    /// `dt > max_dt`.
    Temporal,
    /// `angular_separation(a, b) / dt > max_angular_speed`.
    Kinematic,
    /// `|mag_a - mag_b| > max_mag_difference`.
    Photometric,
    /// The pairwise gate itself would have accepted this pair, but one of
    /// its two observations was claimed by an existing lineage's candidate
    /// extension that was later pruned the same night — see the module
    /// doc's stage 2. Not a `PairConfig` issue.
    ClaimedByPrunedCandidate,
    /// The pairwise gate would have accepted this pair and neither
    /// observation was claimed-then-pruned, but replaying the real
    /// `link_tracklets` on that night's observations still doesn't emit
    /// this pair — consistent with the object's true partner observation
    /// having been greedily claimed by a competing candidate track. See the
    /// module doc's stage 3.
    LostToGreedyMatching,
    /// None of the above explains it: the pairwise gate passes, neither
    /// observation was claimed-then-pruned, and replaying `link_tracklets`
    /// in isolation *does* emit the pair. Worth a manual look — something
    /// this diagnostic doesn't model is at play.
    StillUnexplained,
}

impl GateFailure {
    pub fn label(self) -> &'static str {
        match self {
            GateFailure::Temporal => "temporal gate (max_dt)",
            GateFailure::Kinematic => "kinematic gate (max_angular_speed)",
            GateFailure::Photometric => "photometric gate (max_mag_difference)",
            GateFailure::ClaimedByPrunedCandidate => "claimed by an existing lineage, then pruned",
            GateFailure::LostToGreedyMatching => "lost to greedy tracklet-linker competition",
            GateFailure::StillUnexplained => "still unexplained",
        }
    }

    pub fn all() -> [GateFailure; 6] {
        [
            GateFailure::Temporal,
            GateFailure::Kinematic,
            GateFailure::Photometric,
            GateFailure::ClaimedByPrunedCandidate,
            GateFailure::LostToGreedyMatching,
            GateFailure::StillUnexplained,
        ]
    }
}

/// One object's diagnosed same-night pairwise gate check, using its first
/// two observations (chronological) on the night it had its best same-night
/// observation count.
///
/// Only the first two observations of that night are checked: the real
/// linker's "no fit yet" gate only applies to a track's first extension,
/// exactly matching a 2-detections/night object. For an object with >= 3
/// same-night detections, the real linker may establish a fit after the
/// second point and gate subsequent points differently (see
/// `fink_fat_engine::seeding::tracklet_linker`'s `fitted_match_score`) —
/// this diagnostic doesn't replicate that path, so its verdict for such
/// objects is a proxy, not a guarantee.
pub struct GateDiagnosisRecord {
    pub traj_id: TrajId,
    pub night_id: NightId,
    pub dt_days: f64,
    pub angular_speed_rad_per_day: f64,
    pub mag_diff: f64,
    pub failure: GateFailure,
}

/// Run the diagnosis for every trajectory id in `never_touched`. Skips
/// trajectories whose best same-night observation count is < 2 (shouldn't
/// happen for genuinely trackable objects) or whose recorded night has no
/// observations of that object in `obs_dataset` (defensive).
///
/// `consumed_then_pruned_ids` is the cumulative union, over every night
/// processed so far, of
/// `BranchCollection::last_night_consumed_then_pruned_ids` — see stage 2 in
/// the module doc.
#[allow(clippy::too_many_arguments)]
pub fn diagnose_never_touched_gating(
    never_touched: &[TrajId],
    obs_dataset: &ObsDataset,
    ground_truth: &ObsTrajLookup,
    gold_tracker: &GoldTrajectoryTracker,
    night_ids: &[NightId],
    pair_config: &PairConfig,
    consumed_then_pruned_ids: &AHashSet<ObsId>,
    spatial_binner: &HealpixBinner,
) -> Vec<GateDiagnosisRecord> {
    never_touched
        .iter()
        .filter_map(|traj_id| {
            let step = gold_tracker.best_same_night_step(traj_id)?;
            let night_id = *night_ids.get(step)?;
            let night_obs: Vec<&Observation> =
                obs_dataset.iter_night_observations(&night_id)?.collect();

            let mut obs_tonight: Vec<&Observation> = night_obs
                .iter()
                .copied()
                .filter(|obs| ground_truth.traj_of(*obs.id()) == Some(traj_id))
                .collect();
            if obs_tonight.len() < 2 {
                return None;
            }
            obs_tonight.sort_unstable_by(|a, b| a.mjd_tt().total_cmp(&b.mjd_tt()));

            let (a, b) = (obs_tonight[0], obs_tonight[1]);
            let dt = b.mjd_tt() - a.mjd_tt();
            if dt <= 0.0 {
                return None;
            }
            let angular_speed = a.equ_coord().angular_separation(b.equ_coord()) / dt;
            let mag_diff = (a.photometry().magnitude - b.photometry().magnitude).abs();

            // Stage 1: pairwise gate.
            let failure = if dt > pair_config.max_dt {
                GateFailure::Temporal
            } else if angular_speed > pair_config.max_angular_speed {
                GateFailure::Kinematic
            } else if mag_diff > pair_config.max_mag_difference {
                GateFailure::Photometric
            // Stage 2: claimed by an existing lineage's candidate, then pruned.
            } else if consumed_then_pruned_ids.contains(a.id())
                || consumed_then_pruned_ids.contains(b.id())
            {
                GateFailure::ClaimedByPrunedCandidate
            } else {
                // Stage 3: replay the real greedy/exclusive linker on the
                // full night. Approximation of the true `leftover_obs` (it
                // doesn't exclude observations consumed by continuing
                // lineages before seeding ran), so this is a conservative
                // (harder-to-pass) test than reality — a pair that still
                // forms here would certainly have formed on the true,
                // smaller leftover set too.
                let pairs = link_tracklets(&night_obs, spatial_binner, pair_config);
                let pair_formed = pairs.iter().any(|p| {
                    (*p.a.id() == *a.id() && *p.b.id() == *b.id())
                        || (*p.a.id() == *b.id() && *p.b.id() == *a.id())
                });
                if pair_formed {
                    GateFailure::StillUnexplained
                } else {
                    GateFailure::LostToGreedyMatching
                }
            };

            Some(GateDiagnosisRecord {
                traj_id: traj_id.clone(),
                night_id,
                dt_days: dt,
                angular_speed_rad_per_day: angular_speed,
                mag_diff,
                failure,
            })
        })
        .collect()
}

/// Count, for each [`GateFailure`] variant, how many diagnosed records fell
/// into it. Mutually exclusive (see [`GateFailure`]'s doc), so these counts
/// sum to `records.len()`.
pub fn count_by_failure(records: &[GateDiagnosisRecord]) -> Vec<(&'static str, usize)> {
    GateFailure::all()
        .into_iter()
        .map(|failure| {
            let count = records.iter().filter(|r| r.failure == failure).count();
            (failure.label(), count)
        })
        .collect()
}
