//! Incremental, visit-by-visit advance of a [`BranchCollection`] through one
//! night: [`NightSession::begin`] → [`NightSession::advance_visit`] (once per
//! visit, in chronological order) → [`NightSession::close`].
//!
//! # Why a session and not one call per night
//!
//! Association is inherently **per visit** (one exposure epoch): clutter
//! density, top-K candidate selection and the single null branch per lineage
//! all need the visit's complete set of alerts (see the
//! [`orchestrate`](super::orchestrate) module doc). Pruning (N-scan, stale
//! purge) and discovery of new lineages are inherently **per night**. A
//! batch run has the whole night up front and could do both in one call, but
//! a streaming consumer receives visits one at a time over ~8 hours; the
//! session holds the night's in-flight state between those visits so both
//! callers drive exactly the same code.
//!
//! [`BranchCollection::advance_one_night`] is the batch wrapper: it groups the
//! night into visits ([`group_observations_into_visits`](super::visit::group_observations_into_visits))
//! and feeds them through a session.
//!
//! # Borrowing contract
//!
//! The session never retains a borrow of the [`ObsDataset`] or of any
//! [`Observation`] across calls: every method takes them as arguments. A
//! streaming caller is therefore free to grow its dataset (e.g. with
//! `ObsDataset::push_observation`) between two visits.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicU64;

use photom::{
    NightId,
    observation_dataset::{ObsDataset, ObsId, observation::Observation},
};
use rayon::prelude::*;

use crate::{
    engine_config::{
        kalman_context::KalmanContext, main_config::EngineConfig,
        night_advance_params::NightAdvanceParams,
    },
    error::EngineError,
    spacetime_bucket::{bucket::build_alert_bucket_index, healpix_binner::HealpixBinner},
    topocentric_kf::branching::{
        Branch, BranchCollection,
        candidate_search::SingleBinTimeBinner,
        discovery::seed_new_lineages_from_leftovers,
        orchestrate::{
            GateRecord, OrchestrateEvent, PrefilterAnchor, compute_prefilter_anchors,
            lineage_might_be_in_visit, resolve_observer_state, spawn_branches_for_lineage,
        },
        pruning::{apply_n_scan_pruning, cap_top_b_per_lineage, purge_stale_lineages},
        visit::Visit,
    },
};

/// Next free `branch_id` / `lineage_id` — the only source of new ids during a
/// night, shared by the visit-level branching step and the end-of-night
/// discovery step so neither can hand out an id the other already used.
///
/// Two separate counters on purpose: `branch_id`s grow at every branching
/// event, `lineage_id`s only when discovery seeds a new lineage (see
/// [`seed_new_lineages_from_leftovers`]).
///
/// Persisted with the collection (see
/// [`BranchCollection::ids`]), so both counters are monotone across nights
/// and process restarts: an id is never handed out twice, even once its last
/// holder has been pruned or archived.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct IdCounters {
    /// Next `branch_id` to hand out.
    pub next_branch_id: u64,
    /// Next `lineage_id` to hand out.
    pub next_lineage_id: u64,
}

impl IdCounters {
    /// Counters strictly above every `(branch_id, lineage_id)` pair of `ids`
    /// (`0` when empty).
    ///
    /// Only a lower bound on the true counters: an id whose last holder was
    /// pruned or archived is not in `ids` and could be re-issued. Used to
    /// rebuild counters that were never persisted (legacy snapshots) and to
    /// guard against hand-assembled collections.
    pub fn above(ids: impl IntoIterator<Item = (u64, u64)>) -> Self {
        ids.into_iter()
            .fold(Self::default(), |acc, (branch_id, lineage_id)| Self {
                next_branch_id: acc.next_branch_id.max(branch_id + 1),
                next_lineage_id: acc.next_lineage_id.max(lineage_id + 1),
            })
    }

    /// [`Self::above`] over the ids of live `branches`.
    pub fn above_live_branches(branches: &[Branch<'_, '_>]) -> Self {
        Self::above(branches.iter().map(|b| (b.branch_id, b.lineage_id)))
    }

    /// Component-wise maximum: counters safe with respect to both `self` and
    /// `other`.
    pub fn max(self, other: Self) -> Self {
        Self {
            next_branch_id: self.next_branch_id.max(other.next_branch_id),
            next_lineage_id: self.next_lineage_id.max(other.next_lineage_id),
        }
    }
}

/// What one [`NightSession::advance_visit`] call did — cheap counters for a
/// caller's observability (the session itself only logs).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VisitOutcome {
    /// `false` when the visit's observer state could not be resolved: the
    /// visit was skipped and every lineage carried over unchanged.
    pub observer_resolved: bool,
    /// Lineages that ran the full propagation + gate this visit (passed the
    /// coarse pre-filter and propagated successfully).
    pub n_lineages_gated: usize,
    /// Live branches after this visit's top-B cap.
    pub n_branches: usize,
}

/// In-flight state of one night being advanced visit by visit — see the
/// [module doc](self).
pub struct NightSession<'state_lf, 'bank_config> {
    /// Night index being advanced (N-scan / staleness bookkeeping).
    current_step: usize,
    /// What [`BranchCollection::last_closed_night_id`] becomes at
    /// [`Self::close`]: the night being advanced, or the previous value when
    /// the caller did not name the night.
    last_closed_night_id: Option<NightId>,
    /// Live branches, replaced after every resolved visit.
    branches: Vec<Branch<'state_lf, 'bank_config>>,
    /// Pre-filter origins of the lineages that entered the night, built at
    /// the first visit whose observer state resolves (see
    /// [`PrefilterAnchor`]); `None` until then.
    prefilter_anchors: Option<HashMap<u64, PrefilterAnchor>>,
    /// Every observation that passed some lineage's gate tonight.
    consumed_observation_ids: HashSet<ObsId>,
    /// Gate diagnostics, one per (lineage, visit) that gated a candidate.
    gate_records: Vec<GateRecord>,
    /// Id source for branches spawned tonight and lineages seeded at close.
    ids: IdCounters,
    /// Visits fed to [`Self::advance_visit`] so far.
    n_visits: usize,
    /// Of those, visits skipped because the observer state did not resolve.
    n_unresolved_visits: usize,
}

impl<'state_lf, 'bank_config> NightSession<'state_lf, 'bank_config> {
    /// Open a night on top of `collection`.
    ///
    /// # Arguments
    /// * `collection` – State left by the previous night; its branches are
    ///   cloned, `collection` itself is left untouched.
    /// * `night_id` – Night being advanced, when the caller knows it. Recorded
    ///   as the collection's last closed night at [`Self::close`], which lets
    ///   a resumed run skip nights it already processed. `None` (e.g. an
    ///   input without a night index) disables that check for this night.
    /// * `current_step` – Night index being advanced.
    ///
    /// # Errors
    /// [`EngineError::NightAlreadyClosed`] when `night_id` is not after
    /// `collection`'s last closed night (see
    /// [`BranchCollection::is_night_closed`]).
    pub fn begin(
        collection: &BranchCollection<'state_lf, 'bank_config>,
        night_id: Option<NightId>,
        current_step: usize,
    ) -> Result<Self, EngineError> {
        if let Some(night_id) = night_id
            && let Some(last_closed) = collection.last_closed_night_id
            && night_id <= last_closed
        {
            return Err(EngineError::NightAlreadyClosed {
                night_id,
                last_closed,
            });
        }
        Ok(Self {
            current_step,
            last_closed_night_id: night_id.or(collection.last_closed_night_id),
            branches: collection.branches.clone(),
            prefilter_anchors: None,
            consumed_observation_ids: HashSet::new(),
            gate_records: Vec::new(),
            // The persisted counters are authoritative; the scan only guards
            // a collection assembled by hand through its public fields.
            ids: collection
                .ids
                .max(IdCounters::above_live_branches(&collection.branches)),
            n_visits: 0,
            n_unresolved_visits: 0,
        })
    }

    /// Night index being advanced.
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Live branches right now (after the last visit's cap).
    pub fn branches(&self) -> &[Branch<'state_lf, 'bank_config>] {
        &self.branches
    }

    /// Visits fed to [`Self::advance_visit`] so far.
    pub fn n_visits(&self) -> usize {
        self.n_visits
    }

    /// Visits skipped because their observer state could not be resolved.
    pub fn n_unresolved_visits(&self) -> usize {
        self.n_unresolved_visits
    }

    /// Fold one visit into the night: propagate every lineage that may be in
    /// the visit's footprint, gate its candidates, branch (one branch per
    /// candidate plus the null branch), then cap to the top-B branches per
    /// lineage.
    ///
    /// Visits must be fed in chronological order; each must be complete (all
    /// of its alerts), since clutter density and the null branch are
    /// per-visit quantities.
    ///
    /// # Arguments
    /// * `visit` – One exposure's observations (see [`Visit`]).
    /// * `obs_dataset`, `kalman_context` – Resolve the visit's observer
    ///   heliocentric state at its epoch.
    /// * `params` – Branching / gating tuning.
    /// * `spatial_binner` – HEALPix binner for the visit's alert index.
    pub fn advance_visit(
        &mut self,
        visit: &Visit<'_>,
        obs_dataset: &ObsDataset,
        kalman_context: &KalmanContext,
        params: &NightAdvanceParams,
        spatial_binner: &HealpixBinner,
    ) -> VisitOutcome {
        self.n_visits += 1;
        if self.branches.is_empty() {
            return VisitOutcome {
                observer_resolved: true,
                ..VisitOutcome::default()
            };
        }

        let Ok((r_obs, v_obs)) = resolve_observer_state(
            obs_dataset,
            kalman_context,
            visit.representative_obs,
            visit.epoch,
        ) else {
            OrchestrateEvent::ObserverResolutionFailed { epoch: visit.epoch }.emit();
            self.n_unresolved_visits += 1;
            return VisitOutcome {
                observer_resolved: false,
                n_lineages_gated: 0,
                n_branches: self.branches.len(),
            };
        };

        // Fresh origin for tonight's coarse pre-filter — one Kepler solve per
        // incoming branch, against `predict_to`'s `n_hypotheses` solves per
        // branch per passing visit. See `PrefilterAnchor` for why
        // extrapolating from a weeks-old epoch was silently starving the
        // association.
        //
        // Anchored at the first visit whose observer state resolves, so `dt`
        // inside `lineage_might_be_in_visit` stays within one night (~8h).
        // Until that visit, every earlier one was skipped without touching
        // `branches`, so `branches` is still exactly the incoming lineages
        // here. Branches spawned later tonight need no anchor: their bank is
        // already at the visit epoch they were born at.
        let branches = &self.branches;
        let prefilter_anchors = self
            .prefilter_anchors
            .get_or_insert_with(|| compute_prefilter_anchors(branches, visit.epoch, r_obs, v_obs));

        // Built once per VISIT, reused by every lineage's cheap pre-filter,
        // candidate search and clutter-density lookup this visit.
        let visit_bucket_index = build_alert_bucket_index(
            visit.observations.iter().copied(),
            spatial_binner,
            &SingleBinTimeBinner,
        );

        // Per-lineage work is independent within a visit (each lineage only
        // ever reads its own bank) and dominated by the two-body Kepler
        // propagation in `spawn_branches_for_lineage` — parallelize across
        // lineages via rayon. `next_branch_id` is a shared atomic counter
        // (branch ids are opaque unique keys, never relied on for ordering
        // — see `branch_id.rs`); `consumed_observation_ids` can't be
        // mutated from parallel closures, so each lineage returns its own
        // consumed ids and they're merged sequentially below (cheap: a
        // `HashSet::extend`, not a bottleneck).
        let next_branch_id = AtomicU64::new(self.ids.next_branch_id);
        let current_step = self.current_step;
        let outcomes: Vec<LineageOutcome<'state_lf, 'bank_config>> = self
            .branches
            .par_iter()
            .map(|lineage| {
                if !lineage_might_be_in_visit(
                    lineage,
                    prefilter_anchors.get(&lineage.branch_id),
                    visit,
                    &visit_bucket_index,
                    params,
                    spatial_binner,
                ) {
                    // No alert anywhere near this lineage's extrapolated
                    // position — skip the two Kepler solves entirely. The
                    // lineage carries over unchanged, picked up again
                    // whenever a later visit's alerts land nearby.
                    return LineageOutcome::Unchanged(lineage.clone());
                }

                match spawn_branches_for_lineage(
                    lineage,
                    &visit_bucket_index,
                    visit.epoch,
                    r_obs,
                    v_obs,
                    params,
                    spatial_binner,
                    &next_branch_id,
                    current_step,
                ) {
                    Some((spawned, consumed)) => LineageOutcome::Spawned(
                        spawned,
                        GateRecord {
                            lineage_id: lineage.lineage_id,
                            nights_since_seed: current_step
                                .saturating_sub(lineage.last_real_update_step),
                            gated_obs_ids: consumed,
                            n_hypotheses: lineage.bank.len(),
                            n_steps: lineage.bank.n_steps(),
                        },
                    ),
                    // Every hypothesis in the bank failed to propagate this
                    // visit (rare, but not impossible — see
                    // `outfit_propagate_universal_failures.md`). Treat it
                    // like "no candidate nearby": carry the lineage over
                    // unchanged rather than losing its whole track history.
                    None => LineageOutcome::Unchanged(lineage.clone()),
                }
            })
            .collect();
        self.ids.next_branch_id = next_branch_id.into_inner();

        let mut n_lineages_gated = 0;
        let mut visit_branches = Vec::new();
        for outcome in outcomes {
            match outcome {
                LineageOutcome::Unchanged(branch) => visit_branches.push(branch),
                LineageOutcome::Spawned(spawned, record) => {
                    n_lineages_gated += 1;
                    visit_branches.extend(spawned);
                    self.consumed_observation_ids
                        .extend(record.gated_obs_ids.iter().copied());
                    self.gate_records.push(record);
                }
            }
        }

        // Top-B pruning after every visit — branch counts multiply at every
        // branching event; left unbounded across a night's worth of visits
        // this would explode long before the night ends.
        self.branches = cap_top_b_per_lineage(visit_branches, params.branch_cap);

        VisitOutcome {
            observer_resolved: true,
            n_lineages_gated,
            n_branches: self.branches.len(),
        }
    }

    /// Close the night: N-scan pruning and stale-lineage purge, then seed new
    /// lineages from the observations no surviving branch claimed.
    ///
    /// # Arguments
    /// * `night_obs` – Every observation of the night, in the order discovery
    ///   should see them (the batch path passes the dataset's night order).
    ///   Observations that reached no [`Self::advance_visit`] call (e.g. a
    ///   streaming consumer's late alerts) belong here too: they simply land
    ///   in the leftover pool.
    /// * `obs_dataset`, `kalman_context`, `engine_config`, `spatial_binner` –
    ///   Same context as the visits; `engine_config` supplies both the
    ///   pruning and the discovery parameters.
    ///
    /// # Returns
    /// The collection for the next night (`current_step + 1`), carrying this
    /// night's archived arcs, gate records and consumed-then-pruned ids, the
    /// advanced id counters and the night as its last closed one.
    ///
    /// # Errors
    /// Propagates a discovery failure (see [`seed_new_lineages_from_leftovers`]).
    pub fn close(
        mut self,
        night_obs: &[&Observation],
        obs_dataset: &ObsDataset,
        kalman_context: &'state_lf KalmanContext,
        engine_config: &'bank_config EngineConfig,
        spatial_binner: &HealpixBinner,
    ) -> Result<BranchCollection<'state_lf, 'bank_config>, EngineError> {
        let params = &engine_config.advance_params;

        // N-scan stays at night granularity: one call, after every visit this
        // night has been folded in.
        let n_branches_before_n_scan = self.branches.len();
        let branches = apply_n_scan_pruning(self.branches, params.n_scan, self.current_step);
        let (mut branches, archived) = purge_stale_lineages(
            branches,
            params.max_lineage_lifetime_nights,
            params.stale_llr_floor,
            params.hard_stale_nights,
            params.archive_min_real_updates,
            self.current_step,
        );
        OrchestrateEvent::NightPruningSummary {
            n_branches_before_n_scan,
            n_branches_after: branches.len(),
        }
        .emit();

        let consumed_then_pruned_ids = consumed_then_pruned(
            &self.consumed_observation_ids,
            branches.iter().map(Branch::track_ids),
        );

        // Observations an existing lineage's candidate extension merely
        // *tried* tonight, but that didn't survive tonight's pruning
        // (`cap_top_b_per_lineage`/`apply_n_scan_pruning`), are not truly
        // claimed by anything: give them back to the leftover pool so a real
        // object doesn't lose its only same-night seeding chance to an
        // unrelated candidate that ultimately went nowhere. An observation
        // only ever appears in one night, so there is no later night where it
        // could be revisited.
        let truly_claimed_ids: HashSet<ObsId> = self
            .consumed_observation_ids
            .difference(&consumed_then_pruned_ids)
            .copied()
            .collect();

        // Separate, differently-binned index — built once inside
        // `build_kf_bank_collection_from_observations` (see `discovery`
        // module docs).
        let new_lineages = seed_new_lineages_from_leftovers(
            night_obs,
            &truly_claimed_ids,
            obs_dataset,
            kalman_context,
            engine_config,
            spatial_binner,
            &mut self.ids.next_lineage_id,
            &mut self.ids.next_branch_id,
            self.current_step,
        )?;
        branches.extend(new_lineages);

        Ok(BranchCollection {
            branches,
            last_night_consumed_then_pruned_ids: consumed_then_pruned_ids,
            last_night_gate_records: self.gate_records,
            current_step: self.current_step + 1,
            ids: self.ids,
            last_closed_night_id: self.last_closed_night_id,
            archived,
        })
    }
}

/// Outcome of checking one lineage against one visit — either it carries
/// over unchanged (no candidate nearby), or it spawned branches plus the
/// gate record of the observations they consumed. Kept separate from
/// mutating shared state directly so the per-lineage work in
/// [`NightSession::advance_visit`] can run in parallel via rayon.
enum LineageOutcome<'state_lf, 'bank_config> {
    Unchanged(Branch<'state_lf, 'bank_config>),
    Spawned(Vec<Branch<'state_lf, 'bank_config>>, GateRecord),
}

/// Observations some lineage gated tonight (`consumed`) that ended up in no
/// surviving branch's track history — claimed by a candidate extension that
/// was itself pruned before the night closed.
///
/// Deliberately computed against live branches only, not archived arcs: a
/// lineage purged for staleness consumed none of tonight's observations
/// (that is what made it stale).
///
/// # Arguments
/// * `consumed` – Every observation gated by some lineage tonight.
/// * `surviving_tracks` – Track histories of the branches that survived the
///   night's pruning.
fn consumed_then_pruned<'a>(
    consumed: &HashSet<ObsId>,
    surviving_tracks: impl Iterator<Item = &'a [ObsId]>,
) -> HashSet<ObsId> {
    let surviving_ids: HashSet<ObsId> = surviving_tracks.flatten().copied().collect();
    consumed.difference(&surviving_ids).copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consumed_then_pruned_keeps_only_ids_absent_from_every_survivor() {
        let consumed: HashSet<ObsId> = [1, 2, 3, 4].into();
        let tracks: [&[ObsId]; 2] = [&[10, 1, 2], &[11, 3]];
        assert_eq!(
            consumed_then_pruned(&consumed, tracks.into_iter()),
            HashSet::from([4])
        );
    }

    #[test]
    fn consumed_then_pruned_without_survivors_returns_everything() {
        let consumed: HashSet<ObsId> = [5, 6].into();
        assert_eq!(
            consumed_then_pruned(&consumed, std::iter::empty()),
            consumed
        );
    }

    #[test]
    fn consumed_then_pruned_with_nothing_consumed_is_empty() {
        let tracks: [&[ObsId]; 1] = [&[1, 2]];
        assert!(consumed_then_pruned(&HashSet::new(), tracks.into_iter()).is_empty());
    }

    #[test]
    fn id_counters_above_is_one_past_each_max() {
        assert_eq!(IdCounters::above([]), IdCounters::default());
        assert_eq!(
            IdCounters::above([(7, 2), (3, 5)]),
            IdCounters {
                next_branch_id: 8,
                next_lineage_id: 6,
            }
        );
    }

    #[test]
    fn id_counters_max_is_component_wise() {
        let a = IdCounters {
            next_branch_id: 10,
            next_lineage_id: 1,
        };
        let b = IdCounters {
            next_branch_id: 4,
            next_lineage_id: 9,
        };
        assert_eq!(
            a.max(b),
            IdCounters {
                next_branch_id: 10,
                next_lineage_id: 9,
            }
        );
    }

    /// Empty collection whose last closed night is `last_closed`.
    fn closed_at(last_closed: Option<u32>) -> BranchCollection<'static, 'static> {
        BranchCollection {
            last_closed_night_id: last_closed.map(NightId),
            ..BranchCollection::empty()
        }
    }

    #[test]
    fn begin_refuses_a_night_not_after_the_last_closed_one() {
        let collection = closed_at(Some(5));
        for night in [4, 5] {
            assert!(matches!(
                NightSession::begin(&collection, Some(NightId(night)), 0),
                Err(EngineError::NightAlreadyClosed { night_id, last_closed })
                    if night_id == NightId(night) && last_closed == NightId(5)
            ));
        }
    }

    #[test]
    fn begin_accepts_a_later_night_and_records_it_at_close_time() {
        let session = NightSession::begin(&closed_at(Some(5)), Some(NightId(6)), 0).unwrap();
        assert_eq!(session.last_closed_night_id, Some(NightId(6)));
        let session = NightSession::begin(&closed_at(None), Some(NightId(0)), 0).unwrap();
        assert_eq!(session.last_closed_night_id, Some(NightId(0)));
    }

    #[test]
    fn begin_without_night_id_keeps_the_previous_last_closed_night() {
        let session = NightSession::begin(&closed_at(Some(5)), None, 0).unwrap();
        assert_eq!(session.last_closed_night_id, Some(NightId(5)));
    }

    #[test]
    fn begin_starts_from_the_persisted_counters() {
        let ids = IdCounters {
            next_branch_id: 42,
            next_lineage_id: 7,
        };
        let collection = BranchCollection {
            ids,
            ..BranchCollection::empty()
        };
        assert_eq!(NightSession::begin(&collection, None, 0).unwrap().ids, ids);
    }
}
