//! [`LiveNight`]: one open night, fed visit by visit — the only part of the
//! driver that talks to the tracking engine.
//!
//! # Role
//!
//! The engine's [`NightSession`] advances a [`BranchCollection`] through one
//! night: `begin` → `advance_visit` once per visit → `close`. Its methods
//! take the observations and the [`ObsDataset`] as arguments and keep no
//! borrow of them between calls. `LiveNight` pairs the session with the
//! dataset those observations live in, and turns the driver's vocabulary
//! (a visit = a `Vec<ObservationInput>`, a late alert = a leftover) into the
//! session's.
//!
//! # Life of a night
//!
//! ```text
//! begin(collection, night_id)        session opened, empty dataset
//!   ├─ advance_visit(inputs)  ×N     inputs appended to the dataset, then
//!   │                                folded into the branches as one Visit
//!   ├─ add_leftovers(inputs)  ×M     inputs appended to the dataset only
//!   └─ close()                       pruning + discovery over the whole
//!                                    dataset → next BranchCollection;
//!                                    the dataset is dropped
//! ```
//!
//! Visits must be fed in chronological order and each must be complete (all
//! of its alerts): clutter density, top-K selection and the single null
//! branch per lineage are per-visit quantities. Guaranteeing that is the
//! [`VisitAssembler`](crate::assembler::VisitAssembler)'s job.
//!
//! # A dataset per night
//!
//! A live night owns an [`ObsDataset`] holding **only that night's**
//! observations, grown with `push_observation` and dropped at close. This
//! keeps memory bounded over a survey and makes a restart independent of
//! past nights. It is enough for the engine because:
//!
//! * branches carry observation *ids*, never references into a dataset;
//! * a visit's association reads the visit's own observations, and the
//!   dataset only to resolve the observer;
//! * discovery reads the night's observations passed to
//!   [`NightSession::close`], not the dataset's night index.
//!
//! The dataset carries an astrometric error model: without it, photom cannot
//! resolve an MPC-coded observer (Rubin's `X05`) and every visit would be
//! skipped as unresolved.
//!
//! `tests/night_session_regression.rs` (workspace root) checks that a night
//! fed this way lands on the same golden fingerprint as the batch pipeline
//! (`BranchCollection::advance_one_night` over one dataset of all nights).

use fink_fat_engine::{
    engine_config::{EngineConfig, kalman_context::KalmanContext},
    spacetime_bucket::healpix_binner::HealpixBinner,
    topocentric_kf::branching::{
        BranchCollection, NightRefused, NightSession, VisitOutcome, visit::Visit,
    },
};
use photom::{
    NightId,
    observation_dataset::{
        ObsDataset,
        observation::{Observation, ObservationInput},
    },
    observer::error_model::ObsErrorModel,
};

use crate::error::{Result, StreamError};

/// An open night: the engine's [`NightSession`] plus the night's dataset
/// (see the [module doc](self)).
///
/// `'ctx` / `'cfg` are the lifetimes of the [`KalmanContext`] and
/// [`EngineConfig`] the branches borrow; they outlive the night and are
/// handed back with the collection at [`Self::close`].
pub struct LiveNight<'ctx, 'cfg> {
    /// The night being advanced (MJD of its `dayObs`).
    night_id: NightId,
    /// The engine's in-flight state of the night (owns the branches).
    session: NightSession<'ctx, 'cfg>,
    /// Every observation of the night so far: visits and leftovers.
    dataset: ObsDataset,
    /// Engine context: observer ephemerides, Kalman parameters.
    kalman_context: &'ctx KalmanContext,
    /// Engine configuration: advance / pruning / discovery parameters.
    engine_config: &'cfg EngineConfig,
    /// HEALPix binner at the configured depth, built once per night and
    /// shared by every visit and by discovery.
    spatial_binner: HealpixBinner,
    /// Observations added through [`Self::add_leftovers`].
    n_leftovers: usize,
}

impl<'ctx, 'cfg> LiveNight<'ctx, 'cfg> {
    /// Open `night_id` on top of `collection`.
    ///
    /// The night is advanced as step `collection.current_step` (the engine's
    /// night counter, used by N-scan pruning and staleness), and recorded as
    /// the collection's last closed night once [`Self::close`]d.
    ///
    /// # Arguments
    /// * `collection` – State left by the previous night, consumed without
    ///   copy: its branches move into the session (see
    ///   [`NightSession::begin`]). Its per-night outputs (`archived`, …) are
    ///   dropped — flush them before.
    /// * `night_id` – The night to open.
    /// * `error_model` – Astrometric error model of the night's dataset,
    ///   needed to resolve MPC-coded observers (see the
    ///   [module doc](self#a-dataset-per-night)).
    /// * `kalman_context`, `engine_config` – Engine context, borrowed for
    ///   the whole night.
    ///
    /// # Errors
    /// [`NightRefused`] when `night_id` is not after the collection's last
    /// closed night (it would be advanced twice); it carries the untouched
    /// collection back so the caller keeps its state.
    pub fn begin(
        collection: BranchCollection<'ctx, 'cfg>,
        night_id: NightId,
        error_model: ObsErrorModel,
        kalman_context: &'ctx KalmanContext,
        engine_config: &'cfg EngineConfig,
    ) -> std::result::Result<Self, NightRefused<'ctx, 'cfg>> {
        let current_step = collection.current_step;
        let session = NightSession::begin(collection, Some(night_id), current_step)?;
        Ok(Self {
            night_id,
            session,
            dataset: ObsDataset::empty().with_error_model(error_model),
            kalman_context,
            engine_config,
            spatial_binner: HealpixBinner::new(engine_config.healpix_depth),
            n_leftovers: 0,
        })
    }

    /// The open night (MJD of its `dayObs`).
    pub fn night_id(&self) -> NightId {
        self.night_id
    }

    /// Observations in tonight's dataset: every visit's plus the leftovers.
    pub fn n_observations(&self) -> usize {
        self.dataset.observation_count()
    }

    /// Visits handed to the engine so far (empty visits excluded, see
    /// [`Self::advance_visit`]).
    pub fn n_visits(&self) -> usize {
        self.session.n_visits()
    }

    /// Of [`Self::n_visits`], those the engine skipped because the visit's
    /// observer state did not resolve (every lineage carried over
    /// unchanged).
    pub fn n_unresolved_visits(&self) -> usize {
        self.session.n_unresolved_visits()
    }

    /// Observations added through [`Self::add_leftovers`] so far.
    pub fn n_leftovers(&self) -> usize {
        self.n_leftovers
    }

    /// Fold one complete visit into the night.
    ///
    /// # Steps
    /// 1. Append `inputs` to tonight's dataset (after checking none of their
    ///    ids is already there).
    /// 2. Build the engine's [`Visit`] over the freshly inserted
    ///    observations: its epoch and representative observation are the
    ///    *earliest* observation's, which is what the batch grouping
    ///    ([`group_observations_into_visits`](fink_fat_engine::topocentric_kf::branching::visit::group_observations_into_visits))
    ///    picks after sorting by epoch. All alerts of a Rubin visit share one
    ///    mid-point, so in practice any of them gives the same epoch.
    /// 3. [`NightSession::advance_visit`]: every lineage near the visit is
    ///    propagated to its epoch, gated against its observations, branched
    ///    and capped.
    ///
    /// The [`Visit`] only borrows the dataset for the duration of the call,
    /// which is what lets the next call append to the dataset again.
    ///
    /// # Arguments
    /// * `inputs` – Every observation of the visit, in any order.
    ///
    /// # Returns
    /// The engine's per-visit counters. An empty `inputs` is a no-op that
    /// reports the current branch count, without reaching the engine.
    ///
    /// # Errors
    /// [`StreamError::DuplicateObservation`] when an id is already in the
    /// night or repeated in `inputs`; the dataset is then left untouched.
    pub fn advance_visit(&mut self, inputs: Vec<ObservationInput>) -> Result<VisitOutcome> {
        // 1. Append.
        let indices = self.push(inputs)?;
        let observations: Vec<&Observation> = indices
            .into_iter()
            .map(|index| {
                self.dataset
                    .get_obs_by_index(index)
                    .expect("index returned by push_observation")
            })
            .collect();

        // 2. Visit anchored on the earliest observation (`min_by` keeps the
        // first of equal epochs, like the batch's stable sort).
        let Some(&representative_obs) = observations
            .iter()
            .min_by(|a, b| a.mjd_tt().total_cmp(&b.mjd_tt()))
        else {
            return Ok(VisitOutcome {
                observer_resolved: true,
                n_lineages_gated: 0,
                n_branches: self.session.branches().len(),
            });
        };
        let visit = Visit {
            epoch: representative_obs.mjd_tt(),
            representative_obs,
            observations,
        };

        // 3. Engine. `session` and `dataset` are distinct fields, so the
        // session can be borrowed mutably while the visit borrows the dataset.
        Ok(self.session.advance_visit(
            &visit,
            &self.dataset,
            self.kalman_context,
            &self.engine_config.advance_params,
            &self.spatial_binner,
        ))
    }

    /// Add observations of tonight that reach no visit.
    ///
    /// These are alerts of a visit already advanced (they arrived after the
    /// visit was declared complete). Folding them in now would re-run that
    /// visit's association with an incomplete clutter picture, so they skip
    /// association entirely; but they sit in the dataset, so
    /// [`Self::close`]'s discovery step sees them like any unclaimed
    /// observation and may seed new lineages from them.
    ///
    /// # Arguments
    /// * `inputs` – Leftover observations, any order.
    ///
    /// # Errors
    /// As [`Self::advance_visit`].
    pub fn add_leftovers(&mut self, inputs: Vec<ObservationInput>) -> Result<()> {
        self.n_leftovers += self.push(inputs)?.len();
        Ok(())
    }

    /// Close the night and drop its dataset.
    ///
    /// Runs [`NightSession::close`] over **every** observation of the
    /// night's dataset, in insertion order (visits as advanced, leftovers
    /// where they arrived):
    ///
    /// 1. N-scan pruning and stale-lineage purge (stale lineages with enough
    ///    evidence become archived trajectories);
    /// 2. discovery: observations no surviving branch claimed tonight are
    ///    linked into tracklets and seed new lineages.
    ///
    /// Discovery does not depend on that order (its tracklet sweep sorts by
    /// `(epoch, id)`), so the result is the batch pipeline's whatever order
    /// the alerts arrived in.
    ///
    /// # Returns
    /// The collection for the next night (`current_step + 1`, this night as
    /// its last closed one), carrying this night's freshly archived
    /// trajectories in [`BranchCollection::archived`] — for the caller to
    /// persist.
    ///
    /// # Errors
    /// The engine's discovery error.
    pub fn close(self) -> Result<BranchCollection<'ctx, 'cfg>> {
        let night_obs: Vec<&Observation> = self.dataset.iter_observations().collect();
        Ok(self.session.close(
            &night_obs,
            &self.dataset,
            self.kalman_context,
            self.engine_config,
            &self.spatial_binner,
        )?)
    }

    /// Append `inputs` to tonight's dataset.
    ///
    /// photom's `ObsDataset::push_observation` takes the dataset by value and
    /// does **not** give it back when it refuses a duplicate id: calling it
    /// with a duplicate would lose the whole night. Duplicates are therefore
    /// detected first ([`first_duplicate_id`]), and the dataset is only
    /// moved out (`std::mem::take`, leaving a throw-away empty dataset for
    /// the instant of the call) once the push cannot fail on them.
    ///
    /// # Returns
    /// The positions of the new observations in the dataset (photom's
    /// `ObsIndex`), in the order of `inputs`.
    ///
    /// # Errors
    /// [`StreamError::DuplicateObservation`] for the first duplicate id; the
    /// dataset is untouched.
    fn push(&mut self, inputs: Vec<ObservationInput>) -> Result<Vec<usize>> {
        if let Some(id) = first_duplicate_id(&self.dataset, &inputs) {
            return Err(StreamError::DuplicateObservation(id));
        }
        let (dataset, indices) = std::mem::take(&mut self.dataset).push_observation(inputs)?;
        self.dataset = dataset;
        Ok(indices)
    }
}

/// First id of `inputs` that is already in `dataset` or appears twice within
/// `inputs` — exactly the ids photom's `push_observation` would refuse.
///
/// # Returns
/// `None` when `inputs` can be pushed safely.
fn first_duplicate_id(dataset: &ObsDataset, inputs: &[ObservationInput]) -> Option<u64> {
    let mut seen = std::collections::HashSet::with_capacity(inputs.len());
    inputs
        .iter()
        .map(|input| input.id)
        .find(|&id| dataset.get_observation(id).is_some() || !seen.insert(id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use photom::{
        coordinates::equatorial::EquCoord,
        photometry::{Filter, Photometry},
    };

    fn input(id: u64) -> ObservationInput {
        ObservationInput::new(
            id,
            EquCoord::new(0.0, 1e-6, 0.0, 1e-6),
            Photometry {
                magnitude: 20.0,
                error: 0.1,
                filter: Filter::Int(2),
            },
            61_000.0,
            None,
        )
    }

    #[test]
    fn duplicates_against_the_dataset_or_within_the_batch_are_found() {
        let (dataset, _) = ObsDataset::empty()
            .push_observation(vec![input(1)])
            .unwrap();
        assert_eq!(first_duplicate_id(&dataset, &[input(2), input(3)]), None);
        assert_eq!(first_duplicate_id(&dataset, &[input(2), input(1)]), Some(1));
        assert_eq!(first_duplicate_id(&dataset, &[input(4), input(4)]), Some(4));
    }
}
