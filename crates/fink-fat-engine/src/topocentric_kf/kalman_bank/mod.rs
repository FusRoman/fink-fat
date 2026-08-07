//! Multi-hypothesis Kalman filter bank for angles-only initial orbit tracking.
//!
//! A single observation pair (tracklet) only constrains 4 of the 6 state
//! degrees of freedom: the two sky angles and the two angular rates. The
//! topocentric range `ρ` and range-rate `ρ̇` are **unobservable** and must be
//! hypothesized. Rather than committing to one (poor) guess, the bank carries
//! a population of [`KFState`](crate::topocentric_kf::single_kalman::KFState) hypotheses — one per `(ρ, ρ̇)` seed drawn from
//! the admissible region — each with a Bayesian weight.
//!
//! As new observations arrive, every hypothesis is propagated and updated;
//! its weight is multiplied by the predictive likelihood of the observation;
//! clearly-inconsistent hypotheses are gated out and the weakest are pruned.
//! Within a few nights the range ambiguity collapses to a single dominant mode.
//!
//! # Robustness improvements over the original design
//!
//! ## MAP-hypothesis gate protection
//!
//! The chi-square gate now **never discards the highest-weight (MAP) hypothesis**.
//! The gate's purpose is to cull clearly implausible range hypotheses from a large
//! population; applying it to the sole surviving best estimate is counter-productive:
//! it kills the filter right when it should adapt. Instead, a MAP hypothesis that
//! exceeds the gate is logged and passed through; filter inconsistency is tracked
//! via the NIS diagnostic.
//!
//! ## Scheduled hypothesis-count decay
//!
//! Early in the arc many hypotheses are needed to cover the `(ρ, ρ̇)` ambiguity.
//! As more observations constrain the orbit, the bank can be pruned more
//! aggressively. [`HypothesisCapSchedule`](crate::engine_config::hypothesis_cap::HypothesisCapSchedule) makes this decay explicit and tunable
//! via linear, logarithmic, or exponential schedules, rather than relying on the
//! implicit weight-collapse dynamics alone.
//!
//! A `min_hypotheses` floor is always respected so the bank never collapses below
//! a user-defined minimum.
//!
//! ## Smoothed likelihood pruning
//!
//! The classical pruning criterion (weight floor on the cumulative Bayesian weight)
//! can kill a valid hypothesis after a single outlier observation. The new scheme
//! accumulates a **sliding window of per-step log-likelihoods** for each hypothesis
//! and prunes on the *mean* log-likelihood over that window. This averages out
//! single-epoch astrometric anomalies while still converging to the same result
//! once the window fills with consistent data.
//!
//! Setting `likelihood_window = 0` disables smoothing and falls back to the
//! original weight-floor approach.
//!
//! # Integration point
//!
//! [`KFBank::from_grid`] takes pre-built `(KFState, weight)` pairs produced by
//! the admissible-region grid generator:
//!
//! ```ignore
//! let seeds = admissible_region_grid(obs_dataset, obs1, obs2, &ctx, &grid_cfg)?;
//! let mut bank = KFBank::from_grid(seeds, KFBankConfig::default());
//! for obs in observations {
//!     let report = bank.step(obs_dataset, obs);
//!     if report.collapsed { break; }
//! }
//! let best_orbit = bank.best().unwrap().kf.to_orbit();
//! ```
//!
//! # Lifetime
//!
//! [`KFBank`] and [`Hypothesis`] share the `'state_lf` lifetime of [`KFState`](crate::topocentric_kf::single_kalman::KFState):
//! every hypothesis borrows the same ephemeris context that was used to build its
//! seed.

pub mod ellipse_region_finder;
pub mod from_seeds;
pub mod hypothesis;
pub mod logging;
pub mod seed_grid;

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use photom::observation_dataset::{ObsDataset, ObsId, observation::Observation};

use nalgebra::{Matrix2, Vector2, Vector3, Vector6};

use crate::topocentric_kf::kalman_bank::logging::BankEvent;
use crate::{
    engine_config::{
        grid_population::GridConfig, kalman_context::KalmanContext, kf_bank_config::KFBankConfig,
    },
    error::EngineError,
    topocentric_kf::{
        branching::detection_probability::{
            implied_absolute_magnitude, update_running_magnitude_estimate,
        },
        kalman_bank::{
            ellipse_region_finder::SearchComponent,
            hypothesis::{Hypothesis, HypothesisSnapshot, HypothesisStepResult},
            seed_grid::admissible_region_grid,
        },
        observer_state::get_observer,
    },
};

// ── Per-step diagnostics ──────────────────────────────────────────────────────

/// Per-step diagnostics returned by [`KFBank::step`].
#[derive(Debug, Clone)]
pub struct BankStep {
    /// Observation epoch (MJD TT) processed in this step.
    pub epoch: f64,
    /// Number of live hypotheses entering the step.
    pub n_before: usize,
    /// Number of live hypotheses after gating / pruning / merging.
    pub n_after: usize,
    /// Hypotheses discarded by the chi-square gate.
    pub n_gated: usize,
    /// Hypotheses lost to a propagation / innovation / update failure.
    pub n_failed: usize,
    /// Effective sample size `1 / Σ wᵢ²` — low values mean one mode dominates.
    pub n_effective: f64,
    /// Weight of the current best (MAP) hypothesis.
    pub best_weight: f64,
    /// `true` if no hypothesis survived (filter lost).
    pub collapsed: bool,
    /// Effective hypothesis cap applied at this step (schedule + min floor).
    ///
    /// Useful for diagnosing whether the cap schedule is driving compression
    /// or whether weight dynamics are doing it faster.
    pub scheduled_cap: usize,
}

// ── Bank ──────────────────────────────────────────────────────────────────────

/// A bank of weighted [`KFState`](crate::topocentric_kf::single_kalman::KFState) hypotheses tracking a single object under
/// range / range-rate ambiguity.
///
/// `Clone` is implemented manually (see below) rather than derived, because
/// `AtomicUsize` (used for `best_index_cache`) does not implement `Clone` —
/// cloning it means creating a fresh atomic seeded with the current cached
/// value, not bitwise-copying the atomic itself.
pub struct KFBank<'state_lf, 'bank_config> {
    /// `Arc`-wrapped so that cloning a bank whose hypotheses end up
    /// unmodified (e.g. [`Self::branch_null`], or a lineage carried over
    /// unchanged across a visit with no nearby candidate) is an O(1)
    /// refcount bump instead of a deep clone of up to a few hundred
    /// [`Hypothesis`] structs. Any in-place mutation goes through
    /// `Arc::make_mut`, which clones-on-write only when the `Arc` is
    /// actually shared. `Arc` (not `Rc`) so that `KFBank`/`Branch` stay
    /// `Send + Sync` — required to parallelize the per-lineage work in
    /// `advance_bank_collection_one_night` with rayon.
    ///
    /// Private: the only ways to change this field are [`Self::with_hypotheses`]
    /// (construction) and the [`Self::set_hypotheses`]/[`Self::hypotheses_mut`]
    /// helpers (mutation) — both keep `best_index_cache` in sync, which would
    /// otherwise be easy to forget at one of the many pruning/merging call
    /// sites.
    hypotheses: Arc<Vec<Hypothesis<'state_lf>>>,
    /// Cached index (into `hypotheses`) of the highest-`log_weight`
    /// hypothesis, as last computed by [`Self::best`]. `usize::MAX` means
    /// "not computed yet for the current `hypotheses`" — reset by every
    /// path that can change `hypotheses`'s contents. `AtomicUsize` (not
    /// `Cell`) so `KFBank` stays `Sync`; each bank is only ever touched by
    /// one rayon worker thread at a time, so `Ordering::Relaxed` is enough
    /// — this is a memoization cache, not a cross-thread synchronization
    /// point. `AtomicUsize` has no `Clone` impl of its own, so `KFBank`'s
    /// manual `Clone` impl (below) seeds the clone's atomic from a relaxed
    /// load of `self`'s — cheap, and carries a valid cache over for free
    /// whenever a bank is cloned unchanged.
    best_index_cache: AtomicUsize,
    pub config: &'bank_config KFBankConfig,

    /// Number of `step()` calls completed so far.
    ///
    /// Drives the [`HypothesisCapSchedule`](crate::engine_config::hypothesis_cap::HypothesisCapSchedule) decay: cap = schedule.cap(n_steps).
    n_steps: usize,
    /// Ids of every observation associated to this bank so far, in order.
    ///
    /// Lives on the bank rather than on [`Hypothesis`]: every hypothesis in a
    /// bank survives (or is gated out) together, so all of them share exactly
    /// the same association history by construction — see the branch-level
    /// invariant documented in `kalman_update_instruction.md`.
    track_ids: Vec<ObsId>,
    /// Running mean of the absolute magnitude `H` implied by every
    /// observation associated so far (see
    /// [`branching::detection_probability`](crate::topocentric_kf::branching::detection_probability)).
    /// `None` until the first successful [`Self::branch_with`] call.
    absolute_magnitude_estimate: Option<f64>,
    /// Number of samples folded into `absolute_magnitude_estimate`.
    absolute_magnitude_sample_count: u32,
}

/// Owned, borrow-free snapshot of a [`KFBank`], for persisting a
/// [`BranchCollection`](crate::topocentric_kf::branching::BranchCollection)
/// to disk across nights (see [`KFBank::to_snapshot`]).
///
/// `best_index_cache` is deliberately absent: it is a memoization cache
/// (`usize::MAX` = "not computed"), always reset to that sentinel by
/// `KFBank::with_hypotheses` — the same constructor
/// [`KFBank::from_snapshot`] goes through — so it comes back correct by
/// construction. `config` is likewise absent: it borrows the (not
/// serializable, caller-owned) [`KFBankConfig`], re-supplied by the caller
/// via [`KFBank::from_snapshot`].
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct KFBankSnapshot {
    pub hypotheses: Vec<HypothesisSnapshot>,
    pub n_steps: usize,
    pub track_ids: Vec<ObsId>,
    pub absolute_magnitude_estimate: Option<f64>,
    pub absolute_magnitude_sample_count: u32,
}

impl<'state_lf, 'bank_config> KFBank<'state_lf, 'bank_config> {
    /// Convert to an owned, borrow-free snapshot suitable for on-disk
    /// persistence (see [`KFBankSnapshot`]).
    pub fn to_snapshot(&self) -> KFBankSnapshot {
        KFBankSnapshot {
            hypotheses: self
                .hypotheses
                .iter()
                .map(Hypothesis::to_snapshot)
                .collect(),
            n_steps: self.n_steps,
            track_ids: self.track_ids.clone(),
            absolute_magnitude_estimate: self.absolute_magnitude_estimate,
            absolute_magnitude_sample_count: self.absolute_magnitude_sample_count,
        }
    }

    /// Reattach `shared_ctx`/`config` (supplied by the caller, who already
    /// holds the live [`KalmanContext`]/[`KFBankConfig`]) to rebuild a full
    /// [`KFBank`]. Goes through `Self::with_hypotheses`, so
    /// `best_index_cache` comes back correctly reset.
    pub fn from_snapshot(
        snapshot: KFBankSnapshot,
        shared_ctx: &'state_lf KalmanContext,
        config: &'bank_config KFBankConfig,
    ) -> Self {
        let hypotheses = snapshot
            .hypotheses
            .into_iter()
            .map(|h| h.into_hypothesis(shared_ctx))
            .collect();
        Self::with_hypotheses(
            hypotheses,
            config,
            snapshot.n_steps,
            snapshot.track_ids,
            snapshot.absolute_magnitude_estimate,
            snapshot.absolute_magnitude_sample_count,
        )
    }
}

impl<'state_lf, 'bank_config> Clone for KFBank<'state_lf, 'bank_config> {
    fn clone(&self) -> Self {
        Self {
            hypotheses: Arc::clone(&self.hypotheses),
            best_index_cache: AtomicUsize::new(self.best_index_cache.load(Ordering::Relaxed)),
            config: self.config,
            n_steps: self.n_steps,
            track_ids: self.track_ids.clone(),
            absolute_magnitude_estimate: self.absolute_magnitude_estimate,
            absolute_magnitude_sample_count: self.absolute_magnitude_sample_count,
        }
    }
}

impl<'state_lf, 'bank_config> KFBank<'state_lf, 'bank_config> {
    // ── Construction ──────────────────────────────────────────────────────

    /// Build a bank from pre-constructed `(state, weight)` seeds.
    ///
    /// Weights need not be normalized; they are converted to normalized
    /// log-weights internally.  Non-positive weights are silently ignored.
    ///
    /// The raw admissible-region grid can carry hundreds of nodes (up to
    /// `n_rho × n_rho_dot`) before any weight-based pruning — `post_step_cleanup`
    /// is applied once here, immediately, so a freshly seeded bank is capped
    /// the same way any subsequent real update would cap it. Without this, a
    /// lineage that never gets a second real observation (common: noise
    /// pairs, objects not revisited soon) would keep its full birth-time
    /// hypothesis count forever — `branch_null` deliberately never cleans up
    /// (see its doc), and `cap_schedule`/`min_hypotheses`/`weight_floor`
    /// would never even be consulted for such a bank.
    pub fn from_grid(
        obs_dataset: &ObsDataset,
        first_obs: &Observation,
        second_obs: &Observation,
        state: &'state_lf KalmanContext,
        grid_config: &GridConfig,
        bank_config: &'bank_config KFBankConfig,
    ) -> Result<Self, EngineError> {
        let seeds = admissible_region_grid(obs_dataset, first_obs, second_obs, state, grid_config)?;

        let hypotheses = seeds
            .into_iter()
            .filter(|(_, w)| *w > 0.0)
            .enumerate()
            .map(|(id, (kf, w))| Hypothesis {
                kf,
                log_weight: w.ln(),
                id: id as u64,
                recent_log_liks: VecDeque::new(),
            })
            .collect();

        let bank = Self::with_hypotheses(
            hypotheses,
            bank_config,
            0,
            vec![*first_obs.id(), *second_obs.id()],
            None,
            0,
        );
        // bank.post_step_cleanup();
        Ok(bank)
    }

    /// Sole constructor: builds a `KFBank` from a fresh `hypotheses` vector
    /// plus the rest of the bank's state. Always starts with an empty
    /// `best_index_cache` — correct by construction, since a freshly built
    /// `hypotheses` vector has never been scanned for its MAP hypothesis.
    #[allow(clippy::too_many_arguments)]
    fn with_hypotheses(
        hypotheses: Vec<Hypothesis<'state_lf>>,
        config: &'bank_config KFBankConfig,
        n_steps: usize,
        track_ids: Vec<ObsId>,
        absolute_magnitude_estimate: Option<f64>,
        absolute_magnitude_sample_count: u32,
    ) -> Self {
        Self {
            hypotheses: Arc::new(hypotheses),
            best_index_cache: AtomicUsize::new(usize::MAX),
            config,
            n_steps,
            track_ids,
            absolute_magnitude_estimate,
            absolute_magnitude_sample_count,
        }
    }

    /// Replace `hypotheses` wholesale (e.g. after gating/scoring produced a
    /// brand-new survivor list). Invalidates `best_index_cache` — the only
    /// way this should happen is through this method or
    /// [`Self::hypotheses_mut`].
    fn set_hypotheses(&mut self, new: Vec<Hypothesis<'state_lf>>) {
        self.best_index_cache.store(usize::MAX, Ordering::Relaxed);
        self.hypotheses = Arc::new(new);
    }

    /// Mutable access to `hypotheses` for in-place pruning/merging
    /// (`retain`, `sort_by`, `truncate`, `iter_mut`, `mem::take`).
    /// Invalidates `best_index_cache` unconditionally — cheap, and correct
    /// even for mutations that turn out not to move the MAP hypothesis.
    fn hypotheses_mut(&mut self) -> &mut Vec<Hypothesis<'state_lf>> {
        self.best_index_cache.store(usize::MAX, Ordering::Relaxed);
        Arc::make_mut(&mut self.hypotheses)
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    /// Number of live hypotheses.
    pub fn len(&self) -> usize {
        self.hypotheses.len()
    }

    /// Whether this bank holds no live hypotheses.
    pub fn is_empty(&self) -> bool {
        self.hypotheses.is_empty()
    }

    /// Ids of every observation associated to this bank so far, in
    /// chronological order (the two seed-pair observations, then one id per
    /// successful [`Self::branch_with`] call).
    pub fn track_ids(&self) -> &[ObsId] {
        &self.track_ids
    }

    /// Running absolute-magnitude (`H`) estimate built from every observation
    /// associated so far, or `None` before the first successful
    /// [`Self::branch_with`] call. Feeds the null-branch detection
    /// probability (see
    /// [`branching::detection_probability`](crate::topocentric_kf::branching::detection_probability)).
    pub fn absolute_magnitude_estimate(&self) -> Option<f64> {
        self.absolute_magnitude_estimate
    }

    /// How many observations were folded into
    /// [`Self::absolute_magnitude_estimate`] — the evidence volume behind
    /// that running mean, needed by anything comparing two banks'
    /// photometry (e.g. fragment linkage).
    pub fn absolute_magnitude_sample_count(&self) -> u32 {
        self.absolute_magnitude_sample_count
    }

    /// `true` if at least one hypothesis is alive.
    pub fn is_alive(&self) -> bool {
        !self.hypotheses.is_empty()
    }

    /// Immutable view of the live hypotheses.
    pub fn hypotheses(&self) -> &[Hypothesis<'state_lf>] {
        &self.hypotheses
    }

    /// The maximum-a-posteriori (highest log-weight) hypothesis, if any.
    ///
    /// Cached: repeated calls on an unchanged bank (the common case — e.g.
    /// a lineage re-checked against many visits without a nearby candidate)
    /// are O(1) instead of rescanning all hypotheses every time. See
    /// `best_index_cache`.
    pub fn best(&self) -> Option<&Hypothesis<'state_lf>> {
        let cached = self.best_index_cache.load(Ordering::Relaxed);
        if cached != usize::MAX && cached < self.hypotheses.len() {
            return self.hypotheses.get(cached);
        }

        let idx = self
            .hypotheses
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.log_weight.partial_cmp(&b.log_weight).unwrap())
            .map(|(idx, _)| idx)?;
        self.best_index_cache.store(idx, Ordering::Relaxed);
        self.hypotheses.get(idx)
    }

    /// Effective sample size `1 / Σ wᵢ²` (weights assumed normalized).
    ///
    /// A value close to 1 means one hypothesis dominates; a value close to
    /// `len()` means the weights are approximately uniform.
    pub fn effective_sample_size(&self) -> f64 {
        let sum_sq: f64 = self.hypotheses.iter().map(|h| h.weight().powi(2)).sum();
        if sum_sq > 0.0 { 1.0 / sum_sq } else { 0.0 }
    }

    /// Weighted mean of the 6-D attributable state across all hypotheses.
    ///
    /// **Note:** only meaningful once the mixture is roughly unimodal.  While
    /// several range modes coexist the MAP hypothesis ([`KFBank::best`]) is a
    /// better point estimate.  Averaging angular components $(\alpha, \delta)$
    /// linearly does not account for the $0/2\pi$ wraparound.
    pub fn weighted_mean_state(&self) -> Option<Vector6<f64>> {
        if self.hypotheses.is_empty() {
            return None;
        }
        let mean = self
            .hypotheses
            .iter()
            .fold(Vector6::zeros(), |acc, h| acc + h.kf.state * h.weight());
        Some(mean)
    }

    // ── Main step ─────────────────────────────────────────────────────────

    /// Process one new observation: propagate → score → update → prune → merge.
    ///
    /// The pipeline for each hypothesis is:
    /// 1. **Propagate** to the observation epoch (Kepler + covariance transport).
    /// 2. **Gate** on the Mahalanobis² distance (the MAP hypothesis is exempt).
    /// 3. **Score** with the predictive log-likelihood.
    /// 4. **Update** the Kalman state.
    /// 5. **Push** the log-likelihood into the sliding window.
    ///
    /// After all hypotheses are processed, the bank runs:
    /// - Normalize weights.
    /// - Prune: smoothed-score floor (or classic weight floor if window = 0).
    /// - Apply scheduled cap (decay curve).
    /// - Merge spatially coincident modes.
    /// - Re-normalize.
    pub fn step(&mut self, obs_dataset: &ObsDataset, obs: &Observation) -> BankStep {
        let geometry = self.resolve_step_geometry(obs_dataset, obs);
        self.step_impl(geometry, obs)
    }

    /// Same as [`Self::step`], but with the observer heliocentric geometry
    /// at `obs`'s epoch supplied by the caller instead of resolved
    /// internally.
    ///
    /// For a caller that already knows `(r_obs, v_obs)` for this epoch —
    /// e.g. from a precomputed per-observation cache, or from its own
    /// `predict_search_region` call moments earlier — this skips the ephemeris lookup that [`Self::step`]
    /// would otherwise perform. See [`Self::resolve_step_geometry`]'s doc
    /// for why that lookup is worth avoiding when it's already available.
    pub fn step_with_geometry(
        &mut self,
        r_obs: Vector3<f64>,
        v_obs: Vector3<f64>,
        obs: &Observation,
    ) -> BankStep {
        self.step_impl(Some((r_obs, v_obs)), obs)
    }

    /// Resolve the observer heliocentric geometry needed to propagate every
    /// hypothesis to `obs`'s epoch — **once**, not once per hypothesis.
    ///
    /// This geometry depends only on `(obs_dataset, obs)`, never on which
    /// hypothesis is asking for it: every live hypothesis in a bank shares
    /// the exact same `shared_ctx` by construction (see the module-level
    /// "Lifetime" doc). Reading it off `self.hypotheses.first()` and
    /// resolving it once is therefore exactly equivalent to — but far
    /// cheaper than — resolving it independently inside each hypothesis's
    /// own propagate call (the ephemeris lookup itself, not the per-hypothesis
    /// Kepler solve that follows it, is the expensive part).
    ///
    /// `None` if the bank has no hypotheses to read a context from, or if
    /// the observer/ephemeris lookup itself fails — a failure every
    /// hypothesis would have hit identically, for the same reason.
    pub fn resolve_step_geometry(
        &self,
        obs_dataset: &ObsDataset,
        obs: &Observation,
    ) -> Option<(Vector3<f64>, Vector3<f64>)> {
        let context = self.hypotheses.first()?.kf.shared_ctx;
        let observer = get_observer(obs_dataset, obs).ok()?;
        let helio_state = context
            .get_ephem()
            .helio_observer_state(observer, obs.mjd_tt())
            .ok()?;
        Some((helio_state.helio_cart_pos, helio_state.helio_cart_vel))
    }

    /// Shared body of [`Self::step`]/[`Self::step_with_geometry`]: propagate
    /// → score → update → prune → merge.
    ///
    /// The pipeline for each hypothesis is:
    /// 1. **Propagate** to the observation epoch (Kepler + covariance transport).
    /// 2. **Gate** on the Mahalanobis² distance (the MAP hypothesis is exempt).
    /// 3. **Score** with the predictive log-likelihood.
    /// 4. **Update** the Kalman state.
    /// 5. **Push** the log-likelihood into the sliding window.
    ///
    /// After all hypotheses are processed, the bank runs:
    /// - Normalize weights.
    /// - Prune: smoothed-score floor (or classic weight floor if window = 0).
    /// - Apply scheduled cap (decay curve).
    /// - Merge spatially coincident modes.
    /// - Re-normalize.
    ///
    /// `geometry: None` (only reachable from [`Self::step`] when
    /// [`Self::resolve_step_geometry`] itself fails) treats every
    /// hypothesis as a propagation failure, matching what would happen if
    /// each one independently hit the same ephemeris-lookup error.
    fn step_impl(
        &mut self,
        geometry: Option<(Vector3<f64>, Vector3<f64>)>,
        obs: &Observation,
    ) -> BankStep {
        let epoch = obs.mjd_tt();
        let n_before = self.hypotheses.len();

        let _enter = BankEvent::span(epoch, n_before).entered();

        BankEvent::StepStart {
            n_hypotheses: n_before,
        }
        .emit();

        let (survivors, n_gated, n_failed) = self.process_hypotheses(geometry, epoch, obs);

        BankEvent::PredictUpdateComplete {
            n_survivors: survivors.len(),
            n_gated,
            n_failed,
        }
        .emit();

        self.set_hypotheses(survivors);

        // Compute the scheduled cap *before* cleanup so we can report it.
        let scheduled_cap = self
            .config
            .cap_schedule
            .cap(self.n_steps)
            .max(self.config.min_hypotheses.max(1));

        self.post_step_cleanup();

        // Increment the observation counter *after* the step so that cap(0)
        // applies to the very first observation (full initial population).
        self.n_steps += 1;

        BankStep {
            epoch,
            n_before,
            n_after: self.hypotheses.len(),
            n_gated,
            n_failed,
            n_effective: self.effective_sample_size(),
            best_weight: self.best().map(|h| h.weight()).unwrap_or(0.0),
            collapsed: self.hypotheses.is_empty(),
            scheduled_cap,
        }
    }

    // ── Internal predict/update pipeline ─────────────────────────────────

    /// Run the predict/score/update cycle over all current hypotheses.
    ///
    /// Identifies the top `min_hypotheses` hypotheses **before** the step so
    /// they are marked as exempt from the chi-square gate.
    ///
    /// Protecting only the MAP (single best) was insufficient: if the MAP
    /// survived but the next 4 best hypotheses were gated out on the same
    /// step, the bank collapsed to 1 even though `min_hypotheses = 5`.  The
    /// weight-floor pruning already respects `min_hypotheses`; the gate must
    /// honour the same floor.
    ///
    /// Returns `(survivors, n_gated, n_failed)`.
    ///
    /// Composed of [`Self::propagate_in_place`] followed by
    /// [`Self::score_and_update_hypotheses`] — split into two phases so that
    /// [`Self::branch_with`] can reuse only the second phase on a bank that
    /// was already propagated once via [`Self::predict_to`].
    fn process_hypotheses(
        &mut self,
        geometry: Option<(Vector3<f64>, Vector3<f64>)>,
        epoch: f64,
        obs: &Observation,
    ) -> (Vec<Hypothesis<'state_lf>>, usize, usize) {
        let n_propagate_failed = self.propagate_in_place(geometry, epoch);
        let (survivors, n_gated, n_score_failed) = self.score_and_update_hypotheses(obs);
        (survivors, n_gated, n_score_failed + n_propagate_failed)
    }

    /// Propagate every live hypothesis to `epoch`, in place, using the
    /// already-resolved `geometry` (see [`Self::resolve_step_geometry`]) —
    /// reuses [`Self::predict_hypotheses`], the same per-hypothesis
    /// Kepler-propagation primitive [`Self::predict_to`] uses, so a
    /// hypothesis that fails here (Kepler solver or Jacobian failure) fails
    /// for exactly the same reasons it would have under the old
    /// per-hypothesis-lookup code path.
    ///
    /// `geometry: None` drops every hypothesis (see [`Self::step_impl`]'s doc).
    ///
    /// # Returns
    /// The number of hypotheses dropped due to a propagation failure.
    fn propagate_in_place(
        &mut self,
        geometry: Option<(Vector3<f64>, Vector3<f64>)>,
        epoch: f64,
    ) -> usize {
        let n_before = self.hypotheses.len();
        let survivors = match geometry {
            Some((r_obs, v_obs)) => self.predict_hypotheses(epoch, r_obs, v_obs),
            None => Vec::new(),
        };
        self.set_hypotheses(survivors);
        n_before - self.hypotheses.len()
    }

    /// Gate, score and update every live hypothesis against `obs`, assuming
    /// they are **already propagated** to `obs`'s epoch.
    ///
    /// Identifies the top `min_hypotheses` hypotheses **before** the step so
    /// they are marked as exempt from the chi-square gate.
    ///
    /// Protecting only the MAP (single best) was insufficient: if the MAP
    /// survived but the next 4 best hypotheses were gated out on the same
    /// step, the bank collapsed to 1 even though `min_hypotheses = 5`.  The
    /// weight-floor pruning already respects `min_hypotheses`; the gate must
    /// honour the same floor.
    ///
    /// # Returns
    /// `(survivors, n_gated, n_failed)`.
    fn score_and_update_hypotheses(
        &mut self,
        obs: &Observation,
    ) -> (Vec<Hypothesis<'state_lf>>, usize, usize) {
        // Compute the protected set *before* weights change.
        // Every hypothesis in the top `min_hypotheses` by current log-weight
        // is immune to the chi-square gate for this step.
        let protected_ids = top_k_ids(&self.hypotheses, self.config.min_hypotheses.max(1));

        let results: Vec<HypothesisStepResult<'state_lf>> = std::mem::take(self.hypotheses_mut())
            .into_iter()
            .map(|hyp| {
                let is_protected = protected_ids.contains(&hyp.id);
                hyp.finalize_score_and_update(self.config, obs, is_protected)
            })
            .collect();

        let n_gated = results
            .iter()
            .filter(|r| matches!(r, HypothesisStepResult::Gated))
            .count();
        let n_failed = results
            .iter()
            .filter(|r| matches!(r, HypothesisStepResult::Failed))
            .count();
        let survivors = results
            .into_iter()
            .filter_map(|r| match r {
                HypothesisStepResult::Survived(h) => Some(*h),
                _ => None,
            })
            .collect();

        (survivors, n_gated, n_failed)
    }

    // ── Branching primitives ─────────────────────────────────────────────
    //
    // NOTE ON TEST COVERAGE: every `KFState` borrows a live `KalmanContext`
    // (JPL ephemeris + UT1 provider, loaded over the network), so no
    // `KFBank`/`Hypothesis` value — and therefore none of `predict_to`,
    // `branch_with` or `branch_null` — can be constructed in a fast, offline
    // unit test. This is a pre-existing limitation of the crate (`step()`
    // itself has never had unit tests for the same reason); it is not
    // introduced by this branching work. These primitives are exercised by
    // `cargo build`/`cargo clippy` plus manual review here; correctness is
    // instead covered indirectly through the pure-function tests in
    // `topocentric_kf::branching` (LLR scoring, clutter density, detection
    // probability, candidate search), which do not require a `KFState`.

    /// Propagate every hypothesis to `t_prop`, read-only, without consuming
    /// an observation.
    ///
    /// This is the shared "predict" half of a branching step: propagation is
    /// the expensive two-body operation, so it is paid **once** per bank and
    /// reused by every branch spawned from the result (via
    /// [`Self::branch_with`] / [`Self::branch_null`]), instead of being
    /// repeated per candidate observation.
    ///
    /// # Arguments
    /// * `t_prop` – Target epoch (MJD TT).
    /// * `r_obs_new`, `v_obs_new` – Observer heliocentric state at `t_prop`.
    ///
    /// # Returns
    /// A new bank whose hypotheses are predicted to `t_prop`. Hypotheses that
    /// fail to propagate are dropped (logged at `trace`). `n_steps` and
    /// `track_ids` are left unchanged — no observation has been consumed yet.
    pub fn predict_to(
        &self,
        t_prop: f64,
        r_obs_new: Vector3<f64>,
        v_obs_new: Vector3<f64>,
    ) -> Self {
        Self::with_hypotheses(
            self.predict_hypotheses(t_prop, r_obs_new, v_obs_new),
            self.config,
            self.n_steps,
            self.track_ids.clone(),
            self.absolute_magnitude_estimate,
            self.absolute_magnitude_sample_count,
        )
    }

    /// Predict every hypothesis to `t_prop`, keeping each hypothesis's
    /// weight, id and likelihood window unchanged — only `kf` moves.
    ///
    /// Used by [`Self::predict_to`], the sole propagation entry point —
    /// [`predict_search_region`](super::ellipse_region_finder::KFBank::predict_search_region)
    /// now goes through `predict_to` too (via
    /// [`search_region`](super::ellipse_region_finder::KFBank::search_region)),
    /// rather than propagating a second time.
    fn predict_hypotheses(
        &self,
        t_prop: f64,
        r_obs_new: Vector3<f64>,
        v_obs_new: Vector3<f64>,
    ) -> Vec<Hypothesis<'state_lf>> {
        self.hypotheses
            .iter()
            .filter_map(|hyp| match hyp.kf.predict(t_prop, r_obs_new, v_obs_new) {
                Ok(kf) => Some(Hypothesis { kf, ..hyp.clone() }),
                Err(error) => {
                    BankEvent::PredictionFailed {
                        hyp_id: hyp.id,
                        error: format!("{error:?}"),
                    }
                    .emit();
                    None
                }
            })
            .collect()
    }

    /// Branch this (already [`Self::predict_to`]'d) bank by applying a
    /// Kalman update with `obs` — the "observation branch" primitive of the
    /// track-oriented MHT scheme.
    ///
    /// `self` must already be at `obs`'s epoch: this method does **not**
    /// re-propagate, it only gates/scores/updates (see
    /// `Self::score_and_update_hypotheses`) and runs the usual intra-bank
    /// cleanup (prune → cap → merge, see `Self::post_step_cleanup`).
    ///
    /// # Arguments
    /// * `obs` – Candidate observation to associate with this branch.
    ///
    /// # Returns
    /// * `Some((branch, mixture_likelihood_z))` – The branched bank, with
    ///   `track_ids` extended by `obs.id()`, and the **pre-update** mixture
    ///   predictive likelihood of `obs` under `self` — the `L(z)` term
    ///   consumed by the branch's log-likelihood-ratio score.
    /// * `None` – Every hypothesis was gated or failed; this branch is not
    ///   viable.
    pub fn branch_with(&self, obs: &Observation) -> Option<(Self, f64)> {
        self.branch_with_diag(obs).0
    }

    /// Like [`Self::branch_with`], but also reports how many hypotheses
    /// were rejected by the chi-square gate vs. failed numerically
    /// (propagation/innovation/update failure) — the same breakdown
    /// [`Self::step`] already exposes via `BankStep`'s `n_gated`/`n_failed`,
    /// for this single-observation branching path. Used by diagnostic
    /// tooling (`mot_analysis`) to classify *why* a lineage's real-
    /// observation update collapsed, not just *that* it did.
    ///
    /// Returns `(branch_with_result, n_gated, n_failed)`.
    pub fn branch_with_diag(&self, obs: &Observation) -> (Option<(Self, f64)>, usize, usize) {
        let mixture_likelihood_z = self.mixture_predictive_likelihood(obs);

        let mut branch = self.clone();
        let (survivors, n_gated, n_failed) = branch.score_and_update_hypotheses(obs);
        branch.set_hypotheses(survivors);
        if branch.hypotheses.is_empty() {
            return (None, n_gated, n_failed);
        }

        branch.post_step_cleanup();
        branch.n_steps += 1;
        branch.track_ids.push(*obs.id());
        branch.update_absolute_magnitude_estimate(obs);

        (Some((branch, mixture_likelihood_z)), n_gated, n_failed)
    }

    /// Fold the apparent magnitude of a just-associated observation into the
    /// bank's running absolute-magnitude estimate, using the MAP
    /// hypothesis's geometry at the observation's epoch.
    ///
    /// Silently a no-op if the bank collapsed to no hypotheses (already
    /// guarded against by [`Self::branch_with`]'s early return) — kept
    /// defensive here since [`Self::best`] is a plain `Option`.
    fn update_absolute_magnitude_estimate(&mut self, obs: &Observation) {
        let Some(best) = self.best() else {
            return;
        };

        let r_helio_au = best.kf.to_cartesian().pos.norm();
        let delta_topocentric_au = best.kf.state[4];
        let implied_h = implied_absolute_magnitude(
            obs.photometry().magnitude,
            r_helio_au,
            delta_topocentric_au,
        );

        let (mean, count) = update_running_magnitude_estimate(
            self.absolute_magnitude_estimate,
            self.absolute_magnitude_sample_count,
            implied_h,
        );
        self.absolute_magnitude_estimate = Some(mean);
        self.absolute_magnitude_sample_count = count;
    }

    /// Branch this (already [`Self::predict_to`]'d) bank as the "null"
    /// (missed-detection) hypothesis: no observation is associated.
    ///
    /// Mandatory at LSST cadence: without it, any field not revisited (or an
    /// object dipping below the limiting magnitude) would force a false
    /// association or kill the bank outright. State, weights, `track_ids`
    /// and `n_steps` are all left unchanged — no observation was processed,
    /// so nothing about the bank's posterior or its hypothesis-cap-schedule
    /// pacing should move. `n_steps` deliberately does **not** advance here
    /// (unlike earlier revisions of this method): at LSST cadence a single
    /// quiet night can call `branch_null` hundreds of times (once per visit
    /// with no candidate), and `HypothesisCapSchedule` is documented as "a
    /// function of the number of observations processed" — counting null
    /// branches would collapse the hypothesis population almost
    /// immediately, defeating the schedule's purpose.
    pub fn branch_null(&self) -> Self {
        self.clone()
    }

    /// Mixture predictive likelihood $\ell(z) = \sum_i w_i\,\mathcal{N}(z;\,
    /// \mu_i,\, S_i)$ of `obs` under this (already-propagated) bank's
    /// hypotheses, evaluated *before* any update is applied.
    ///
    /// Reuses [`SearchComponent`] purely for its Gaussian-density evaluation;
    /// the gate threshold it carries is irrelevant here (no gating is
    /// performed) and is set to `0.0`.
    fn mixture_predictive_likelihood(&self, obs: &Observation) -> f64 {
        let coord = obs.equ_coord();
        let observation_noise = Matrix2::from_diagonal(&Vector2::new(
            coord.ra_error * coord.ra_error,
            coord.dec_error * coord.dec_error,
        ));

        self.hypotheses
            .iter()
            .filter_map(|hyp| {
                let sky_covariance = hyp.kf.sky_covariance().ok()? + observation_noise;
                SearchComponent::new(
                    hyp.weight(),
                    hyp.kf.state[0],
                    hyp.kf.state[1],
                    sky_covariance,
                    0.0,
                )
            })
            .map(|component| component.weighted_density(coord.ra, coord.dec))
            .sum()
    }

    // ── Post-step cleanup pipeline ────────────────────────────────────────

    /// Full post-step cleanup: normalize → prune → cap → merge → normalize.
    fn post_step_cleanup(&mut self) {
        self.normalize_weights();

        let n0 = self.hypotheses.len();
        self.prune_by_smoothed_score();
        let n1 = self.hypotheses.len();
        if n0 != n1 {
            BankEvent::PruningPhase {
                phase: "smoothed_prune",
                n_before: n0,
                n_after: n1,
            }
            .emit();
        }

        let n2 = n1;
        self.cap_to_scheduled_max();
        let n3 = self.hypotheses.len();
        if n2 != n3 {
            BankEvent::PruningPhase {
                phase: "scheduled_cap",
                n_before: n2,
                n_after: n3,
            }
            .emit();
        }

        let n4 = n3;
        self.merge_coincident_modes();
        let n5 = self.hypotheses.len();
        if n4 != n5 {
            BankEvent::PruningPhase {
                phase: "merge",
                n_before: n4,
                n_after: n5,
            }
            .emit();
        }

        self.normalize_weights();
    }

    /// Prune low-quality hypotheses using a sliding-window likelihood score.
    ///
    /// # Strategy
    ///
    /// When `likelihood_window > 0` (smoothed mode):
    ///
    /// Each hypothesis carries a bounded deque of its last `W` per-step
    /// log-likelihoods.  The **smoothed score** is the mean of this deque.
    /// A hypothesis is pruned if its smoothed score falls more than
    /// `−ln(weight_floor)` below the best smoothed score:
    ///
    /// ```text
    /// prune if: smoothed(h) < smoothed(best) + ln(weight_floor)
    /// ```
    ///
    /// The top `min_hypotheses` by smoothed score are **always protected**,
    /// regardless of the threshold, to prevent the bank from collapsing to
    /// fewer modes than desired.
    ///
    /// When `likelihood_window == 0` (classic mode), falls back to the
    /// original raw weight-floor approach on the cumulative `log_weight`.
    fn prune_by_smoothed_score(&mut self) {
        let window = self.config.likelihood_window;

        if window == 0 {
            // Classic path: weight floor on the cumulative Bayesian weight.
            self.prune_by_weight_floor();
            return;
        }

        let min_keep = self.config.min_hypotheses.max(1);
        if self.hypotheses.len() <= min_keep {
            // Nothing left to prune; protecting min_hypotheses already.
            return;
        }

        // Best smoothed score across all live hypotheses.
        let best_smoothed = self
            .hypotheses
            .iter()
            .map(|h| h.smoothed_log_lik())
            .fold(f64::NEG_INFINITY, f64::max);

        if !best_smoothed.is_finite() {
            // No hypothesis has accumulated any window data yet — skip.
            BankEvent::SmoothedPruningSkipped.emit();
            return;
        }

        // The score that the min_keep-th best hypothesis achieves.
        // Any hypothesis at or above this floor is in the "protected" group.
        let protection_floor = nth_largest_smoothed_score(&self.hypotheses, min_keep);

        // Relative threshold: a hypothesis must be within weight_floor of the best.
        let log_threshold = best_smoothed + self.config.weight_floor.ln();

        let n_before = self.hypotheses.len();

        // Retain a hypothesis if EITHER:
        //   (a) its smoothed score is above the weight-floor threshold, OR
        //   (b) it is among the top `min_keep` (score >= protection_floor).
        self.hypotheses_mut().retain(|h| {
            let s = h.smoothed_log_lik();
            s >= log_threshold || s >= protection_floor
        });

        let n_removed = n_before - self.hypotheses.len();
        if n_removed > 0 {
            BankEvent::SmoothedPruning {
                n_removed,
                log_threshold,
                best_smoothed,
                window,
                min_kept: min_keep,
            }
            .emit();
        }
    }

    /// Classic weight-floor pruning on the cumulative `log_weight`.
    ///
    /// Used as the fallback when `likelihood_window == 0`, and also respects
    /// the `min_hypotheses` floor.
    fn prune_by_weight_floor(&mut self) {
        let min_keep = self.config.min_hypotheses.max(1);
        if self.hypotheses.len() <= min_keep {
            return;
        }

        // Score of the min_keep-th best hypothesis by cumulative weight.
        let protection_floor = nth_largest_log_weight(&self.hypotheses, min_keep);

        let floor_log = self.config.weight_floor.ln(); // e.g. ln(1e-4) ≈ −9.2
        // After normalization, the best hypothesis has log_weight ≈ 0, so the
        // threshold is just floor_log.
        let n_before = self.hypotheses.len();

        self.hypotheses_mut()
            .retain(|h| h.log_weight >= floor_log || h.log_weight >= protection_floor);

        let n_removed = n_before - self.hypotheses.len();
        if n_removed > 0 {
            BankEvent::WeightFloorPruning {
                n_removed,
                weight_floor: self.config.weight_floor,
                min_kept: min_keep,
            }
            .emit();
        }
    }

    /// Apply the scheduled hypothesis cap, respecting the `min_hypotheses` floor.
    ///
    /// Keeps the `effective_cap` highest-weighted hypotheses (by `log_weight`),
    /// where `effective_cap = max(schedule.cap(n_steps), min_hypotheses)`.
    fn cap_to_scheduled_max(&mut self) {
        let raw_cap = self.config.cap_schedule.cap(self.n_steps);
        let effective_cap = raw_cap.max(self.config.min_hypotheses.max(1));

        if self.hypotheses.len() <= effective_cap {
            return;
        }

        // Sort descending by cumulative weight to keep the most plausible ones.
        let hypotheses = self.hypotheses_mut();
        hypotheses.sort_by(|a, b| b.log_weight.partial_cmp(&a.log_weight).unwrap());

        let n_truncated = hypotheses.len() - effective_cap;
        hypotheses.truncate(effective_cap);

        BankEvent::ScheduledCapTruncation {
            n_truncated,
            effective_cap,
            raw_cap,
            n_steps: self.n_steps,
        }
        .emit();
    }

    /// Greedily merge modes whose mean heliocentric positions are within
    /// `merge_position_au`.
    fn merge_coincident_modes(&mut self) {
        let thresh = self.config.merge_position_au;
        let mut merged: Vec<Hypothesis<'state_lf>> = Vec::with_capacity(self.hypotheses.len());

        for hyp in std::mem::take(self.hypotheses_mut()) {
            match merged
                .iter_mut()
                .find(|m| m.kf.position_distance_au(&hyp.kf) < thresh)
            {
                Some(existing) => {
                    BankEvent::ModesMerged {
                        hyp_id_kept: existing.id,
                        hyp_id_merged: hyp.id,
                        threshold_au: thresh,
                    }
                    .emit();
                    *existing = existing.moment_match_merge(&hyp);
                }
                None => merged.push(hyp),
            }
        }

        self.set_hypotheses(merged);
    }

    /// Normalize log-weights so that `Σ exp(log_weight) = 1`, using the
    /// log-sum-exp trick for numerical stability.
    fn normalize_weights(&mut self) {
        if self.hypotheses.is_empty() {
            return;
        }

        let max = self
            .hypotheses
            .iter()
            .map(|h| h.log_weight)
            .fold(f64::NEG_INFINITY, f64::max);

        if !max.is_finite() {
            // All weights underflowed — fall back to uniform to avoid losing
            // the bank entirely rather than declaring the filter lost.
            let uniform = -(self.hypotheses.len() as f64).ln();
            self.hypotheses_mut()
                .iter_mut()
                .for_each(|h| h.log_weight = uniform);
            return;
        }

        let log_norm = max
            + self
                .hypotheses
                .iter()
                .map(|h| (h.log_weight - max).exp())
                .sum::<f64>()
                .ln();

        self.hypotheses_mut()
            .iter_mut()
            .for_each(|h| h.log_weight -= log_norm);
    }
}

// ── Free functions ────────────────────────────────────────────────────────────

/// Collect the ids of the `k` highest-weight hypotheses (by `log_weight`).
///
/// Used by [`KFBank::process_hypotheses`] to build the protected set before
/// the predict/gate/update cycle: any hypothesis whose id is in this set is
/// exempt from the chi-square gate for the current step.
///
/// Returns a `Vec` rather than a `HashSet` because `k` (= `min_hypotheses`)
/// is typically small (≤ 10), making a linear `.contains()` search cheaper
/// than the overhead of hashing.
fn top_k_ids(hypotheses: &[Hypothesis<'_>], k: usize) -> Vec<u64> {
    if k == 0 || hypotheses.is_empty() {
        return Vec::new();
    }
    // Partial sort descending by log_weight; no full sort needed.
    let actual_k = k.min(hypotheses.len());
    let mut indexed: Vec<(f64, u64)> = hypotheses.iter().map(|h| (h.log_weight, h.id)).collect();
    // select_nth_unstable_by places the top `actual_k` elements in the first
    // `actual_k` slots (unordered) — O(n) average.
    indexed.select_nth_unstable_by(actual_k - 1, |a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed[..actual_k].iter().map(|(_, id)| *id).collect()
}

/// Smoothed score of the `k`-th best hypothesis (1-indexed) by smoothed log-likelihood.
///
/// Used to compute the protection floor for `prune_by_smoothed_score`:
/// any hypothesis with a score ≥ this value is in the protected top-k group.
///
/// Returns `f64::NEG_INFINITY` if `k` exceeds the number of hypotheses (all
/// are protected).
fn nth_largest_smoothed_score(hypotheses: &[Hypothesis<'_>], k: usize) -> f64 {
    if k >= hypotheses.len() {
        return f64::NEG_INFINITY;
    }
    let mut scores: Vec<f64> = hypotheses.iter().map(|h| h.smoothed_log_lik()).collect();
    // Partial sort: we only need the k-th element (0-indexed: k-1).
    scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    scores[k - 1]
}

/// Cumulative log-weight of the `k`-th best hypothesis (1-indexed).
///
/// Used by `prune_by_weight_floor` to compute the protection floor.
/// Returns `f64::NEG_INFINITY` if `k` exceeds the number of hypotheses.
fn nth_largest_log_weight(hypotheses: &[Hypothesis<'_>], k: usize) -> f64 {
    if k >= hypotheses.len() {
        return f64::NEG_INFINITY;
    }
    let mut weights: Vec<f64> = hypotheses.iter().map(|h| h.log_weight).collect();
    weights.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    weights[k - 1]
}
