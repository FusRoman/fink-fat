//! Shadow study of fragment linkage: would merging arcs recover completeness
//! without manufacturing contamination?
//!
//! # Why measure before wiring
//!
//! 42 % of trackable trajectories come out fragmented, and 40 676 of them are
//! split into pieces that are each individually pure — reconstructions that
//! are already correct and only need stitching. That is the largest remaining
//! completeness deficit.
//!
//! But a *wrong* merge is worse than a missed one: it fabricates a
//! contaminated trajectory out of two clean ones. So the linkage criteria in
//! [`fink_fat_engine::topocentric_kf::branching::merge`] are deliberately left
//! unwired, and this module scores them against ground truth first. Nothing
//! here merges anything — it only answers "how many of the merges these
//! thresholds would make are wrong?".
//!
//! # How the ground truth is obtained
//!
//! A fragment is one entry of
//! [`all_reconstructions`](crate::snapshot_report::efficacy::all_reconstructions)
//! — live branches followed by archived arcs, the exact pool (and order) the
//! efficacy report scores. Classifying each fragment's `track_ids` with
//! [`ObsTrajLookup::classify`] tells us which object it belongs to, so:
//!
//! * two `Pure` fragments of the **same** trajectory *should* be linked —
//!   these are the ground-truth positives, and their count is the recall
//!   denominator;
//! * two `Pure` fragments of **different** trajectories must **never** be
//!   linked — every such link is exactly the contamination to avoid.
//!
//! # Reading the report
//!
//! The acceptance bar is **zero wrong merges** on the retained cascade,
//! broken down by population so a handful of NEO/Centaur/KBO errors cannot
//! hide inside an MBA-dominated aggregate. Recall is secondary: a criterion
//! that links nothing is useless but harmless, one that links wrongly is not.
//!
//! Read the orbital index's own recall first — it caps everything downstream.
//! If the index never proposes a true pair, no threshold can recover it, and
//! the deficit belongs to the pairing, not to the gates.

use ahash::{AHashMap, AHashSet};
use photom::{
    TrajId,
    observation_dataset::{ObsDataset, ObsId},
};
use rayon::prelude::*;

use fink_fat_engine::{
    engine_config::kalman_context::KalmanContext,
    spacetime_bucket::{
        healpix_binner::HealpixBinner,
        spatial_binner::{SpatialBinner, SpatialKey},
        time_binner::{TimeBin, TimeBinner},
        uniform_time_binner::UniformTimeBinner,
    },
    topocentric_kf::{
        branching::{
            BranchCollection,
            merge::{
                LinkTestPoint, arc_is_converged, attributable_at, orbit_elements, rates_compatible,
                sequential_fit, sky_cell_key, temporally_disjoint,
            },
        },
        single_kalman::{KFState, propagate::PropagateError},
    },
};

use crate::{
    population::Population,
    seed_bank_report::ground_truth::{ObsTrajLookup, SeedPurity},
    snapshot_report::efficacy::all_reconstructions,
};

/// HEALPix depth of the attributable index's sky cell. Depth 5 is ~1.8°
/// across — coarse enough to absorb the propagation error to the reference
/// epoch, fine enough that a cell holds few unrelated arcs.
const INDEX_HEALPIX_DEPTH: u8 = 5;
/// Width of one index time slice, days.
///
/// One night: arcs are nightly tracklets, and within a night an object moves
/// far less than a cell, so a finer slice would only multiply propagations
/// without separating anything.
const TIME_SLICE_DAYS: f64 = 1.0;

/// How far, in days, a predictor is carried either side of its own span to look
/// for a continuation.
///
/// This is a hard ceiling on recall — a true pair separated by more than this
/// is never proposed — so the report counts the true pairs it excludes rather
/// than leaving the cap silent. At 30 days it cost 3391 true pairs, 21 % of the
/// total; the budget freed by the tighter rate gate below is spent here.
const MAX_GAP_DAYS: f64 = 60.0;

/// Rate agreement required to survive the index, radians/day (3 arcmin/day).
///
/// Sized on measurement, not caution. At 30 arcmin/day this gate pruned nothing
/// (6.48 M candidate pairs against 6.42 M) and the pool reached 6.98 M — 6.4×
/// the previous index, which multiplies false merges by the same factor at
/// equal specificity. The cascade sweep showed `rate <= 3` keeping **6503 of
/// 7041 correct pairs (92 %) while dividing the pool by four**: proper motion
/// is nearly independent of the position residual, which makes it the cheapest
/// specificity available.
const INDEX_MAX_RATE_DIFF_RAD_PER_DAY: f64 = 8.726_646_259_971_648e-4;

/// How many observations of the tested arc are walked through.
///
/// Each costs a Kepler propagation plus a Kalman update. A spread sample is
/// nearly as decisive as the whole arc: the covariance collapses after the
/// first one or two absorbed points, so the discrimination is already there
/// by the third.
const MAX_TEST_POINTS: usize = 5;

/// Minimum observations an arc needs to enter the candidate pool.
///
/// The dominant lever of the whole study: it acts on the base rate rather
/// than on any single criterion. See
/// [`MergeParams::min_arc_points`](fink_fat_engine::topocentric_kf::branching::merge::MergeParams::min_arc_points).
const MIN_ARC_POINTS: usize = 6;

/// Photometric thresholds swept in one pass (magnitudes). `INFINITY` = off.
///
/// Sized for a *phase-corrected* `H`. Once the H-G term is removed, what is
/// left between two arcs of one object is lightcurve amplitude plus photometric
/// noise — a few tenths — so the wide thresholds that used to be needed to
/// absorb the geometry systematic no longer discriminate anything.
///
/// The disabled setting is included deliberately, and now actually works: the
/// previous `0.0` sentinel was read as `|ΔH| <= 0` and made that row the
/// strictest test in the sweep rather than the loosest.
const DELTA_H_THRESHOLDS: [f64; 4] = [f64::INFINITY, 0.6, 0.3, 0.15];
/// Reduced chi-square gates swept in one pass. `INFINITY` = off.
///
/// Swept on both the first predicted point and the whole absorbed walk, since
/// the two separate the populations differently: on a full run the walk
/// statistic put correct pairs at p10 = 0.001 against wrong pairs at p10 = 1.57.
const REDUCED_CHI2_THRESHOLDS: [f64; 4] = [f64::INFINITY, 2.0, 5.0, 10.0];
/// Absolute residual caps swept in one pass (arcsec). `INFINITY` = off.
const RESIDUAL_THRESHOLDS: [f64; 4] = [f64::INFINITY, 60.0, 300.0, 900.0];
/// Proper-motion agreement caps swept in one pass (arcmin/day). `INFINITY` = off.
///
/// Nearly independent of the position residual: two unrelated arcs can share a
/// sky cell by coincidence, but sharing a cell *and* a proper motion is a much
/// stronger statement.
///
/// These sit **below** [`INDEX_MAX_RATE_DIFF_RAD_PER_DAY`] on purpose. The index
/// already refuses anything looser, so sweeping values above its floor would
/// measure nothing — every such row would be identical to `rate:off`.
const RATE_THRESHOLDS_ARCMIN_PER_DAY: [f64; 4] = [f64::INFINITY, 3.0, 1.0, 0.3];

/// Minimum overall precision the retained operating point must reach, percent.
///
/// The exotic-population constraint cannot carry this on its own: NEO, Centaur,
/// KBO and SDO together account for 255 of 169 862 trackable objects (0.15 %),
/// so "zero exotic contamination" is satisfied by chance at almost any
/// threshold — a selection rule resting on it alone degenerates to the most
/// permissive row in the sweep, which is exactly what happened.
const MIN_PRECISION_PCT: f64 = 99.0;

/// Gap bands the retained operating point is broken down over, days.
///
/// Widening [`MAX_GAP_DAYS`] lengthens every propagation, which loosens the
/// prediction and may cost precision. That is a risk to measure, not to assume:
/// if the widest band degrades, the window comes back down on evidence.
const GAP_BANDS_DAYS: [f64; 2] = [10.0, 30.0];

/// `|dt|` bands the round-trip error is reported over, days.
///
/// The shape across these bands is the whole point: an error that grows
/// smoothly with the propagation length is physics, while one already present
/// at a few days is a defect.
const DT_BANDS_DAYS: [f64; 4] = [5.0, 20.0, 50.0, 100.0];

/// Range considered physically sane for a solar-system object, AU. Outside it
/// the attributable↔cartesian Jacobian is ill-conditioned and the state is
/// junk regardless of what the solver returned.
const SANE_RHO_MAX_AU: f64 = 200.0;

/// What a fragment is, for linkage purposes.
struct Fragment<'a> {
    /// Whether this arc carries enough observations to be linkable at all.
    /// Non-converged fragments stay in the list — they are needed to build
    /// the honest ground-truth denominator and to measure what the gate
    /// costs — but they are never indexed nor paired.
    converged: bool,
    track_ids: &'a [ObsId],
    /// MAP state at the arc's own epoch — what the dynamical test propagates
    /// from.
    state: KFState<'a>,
    h: Option<f64>,
    /// `(first, last)` observation epoch, MJD TT.
    span: (f64, f64),
    /// Ground-truth identity, `None` when the fragment is `Mixed`/`Unknown`
    /// and therefore not scorable.
    traj: Option<TrajId>,
}

/// Continuous quantities for one ordered pair, computed once so every
/// threshold combination can be derived without re-propagating.
struct PairMeasurement {
    /// Ground-truth verdict, `None` when either fragment is unscorable.
    same_object: Option<bool>,
    traj: Option<TrajId>,
    temporal_ok: bool,
    /// Days between the two arcs' spans, for the gap breakdown of the retained
    /// operating point. `NaN` propagates as "unclassifiable" and is skipped.
    gap_days: Option<f64>,
    delta_h: Option<f64>,
    /// Absolute prediction-to-observation miss (arcsec), one per sampled
    /// point. The raw `d²` values are deliberately not kept: only the reduced
    /// chi-square below is used to judge, and holding a `Vec` per pair across
    /// a million-pair sweep is wasted memory.
    separation_arcsec: Vec<f64>,
    /// `d²₁ / 2` — the prediction test taken before the filter adapts to
    /// anything. `None` for an unjudgeable pair.
    reduced_chi2_first: Option<f64>,
    /// Largest of `|Δα̇·cos δ|` and `|Δδ̇|` between the predictor propagated to
    /// the tested arc's epoch and that arc's own attributable, arcmin/day.
    /// `None` when the propagation fails.
    rate_diff_arcmin_per_day: Option<f64>,
    /// `Σd² / (2n)` over the whole walk.
    reduced_chi2_all: Option<f64>,
}

/// Counters for one threshold combination.
#[derive(Default, Clone, Copy)]
struct StageCounters {
    linked: u64,
    correct: u64,
    wrong: u64,
    unscorable: u64,
}

impl StageCounters {
    fn record(&mut self, linked: bool, same_object: Option<bool>) {
        if !linked {
            return;
        }
        self.linked += 1;
        match same_object {
            Some(true) => self.correct += 1,
            Some(false) => self.wrong += 1,
            None => self.unscorable += 1,
        }
    }

    fn precision_pct(&self) -> f64 {
        let scorable = self.correct + self.wrong;
        if scorable == 0 {
            f64::NAN
        } else {
            100.0 * self.correct as f64 / scorable as f64
        }
    }
}

/// The instant every arc is propagated to before being indexed, with the
/// observer geometry there.
///
/// One shared epoch *and* one shared observer state: the attributable is
/// topocentric, so two arcs are only comparable if both are expressed from the
/// same vantage point at the same time.
#[derive(Debug, Clone, Copy)]
struct ReferenceEpoch {
    epoch: f64,
    r_obs: nalgebra::Vector3<f64>,
    v_obs: nalgebra::Vector3<f64>,
}

/// Pick the reference epoch near the middle of the run's observations.
///
/// The middle bounds how far any arc has to be propagated, and propagation
/// error is what the index bins must absorb. The geometry is taken from the
/// real observation closest to that instant, so the observer state is a
/// genuine one rather than an extrapolation.
fn pick_reference_epoch(
    fragments_track_ids: &[&[ObsId]],
    obs_dataset: &ObsDataset,
    kalman_context: &KalmanContext,
) -> Option<ReferenceEpoch> {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for ids in fragments_track_ids {
        for id in ids.iter() {
            if let Some(obs) = obs_dataset.get_observation(*id) {
                let t = obs.mjd_tt();
                if t.is_finite() {
                    lo = lo.min(t);
                    hi = hi.max(t);
                }
            }
        }
    }
    if !(lo.is_finite() && hi.is_finite()) {
        return None;
    }
    let target = 0.5 * (lo + hi);

    // Closest real observation to the midpoint, so the observer state below is
    // one the survey actually had.
    let mut best: Option<(f64, ObsId)> = None;
    for ids in fragments_track_ids {
        for id in ids.iter() {
            if let Some(obs) = obs_dataset.get_observation(*id) {
                let d = (obs.mjd_tt() - target).abs();
                if d.is_finite() && best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, *id));
                }
            }
        }
    }

    let (_, id) = best?;
    let obs = obs_dataset.get_observation(id)?;
    let (r_obs, v_obs) = crate::kalman_traj::resolve_geometry(obs_dataset, kalman_context, obs)?;
    Some(ReferenceEpoch {
        epoch: obs.mjd_tt(),
        r_obs,
        v_obs,
    })
}

/// Build the fragment list, in `all_reconstructions` order.
///
/// Fragments without a usable MAP state are skipped and counted: a lineage
/// with no live hypothesis, or an archived arc whose observations cannot be
/// resolved, has nothing to propagate and so cannot be linked either way.
fn build_fragments<'a>(
    collection: &'a BranchCollection<'_, '_>,
    obs_dataset: &ObsDataset,
    kalman_context: &'a KalmanContext,
    ground_truth: &ObsTrajLookup,
) -> (Vec<Fragment<'a>>, usize) {
    let pool = all_reconstructions(collection);
    let n_live = collection.branches.len();

    let mut fragments = Vec::with_capacity(pool.len());
    let mut n_skipped = 0usize;

    for (index, track_ids) in pool.iter().enumerate() {
        let Some(span) = observation_span(track_ids, obs_dataset) else {
            n_skipped += 1;
            continue;
        };

        let state = if index < n_live {
            match collection.branches[index].bank.best() {
                Some(best) => best.kf.clone(),
                None => {
                    n_skipped += 1;
                    continue;
                }
            }
        } else {
            collection.archived[index - n_live]
                .map_state
                .clone()
                .into_kf_state(kalman_context)
        };

        let h = if index < n_live {
            collection.branches[index]
                .bank
                .absolute_magnitude_estimate()
        } else {
            collection.archived[index - n_live].absolute_magnitude_estimate
        };

        let traj = match ground_truth.classify(track_ids) {
            SeedPurity::Pure(traj_id) => Some(traj_id),
            SeedPurity::Mixed | SeedPurity::Unknown => None,
        };

        fragments.push(Fragment {
            converged: arc_is_converged(track_ids.len(), MIN_ARC_POINTS),
            track_ids,
            state,
            h,
            span,
            traj,
        });
    }

    (fragments, n_skipped)
}

/// `(first, last)` observation epoch of an arc, or `None` if no id resolves.
fn observation_span(track_ids: &[ObsId], obs_dataset: &ObsDataset) -> Option<(f64, f64)> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for id in track_ids {
        if let Some(obs) = obs_dataset.get_observation(*id) {
            let t = obs.mjd_tt();
            if t.is_finite() {
                lo = lo.min(t);
                hi = hi.max(t);
            }
        }
    }
    (lo.is_finite() && hi.is_finite()).then_some((lo, hi))
}

/// Up to [`MAX_TEST_POINTS`] observations of `fragment`, spread across its
/// arc, each with the observer geometry needed to propagate against it.
fn sample_test_points<'obs>(
    fragment: &Fragment<'_>,
    obs_dataset: &'obs ObsDataset,
    kalman_context: &KalmanContext,
) -> Vec<LinkTestPoint<'obs>> {
    sample_test_points_for(
        fragment.track_ids,
        MAX_TEST_POINTS,
        obs_dataset,
        kalman_context,
    )
}

/// Spread sample of an arc's observations, with the observer geometry at each
/// epoch resolved.
///
/// Shared with the applied-merge path so both judge a pair on the same
/// evidence; a validation that sampled differently from the sweep would not be
/// validating the sweep's conclusion.
pub fn sample_test_points_for<'obs>(
    track_ids: &[ObsId],
    max_points: usize,
    obs_dataset: &'obs ObsDataset,
    kalman_context: &KalmanContext,
) -> Vec<LinkTestPoint<'obs>> {
    let n = track_ids.len();
    if n == 0 || max_points == 0 {
        return Vec::new();
    }
    let stride = n.div_ceil(max_points).max(1);

    let mut points: Vec<LinkTestPoint<'obs>> = track_ids
        .iter()
        .step_by(stride)
        .take(max_points)
        .filter_map(|id| {
            let observation = obs_dataset.get_observation(*id)?;
            let (r_obs, v_obs) =
                crate::kalman_traj::resolve_geometry(obs_dataset, kalman_context, observation)?;
            Some(LinkTestPoint {
                observation,
                r_obs,
                v_obs,
            })
        })
        .collect();

    // `sequential_fit` absorbs points as it walks, so they must be in
    // chronological order for the covariance to tighten meaningfully.
    points.sort_by(|a, b| a.observation.mjd_tt().total_cmp(&b.observation.mjd_tt()));
    points
}

/// Canonical ordering of a candidate pair: `(predictor, tested)`.
///
/// The predictor is the **longer** arc, not the earlier one. The test is
/// directional — one arc predicts the other's observations — but which arc
/// plays that role is a free choice, and measurements showed 48 % of true
/// pairs have the *short* arc first. Letting the converged arc predict, in
/// whichever time direction, is what makes those testable.
///
/// `None` when the spans overlap: one object holds one position per epoch, so
/// overlapping arcs are duplicates rather than continuations.
///
/// Used by both the candidate enumeration and the ground-truth enumeration so
/// the two index the same pairs and can be intersected.
fn order_pair(fragments: &[Fragment<'_>], a: usize, b: usize) -> Option<(usize, usize)> {
    let (a_span, b_span) = (fragments[a].span, fragments[b].span);
    if !(a_span.1 <= b_span.0 || b_span.1 <= a_span.0) {
        return None;
    }
    let (na, nb) = (fragments[a].track_ids.len(), fragments[b].track_ids.len());
    Some(if (nb, b) > (na, a) { (b, a) } else { (a, b) })
}

/// One night's worth of indexed candidate arcs.
///
/// The observer state is harvested from whichever arc created the slice: the
/// predictor has to be propagated *somewhere* inside this night, and every arc
/// in it sits within a day of every other, which is far below the cell size the
/// probe already tolerates.
struct TimeSlice {
    epoch: f64,
    r_obs: nalgebra::Vector3<f64>,
    v_obs: nalgebra::Vector3<f64>,
    /// Sky cell → fragment slots indexed in this night.
    cells: AHashMap<u64, Vec<usize>>,
}

/// Enumerate the ordered candidate pairs a **time-sliced** index proposes.
///
/// # Why the index is asymmetric
///
/// The previous design carried every arc to one shared reference epoch and
/// bucketed the result. That forced a long propagation on both sides, and a
/// propagation is only as good as `ρ` — which is exactly the badly-determined
/// half of a topocentric attributable. So it had to demand convergence of both
/// arcs, and measurement showed the price: of 12 613 true pairs the index
/// missed, **12 090 were missed because one arc was too short to index**,
/// against 523 genuinely mis-binned. A 17× ratio.
///
/// The two sides do not need the same treatment:
///
/// * **Candidates** are indexed at *their own epoch*, with no propagation at
///   all. `(α, δ, α̇, δ̇)` is what the observations directly measure — a
///   two-point tracklet already pins it down — so arc length stops mattering
///   and short arcs enter the index.
/// * **Predictors** must be converged, and are propagated to each night slice
///   within [`MAX_GAP_DAYS`] of their span. Round-trip instrumentation cleared
///   the propagator itself (error ~0 at p99 in every band, both directions),
///   but its *one-way* error still grows with distance through `ρ` ignorance —
///   hence short hops instead of one ~100-day jump to a shared epoch.
///
/// Only bins strictly outside the predictor's own span are visited: an arc
/// overlapping it is a duplicate, not a continuation, and `order_pair` would
/// reject it anyway.
///
/// # Why proper motion is not in the key
///
/// Quantising it would add a ±1 probe on two more axes — 81 lookups per slice
/// instead of 9 — while still cutting arbitrarily at bin edges. Comparing rates
/// exactly on the short list a cell lookup returns is cheaper and sharper; see
/// [`rates_compatible`].
fn candidate_pairs(fragments: &[Fragment<'_>]) -> (Vec<(usize, usize)>, IndexStats) {
    let binner = HealpixBinner::new(INDEX_HEALPIX_DEPTH);

    // Every arc's attributable at its own epoch. No propagation happens here —
    // this is precisely what makes a short arc indexable.
    let attributables: Vec<Option<(f64, f64, f64, f64)>> = fragments
        .iter()
        .map(|f| attributable_at(&f.state))
        .collect();

    let t0 = fragments
        .iter()
        .map(|f| f.state.epoch)
        .filter(|e| e.is_finite())
        .fold(f64::INFINITY, f64::min);
    if !t0.is_finite() {
        return (Vec::new(), IndexStats::default());
    }
    let time_binner = UniformTimeBinner::new(t0, TIME_SLICE_DAYS);

    let mut slices: AHashMap<TimeBin, TimeSlice> = AHashMap::default();
    let mut n_indexed = 0usize;
    for (slot, fragment) in fragments.iter().enumerate() {
        if attributables[slot].is_none() || !fragment.state.epoch.is_finite() {
            continue;
        }
        let Some(SpatialKey(cell)) = sky_cell_key(&fragment.state, &binner) else {
            continue;
        };
        let epoch = fragment.state.epoch;
        let slice = slices
            .entry(time_binner.bin_for(epoch))
            .or_insert_with(|| TimeSlice {
                epoch,
                r_obs: fragment.state.r_obs,
                v_obs: fragment.state.v_obs,
                cells: AHashMap::default(),
            });
        slice.cells.entry(cell).or_default().push(slot);
        n_indexed += 1;
    }

    let mut occupancy: Vec<usize> = slices
        .values()
        .flat_map(|s| s.cells.values().map(Vec::len))
        .collect();
    occupancy.sort_unstable();
    let stats = IndexStats {
        n_predictors: fragments.iter().filter(|f| f.converged).count(),
        n_indexed,
        n_slices: slices.len(),
        n_buckets: occupancy.len(),
        occupancy_median: percentile(&occupancy, 0.50),
        occupancy_p90: percentile(&occupancy, 0.90),
        occupancy_max: occupancy.last().copied().unwrap_or(0),
    };

    // An arc near a cell edge lands either side depending on which arc you ask,
    // so the exact cell alone would lose precisely the pairs of interest.
    let cell_probe_radius = binner.cell_radius() * 1.5;

    let mut pairs: Vec<(usize, usize)> = fragments
        .par_iter()
        .enumerate()
        .filter(|(_, f)| f.converged)
        .flat_map_iter(|(a, predictor)| {
            let mut local: Vec<(usize, usize)> = Vec::new();
            let (span_start, span_end) = predictor.span;

            for (lo, hi) in [
                (span_start - MAX_GAP_DAYS, span_start),
                (span_end, span_end + MAX_GAP_DAYS),
            ] {
                for bin in time_binner.bins_in_range(lo, hi) {
                    let Some(slice) = slices.get(&bin) else {
                        continue;
                    };
                    let Ok(propagated) =
                        predictor
                            .state
                            .predict(slice.epoch, slice.r_obs, slice.v_obs)
                    else {
                        continue;
                    };
                    let Some(attr_pred) = attributable_at(&propagated) else {
                        continue;
                    };
                    let Some(cell) = sky_cell_key(&propagated, &binner) else {
                        continue;
                    };

                    for SpatialKey(neighbour) in binner.neighbors(cell, cell_probe_radius) {
                        let Some(slots) = slice.cells.get(&neighbour) else {
                            continue;
                        };
                        for &b in slots {
                            if a == b {
                                continue;
                            }
                            let Some(attr_b) = attributables[b] else {
                                continue;
                            };
                            if !rates_compatible(attr_pred, attr_b, INDEX_MAX_RATE_DIFF_RAD_PER_DAY)
                            {
                                continue;
                            }
                            let Some(pair) = order_pair(fragments, a, b) else {
                                continue; // overlapping: never mergeable
                            };
                            local.push(pair);
                        }
                    }
                }
            }
            local
        })
        .collect();

    // The same pair reaches the list once per slice that proposed it.
    pairs.sort_unstable();
    pairs.dedup();

    (pairs, stats)
}

/// How the index behaved — reported because a low recall downstream can come
/// from the index never proposing a pair rather than from the gates rejecting
/// it.
#[derive(Default)]
struct IndexStats {
    /// Converged arcs, the only ones allowed to predict.
    n_predictors: usize,
    /// Arcs placed in the candidate index — **all lengths**, which is the
    /// point of the time-sliced design.
    n_indexed: usize,
    n_slices: usize,
    n_buckets: usize,
    occupancy_median: usize,
    occupancy_p90: usize,
    occupancy_max: usize,
}

/// `q`-quantile of an already-sorted slice.
fn percentile(sorted: &[usize], q: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

/// Ground truth about how objects are actually fragmented.
struct TruthStats {
    /// Ordered pairs of pure fragments of the same object with disjoint
    /// spans — what linkage could, at best, recover.
    pairs: AHashSet<(usize, usize)>,
    /// Of those, how many have a converged **predictor** (the longer arc) and
    /// are therefore testable at all.
    pairs_testable: usize,
    /// True pairs where even the longer arc falls short of the threshold —
    /// genuinely beyond reach, not something a different gate policy would
    /// recover.
    lost_both_short: usize,
    /// Testable pairs separated by more than [`MAX_GAP_DAYS`], which the index
    /// therefore cannot propose. Reported so the cap never silently caps
    /// recall.
    lost_beyond_max_gap: usize,
    /// Trajectories whose pure fragments are all temporally disjoint — the
    /// linkable kind of fragmentation.
    traj_split: usize,
    /// Trajectories with at least one overlapping pair of pure fragments.
    /// These are duplicates, not continuations: linkage rejects them by
    /// design and they need de-duplication instead.
    traj_duplicated: usize,
}

/// `x > limit`, written so a NaN `x` counts as exceeding.
///
/// The plain `>` would silently *accept* a NaN, which for every use below —
/// gaps, magnitudes, residuals — means letting an unjudgeable pair through as
/// if it had passed. The negated form is the NaN-safe one; clippy's suggestion
/// to use `partial_cmp` would only restate it less legibly.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
#[inline]
fn exceeds(x: f64, limit: f64) -> bool {
    !(x <= limit)
}

/// Shortest separation between two non-overlapping spans, days. `f64::NAN` for
/// non-finite input, which every comparison below reads as "too far".
fn gap_days(a: (f64, f64), b: (f64, f64)) -> f64 {
    if a.1 <= b.0 { b.0 - a.1 } else { a.0 - b.1 }
}

/// Ground-truth positives, plus what the convergence gate costs and how much
/// of the observed fragmentation is actually linkable.
fn ground_truth_stats(fragments: &[Fragment<'_>]) -> TruthStats {
    let mut by_traj: AHashMap<&TrajId, Vec<usize>> = AHashMap::default();
    for (slot, fragment) in fragments.iter().enumerate() {
        if let Some(traj) = fragment.traj.as_ref() {
            by_traj.entry(traj).or_default().push(slot);
        }
    }

    let mut stats = TruthStats {
        pairs: AHashSet::default(),
        pairs_testable: 0,
        lost_both_short: 0,
        lost_beyond_max_gap: 0,
        traj_split: 0,
        traj_duplicated: 0,
    };

    for slots in by_traj.values() {
        if slots.len() < 2 {
            continue;
        }
        let mut any_overlap = false;

        for (i, &a) in slots.iter().enumerate() {
            for &b in &slots[i + 1..] {
                let Some((predictor, tested)) = order_pair(fragments, a, b) else {
                    any_overlap = true;
                    continue;
                };
                stats.pairs.insert((predictor, tested));
                if fragments[predictor].converged {
                    stats.pairs_testable += 1;
                    let gap = gap_days(fragments[predictor].span, fragments[tested].span);
                    if exceeds(gap, MAX_GAP_DAYS) {
                        stats.lost_beyond_max_gap += 1;
                    }
                } else {
                    // Predictor is the longer arc, so this means *both* are
                    // short — genuinely beyond reach, not a gate artefact.
                    stats.lost_both_short += 1;
                }
            }
        }

        if any_overlap {
            stats.traj_duplicated += 1;
        } else {
            stats.traj_split += 1;
        }
    }

    stats
}

/// Why each true pair the index failed to propose was missed, and — for the
/// ones that had every reason to be found — how far off they actually were.
///
/// The breakdown matters more than the distances. Splitting the misses by cause
/// is what showed the previous, reference-epoch index was losing 12 090 pairs to
/// an indexing rule and only 523 to cell sizing; without that split the two are
/// indistinguishable and the effort goes to the wrong place.
///
/// Distances are measured on the **short hop the index actually performs** —
/// predictor propagated to the tested arc's own epoch — not against a distant
/// shared epoch. That is what makes them usable to size the cell: they describe
/// the propagation the index really does.
fn print_missed_pairs(
    fragments: &[Fragment<'_>],
    truth_pairs: &AHashSet<(usize, usize)>,
    proposed: &AHashSet<(usize, usize)>,
) {
    const RAD_TO_ARCMIN: f64 = 3_437.746_770_784_939;

    let (mut d_sky, mut d_ra_rate, mut d_dec_rate) = (Vec::new(), Vec::new(), Vec::new());
    let (mut both_short, mut beyond_gap, mut propagation_failed) = (0usize, 0usize, 0usize);

    for &(predictor_slot, tested_slot) in truth_pairs.iter() {
        if proposed.contains(&(predictor_slot, tested_slot)) {
            continue;
        }
        let (predictor, tested) = (&fragments[predictor_slot], &fragments[tested_slot]);

        // Out of reach by construction: the longer arc is itself too short to
        // predict anything. No index change recovers these.
        if !predictor.converged {
            both_short += 1;
            continue;
        }
        // Outside the search window — a `MAX_GAP_DAYS` cost, not a binning one.
        if exceeds(gap_days(predictor.span, tested.span), MAX_GAP_DAYS) {
            beyond_gap += 1;
            continue;
        }

        let Ok(propagated) =
            predictor
                .state
                .predict(tested.state.epoch, tested.state.r_obs, tested.state.v_obs)
        else {
            propagation_failed += 1;
            continue;
        };
        let (Some(pred_attr), Some(tested_attr)) =
            (attributable_at(&propagated), attributable_at(&tested.state))
        else {
            propagation_failed += 1;
            continue;
        };

        d_sky.push(
            angular_separation_rad(pred_attr.0, pred_attr.1, tested_attr.0, tested_attr.1)
                .to_degrees(),
        );
        d_ra_rate.push((pred_attr.2 - tested_attr.2).abs() * RAD_TO_ARCMIN);
        d_dec_rate.push((tested_attr.3 - pred_attr.3).abs() * RAD_TO_ARCMIN);
    }

    let missed = truth_pairs.len() - proposed.intersection(truth_pairs).count();
    println!("\n  Why the index missed {missed} true pairs");
    let show = |label: &str, n: usize| {
        println!(
            "  {label:<40} {n:>9}  ({:>5.1}%)",
            100.0 * n as f64 / missed.max(1) as f64
        )
    };
    show("both arcs too short to predict", both_short);
    show("gap beyond MAX_GAP_DAYS", beyond_gap);
    show("propagation failed", propagation_failed);
    show("reachable but mis-binned", d_sky.len());

    if d_sky.is_empty() {
        return;
    }
    println!("\n  How far off the mis-binned ones were, at the tested arc's own epoch");
    println!(
        "  {:<26} {:>10} {:>10} {:>10} {:>10}   (cell/tol)",
        "", "p10", "p50", "p90", "p99"
    );
    let cell_deg = HealpixBinner::new(INDEX_HEALPIX_DEPTH)
        .cell_radius()
        .to_degrees();
    print_distance_row("sky separation (deg)", &mut d_sky, cell_deg);
    print_distance_row(
        "|d ra_dot| (arcmin/day)",
        &mut d_ra_rate,
        INDEX_MAX_RATE_DIFF_RAD_PER_DAY * RAD_TO_ARCMIN,
    );
    print_distance_row(
        "|d dec_dot| (arcmin/day)",
        &mut d_dec_rate,
        INDEX_MAX_RATE_DIFF_RAD_PER_DAY * RAD_TO_ARCMIN,
    );
}

/// One percentile row, with the current bin width alongside for comparison.
fn print_distance_row(label: &str, xs: &mut [f64], bin: f64) {
    xs.sort_by(|a, b| a.total_cmp(b));
    let q = |f: f64| xs[(((xs.len() - 1) as f64) * f).round() as usize];
    println!(
        "  {label:<26} {:>10.4} {:>10.4} {:>10.4} {:>10.4}   {bin:.4}",
        q(0.10),
        q(0.50),
        q(0.90),
        q(0.99)
    );
}

/// Great-circle separation between two equatorial directions, radians.
///
/// Haversine rather than a tangent-plane approximation: the separations under
/// investigation reach 180°, where a tangent plane is meaningless.
fn angular_separation_rad(ra_a: f64, dec_a: f64, ra_b: f64, dec_b: f64) -> f64 {
    let d_dec = dec_b - dec_a;
    let d_ra = ra_b - ra_a;
    let h = (d_dec * 0.5).sin().powi(2) + dec_a.cos() * dec_b.cos() * (d_ra * 0.5).sin().powi(2);
    2.0 * h.sqrt().clamp(0.0, 1.0).asin()
}

/// How one fragment's propagation to the reference epoch behaved.
struct PropagationProbe {
    /// Signed propagation length, days. Its sign is what separates the
    /// forward from the backward regime.
    dt_days: f64,
    /// Sky error after propagating out to the reference epoch and straight
    /// back, arcsec. `None` when either leg failed.
    round_trip_arcsec: Option<f64>,
    rho_before: f64,
    rho_after: Option<f64>,
    /// Whether the propagated two-body orbit is closed. `None` when the
    /// outbound leg failed.
    bound: Option<bool>,
    outcome: ProbeOutcome,
}

/// Why a probe stopped where it did — the counters `.ok()` used to discard.
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
enum ProbeOutcome {
    Ok,
    OutboundKepler,
    OutboundSingularJacobian,
    OutboundIllConditionedRange,
    ReturnLegFailed,
}

impl ProbeOutcome {
    fn label(self) -> &'static str {
        match self {
            ProbeOutcome::Ok => "ok",
            ProbeOutcome::OutboundKepler => "outbound: Kepler solver",
            ProbeOutcome::OutboundSingularJacobian => "outbound: singular Jacobian",
            ProbeOutcome::OutboundIllConditionedRange => "outbound: rho out of bounds",
            ProbeOutcome::ReturnLegFailed => "return leg failed",
        }
    }

    fn of(err: &PropagateError) -> Self {
        match err {
            PropagateError::Kepler(_) => ProbeOutcome::OutboundKepler,
            PropagateError::SingularJacobian => ProbeOutcome::OutboundSingularJacobian,
            PropagateError::IllConditionedRange { .. } => ProbeOutcome::OutboundIllConditionedRange,
        }
    }
}

/// Probe the propagation the index depends on, one fragment at a time.
///
/// The round trip is the decisive measurement. Propagating a state out to the
/// reference epoch and straight back must return the starting angles **however
/// wrong the arc's range is**: the same (possibly bad) range is carried out and
/// back, so its effect cancels exactly. Whatever survives is propagator error
/// and nothing else.
///
/// That is what separates the two readings of the missed-pair diagnostic, which
/// showed two arcs of the *same* object landing up to 65° apart at the
/// reference epoch:
/// - round trip clean → the propagator is sound and that spread is genuine
///   range ignorance amplified over ~100-day arcs, so the fix is shorter hops;
/// - round trip dirty → a defect, and its dependence on `sign(dt)` and `|dt|`
///   says where.
fn probe_propagation(
    fragments: &[Fragment<'_>],
    reference: ReferenceEpoch,
) -> Vec<PropagationProbe> {
    fragments
        .par_iter()
        .filter(|f| f.converged)
        .map(|fragment| {
            let state = &fragment.state;
            let dt_days = reference.epoch - state.epoch;
            let rho_before = state.state[4];

            let out = match state.predict(reference.epoch, reference.r_obs, reference.v_obs) {
                Ok(out) => out,
                Err(err) => {
                    return PropagationProbe {
                        dt_days,
                        round_trip_arcsec: None,
                        rho_before,
                        rho_after: None,
                        bound: None,
                        outcome: ProbeOutcome::of(&err),
                    };
                }
            };

            let rho_after = Some(out.state[4]);
            let bound = Some(orbit_elements(&out).is_some());

            // Return leg, warm-started from the outbound anomaly exactly as
            // `propagate_to_epoch` does — so a guess left pointing the wrong
            // way for this direction is exercised, not bypassed.
            let Ok(back) = out.predict(state.epoch, state.r_obs, state.v_obs) else {
                return PropagationProbe {
                    dt_days,
                    round_trip_arcsec: None,
                    rho_before,
                    rho_after,
                    bound,
                    outcome: ProbeOutcome::ReturnLegFailed,
                };
            };

            let err_rad = angular_separation_rad(
                state.state[0],
                state.state[1],
                back.state[0],
                back.state[1],
            );

            PropagationProbe {
                dt_days,
                round_trip_arcsec: Some(err_rad.to_degrees() * 3600.0),
                rho_before,
                rho_after,
                bound,
                outcome: ProbeOutcome::Ok,
            }
        })
        .collect()
}

/// Print the propagation health report.
fn print_propagation_diagnostics(probes: &[PropagationProbe]) {
    println!(
        "\n  Propagation health of the {} indexed arcs",
        probes.len()
    );
    if probes.is_empty() {
        return;
    }

    // ── Round-trip error, by |dt| band and direction ──────────────────────
    // Read the shape, not the absolute value: growth with |dt| is physics
    // leaking through numerics, a floor already present at small |dt| is a
    // defect, and a forward/backward gap at equal |dt| points straight at the
    // solver's warm start.
    println!(
        "\n  Round-trip sky error (arcsec) — must be ~0 whatever rho is, so any signal here is a bug"
    );
    println!(
        "  {:<18} {:>9} {:>12} {:>12} {:>12} {:>12}",
        "|dt| band (days)", "n", "p50", "p90", "p99", "max"
    );

    let mut bands: Vec<(f64, f64)> = Vec::with_capacity(DT_BANDS_DAYS.len() + 1);
    let mut lo = 0.0;
    for &hi in &DT_BANDS_DAYS {
        bands.push((lo, hi));
        lo = hi;
    }
    bands.push((lo, f64::INFINITY));

    for (lo, hi) in bands {
        for (dir_label, forward) in [("forward", true), ("backward", false)] {
            let mut xs: Vec<f64> = probes
                .iter()
                .filter(|p| {
                    let a = p.dt_days.abs();
                    a >= lo && a < hi && (p.dt_days >= 0.0) == forward
                })
                .filter_map(|p| p.round_trip_arcsec)
                .collect();
            if xs.is_empty() {
                continue;
            }
            xs.sort_by(|a, b| a.total_cmp(b));
            let q = |f: f64| xs[(((xs.len() - 1) as f64) * f).round() as usize];
            let band = if hi.is_finite() {
                format!("{lo:.0}-{hi:.0} {dir_label}")
            } else {
                format!("{lo:.0}+ {dir_label}")
            };
            println!(
                "  {:<18} {:>9} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
                band,
                xs.len(),
                q(0.50),
                q(0.90),
                q(0.99),
                xs[xs.len() - 1],
            );
        }
    }

    // A `max` alone cannot say whether the tail is three arcs or three hundred,
    // and that is the difference between a curiosity and a defect worth fixing.
    // The 1 degree row is the one that matters: nothing physical can send a
    // round trip that far.
    println!("\n  Round-trip tail, counted rather than glimpsed");
    for (label, limit_arcsec) in [("> 1\"", 1.0), ("> 60\"", 60.0), ("> 1 deg", 3600.0)] {
        let n = probes
            .iter()
            .filter(|p| p.round_trip_arcsec.is_some_and(|e| e > limit_arcsec))
            .count();
        println!(
            "  {label:<34} {n:>9}  ({:>5.3}%)",
            100.0 * n as f64 / probes.len() as f64
        );
    }

    // ── Everything `.ok()` used to swallow ────────────────────────────────
    let mut by_outcome: AHashMap<&'static str, usize> = AHashMap::default();
    for p in probes {
        *by_outcome.entry(p.outcome.label()).or_default() += 1;
    }
    let mut outcomes: Vec<_> = by_outcome.into_iter().collect();
    outcomes.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
    println!("\n  Propagation outcomes (previously discarded by `.ok()`)");
    for (label, n) in outcomes {
        println!(
            "  {label:<34} {n:>9}  ({:>5.2}%)",
            100.0 * n as f64 / probes.len() as f64
        );
    }

    // ── State sanity, before and after ────────────────────────────────────
    let count = |f: &dyn Fn(&PropagationProbe) -> bool| probes.iter().filter(|p| f(p)).count();
    let pct = |n: usize| 100.0 * n as f64 / probes.len() as f64;

    let before_bad = count(&|p| !p.rho_before.is_finite() || p.rho_before <= 0.0);
    let before_absurd = count(&|p| p.rho_before > SANE_RHO_MAX_AU);
    let after_bad = count(&|p| p.rho_after.is_some_and(|r| !r.is_finite() || r <= 0.0));
    let after_absurd = count(&|p| p.rho_after.is_some_and(|r| r > SANE_RHO_MAX_AU));
    let unbound = count(&|p| p.bound == Some(false));

    println!("\n  State sanity");
    println!(
        "  {:<34} {:>9}  ({:>5.2}%)",
        "rho <= 0 or non-finite, before",
        before_bad,
        pct(before_bad)
    );
    println!(
        "  {:<34} {:>9}  ({:>5.2}%)",
        format!("rho > {SANE_RHO_MAX_AU:.0} AU, before"),
        before_absurd,
        pct(before_absurd)
    );
    println!(
        "  {:<34} {:>9}  ({:>5.2}%)",
        "rho <= 0 or non-finite, after",
        after_bad,
        pct(after_bad)
    );
    println!(
        "  {:<34} {:>9}  ({:>5.2}%)",
        format!("rho > {SANE_RHO_MAX_AU:.0} AU, after"),
        after_absurd,
        pct(after_absurd)
    );
    println!(
        "  {:<34} {:>9}  ({:>5.2}%)   <- the real figure; the index never tested boundedness",
        "unbound orbit after propagation",
        unbound,
        pct(unbound)
    );
}

/// Run the study and print its report.
pub fn print_merge_shadow_study(
    collection: &BranchCollection<'_, '_>,
    obs_dataset: &ObsDataset,
    kalman_context: &KalmanContext,
    ground_truth: &ObsTrajLookup,
    traj_population: &AHashMap<TrajId, Population>,
) {
    let sep = "=".repeat(90);
    println!("\n{sep}");
    println!("[Fragment linkage shadow study] merge criteria scored against ground truth");
    println!("{sep}");

    // One shared instant for the whole pool: the attributable index compares
    // sky positions, which only means anything at a common epoch.
    let pool_track_ids = all_reconstructions(collection);
    let reference = pick_reference_epoch(&pool_track_ids, obs_dataset, kalman_context);
    let (fragments, n_skipped) =
        build_fragments(collection, obs_dataset, kalman_context, ground_truth);
    if fragments.is_empty() {
        println!("  (no linkable fragment)");
        println!("{sep}");
        return;
    }

    let truth = ground_truth_stats(&fragments);
    let (pairs, index) = candidate_pairs(&fragments);

    // Runs before anything is concluded from the index: if the propagation the
    // index rests on is broken, no cell size and no threshold read below means
    // anything.
    if let Some(reference) = reference {
        print_propagation_diagnostics(&probe_propagation(&fragments, reference));
    }

    // How many true pairs the orbital index actually proposes — this caps
    // every recall figure below, so it is reported on its own.
    let proposed: AHashSet<(usize, usize)> = pairs.iter().copied().collect();
    let n_truth_found = truth.pairs.iter().filter(|p| proposed.contains(p)).count();

    let n_live = collection.branches.len();
    println!(
        "  Fragments in pool                     : {} ({} live + {} archived, {} skipped)",
        fragments.len(),
        n_live.min(fragments.len()),
        fragments.len().saturating_sub(n_live),
        n_skipped
    );
    println!(
        "  Predictors (>={MIN_ARC_POINTS} pts, may predict)   : {} ({:.1}% of pool)",
        index.n_predictors,
        100.0 * index.n_predictors as f64 / fragments.len().max(1) as f64
    );
    println!(
        "  Candidates indexed (any length)       : {} ({:.1}% of pool, over {} nightly slices)",
        index.n_indexed,
        100.0 * index.n_indexed as f64 / fragments.len().max(1) as f64,
        index.n_slices
    );
    println!(
        "  Sky-cell buckets                      : {} (occupancy med {} / p90 {} / max {})",
        index.n_buckets, index.occupancy_median, index.occupancy_p90, index.occupancy_max
    );
    println!(
        "  Search window / rate tolerance        : +/-{MAX_GAP_DAYS:.0} d / {:.0} arcmin per day",
        INDEX_MAX_RATE_DIFF_RAD_PER_DAY * 3_437.746_770_784_939
    );
    println!(
        "  Ground-truth mergeable pairs          : {} ({} testable: longer arc converged)",
        truth.pairs.len(),
        truth.pairs_testable
    );
    println!(
        "    out of reach (both arcs short)      : {}",
        truth.lost_both_short
    );
    println!(
        "    testable but gap > {MAX_GAP_DAYS:.0} d           : {}",
        truth.lost_beyond_max_gap
    );
    println!(
        "  Multi-fragment trajectories           : {} split (linkable) / {} duplicated (overlapping — needs de-dup, not linkage)",
        truth.traj_split, truth.traj_duplicated
    );
    println!(
        "  Pairs proposed by the index           : {} (index recall {:.1}% of all true pairs, {:.1}% of testable ones)",
        pairs.len(),
        100.0 * n_truth_found as f64 / truth.pairs.len().max(1) as f64,
        100.0 * n_truth_found as f64 / truth.pairs_testable.max(1) as f64
    );
    println!("  Test points per pair (max)            : {MAX_TEST_POINTS}");

    // The expensive part: per (pair, test point), one Kepler propagation plus
    // one Kalman update.
    let measurements: Vec<PairMeasurement> = pairs
        .par_iter()
        .map(|&(predictor_slot, tested_slot)| {
            let predictor = &fragments[predictor_slot];
            let tested = &fragments[tested_slot];

            let same_object = match (predictor.traj.as_ref(), tested.traj.as_ref()) {
                (Some(a), Some(b)) => Some(a == b),
                _ => None,
            };
            let delta_h = match (predictor.h, tested.h) {
                (Some(a), Some(b)) if a.is_finite() && b.is_finite() => Some((a - b).abs()),
                _ => None,
            };

            // A fit that cannot be computed at all leaves both vectors empty,
            // which every threshold below reads as "not a link" — an
            // unjudgeable pair is never linked.
            // One extra propagation, to the tested arc's own epoch, so proper
            // motion can be compared at a common instant.
            let rate_diff_arcmin_per_day = predictor
                .state
                .predict(tested.state.epoch, tested.state.r_obs, tested.state.v_obs)
                .ok()
                .and_then(|propagated| {
                    let a = attributable_at(&propagated)?;
                    let b = attributable_at(&tested.state)?;
                    Some(((a.2 - b.2).abs()).max((a.3 - b.3).abs()) * 3_437.746_770_784_939)
                });

            let points = sample_test_points(tested, obs_dataset, kalman_context);
            let fit = sequential_fit(&predictor.state, &points);
            let reduced_chi2_first = fit.as_ref().and_then(|f| f.reduced_chi2_first());
            let reduced_chi2_all = fit.as_ref().and_then(|f| f.reduced_chi2());
            let separation_arcsec = fit.map(|f| f.separation_arcsec).unwrap_or_default();

            PairMeasurement {
                same_object,
                traj: predictor.traj.clone(),
                // Order-independent: `order_pair` already guaranteed the spans
                // do not overlap, whichever arc predicts.
                temporal_ok: temporally_disjoint(predictor.span, tested.span, 0.0)
                    || temporally_disjoint(tested.span, predictor.span, 0.0),
                gap_days: Some(gap_days(predictor.span, tested.span)).filter(|g| g.is_finite()),
                delta_h,
                separation_arcsec,
                rate_diff_arcmin_per_day,
                reduced_chi2_first,
                reduced_chi2_all,
            }
        })
        .collect();

    // Denominator stays "all true pairs" rather than the converged subset, so
    // recall remains comparable when the convergence threshold changes.
    print_missed_pairs(&fragments, &truth.pairs, &proposed);
    let rows = sweep(&measurements, traj_population);
    print_sweep(&rows, &measurements, truth.pairs.len());
    print_delta_h_distributions(&measurements);
    print_reduced_chi2_distributions(&measurements);
    println!("{sep}");
}

/// One combination of gates. `f64::INFINITY` disables a test.
///
/// A single explicit sentinel throughout, because the previous `0.0`-means-off
/// convention silently inverted one gate: `photometric_pass(d, 0.0)` read as
/// `d <= 0.0`, so the row labelled `h:off` was in fact the *strictest*
/// photometric test ever run — it kept only pairs with no magnitude estimate,
/// which are also the pairs with no ground truth. Two full reports concluded
/// "zero contamination" from that row.
#[derive(Clone, Copy)]
struct Thresholds {
    max_delta_h: f64,
    max_reduced_chi2: f64,
    /// Judge the reduced chi-square on the whole absorbed walk rather than on
    /// the first predicted point alone.
    chi2_on_whole_walk: bool,
    max_residual_arcsec: f64,
    max_rate_diff_arcmin_per_day: f64,
}

impl Thresholds {
    fn label(&self) -> String {
        let opt = |name: &str, v: f64, unit: &str| {
            if v.is_finite() {
                format!("{name}<={v}{unit}")
            } else {
                format!("{name}:off")
            }
        };
        format!(
            "{} {} {} {}",
            opt("h", self.max_delta_h, ""),
            opt(
                if self.chi2_on_whole_walk {
                    "rc2w"
                } else {
                    "rc2f"
                },
                self.max_reduced_chi2,
                ""
            ),
            opt("res", self.max_residual_arcsec, "\""),
            opt("rate", self.max_rate_diff_arcmin_per_day, ""),
        )
    }

    /// Whether this combination would link the pair.
    fn links(&self, m: &PairMeasurement) -> bool {
        if !m.temporal_ok || m.separation_arcsec.is_empty() {
            return false;
        }
        // A missing magnitude never vetoes on its own, mirroring the engine's
        // `photometric_compatible`.
        if m.delta_h.is_some_and(|d| exceeds(d, self.max_delta_h)) {
            return false;
        }
        let chi2 = if self.chi2_on_whole_walk {
            m.reduced_chi2_all
        } else {
            m.reduced_chi2_first
        };
        if self.max_reduced_chi2.is_finite() && !chi2.is_some_and(|c| c <= self.max_reduced_chi2) {
            return false;
        }
        if self.max_residual_arcsec.is_finite()
            && !m
                .separation_arcsec
                .iter()
                .all(|s| *s <= self.max_residual_arcsec)
        {
            return false;
        }
        if self.max_rate_diff_arcmin_per_day.is_finite()
            && !m
                .rate_diff_arcmin_per_day
                .is_some_and(|r| r <= self.max_rate_diff_arcmin_per_day)
        {
            return false;
        }
        true
    }
}

/// Scored outcome of one threshold combination.
struct SweepRow {
    thresholds: Thresholds,
    counters: StageCounters,
    wrong_by_pop: AHashMap<Population, u64>,
}

impl SweepRow {
    /// Wrong merges on the populations the science case cannot absorb.
    ///
    /// NEO / Centaur / KBO / SDO carry a handful of objects each, so a single
    /// fabricated trajectory is a large relative error — and these are exactly
    /// the objects the pipeline exists to find. MBA and Unknown are dense
    /// enough to dilute one.
    fn exotic_wrong(&self) -> u64 {
        [
            Population::Neo,
            Population::Centaur,
            Population::Kbo,
            Population::Sdo,
        ]
        .iter()
        .filter_map(|p| self.wrong_by_pop.get(p))
        .sum()
    }
}

/// Score every threshold combination against ground truth.
fn sweep(
    measurements: &[PairMeasurement],
    traj_population: &AHashMap<TrajId, Population>,
) -> Vec<SweepRow> {
    let mut combinations = Vec::new();
    for &max_delta_h in &DELTA_H_THRESHOLDS {
        for &max_reduced_chi2 in &REDUCED_CHI2_THRESHOLDS {
            for &chi2_on_whole_walk in &[false, true] {
                for &max_residual_arcsec in &RESIDUAL_THRESHOLDS {
                    for &max_rate_diff_arcmin_per_day in &RATE_THRESHOLDS_ARCMIN_PER_DAY {
                        combinations.push(Thresholds {
                            max_delta_h,
                            max_reduced_chi2,
                            chi2_on_whole_walk,
                            max_residual_arcsec,
                            max_rate_diff_arcmin_per_day,
                        });
                    }
                }
            }
        }
    }

    combinations
        .into_par_iter()
        .map(|thresholds| {
            let mut counters = StageCounters::default();
            let mut wrong_by_pop: AHashMap<Population, u64> = AHashMap::default();
            for m in measurements {
                let linked = thresholds.links(m);
                counters.record(linked, m.same_object);
                if linked && m.same_object == Some(false) {
                    let population = m
                        .traj
                        .as_ref()
                        .and_then(|t| traj_population.get(t))
                        .copied()
                        .unwrap_or(Population::Unknown);
                    *wrong_by_pop.entry(population).or_insert(0) += 1;
                }
            }
            SweepRow {
                thresholds,
                counters,
                wrong_by_pop,
            }
        })
        .collect()
}

/// Whether a combination is admissible as an operating point.
///
/// Two independent constraints, because neither covers the other: the exotic
/// rule protects the objects the science case exists for, and the precision
/// floor protects the aggregate. Resting on the exotic rule alone made the
/// selection degenerate to the most permissive row in the sweep — 61.6 %
/// precision and 3900 fabricated trajectories — since exotic objects are too
/// rare to constrain anything.
fn is_admissible(row: &SweepRow) -> bool {
    row.exotic_wrong() == 0 && row.counters.precision_pct() >= MIN_PRECISION_PCT
}

/// The retained operating point: the admissible combination recovering the most
/// true pairs.
///
/// Chosen by rule rather than by eye so the report cannot drift into quoting a
/// row that happens to look good. Ties break on fewer total wrong merges.
fn best_operating_point(rows: &[SweepRow]) -> Option<&SweepRow> {
    rows.iter()
        .filter(|r| is_admissible(r))
        .max_by_key(|r| (r.counters.correct, std::cmp::Reverse(r.counters.wrong)))
}

/// Precision bands the frontier is reported over, percent.
const PRECISION_BANDS_PCT: [f64; 6] = [99.9, 99.5, 99.0, 98.0, 95.0, 90.0];

/// The sweep's precision/recall frontier, the retained operating point, and
/// that point's contamination broken down by population and by gap length.
///
/// # Why a frontier and not a sorted list
///
/// The previous version sorted by recall and printed the top 40 of 512
/// combinations. Precise combinations have mechanically less recall, so *every*
/// one of them fell past the cut — the table could not show a single row above
/// 65 % precision even though rows above 99 % existed. A frontier reports the
/// best recall reachable at each precision level, so the decision-relevant rows
/// are present by construction whatever the sweep's width.
fn print_sweep(rows: &[SweepRow], measurements: &[PairMeasurement], n_truth_pairs: usize) {
    let tested = measurements.len() as u64;
    let recall_of = |correct: u64| 100.0 * correct as f64 / n_truth_pairs.max(1) as f64;

    let print_row = |label: String, c: &StageCounters| {
        println!(
            "  {:<38} {:>9} {:>8} {:>10} {:>8} {:>11.2} {:>9.2}",
            label,
            tested,
            c.linked,
            c.correct,
            c.wrong,
            c.precision_pct(),
            recall_of(c.correct),
        );
    };

    println!("\n  Precision/recall frontier — best recall reachable at each precision level");
    println!(
        "  {:<38} {:>9} {:>8} {:>10} {:>8} {:>11} {:>9}",
        "Criterion", "tested", "linked", "correct", "WRONG", "precision%", "recall%"
    );

    // Reference row: what the index alone proposes, i.e. the base rate every
    // gate below has to fight against.
    let mut baseline = StageCounters::default();
    for m in measurements {
        baseline.record(m.temporal_ok, m.same_object);
    }
    print_row("index baseline (no gate)".to_string(), &baseline);

    for floor in PRECISION_BANDS_PCT {
        let best = rows
            .iter()
            .filter(|r| r.counters.linked > 0 && r.counters.precision_pct() >= floor)
            .max_by_key(|r| (r.counters.correct, std::cmp::Reverse(r.counters.wrong)));
        match best {
            Some(row) => print_row(
                format!("[>={floor:>5.1}%] {}", row.thresholds.label()),
                &row.counters,
            ),
            None => println!("  [>={floor:>5.1}%] none"),
        }
    }

    // The unconstrained best, to show what precision is being traded away.
    if let Some(row) = rows
        .iter()
        .filter(|r| r.counters.linked > 0)
        .max_by_key(|r| (r.counters.correct, std::cmp::Reverse(r.counters.wrong)))
    {
        print_row(
            format!("[max recall] {}", row.thresholds.label()),
            &row.counters,
        );
    }

    println!(
        "\n  Selection rule: precision >= {MIN_PRECISION_PCT:.1}% AND zero wrong merge on NEO/Centaur/KBO/SDO"
    );
    let Some(best) = best_operating_point(rows) else {
        println!("  No combination satisfies both constraints.");
        // Show what is blocking, rather than falling silently back to nothing.
        if let Some(row) = rows
            .iter()
            .filter(|r| r.counters.linked > 0 && r.exotic_wrong() == 0)
            .max_by(|a, b| {
                a.counters
                    .precision_pct()
                    .total_cmp(&b.counters.precision_pct())
            })
        {
            print_row(
                format!("  closest (exotic-clean): {}", row.thresholds.label()),
                &row.counters,
            );
        }
        return;
    };

    println!("  Retained: {}", best.thresholds.label());
    print_row("  ->".to_string(), &best.counters);

    println!(
        "\n  WRONG merges by population at that point — total {}",
        best.counters.wrong
    );
    if best.counters.wrong == 0 {
        println!("    none — this cascade manufactures no contamination at all");
    } else {
        for population in Population::all() {
            let n = best.wrong_by_pop.get(&population).copied().unwrap_or(0);
            if n > 0 {
                println!("    {:<24} {n}", population.label());
            }
        }
    }

    print_gap_breakdown(best, measurements);
}

/// Correct/wrong merges of the retained point, split by how far apart the two
/// arcs are.
///
/// Widening the search window lengthens every propagation, which loosens the
/// prediction and can cost precision. This is where that shows: if the widest
/// band is markedly dirtier than the others, [`MAX_GAP_DAYS`] is too generous
/// and comes back down on evidence rather than on a hunch.
fn print_gap_breakdown(best: &SweepRow, measurements: &[PairMeasurement]) {
    let mut bands: Vec<(f64, f64)> = Vec::with_capacity(GAP_BANDS_DAYS.len() + 1);
    let mut lo = 0.0;
    for &hi in &GAP_BANDS_DAYS {
        bands.push((lo, hi));
        lo = hi;
    }
    bands.push((lo, MAX_GAP_DAYS));

    println!("\n  Retained point by gap between the two arcs");
    println!(
        "  {:<20} {:>10} {:>10} {:>12}",
        "gap band (days)", "correct", "WRONG", "precision%"
    );
    for (lo, hi) in bands {
        let (mut correct, mut wrong) = (0u64, 0u64);
        for m in measurements {
            let Some(gap) = m.gap_days else { continue };
            if !(gap >= lo && gap < hi && best.thresholds.links(m)) {
                continue;
            }
            match m.same_object {
                Some(true) => correct += 1,
                Some(false) => wrong += 1,
                None => {}
            }
        }
        let judged = correct + wrong;
        if judged == 0 {
            continue;
        }
        println!(
            "  {:<20} {correct:>10} {wrong:>10} {:>12.2}",
            format!("{lo:.0}-{hi:.0}"),
            100.0 * correct as f64 / judged as f64
        );
    }
}

/// Absolute-magnitude agreement between the two arcs, split by ground truth.
///
/// This is the direct verdict on the H-G phase correction, and it reads
/// independently of whether the cascade ends up using photometry: with the
/// geometry systematic removed, the **correct** pairs' distribution must
/// tighten, since what remains is lightcurve amplitude plus photometric noise.
/// If it widens instead, the phase term is being applied with the wrong sign
/// somewhere and nothing else in the report is worth reading.
///
/// Without this, "photometry does not help" and "photometry is miswired" look
/// identical from the sweep table alone.
fn print_delta_h_distributions(measurements: &[PairMeasurement]) {
    println!("\n  |dH| between the two arcs (magnitudes)");
    println!(
        "  {:<28} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "", "n", "p10", "p50", "p90", "p99"
    );

    for (kind, want) in [("correct pairs", Some(true)), ("WRONG pairs", Some(false))] {
        let mut xs: Vec<f64> = measurements
            .iter()
            .filter(|m| m.same_object == want)
            .filter_map(|m| m.delta_h)
            .filter(|v| v.is_finite())
            .collect();
        if xs.is_empty() {
            continue;
        }
        xs.sort_by(|a, b| a.total_cmp(b));
        let q = |f: f64| xs[(((xs.len() - 1) as f64) * f).round() as usize];
        println!(
            "  {:<28} {:>8} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
            kind,
            xs.len(),
            q(0.10),
            q(0.50),
            q(0.90),
            q(0.99)
        );
    }
}

/// Reduced chi-square distributions, split by ground truth.
///
/// The evidence behind any threshold choice: if the true and false populations
/// separate on this statistic a principled cut exists; if they overlap, no
/// amount of tuning will make it discriminate and the signal has to come from
/// elsewhere.
fn print_reduced_chi2_distributions(measurements: &[PairMeasurement]) {
    println!("\n  Reduced chi-square (expected ~1 if the filter were calibrated)");
    println!(
        "  {:<28} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "", "n", "p10", "p50", "p90", "p99"
    );

    for (label, first) in [
        ("first point (d2_1/2)", true),
        ("whole walk (sum/2n)", false),
    ] {
        for (kind, want) in [("correct pairs", Some(true)), ("WRONG pairs", Some(false))] {
            let mut xs: Vec<f64> = measurements
                .iter()
                .filter(|m| m.same_object == want)
                .filter_map(|m| {
                    if first {
                        m.reduced_chi2_first
                    } else {
                        m.reduced_chi2_all
                    }
                })
                .filter(|v| v.is_finite())
                .collect();
            if xs.is_empty() {
                continue;
            }
            xs.sort_by(|a, b| a.total_cmp(b));
            let q = |f: f64| xs[(((xs.len() - 1) as f64) * f).round() as usize];
            println!(
                "  {:<28} {:>8} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
                format!("{label} / {kind}"),
                xs.len(),
                q(0.10),
                q(0.50),
                q(0.90),
                q(0.99)
            );
        }
    }
}
