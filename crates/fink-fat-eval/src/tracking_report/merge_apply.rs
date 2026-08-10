//! Run the engine's fragment linkage and score what it produced.
//!
//! # Why this is separate from the shadow study
//!
//! [`merge_shadow`](super::merge_shadow) *explores*: it sweeps hundreds of
//! threshold combinations against ground truth to find an operating point. This
//! module *validates*: it runs the exact code path the pipeline would run,
//! with the retained parameters, and scores the result.
//!
//! They deliberately keep separate index implementations. The sweep needs to
//! vary things the engine hard-codes into [`MergeParams`], and having the
//! validation path go through
//! [`merge_fragments`](fink_fat_engine::topocentric_kf::branching::merge::merge_fragments)
//! is the whole point — a validation that re-implemented the engine would
//! prove nothing about the engine.
//!
//! # Why component purity, not pair precision
//!
//! Linkage is transitive: accepting A–B and B–C yields `{A, B, C}`. A single
//! wrong link therefore merges two entire components, so **99 % precision per
//! pair does not mean 99 % of merged components are pure** — and it is
//! component purity that feeds `contaminated` in the efficacy report. The
//! numbers below are the ones that decide whether merging is worth enabling.

use ahash::AHashMap;
use photom::{
    TrajId,
    observation_dataset::{ObsDataset, ObsId},
};

use fink_fat_engine::{
    engine_config::kalman_context::KalmanContext,
    topocentric_kf::branching::{
        BranchCollection,
        merge::{MergeFragment, MergeOutcome, MergeParams, component_track_ids, merge_fragments},
    },
};

use crate::{
    seed_bank_report::ground_truth::{ObsTrajLookup, SeedPurity},
    snapshot_report::efficacy::all_reconstructions,
};

/// Component sizes the report calls out individually before lumping the rest
/// together. A component past a handful of arcs is the signature of a chain of
/// wrong links rather than a heavily fragmented object.
const SIZE_BUCKETS: [usize; 5] = [1, 2, 3, 4, 5];

/// The merged reconstructions plus everything needed to judge them.
pub struct MergeApplication {
    /// Consolidated observation lists — the input to a second efficacy report.
    pub reconstructions: Vec<Vec<ObsId>>,
    outcome: MergeOutcome,
    /// Arcs the fragment build could not use at all (no MAP state, or no
    /// resolvable observation).
    n_skipped: usize,
}

impl MergeApplication {
    /// Borrowed view for
    /// [`compute_reconstruction_efficacy_for`](crate::snapshot_report::efficacy::compute_reconstruction_efficacy_for).
    pub fn as_slices(&self) -> Vec<&[ObsId]> {
        self.reconstructions.iter().map(Vec::as_slice).collect()
    }
}

/// Run the engine's linkage over a finished collection.
///
/// Test points are resolved once per fragment rather than per candidate pair:
/// every arc is a potential *tested* arc, and resolving lazily would repeat the
/// same ephemeris lookups across millions of pairs.
pub fn apply_merge<'a>(
    collection: &'a BranchCollection<'_, '_>,
    obs_dataset: &'a ObsDataset,
    kalman_context: &'a KalmanContext,
    params: &MergeParams,
) -> MergeApplication {
    let pool = all_reconstructions(collection);
    let n_live = collection.branches.len();

    let mut fragments: Vec<MergeFragment<'a, 'a>> = Vec::with_capacity(pool.len());
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

        fragments.push(MergeFragment {
            track_ids,
            state,
            h,
            span,
            test_points: crate::tracking_report::merge_shadow::sample_test_points_for(
                track_ids,
                params.max_test_points,
                obs_dataset,
                kalman_context,
            ),
        });
    }

    let outcome = merge_fragments(&fragments, params);

    let epoch_of = |id: ObsId| obs_dataset.get_observation(id).map(|o| o.mjd_tt());
    let reconstructions = outcome
        .components
        .iter()
        .map(|component| component_track_ids(&fragments, component, epoch_of))
        .collect();

    MergeApplication {
        reconstructions,
        outcome,
        n_skipped,
    }
}

/// `(first, last)` observation epoch of an arc, or `None` if no id resolves.
fn observation_span(track_ids: &[ObsId], obs_dataset: &ObsDataset) -> Option<(f64, f64)> {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
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

/// Report what linkage did, scored per **component**.
pub fn print_merge_application(
    application: &MergeApplication,
    ground_truth: &ObsTrajLookup,
    n_fragments_before: usize,
) {
    let sep = "=".repeat(90);
    println!("\n{sep}");
    println!("[Fragment linkage applied] engine merge_fragments, scored per component");
    println!("{sep}");

    let outcome = &application.outcome;
    println!(
        "  Arcs in                               : {} ({} skipped: no usable state)",
        n_fragments_before, application.n_skipped
    );
    println!(
        "  Candidate pairs / links accepted      : {} / {}",
        outcome.n_pairs_tested, outcome.n_links_accepted
    );
    println!(
        "  Components out                        : {}",
        outcome.components.len()
    );
    println!(
        "  Oversize components refused           : {} ({} arcs dissolved back to singletons)",
        outcome.n_components_rejected_oversize, outcome.n_arcs_in_rejected_components
    );

    // ── Size distribution ────────────────────────────────────────────────
    // A tail past a handful of arcs is a chain of wrong links, not a heavily
    // fragmented object, and it is invisible in any pairwise number.
    let mut by_size: AHashMap<usize, usize> = AHashMap::default();
    for component in &outcome.components {
        *by_size.entry(component.len()).or_default() += 1;
    }
    println!("\n  Component sizes");
    let mut beyond = 0usize;
    let mut largest = 0usize;
    for (&size, &count) in by_size.iter() {
        largest = largest.max(size);
        if !SIZE_BUCKETS.contains(&size) {
            beyond += count;
        }
    }
    for size in SIZE_BUCKETS {
        let n = by_size.get(&size).copied().unwrap_or(0);
        println!("  {:<20} {n:>9}", format!("{size} arc(s)"));
    }
    println!("  {:<20} {beyond:>9}  (largest: {largest})", "6+ arcs");

    // ── Purity of what merging produced ──────────────────────────────────
    // The translation of pairwise precision into the metric the efficacy
    // report actually uses.
    let (mut merged_pure, mut merged_mixed, mut merged_unknown) = (0usize, 0usize, 0usize);
    let mut objects_recovered: AHashMap<TrajId, usize> = AHashMap::default();
    for (component, track_ids) in outcome
        .components
        .iter()
        .zip(application.reconstructions.iter())
    {
        if component.len() < 2 {
            continue; // untouched by linkage: says nothing about merging
        }
        match ground_truth.classify(track_ids) {
            SeedPurity::Pure(traj) => {
                merged_pure += 1;
                *objects_recovered.entry(traj).or_default() += 1;
            }
            SeedPurity::Mixed => merged_mixed += 1,
            SeedPurity::Unknown => merged_unknown += 1,
        }
    }

    let judged = merged_pure + merged_mixed;
    println!("\n  Purity of the components linkage actually built (size >= 2)");
    println!("  {:<28} {merged_pure:>9}", "pure");
    println!("  {:<28} {merged_mixed:>9}", "MIXED (contamination)");
    println!("  {:<28} {merged_unknown:>9}", "unknown (unscorable)");
    if judged > 0 {
        println!(
            "  {:<28} {:>8.2}%   <- compare against the sweep's pairwise precision",
            "component purity",
            100.0 * merged_pure as f64 / judged as f64
        );
    }
    println!(
        "  {:<28} {:>9}",
        "distinct objects reunited",
        objects_recovered.len()
    );
    println!("{sep}");
}
