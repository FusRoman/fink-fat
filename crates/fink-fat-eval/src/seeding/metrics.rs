//! Quality metrics for intra-night seeding (pairs & triplets).
//!
//! Overview
//! --------
//! This module evaluates the *raw seeding stage* of `fink-fat-engine` (pairs and
//! triplets) against **ground-truth associations** stored in
//! [`AlertStoreWithTruth`].
//!
//! In the evaluation pipeline, seed generation produces candidate hypotheses:
//! - **Pairs**   `(a, b)`  — minimal intra-night linkage unit,
//! - **Triplets** `(a, b, c)` — higher-confidence seed built from pairs.
//!
//! These seeds are later consumed by downstream stages (graph building,
//! solvers, orbit fitting). The goal of this module is to quantify how *clean*
//! and how *complete* the seeds are **before** any global solver is applied.
//!
//! Truth model
//! -----------
//! The dataset provides a per-alert integer `trajectory_id` aligned with dense
//! [`AlertId`]. We interpret truth as:
//! - `trajectory_id > 0`  → alert is associated with a truth trajectory,
//! - `trajectory_id <= 0` → alert has no truth association (unknown/unmatched).
//!
//! Definitions
//! -----------
//! **True pair**
//! : A pair `(a, b)` is *true* if `tid(a) == tid(b)` and `tid(a) > 0`.
//!
//! **Contaminated pair**
//! : A pair is *contaminated* if both endpoints are truth-associated
//!   (`tid(a) > 0` and `tid(b) > 0`) but `tid(a) != tid(b)`.
//!
//! **True triplet**
//! : A triplet `(a, b, c)` is *true* if `tid(a) == tid(b) == tid(c)` and
//!   `tid(a) > 0`.
//!
//! **Contaminated triplet**
//! : A triplet is *contaminated* if all three endpoints are truth-associated
//!   (`tid > 0`) but they do not all share the same `trajectory_id`.
//!
//! Metrics reported
//! ----------------
//! The module computes two families of metrics:
//!
//! 1) **Purity / Precision (local correctness)**
//!    - `purity_overall`  = `n_true / n_total`
//!    - `precision_on_truth` = `n_true / n_both_truth` (pairs) or
//!      `n_true / n_all_truth` (triplets)
//!
//!    Interpretation:
//!    - `purity_overall` answers: *"Among all generated seeds, what fraction is
//!      actually correct?"*
//!    - `precision_on_truth` answers: *"If we restrict to seeds where truth is
//!      defined on all endpoints, how clean is the generator?"*
//!
//! 2) **Coverage proxy (continuity / recall-like)**
//!    Exact recall over *all possible same-trajectory pairs* is infeasible in
//!    large surveys because each truth trajectory of length `k` has O(k²) pairs.
//!    Instead, we use a scalable, meaningful proxy:
//!
//!    - For each truth trajectory, collect its alert ids.
//!    - Sort them by observation time (`mjd_tt`).
//!    - Define consecutive truth seeds:
//!      - pairs:    `(id[i], id[i+1])`
//!      - triplets: `(id[i], id[i+1], id[i+2])`
//!    - `consecutive_recall` is the fraction of those consecutive seeds that
//!      appear in the generated output.
//!
//!    Interpretation:
//!    - This measures whether the generator preserves the *temporal continuity*
//!      of each truth track, which is often the most important property for
//!      downstream linking.
//!
//! Complexity & performance notes
//! ------------------------------
//! - Building the seed membership sets (`HashSet`) is O(M) where `M` is the
//!   number of generated pairs/triplets.
//! - Grouping alerts by truth id is O(N) where `N` is the number of alerts.
//! - Sorting per trajectory drives the coverage proxy and yields an overall
//!   O(N log N) behavior in practice (sum of per-trajectory sorts).
//!
//! This module is deterministic as long as:
//! - the input seed lists are deterministic,
//! - the observation times are deterministic (they are data-driven).
//!
//! See also
//! --------
//! - [`crate::dataset::ztf_alerts::AlertStoreWithTruth`] — engine store + truth sidecar.
//! - `fink-fat-engine::seeding::{pairs, triplets}` — seed generation algorithms.

use std::collections::{HashMap, HashSet};

use fink_fat_engine::{
    AlertId,
    seeding::{
        pairs::{Pair, Pairs},
        triplets::{Triplet, Triplets},
    },
};

use crate::dataset::ztf_alerts::AlertStoreWithTruth;

/// Aggregate metrics for a set of generated pairs.
///
/// Notes
/// -----
/// - `n_both_truth` counts pairs where both endpoints have `trajectory_id > 0`.
/// - `precision_on_truth` is computed only over `n_both_truth` to avoid being
///   dominated by unassociated alerts.
/// - `consecutive_recall` uses the *coverage proxy* described in the module docs.
#[derive(Debug, Clone)]
pub struct PairMetrics {
    /// Total number of generated pairs.
    pub n_total: usize,

    /// Pairs where both endpoints are truth-associated (`tid(a) > 0 && tid(b) > 0`).
    pub n_both_truth: usize,
    /// Pairs where exactly one endpoint is truth-associated.
    pub n_one_truth: usize,
    /// Pairs where neither endpoint is truth-associated.
    pub n_none_truth: usize,

    /// Number of true pairs (`tid(a) == tid(b) > 0`).
    pub n_true: usize,
    /// Number of contaminated pairs (both truth-associated but different truth ids).
    pub n_contaminated: usize,

    /// Precision restricted to truth-defined pairs: `n_true / n_both_truth`.
    pub precision_on_truth: f64,
    /// Overall purity among all generated pairs: `n_true / n_total`.
    pub purity_overall: f64,

    /// Coverage proxy: `n_consecutive_truth_pairs_found / n_consecutive_truth_pairs`.
    pub consecutive_recall: f64,
    /// Number of consecutive truth pairs available in the dataset.
    pub n_consecutive_truth_pairs: usize,
    /// Number of consecutive truth pairs present in the generated output.
    pub n_consecutive_truth_pairs_found: usize,
}

/// Aggregate metrics for a set of generated triplets.
///
/// Notes
/// -----
/// - `n_all_truth` counts triplets where all 3 endpoints have `trajectory_id > 0`.
/// - `precision_on_truth` is computed only over `n_all_truth`.
/// - `consecutive_recall` uses the *coverage proxy* described in the module docs.
#[derive(Debug, Clone)]
pub struct TripletMetrics {
    /// Total number of generated triplets.
    pub n_total: usize,

    /// Triplets where all 3 endpoints are truth-associated (`tid > 0`).
    pub n_all_truth: usize,
    /// Triplets with at least one truth endpoint, but not all three.
    pub n_partial_truth: usize,
    /// Triplets where none of the endpoints are truth-associated.
    pub n_none_truth: usize,

    /// Number of true triplets (`tid(a) == tid(b) == tid(c) > 0`).
    pub n_true: usize,
    /// Number of contaminated triplets (all truth-associated but not all equal).
    pub n_contaminated: usize,

    /// Precision restricted to truth-defined triplets: `n_true / n_all_truth`.
    pub precision_on_truth: f64,
    /// Overall purity among all generated triplets: `n_true / n_total`.
    pub purity_overall: f64,

    /// Coverage proxy: `n_consecutive_truth_triplets_found / n_consecutive_truth_triplets`.
    pub consecutive_recall: f64,
    /// Number of consecutive truth triplets available in the dataset.
    pub n_consecutive_truth_triplets: usize,
    /// Number of consecutive truth triplets present in the generated output.
    pub n_consecutive_truth_triplets_found: usize,
}

/// Return the truth id (`trajectory_id`) for a given alert id.
///
/// Parameters
/// ----------
/// store : &AlertStoreWithTruth
///     Store holding the truth sidecar aligned with dense [`AlertId`].
/// id : AlertId
///     Alert identifier (dense row index).
///
/// Returns
/// -------
/// i32
///     The truth trajectory id for this alert.
///     Values `<= 0` are considered "unassociated / unknown truth".
///
/// Notes
/// -----
/// This is an O(1) lookup into a sidecar vector:
/// `store.trajectory_id[id.idx()]`.
#[inline]
fn tid(store: &AlertStoreWithTruth, id: AlertId) -> i32 {
    store.trajectory_id[id.idx()]
}

/// Safe division helper for ratios.
///
/// Parameters
/// ----------
/// num : usize
///     Numerator.
/// den : usize
///     Denominator.
///
/// Returns
/// -------
/// f64
///     `num / den` as `f64`, or `0.0` if `den == 0`.
#[inline]
fn frac(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        (num as f64) / (den as f64)
    }
}

/// Build a membership set for generated pairs `(a_idx, b_idx)`.
///
/// Parameters
/// ----------
/// pairs : &Pairs
///     Generated pairs to index.
///
/// Returns
/// -------
/// HashSet<(usize, usize)>
///     A set of keys `(a_idx, b_idx)` where `a_idx = AlertId::idx()`
///     and similarly for `b_idx`.
///
/// Notes
/// -----
/// - This set is used to test whether a consecutive truth pair is present
///   in the generated output in O(1) average time.
/// - The order `(a, b)` is preserved. If your generator can produce both
///   directions, you may want to also check `(b, a)` depending on semantics.
#[inline]
fn pair_key_set(pairs: &Pairs) -> HashSet<(usize, usize)> {
    pairs.iter().map(|p| (p.a.idx(), p.b.idx())).collect()
}

/// Build a membership set for generated triplets `(a_idx, b_idx, c_idx)`.
///
/// Parameters
/// ----------
/// triplets : &Triplets
///     Generated triplets to index.
///
/// Returns
/// -------
/// HashSet<(usize, usize, usize)>
///     A set of keys `(a_idx, b_idx, c_idx)` (dense alert indices).
///
/// Notes
/// -----
/// - Used to test whether a consecutive truth triplet is present in O(1)
///   average time.
/// - The order `(a, b, c)` is preserved.
#[inline]
fn triplet_key_set(triplets: &Triplets) -> HashSet<(usize, usize, usize)> {
    triplets
        .iter()
        .map(|t| (t.a.idx(), t.b.idx(), t.c.idx()))
        .collect()
}

/// Group alert indices by truth trajectory id (`trajectory_id > 0`).
///
/// Parameters
/// ----------
/// store : &AlertStoreWithTruth
///     Store holding the truth sidecar aligned with dense alert order.
///
/// Returns
/// -------
/// HashMap<i32, Vec<usize>>
///     Map `trajectory_id -> list of alert indices` (dense row indices),
///     containing only truth-associated alerts (`trajectory_id > 0`).
///
/// Notes
/// -----
/// - This is the core preprocessing step for the coverage proxy.
/// - The returned vectors are *not sorted*; callers typically sort them by time.
/// - Allocation size is proportional to the number of truth-associated alerts.
#[inline]
fn group_indices_by_tid(store: &AlertStoreWithTruth) -> HashMap<i32, Vec<usize>> {
    store
        .trajectory_id
        .iter()
        .enumerate()
        .filter_map(|(idx, &t)| (t > 0).then_some((t, idx)))
        .fold(HashMap::new(), |mut acc, (t, idx)| {
            acc.entry(t).or_default().push(idx);
            acc
        })
}

/// Sort alert indices by observation time (`mjd_tt`) in-place.
///
/// Parameters
/// ----------
/// store : &AlertStoreWithTruth
///     Store providing the observation time `mjd_tt` for each alert.
/// ids : &mut [usize]
///     Mutable slice of dense alert indices to sort.
///
/// Returns
/// -------
/// ()
///
/// Notes
/// -----
/// - Sorting by `mjd_tt` makes the coverage proxy robust to datasets where row
///   order is not strictly chronological.
/// - If `mjd_tt` contains NaNs, sorting uses a fallback ordering that may hide
///   data issues. In well-formed datasets, `mjd_tt` should be finite.
#[inline]
fn sort_ids_by_time(store: &AlertStoreWithTruth, ids: &mut [usize]) {
    ids.sort_by(|&i, &j| {
        let ti = store.store.alerts[i].mjd_tt;
        let tj = store.store.alerts[j].mjd_tt;
        ti.partial_cmp(&tj).unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Compute the coverage proxy for "consecutive truth seeds".
///
/// Overview
/// --------
/// This routine implements the recall-like proxy described at the module level.
/// For each truth trajectory, it sorts alerts by time and enumerates consecutive
/// windows of size `k`. It then counts how many of those windows are present in
/// the generated output (via the user-provided membership predicate).
///
/// Parameters
/// ----------
/// store : &AlertStoreWithTruth
///     Store holding alerts and truth association.
/// k : usize
///     Window size:
///     - `k = 2` for consecutive truth pairs `(i, i+1)`
///     - `k = 3` for consecutive truth triplets `(i, i+1, i+2)`
/// contains : F
///     Membership predicate called with a slice `w` of length `k` containing
///     dense alert indices. Should return `true` if that window exists among
///     generated seeds.
///
/// Returns
/// -------
/// (usize, usize)
///     `(n_possible, n_found)` where:
///     - `n_possible` is the number of consecutive truth windows across all truth
///       trajectories,
///     - `n_found` is how many of those windows exist in the generated output.
///
/// Notes
/// -----
/// - Complexity is dominated by per-trajectory sorting: overall O(N log N)
///   in typical survey-like data.
/// - This proxy is intentionally conservative: it requires exact consecutive
///   neighbors by time. It does not count non-consecutive same-trajectory seeds.
fn consecutive_truth_coverage<F>(
    store: &AlertStoreWithTruth,
    k: usize,
    mut contains: F,
) -> (usize, usize)
where
    F: FnMut(&[usize]) -> bool,
{
    // Step 1: group all truth-associated alerts by trajectory id.
    let mut by_tid = group_indices_by_tid(store);

    // Step 2: for each trajectory, sort by time and scan consecutive windows.
    by_tid
        .values_mut()
        .filter(|ids| ids.len() >= k)
        .map(|ids| {
            sort_ids_by_time(store, ids);

            // windows(k): consecutive sequences of length k.
            ids.windows(k)
                .fold((0usize, 0usize), |(possible, found), w| {
                    let possible = possible + 1;
                    let found = found + (contains(w) as usize);
                    (possible, found)
                })
        })
        // Step 3: aggregate across trajectories.
        .fold((0usize, 0usize), |(p0, f0), (p1, f1)| (p0 + p1, f0 + f1))
}

/// Compute pair-quality metrics using truth association.
///
/// Parameters
/// ----------
/// store : &AlertStoreWithTruth
///     Store holding the engine alerts and the truth sidecar (`trajectory_id`).
/// pairs : &Pairs
///     Generated pairs to evaluate.
///
/// Returns
/// -------
/// PairMetrics
///     Aggregate counts and ratios:
///     - truth availability breakdown (`n_both_truth`, `n_one_truth`, `n_none_truth`)
///     - local correctness (`n_true`, `n_contaminated`, precision/purity)
///     - continuity proxy (`consecutive_recall`, `n_consecutive_*`)
///
/// Notes
/// -----
/// - "True" pairs require `trajectory_id > 0` and equality across endpoints.
/// - "Contaminated" pairs require both endpoints truth-associated but different.
/// - Coverage proxy is evaluated against consecutive truth pairs sorted by `mjd_tt`.
pub fn pair_metrics(store: &AlertStoreWithTruth, pairs: &Pairs) -> PairMetrics {
    let n_total = pairs.len();

    // Membership set for fast O(1) checks in the coverage proxy.
    let pair_set = pair_key_set(pairs);

    // Single pass: count truth-availability buckets and correctness categories.
    let (n_both_truth, n_one_truth, n_none_truth, n_true, n_contaminated) = pairs.iter().fold(
        (0usize, 0usize, 0usize, 0usize, 0usize),
        |(both, one, none, true_n, contam), Pair { a, b }| {
            let ta = tid(store, *a);
            let tb = tid(store, *b);

            let a_truth = ta > 0;
            let b_truth = tb > 0;

            match (a_truth, b_truth) {
                // Both endpoints have truth: can be true or contaminated.
                (true, true) => (
                    both + 1,
                    one,
                    none,
                    true_n + (ta == tb) as usize,
                    contam + (ta != tb) as usize,
                ),
                // One endpoint has truth: neither true nor contaminated by definition.
                (true, false) | (false, true) => (both, one + 1, none, true_n, contam),
                // No truth information: cannot assess correctness.
                (false, false) => (both, one, none + 1, true_n, contam),
            }
        },
    );

    let precision_on_truth = frac(n_true, n_both_truth);
    let purity_overall = frac(n_true, n_total);

    // Coverage proxy: consecutive truth pairs that appear in generated output.
    let (n_consecutive_truth_pairs, n_consecutive_truth_pairs_found) =
        consecutive_truth_coverage(store, 2, |w| pair_set.contains(&(w[0], w[1])));

    let consecutive_recall = frac(n_consecutive_truth_pairs_found, n_consecutive_truth_pairs);

    PairMetrics {
        n_total,
        n_both_truth,
        n_one_truth,
        n_none_truth,
        n_true,
        n_contaminated,
        precision_on_truth,
        purity_overall,
        consecutive_recall,
        n_consecutive_truth_pairs,
        n_consecutive_truth_pairs_found,
    }
}

/// Compute triplet-quality metrics using truth association.
///
/// Parameters
/// ----------
/// store : &AlertStoreWithTruth
///     Store holding the engine alerts and the truth sidecar (`trajectory_id`).
/// triplets : &Triplets
///     Generated triplets to evaluate.
///
/// Returns
/// -------
/// TripletMetrics
///     Aggregate counts and ratios:
///     - truth availability breakdown (`n_all_truth`, `n_partial_truth`, `n_none_truth`)
///     - local correctness (`n_true`, `n_contaminated`, precision/purity)
///     - continuity proxy (`consecutive_recall`, `n_consecutive_*`)
///
/// Notes
/// -----
/// - "True" triplets require `trajectory_id > 0` on all endpoints and equality.
/// - "Contaminated" triplets require all endpoints truth-associated but not all equal.
/// - Coverage proxy is evaluated against consecutive truth triplets sorted by `mjd_tt`.
pub fn triplet_metrics(store: &AlertStoreWithTruth, triplets: &Triplets) -> TripletMetrics {
    let n_total = triplets.len();

    // Membership set for O(1) checks in the coverage proxy.
    let triplet_set = triplet_key_set(triplets);

    // Single pass: truth-availability buckets and correctness categories.
    let (n_all_truth, n_partial_truth, n_none_truth, n_true, n_contaminated) =
        triplets.iter().fold(
            (0usize, 0usize, 0usize, 0usize, 0usize),
            |(all, partial, none, true_n, contam), Triplet { a, b, c }| {
                let ta = tid(store, *a);
                let tb = tid(store, *b);
                let tc = tid(store, *c);

                let a_truth = ta > 0;
                let b_truth = tb > 0;
                let c_truth = tc > 0;

                // Count how many endpoints have truth info.
                let n_truth = (a_truth as u8) + (b_truth as u8) + (c_truth as u8);

                match n_truth {
                    3 => {
                        let is_true = (ta == tb) && (tb == tc);
                        (
                            all + 1,
                            partial,
                            none,
                            true_n + (is_true as usize),
                            contam + ((!is_true) as usize),
                        )
                    }
                    0 => (all, partial, none + 1, true_n, contam),
                    _ => (all, partial + 1, none, true_n, contam),
                }
            },
        );

    let precision_on_truth = frac(n_true, n_all_truth);
    let purity_overall = frac(n_true, n_total);

    // Coverage proxy: consecutive truth triplets that appear in generated output.
    let (n_consecutive_truth_triplets, n_consecutive_truth_triplets_found) =
        consecutive_truth_coverage(store, 3, |w| triplet_set.contains(&(w[0], w[1], w[2])));

    let consecutive_recall = frac(
        n_consecutive_truth_triplets_found,
        n_consecutive_truth_triplets,
    );

    TripletMetrics {
        n_total,
        n_all_truth,
        n_partial_truth,
        n_none_truth,
        n_true,
        n_contaminated,
        precision_on_truth,
        purity_overall,
        consecutive_recall,
        n_consecutive_truth_triplets,
        n_consecutive_truth_triplets_found,
    }
}
