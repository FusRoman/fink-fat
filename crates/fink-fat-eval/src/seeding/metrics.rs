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

use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use comfy_table::{Cell, Table, presets::UTF8_FULL};
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

/// Format a ratio as a human-friendly percentage string.
///
/// Overview
/// --------
/// This helper converts a ratio expressed as a fraction in `[0, 1]` into a
/// percentage string with two decimals, e.g.:
/// - `0.0  -> "0.00%"`
/// - `0.5  -> "50.00%"`
/// - `1.0  -> "100.00%"`
///
/// Parameters
/// ----------
/// x : f64
///     Ratio as a floating-point fraction (typically in `[0, 1]`).
///
/// Returns
/// -------
/// String
///     Percentage string formatted as `"xx.xx%"`.
///
/// Notes
/// -----
/// - This function does not clamp the input. If `x < 0.0` or `x > 1.0`,
///   the output will reflect that (e.g. `1.2 -> "120.00%"`).
/// - `NaN` and infinities are formatted using Rust's default float formatting.
#[inline]
fn pct(x: f64) -> String {
    format!("{:.2}%", 100.0 * x)
}

/// Format an integer value for terminal display.
///
/// Overview
/// --------
/// This helper keeps numeric rendering intentionally simple and dependency-free.
/// It currently formats a `usize` using the standard decimal representation.
///
/// Parameters
/// ----------
/// x : usize
///     Value to format.
///
/// Returns
/// -------
/// String
///     Decimal string representation of `x`.
///
/// Notes
/// -----
/// - We intentionally do not add thousand separators here to avoid extra
///   dependencies and locale issues.
/// - If you want grouped formatting (e.g. `1_000_000`), consider adding an
///   optional feature using a dedicated crate and formatting only at the UI layer.
#[inline]
fn n(x: usize) -> String {
    x.to_string()
}

/// Build a 2-column row for a [`comfy_table::Table`].
///
/// Overview
/// --------
/// The pretty terminal output for metrics uses a fixed 2-column layout:
/// - left column: a descriptive label (`k`)
/// - right column: a value (`v`)
///
/// This helper ensures:
/// - a consistent row shape (`[Cell; 2]`),
/// - a small call-site footprint (`t.add_row(row2(...))`),
/// - uniform handling of value types through `Into<Cell>`.
///
/// Parameters
/// ----------
/// k : &str
///     Row label (left column).
/// v : impl Into<Cell>
///     Row value (right column). Typically a `String`, `&str`, or `Cell`.
///
/// Returns
/// -------
/// [Cell; 2]
///     Two cells representing one table row: `[label, value]`.
///
/// Notes
/// -----
/// - This is purely a UI helper and has no impact on metric computation.
/// - The returned array can be passed directly to `Table::add_row`.
#[inline]
fn row2<'a>(k: &str, v: impl Into<Cell>) -> [Cell; 2] {
    [Cell::new(k), v.into()]
}

/// Render [`PairMetrics`] as a pretty terminal table.
///
/// Overview
/// --------
/// This function builds a [`comfy_table::Table`] (UTF-8 box drawing preset)
/// describing the pair metrics in a compact, readable layout.
///
/// The table is intended for:
/// - evaluation logs,
/// - benchmark summaries,
/// - quick CLI inspection of seeding quality.
///
/// Parameters
/// ----------
/// m : &PairMetrics
///     Pair metrics to render.
///
/// Returns
/// -------
/// Table
///     A fully populated table ready to be printed (e.g. via `println!("{table}")`).
///
/// Notes
/// -----
/// - Rendering is deterministic.
/// - This function is UI-only: it does not recompute metrics, it only formats
///   the already aggregated fields of [`PairMetrics`].
/// - Column names are fixed to keep output stable for parsing / log scraping.
pub fn pair_table(m: &PairMetrics) -> Table {
    let mut t = Table::new();
    t.load_preset(UTF8_FULL);
    t.set_header(vec!["Pair metrics", "Value"]);

    t.add_row(row2("total pairs", n(m.n_total)));
    t.add_row(row2("true pairs", n(m.n_true)));
    t.add_row(row2("contaminated (truth mismatch)", n(m.n_contaminated)));

    t.add_row(row2(
        "truth-defined pairs (both endpoints)",
        n(m.n_both_truth),
    ));
    t.add_row(row2("one truth endpoint", n(m.n_one_truth)));
    t.add_row(row2("no truth endpoints", n(m.n_none_truth)));

    t.add_row(row2(
        "precision on truth-defined",
        pct(m.precision_on_truth),
    ));
    t.add_row(row2("purity overall", pct(m.purity_overall)));

    t.add_row(row2(
        "consecutive truth pairs (possible)",
        n(m.n_consecutive_truth_pairs),
    ));
    t.add_row(row2(
        "consecutive truth pairs (found)",
        n(m.n_consecutive_truth_pairs_found),
    ));
    t.add_row(row2("consecutive recall", pct(m.consecutive_recall)));

    t
}

/// Render [`TripletMetrics`] as a pretty terminal table.
///
/// Overview
/// --------
/// This function builds a [`comfy_table::Table`] (UTF-8 box drawing preset)
/// describing the triplet metrics in a compact, readable layout.
///
/// Parameters
/// ----------
/// m : &TripletMetrics
///     Triplet metrics to render.
///
/// Returns
/// -------
/// Table
///     A fully populated table ready to be printed.
///
/// Notes
/// -----
/// - Rendering is deterministic.
/// - This function is UI-only and does not recompute metrics.
/// - The layout mirrors [`pair_table`] to keep logs consistent across seed types.
pub fn triplet_table(m: &TripletMetrics) -> Table {
    let mut t = Table::new();
    t.load_preset(UTF8_FULL);
    t.set_header(vec!["Triplet metrics", "Value"]);

    t.add_row(row2("total triplets", n(m.n_total)));
    t.add_row(row2("true triplets", n(m.n_true)));
    t.add_row(row2("contaminated (truth mismatch)", n(m.n_contaminated)));

    t.add_row(row2(
        "truth-defined triplets (all 3 endpoints)",
        n(m.n_all_truth),
    ));
    t.add_row(row2("partial truth (1-2 endpoints)", n(m.n_partial_truth)));
    t.add_row(row2("no truth endpoints", n(m.n_none_truth)));

    t.add_row(row2(
        "precision on truth-defined",
        pct(m.precision_on_truth),
    ));
    t.add_row(row2("purity overall", pct(m.purity_overall)));

    t.add_row(row2(
        "consecutive truth triplets (possible)",
        n(m.n_consecutive_truth_triplets),
    ));
    t.add_row(row2(
        "consecutive truth triplets (found)",
        n(m.n_consecutive_truth_triplets_found),
    ));
    t.add_row(row2("consecutive recall", pct(m.consecutive_recall)));

    t
}

/// Pretty display for [`PairMetrics`].
///
/// Overview
/// --------
/// This implementation prints pair metrics as a terminal-friendly table.
/// It is intended for interactive evaluation and benchmark logs.
///
/// Parameters
/// ----------
/// f : &mut fmt::Formatter<'_>
///     Formatter provided by the standard formatting machinery.
///
/// Returns
/// -------
/// fmt::Result
///     Formatting result from `write!`.
///
/// Notes
/// -----
/// - This is a presentation layer only: the values are assumed to have been
///   computed already by [`pair_metrics`].
/// - If you want a compact, one-line output for high-volume logs, consider
///   adding a dedicated `to_compact_line()` helper alongside this `Display`.
impl fmt::Display for PairMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let table = pair_table(self);
        return write!(f, "{table}");
    }
}

/// Pretty display for [`TripletMetrics`].
///
/// Overview
/// --------
/// This implementation prints triplet metrics as a terminal-friendly table.
/// It mirrors the layout of [`PairMetrics`] to keep logs consistent.
///
/// Parameters
/// ----------
/// f : &mut fmt::Formatter<'_>
///     Formatter provided by the standard formatting machinery.
///
/// Returns
/// -------
/// fmt::Result
///     Formatting result from `write!`.
///
/// Notes
/// -----
/// - This is presentation-only and assumes metrics were computed by
///   [`triplet_metrics`].
impl fmt::Display for TripletMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let table = triplet_table(self);
        return write!(f, "{table}");
    }
}

#[cfg(test)]
mod seed_metrics_tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use prop_test::proptest::{self, strategy::ValueTree, test_runner::TestRunner};
    use proptest::prelude::*;

    use fink_fat_engine::{Alert, alerts::AlertStore};

    fn mk_alert(id: usize, mjd_tt: f64) -> Alert {
        Alert {
            id: AlertId::from(id),
            dia_source_id: id as u64,
            ra: 0.0,
            ra_err: 0.0,
            dec: 0.0,
            dec_err: 0.0,
            mjd_tt,
            flux: 0.0,
            flux_err: 0.0,
            band: 1,
        }
    }

    /// Build a tiny `AlertStoreWithTruth` for testing.
    ///
    /// Notes
    /// -----
    /// - `trajectory_id.len()` must match `times.len()`.
    /// - Alerts are created with dense `AlertId` matching row order.
    fn mk_store_with_truth(trajectory_id: Vec<i32>, times: Vec<f64>) -> AlertStoreWithTruth {
        assert_eq!(trajectory_id.len(), times.len());
        let alerts: Vec<Alert> = times
            .into_iter()
            .enumerate()
            .map(|(i, t)| mk_alert(i, t))
            .collect();

        let min_mjd = alerts
            .iter()
            .map(|a| a.mjd_tt)
            .fold(f64::INFINITY, f64::min);

        let store = AlertStore::new(min_mjd.floor(), alerts);
        AlertStoreWithTruth {
            store,
            trajectory_id,
        }
    }

    fn pair_builder(a: usize, b: usize) -> Pair {
        Pair {
            a: AlertId::from(a),
            b: AlertId::from(b),
        }
    }

    fn triplet_builder(a: usize, b: usize, c: usize) -> Triplet {
        Triplet {
            a: AlertId::from(a),
            b: AlertId::from(b),
            c: AlertId::from(c),
        }
    }

    #[test]
    fn unit_pair_metrics_basic_counts() {
        // Two truth trajectories:
        // tid=10: alerts 0,1,2
        // tid=20: alerts 3,4
        // tid<=0: alert 5 (unassociated)
        let store = mk_store_with_truth(
            vec![10, 10, 10, 20, 20, 0],
            vec![1.0, 2.0, 3.0, 1.5, 2.5, 9.0],
        );

        // Pairs: 4 total
        // (0,1) true (both truth, same tid)
        // (1,3) contaminated (both truth, different tid)
        // (4,5) one_truth (tid(5)=0)
        // (2,5) one_truth
        let pairs: Pairs = vec![
            pair_builder(0, 1),
            pair_builder(1, 3),
            pair_builder(4, 5),
            pair_builder(2, 5),
        ];

        let m = pair_metrics(&store, &pairs);

        assert_eq!(m.n_total, 4);
        assert_eq!(m.n_both_truth, 2); // (0,1), (1,3)
        assert_eq!(m.n_one_truth, 2); // (4,5), (2,5)
        assert_eq!(m.n_none_truth, 0);

        assert_eq!(m.n_true, 1);
        assert_eq!(m.n_contaminated, 1);

        assert_relative_eq!(m.precision_on_truth, 1.0 / 2.0, epsilon = 1e-12);
        assert_relative_eq!(m.purity_overall, 1.0 / 4.0, epsilon = 1e-12);

        // Consecutive truth pairs:
        // tid=10 times: (0@1.0, 1@2.0, 2@3.0) -> (0,1) and (1,2)
        // tid=20 times: (3@1.5, 4@2.5) -> (3,4)
        // possible = 3
        // found: only (0,1) is present -> 1
        assert_eq!(m.n_consecutive_truth_pairs, 3);
        assert_eq!(m.n_consecutive_truth_pairs_found, 1);
        assert_relative_eq!(m.consecutive_recall, 1.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn unit_triplet_metrics_basic_counts() {
        // tid=10: 0,1,2
        // tid=20: 3,4,5
        // plus an unassociated 6
        let store = mk_store_with_truth(
            vec![10, 10, 10, 20, 20, 20, 0],
            vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 0.5],
        );

        // Triplets:
        // (0,1,2) true
        // (3,4,6) partial truth (6 is unassociated)
        // (1,3,4) contaminated (all truth but different tids)
        let triplets: Triplets = vec![
            triplet_builder(0, 1, 2),
            triplet_builder(3, 4, 6),
            triplet_builder(1, 3, 4),
        ];

        let m = triplet_metrics(&store, &triplets);

        assert_eq!(m.n_total, 3);
        assert_eq!(m.n_all_truth, 2); // (0,1,2) and (1,3,4)
        assert_eq!(m.n_partial_truth, 1); // (3,4,6)
        assert_eq!(m.n_none_truth, 0);

        assert_eq!(m.n_true, 1);
        assert_eq!(m.n_contaminated, 1);

        assert_relative_eq!(m.precision_on_truth, 1.0 / 2.0, epsilon = 1e-12);
        assert_relative_eq!(m.purity_overall, 1.0 / 3.0, epsilon = 1e-12);

        // Consecutive truth triplets:
        // tid=10: 0,1,2 -> one consecutive triplet (0,1,2)
        // tid=20: 3,4,5 -> one consecutive triplet (3,4,5)
        // possible=2, found=1 (only (0,1,2) present)
        assert_eq!(m.n_consecutive_truth_triplets, 2);
        assert_eq!(m.n_consecutive_truth_triplets_found, 1);
        assert_relative_eq!(m.consecutive_recall, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn unit_perfect_consecutive_pairs_have_recall_1() {
        // One trajectory tid=7 of length 5
        let store = mk_store_with_truth(vec![7, 7, 7, 7, 7], vec![5.0, 4.0, 3.0, 2.0, 1.0]);

        // Build exactly the consecutive truth pairs, *in time order*.
        // Times are decreasing in row order, but the metric sorts by mjd_tt.
        // Sorted by time: ids (4,3,2,1,0)
        let pairs: Pairs = vec![
            pair_builder(4, 3),
            pair_builder(3, 2),
            pair_builder(2, 1),
            pair_builder(1, 0),
        ];

        let m = pair_metrics(&store, &pairs);

        assert_eq!(m.n_consecutive_truth_pairs, 4);
        assert_eq!(m.n_consecutive_truth_pairs_found, 4);
        assert_abs_diff_eq!(m.consecutive_recall, 1.0, epsilon = 1e-12);

        // All pairs are true and both endpoints truth-associated.
        assert_eq!(m.n_both_truth, 4);
        assert_eq!(m.n_true, 4);
        assert_abs_diff_eq!(m.precision_on_truth, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(m.purity_overall, 1.0, epsilon = 1e-12);
    }

    /* --------------------------------------------------------------------- */
    /* Property-based tests (proptest)                                        */
    /* --------------------------------------------------------------------- */

    // Generate a small store with:
    // - N alerts (N <= 25)
    // - truth ids in [0..=5] (0 means unassociated)
    // - times are arbitrary finite floats
    prop_compose! {
        fn arb_store()
            (n in 1usize..=25)
            (tids in prop::collection::vec(0i32..=5, n),
             // avoid NaN/infinite to keep sorting/metrics stable
             times in prop::collection::vec(-1.0e6f64..=1.0e6f64, n))
            -> AlertStoreWithTruth
        {
            mk_store_with_truth(tids, times)
        }
    }

    prop_compose! {
        fn arb_pairs(n_alerts: usize)
            (m in 0usize..=200)
            (pairs in prop::collection::vec((0usize..n_alerts, 0usize..n_alerts), m))
            -> Pairs
        {
            pairs.into_iter().map(|(a,b)| pair_builder(a,b)).collect()
        }
    }

    prop_compose! {
        fn arb_triplets(n_alerts: usize)
            (m in 0usize..=200)
            (triplets in prop::collection::vec((0usize..n_alerts, 0usize..n_alerts, 0usize..n_alerts), m))
            -> Triplets
        {
            triplets.into_iter().map(|(a,b,c)| triplet_builder(a,b,c)).collect()
        }
    }

    proptest! {
        #[test]
        fn prop_pair_metrics_invariants(store in arb_store()) {
            let n = store.trajectory_id.len();
            let pairs = arb_pairs(n).new_tree(&mut TestRunner::default()).unwrap().current();

            let m = pair_metrics(&store, &pairs);

            // Basic partition invariants.
            prop_assert_eq!(m.n_total, pairs.len());
            prop_assert_eq!(m.n_both_truth + m.n_one_truth + m.n_none_truth, m.n_total);

            // True/contaminated must be within both-truth pairs.
            prop_assert!(m.n_true <= m.n_both_truth);
            prop_assert!(m.n_contaminated <= m.n_both_truth);

            // Ratios must be in [0,1] (or 0 when denominator 0).
            prop_assert!(0.0 <= m.precision_on_truth && m.precision_on_truth <= 1.0);
            prop_assert!(0.0 <= m.purity_overall && m.purity_overall <= 1.0);
            prop_assert!(0.0 <= m.consecutive_recall && m.consecutive_recall <= 1.0);

            // Coverage counters are consistent.
            prop_assert!(m.n_consecutive_truth_pairs_found <= m.n_consecutive_truth_pairs);
        }

        #[test]
        fn prop_triplet_metrics_invariants(store in arb_store()) {
            let n = store.trajectory_id.len();
            let triplets = arb_triplets(n).new_tree(&mut TestRunner::default()).unwrap().current();

            let m = triplet_metrics(&store, &triplets);

            prop_assert_eq!(m.n_total, triplets.len());
            prop_assert_eq!(m.n_all_truth + m.n_partial_truth + m.n_none_truth, m.n_total);

            prop_assert!(m.n_true <= m.n_all_truth);
            prop_assert!(m.n_contaminated <= m.n_all_truth);

            prop_assert!(0.0 <= m.precision_on_truth && m.precision_on_truth <= 1.0);
            prop_assert!(0.0 <= m.purity_overall && m.purity_overall <= 1.0);
            prop_assert!(0.0 <= m.consecutive_recall && m.consecutive_recall <= 1.0);

            prop_assert!(m.n_consecutive_truth_triplets_found <= m.n_consecutive_truth_triplets);
        }
    }

    proptest! {
        #[test]
        fn prop_perfect_consecutive_pairs_recall_is_one(
            // one truth trajectory id > 0
            n in 2usize..=25,
            tid in 1i32..=10,
            // strict monotone times guarantee unique ordering
            base in -1.0e6f64..=1.0e6f64,
            step in 1.0e-3f64..=1.0e3f64
        ) {
            let tids = vec![tid; n];

            // times increasing: base + i*step
            let times: Vec<f64> = (0..n).map(|i| base + (i as f64) * step).collect();
            let store = mk_store_with_truth(tids, times);

            // consecutive pairs in time order are exactly (0,1), (1,2), ...
            let pairs: Pairs = (0..(n-1)).map(|i| pair_builder(i, i+1)).collect();

            let m = pair_metrics(&store, &pairs);

            prop_assert_eq!(m.n_consecutive_truth_pairs, n-1);
            prop_assert_eq!(m.n_consecutive_truth_pairs_found, n-1);
            prop_assert!( (m.consecutive_recall - 1.0).abs() <= 1e-12 );

            // all true
            prop_assert_eq!(m.n_true, n-1);
            prop_assert_eq!(m.n_contaminated, 0);
            prop_assert!( (m.precision_on_truth - 1.0).abs() <= 1e-12 );
            prop_assert!( (m.purity_overall - 1.0).abs() <= 1e-12 );
        }
    }
}
