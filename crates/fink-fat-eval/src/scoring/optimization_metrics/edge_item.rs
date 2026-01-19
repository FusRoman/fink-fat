use crate::scoring::optimization_metrics::{BestYouden, metrics::{BestF1, Metrics, OperatingPoint}};

/// Iterate over contiguous **tie blocks** of equal keys in a `(key, is_good)` array.
///
/// Overview
/// --------
/// This helper groups together consecutive elements that share the same `key`
/// value (a *tie block*) after sorting the input in ascending order of `key`.
/// For each tie block, it reports:
/// - the common key value,
/// - the total number of elements in the block,
/// - how many elements belong to the "good" class,
/// - how many belong to the "bad" class.
///
/// This pattern appears repeatedly in metric computations where thresholds
/// are evaluated **only at unique key values**, such as:
/// - empirical CDF updates for the **KS statistic**,
/// - threshold sweeps for **Youden’s J**, **F1**, or **FPR@TPR**,
/// - rank-based metrics when combined with explicit rank bookkeeping.
///
/// By centralizing this logic, we ensure:
/// - consistent handling of ties,
/// - stable numerical behavior when keys are discretized or quantized,
/// - simpler and less error-prone metric implementations.
///
/// Arguments
/// ---------
/// * `pairs`  
///   Mutable slice of `(key, is_good)` tuples, where:
///   - `key` is the scalar value used for ordering (e.g. `cost` or `score`),
///   - `is_good` indicates class membership (`true` = good, `false` = bad).
///
///   The vector is **sorted in-place** by ascending `key`.
///
/// * `on_block`  
///   Callback invoked once per tie block, with the following arguments:
///   - `key` – the common key value of the block,
///   - `block_size` – total number of elements in the block,
///   - `n_good_in_block` – number of good elements in the block,
///   - `n_bad_in_block` – number of bad elements in the block.
///
/// Return
/// ------
/// * `()`  
///   This function returns nothing. All effects are performed via the
///   `on_block` callback.
///
/// Notes
/// -----
/// - The input vector is modified (sorted) as part of the computation.
/// - Equality between keys is tested using `==`; callers should ensure that
///   the keys are suitable for exact comparison (e.g. costs produced by the
///   same computation pipeline).
/// - This function assumes all keys are finite; non-finite values should be
///   filtered upstream.
fn for_each_tie_block(
    pairs: &mut Vec<(f64, bool)>,
    mut on_block: impl FnMut(f64, usize, usize, usize),
) {
    // Sort all pairs by ascending key value so that equal keys form
    // contiguous segments (tie blocks).
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let total_len = pairs.len();
    let mut start_idx = 0usize;

    // Walk through the sorted array and process one tie block at a time.
    while start_idx < total_len {
        let key_value = pairs[start_idx].0;

        // Find the end of the tie block [start_idx, end_idx).
        let mut end_idx = start_idx + 1;
        while end_idx < total_len && pairs[end_idx].0 == key_value {
            end_idx += 1;
        }

        // Count how many elements in the block belong to the "good" class.
        let mut n_good_in_block = 0usize;
        for idx in start_idx..end_idx {
            if pairs[idx].1 {
                n_good_in_block += 1;
            }
        }

        let block_size = end_idx - start_idx;
        let n_bad_in_block = block_size - n_good_in_block;

        // Delegate block-level logic to the caller.
        on_block(key_value, block_size, n_good_in_block, n_bad_in_block);

        // Advance to the next block.
        start_idx = end_idx;
    }
}

/// One edge sample used for metric computations.
///
/// Overview
/// --------
/// `EdgeItem` is a lightweight representation of a scored edge used exclusively
/// for **metric evaluation**. It strips a full [`ScoredEdge`] down to the minimal
/// information required to compute ranking- and threshold-based metrics.
///
/// This structure is typically produced after:
/// - filtering invalid edges (non-finite costs),
/// - attaching a binary ground-truth label,
/// - converting the score into a convenient ranking variable.
///
/// Fields
/// ------
/// * `cost`  
///   Scalar edge cost (**lower is better**).  
///   This is the quantity used for threshold sweeps (`cost ≤ threshold`)
///   in metrics such as KS, Youden’s J, F1, FPR@TPR, etc.
///
/// * `score`  
///   Ranking score (**higher is better**), typically defined as `-cost`.  
///   This field is used for **ranking-based metrics** such as ROC-AUC
///   (Mann–Whitney) and PR-AUC (Average Precision).
///
/// * `is_good`  
///   Ground-truth label:
///   - `true`  → the edge links observations of the same asteroid,
///   - `false` → the edge links different asteroids.
///
/// Notes
/// -----
/// - `EdgeItem` is intentionally small and `Copy` to allow fast aggregation
///   and repeated metric evaluations during optimization.
/// - The distinction between `cost` and `score` avoids ambiguity between
///   *threshold-based* and *ranking-based* metrics.
#[derive(Debug, Clone, Copy)]
pub struct EdgeItem {
    pub cost: f64,
    pub score: f64, // typically -cost
    pub is_good: bool,
}

impl Metrics for [EdgeItem] {
    fn auc_mann_whitney_on_score(&self, n_good: usize, n_bad: usize) -> f64 {
        // Sanity check: the population must contain exactly n_good + n_bad samples.
        debug_assert_eq!(self.len(), n_good + n_bad);

        // Build a working array of (score, is_good) pairs.
        // The score is the ranking variable: higher score = more likely good.
        let mut pairs: Vec<(f64, bool)> = self.iter().map(|it| (it.score, it.is_good)).collect();

        // Sort by score in ascending order so that ranks are assigned from
        // lowest score (rank 1) to highest score (rank N).
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let n = pairs.len();

        // Accumulator for the sum of ranks assigned to good edges.
        // This is the core quantity in the Mann–Whitney U formulation.
        let mut rank_sum_good = 0.0f64;

        // Walk through the sorted array and process one tie block at a time.
        let mut i = 0usize;
        while i < n {
            // Score value shared by the current tie block.
            let score_value = pairs[i].0;

            // Find the end of the tie block [i, j),
            // i.e. all consecutive entries with identical score.
            let mut j = i + 1;
            while j < n && pairs[j].0 == score_value {
                j += 1;
            }

            // Compute the ranks covered by this tie block.
            //
            // Ranks are 1-based:
            // - index i corresponds to rank i + 1,
            // - index j - 1 corresponds to rank j.
            let rank_lo = (i + 1) as f64;
            let rank_hi = j as f64;

            // All elements in the tie block receive the average rank.
            let avg_rank = 0.5 * (rank_lo + rank_hi);

            // Add the average rank once for each good element in the block.
            for k in i..j {
                if pairs[k].1 {
                    rank_sum_good += avg_rank;
                }
            }

            // Move to the next tie block.
            i = j;
        }

        // Convert class counts to floating point.
        let ng = n_good as f64;
        let nb = n_bad as f64;

        // Final Mann–Whitney AUC formula:
        //
        // AUC = (R_good − ng(ng + 1)/2) / (ng · nb)
        //
        // where R_good is the sum of ranks of the good samples.
        (rank_sum_good - ng * (ng + 1.0) * 0.5) / (ng * nb)
    }

    fn average_precision_on_score(&self, n_good: usize, n_bad: usize) -> f64 {
        // Sanity check: the population must contain exactly n_good + n_bad samples.
        debug_assert_eq!(self.len(), n_good + n_bad);

        // If there are no positive (good) samples, recall is undefined.
        // By convention, Average Precision is returned as 0.0.
        if n_good == 0 {
            return 0.0;
        }

        // Build a working array of (score, is_good) pairs.
        // The score is the ranking variable: higher score = more likely good.
        let mut items: Vec<(f64, bool)> = self.iter().map(|it| (it.score, it.is_good)).collect();

        // Sort by score in descending order so that we sweep thresholds
        // from the most confident predictions to the least confident ones.
        items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Counters for true positives and false positives encountered so far
        // when moving the threshold downward.
        let mut tp = 0usize;
        let mut fp = 0usize;

        // Accumulator for the Average Precision integral.
        let mut ap = 0.0f64;

        // Recall value at the previous threshold.
        // Used to compute stepwise increments along the recall axis.
        let mut prev_recall = 0.0f64;

        // Walk through the sorted array and process one tie block at a time.
        // This ensures that all edges with identical scores are added
        // simultaneously, producing a stable PR curve.
        let n = items.len();
        let mut i = 0usize;
        while i < n {
            // Score value shared by the current tie block.
            let score_value = items[i].0;

            // Find the end of the tie block [i, j).
            let mut j = i + 1;
            while j < n && items[j].0 == score_value {
                j += 1;
            }

            // Count how many good and bad edges are in this tie block.
            let mut good_in_block = 0usize;
            let mut bad_in_block = 0usize;
            for k in i..j {
                if items[k].1 {
                    good_in_block += 1;
                } else {
                    bad_in_block += 1;
                }
            }

            // When the threshold crosses this score value, all edges in the
            // block become predicted positive at once.
            tp += good_in_block;
            fp += bad_in_block;

            // Compute recall after including this block.
            let recall = (tp as f64) / (n_good as f64);

            // Compute precision after including this block.
            let precision = {
                let denom = (tp + fp) as f64;
                if denom > 0.0 {
                    (tp as f64) / denom
                } else {
                    0.0
                }
            };

            // Average Precision is the stepwise integral of precision
            // with respect to recall.
            //
            // Only positive recall increments contribute to the area.
            let delta_recall = recall - prev_recall;
            if delta_recall > 0.0 {
                ap += precision * delta_recall;
                prev_recall = recall;
            }

            // Advance to the next tie block.
            i = j;
        }

        ap
    }

    fn ks_stat_on_cost(&self, n_good: usize, n_bad: usize) -> f64 {
        // Sanity check: the population must contain exactly n_good + n_bad samples.
        debug_assert_eq!(self.len(), n_good + n_bad);

        // Convert class counts to floating point for CDF computations.
        let n_good_f = n_good as f64;
        let n_bad_f = n_bad as f64;

        // Cumulative counts of samples seen so far when sweeping cost thresholds
        // from low to high (i.e., when increasing the acceptance threshold).
        let mut cum_good = 0usize;
        let mut cum_bad = 0usize;

        // Current best KS value: maximum absolute separation between empirical CDFs.
        let mut ks = 0.0f64;

        // Build a working array of (cost, is_good) pairs.
        // We will sweep thresholds on cost (lower cost = more likely good).
        let mut pairs: Vec<(f64, bool)> = self.iter().map(|it| (it.cost, it.is_good)).collect();

        // Iterate through unique cost values (tie blocks) in ascending order.
        // At each cost value c:
        // - all edges with cost == c become included in the empirical CDFs at once,
        // - we update F_good(c) and F_bad(c),
        // - we update ks = max |F_good(c) - F_bad(c)|.
        for_each_tie_block(
            &mut pairs,
            |_cost_value, _block_size, good_in_block, bad_in_block| {
                // Update cumulative counts with the content of this tie block.
                cum_good += good_in_block;
                cum_bad += bad_in_block;

                // Empirical CDF values at the current threshold:
                // F_good(c) = (# good with cost <= c) / n_good
                // F_bad(c)  = (# bad  with cost <= c) / n_bad
                let cdf_good = (cum_good as f64) / n_good_f;
                let cdf_bad = (cum_bad as f64) / n_bad_f;

                // KS statistic is the maximum absolute vertical difference between the two CDFs.
                ks = ks.max((cdf_good - cdf_bad).abs());
            },
        );

        ks
    }

    fn best_youden_j_on_cost(&self, n_good: usize, n_bad: usize) -> BestYouden {
        // Sanity check: the population must contain exactly n_good + n_bad samples.
        debug_assert_eq!(self.len(), n_good + n_bad);

        // Convert class counts to floating point for rate computations.
        let n_good_f = n_good as f64;
        let n_bad_f = n_bad as f64;

        // Cumulative counts of true positives (TP) and false positives (FP)
        // as we sweep the cost threshold from low to high.
        let mut tp = 0usize;
        let mut fp = 0usize;

        // Build a working array of (cost, is_good) pairs.
        // Lower cost means more likely to be a correct link.
        let mut pairs: Vec<(f64, bool)> = self.iter().map(|it| (it.cost, it.is_good)).collect();

        // Sort by cost in ascending order so that increasing the threshold
        // gradually accepts more edges.
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Default threshold used as a fallback (e.g. empty or degenerate cases).
        let default_threshold = pairs.first().map(|x| x.0).unwrap_or(0.0);

        // Initialize the best Youden operating point.
        // We start with a very low J value so that any valid threshold improves it.
        let mut best = BestYouden {
            best_j: -1.0,
            threshold_cost: default_threshold,
            tpr: 0.0,
            fpr: 0.0,
            precision: 0.0,
        };

        // Iterate over unique cost values (tie blocks).
        // At each step, all edges with cost == cost_i are included simultaneously.
        for_each_tie_block(
            &mut pairs,
            |cost_i, _block_size, good_in_block, bad_in_block| {
                // Update cumulative TP / FP counts when the threshold reaches cost_i.
                tp += good_in_block;
                fp += bad_in_block;

                // Compute true positive rate (recall) at this threshold:
                // TPR = TP / n_good
                let tpr = (tp as f64) / n_good_f;

                // Compute false positive rate at this threshold:
                // FPR = FP / n_bad
                let fpr = (fp as f64) / n_bad_f;

                // Youden’s J statistic at this threshold:
                // J = TPR - FPR
                let j_stat = tpr - fpr;

                // Precision is reported as an auxiliary diagnostic:
                // precision = TP / (TP + FP)
                let denom = (tp + fp) as f64;
                let precision = if denom > 0.0 {
                    (tp as f64) / denom
                } else {
                    0.0
                };

                // Keep this threshold if it improves the Youden statistic.
                if j_stat > best.best_j {
                    best = BestYouden {
                        best_j: j_stat,
                        threshold_cost: cost_i,
                        tpr,
                        fpr,
                        precision,
                    };
                }
            },
        );

        best
    }

    fn best_f1_on_cost(&self, n_good: usize, n_bad: usize) -> BestF1 {
        // Sanity check: the population must contain exactly n_good + n_bad samples.
        debug_assert_eq!(self.len(), n_good + n_bad);

        // Cumulative counts of true positives (TP) and false positives (FP)
        // as we sweep the cost threshold from low to high.
        let mut tp = 0usize;
        let mut fp = 0usize;

        // Build a working array of (cost, is_good) pairs.
        // Lower cost means more likely to be a correct link.
        let mut pairs: Vec<(f64, bool)> = self.iter().map(|it| (it.cost, it.is_good)).collect();

        // Sort by cost in ascending order so that increasing the threshold
        // gradually accepts more edges.
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Default threshold used as a fallback (e.g. empty or degenerate cases).
        let default_threshold = pairs.first().map(|x| x.0).unwrap_or(0.0);

        // Track the best operating point found so far.
        // We initialize at a threshold where no edge is accepted (TP=0, FP=0).
        let mut best_op = OperatingPoint::from_counts(default_threshold, 0, 0, n_good, n_bad);

        // Track the best F1 value found so far. Start below 0 so any valid F1 wins.
        let mut best_f1 = -1.0f64;

        // Iterate over unique cost values (tie blocks).
        // At each step, all edges with cost == cost_i are included simultaneously.
        for_each_tie_block(
            &mut pairs,
            |cost_i, _block_size, good_in_block, bad_in_block| {
                // When the threshold reaches cost_i, all edges in this block become
                // predicted positives. Update cumulative TP/FP accordingly.
                tp += good_in_block;
                fp += bad_in_block;

                // Build the operating point at this threshold:
                // - threshold_cost = cost_i
                // - counts: TP/FP and derived TN/FN
                // - derived rates: precision/recall/TPR/FPR
                let op = OperatingPoint::from_counts(cost_i, tp, fp, n_good, n_bad);

                // Compute F1 at this threshold:
                // F1 = 2 * precision * recall / (precision + recall)
                let f1 = op.f1();

                // Keep this threshold if it improves the best F1 score.
                if f1 > best_f1 {
                    best_f1 = f1;
                    best_op = op;
                }
            },
        );

        // Clamp to 0 in case all candidates were degenerate (should be rare if both classes exist).
        BestF1 {
            best_f1: best_f1.max(0.0),
            op: best_op,
        }
    }

    fn fpr_at_tpr_on_cost(&self, n_good: usize, n_bad: usize, target_tpr: f64) -> f64 {
        // Sanity check: the population must contain exactly n_good + n_bad samples.
        debug_assert_eq!(self.len(), n_good + n_bad);

        // Clamp the target recall (TPR) to a valid probability range.
        let target_tpr = target_tpr.clamp(0.0, 1.0);

        // If one class is missing, the metric is not meaningful.
        // Return 0.0 by convention (caller should typically avoid this situation).
        if n_good == 0 || n_bad == 0 {
            return 0.0;
        }

        // Cumulative counts as we increase the cost threshold (accept more edges).
        let mut tp = 0usize; // good edges accepted so far
        let mut fp = 0usize; // bad edges accepted so far

        // Best (minimal) FPR observed among thresholds achieving TPR >= target_tpr.
        let mut best_fpr = 1.0f64;

        // Build a working array of (cost, is_good) pairs.
        // Lower cost means more likely to be a correct link.
        let mut pairs: Vec<(f64, bool)> = self.iter().map(|it| (it.cost, it.is_good)).collect();

        // Iterate thresholds at unique cost values (tie blocks) in ascending order.
        // At each cost value, all edges with that exact cost are accepted at once.
        for_each_tie_block(
            &mut pairs,
            |_cost_value, _block_size, good_in_block, bad_in_block| {
                // Update cumulative acceptance counts.
                tp += good_in_block;
                fp += bad_in_block;

                // True positive rate (recall) at the current threshold:
                // TPR = TP / n_good
                let tpr = (tp as f64) / (n_good as f64);

                // If we meet the recall constraint, compute the corresponding FPR
                // and keep the minimal one across all satisfying thresholds.
                //
                // The small epsilon avoids missing the boundary due to floating-point noise.
                if tpr + 1e-15 >= target_tpr {
                    // False positive rate at the current threshold:
                    // FPR = FP / n_bad
                    let fpr = (fp as f64) / (n_bad as f64);

                    if fpr < best_fpr {
                        best_fpr = fpr;
                    }
                }
            },
        );

        best_fpr
    }

    fn tpr_at_fpr_on_cost(&self, n_good: usize, n_bad: usize, target_fpr: f64) -> f64 {
        // Sanity check: the population must contain exactly n_good + n_bad samples.
        debug_assert_eq!(self.len(), n_good + n_bad);

        // Clamp the target false positive rate to a valid probability range.
        let target_fpr = target_fpr.clamp(0.0, 1.0);

        // If one class is missing, the metric is not meaningful.
        // Return 0.0 by convention (caller should typically avoid this situation).
        if n_good == 0 || n_bad == 0 {
            return 0.0;
        }

        // Cumulative counts as we increase the cost threshold (accept more edges).
        let mut tp = 0usize; // good edges accepted so far
        let mut fp = 0usize; // bad edges accepted so far

        // Best (maximal) TPR observed among thresholds satisfying FPR <= target_fpr.
        let mut best_tpr = 0.0f64;

        // Build a working array of (cost, is_good) pairs.
        // Lower cost means more likely to be a correct link.
        let mut pairs: Vec<(f64, bool)> = self.iter().map(|it| (it.cost, it.is_good)).collect();

        // Iterate thresholds at unique cost values (tie blocks) in ascending order.
        // At each cost value, all edges with that exact cost are accepted at once.
        for_each_tie_block(
            &mut pairs,
            |_cost_value, _block_size, good_in_block, bad_in_block| {
                // Update cumulative acceptance counts.
                tp += good_in_block;
                fp += bad_in_block;

                // False positive rate at the current threshold:
                // FPR = FP / n_bad
                let fpr = (fp as f64) / (n_bad as f64);

                // If we satisfy the contamination constraint, compute the corresponding TPR
                // and keep the maximal one across all satisfying thresholds.
                //
                // The small epsilon avoids rejecting a threshold at the boundary due to
                // floating-point noise.
                if fpr <= target_fpr + 1e-15 {
                    // True positive rate (recall) at the current threshold:
                    // TPR = TP / n_good
                    let tpr = (tp as f64) / (n_good as f64);

                    if tpr > best_tpr {
                        best_tpr = tpr;
                    }
                }
            },
        );

        best_tpr
    }
}
