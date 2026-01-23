use std::fmt;

/// Operating characteristics of a binary decision threshold on `cost`.
///
/// Overview
/// --------
/// `OperatingPoint` summarizes the **confusion matrix** and derived rates
/// obtained when applying a **threshold on edge cost**:
///
/// ```text
/// predicted_good = (cost ≤ threshold_cost)
/// ```
///
/// It represents a single point on the ROC / PR curves and is used to:
/// - report optimal thresholds (Youden, F1),
/// - evaluate operating regimes under recall or contamination constraints,
/// - compute secondary metrics such as F1 or MCC.
///
/// Fields
/// ------
/// * `threshold_cost`  
///   Cost threshold defining the decision rule (`cost ≤ threshold_cost`).
///
/// * `tp`  
///   Number of **true positives**: good edges correctly accepted.
///
/// * `fp`  
///   Number of **false positives**: bad edges incorrectly accepted.
///
/// * `tn`  
///   Number of **true negatives**: bad edges correctly rejected.
///
/// * `fn_`  
///   Number of **false negatives**: good edges incorrectly rejected.
///
/// * `tpr`  
///   **True positive rate** (recall):  
///   `TPR = tp / n_good`.
///
/// * `fpr`  
///   **False positive rate**:  
///   `FPR = fp / n_bad`.
///
/// * `precision`  
///   Fraction of accepted edges that are correct:  
///   `precision = tp / (tp + fp)`.
///
/// * `recall`  
///   Alias for `tpr`. Included for clarity in precision–recall contexts.
///
/// Notes
/// -----
/// - All rates are stored explicitly to avoid recomputation and to make
///   diagnostic logging easier.
/// - `recall` is duplicated from `tpr` to emphasize its use in PR-based metrics.
#[derive(Debug, Clone, Copy, Default)]
pub struct OperatingPoint {
    pub threshold_cost: f64,
    pub tp: usize,
    pub fp: usize,
    pub tn: usize,
    pub fn_: usize,
    pub tpr: f64,
    pub fpr: f64,
    pub precision: f64,
    pub recall: f64, // == tpr
}

impl OperatingPoint {
    /// Construct an [`OperatingPoint`] from cumulative counts at a given threshold.
    ///
    /// Overview
    /// --------
    /// This constructor builds a fully populated [`OperatingPoint`] from:
    /// - the decision threshold on `cost`,
    /// - cumulative counts of accepted good and bad edges,
    /// - the total number of good and bad edges in the population.
    ///
    /// It is typically called while **sweeping thresholds in ascending cost**
    /// during metric computations (Youden, F1, FPR@TPR, etc.).
    ///
    /// Arguments
    /// ---------
    /// * `threshold_cost`  
    ///   Cost threshold defining the decision rule (`cost ≤ threshold_cost`).
    ///
    /// * `tp`  
    ///   Number of true positives at this threshold.
    ///
    /// * `fp`  
    ///   Number of false positives at this threshold.
    ///
    /// * `n_good`  
    ///   Total number of good edges in the population.
    ///
    /// * `n_bad`  
    ///   Total number of bad edges in the population.
    ///
    /// Notes
    /// -----
    /// - `tn` and `fn_` are derived from `n_bad − fp` and `n_good − tp`,
    ///   respectively, using saturating subtraction for safety.
    /// - Rates (`TPR`, `FPR`, `precision`) are defined as zero when their
    ///   denominators vanish.
    pub(in crate::scoring::optimization_metrics) fn from_counts(
        threshold_cost: f64,
        tp: usize,
        fp: usize,
        n_good: usize,
        n_bad: usize,
    ) -> Self {
        let fn_ = n_good.saturating_sub(tp);
        let tn = n_bad.saturating_sub(fp);

        let ng = n_good as f64;
        let nb = n_bad as f64;

        let tpr = if n_good > 0 { (tp as f64) / ng } else { 0.0 };
        let fpr = if n_bad > 0 { (fp as f64) / nb } else { 0.0 };

        let denom = (tp + fp) as f64;
        let precision = if denom > 0.0 {
            (tp as f64) / denom
        } else {
            0.0
        };

        Self {
            threshold_cost,
            tp,
            fp,
            tn,
            fn_,
            tpr,
            fpr,
            precision,
            recall: tpr,
        }
    }

    /// Compute the **F1 score** at this operating point.
    ///
    /// Definition
    /// ----------
    /// ```text
    /// F1 = 2 · precision · recall / (precision + recall)
    /// ```
    ///
    /// Interpretation
    /// --------------
    /// - F1 emphasizes a balance between contamination (precision) and
    ///   completeness (recall),
    /// - It is appropriate when both are considered equally important,
    /// - It is **not optimized for ultra-low false-positive regimes**.
    ///
    /// Returns
    /// -------
    /// * A value in `[0, 1]`, or `0.0` if undefined.
    pub(in crate::scoring::optimization_metrics) fn f1(&self) -> f64 {
        let denom = self.precision + self.recall;
        if denom > 0.0 {
            2.0 * self.precision * self.recall / denom
        } else {
            0.0
        }
    }

    /// Compute the **Matthews Correlation Coefficient (MCC)** at this operating point.
    ///
    /// Definition
    /// ----------
    /// ```text
    /// MCC = (TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    /// ```
    ///
    /// Interpretation
    /// --------------
    /// - MCC is a balanced scalar metric that accounts for all four entries
    ///   of the confusion matrix,
    /// - It remains informative under strong class imbalance,
    /// - `MCC = 1` indicates perfect classification,
    /// - `MCC = 0` indicates no better than random,
    /// - `MCC < 0` indicates systematic misclassification.
    ///
    /// Returns
    /// -------
    /// * A value in `[-1, 1]`, or `0.0` if undefined.
    pub(in crate::scoring::optimization_metrics) fn mcc(&self) -> f64 {
        let tp = self.tp as f64;
        let tn = self.tn as f64;
        let fp = self.fp as f64;
        let fn_ = self.fn_ as f64;

        let num = tp * tn - fp * fn_;
        let denom = (tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_);
        if denom > 0.0 { num / denom.sqrt() } else { 0.0 }
    }
}

/// Summary of the operating point that maximizes **Youden’s J statistic**
/// for thresholds on `cost`.
///
/// Overview
/// --------
/// `BestYouden` describes the **single cost threshold** (lower is better)
/// that maximizes **Youden’s J statistic**, defined as:
///
/// ```text
/// J = TPR − FPR
/// ```
///
/// This operating point corresponds to the maximum vertical distance between
/// the ROC curve and the diagonal, and is often used as a **baseline global
/// threshold** for binary classification.
///
/// In the context of inter-night asteroid linking, this structure captures:
/// - where the best global separation between true and false edges occurs,
/// - how much recall and contamination are associated with that threshold.
///
/// Fields
/// ------
/// * `best_j`  
///   Maximum value of Youden’s J statistic achieved across all thresholds.
///
/// * `threshold_cost`  
///   Cost threshold at which `best_j` is attained.  
///   The decision rule is:
///
///   ```text
///   predicted_good = (cost ≤ threshold_cost)
///   ```
///
/// * `tpr`  
///   True positive rate (recall) at the optimal threshold:
///   fraction of true edges that are accepted.
///
/// * `fpr`  
///   False positive rate at the optimal threshold:
///   fraction of false edges that are incorrectly accepted.
///
/// * `precision`  
///   Precision at the optimal threshold:
///   fraction of accepted edges that are correct.
///
/// Notes
/// -----
/// - Youden’s J treats recall and false positive rate **symmetrically**.
/// - The resulting threshold is often a reasonable starting point, but is
///   **not tailored for low-contamination regimes** typically required by
///   sparse graph construction.
///
/// See also
/// --------
/// * [`BestF1`] – threshold optimized for precision–recall balance.
/// * [`fpr_at_tpr_on_cost`] – thresholds under fixed recall constraints.
#[derive(Debug, Clone, Copy, Default)]
pub struct BestYouden {
    pub best_j: f64,
    pub threshold_cost: f64,
    pub tpr: f64,
    pub fpr: f64,
    pub precision: f64,
}

/// Summary of the operating point that maximizes the **F1 score**
/// for thresholds on `cost`.
///
/// Overview
/// --------
/// `BestF1` describes the **cost threshold** (lower is better) that maximizes
/// the **F1 score**, defined as the harmonic mean of precision and recall.
///
/// This operating point emphasizes a **balance between contamination and
/// completeness** in the selected edge set.
///
/// Fields
/// ------
/// * `best_f1`  
///   Maximum F1 score achieved across all thresholds.
///
/// * `op`  
///   Full [`OperatingPoint`] corresponding to the F1-optimal threshold,
///   including:
///   - confusion matrix counts (TP, FP, TN, FN),
///   - derived rates (precision, recall/TPR, FPR),
///   - the selected `threshold_cost`.
///
/// Interpretation
/// --------------
/// - High `best_f1` indicates that a threshold exists where both precision and
///   recall are reasonably good,
/// - Maximizing F1 often tolerates **moderate false positive rates** if that
///   improves recall,
/// - This metric is useful for diagnostic and comparison purposes, but is
///   **not optimal** when the operational regime requires extremely low
///   contamination.
///
/// See also
/// --------
/// * [`BestYouden`] – symmetric ROC-based threshold.
/// * [`tpr_at_fpr_on_cost`] – thresholds under strict contamination limits.
#[derive(Debug, Clone, Copy, Default)]
pub struct BestF1 {
    pub best_f1: f64,
    pub op: OperatingPoint,
}

/// Aggregate summary of metrics used for **score and gate optimization**.
///
/// Overview
/// --------
/// `MetricsSummary` bundles together a comprehensive set of evaluation metrics
/// computed from a labeled population of edges. It is designed as a
/// **single snapshot** describing how well a scoring configuration:
/// - separates true and false edges,
/// - ranks candidates,
/// - behaves under practical operating constraints.
///
/// This structure is typically produced by [`Metrics::summary`] and is intended
/// for:
/// - offline hyper-parameter optimization,
/// - comparative benchmarking of score configurations,
/// - logging, plotting, and reporting.
///
/// Fields
/// ------
/// ### Global / threshold-free metrics
/// * `auc`  
///   ROC-AUC computed via the Mann–Whitney formulation on `score`.
///   Measures global ranking separability, independent of thresholds.
///
/// * `auc_pr`  
///   PR-AUC (Average Precision) on `score`.
///   More informative than ROC-AUC when true edges are rare.
///
/// * `ks`  
///   Kolmogorov–Smirnov statistic between the cost distributions of good and
///   bad edges. Measures maximum global separation of empirical CDFs.
///
/// ### Threshold-based reference points
/// * `best_youden`  
///   Operating point that maximizes Youden’s J statistic (`TPR − FPR`).
///
/// * `best_f1`  
///   Operating point that maximizes the F1 score.
///
/// * `mcc_at_best_f1`  
///   Matthews correlation coefficient evaluated at the F1-optimal threshold.
///   Provides a balanced scalar summary even under class imbalance.
///
/// ### Low-contamination / graph-oriented metrics
/// * `fpr_at_tpr_90`, `fpr_at_tpr_95`, `fpr_at_tpr_99`  
///   Minimal achievable false positive rate when enforcing recall levels of
///   90%, 95%, and 99%, respectively.
///
/// * `tpr_at_fpr_1e3`  
///   Maximum achievable recall when constraining the false positive rate
///   to at most `10⁻³`.
///
/// Interpretation
/// --------------
/// - No single metric fully characterizes score quality.
/// - This summary is meant to be read **as a whole**, combining:
///   - global separability,
///   - ranking quality,
///   - threshold behavior,
///   - suitability for sparse graph construction.
///
/// Notes
/// -----
/// - Metrics are computed under the assumption that both classes are present.
/// - Values near pathological limits (e.g. `AUC ≈ 0.5`, `KS ≈ 0`) indicate
///   weak or unusable scoring configurations.
///
/// See also
/// --------
/// * [`Metrics::summary`] – construction of this aggregate.
/// * [`graph_ranking_metrics`] – graph-aware Top-K and ranking metrics.
#[derive(Debug, Clone, Copy)]
pub struct MetricsSummary {
    pub auc: f64,
    pub auc_pr: f64, // Average Precision on score (higher is better)
    pub ks: f64,
    pub best_youden: BestYouden,
    pub best_f1: BestF1,
    pub mcc_at_best_f1: f64,
    // Handy for "graph-like" constraints:
    pub fpr_at_tpr_90: f64,
    pub fpr_at_tpr_95: f64,
    pub fpr_at_tpr_99: f64,
    pub tpr_at_fpr_1e3: f64,
}

impl fmt::Display for MetricsSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Metrics summary")?;
        writeln!(f, "---------------")?;

        writeln!(f, "Global metrics")?;
        writeln!(f, "  AUC (ROC)          : {:.6}", self.auc)?;
        writeln!(f, "  AUC (PR)           : {:.6}", self.auc_pr)?;
        writeln!(f, "  KS                 : {:.6}", self.ks)?;

        writeln!(f)?;
        writeln!(f, "Best operating points")?;

        writeln!(
            f,
            "  Youden J           : J={:.6} @ cost={:.6}  (TPR={:.4}, FPR={:.4}, P={:.4})",
            self.best_youden.best_j,
            self.best_youden.threshold_cost,
            self.best_youden.tpr,
            self.best_youden.fpr,
            self.best_youden.precision,
        )?;

        writeln!(
            f,
            "  F1 max             : F1={:.6} @ cost={:.6}",
            self.best_f1.best_f1, self.best_f1.op.threshold_cost,
        )?;
        writeln!(f, "    └─ MCC @ F1      : {:.6}", self.mcc_at_best_f1)?;

        writeln!(f)?;
        writeln!(f, "Low-contamination regime")?;
        writeln!(f, "  FPR @ TPR=0.90     : {:.2e}", self.fpr_at_tpr_90)?;
        writeln!(f, "  FPR @ TPR=0.95     : {:.2e}", self.fpr_at_tpr_95)?;
        writeln!(f, "  FPR @ TPR=0.99     : {:.2e}", self.fpr_at_tpr_99)?;
        writeln!(f, "  TPR @ FPR=1e-3     : {:.4}", self.tpr_at_fpr_1e3)?;

        Ok(())
    }
}

/// Metrics computed from a set of [`EdgeItem`].
pub trait Metrics {
    /// Compute the ROC AUC using the Mann–Whitney U formulation on `score`.
    ///
    /// Overview
    /// --------
    /// This function computes the **Area Under the ROC Curve (AUC)** using the
    /// **Mann–Whitney rank statistic**, applied to the edge `score`
    /// (**higher score = more likely to be a correct link**).
    ///
    /// In the context of inter-night asteroid linking, this metric evaluates how
    /// well the scoring function **ranks true edges ahead of false ones**, independently
    /// of any decision threshold. It measures the intrinsic separability between
    /// "same-asteroid" and "different-asteroid" edges.
    ///
    /// Mathematical definition
    /// -----------------------
    /// Consider a population of labeled edges split into two classes:
    ///
    /// - **Good edges**: edges linking observations of the same asteroid
    ///   (`is_good = true`), with total count `n_good`,
    /// - **Bad edges**: edges linking different asteroids
    ///   (`is_good = false`), with total count `n_bad`.
    ///
    /// All edges are sorted by increasing `score`, and **1-based ranks** are assigned:
    ///
    /// - rank 1 corresponds to the lowest score,
    /// - rank `N = n_good + n_bad` corresponds to the highest score,
    /// - ties in `score` receive the **average rank** of the tie block.
    ///
    /// Let:
    ///
    /// - `R_good` be the **sum of the ranks** of all good edges.
    ///
    /// The AUC is then computed as:
    ///
    /// ```text
    /// AUC = (R_good − n_good · (n_good + 1) / 2) / (n_good · n_bad)
    /// ```
    ///
    /// This expression is algebraically equivalent to the **Mann–Whitney U statistic**
    /// normalized to the interval `[0, 1]`.
    ///
    /// Probabilistic interpretation
    /// ----------------------------
    /// The AUC can be interpreted as:
    ///
    /// > The probability that a randomly chosen good edge has a higher score than
    /// > a randomly chosen bad edge (with ties counted as 0.5).
    ///
    /// Interpretation
    /// --------------
    /// - `AUC = 0.5` : no discriminative power (random ranking),
    /// - `AUC = 1.0` : perfect ranking (all good edges ranked above all bad ones),
    /// - `AUC < 0.5` : pathological case (ranking is worse than random).
    ///
    /// In practice:
    /// - **Higher AUC** indicates better global separation between true and false links,
    /// - AUC is **threshold-independent**, making it well suited for offline tuning
    ///   of scoring weights and noise models,
    /// - AUC does **not** directly capture performance in the very low false-positive
    ///   regime required for graph linking (use FPR@TPR, Hit@K, or MRR for that).
    ///
    /// Tie handling
    /// ------------
    /// Edges with identical `score` values are processed as **tie blocks** and are
    /// assigned the **average rank** of the block. This ensures an unbiased AUC
    /// estimate even when scores are discretized or weakly quantized.
    ///
    /// Assumptions
    /// -----------
    /// - `self.len() == n_good + n_bad`,
    /// - both classes are present (`n_good > 0` and `n_bad > 0`),
    /// - all scores are finite (non-finite values should be filtered upstream).
    ///
    /// Returns
    /// -------
    /// * A value in `[0, 1]` representing the ROC AUC computed via the
    ///   Mann–Whitney formulation.
    ///
    /// See also
    /// --------
    /// * [`average_precision_on_score`] – PR-AUC, more informative for rare positives.
    /// * [`ks_stat_on_cost`] – Maximum separation between empirical CDFs.
    /// * [`best_youden_j_on_cost`] – Threshold-based operating point.
    /// * [`graph_ranking_metrics`] – Graph-aware metrics (Hit@K, MRR).
    fn auc_mann_whitney_on_score(&self, n_good: usize, n_bad: usize) -> f64;

    /// Compute the PR-AUC as **Average Precision (AP)** on `score`.
    ///
    /// Overview
    /// --------
    /// This function computes the **area under the Precision–Recall (PR) curve**
    /// using the **Average Precision (AP)** formulation, evaluated on the edge
    /// `score` (**higher score = more likely to be a correct link**).
    ///
    /// In inter-night asteroid linking, correct edges are often relatively rare
    /// compared to incorrect ones. In such imbalanced settings, **PR-based metrics**
    /// are typically more informative than ROC-AUC because they directly capture
    /// the trade-off between:
    /// - **precision** (contamination of selected edges), and
    /// - **recall** (fraction of true edges recovered).
    ///
    /// Definitions
    /// -----------
    /// Consider a population of labeled edges split into two classes:
    ///
    /// - **Good edges**: edges linking observations of the same asteroid
    ///   (`is_good = true`), with total count `n_good`,
    /// - **Bad edges**: edges linking different asteroids
    ///   (`is_good = false`), with total count `n_bad`.
    ///
    /// We conceptually sweep a threshold on `score` from high to low, and at each
    /// threshold we classify as "predicted good" all edges with `score` greater than
    /// or equal to the threshold.
    ///
    /// For a given threshold, define:
    /// - `TP` (true positives): number of predicted-good edges that are actually good,
    /// - `FP` (false positives): number of predicted-good edges that are actually bad,
    ///
    /// and the standard PR quantities:
    ///
    /// ```text
    /// precision = TP / (TP + FP)
    /// recall    = TP / n_good
    /// ```
    ///
    /// Average Precision definition
    /// ----------------------------
    /// The PR curve is a piecewise-constant function of recall when thresholds are
    /// moved across the sorted list of scores. **Average Precision (AP)** is the
    /// stepwise integral of precision over recall:
    ///
    /// ```text
    /// AP = Σ_k precision_k · (recall_k − recall_{k−1})
    /// ```
    ///
    /// where each step `k` corresponds to advancing the threshold past one or more
    /// items (including tied-score blocks).
    ///
    /// Interpretation
    /// --------------
    /// - `AP = 1.0` : perfect ranking (all good edges appear before any bad edge),
    /// - `AP` near the base rate `n_good / (n_good + n_bad)` : weak ranking
    ///   (close to random ordering),
    /// - Higher AP means that selecting the **top-scoring edges** yields **high
    ///   precision at high recall**, which is desirable before Top-K pruning and
    ///   graph solvers.
    ///
    /// Compared to ROC-AUC, AP is more sensitive to errors among the top-ranked
    /// edges, which often matches the operational regime of linking pipelines.
    ///
    /// Tie handling
    /// ------------
    /// Scores are processed in **tie blocks**: all edges with identical `score` are
    /// added to `TP`/`FP` together before computing the next PR point. This produces
    /// a stable, well-defined PR curve when scores are discretized or quantized.
    ///
    /// Assumptions
    /// -----------
    /// - `self.len() == n_good + n_bad`,
    /// - `n_good > 0` for a meaningful recall axis (otherwise AP is defined as 0),
    /// - all scores are finite (non-finite values should be filtered upstream).
    ///
    /// Returns
    /// -------
    /// * A value in `[0, 1]` representing the PR-AUC computed as Average Precision
    ///   on `score`.
    ///
    /// See also
    /// --------
    /// * [`auc_mann_whitney_on_score`] – ROC-AUC, threshold-independent but often less
    ///   informative under severe class imbalance.
    /// * [`fpr_at_tpr_on_cost`] / [`tpr_at_fpr_on_cost`] – operating points targeting
    ///   the low-FPR regime.
    /// * [`graph_ranking_metrics`] – graph-aware ranking metrics (Hit@K, MRR).
    fn average_precision_on_score(&self, n_good: usize, n_bad: usize) -> f64;

    /// Compute the Kolmogorov–Smirnov (KS) statistic between GOOD and BAD edges
    /// using the empirical CDFs of `cost`.
    ///
    /// Overview
    /// --------
    /// This function computes the **two-sample Kolmogorov–Smirnov (KS) statistic**
    /// between the distributions of `cost` for:
    /// - **Good edges** (`is_good = true`), and
    /// - **Bad edges** (`is_good = false`).
    ///
    /// The KS statistic measures the **maximum vertical separation** between the
    /// empirical cumulative distribution functions (CDFs) of the two classes.
    /// In the context of inter-night asteroid linking, it quantifies how well the
    /// scalar cost separates true links from false associations **across the entire
    /// cost range**, without committing to a specific threshold.
    ///
    /// Definitions
    /// -----------
    /// Consider a population of labeled edges split into two classes:
    ///
    /// - **Good edges**: edges linking observations of the same asteroid,
    ///   with total count `n_good`,
    /// - **Bad edges**: edges linking different asteroids,
    ///   with total count `n_bad`.
    ///
    /// For any cost value `c`, define the empirical CDFs:
    ///
    /// ```text
    /// F_good(c) = (# of good edges with cost ≤ c) / n_good
    /// F_bad(c)  = (# of bad edges  with cost ≤ c) / n_bad
    /// ```
    ///
    /// The KS statistic is then:
    ///
    /// ```text
    /// KS = max_c | F_good(c) − F_bad(c) |
    /// ```
    ///
    /// Interpretation
    /// --------------
    /// - `KS = 0` : the two cost distributions are identical,
    /// - `KS = 1` : the distributions are completely separated,
    /// - Larger KS values indicate stronger separation between good and bad edges.
    ///
    /// In practice:
    /// - A **large KS** means there exists a cost region where good edges accumulate
    ///   much faster than bad ones (or vice versa),
    /// - The cost value at which the maximum separation occurs is often close to a
    ///   **useful operating threshold** for gating or pruning,
    /// - KS is sensitive to **global distribution differences**, not only to the
    ///   extreme tails.
    ///
    /// Tie handling
    /// ------------
    /// Costs are processed in **tie blocks**: all edges with identical `cost` values
    /// are incorporated together when updating the empirical CDFs. This ensures:
    /// - a well-defined KS statistic when costs are discretized or quantized,
    /// - numerical stability when many edges share the same cost.
    ///
    /// Assumptions
    /// -----------
    /// - `self.len() == n_good + n_bad`,
    /// - both classes are present (`n_good > 0` and `n_bad > 0`),
    /// - all costs are finite (non-finite values should be filtered upstream).
    ///
    /// Returns
    /// -------
    /// * A value in `[0, 1]` representing the Kolmogorov–Smirnov statistic computed
    ///   from the empirical CDFs of `cost`.
    ///
    /// See also
    /// --------
    /// * [`best_youden_j_on_cost`] – threshold selection based on KS-like separation.
    /// * [`auc_mann_whitney_on_score`] – rank-based global separability metric.
    /// * [`average_precision_on_score`] – PR-based metric focused on top-ranked edges.
    /// * [`fpr_at_tpr_on_cost`] – low-contamination operating points.
    fn ks_stat_on_cost(&self, n_good: usize, n_bad: usize) -> f64;

    /// Compute the best **Youden’s J statistic** over thresholds on `cost`.
    ///
    /// Overview
    /// --------
    /// This function finds the **operating threshold on `cost`** (lower is better)
    /// that maximizes **Youden’s J statistic**, defined as:
    ///
    /// ```text
    /// J = TPR − FPR
    /// ```
    ///
    /// where:
    /// - `TPR` is the true positive rate (recall),
    /// - `FPR` is the false positive rate.
    ///
    /// In the context of inter-night asteroid linking, this metric identifies the
    /// cost threshold that best separates **true links** from **false associations**
    /// by maximizing the vertical distance between the ROC curve and the diagonal.
    ///
    /// Definitions
    /// -----------
    /// Consider a population of labeled edges split into two classes:
    ///
    /// - **Good edges**: edges linking observations of the same asteroid
    ///   (`is_good = true`), with total count `n_good`,
    /// - **Bad edges**: edges linking different asteroids
    ///   (`is_good = false`), with total count `n_bad`.
    ///
    /// For a given threshold value `c` on `cost`, edges are classified as:
    ///
    /// ```text
    /// predicted_good = (cost ≤ c)
    /// ```
    ///
    /// From this classification, define:
    ///
    /// - `TP` (true positives): number of good edges with `cost ≤ c`,
    /// - `FP` (false positives): number of bad edges with `cost ≤ c`,
    ///
    /// and the corresponding rates:
    ///
    /// ```text
    /// TPR = TP / n_good
    /// FPR = FP / n_bad
    /// ```
    ///
    /// The Youden statistic at threshold `c` is:
    ///
    /// ```text
    /// J(c) = TPR(c) − FPR(c)
    /// ```
    ///
    /// Algorithm
    /// ---------
    /// - All edges are sorted by increasing `cost`,
    /// - Thresholds are evaluated **only at unique cost values**,
    /// - Costs are processed in **tie blocks**: when the threshold reaches a given
    ///   cost value, all edges with that cost are included simultaneously,
    /// - The threshold that maximizes `J(c)` is retained.
    ///
    /// Returned quantities
    /// -------------------
    /// The returned [`BestYouden`] structure contains:
    ///
    /// - `best_j`: the maximum value of Youden’s J,
    /// - `threshold_cost`: the cost threshold at which this maximum is attained,
    /// - `tpr`: true positive rate at the optimal threshold,
    /// - `fpr`: false positive rate at the optimal threshold,
    /// - `precision`: precision at the optimal threshold.
    ///
    /// Interpretation
    /// --------------
    /// - `best_j = 1.0` : perfect separation between good and bad edges,
    /// - `best_j = 0.0` : no discriminative power (equivalent to random choice),
    /// - Negative values indicate pathological behavior.
    ///
    /// In practice:
    /// - The optimal threshold often corresponds to a **reasonable global gate**
    ///   on edge cost,
    /// - Youden’s J implicitly balances recall and false positives **symmetrically**,
    /// - This metric is useful for **initial calibration**, but may not be optimal
    ///   when the operational regime requires extremely low false-positive rates.
    ///
    /// Tie handling
    /// ------------
    /// When multiple edges share the same `cost`, they are incorporated together
    /// before evaluating the statistic. This ensures:
    /// - a stable and unbiased estimate of the optimal threshold,
    /// - consistent behavior when costs are discretized or weakly quantized.
    ///
    /// Assumptions
    /// -----------
    /// - `self.len() == n_good + n_bad`,
    /// - both classes are present (`n_good > 0` and `n_bad > 0`),
    /// - all costs are finite (non-finite values should be filtered upstream).
    ///
    /// Returns
    /// -------
    /// * A [`BestYouden`] structure describing the optimal threshold and associated
    ///   operating characteristics.
    ///
    /// See also
    /// --------
    /// * [`ks_stat_on_cost`] – maximum CDF separation, closely related to Youden’s J.
    /// * [`fpr_at_tpr_on_cost`] – threshold selection under fixed recall constraints.
    /// * [`best_f1_on_cost`] – threshold optimizing F1 score.
    /// * [`graph_ranking_metrics`] – graph-aware metrics for Top-K linking.
    fn best_youden_j_on_cost(&self, n_good: usize, n_bad: usize) -> BestYouden;

    /// Compute the operating threshold on `cost` that maximizes the **F1 score**.
    ///
    /// Overview
    /// --------
    /// This function finds the **cost threshold** (lower is better) that maximizes
    /// the **F1 score**, defined as the harmonic mean of **precision** and **recall**
    /// (true positive rate).
    ///
    /// In inter-night asteroid linking, this metric identifies a threshold that
    /// balances:
    /// - **precision**: how many selected edges are correct,
    /// - **recall**: how many true edges are recovered.
    ///
    /// Unlike Youden’s J, which balances recall against false-positive rate,
    /// F1 directly optimizes the trade-off between **purity** and **completeness**
    /// of the selected edge set.
    ///
    /// Definitions
    /// -----------
    /// Consider a population of labeled edges split into two classes:
    ///
    /// - **Good edges**: edges linking observations of the same asteroid
    ///   (`is_good = true`), with total count `n_good`,
    /// - **Bad edges**: edges linking different asteroids
    ///   (`is_good = false`), with total count `n_bad`.
    ///
    /// For a given threshold value `c` on `cost`, edges are classified as:
    ///
    /// ```text
    /// predicted_good = (cost ≤ c)
    /// ```
    ///
    /// From this classification, define:
    ///
    /// - `TP` (true positives): number of good edges with `cost ≤ c`,
    /// - `FP` (false positives): number of bad edges with `cost ≤ c`,
    /// - `FN` (false negatives): number of good edges with `cost > c`,
    ///
    /// and the derived quantities:
    ///
    /// ```text
    /// precision = TP / (TP + FP)
    /// recall    = TP / n_good
    /// ```
    ///
    /// The F1 score at threshold `c` is:
    ///
    /// ```text
    /// F1(c) = 2 · precision(c) · recall(c) / (precision(c) + recall(c))
    /// ```
    ///
    /// Algorithm
    /// ---------
    /// - All edges are sorted by increasing `cost`,
    /// - Thresholds are evaluated **only at unique cost values**,
    /// - Costs are processed in **tie blocks** so that all edges with identical
    ///   `cost` values are included simultaneously,
    /// - The threshold that maximizes `F1(c)` is retained.
    ///
    /// Returned quantities
    /// -------------------
    /// The returned [`BestF1`] structure contains:
    ///
    /// - `best_f1`: the maximum F1 score achieved,
    /// - `op`: an [`OperatingPoint`] describing the optimal threshold, including:
    ///   - `threshold_cost`,
    ///   - `precision`,
    ///   - `recall` (true positive rate),
    ///   - `fpr`, `tp`, `fp`, `tn`, `fn`.
    ///
    /// Interpretation
    /// --------------
    /// - `F1 = 1.0` : perfect precision and recall (all selected edges are correct
    ///   and all true edges are recovered),
    /// - `F1 = 0.0` : either zero precision or zero recall,
    /// - Intermediate values reflect the balance between contamination and completeness.
    ///
    /// In practice:
    /// - Maximizing F1 tends to favor **moderate thresholds** that include a
    ///   non-negligible number of false positives if that improves recall,
    /// - F1 is appropriate when **precision and recall are equally important**,
    /// - F1 is **not well suited** when operating in regimes requiring extremely
    ///   low false-positive rates (e.g., Top-K graph pruning with strict budgets).
    ///
    /// Tie handling
    /// ------------
    /// When multiple edges share the same `cost`, they are incorporated together
    /// before evaluating the F1 score. This ensures:
    /// - a stable and unbiased estimate of the optimal threshold,
    /// - consistent behavior when costs are discretized or weakly quantized.
    ///
    /// Assumptions
    /// -----------
    /// - `self.len() == n_good + n_bad`,
    /// - both classes are present (`n_good > 0` and `n_bad > 0`),
    /// - all costs are finite (non-finite values should be filtered upstream).
    ///
    /// Returns
    /// -------
    /// * A [`BestF1`] structure describing the threshold that maximizes the F1 score
    ///   and the associated operating characteristics.
    ///
    /// See also
    /// --------
    /// * [`best_youden_j_on_cost`] – symmetric balance between recall and false-positive rate.
    /// * [`fpr_at_tpr_on_cost`] – threshold selection under fixed recall constraints.
    /// * [`tpr_at_fpr_on_cost`] – threshold selection under strict contamination limits.
    /// * [`graph_ranking_metrics`] – Top-K and ranking-based metrics for linking graphs.
    fn best_f1_on_cost(&self, n_good: usize, n_bad: usize) -> BestF1;

    /// Compute the **false positive rate (FPR)** achieved at a target
    /// **true positive rate (TPR)** using thresholds on `cost`.
    ///
    /// Overview
    /// --------
    /// This function evaluates the **best achievable contamination level**
    /// (false positive rate) when enforcing a **minimum recall constraint**
    /// on true edges.
    ///
    /// More precisely, it computes the **minimum FPR** over all cost thresholds
    /// (lower is better) such that the resulting **TPR is greater than or equal to**
    /// the requested `target_tpr`.
    ///
    /// In inter-night asteroid linking, this metric directly answers the question:
    ///
    /// > *“If I require at least X% of true links to be kept, what is the lowest
    /// > fraction of false links I must accept?”*
    ///
    /// Definitions
    /// -----------
    /// Consider a population of labeled edges split into two classes:
    ///
    /// - **Good edges**: edges linking observations of the same asteroid
    ///   (`is_good = true`), with total count `n_good`,
    /// - **Bad edges**: edges linking different asteroids
    ///   (`is_good = false`), with total count `n_bad`.
    ///
    /// For a given threshold value `c` on `cost`, edges are classified as:
    ///
    /// ```text
    /// predicted_good = (cost ≤ c)
    /// ```
    ///
    /// From this classification, define:
    ///
    /// - `TP` (true positives): number of good edges with `cost ≤ c`,
    /// - `FP` (false positives): number of bad edges with `cost ≤ c`,
    ///
    /// and the corresponding rates:
    ///
    /// ```text
    /// TPR = TP / n_good
    /// FPR = FP / n_bad
    /// ```
    ///
    /// Target constraint
    /// -----------------
    /// Let `target_tpr` be a value in `[0, 1]` specifying the **minimum acceptable
    /// recall**. Among all thresholds `c` such that:
    ///
    /// ```text
    /// TPR(c) ≥ target_tpr
    /// ```
    ///
    /// this function returns:
    ///
    /// ```text
    /// min_c FPR(c)
    /// ```
    ///
    /// Algorithm
    /// ---------
    /// - All edges are sorted by increasing `cost`,
    /// - Thresholds are evaluated **only at unique cost values**,
    /// - Costs are processed in **tie blocks** so that all edges with identical
    ///   `cost` values are included simultaneously,
    /// - The minimal FPR among thresholds satisfying the recall constraint is retained.
    ///
    /// Interpretation
    /// --------------
    /// - A **small returned value** indicates that high recall can be achieved with
    ///   little contamination,
    /// - A **large value** indicates that maintaining the requested recall requires
    ///   accepting many false edges,
    /// - If the value is close to `1.0`, the requested recall level is essentially
    ///   unattainable without accepting almost all false edges.
    ///
    /// This metric is especially informative when:
    /// - operating in **high-recall regimes** (e.g. `TPR ≥ 0.95` or `0.99`),
    /// - tuning gates or pruning strategies that must preserve most true links
    ///   while keeping the graph sparse.
    ///
    /// Numerical considerations
    /// ------------------------
    /// - `target_tpr` is clamped to `[0, 1]`,
    /// - A small tolerance is used when comparing `TPR` to `target_tpr` to avoid
    ///   floating-point boundary effects,
    /// - If one of the classes is missing (`n_good == 0` or `n_bad == 0`), the
    ///   function returns `0.0` by convention.
    ///
    /// Assumptions
    /// -----------
    /// - `self.len() == n_good + n_bad`,
    /// - both classes are present for meaningful results,
    /// - all costs are finite (non-finite values should be filtered upstream).
    ///
    /// Returns
    /// -------
    /// * A value in `[0, 1]` representing the **minimum achievable FPR** under the
    ///   specified recall constraint.
    ///
    /// See also
    /// --------
    /// * [`tpr_at_fpr_on_cost`] – dual metric: maximum recall at fixed contamination.
    /// * [`best_youden_j_on_cost`] – unconstrained threshold selection.
    /// * [`best_f1_on_cost`] – balance between precision and recall.
    /// * [`graph_ranking_metrics`] – Top-K oriented metrics for graph construction.
    fn fpr_at_tpr_on_cost(&self, n_good: usize, n_bad: usize, target_tpr: f64) -> f64;

    /// Compute the **true positive rate (TPR)** achievable at a target
    /// **false positive rate (FPR)** using thresholds on `cost`.
    ///
    /// Overview
    /// --------
    /// This function evaluates the **maximum achievable recall**
    /// (true positive rate) when enforcing a **strict upper bound on contamination**
    /// (false positive rate).
    ///
    /// More precisely, it computes the **maximum TPR** over all cost thresholds
    /// (lower is better) such that the resulting **FPR is less than or equal to**
    /// the requested `target_fpr`.
    ///
    /// In inter-night asteroid linking, this metric directly answers the question:
    ///
    /// > *“If I can tolerate at most X% of false links, what fraction of true links
    /// > can I recover?”*
    ///
    /// Definitions
    /// -----------
    /// Consider a population of labeled edges split into two classes:
    ///
    /// - **Good edges**: edges linking observations of the same asteroid
    ///   (`is_good = true`), with total count `n_good`,
    /// - **Bad edges**: edges linking different asteroids
    ///   (`is_good = false`), with total count `n_bad`.
    ///
    /// For a given threshold value `c` on `cost`, edges are classified as:
    ///
    /// ```text
    /// predicted_good = (cost ≤ c)
    /// ```
    ///
    /// From this classification, define:
    ///
    /// - `TP` (true positives): number of good edges with `cost ≤ c`,
    /// - `FP` (false positives): number of bad edges with `cost ≤ c`,
    ///
    /// and the corresponding rates:
    ///
    /// ```text
    /// TPR = TP / n_good
    /// FPR = FP / n_bad
    /// ```
    ///
    /// Target constraint
    /// -----------------
    /// Let `target_fpr` be a value in `[0, 1]` specifying the **maximum acceptable
    /// false positive rate**. Among all thresholds `c` such that:
    ///
    /// ```text
    /// FPR(c) ≤ target_fpr
    /// ```
    ///
    /// this function returns:
    ///
    /// ```text
    /// max_c TPR(c)
    /// ```
    ///
    /// Algorithm
    /// ---------
    /// - All edges are sorted by increasing `cost`,
    /// - Thresholds are evaluated **only at unique cost values**,
    /// - Costs are processed in **tie blocks** so that all edges with identical
    ///   `cost` values are included simultaneously,
    /// - The maximum TPR among thresholds satisfying the contamination constraint
    ///   is retained.
    ///
    /// Interpretation
    /// --------------
    /// - A **high returned value** indicates that most true edges can be preserved
    ///   while keeping contamination below the requested level,
    /// - A **low value** indicates that the contamination constraint is very strict
    ///   and only a small fraction of true edges can be retained,
    /// - If the value is close to `0.0`, the requested contamination level is
    ///   essentially incompatible with the current score distribution.
    ///
    /// This metric is especially relevant when:
    /// - operating in **low-contamination regimes** (e.g. `FPR ≤ 10⁻³`),
    /// - tuning pre-filtering gates or Top-K pruning thresholds,
    /// - building sparse graphs for downstream assignment or flow solvers.
    ///
    /// Numerical considerations
    /// ------------------------
    /// - `target_fpr` is clamped to `[0, 1]`,
    /// - A small tolerance is used when comparing `FPR` to `target_fpr` to avoid
    ///   floating-point boundary effects,
    /// - If one of the classes is missing (`n_good == 0` or `n_bad == 0`), the
    ///   function returns `0.0` by convention.
    ///
    /// Assumptions
    /// -----------
    /// - `self.len() == n_good + n_bad`,
    /// - both classes are present for meaningful results,
    /// - all costs are finite (non-finite values should be filtered upstream).
    ///
    /// Returns
    /// -------
    /// * A value in `[0, 1]` representing the **maximum achievable TPR** under the
    ///   specified contamination constraint.
    ///
    /// See also
    /// --------
    /// * [`fpr_at_tpr_on_cost`] – dual metric: minimal contamination at fixed recall.
    /// * [`best_youden_j_on_cost`] – unconstrained threshold selection.
    /// * [`best_f1_on_cost`] – balance between precision and recall.
    /// * [`graph_ranking_metrics`] – Top-K oriented metrics for graph construction.
    fn tpr_at_fpr_on_cost(&self, n_good: usize, n_bad: usize, target_fpr: f64) -> f64;

    /// Compute a comprehensive set of performance metrics for edge scoring.
    ///
    /// Overview
    /// --------
    /// This method computes **all commonly used evaluation metrics at once** and
    /// returns them bundled in a [`MetricsSummary`] structure. It is intended as a
    /// **one-liner convenience entry point** for:
    /// - offline tuning of scoring weights and gates,
    /// - comparative evaluation of different score configurations,
    /// - logging and visualization in optimization pipelines.
    ///
    /// The returned metrics jointly characterize:
    /// - **global separability** between true and false edges,
    /// - **ranking quality** of the score,
    /// - **threshold-based operating points**,
    /// - **behavior in low-contamination regimes**, which are critical for building
    ///   sparse inter-night linking graphs.
    ///
    /// Definitions
    /// -----------
    /// Consider a population of labeled edges split into two classes:
    ///
    /// - **Good edges**: edges linking observations of the same asteroid
    ///   (`is_good = true`), with total count `n_good`,
    /// - **Bad edges**: edges linking different asteroids
    ///   (`is_good = false`), with total count `n_bad`.
    ///
    /// Assumptions
    /// -----------
    /// - `self.len() == n_good + n_bad`,
    /// - both classes are present (`n_good > 0` and `n_bad > 0`),
    /// - all costs and scores are finite.
    ///
    /// Metrics included
    /// ----------------
    /// The [`MetricsSummary`] contains the following fields:
    ///
    /// ### Global / threshold-free metrics
    /// - `auc`  
    ///   ROC-AUC computed via the Mann–Whitney formulation on `score`.
    ///   Measures global ranking separability, independent of any threshold.
    ///
    /// - `auc_pr`  
    ///   PR-AUC (Average Precision) on `score`.
    ///   More informative than ROC-AUC when true edges are rare.
    ///
    /// - `ks`  
    ///   Kolmogorov–Smirnov statistic between the cost distributions of good and bad
    ///   edges. Measures maximum global separation of empirical CDFs.
    ///
    /// ### Threshold-based metrics
    /// - `best_youden`  
    ///   Cost threshold maximizing Youden’s J (`TPR − FPR`).
    ///   Useful as a baseline global gate.
    ///
    /// - `best_f1`  
    ///   Cost threshold maximizing the F1 score (harmonic mean of precision and recall).
    ///   Emphasizes balance between contamination and completeness.
    ///
    /// - `mcc_at_best_f1`  
    ///   Matthews correlation coefficient evaluated at the F1-optimal threshold.
    ///   Provides a balanced scalar measure even under class imbalance.
    ///
    /// ### Low-contamination operating points
    /// - `fpr_at_tpr_90`, `fpr_at_tpr_95`, `fpr_at_tpr_99`  
    ///   Minimal achievable false positive rate when enforcing recall levels of
    ///   90%, 95%, and 99%, respectively.
    ///
    /// - `tpr_at_fpr_1e3`  
    ///   Maximum achievable recall when constraining the false positive rate to
    ///   at most `10⁻³`.
    ///
    /// Interpretation
    /// --------------
    /// - Use **AUC / PR-AUC** to assess whether the score contains meaningful
    ///   discriminative information.
    /// - Use **KS** to diagnose global distribution separation.
    /// - Use **Youden / F1** as reference thresholds during early calibration.
    /// - Use **FPR@TPR** and **TPR@FPR** to evaluate suitability for sparse graph
    ///   construction and Top-K pruning.
    ///
    /// No single metric is sufficient on its own; the summary is designed to give a
    /// **multi-angle view** of score quality aligned with the needs of inter-night
    /// linking pipelines.
    ///
    /// Returns
    /// -------
    /// * A [`MetricsSummary`] structure containing all computed metrics.
    ///
    /// See also
    /// --------
    /// * [`graph_ranking_metrics`] – graph-aware Top-K and ranking metrics.
    /// * [`Metrics`] – individual metric definitions.
    /// * `fink-fat-engine::graph::score` – edge cost construction.
    fn summary(&self, n_good: usize, n_bad: usize) -> MetricsSummary {
        let auc = self.auc_mann_whitney_on_score(n_good, n_bad);
        let auc_pr = self.average_precision_on_score(n_good, n_bad);
        let ks = self.ks_stat_on_cost(n_good, n_bad);
        let best_youden = self.best_youden_j_on_cost(n_good, n_bad);
        let best_f1 = self.best_f1_on_cost(n_good, n_bad);

        MetricsSummary {
            auc,
            auc_pr,
            ks,
            best_youden,
            mcc_at_best_f1: best_f1.op.mcc(),
            best_f1,
            fpr_at_tpr_90: self.fpr_at_tpr_on_cost(n_good, n_bad, 0.90),
            fpr_at_tpr_95: self.fpr_at_tpr_on_cost(n_good, n_bad, 0.95),
            fpr_at_tpr_99: self.fpr_at_tpr_on_cost(n_good, n_bad, 0.99),
            tpr_at_fpr_1e3: self.tpr_at_fpr_on_cost(n_good, n_bad, 1e-3),
        }
    }
}
