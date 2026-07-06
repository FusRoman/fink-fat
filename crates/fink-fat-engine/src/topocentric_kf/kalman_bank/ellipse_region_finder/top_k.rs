use crate::topocentric_kf::KFState;

/// Controls which hypotheses from the bank are used to build a [`SearchRegion`].
///
/// When the bank holds many low-weight hypotheses, restricting the region to
/// the most probable ones yields a tighter, more actionable search area while
/// preserving the probabilistic guarantees that matter.
///
/// Variants
/// --------
/// - [`TopK::All`] – conservative fallback: every live hypothesis contributes.
/// - [`TopK::Map`] – only the single highest-weight hypothesis (Maximum A
///   Posteriori). Equivalent to `TopK::Best(1)`.
/// - [`TopK::Best(k)`] – the `k` hypotheses with the highest weights,
///   renormalized to sum to 1.
/// - [`TopK::WeightThreshold(theta)`] – retains the minimal set of hypotheses
///   (sorted by descending weight) whose cumulative weight reaches `theta`.
///   For example, `WeightThreshold(0.99)` discards all hypotheses beyond the
///   99 % credible set, eliminating low-weight spatial outliers that would
///   otherwise inflate the search region.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TopK {
    /// Use all live hypotheses (original conservative behaviour).
    All,
    /// Use only the single best hypothesis (MAP estimate).
    Map,
    /// Use the `k` best hypotheses by descending weight, renormalized.
    Best(usize),
    /// Keep the minimal prefix of hypotheses (sorted by descending weight)
    /// whose cumulative weight reaches `theta ∈ (0, 1]`, then renormalize.
    WeightThreshold(f64),
}

impl Default for TopK {
    /// Defaults to [`TopK::All`] to preserve backward-compatible behaviour.
    fn default() -> Self {
        TopK::All
    }
}

impl TopK {
    /// Select and renormalize hypotheses in-place according to `self`.
    pub fn apply(self, predicted: &mut Vec<(f64, KFState)>) {
        match self {
            TopK::All => {}
            TopK::Map => Self::keep_map(predicted),
            TopK::Best(k) => Self::keep_best(predicted, k),
            TopK::WeightThreshold(theta) => Self::keep_threshold(predicted, theta),
        }
    }

    /// Keep only the highest-weight hypothesis, weight forced to 1.0.
    fn keep_map(predicted: &mut Vec<(f64, KFState)>) {
        let best_idx = predicted
            .iter()
            .enumerate()
            .max_by(|(_, (wa, _)), (_, (wb, _))| wa.total_cmp(wb))
            .map(|(i, _)| i)
            .unwrap_or(0);
        predicted.swap(0, best_idx);
        predicted.truncate(1);
        predicted[0].0 = 1.0;
    }

    /// Keep the `k` highest-weight hypotheses, renormalized.
    fn keep_best(predicted: &mut Vec<(f64, KFState)>, k: usize) {
        let k = k.max(1).min(predicted.len());
        predicted.select_nth_unstable_by(k - 1, |(wa, _), (wb, _)| wb.total_cmp(wa));
        predicted.truncate(k);
        renormalize(predicted);
    }

    /// Keep the minimal prefix (by descending weight) reaching cumulative
    /// weight `theta`, renormalized.
    fn keep_threshold(predicted: &mut Vec<(f64, KFState)>, theta: f64) {
        let theta = theta.clamp(0.0, 1.0);
        predicted.sort_unstable_by(|(wa, _), (wb, _)| wb.total_cmp(wa));
        let mut cumul = 0.0;
        let mut keep = 0;
        for (w, _) in predicted.iter() {
            cumul += w;
            keep += 1;
            if cumul >= theta {
                break;
            }
        }
        predicted.truncate(keep);
        renormalize(predicted);
    }
}

/// Renormalize weights in-place so they sum to 1.
fn renormalize(predicted: &mut [(f64, KFState)]) {
    let total: f64 = predicted.iter().map(|(w, _)| w).sum();
    if total > 0.0 {
        predicted.iter_mut().for_each(|(w, _)| *w /= total);
    }
}
