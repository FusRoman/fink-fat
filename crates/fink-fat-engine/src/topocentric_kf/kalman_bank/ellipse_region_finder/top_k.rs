use crate::topocentric_kf::single_kalman::KFState;
use serde::{Deserialize, Serialize};

/// Controls which hypotheses from the bank are used to build a [`SearchRegion`](super::SearchRegion).
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
/// - [`TopK::Best`]`(k)` – the `k` hypotheses with the highest weights,
///   renormalized to sum to 1.
/// - [`TopK::WeightThreshold`]`(theta)` – retains the minimal set of hypotheses
///   (sorted by descending weight) whose cumulative weight reaches `theta`.
///   For example, `WeightThreshold(0.99)` discards all hypotheses beyond the
///   99 % credible set, eliminating low-weight spatial outliers that would
///   otherwise inflate the search region.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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
    ///
    /// # Why the input is normalized *first*
    ///
    /// "The minimal set whose cumulative weight reaches θ" is only meaningful
    /// on a probability distribution. On raw weights the cut point is
    /// scale-dependent and therefore arbitrary.
    ///
    /// This matters in practice, not just in principle: a freshly seeded bank
    /// carries **unnormalized** population priors — `KFBank::from_grid` stores
    /// `log_weight = w.ln()` and the `post_step_cleanup` that would normalize
    /// them is not applied at birth, so `normalize_weights` first runs after
    /// the first real update. Measured on real banks, `Σ exp(log_weight)` has
    /// a median of ~90. Accumulating those raw weights against θ = 0.99 meant
    /// the single highest-weight node cleared the threshold on its own, and a
    /// ~418-hypothesis bank collapsed to **one** hypothesis — reducing the
    /// predicted search region to a ~138″ cone around one arbitrary (ρ, ρ̇)
    /// grid node. That was the dominant cause of the step-1/2 `not_matched%`
    /// in `mot_analysis` (26.3% / 11.3%), and of why it decayed to ~2% by
    /// step 3: the first real update normalizes the bank, after which this
    /// selection started behaving as intended.
    fn keep_threshold(predicted: &mut Vec<(f64, KFState)>, theta: f64) {
        predicted.sort_unstable_by(|(wa, _), (wb, _)| wb.total_cmp(wa));
        let weights: Vec<f64> = predicted.iter().map(|(w, _)| *w).collect();
        predicted.truncate(threshold_keep_count(&weights, theta));
        renormalize(predicted);
    }
}

/// How many entries of `sorted_desc` (weights, descending, **not** required to
/// be normalized) are needed for their share of the total to reach `theta`.
///
/// Normalizing inside is the whole point — see [`TopK::keep_threshold`]'s doc
/// for the bank-seeding bug that made this necessary. Split out from
/// `keep_threshold` so the selection arithmetic is unit-testable without
/// constructing `KFState`s, which would require a loaded ephemeris.
///
/// Returns `sorted_desc.len()` when the weights sum to zero or less (nothing
/// to discriminate on, so keep everything) and `0` for an empty input.
fn threshold_keep_count(sorted_desc: &[f64], theta: f64) -> usize {
    let theta = theta.clamp(0.0, 1.0);
    let total: f64 = sorted_desc.iter().sum();
    if total <= 0.0 {
        return sorted_desc.len();
    }
    let mut cumul = 0.0;
    for (i, w) in sorted_desc.iter().enumerate() {
        cumul += w / total;
        if cumul >= theta {
            return i + 1;
        }
    }
    sorted_desc.len()
}

/// Renormalize weights in-place so they sum to 1.
fn renormalize(predicted: &mut [(f64, KFState)]) {
    let total: f64 = predicted.iter().map(|(w, _)| w).sum();
    if total > 0.0 {
        predicted.iter_mut().for_each(|(w, _)| *w /= total);
    }
}

#[cfg(test)]
mod threshold_keep_count_tests {
    use super::threshold_keep_count;

    /// The regression this function exists for: a freshly seeded `KFBank`
    /// stores raw population priors, so its weights sum to ~90 rather than 1
    /// (measured median over real banks). Thresholding those directly let the
    /// single largest weight clear θ = 0.99 on its own and collapsed a
    /// ~418-hypothesis bank to one, shrinking the search region to a cone
    /// around one arbitrary grid node.
    #[test]
    fn near_uniform_unnormalized_weights_keep_almost_everything() {
        let weights = vec![0.215_f64; 418]; // sums to ~90, like a real seeded bank
        let keep = threshold_keep_count(&weights, 0.99);
        assert!(
            keep >= 414,
            "expected a 99% credible set of near-uniform weights to keep almost all \
             418 hypotheses, kept {keep}"
        );
    }

    /// Scale invariance is the property that was broken: multiplying every
    /// weight by a constant must not change the selection.
    #[test]
    fn selection_is_scale_invariant() {
        let base = [0.5_f64, 0.3, 0.15, 0.04, 0.01];
        let expected = threshold_keep_count(&base, 0.95);
        for scale in [1e-6, 0.5, 3.0, 90.0, 1e6] {
            let scaled: Vec<f64> = base.iter().map(|w| w * scale).collect();
            assert_eq!(
                threshold_keep_count(&scaled, 0.95),
                expected,
                "selection changed at scale {scale}"
            );
        }
    }

    /// A genuinely dominant hypothesis should still collapse the set — the fix
    /// must not turn `WeightThreshold` into `All`.
    #[test]
    fn dominant_weight_still_collapses_the_set() {
        assert_eq!(threshold_keep_count(&[0.999, 0.0005, 0.0005], 0.99), 1);
    }

    #[test]
    fn degenerate_inputs() {
        assert_eq!(threshold_keep_count(&[], 0.99), 0);
        // All-zero weights carry no information: keep everything rather than
        // truncating to an arbitrary prefix.
        assert_eq!(threshold_keep_count(&[0.0, 0.0, 0.0], 0.99), 3);
        // theta == 0 still keeps at least one entry.
        assert_eq!(threshold_keep_count(&[0.6, 0.4], 0.0), 1);
    }
}
