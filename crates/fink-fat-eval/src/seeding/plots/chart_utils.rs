//! Pure data utilities: histogram binning, percentile computation.
//!
//! All functions here are free of plotters; they only transform `Vec<f64>` data.

// ─────────────────────────────────────────────────────────────────────────────
// Histogram
// ─────────────────────────────────────────────────────────────────────────────

/// Compute equispaced histogram bins from a **pre-sorted** slice.
///
/// Returns `(edges, counts)` where `edges` has `n_bins + 1` entries and
/// `counts` has `n_bins` entries.  Values are partitioned into
/// `[edges[i], edges[i+1])` for `i < n_bins - 1`, and the last bin is
/// inclusive on the right.
///
/// # Panics
/// Does not panic; an empty slice returns a single trivial bin.
pub fn histogram_bins(sorted: &[f64], n_bins: usize) -> (Vec<f64>, Vec<u32>) {
    let n_bins = n_bins.max(1);
    if sorted.is_empty() {
        return (vec![0.0, 1.0], vec![0]);
    }
    let lo = sorted[0];
    let hi = sorted[sorted.len() - 1];
    let range = (hi - lo).max(f64::EPSILON);

    let edges: Vec<f64> = (0..=n_bins)
        .map(|i| lo + (i as f64 / n_bins as f64) * range)
        .collect();

    let mut counts = vec![0u32; n_bins];
    let mut bin = 0usize;
    for &v in sorted {
        while bin + 1 < n_bins && v >= edges[bin + 1] {
            bin += 1;
        }
        counts[bin] += 1;
    }
    (edges, counts)
}

// ─────────────────────────────────────────────────────────────────────────────
// Percentiles
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a quantile from a **pre-sorted** slice.
///
/// `p` is in `[0, 1]`.  Uses the nearest-rank method.
/// Returns `f64::NAN` for empty slices.
pub fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((p * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1);
    sorted[idx]
}

/// Standard percentile reference points (expressed in `[0, 100]`) used across
/// all metric charts.
pub const PERCENTILE_PS: &[f64] = &[
    0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.5,
];

/// Compute `(percentile, value)` pairs for `PERCENTILE_PS`.
///
/// Input slice need not be sorted; it is sorted internally.
/// Returns an empty vec if `values` is empty.
pub fn compute_percentile_table(values: &[f64]) -> Vec<(f64, f64)> {
    if values.is_empty() {
        return vec![];
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    PERCENTILE_PS
        .iter()
        .map(|&p| (p, percentile_sorted(&sorted, p / 100.0)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_basic() {
        let data = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let (edges, counts) = histogram_bins(&data, 2);
        assert_eq!(edges.len(), 3);
        assert_eq!(counts.len(), 2);
        assert_eq!(counts.iter().sum::<u32>(), 5);
    }

    #[test]
    fn percentile_median() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile_sorted(&sorted, 0.5), 3.0);
    }
}
