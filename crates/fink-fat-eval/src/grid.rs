//! Sweep grid helpers (linspace/logspace).
//!
//! This module provides small, dependency-free helpers to build 1D parameter
//! grids for diagnostic sweeps (e.g., threshold scans). Both functions return
//! `n` points **including the endpoints** when `n >= 2`, which is convenient
//! for reproducible experiments and plot generation.

/// Generate a log-spaced grid between `min` and `max` (inclusive).
///
/// This routine returns `n` values `x[i]` such that `ln(x[i])` is linearly
/// spaced between `ln(min)` and `ln(max)`:
///
/// ```text
/// x[i] = exp( ln(min) + t * (ln(max) - ln(min)) ),  with  t = i/(n-1)
/// ```
///
/// Arguments
/// ---------
/// * `min` – Lower bound of the interval. Must be strictly positive and finite.
/// * `max` – Upper bound of the interval. Must be strictly positive and finite.
/// * `n` – Number of points to generate.
///   - If `n == 0`, returns an empty vector.
///   - If `n == 1`, returns `[min]`.
///
/// Return
/// ------
/// * `Vec<f64>` – A vector of `n` log-spaced points in `[min, max]`.
///   When `n >= 2`, both endpoints are included (subject to floating-point
///   rounding).
///
/// Notes
/// -----
/// * This function does not enforce input validity beyond what `ln()`/`exp()`
///   naturally produce. If `min <= 0` or `max <= 0`, results will contain
///   `NaN`/`Inf`. Prefer validating bounds at the call site (especially for CLI).
/// * Log spacing is often preferred when sweeping parameters across multiple
///   orders of magnitude (e.g., angular thresholds), because it allocates more
///   resolution near small values where metrics can change rapidly.
pub fn logspace(min: f64, max: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![min];
    }
    let a = min.ln();
    let b = max.ln();
    (0..n)
        .map(|i| {
            let t = (i as f64) / ((n - 1) as f64);
            (a + t * (b - a)).exp()
        })
        .collect()
}

/// Generate a linearly spaced grid between `min` and `max` (inclusive).
///
/// This routine returns `n` values `x[i]` uniformly spaced on the real line:
///
/// ```text
/// x[i] = min + t * (max - min),  with  t = i/(n-1)
/// ```
///
/// Arguments
/// ---------
/// * `min` – Lower bound of the interval. Must be finite for meaningful output.
/// * `max` – Upper bound of the interval. Must be finite for meaningful output.
/// * `n` – Number of points to generate.
///   - If `n == 0`, returns an empty vector.
///   - If `n == 1`, returns `[min]`.
///
/// Return
/// ------
/// * `Vec<f64>` – A vector of `n` linearly spaced points in `[min, max]`.
///   When `n >= 2`, both endpoints are included (subject to floating-point
///   rounding).
///
/// Notes
/// -----
/// * If `min > max`, the function still produces a valid grid, but it will be
///   decreasing.
/// * Linear spacing is best suited when the metric behaves roughly linearly
///   over the swept interval (e.g., additive penalties, durations, bin widths).
pub fn linspace(min: f64, max: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![min];
    }
    (0..n)
        .map(|i| {
            let t = (i as f64) / ((n - 1) as f64);
            min + t * (max - min)
        })
        .collect()
}
