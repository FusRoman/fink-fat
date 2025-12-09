// src/seeding/time_binner.rs

//! Uniform time binning for MJD(TT) streams.
//!
//! This module provides [`UniformTimeBinner`], a concrete implementation
//! of the [`TimeBinner`] trait. It divides a continuous time axis expressed
//! in Modified Julian Days (TT timescale) into **uniform, non-overlapping bins**
//! of fixed width `dt`.
//!
//! ## Interval Convention
//!
//! Each bin corresponds to a half-open interval
//!
//! ```text
//! [t0 + k·dt, t0 + (k+1)·dt)
//! ```
//!
//! where
//! - `t0` is the origin (MJD TT, in days),
//! - `dt` is the bin width (> 0, in days),
//! - `k ∈ ℤ` is the integer bin index stored in [`TimeBin`].
//!
//! ## Boundary Handling
//!
//! The method [`bins_in_range`](crate::seeding::space_time_bucket::TimeBinner::bins_in_range) returns all bins overlapping the **closed**
//! interval `[a, b]`. To achieve closed-interval semantics while remaining
//! robust against floating-point roundoff, the upper bound `b` is nudged
//! downward by a small epsilon proportional to the bin width.

use crate::{
    MjdTt,
    spacetime_bucket::time_binner::{TimeBin, TimeBinner},
};

/// Uniform time binner for Modified Julian Date (TT) streams.
///
/// This structure partitions continuous time (days in MJD TT) into
/// fixed-width bins of duration `dt`. Each bin is addressed by an
/// integer index `k ∈ ℤ`, with start time `t0 + k·dt`.
#[derive(Clone, Copy, Debug)]
pub struct UniformTimeBinner {
    /// Origin of the binning scheme (days, MJD TT).
    t0: MjdTt,
    /// Bin width (days, strictly > 0).
    dt: MjdTt,
}

impl UniformTimeBinner {
    /// Construct a new uniform time binner.
    ///
    /// Parameters
    /// ----------
    /// * `t0` — Origin of the binning (days, MJD TT).
    /// * `dt_days` — Bin width (days, must be strictly positive).
    ///
    /// Panics
    /// ------
    /// Panics if `dt_days <= 0.0`.
    pub fn new(t0: MjdTt, dt_days: MjdTt) -> Self {
        assert!(dt_days > 0.0, "bin width must be > 0");
        Self { t0, dt: dt_days }
    }

    /// Return the origin `t0` of the binning (days, MJD TT).
    #[inline]
    pub fn origin(&self) -> MjdTt {
        self.t0
    }

    /// Compute the start time of bin `k`.
    ///
    /// Formula
    /// -------
    /// `t_start = t0 + k·dt`
    ///
    /// Parameters
    /// ----------
    /// * `k` — Bin index (integer).
    ///
    /// Returns
    /// -------
    /// Start time of bin `k` (days, MJD TT).
    #[inline]
    pub fn bin_start(&self, k: i64) -> MjdTt {
        self.t0 + (k as f64) * self.dt
    }

    /// Compute the end time of bin `k`.
    ///
    /// Formula
    /// -------
    /// `t_end = t0 + (k+1)·dt`
    ///
    /// Parameters
    /// ----------
    /// * `k` — Bin index (integer).
    ///
    /// Returns
    /// -------
    /// End time of bin `k` (days, MJD TT).
    #[inline]
    pub fn bin_end(&self, k: i64) -> MjdTt {
        self.t0 + ((k + 1) as f64) * self.dt
    }

    /// Compute the bin index for time `t`.
    ///
    /// Formula
    /// -------
    /// `k = floor((t - t0)/dt)`
    ///
    /// Notes
    /// -----
    /// Handles negative values correctly by relying on `floor`.
    #[inline]
    fn bin_index(&self, t: MjdTt) -> i64 {
        ((t - self.t0) / self.dt).floor() as i64
    }

    /// Return a small epsilon proportional to the bin width.
    ///
    /// This epsilon is subtracted from the upper bound `b` in
    /// [`bins_in_range`] to ensure closed-interval semantics while
    /// avoiding floating-point roundoff issues at exact bin edges.
    ///
    /// Scaling `eps` with `dt` ensures consistent behavior across
    /// bin widths.
    #[inline]
    fn eps(&self) -> f64 {
        1e-12 * self.dt
    }
}

impl TimeBinner for UniformTimeBinner {
    /// Return the bin index containing `mjd_tt`.
    ///
    /// Parameters
    /// ----------
    /// * `mjd_tt` — Time to locate (days, MJD TT).
    ///
    /// Returns
    /// -------
    /// * [`TimeBin`] — Bin containing `mjd_tt`.
    #[inline]
    fn bin_for(&self, mjd_tt: MjdTt) -> TimeBin {
        TimeBin(self.bin_index(mjd_tt))
    }

    /// Return all bins overlapping the interval `[t0, t1]`.
    ///
    /// Parameters
    /// ----------
    /// * `t0` — Lower bound of the interval (days, MJD TT).
    /// * `t1` — Upper bound of the interval (days, MJD TT).
    ///
    /// Behavior
    /// --------
    /// - The order of `(t0, t1)` is normalized (swapped if `t0 > t1`).
    /// - NaN values yield an empty result.
    /// - The interval is treated as **closed** `[a,b]`, ensured by
    ///   subtracting a small epsilon from `b` before computing indices.
    ///
    /// Returns
    /// -------
    /// Vector of [`TimeBin`] covering the interval `[t0, t1]`.
    fn bins_in_range(&self, t0: MjdTt, t1: MjdTt) -> Vec<TimeBin> {
        if t0.is_nan() || t1.is_nan() {
            return Vec::new();
        }

        let (a, b) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };

        let b_adj = b - self.eps();

        let k_start = self.bin_index(a);
        let k_end = self.bin_index(b_adj);

        if k_end < k_start {
            return Vec::new();
        }

        let n = (k_end - k_start + 1) as usize;
        let mut out = Vec::with_capacity(n);
        for k in k_start..=k_end {
            out.push(TimeBin(k));
        }
        out
    }

    /// Return the bin width (days, MJD TT).
    #[inline]
    fn bin_width(&self) -> MjdTt {
        self.dt
    }
}

#[cfg(test)]
mod uniform_bin_tests {
    use super::*;

    #[test]
    fn test_basic_indexing() {
        let b = UniformTimeBinner::new(59000.0, 1.0);
        // Bin 0 covers [59000.0, 59001.0)
        assert_eq!(b.bin_for(59000.0), TimeBin(0));
        assert_eq!(b.bin_for(59000.0999999), TimeBin(0));
        assert_eq!(b.bin_for(59000.1), TimeBin(0));
        assert_eq!(b.bin_for(59012.5), TimeBin(12));
        assert_eq!(b.bin_start(1), 59001.0);
        assert_eq!(b.bin_end(5), 59006.0);
    }

    #[test]
    fn test_bins_in_range_closed_interval() {
        let b = UniformTimeBinner::new(59000.0, 0.5);
        // bins: [59000.0,59000.5) -> k=0 ; [59000.5,59001.0) -> k=1 ; [59001.0,59001.5) -> k=2

        // Exact boundaries: include k=0 and k=1
        let ks = b.bins_in_range(59000.0, 59001.0);
        let v: Vec<i64> = ks.into_iter().map(|TimeBin(k)| k).collect();
        assert_eq!(v, vec![0, 1, 2]); // because [a,b] is closed and 59001.0 falls into k=2 with the epsilon trick

        // Small interior range
        let ks = b.bins_in_range(59000.1, 59000.6);
        let v: Vec<i64> = ks.into_iter().map(|TimeBin(k)| k).collect();
        assert_eq!(v, vec![0, 1]);

        // Reversed input order should be handled
        let ks = b.bins_in_range(59000.6, 59000.1);
        let v: Vec<i64> = ks.into_iter().map(|TimeBin(k)| k).collect();
        assert_eq!(v, vec![0, 1]);
    }

    #[test]
    fn test_negative_offsets() {
        let b = UniformTimeBinner::new(59000.0, 1.0);
        // Times before origin
        assert_eq!(b.bin_for(58999.9), TimeBin(-1));
        assert_eq!(b.bin_for(58998.1), TimeBin(-2));
        let ks = b.bins_in_range(58998.9, 59000.1);
        let v: Vec<i64> = ks.into_iter().map(|TimeBin(k)| k).collect();
        assert_eq!(v, vec![-2, -1, 0]);
    }
}
