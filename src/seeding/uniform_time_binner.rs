// src/seeding/time_binner.rs

//! Uniform time binning for MJD(TT) streams.
//!
//! Bins are half-open intervals: [t0 + k*dt, t0 + (k+1)*dt)
//! with integer k ∈ ℤ stored in `TimeBin(i64)`.
//!
//! `bins_in_range([a,b])` returns all bins that overlap the *closed*
//! interval \[a,b\]. This is achieved by slightly nudging `b` downward
//! to handle exact-boundary cases in presence of floating-point roundoff.

use crate::seeding::space_time_bucket::{MjdTt, TimeBin, TimeBinner};

#[derive(Clone, Copy, Debug)]
pub struct UniformTimeBinner {
    t0: MjdTt, // origin (days, MJD TT)
    dt: MjdTt, // bin width (days), strictly > 0
}

impl UniformTimeBinner {
    /// Create a new uniform time binner with origin `t0` and bin width `dt_days` (> 0).
    pub fn new(t0: MjdTt, dt_days: MjdTt) -> Self {
        assert!(dt_days > 0.0, "bin width must be > 0");
        Self { t0, dt: dt_days }
    }

    /// Origin (MJD TT) of the binning.
    #[inline]
    pub fn origin(&self) -> MjdTt {
        self.t0
    }

    /// Start time of bin `k`: t0 + k*dt
    #[inline]
    pub fn bin_start(&self, k: i64) -> MjdTt {
        self.t0 + (k as f64) * self.dt
    }

    /// End time of bin `k`: t0 + (k+1)*dt
    #[inline]
    pub fn bin_end(&self, k: i64) -> MjdTt {
        self.t0 + ((k + 1) as f64) * self.dt
    }

    /// Internal: compute floor((t - t0)/dt) in i64, robust to negative values.
    #[inline]
    fn bin_index(&self, t: MjdTt) -> i64 {
        ((t - self.t0) / self.dt).floor() as i64
    }

    /// Small epsilon relative to `dt`, used to make [a,b] closed on the upper bound.
    #[inline]
    fn eps(&self) -> f64 {
        // conservative, scale with dt to keep the same “fraction” of a bin across scales
        1e-12 * self.dt
    }
}

impl TimeBinner for UniformTimeBinner {
    #[inline]
    fn bin_for(&self, mjd_tt: MjdTt) -> TimeBin {
        TimeBin(self.bin_index(mjd_tt))
    }

    fn bins_in_range(&self, t0: MjdTt, t1: MjdTt) -> Vec<TimeBin> {
        if t0.is_nan() || t1.is_nan() {
            return Vec::new();
        }
        // Normalize order
        let (a, b) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };

        // Make the upper bound "belong" to its natural bin even if it falls exactly
        // on a boundary; subtract a tiny epsilon scaled to dt.
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
