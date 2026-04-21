use photom::MJDTT;

/// Compact time bin identifier.
///
/// Usually an integer index of uniform-width bins on the MJD(TT) axis.
/// Signed 64-bit to support long spans and negative offsets if needed.
///
/// ### Notes
/// - The exact mapping `MJD → TimeBin` depends on the `TimeBinner`.
/// - Comparable and hashable to serve as a map key.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct TimeBin(pub i64);

/// Time binning interface.
///
/// Implement this for your time partitioner (uniform bins, cadence-aware bins…).
pub trait TimeBinner: Sync {
    /// Return the **time bin** covering `mjd_tt`.
    ///
    /// Parameters
    /// ----------
    /// - `mjd_tt`: Time stamp in MJD(TT) days.
    fn bin_for(&self, mjd_tt: MJDTT) -> TimeBin;

    /// Enumerate all bins **overlapping** the closed interval `[t0, t1]`.
    ///
    /// Parameters
    /// ----------
    /// - `t0`, `t1`: Start/end in MJD(TT) days (no ordering required; implementations may swap).
    fn bins_in_range(&self, t0: MJDTT, t1: MJDTT) -> Vec<TimeBin>;

    /// The **bin width** in days.
    fn bin_width(&self) -> MJDTT;

    /// Return the start time of bin `k`.
    fn bin_start(&self, k: i64) -> MJDTT;

    /// Return the end time of bin `k`.
    fn bin_end(&self, k: i64) -> MJDTT {
        self.bin_start(k) + self.bin_width()
    }
}

/// Enumerate **target time bins** starting from `k0`, bounded by `max_dt`.
///
/// If `include_same == false`, the enumeration starts at `k0 + 1`.
///
/// Arguments
/// ---------
/// * `tb` – Time binner.
/// * `k0` – Base time bin.
/// * `max_dt` – Inclusive time horizon after `k0` (days).
/// * `include_same` – Whether to include the same bin `k0`.
///
/// Return
/// ------
/// Iterator over `TimeBin` values: `k0 (+0|+1) .. k0 + ceil(max_dt / bin_width)`.
pub fn time_targets<Bt: TimeBinner + Sync>(
    tb: &Bt,
    k0: TimeBin,
    max_dt: f64,
    include_same: bool,
) -> impl Iterator<Item = TimeBin> {
    let w = tb.bin_width().max(1e-12);
    let max_steps = (max_dt / w).ceil().max(0.0) as i64;
    let start = if include_same { 0 } else { 1 };
    (start..=max_steps).map(move |dk| TimeBin(k0.0 + dk))
}
