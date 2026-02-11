use crate::Radian;
use std::fmt::Debug;

/// Compact spatial cell identifier.
///
/// Typically the output of a sky partitioner (e.g., HEALPix, HTM, or a lon/lat grid).
/// Stores the cell id as an unsigned 64-bit integer to accommodate deep tessellations.
///
/// ### Notes
/// - The specific **encoding** depends on the `SpatialBinner` implementation.
/// - Comparable and hashable to serve as a map key.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct SpatialKey(pub u64);

/// Spatial binning interface.
///
/// Implement this for your sky partitioner (HEALPix, HTM, lon/lat grid…).
pub trait SpatialBinner: Sync {
    /// Return the spatial cell for a given sky position.
    ///
    /// Parameters
    /// ----------
    /// - `ra`: Right ascension (radians).
    /// - `dec`: Declination (radians).
    ///
    /// Return
    /// ------
    /// `SpatialKey` – the spatial cell id covering `(ra, dec)`.
    fn key_for(&self, ra: Radian, dec: Radian) -> SpatialKey;

    /// Write neighbor keys into `out` (which is cleared by the callee).
    ///
    /// This avoids allocating a new Vec on every query.
    fn neighbors_into(&self, key: SpatialKey, ang_radius: Radian, out: &mut Vec<SpatialKey>);

    /// Enumerate neighbor cells needed to cover an **angular radius** around `key`.
    ///
    /// The radius is in **radians** and typically chosen as a small multiple of the
    /// cell's characteristic scale (see [`cell_radius`](crate::seeding::space_time_bucket::SpatialBinner::cell_radius)).
    ///
    /// Notes
    /// -----
    /// Implementations usually **include `key` itself** in the returned list,
    /// but callers should not rely on this unless documented by the concrete type.
    #[inline]
    fn neighbors(&self, key: SpatialKey, ang_radius: Radian) -> Vec<SpatialKey> {
        let mut out = Vec::new();
        self.neighbors_into(key, ang_radius, &mut out);
        out
    }

    /// Characteristic angular **radius** for a single cell (radians).
    ///
    /// This can drive the choice of neighbor coverage (e.g., `k × cell_radius()`).
    fn cell_radius(&self) -> Radian;
}
