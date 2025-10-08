use pyo3::pyclass;

use crate::{
    alerts::{Alert, AlertStore},
    AlertId,
};

pub mod geometrical_seeding;
pub mod healpix_binners;
pub mod space_time_bucket;
pub mod uniform_time_binner;

/// A seed made of two detections (t_b > t_a).
#[pyclass(module = "fink_fat")]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Pair {
    pub a: AlertId,
    pub b: AlertId,
}
pub type Pairs = Vec<Pair>;

/// A seed made of three detections (t_a < t_b < t_c).
#[pyclass(module = "fink_fat")]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Triplet {
    pub a: AlertId,
    pub b: AlertId,
    pub c: AlertId,
}
pub type Triplets = Vec<Triplet>;

/* ---------- Backward compatibility (tuples <-> wrappers) ---------- */

impl From<(AlertId, AlertId)> for Pair {
    #[inline]
    fn from(t: (AlertId, AlertId)) -> Self {
        Self { a: t.0, b: t.1 }
    }
}
impl From<Pair> for (AlertId, AlertId) {
    #[inline]
    fn from(p: Pair) -> Self {
        (p.a, p.b)
    }
}

impl From<(AlertId, AlertId, AlertId)> for Triplet {
    #[inline]
    fn from(t: (AlertId, AlertId, AlertId)) -> Self {
        Self {
            a: t.0,
            b: t.1,
            c: t.2,
        }
    }
}
impl From<Triplet> for (AlertId, AlertId, AlertId) {
    #[inline]
    fn from(t: Triplet) -> Self {
        (t.a, t.b, t.c)
    }
}

/* ---------------------- Convenience resolvers --------------------- */

impl Pair {
    /// Resolve to borrowed alerts (checked).
    ///
    /// Return
    /// ------
    /// `Some((&Alert, &Alert))` if both ids are in bounds, otherwise `None`.
    #[inline]
    pub fn resolve(self, store: &AlertStore) -> Option<(&Alert, &Alert)> {
        let a = store.alerts.get(self.a.idx())?;
        let b = store.alerts.get(self.b.idx())?;
        Some((a, b))
    }

    /// Resolve with a debug bound check and then unchecked indexing (fast path).
    #[inline]
    pub fn resolve_fast(self, store: &AlertStore) -> (&Alert, &Alert) {
        debug_assert!(self.a.idx() < store.alerts.len());
        debug_assert!(self.b.idx() < store.alerts.len());
        // Safe after asserts:
        let a = unsafe { store.alerts.get_unchecked(self.a.idx()) };
        let b = unsafe { store.alerts.get_unchecked(self.b.idx()) };
        (a, b)
    }
}

impl Triplet {
    /// Resolve to borrowed alerts (checked).
    #[inline]
    pub fn resolve(self, store: &AlertStore) -> Option<(&Alert, &Alert, &Alert)> {
        Some((
            store.alerts.get(self.a.idx())?,
            store.alerts.get(self.b.idx())?,
            store.alerts.get(self.c.idx())?,
        ))
    }

    /// Resolve with a debug bound check and then unchecked indexing (fast path).
    #[inline]
    pub fn resolve_fast(self, store: &AlertStore) -> (&Alert, &Alert, &Alert) {
        debug_assert!(self.a.idx() < store.alerts.len());
        debug_assert!(self.b.idx() < store.alerts.len());
        debug_assert!(self.c.idx() < store.alerts.len());
        unsafe {
            (
                store.alerts.get_unchecked(self.a.idx()),
                store.alerts.get_unchecked(self.b.idx()),
                store.alerts.get_unchecked(self.c.idx()),
            )
        }
    }
}
