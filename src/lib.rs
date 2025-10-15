use pyo3::prelude::*;

#[cfg(feature = "python-extension")]
use crate::propagation::flow::FlowUpdate;
#[cfg(feature = "python-extension")]
use crate::propagation::linking::RollingLinkState;
#[cfg(feature = "python-extension")]
use crate::propagation::linking_flow::RollingFlowState;

pub mod alerts;
pub mod errors;
pub mod params;
pub(crate) mod progress;
pub mod seeding;

pub mod propagation;

pub mod track_registry;

/// Strong-typed aliases (adapt to your real types).
pub type MjdTt = f64; // days (TT)
pub type Radians = f64; // radians

/// Night identifier (monotonic).
pub type NightId = u32;

/// Dense identifier used to index into `AlertStore::alerts`.
///
/// Why a newtype?
/// --------------
/// A `type` alias like `type AlertId = u32` reads as a primitive everywhere,
/// which makes signatures harder to understand and refactors brittle. A
/// newtype gives:
/// - type safety (no accidental mix with other integers),
/// - discoverable methods (`id.idx()`),
/// - trait impls (`Display`, `From<usize>`, etc.).
#[pyclass(module = "fink_fat")]
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AlertId(pub u32);

impl AlertId {
    /// Convert to a slice index (panics in debug on overflow).
    #[inline]
    pub fn idx(self) -> usize {
        self.0 as usize
    }
}

impl From<usize> for AlertId {
    #[inline]
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}
impl From<u32> for AlertId {
    #[inline]
    fn from(v: u32) -> Self {
        Self(v)
    }
}
impl From<AlertId> for usize {
    #[inline]
    fn from(v: AlertId) -> usize {
        v.0 as usize
    }
}

/// A Python module implemented in Rust.
#[cfg(feature = "python-extension")]
#[pymodule]
fn fink_fat(m: &Bound<'_, PyModule>) -> PyResult<()> {
    use crate::{
        params::params_binding::register_params_module, track_registry::DetectConflictPolicy,
    };

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<alerts::Alert>()?;
    m.add_class::<alerts::AlertStore>()?;
    m.add_class::<RollingLinkState>()?;

    m.add_class::<RollingFlowState>()?;
    m.add_class::<FlowUpdate>()?;

    m.add_class::<DetectConflictPolicy>()?;

    register_params_module(m)?;

    Ok(())
}
