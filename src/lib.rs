use pyo3::prelude::*;

use crate::params::params_binding::register_params_module;

pub mod alerts;
pub mod errors;
pub mod params;
pub(crate) mod progress;
pub mod seeding;

/// Strong-typed aliases (adapt to your real types).
pub type MjdTt = f64; // days (TT)
pub type Radians = f64; // radians

/// A Python module implemented in Rust.
#[pymodule]
fn fink_fat(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<alerts::Alert>()?;
    m.add_class::<alerts::AlertStore>()?;

    register_params_module(m)?;

    Ok(())
}
