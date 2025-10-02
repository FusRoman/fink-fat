use pyo3::prelude::*;

pub mod alerts;
pub(crate) mod progress;
pub mod seeding;

/// A Python module implemented in Rust.
#[pymodule]
fn fink_fat(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<alerts::Alert>()?;
    m.add_class::<alerts::AlertStore>()?;

    Ok(())
}
