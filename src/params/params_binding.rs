//! # Python bindings: `FinkFatParams`
//!
//! This module exposes a **Python-friendly** configuration API for Fink-FAT,
//! wrapping the Rust [`FinkFatParams`] and its builder. The Python surface is
//! intentionally **flat** (no closure-based nested setters) to keep the API
//! stable and ergonomic in notebooks and scripts.
//!
//! ## Overview
//! -----------
//! * [`PyFinkFatParams`] – an owning wrapper around the validated Rust
//!   [`FinkFatParams`]. Provides read-only getters, `validate()`, `to_dict()`,
//!   and `__repr__`.
//! * [`PyFinkFatParamsBuilder`] – a fluent builder exposing **flat setters** only
//!   (binning/pairs/triplets/global). It reuses all validation logic from Rust.
//!
//! ## Defaults
//! -----------
//! Tuned for LSST-like cadence:
//! * Binning: `healpix_depth = 10`, `time_bin_width_days = 0.02` (~28.8 min)
//! * Pairs: `max_dt = 0.06 d`, `max_sep = 0.003 rad`, `max_flux_difference = 5.0`
//! * Triplets: `max_dt_between = 0.04 d`, `max_pair_sep = 0.0025 rad`,
//!   `max_predicted_residual = 8e-4 rad`
//! * Global: `show_progress = false`
//!
//! ## Errors
//! ---------
//! Any invalid parameter (non-finite/negative times or angles, inconsistent
//! tolerances, out-of-range HEALPix depth) results in a Rust [`ParamError`],
//! converted here into a Python `ValueError` with a clear message.
//!
//! ## See also
//! -----------
//! * Rust configuration types: [`FinkFatParams`], [`FinkFatParamsBuilder`],
//!   and the sub-groups (`BinningParams`, `PairParams`, `TripletParams`).

#![allow(clippy::needless_pass_by_value)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::errors::ParamError;
use crate::params::{FinkFatParams, FinkFatParamsBuilder};

/// Convert an internal [`ParamError`] into a Python `ValueError`.
///
/// Arguments
/// ---------
/// * `e` – Rust parameter validation error.
///
/// Return
/// ------
/// * `PyErr` – a `ValueError` carrying the error string.
fn to_py_err(e: ParamError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/* ------------------------------ PyFinkFatParams ------------------------------ */

/// Python wrapper around the validated Rust [`FinkFatParams`].
///
/// This object owns the inner Rust configuration and exposes **read-only**
/// Python accessors for all scalar fields, a `validate()` method, a compact
/// `__repr__`, and a convenience `to_dict()` for logging/serialization.
///
/// Overview
/// --------
/// * Construct via [`PyFinkFatParams::default`] or the builder
///   [`PyFinkFatParams::builder`].
/// * All getters are **read-only**; use the builder to create new configs.
/// * `validate()` raises `ValueError` on invalid combinations.
#[pyclass(module = "fink_fat")]
#[derive(Clone)]
pub struct PyFinkFatParams {
    pub(crate) inner: FinkFatParams,
}

#[pymethods]
impl PyFinkFatParams {
    /// Return LSST-like defaults.
    ///
    /// Return
    /// ------
    /// * `PyFinkFatParams` – configuration object with project defaults.
    ///
    /// Example (Python)
    /// ----------------
    /// ```python
    /// from fink_fat import FinkFatParams
    /// p = FinkFatParams.default()
    /// ```
    #[staticmethod]
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self {
            inner: FinkFatParams::default(),
        }
    }

    /// Create a fluent builder (flat setters only).
    ///
    /// Notes
    /// -----
    /// The Python API intentionally **does not** expose nested closure setters.
    /// Use the provided flat setters to configure all fields.
    #[staticmethod]
    pub fn builder() -> PyFinkFatParamsBuilder {
        PyFinkFatParamsBuilder {
            inner: FinkFatParamsBuilder::default(),
        }
    }

    /* ----------------------------- Read-only getters ---------------------------- */

    /// Whether to display progress bars in seeding/linking stages.
    #[getter]
    pub fn show_progress(&self) -> bool {
        self.inner.show_progress
    }

    /// HEALPix depth (NSIDE = 2^depth), valid range: 0..=29.
    #[getter]
    pub fn healpix_depth(&self) -> u8 {
        self.inner.binning.healpix_depth
    }

    /// Temporal bucket width (days, TT). Must be strictly positive.
    #[getter]
    pub fn time_bin_width_days(&self) -> f64 {
        self.inner.binning.time_bin_width_days
    }

    /// Pair: maximum Δt between alerts (days, TT).
    #[getter]
    pub fn pair_max_dt(&self) -> f64 {
        self.inner.pairs.max_dt
    }

    /// Pair: maximum angular separation (radians).
    #[getter]
    pub fn pair_max_sep(&self) -> f64 {
        self.inner.pairs.max_sep
    }

    /// Pair: maximum photometric difference (dimensionless; flux or Δmag proxy).
    #[getter]
    pub fn pair_max_flux_difference(&self) -> f32 {
        self.inner.pairs.max_flux_difference
    }

    /// Pair: whether alerts inside the same time bin can form a pair.
    #[getter]
    pub fn pair_allow_same_timebin(&self) -> bool {
        self.inner.pairs.allow_same_timebin
    }

    /// Triplet: maximum Δt between consecutive neighbors (days, TT).
    #[getter]
    pub fn triplet_max_dt_between(&self) -> f64 {
        self.inner.triplets.max_dt_between
    }

    /// Triplet: maximum neighbor angular separation (radians).
    #[getter]
    pub fn triplet_max_pair_sep(&self) -> f64 {
        self.inner.triplets.max_pair_sep
    }

    /// Triplet: maximum predicted residual at `c` when extrapolating `a→b` (radians).
    #[getter]
    pub fn triplet_max_predicted_residual(&self) -> f64 {
        self.inner.triplets.max_predicted_residual
    }

    /// Triplet: enforce strict time ordering `t(a) < t(b) < t(c)`.
    #[getter]
    pub fn triplet_enforce_time_order(&self) -> bool {
        self.inner.triplets.enforce_time_order
    }

    /// Triplet: maximum photometric difference (dimensionless; flux or Δmag proxy).
    #[getter]
    pub fn triplet_max_flux_difference(&self) -> f32 {
        self.inner.triplets.max_flux_difference
    }

    /// Convert scalar fields to a Python dict.
    ///
    /// Return
    /// ------
    /// * `dict[str, float|bool|int]` – simple mapping of scalar settings.
    ///
    /// Notes
    /// -----
    /// Complex/nested fields (if any are added in the future) are not included.
    pub fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("show_progress", self.show_progress())?;
        d.set_item("healpix_depth", self.healpix_depth())?;
        d.set_item("time_bin_width_days", self.time_bin_width_days())?;
        d.set_item("pair_max_dt", self.pair_max_dt())?;
        d.set_item("pair_max_sep", self.pair_max_sep())?;
        d.set_item("pair_max_flux_difference", self.pair_max_flux_difference())?;
        d.set_item("pair_allow_same_timebin", self.pair_allow_same_timebin())?;
        d.set_item("triplet_max_dt_between", self.triplet_max_dt_between())?;
        d.set_item("triplet_max_pair_sep", self.triplet_max_pair_sep())?;
        d.set_item(
            "triplet_max_predicted_residual",
            self.triplet_max_predicted_residual(),
        )?;
        d.set_item(
            "triplet_enforce_time_order",
            self.triplet_enforce_time_order(),
        )?;
        d.set_item(
            "triplet_max_flux_difference",
            self.triplet_max_flux_difference(),
        )?;
        Ok(d.into())
    }

    /// Compact string representation (for logging / debugging).
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "FinkFatParams(show_progress={}, depth={}, dt_days={:.6}, \
pair_max_dt={:.6}, pair_max_sep={:.6}, pair_max_flux_diff={:.3}, same_bin={}, \
trip_max_dt_between={:.6}, trip_max_pair_sep={:.6}, trip_max_pred_resid={:.6}, \
time_order={}, trip_max_flux_diff={:.3})",
            self.show_progress(),
            self.healpix_depth(),
            self.time_bin_width_days(),
            self.pair_max_dt(),
            self.pair_max_sep(),
            self.pair_max_flux_difference(),
            self.pair_allow_same_timebin(),
            self.triplet_max_dt_between(),
            self.triplet_max_pair_sep(),
            self.triplet_max_predicted_residual(),
            self.triplet_enforce_time_order(),
            self.triplet_max_flux_difference(),
        ))
    }
}

/* --------------------------- PyFinkFatParamsBuilder --------------------------- */

/// Python wrapper over the Rust [`FinkFatParamsBuilder`].
///
/// This builder exposes **flat setters** only, mapping 1:1 to the Rust builder’s
/// flat API. It applies Rust-side validation upon `build()`, raising a Python
/// `ValueError` when constraints are not satisfied.
#[pyclass(module = "fink_fat")]
pub struct PyFinkFatParamsBuilder {
    pub(crate) inner: FinkFatParamsBuilder,
}

#[pymethods]
impl PyFinkFatParamsBuilder {
    /// Set whether to show progress bars during seeding/linking.
    ///
    /// Arguments
    /// ---------
    /// * `v` – `bool`
    pub fn show_progress<'py>(mut slf: PyRefMut<'py, Self>, v: bool) -> PyRefMut<'py, Self> {
        let inner = std::mem::take(&mut slf.inner).show_progress(v);
        slf.inner = inner;
        slf
    }

    /* ------------------------- Binning setters ------------------------- */

    /// Set HEALPix depth (NSIDE = 2^depth).
    ///
    /// Arguments
    /// ---------
    /// * `v` – `int` in **0..=29**
    pub fn healpix_depth<'py>(mut slf: PyRefMut<'py, Self>, v: u8) -> PyRefMut<'py, Self> {
        let inner = std::mem::take(&mut slf.inner).healpix_depth(v);
        slf.inner = inner;
        slf
    }

    /// Set the time bin width (days, TT).
    ///
    /// Arguments
    /// ---------
    /// * `v` – `float` strictly **> 0**
    pub fn time_bin_width_days<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        let inner = std::mem::take(&mut slf.inner).time_bin_width_days(v);
        slf.inner = inner;
        slf
    }

    /* --------------------------- Pair setters -------------------------- */

    /// Set pair maximum Δt (days, TT).
    ///
    /// Arguments
    /// ---------
    /// * `v` – `float` ≥ 0 and finite
    pub fn pair_max_dt<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        slf.inner = std::mem::take(&mut slf.inner).pair_max_dt(v);
        slf
    }

    /// Set pair maximum angular separation (radians).
    ///
    /// Arguments
    /// ---------
    /// * `v` – `float` ≥ 0 and finite
    pub fn pair_max_sep<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        slf.inner = std::mem::take(&mut slf.inner).pair_max_sep(v);
        slf
    }

    /// Set pair maximum photometric difference (dimensionless).
    ///
    /// Arguments
    /// ---------
    /// * `v` – `float` ≥ 0 and finite
    pub fn pair_max_flux_difference<'py>(
        mut slf: PyRefMut<'py, Self>,
        v: f32,
    ) -> PyRefMut<'py, Self> {
        slf.inner = std::mem::take(&mut slf.inner).pair_max_flux_difference(v);
        slf
    }

    /// Allow/disallow pairing within the same time bin.
    ///
    /// Arguments
    /// ---------
    /// * `v` – `bool`
    pub fn pair_allow_same_timebin<'py>(
        mut slf: PyRefMut<'py, Self>,
        v: bool,
    ) -> PyRefMut<'py, Self> {
        slf.inner = std::mem::take(&mut slf.inner).pair_allow_same_timebin(v);
        slf
    }

    /* ------------------------- Triplet setters ------------------------- */

    /// Set maximum Δt between consecutive neighbors (days, TT).
    ///
    /// Arguments
    /// ---------
    /// * `v` – `float` ≥ 0 and finite
    pub fn triplet_max_dt_between<'py>(
        mut slf: PyRefMut<'py, Self>,
        v: f64,
    ) -> PyRefMut<'py, Self> {
        slf.inner = std::mem::take(&mut slf.inner).triplet_max_dt_between(v);
        slf
    }

    /// Set maximum neighbor angular separation (radians).
    ///
    /// Arguments
    /// ---------
    /// * `v` – `float` ≥ 0 and finite
    pub fn triplet_max_pair_sep<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        slf.inner = std::mem::take(&mut slf.inner).triplet_max_pair_sep(v);
        slf
    }

    /// Set maximum predicted residual at `c` (radians) when extrapolating `a→b`.
    ///
    /// Arguments
    /// ---------
    /// * `v` – `float` ≥ 0 and finite
    pub fn triplet_max_predicted_residual<'py>(
        mut slf: PyRefMut<'py, Self>,
        v: f64,
    ) -> PyRefMut<'py, Self> {
        slf.inner = std::mem::take(&mut slf.inner).triplet_max_predicted_residual(v);
        slf
    }

    /// Enforce strict time ordering: `t(a) < t(b) < t(c)`.
    ///
    /// Arguments
    /// ---------
    /// * `v` – `bool`
    pub fn triplet_enforce_time_order<'py>(
        mut slf: PyRefMut<'py, Self>,
        v: bool,
    ) -> PyRefMut<'py, Self> {
        slf.inner = std::mem::take(&mut slf.inner).triplet_enforce_time_order(v);
        slf
    }

    /// Set triplet maximum photometric difference (dimensionless).
    ///
    /// Arguments
    /// ---------
    /// * `v` – `float` ≥ 0 and finite
    pub fn triplet_max_flux_difference<'py>(
        mut slf: PyRefMut<'py, Self>,
        v: f32,
    ) -> PyRefMut<'py, Self> {
        slf.inner = std::mem::take(&mut slf.inner).triplet_max_flux_difference(v);
        slf
    }

    /* ------------------------------ Build ------------------------------ */

    /// Build a validated `FinkFatParams`. Raises `ValueError` on invalid combo.
    ///
    /// Return
    /// ------
    /// * `PyFinkFatParams` – owning wrapper over the validated Rust config.
    ///
    /// Errors
    /// ------
    /// * `ValueError` if validation fails.
    pub fn build(&mut self) -> PyResult<PyFinkFatParams> {
        let built = std::mem::take(&mut self.inner).build().map_err(to_py_err)?;
        Ok(PyFinkFatParams { inner: built })
    }

    /// Minimal builder summary.
    fn __repr__(&self) -> PyResult<String> {
        Ok("FinkFatParamsBuilder(...)".to_string())
    }
}

/* ----------------------- Module registration helper ----------------------- */

/// Register parameter bindings into the Python module.
///
/// This is called from the crate’s Python entry point to expose
/// [`PyFinkFatParams`] and [`PyFinkFatParamsBuilder`] to Python.
///
/// Arguments
/// ---------
/// * `m` – destination Python module.
///
/// Return
/// ------
/// * `PyResult<()>` – `Ok(())` on success.
pub fn register_params_module(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyFinkFatParams>()?;
    m.add_class::<PyFinkFatParamsBuilder>()?;
    Ok(())
}
