//! # Python bindings: `FinkFatParams`
//!
//! This module exposes a **Python-friendly** configuration API for Fink-FAT,
//! wrapping the Rust [`FinkFatParams`] and its builder. The Python surface is
//! intentionally **flat** (no nested closures) to keep the API ergonomic in
//! notebooks and scripts.
//!
//! ## Overview
//! -----------
//! * [`PyFinkFatParams`] – an owning wrapper around the validated Rust
//!   [`FinkFatParams`]. Provides read-only getters, `validate()`, `to_dict()`,
//!   TOML helpers, and `__repr__`.
//! * [`PyFinkFatParamsBuilder`] – a fluent builder exposing **flat setters** only
//!   (binning/pairs/triplets/linking/global). Validation is delegated to Rust.
//!
//! ## Defaults
//! -----------
//! The defaults follow the Rust side. Call `PyFinkFatParams::default()` or use
//! `PyFinkFatParamsBuilder()` then `.build()`.

use std::mem;

use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyModule},
    Bound,
};

use crate::errors::ParamError;
use crate::params::FinkFatParams;
use crate::params::FinkFatParamsBuilder;

/// Convert an internal [`ParamError`] into a Python `ValueError`.
fn to_py_err(e: ParamError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/* ------------------------------ PyFinkFatParams ------------------------------ */

/// Python wrapper around the validated Rust [`FinkFatParams`].
///
/// This object owns the inner Rust configuration and exposes **read-only**
/// Python accessors for all scalar fields, TOML helpers, and a compact `repr`.
#[pyclass(name = "FinkFatParams")]
#[derive(Clone)]
pub struct PyFinkFatParams {
    pub(crate) inner: FinkFatParams,
}

#[pymethods]
impl PyFinkFatParams {
    /// Construct with Rust defaults (validate on creation).
    #[new]
    pub fn new() -> PyResult<Self> {
        let cfg = FinkFatParams::builder().build().map_err(to_py_err)?;
        Ok(Self { inner: cfg })
    }

    /// Return a builder to mutate settings before validation.
    #[staticmethod]
    pub fn builder() -> PyFinkFatParamsBuilder {
        PyFinkFatParamsBuilder {
            inner: FinkFatParamsBuilder::default(),
        }
    }

    /* ------------------------------- TOML I/O ------------------------------- */

    /// Build from a TOML string (validates).
    #[staticmethod]
    pub fn from_toml_str(s: &str) -> PyResult<Self> {
        let cfg = FinkFatParams::from_toml_str(s).map_err(to_py_err)?;
        Ok(Self { inner: cfg })
    }

    /// Dump to a TOML string (pretty).
    pub fn to_toml_str(&self) -> PyResult<String> {
        self.inner.to_toml_string_pretty().map_err(to_py_err)
    }

    /* ----------------------------- Global options --------------------------- */

    /// Whether to show progress bars.
    #[getter]
    pub fn show_progress(&self) -> bool {
        self.inner.show_progress
    }

    /* -------------------------------- Binning ------------------------------- */

    /// HEALPix depth (NSIDE = 2^depth).
    #[getter]
    pub fn healpix_depth(&self) -> u8 {
        self.inner.binning.healpix_depth
    }

    /// Time-bin width in **days (TT)**.
    #[getter]
    pub fn time_bin_width_days(&self) -> f64 {
        self.inner.binning.time_bin_width_days
    }

    /* --------------------------------- Pairs -------------------------------- */

    /// Max time difference for pairs (days, TT).
    #[getter]
    pub fn pair_max_dt(&self) -> f64 {
        self.inner.pairs.max_dt
    }

    /// Max great-circle separation for pairs (radians).
    #[getter]
    pub fn pair_max_sep(&self) -> f64 {
        self.inner.pairs.max_sep
    }

    /// Max photometric difference for pairs (dimensionless).
    #[getter]
    pub fn pair_max_flux_difference(&self) -> f32 {
        self.inner.pairs.max_flux_difference
    }

    /// Allow matching within the same time bin.
    #[getter]
    pub fn pair_allow_same_timebin(&self) -> bool {
        self.inner.pairs.allow_same_timebin
    }

    /* -------------------------------- Triplets ------------------------------ */

    /// Max neighbor Δt for triplets (days, TT).
    #[getter]
    pub fn triplet_max_dt_between(&self) -> f64 {
        self.inner.triplets.max_dt_between
    }

    /// Max per-pair separation in triplets (radians).
    #[getter]
    pub fn triplet_max_pair_sep(&self) -> f64 {
        self.inner.triplets.max_pair_sep
    }

    /// Max predicted residual at `c` when extrapolating `a→b` (radians).
    #[getter]
    pub fn triplet_max_predicted_residual(&self) -> f64 {
        self.inner.triplets.max_predicted_residual
    }

    /// Enforce strict time ordering `t(a) < t(b) < t(c)`.
    #[getter]
    pub fn triplet_enforce_time_order(&self) -> bool {
        self.inner.triplets.enforce_time_order
    }

    /// Max photometric difference for triplets (dimensionless).
    #[getter]
    pub fn triplet_max_flux_difference(&self) -> f32 {
        self.inner.triplets.max_flux_difference
    }

    /* ----------------------------- Linking: Predict ------------------------- */

    /// Predictor cone inflation `k_sigma` (dimensionless).
    #[getter]
    pub fn link_k_sigma(&self) -> f64 {
        self.inner.link.predict.k_sigma
    }

    /// Whether to pad by the spatial cell radius when forming cones.
    #[getter]
    pub fn link_pad_cell_radius(&self) -> bool {
        self.inner.link.predict.pad_cell_radius
    }

    /// Additive model noise (variance floor, rad²).
    #[getter]
    pub fn link_noise_q0(&self) -> f64 {
        self.inner.link.predict.noise.variance_floor
    }

    /// Additive model noise (linear drift per day, rad²/day).
    #[getter]
    pub fn link_noise_q1(&self) -> f64 {
        self.inner.link.predict.noise.drift_per_day
    }

    /// Additive model noise (quadratic curvature per day², rad²/day²).
    #[getter]
    pub fn link_noise_q2(&self) -> f64 {
        self.inner.link.predict.noise.curvature_per_day2
    }

    /* ------------------------------ Linking: Weights ------------------------ */

    #[getter]
    pub fn link_w_pos(&self) -> f64 {
        self.inner.link.scoring.weights.w_pos
    }
    #[getter]
    pub fn link_w_vel_dir(&self) -> f64 {
        self.inner.link.scoring.weights.w_vel_dir
    }
    #[getter]
    pub fn link_w_vel_norm(&self) -> f64 {
        self.inner.link.scoring.weights.w_vel_norm
    }
    #[getter]
    pub fn link_w_flux(&self) -> f64 {
        self.inner.link.scoring.weights.w_flux
    }
    #[getter]
    pub fn link_w_gap(&self) -> f64 {
        self.inner.link.scoring.weights.w_gap
    }
    #[getter]
    pub fn link_w_band_mismatch(&self) -> f64 {
        self.inner.link.scoring.weights.w_band_mismatch
    }

    /* ------------------------------- Linking: Gates ------------------------ */

    /// Gate on Mahalanobis distance (d² on positions).
    #[getter]
    pub fn link_max_d2_pos(&self) -> f64 {
        self.inner.link.scoring.gates.max_d2_pos
    }

    /// Gate on velocity direction mismatch (radians).
    #[getter]
    pub fn link_max_theta_vel(&self) -> f64 {
        self.inner.link.scoring.gates.max_theta_vel()
    }

    /// Gate on absolute speed difference (rad/day).
    #[getter]
    pub fn link_max_speed_diff(&self) -> f64 {
        self.inner.link.scoring.gates.max_speed_diff
    }

    /* ------------------------------ Linking: Scales ------------------------ */

    /// Angular scale used in costs (radians).
    #[getter]
    pub fn link_theta0(&self) -> f64 {
        self.inner.link.scoring.scales.theta0
    }

    /// Speed scale used in costs (rad/day).
    #[getter]
    pub fn link_v0(&self) -> f64 {
        self.inner.link.scoring.scales.v0
    }

    /// Flux sigma floor (dimensionless, cost term).
    #[getter]
    pub fn link_flux_sigma_floor(&self) -> f64 {
        self.inner.link.scoring.scales.flux_sigma_floor
    }

    /// Gap exponent ρ for Δ>1 penalty.
    #[getter]
    pub fn link_gap_rho(&self) -> f64 {
        self.inner.link.scoring.scales.gap_rho
    }

    /// Finite-difference step for j’s plane velocity (days).
    #[getter]
    pub fn link_vel_eps_days(&self) -> f64 {
        self.inner.link.scoring.scales.vel_eps_days
    }

    /* ------------------------------- Linking: Limits ----------------------- */

    /// Keep at most K edges per left node (Top-K).
    #[getter]
    pub fn link_top_k_per_left(&self) -> usize {
        self.inner.link.limits.top_k_per_left
    }

    /// Global cap on total edges (after Top-K); `None` disables it.
    #[getter]
    pub fn link_max_total_edges(&self) -> Option<usize> {
        self.inner.link.limits.max_total_edges
    }

    /// Hard cost cutoff; `None` disables it.
    #[getter]
    pub fn link_max_cost(&self) -> Option<f64> {
        self.inner.link.limits.max_cost
    }

    /// Optional hard cap on per-seed speed (rad/day).
    #[getter]
    pub fn link_max_speed_rad_per_day(&self) -> Option<f64> {
        self.inner.link.max_speed_rad_per_day
    }

    /* -------------------------------- Linking: Min Cost Flow ------------------------ */
    #[getter]
    pub fn link_mcf_lambda_start(&self) -> f64 {
        self.inner.link.mcf.lambda_start
    }

    #[getter]
    pub fn link_mcf_lambda_end(&self) -> f64 {
        self.inner.link.mcf.lambda_end
    }

    #[getter]
    pub fn link_mcf_gap_penalty_weight(&self) -> f64 {
        self.inner.link.mcf.gap_penalty_weight
    }

    #[getter]
    pub fn link_mcf_max_revisit_gap(&self) -> u32 {
        self.inner.link.mcf.max_revisit_gap
    }

    #[getter]
    pub fn link_mcf_max_total_flow(&self) -> Option<u32> {
        self.inner.link.mcf.max_total_flow
    }

    #[getter]
    pub fn link_mcf_horizon_nights(&self) -> usize {
        self.inner.link.mcf.horizon_nights
    }

    /* -------------------------------- Utilities ---------------------------- */

    /// Validate the entire parameter set.
    pub fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(to_py_err)
    }

    /// Convert scalar fields to a Python dict for logging/serialization.
    pub fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);

        // Global
        d.set_item("show_progress", self.show_progress())?;

        // Binning
        d.set_item("healpix_depth", self.healpix_depth())?;
        d.set_item("time_bin_width_days", self.time_bin_width_days())?;

        // Pairs
        d.set_item("pair_max_dt", self.pair_max_dt())?;
        d.set_item("pair_max_sep", self.pair_max_sep())?;
        d.set_item("pair_max_flux_difference", self.pair_max_flux_difference())?;
        d.set_item("pair_allow_same_timebin", self.pair_allow_same_timebin())?;

        // Triplets
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

        // Linking – predictor
        d.set_item("link_k_sigma", self.link_k_sigma())?;
        d.set_item("link_pad_cell_radius", self.link_pad_cell_radius())?;
        d.set_item("link_noise_q0", self.link_noise_q0())?;
        d.set_item("link_noise_q1", self.link_noise_q1())?;
        d.set_item("link_noise_q2", self.link_noise_q2())?;

        // Linking – weights
        d.set_item("link_w_pos", self.link_w_pos())?;
        d.set_item("link_w_vel_dir", self.link_w_vel_dir())?;
        d.set_item("link_w_vel_norm", self.link_w_vel_norm())?;
        d.set_item("link_w_flux", self.link_w_flux())?;
        d.set_item("link_w_gap", self.link_w_gap())?;
        d.set_item("link_w_band_mismatch", self.link_w_band_mismatch())?;

        // Linking – gates
        d.set_item("link_max_d2_pos", self.link_max_d2_pos())?;
        d.set_item("link_max_theta_vel", self.link_max_theta_vel())?;
        d.set_item("link_max_speed_diff", self.link_max_speed_diff())?;

        // Linking – scales
        d.set_item("link_theta0", self.link_theta0())?;
        d.set_item("link_v0", self.link_v0())?;
        d.set_item("link_flux_sigma_floor", self.link_flux_sigma_floor())?;
        d.set_item("link_gap_rho", self.link_gap_rho())?;
        d.set_item("link_vel_eps_days", self.link_vel_eps_days())?;

        // Linking – limits & optional cap
        d.set_item("link_top_k_per_left", self.link_top_k_per_left())?;
        d.set_item("link_max_total_edges", self.link_max_total_edges())?;
        d.set_item("link_max_cost", self.link_max_cost())?;
        d.set_item(
            "link_max_speed_rad_per_day",
            self.link_max_speed_rad_per_day(),
        )?;

        Ok(d.unbind())
    }

    /// Compact `repr` string for debugging.
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "FinkFatParams(healpix_depth={}, time_bin_width_days={}, pair_max_dt={}, pair_max_sep={}, triplet_max_dt_between={}, link.k_sigma={:.3}, ...)",
            self.healpix_depth(),
            self.time_bin_width_days(),
            self.pair_max_dt(),
            self.pair_max_sep(),
            self.triplet_max_dt_between(),
            self.link_k_sigma(),
        ))
    }
}

/* --------------------------- PyFinkFatParamsBuilder ------------------------- */

/// Fluent Python builder mirroring [`FinkFatParamsBuilder`] with flat setters.
///
/// Each setter returns `self` so you can chain calls and finish with `.build()`.
#[pyclass(name = "FinkFatParamsBuilder")]
pub struct PyFinkFatParamsBuilder {
    pub(crate) inner: FinkFatParamsBuilder,
}

impl Default for PyFinkFatParamsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyFinkFatParamsBuilder {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: FinkFatParamsBuilder::default(),
        }
    }

    /// Enable/disable progress bars.
    ///
    /// Returns
    /// -------
    /// self : PyFinkFatParamsBuilder
    ///     The same builder (for chaining).
    pub fn show_progress<'py>(mut slf: PyRefMut<'py, Self>, v: bool) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner); // move out safely (requires Default)
        slf.inner = inner.show_progress(v); // consume + return new builder
        slf // return PyRefMut<Self> for chaining
    }

    /* Binning */
    pub fn healpix_depth<'py>(mut slf: PyRefMut<'py, Self>, v: u8) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.healpix_depth(v);
        slf
    }
    pub fn time_bin_width_days<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.time_bin_width_days(v);
        slf
    }

    /* Pairs */
    pub fn pair_max_dt<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.pair_max_dt(v);
        slf
    }
    pub fn pair_max_sep<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.pair_max_sep(v);
        slf
    }
    pub fn pair_max_flux_difference<'py>(
        mut slf: PyRefMut<'py, Self>,
        v: f32,
    ) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.pair_max_flux_difference(v);
        slf
    }
    pub fn pair_allow_same_timebin<'py>(
        mut slf: PyRefMut<'py, Self>,
        yes: bool,
    ) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.pair_allow_same_timebin(yes);
        slf
    }

    /* Triplets */
    pub fn triplet_max_dt_between<'py>(
        mut slf: PyRefMut<'py, Self>,
        v: f64,
    ) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.triplet_max_dt_between(v);
        slf
    }
    pub fn triplet_max_pair_sep<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.triplet_max_pair_sep(v);
        slf
    }
    pub fn triplet_max_predicted_residual<'py>(
        mut slf: PyRefMut<'py, Self>,
        v: f64,
    ) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.triplet_max_predicted_residual(v);
        slf
    }
    pub fn triplet_enforce_time_order<'py>(
        mut slf: PyRefMut<'py, Self>,
        yes: bool,
    ) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.triplet_enforce_time_order(yes);
        slf
    }
    pub fn triplet_max_flux_difference<'py>(
        mut slf: PyRefMut<'py, Self>,
        v: f32,
    ) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.triplet_max_flux_difference(v);
        slf
    }

    /* Linking – predictor */
    pub fn link_k_sigma<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_k_sigma(v);
        slf
    }
    pub fn link_pad_cell_radius<'py>(
        mut slf: PyRefMut<'py, Self>,
        yes: bool,
    ) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_pad_cell_radius(yes);
        slf
    }
    pub fn link_noise_q0<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_noise_q0(v);
        slf
    }
    pub fn link_noise_q1<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_noise_q1(v);
        slf
    }
    pub fn link_noise_q2<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_noise_q2(v);
        slf
    }

    /* Linking – weights */
    pub fn link_w_pos<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_w_pos(v);
        slf
    }
    pub fn link_w_vel_dir<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_w_vel_dir(v);
        slf
    }
    pub fn link_w_vel_norm<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_w_vel_norm(v);
        slf
    }
    pub fn link_w_flux<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_w_flux(v);
        slf
    }
    pub fn link_w_gap<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_w_gap(v);
        slf
    }
    pub fn link_w_band_mismatch<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_w_band_mismatch(v);
        slf
    }

    /* Linking – gates */
    pub fn link_max_d2_pos<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_max_d2_pos(v);
        slf
    }
    pub fn link_max_theta_vel<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_max_theta_vel(v);
        slf
    }
    pub fn link_max_speed_diff<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_max_speed_diff(v);
        slf
    }

    /* Linking – scales */
    pub fn link_theta0<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_theta0(v);
        slf
    }
    pub fn link_v0<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_v0(v);
        slf
    }
    pub fn link_flux_sigma_floor<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_flux_sigma_floor(v);
        slf
    }
    pub fn link_gap_rho<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_gap_rho(v);
        slf
    }
    pub fn link_vel_eps_days<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_vel_eps_days(v);
        slf
    }

    /* Linking – limits & optional cap */
    pub fn link_top_k_per_left<'py>(mut slf: PyRefMut<'py, Self>, v: usize) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_top_k_per_left(v);
        slf
    }
    pub fn link_max_total_edges<'py>(
        mut slf: PyRefMut<'py, Self>,
        v: Option<usize>,
    ) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_max_total_edges(v);
        slf
    }
    pub fn link_clear_max_total_edges<'py>(mut slf: PyRefMut<'py, Self>) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_clear_max_total_edges();
        slf
    }
    pub fn link_max_cost<'py>(mut slf: PyRefMut<'py, Self>, v: Option<f64>) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_max_cost(v);
        slf
    }
    pub fn link_clear_max_cost<'py>(mut slf: PyRefMut<'py, Self>) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_clear_max_cost();
        slf
    }
    pub fn link_max_speed_rad_per_day<'py>(
        mut slf: PyRefMut<'py, Self>,
        v: Option<f64>,
    ) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_max_speed_rad_per_day(v);
        slf
    }

    /* Linking – Min cost flow */
    pub fn link_mcf_lambda_start<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_mcf_lambda_start(v);
        slf
    }
    pub fn link_mcf_lambda_end<'py>(mut slf: PyRefMut<'py, Self>, v: f64) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_mcf_lambda_end(v);
        slf
    }
    pub fn link_mcf_gap_penalty_weight<'py>(
        mut slf: PyRefMut<'py, Self>,
        v: f64,
    ) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_mcf_gap_penalty_weight(v);
        slf
    }
    pub fn link_mcf_max_revisit_gap<'py>(
        mut slf: PyRefMut<'py, Self>,
        v: u32,
    ) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_mcf_max_revisit_gap(v);
        slf
    }
    pub fn link_mcf_max_total_flow<'py>(
        mut slf: PyRefMut<'py, Self>,
        v: Option<u32>,
    ) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_mcf_max_total_flow(v);
        slf
    }
    pub fn link_mcf_horizon_nights<'py>(
        mut slf: PyRefMut<'py, Self>,
        v: usize,
    ) -> PyRefMut<'py, Self> {
        let inner = mem::take(&mut slf.inner);
        slf.inner = inner.link_mcf_horizon_nights(v);
        slf
    }

    /// Validate and return an owning configuration.
    ///
    /// Note: this borrows `self` (no move). We clone the Rust builder under the hood.
    pub fn build(&self) -> PyResult<PyFinkFatParams> {
        // Ensure `FinkFatParamsBuilder: Clone`
        let inner = self.inner.clone().build().map_err(to_py_err)?;
        Ok(PyFinkFatParams { inner })
    }
}

/* ------------------------------- Registration ------------------------------- */

/// Register [`PyFinkFatParams`] and [`PyFinkFatParamsBuilder`] to Python.
pub fn register_params_module(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyFinkFatParams>()?;
    m.add_class::<PyFinkFatParamsBuilder>()?;
    Ok(())
}
