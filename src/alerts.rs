//! # Alert data model and seeding bridge (Python bindings)
//!
//! This module exposes a minimal **alert data model** to Python and the
//! **intra-night seeding pipeline** entry points.
//!
//! ## Overview
//!
//! - [`Alert`] is a lightweight immutable record (exposed to Python) holding
//!   the per-detection astrometry, photometry and metadata.
//! - [`AlertStore`] is a contiguous vector of [`Alert`] indexed by
//!   [`AlertId`], plus a `start_mjd` anchor used for time binning.
//! - From Python, you can construct an [`AlertStore`] directly from NumPy
//!   arrays via [`AlertStore::from_numpy`], run the **pair/triplet seeding**
//!   pipeline with [`AlertStore::generate_seeds`], and build stable,
//!   human-readable link identifiers with [`AlertStore::build_link_uids_dict`].
//!
//! ## Units & Conventions
//!
//! - `ra`, `dec` are in **radians** (ICRS, J2000).
//! - `mjd_tt` is **Modified Julian Date in TT (Terrestrial Time)**.
//! - `flux` is PSF **difference** flux (e.g., nJy), and `flux_err` its error.
//! - `band` is an **integer** photometric band code.
//! - [`AlertId`] is a **0-based** dense integer index into the `alerts` vector.
//!
//! ## Performance notes
//!
//! - [`AlertStore::from_numpy`] performs a single pass over zero-copied slices
//!   (no Python allocation in the loop). Data is **copied once** into a Rust
//!   `Vec<Alert>` to enable contiguous access and safe parallel iteration later.
//! - The seeding routines can stream progress bars via `indicatif` if requested
//!   in [`PyFinkFatParams`]; otherwise there is no UI overhead.

use std::{collections::BTreeSet, fmt};

use itertools::izip;
use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyList},
    Py, PyAny, PyErr, PyResult, Python,
};

use numpy::PyReadonlyArray1;

use crate::{
    params::{params_binding::PyFinkFatParams, FinkFatParams},
    progress::{make_bar, make_multi_progress},
    propagation::{
        features::{extract_pair_features, extract_triplet_features, SeedNode},
        linking::NightSnapshot,
    },
    seeding::{
        geometrical_seeding::{
            generate_pairs, generate_pairs_with_progress, generate_triplets_from_pairs,
            generate_triplets_from_pairs_with_progress,
        },
        healpix_binners::HealpixBinner,
        space_time_bucket::{
            build_index_from_alerts_precise, build_index_from_alerts_precise_with_progress,
        },
        uniform_time_binner::UniformTimeBinner,
        Pair, Pairs, Triplet, Triplets,
    },
    AlertId, NightId,
};

/// Single detection in the alert stream.
///
/// This record is intentionally compact and cloneable so it can be moved across
/// threads and sent back to Python when needed.
///
/// Fields
/// ------
/// - `id` – [`AlertId`] assigned on ingestion; indexes `alerts[id as usize]`.
/// - `dia_source_id` – LSST `diaSourceId` (stable, 64-bit).
/// - `ra`, `dec` – ICRS coordinates in **radians**.
/// - `ra_err`, `dec_err` – 1-sigma uncertainties on `ra` and `dec` in **radians**.
/// - `mjd_tt` – **MJD (TT)** timestamp of the detection.
/// - `flux`, `flux_err` – PSF **difference** flux and its uncertainty (units depend on upstream).
/// - `band` – integer photometric band code.
#[pyclass(module = "fink_fat")]
#[derive(Clone, Debug, Default)]
pub struct Alert {
    #[pyo3(get)]
    pub id: AlertId,
    #[pyo3(get)]
    pub dia_source_id: u64, // from LSST
    #[pyo3(get)]
    pub ra: f64, // rad
    #[pyo3(get)]
    pub ra_err: f64, // rad
    #[pyo3(get)]
    pub dec: f64, // rad
    #[pyo3(get)]
    pub dec_err: f64, // rad
    #[pyo3(get)]
    pub mjd_tt: f64, // days (TT)
    #[pyo3(get)]
    pub flux: f32, // psf flux (nJy), note: it is a flux difference between template image and the visit image
    #[pyo3(get)]
    pub flux_err: f32, // psf flux error
    #[pyo3(get)]
    pub band: u8, // photometric band
}

/* ------------------------ Display / Debug ------------------------- */

impl fmt::Display for Alert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Compact single-line rendering for logs and __str__.
        write!(
            f,
            "Alert(id={}, dia_source_id={}, ra={:.6} rad, dec={:.6} rad, mjd_tt={:.5}, flux={:.3}±{:.3} nJy, band={})",
            self.id.idx(), self.dia_source_id, self.ra, self.dec, self.mjd_tt, self.flux, self.flux_err, self.band
        )
    }
}

/* --------------------------- Python API --------------------------- */

#[pymethods]
impl Alert {
    /// Return a compact string representation (Python `str(alert)`).
    ///
    /// Examples
    /// --------
    /// >>> str(alert)  # doctest: +SKIP
    /// "Alert(id=42, dia_source_id=..., ra=..., dec=..., ...)"
    fn __str__(&self) -> String {
        format!("{}", self)
    }

    /// Return a detailed representation (Python `repr(alert)`).
    ///
    /// Notes
    /// -----
    /// This is more verbose than `__str__` and includes all scalar fields.
    fn __repr__(&self) -> String {
        format!(
            "Alert(id={}, dia_source_id={}, ra={:.6}, dec={:.6}, mjd_tt={:.5}, flux={:.3}, flux_err={:.3}, band={})",
            self.id.idx(), self.dia_source_id, self.ra, self.dec, self.mjd_tt, self.flux, self.flux_err, self.band
        )
    }
}

/// Contiguous store of alerts for (typically) a single night.
///
/// The vector `alerts` is indexed by [`AlertId`] (0-based) and provides
/// cache-friendly iteration for the seeding pipeline.
///
/// Design
/// ------
/// - `start_mjd` is the **floor** of the minimum `mjd_tt` in the store and
///   serves as origin for uniform time binning.
/// - Alerts are immutable after construction to simplify sharing across threads.
///
/// Python
/// ------
/// Instances are exposed in `fink_fat.AlertStore`.
#[pyclass(module = "fink_fat")]
#[derive(Debug, Clone)]
pub struct AlertStore {
    /// Night anchor (TT): floor of the minimum `mjd_tt` in `alerts`.
    pub start_mjd: f64,
    /// All alerts, densely indexed by [`AlertId`].
    pub alerts: Vec<Alert>,
}

impl fmt::Display for AlertStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Lightweight summary without iterating the alerts (no GIL needed).
        write!(
            f,
            "AlertStore(n_alerts={}, start_mjd={:.5})",
            self.alerts.len(),
            self.start_mjd
        )
    }
}

impl AlertStore {
    /// Build a **snapshot** for one night from alerts by running the pure-Rust seeding
    /// (no UI) then extracting features.
    ///
    /// Use this for the **current** night N+1 right before linking, and persist the
    /// returned snapshot to disk to serve as “previous” on the next run.
    pub fn build_snapshot_from_store(
        &self,
        night_id: NightId,
        params: &FinkFatParams,
    ) -> NightSnapshot {
        // 1) Seeding (no progress UI)
        let sb = HealpixBinner::new(params.binning.healpix_depth);
        let tb = UniformTimeBinner::new(self.start_mjd, params.binning.time_bin_width_days);
        let index = build_index_from_alerts_precise(&self.alerts, &sb, &tb);
        let pairs = generate_pairs(&index, &self.alerts, &sb, &tb, params);
        let triplets = generate_triplets_from_pairs(&index, &self.alerts, &sb, &tb, params, &pairs);

        // 2) Feature extraction
        let mut seeds = Vec::with_capacity(pairs.len() + triplets.len());
        seeds.extend(extract_pair_features(
            self,
            &pairs,
            night_id,
            params.link.max_speed_rad_per_day,
        ));
        seeds.extend(extract_triplet_features(self, &triplets, night_id));
        renumber_seed_ids(&mut seeds);

        NightSnapshot {
            night_id,
            pairs,
            triplets,
            seeds,
        }
    }

    /// Borrow one alert by id (checked).
    ///
    /// Return
    /// ------
    /// `Some(&Alert)` if `id` is in bounds, otherwise `None`.
    #[inline]
    pub fn get(&self, id: AlertId) -> Option<&Alert> {
        self.alerts.get(id.idx())
    }

    /// Borrow one alert by id (debug assert + unchecked; fastest in hot loops).
    #[inline]
    pub fn get_fast(&self, id: AlertId) -> &Alert {
        debug_assert!(id.idx() < self.alerts.len());
        unsafe { self.alerts.get_unchecked(id.idx()) }
    }

    /// Borrow both members of a pair (checked).
    #[inline]
    pub fn get_pair(&self, p: Pair) -> Option<(&Alert, &Alert)> {
        p.resolve(self)
    }

    /// Borrow the three members of a triplet (checked).
    #[inline]
    pub fn get_triplet(&self, t: Triplet) -> Option<(&Alert, &Alert, &Alert)> {
        t.resolve(self)
    }

    /// Resolve an arbitrary list of `AlertId`s into borrowed `&Alert`s (checked).
    ///
    /// The iterator short-circuits to `None` if any id is out-of-bounds.
    #[inline]
    pub fn get_many<'a>(
        &self,
        ids: impl IntoIterator<Item = &'a AlertId>,
    ) -> Option<impl Iterator<Item = &Alert>> {
        let mut v = Vec::new();
        for id in ids {
            v.push(self.alerts.get(id.idx())?);
        }
        Some(v.into_iter())
    }
}

/* ----------------------------- Internals ----------------------------- */

fn renumber_seed_ids(seeds: &mut [SeedNode]) {
    for (k, s) in seeds.iter_mut().enumerate() {
        s.seed_id = k as u64;
    }
}

#[pymethods]
impl AlertStore {
    /// Build an [`AlertStore`] from 1-D NumPy arrays (copying once into Rust).
    ///
    /// Parameters
    /// ----------
    /// dia_source_id : numpy.ndarray\[uint64\]
    ///     LSST diaSource identifiers (shape: `(N,)`).
    /// ra, dec : numpy.ndarray\[float64\]
    ///     ICRS coordinates in **radians** (shape: `(N,)`).
    /// ra_err, dec_err : numpy.ndarray\[float64\]
    ///     ICRS coordinate uncertainties in **radians** (shape: `(N,)`).
    /// mjd_tt : numpy.ndarray\[float64\]
    ///     Detection time as **MJD (TT)** (shape: `(N,)`).
    /// flux, flux_err : numpy.ndarray\[float32\]
    ///     PSF **difference** flux and its uncertainty (shape: `(N,)`).
    /// band : numpy.ndarray\[uint8\]
    ///     Integer photometric band code (shape: `(N,)`).
    ///
    /// Returns
    /// -------
    /// AlertStore
    ///     A new store with `N` alerts and `start_mjd = floor(min(mjd_tt))`.
    ///
    /// Notes
    /// -----
    /// - All arrays must be 1-D and have the same length.
    /// - Values are copied into a contiguous `Vec<Alert>` for performance.
    /// - Field units are not converted here; callers must provide radians and TT.
    #[pyo3(
        text_signature = "(dia_source_id, ra, ra_err, dec, dec_err, mjd_tt, flux, flux_err, band, /)"
    )]
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn from_numpy(
        dia_source_id: PyReadonlyArray1<u64>,
        ra: PyReadonlyArray1<f64>,
        ra_err: PyReadonlyArray1<f64>,
        dec: PyReadonlyArray1<f64>,
        dec_err: PyReadonlyArray1<f64>,
        mjd_tt: PyReadonlyArray1<f64>,
        flux: PyReadonlyArray1<f32>,
        flux_err: PyReadonlyArray1<f32>,
        band: PyReadonlyArray1<u8>,
    ) -> PyResult<Self> {
        let dia_source_id = dia_source_id.as_slice()?;
        let ra = ra.as_slice()?;
        let ra_err = ra_err.as_slice()?;
        let dec = dec.as_slice()?;
        let dec_err = dec_err.as_slice()?;
        let mjd_tt = mjd_tt.as_slice()?;
        let flux = flux.as_slice()?;
        let flux_err = flux_err.as_slice()?;
        let band = band.as_slice()?;

        let n = dia_source_id.len();
        assert_eq!(ra.len(), n);
        assert_eq!(ra_err.len(), n);
        assert_eq!(dec.len(), n);
        assert_eq!(dec_err.len(), n);
        assert_eq!(mjd_tt.len(), n);
        assert_eq!(flux.len(), n);
        assert_eq!(flux_err.len(), n);
        assert_eq!(band.len(), n);

        let mut alerts = Vec::with_capacity(n);
        for (i, (&dia, &ra, &ra_err, &dec, &dec_err, &t, &fl, &flerr, &b)) in izip!(
            dia_source_id,
            ra,
            ra_err,
            dec,
            dec_err,
            mjd_tt,
            flux,
            flux_err,
            band
        )
        .enumerate()
        {
            alerts.push(Alert {
                id: AlertId::from(i),
                dia_source_id: dia,
                ra,
                ra_err,
                dec,
                dec_err,
                mjd_tt: t,
                flux: fl,
                flux_err: flerr,
                band: b,
            });
        }

        let start_mjd = mjd_tt.iter().copied().fold(f64::INFINITY, f64::min).floor();

        Ok(AlertStore { start_mjd, alerts })
    }

    /// Return an owned Python `Alert` (clone), not a Rust reference.
    ///
    /// Parameters
    /// ----------
    /// id : int
    ///     Dense alert identifier (`0 <= id < len(store)`).
    ///
    /// Returns
    /// -------
    /// Alert
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     If `id` is out of bounds.
    #[pyo3(text_signature = "($self, id, /)")]
    pub fn get_py<'py>(&self, py: Python<'py>, id: AlertId) -> PyResult<Py<Alert>> {
        let a = self
            .alerts
            .get(id.idx())
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>("invalid AlertId"))?;
        Py::new(py, a.clone())
    }

    /// Python `len(store)`.
    pub fn __len__(&self) -> usize {
        self.alerts.len()
    }

    /// Python `store[idx]` → owned `Alert` (clone).
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     If `idx` is out of range.
    pub fn __getitem__<'py>(&self, py: Python<'py>, idx: usize) -> PyResult<Py<Alert>> {
        let a = self
            .alerts
            .get(idx)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>("index out of range"))?;
        Py::new(py, a.clone())
    }

    /// Compact string summary (Python `str(store)`).
    fn __str__(&self) -> String {
        format!("{}", self)
    }

    /// Rich summary including time span and band set (Python `repr(store)`).
    ///
    /// Returns
    /// -------
    /// str
    ///     Example: `"AlertStore(n_alerts=..., start_mjd=..., time_span=[tmin, tmax], bands={...})"`.
    fn __repr__(&self) -> PyResult<String> {
        let n = self.alerts.len();
        if n == 0 {
            return Ok("AlertStore(n_alerts=0)".to_string());
        }

        let mut min_t = f64::INFINITY;
        let mut max_t = f64::NEG_INFINITY;
        let mut bands: BTreeSet<u8> = BTreeSet::new();

        for a in &self.alerts {
            let t = a.mjd_tt;
            if t < min_t {
                min_t = t;
            }
            if t > max_t {
                max_t = t;
            }
            bands.insert(a.band);
        }

        Ok(format!(
            "AlertStore(n_alerts={}, start_mjd={:.5}, time_span=[{:.5}, {:.5}], bands={:?})",
            n, self.start_mjd, min_t, max_t, bands
        ))
    }

    /// Generate intra-night **pairs** and **triplets** seeds.
    ///
    /// Parameters
    /// ----------
    /// params : FinkFatParams
    ///     Configuration object. The following fields are consumed here:
    ///     - `healpix_depth`, `time_bin_width_days` (bucketing),
    ///     - pair thresholds (max Δt, max angular sep, etc.),
    ///     - triplet thresholds (Δt between, pair separation, predicted residual),
    ///     - `show_progress` (enable/disable progress bars).
    ///
    /// Returns
    /// -------
    /// (Pairs, Triplets)
    ///     Pair and triplet collections as defined by the seeding module.
    ///
    /// Notes
    /// -----
    /// - If `show_progress == False`, runs the pure compute path without UI.
    /// - Otherwise, attaches `indicatif` progress bars with three phases:
    ///   buckets → pairs → triplets.
    #[pyo3(text_signature = "($self, params, /)")]
    pub fn generate_seeds(&self, params: &PyFinkFatParams) -> PyResult<(Pairs, Triplets)> {
        let sb = HealpixBinner::new(params.healpix_depth());
        let tb = UniformTimeBinner::new(self.start_mjd, params.time_bin_width_days());

        if !params.show_progress() {
            let index = build_index_from_alerts_precise(&self.alerts, &sb, &tb);
            let pairs = generate_pairs(&index, &self.alerts, &sb, &tb, &params.inner);

            let triplets =
                generate_triplets_from_pairs(&index, &self.alerts, &sb, &tb, &params.inner, &pairs);
            return Ok((pairs, triplets));
        }

        // ====== Progress UI ======
        let mp = make_multi_progress();
        let global = make_bar(&mp, 3, "pipeline");
        let pb_buckets = make_bar(&mp, 2 * self.alerts.len() as u64, "buckets");
        let pb_pairs = make_bar(&mp, self.alerts.len() as u64, "pairs");
        // pb_triplets: length set after pairs are known.
        let pb_triplets = make_bar(&mp, 1, "triplets (waiting)");

        // Step 1: bucket index
        let index =
            build_index_from_alerts_precise_with_progress(&self.alerts, &sb, &tb, &pb_buckets);
        global.inc(1);

        // Step 2: pairs
        let pairs =
            generate_pairs_with_progress(&index, &self.alerts, &sb, &tb, &params.inner, &pb_pairs);
        global.inc(1);

        // Step 3: triplets
        pb_triplets.set_length(pairs.len() as u64);
        pb_triplets.set_message("triplets");
        let triplets = generate_triplets_from_pairs_with_progress(
            &index,
            &self.alerts,
            &sb,
            &tb,
            &params.inner,
            &pairs,
            &pb_triplets,
        );
        global.inc(1);
        global.finish_with_message("done ✓");
        // ====== /Progress UI ======

        Ok((pairs, triplets))
    }

    /// Build deterministic, human-readable link UIDs for pairs and triplets.
    ///
    /// The function returns a nested Python `dict` of column lists ready to be
    /// turned into a pandas DataFrame. The UID construction is order-invariant
    /// with respect to member DIA source ids.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Structure:
    ///     - `out["pairs"]`:
    ///       - `"pair_uid"`: list\[str\] — format `"P|{min_dia}|{max_dia}"`.
    ///       - `"a_alert_id"`, `"b_alert_id"`: list\[int\] — store indices.
    ///       - `"a_dia_source_id"`, `"b_dia_source_id"`: list\[int\].
    ///     - `out["triplets"]`:
    ///       - `"trip_uid"`: list\[str\] — format `"T|{dia1}|{dia2}|{dia3}"` with sorted DIAs.
    ///       - `"a_alert_id"`, `"b_alert_id"`, `"c_alert_id"`: list\[int\].
    ///       - `"a_dia_source_id"`, `"b_dia_source_id"`, `"c_dia_source_id"`: list\[int\].
    ///
    /// Notes
    /// -----
    /// - The UID scheme is stable across runs and independent of alert ordering.
    /// - Use this to join seeds across nights or to deduplicate collections.
    #[pyo3(text_signature = "($self, pairs, triplets, /)")]
    pub fn build_link_uids_dict(
        &self,
        py: Python<'_>,
        pairs: Pairs,
        triplets: Triplets,
    ) -> PyResult<Py<PyAny>> {
        // --- Helpers for column builders ---
        #[inline]
        fn get_dia(alerts: &[Alert], id: AlertId) -> u64 {
            alerts[id.idx()].dia_source_id
        }

        // --- Build PAIRS columns ---
        let mut pair_uid: Vec<String> = Vec::with_capacity(pairs.len());
        let mut a_alert_id: Vec<u32> = Vec::with_capacity(pairs.len());
        let mut b_alert_id: Vec<u32> = Vec::with_capacity(pairs.len());
        let mut a_dia: Vec<u64> = Vec::with_capacity(pairs.len());
        let mut b_dia: Vec<u64> = Vec::with_capacity(pairs.len());

        for Pair { a, b } in pairs.iter().copied() {
            let da = get_dia(&self.alerts, a);
            let db = get_dia(&self.alerts, b);
            let (dmin, dmax) = if da <= db { (da, db) } else { (db, da) };
            // Stable, deterministic, human-readable UID
            let uid = format!("P|{}|{}", dmin, dmax);

            pair_uid.push(uid);
            a_alert_id.push(a.idx() as u32);
            b_alert_id.push(b.idx() as u32);
            a_dia.push(da);
            b_dia.push(db);
        }

        // --- Build TRIPLETS columns ---
        let mut trip_uid: Vec<String> = Vec::with_capacity(triplets.len());
        let mut ta_alert_id: Vec<u32> = Vec::with_capacity(triplets.len());
        let mut tb_alert_id: Vec<u32> = Vec::with_capacity(triplets.len());
        let mut tc_alert_id: Vec<u32> = Vec::with_capacity(triplets.len());
        let mut ta_dia: Vec<u64> = Vec::with_capacity(triplets.len());
        let mut tb_dia: Vec<u64> = Vec::with_capacity(triplets.len());
        let mut tc_dia: Vec<u64> = Vec::with_capacity(triplets.len());

        for Triplet { a, b, c } in triplets.iter().copied() {
            let da = get_dia(&self.alerts, a);
            let db = get_dia(&self.alerts, b);
            let dc = get_dia(&self.alerts, c);
            let mut s = [da, db, dc];
            s.sort_unstable();
            let uid = format!("T|{}|{}|{}", s[0], s[1], s[2]);

            trip_uid.push(uid);
            ta_alert_id.push(a.idx() as u32);
            tb_alert_id.push(b.idx() as u32);
            tc_alert_id.push(c.idx() as u32);
            ta_dia.push(da);
            tb_dia.push(db);
            tc_dia.push(dc);
        }

        // --- Convert to Python dicts of columns ---
        let pairs_dict = PyDict::new(py);
        pairs_dict.set_item("pair_uid", PyList::new(py, &pair_uid)?)?;
        pairs_dict.set_item("a_alert_id", PyList::new(py, &a_alert_id)?)?;
        pairs_dict.set_item("b_alert_id", PyList::new(py, &b_alert_id)?)?;
        pairs_dict.set_item("a_dia_source_id", PyList::new(py, &a_dia)?)?;
        pairs_dict.set_item("b_dia_source_id", PyList::new(py, &b_dia)?)?;

        let trips_dict = PyDict::new(py);
        trips_dict.set_item("trip_uid", PyList::new(py, &trip_uid)?)?;
        trips_dict.set_item("a_alert_id", PyList::new(py, &ta_alert_id)?)?;
        trips_dict.set_item("b_alert_id", PyList::new(py, &tb_alert_id)?)?;
        trips_dict.set_item("c_alert_id", PyList::new(py, &tc_alert_id)?)?;
        trips_dict.set_item("a_dia_source_id", PyList::new(py, &ta_dia)?)?;
        trips_dict.set_item("b_dia_source_id", PyList::new(py, &tb_dia)?)?;
        trips_dict.set_item("c_dia_source_id", PyList::new(py, &tc_dia)?)?;

        let out = PyDict::new(py);
        out.set_item("pairs", pairs_dict)?;
        out.set_item("triplets", trips_dict)?;
        Ok(out.into())
    }
}
