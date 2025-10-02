use std::{collections::BTreeSet, fmt};

use itertools::izip;
use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyList},
    Py, PyAny, PyErr, PyResult, Python,
};

use numpy::PyReadonlyArray1;

use crate::{
    params::params_binding::PyFinkFatParams,
    progress::{make_bar, make_multi_progress},
    seeding::{
        geometrical_seeding::{
            generate_pairs, generate_pairs_with_progress, generate_triplets_from_pairs,
            generate_triplets_from_pairs_with_progress, Pairs, Triplets,
        },
        healpix_binners::HealpixBinner,
        space_time_bucket::{
            build_index_from_alerts_precise, build_index_from_alerts_precise_with_progress,
        },
        uniform_time_binner::UniformTimeBinner,
    },
};

pub type AlertId = u32;

#[pyclass(module = "fink_fat")]
#[derive(Clone)]
pub struct Alert {
    #[pyo3(get)]
    pub id: AlertId,
    #[pyo3(get)]
    pub dia_source_id: u64, // from LSST
    #[pyo3(get)]
    pub ra: f64, // rad
    #[pyo3(get)]
    pub dec: f64, // rad
    #[pyo3(get)]
    pub mjd_tt: f64, // days (TT)
    #[pyo3(get)]
    pub flux: f32, // psf flux (nJy), note: it is a flux difference between template image and the visit image
    #[pyo3(get)]
    pub flux_err: f32, // psf flux error
    #[pyo3(get)]
    pub band: u8, // photometric band
}

// -------- Display / Debug --------

impl fmt::Display for Alert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Affichage compact, une ligne
        write!(
            f,
            "Alert(id={}, dia_source_id={}, ra={:.6} rad, dec={:.6} rad, mjd_tt={:.5}, flux={:.3}±{:.3} nJy, band={})",
            self.id, self.dia_source_id, self.ra, self.dec, self.mjd_tt, self.flux, self.flux_err, self.band
        )
    }
}

// -------- API Python --------

#[pymethods]
impl Alert {
    /// Python: str(alert)
    fn __str__(&self) -> String {
        format!("{}", self)
    }

    /// Python: repr(alert)
    fn __repr__(&self) -> String {
        // Plus verbeux que __str__ si tu veux
        format!(
            "Alert(id={}, dia_source_id={}, ra={:.6}, dec={:.6}, mjd_tt={:.5}, flux={:.3}, flux_err={:.3}, band={})",
            self.id, self.dia_source_id, self.ra, self.dec, self.mjd_tt, self.flux, self.flux_err, self.band
        )
    }
}

/// AlertStore holds all alerts of a night, indexed by AlertId as usize
#[pyclass(module = "fink_fat")]
pub struct AlertStore {
    pub start_mjd: f64,     // t0 of the night (TT)
    pub alerts: Vec<Alert>, // indexed by AlertId as usize
}

impl fmt::Display for AlertStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Version "légère", ne lit pas les alertes (pas de GIL)
        write!(
            f,
            "AlertStore(n_alerts={}, start_mjd={:.5})",
            self.alerts.len(),
            self.start_mjd
        )
    }
}

#[pymethods]
impl AlertStore {
    #[staticmethod]
    pub fn from_numpy(
        dia_source_id: PyReadonlyArray1<u64>,
        ra: PyReadonlyArray1<f64>,
        dec: PyReadonlyArray1<f64>,
        mjd_tt: PyReadonlyArray1<f64>,
        flux: PyReadonlyArray1<f32>,
        flux_err: PyReadonlyArray1<f32>,
        band: PyReadonlyArray1<u8>,
    ) -> PyResult<Self> {
        let dia_source_id = dia_source_id.as_slice()?;
        let ra = ra.as_slice()?;
        let dec = dec.as_slice()?;
        let mjd_tt = mjd_tt.as_slice()?;
        let flux = flux.as_slice()?;
        let flux_err = flux_err.as_slice()?;
        let band = band.as_slice()?;

        let n = dia_source_id.len();
        assert_eq!(ra.len(), n);
        assert_eq!(dec.len(), n);
        assert_eq!(mjd_tt.len(), n);
        assert_eq!(flux.len(), n);
        assert_eq!(flux_err.len(), n);
        assert_eq!(band.len(), n);

        let mut alerts = Vec::with_capacity(n);
        for (i, (&dia, &ra, &dec, &t, &fl, &flerr, &b)) in
            izip!(dia_source_id, ra, dec, mjd_tt, flux, flux_err, band).enumerate()
        {
            alerts.push(Alert {
                id: i as AlertId,
                dia_source_id: dia,
                ra,
                dec,
                mjd_tt: t,
                flux: fl,
                flux_err: flerr,
                band: b,
            });
        }

        let start_mjd = mjd_tt.iter().copied().fold(f64::INFINITY, f64::min).floor();

        Ok(AlertStore { start_mjd, alerts })
    }

    /// Retourne un **objet Python possédé** (copie/clône) plutôt qu’une référence Rust.
    pub fn get<'py>(&self, py: Python<'py>, id: AlertId) -> PyResult<Py<Alert>> {
        let a = self
            .alerts
            .get(id as usize)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>("invalid AlertId"))?;
        Py::new(py, a.clone())
    }

    pub fn __len__(&self) -> usize {
        self.alerts.len()
    }

    pub fn __getitem__<'py>(&self, py: Python<'py>, idx: usize) -> PyResult<Py<Alert>> {
        let a = self
            .alerts
            .get(idx)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>("index out of range"))?;
        Py::new(py, a.clone())
    }

    /// Python: str(store)
    fn __str__(&self) -> String {
        format!("{}", self)
    }

    /// Python: repr(store) — résumé plus riche (span temporel + bandes)
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

    #[allow(clippy::too_many_arguments)]
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

        // ====== PROGRESS ======
        let mp = make_multi_progress();
        let global = make_bar(&mp, 3, "pipeline");
        let pb_buckets = make_bar(&mp, 2 * self.alerts.len() as u64, "buckets");
        let pb_pairs = make_bar(&mp, self.alerts.len() as u64, "pairs");
        // pb_triplets: longueur ajustée après avoir les paires.
        let pb_triplets = make_bar(&mp, 1, "triplets (waiting)");

        // Step 1: buckets
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
        // ====== /PROGRESS ======

        Ok((pairs, triplets))
    }

    pub fn build_link_uids_dict(
        &self,
        py: Python<'_>,
        pairs: Pairs,
        triplets: Triplets,
    ) -> PyResult<Py<PyAny>> {
        // ===== Helpers for column builders =====
        #[inline]
        fn get_dia(alerts: &[Alert], id: AlertId) -> u64 {
            alerts[id as usize].dia_source_id
        }

        // ===== Build PAIRS columns =====
        let mut pair_uid: Vec<String> = Vec::with_capacity(pairs.len());
        let mut a_alert_id: Vec<u32> = Vec::with_capacity(pairs.len());
        let mut b_alert_id: Vec<u32> = Vec::with_capacity(pairs.len());
        let mut a_dia: Vec<u64> = Vec::with_capacity(pairs.len());
        let mut b_dia: Vec<u64> = Vec::with_capacity(pairs.len());

        for (a, b) in pairs.iter().copied() {
            let da = get_dia(&self.alerts, a);
            let db = get_dia(&self.alerts, b);
            let (dmin, dmax) = if da <= db { (da, db) } else { (db, da) };
            // Stable, deterministic, human-readable UID
            let uid = format!("P|{}|{}", dmin, dmax);

            pair_uid.push(uid);
            a_alert_id.push(a);
            b_alert_id.push(b);
            a_dia.push(da);
            b_dia.push(db);
        }

        // ===== Build TRIPLETS columns =====
        let mut trip_uid: Vec<String> = Vec::with_capacity(triplets.len());
        let mut ta_alert_id: Vec<u32> = Vec::with_capacity(triplets.len());
        let mut tb_alert_id: Vec<u32> = Vec::with_capacity(triplets.len());
        let mut tc_alert_id: Vec<u32> = Vec::with_capacity(triplets.len());
        let mut ta_dia: Vec<u64> = Vec::with_capacity(triplets.len());
        let mut tb_dia: Vec<u64> = Vec::with_capacity(triplets.len());
        let mut tc_dia: Vec<u64> = Vec::with_capacity(triplets.len());

        for (a, b, c) in triplets.iter().copied() {
            let da = get_dia(&self.alerts, a);
            let db = get_dia(&self.alerts, b);
            let dc = get_dia(&self.alerts, c);
            let mut s = [da, db, dc];
            s.sort_unstable();
            let uid = format!("T|{}|{}|{}", s[0], s[1], s[2]);

            trip_uid.push(uid);
            ta_alert_id.push(a);
            tb_alert_id.push(b);
            tc_alert_id.push(c);
            ta_dia.push(da);
            tb_dia.push(db);
            tc_dia.push(dc);
        }

        // ===== Convert to Python dicts of columns =====
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
