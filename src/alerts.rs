use std::{collections::BTreeSet, fmt};

use itertools::izip;
use pyo3::{pyclass, pymethods, Py, PyErr, PyResult, Python};

use numpy::PyReadonlyArray1;

use crate::seeding::{
    geometrical_seeding::{generate_pairs_and_triplets, PairParams, TripletParams},
    healpix_binners::HealpixBinner,
    space_time_bucket::build_index_from_alerts_precise,
    uniform_time_binner::UniformTimeBinner,
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

        Ok(AlertStore {
            start_mjd: start_mjd,
            alerts,
        })
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

    pub fn generate_seeds(
        &self,
        healpix_depth: u8,
        time_bin_width_days: f64,
        pair_max_dt: f64,
        pair_max_sep: f64,
        allow_same_timebin: bool,
        trip_max_dt_between: f64,
        trip_max_pair_sep: f64,
        trip_max_pred_resid: f64,
        enforce_time_order: bool,
    ) -> PyResult<(Vec<(AlertId, AlertId)>, Vec<(AlertId, AlertId, AlertId)>)> {
        let pair_params = PairParams {
            max_dt: pair_max_dt,
            max_sep: pair_max_sep,
            allow_same_timebin,
        };
        let triplet_params = TripletParams {
            max_dt_between: trip_max_dt_between,
            max_pair_sep: trip_max_pair_sep,
            max_predicted_residual: trip_max_pred_resid,
            enforce_time_order,
        };

        let sb = HealpixBinner::new(healpix_depth);
        let tb = UniformTimeBinner::new(self.start_mjd, time_bin_width_days);

        println!(
            "Generating seeds with Healpix depth {}, time bin width {:.3} days",
            healpix_depth, time_bin_width_days
        );
        let index = build_index_from_alerts_precise(&self.alerts, &sb, &tb);

        println!(
            "Built space-time index with {} buckets (max bucket size {})",
            index.buckets.len(),
            index
                .buckets
                .values()
                .map(|b| b.members.len())
                .max()
                .unwrap_or(0)
        );

        let seeds = generate_pairs_and_triplets(
            &index,
            &self.alerts,
            &sb,
            &tb,
            pair_params,
            triplet_params,
        );

        println!(
            "Generated {} pairs and {} triplets",
            seeds.pairs.len(),
            seeds.triplets.len()
        );

        Ok((seeds.pairs, seeds.triplets))
    }
}
