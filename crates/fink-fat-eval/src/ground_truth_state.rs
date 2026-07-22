//! True topocentric state lookup, used to compute NEES/RMSE against ground
//! truth for real ZTF asteroid observations.
//!
//! The Parquet file consumed here is produced offline by
//! `test_exp/solar_system_data/build_ground_truth.py`, which joins the
//! fink-fat eval dataset to the ZTF alert stream and the per-asteroid
//! light-curve fits (which carry a true topocentric cartesian state per
//! alert epoch) — see that script for the full provenance chain.

use ahash::AHashMap;
use anyhow::{Context, Result};
use camino::Utf8Path;
use nalgebra::Vector3;
use photom::observation_dataset::ObsId;
use polars::lazy::frame::LazyFrame;

/// True topocentric state of an asteroid at the epoch of one observation.
#[derive(Debug, Clone, Copy)]
pub struct TruthState {
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub range_au: f64,
    pub pos_au: Vector3<f64>,
    pub vel_au_day: Vector3<f64>,
}

/// `ObsId -> TruthState` lookup built once per run from the ground-truth
/// Parquet file.
pub struct TruthLookup {
    by_obs_id: AHashMap<ObsId, TruthState>,
}

impl TruthLookup {
    /// Load the ground-truth Parquet file produced by
    /// `build_ground_truth.py`. Rows with a null truth (no light-curve match
    /// within tolerance) are skipped — `get` simply returns `None` for them.
    pub fn load(path: impl AsRef<Utf8Path>) -> Result<Self> {
        let path = path.as_ref();
        let df = LazyFrame::scan_parquet(path.as_str().into(), Default::default())
            .with_context(|| format!("failed to scan ground-truth parquet at {path}"))?
            .collect()
            .with_context(|| format!("failed to collect ground-truth parquet at {path}"))?;

        let ids = df.column("id")?.u64()?;
        let true_ra = df.column("true_ra_deg")?.f32()?;
        let true_dec = df.column("true_dec_deg")?.f32()?;
        let true_range = df.column("true_range_au")?.f32()?;
        let true_px = df.column("true_px_au")?.f32()?;
        let true_py = df.column("true_py_au")?.f32()?;
        let true_pz = df.column("true_pz_au")?.f32()?;
        let true_vx = df.column("true_vx_au_day")?.f32()?;
        let true_vy = df.column("true_vy_au_day")?.f32()?;
        let true_vz = df.column("true_vz_au_day")?.f32()?;

        let mut by_obs_id = AHashMap::default();
        for i in 0..df.height() {
            let (
                Some(id),
                Some(ra),
                Some(dec),
                Some(range),
                Some(px),
                Some(py),
                Some(pz),
                Some(vx),
                Some(vy),
                Some(vz),
            ) = (
                ids.get(i),
                true_ra.get(i),
                true_dec.get(i),
                true_range.get(i),
                true_px.get(i),
                true_py.get(i),
                true_pz.get(i),
                true_vx.get(i),
                true_vy.get(i),
                true_vz.get(i),
            )
            else {
                continue;
            };

            by_obs_id.insert(
                id as ObsId,
                TruthState {
                    ra_deg: ra as f64,
                    dec_deg: dec as f64,
                    range_au: range as f64,
                    pos_au: Vector3::new(px as f64, py as f64, pz as f64),
                    vel_au_day: Vector3::new(vx as f64, vy as f64, vz as f64),
                },
            );
        }

        Ok(Self { by_obs_id })
    }

    /// True topocentric state for a single observation id, if known.
    pub fn get(&self, obs_id: ObsId) -> Option<&TruthState> {
        self.by_obs_id.get(&obs_id)
    }

    /// Whether any ground truth was loaded at all.
    pub fn is_empty(&self) -> bool {
        self.by_obs_id.is_empty()
    }
}
