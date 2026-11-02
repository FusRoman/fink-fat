//! Orbital-class population of a ground-truth trajectory.
//!
//! The evaluation's headline objective is completeness of the **exotic**
//! (non-MBA) populations — NEO, Centaur, KBO, SDO — which are swamped by the
//! MBA majority in every aggregate metric. This module classifies each
//! ground-truth `TrajId` into a population from its heliocentric semi-major
//! axis `a` and perihelion `q`, so efficacy / selectivity can be broken down
//! per population.
//!
//! `a`/`q` come from `ssoBFT` and are added to the ground-truth Parquet by
//! `test_exp/solar_system_data/build_ground_truth.py` (columns `true_a_au`,
//! `true_q_au`). Older Parquets without those columns yield an empty map, so
//! every trajectory falls back to [`Population::Unknown`].

use ahash::AHashMap;
use anyhow::{Context, Result};
use camino::Utf8Path;
use photom::TrajId;
use polars::prelude::*;

/// Orbital-class bins. Boundaries are the usual dynamical cuts; NEO is defined
/// by perihelion (`q < 1.3` AU), the rest by semi-major axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Population {
    /// Near-Earth object: `q < 1.3` AU.
    Neo,
    /// Main belt: `q ≥ 1.3` and `a < 3.3` AU.
    Mba,
    /// Cybele / Hilda / Trojan region: `3.3 ≤ a < 5.5` AU.
    MidOuter,
    /// Centaur: `5.5 ≤ a < 30` AU.
    Centaur,
    /// Trans-Neptunian / Kuiper belt: `30 ≤ a < 50` AU.
    Kbo,
    /// Scattered-disk / detached: `a ≥ 50` AU.
    Sdo,
    /// No `a`/`q` available for this trajectory.
    Unknown,
}

impl Population {
    /// Classify from heliocentric semi-major axis and perihelion (AU).
    pub fn classify(a_au: f64, q_au: f64) -> Self {
        if !a_au.is_finite() || a_au <= 0.0 {
            return Population::Unknown;
        }
        if q_au.is_finite() && q_au < 1.3 {
            return Population::Neo;
        }
        if a_au < 3.3 {
            Population::Mba
        } else if a_au < 5.5 {
            Population::MidOuter
        } else if a_au < 30.0 {
            Population::Centaur
        } else if a_au < 50.0 {
            Population::Kbo
        } else {
            Population::Sdo
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Population::Neo => "NEO",
            Population::Mba => "MBA",
            Population::MidOuter => "Cybele/Hilda/Trojan",
            Population::Centaur => "Centaur",
            Population::Kbo => "KBO",
            Population::Sdo => "SDO/Detached",
            Population::Unknown => "Unknown",
        }
    }

    /// All populations, in a stable display order.
    pub fn all() -> [Population; 7] {
        [
            Population::Neo,
            Population::Mba,
            Population::MidOuter,
            Population::Centaur,
            Population::Kbo,
            Population::Sdo,
            Population::Unknown,
        ]
    }
}

/// Build a `TrajId -> Population` map from the ground-truth Parquet's
/// `true_a_au` / `true_q_au` columns (one entry per trajectory; `a`/`q` are
/// per-object constants so the first non-null row wins).
///
/// Returns an empty map if the columns are absent (old Parquet) — callers then
/// see [`Population::Unknown`] for every trajectory.
pub fn load_traj_populations(path: impl AsRef<Utf8Path>) -> Result<AHashMap<TrajId, Population>> {
    let path = path.as_ref();
    let df = LazyFrame::scan_parquet(path.as_str().into(), Default::default())
        .with_context(|| format!("failed to scan ground-truth parquet at {path}"))?
        .collect()
        .with_context(|| format!("failed to collect ground-truth parquet at {path}"))?;

    // Absent columns ⇒ empty map (every trajectory becomes Unknown downstream).
    if df.column("true_a_au").is_err() {
        return Ok(AHashMap::default());
    }

    let traj = df.column("traj_id")?.u32()?;
    let a = df.column("true_a_au")?.cast(&DataType::Float64)?;
    let a = a.f64()?;
    let q = df.column("true_q_au")?.cast(&DataType::Float64)?;
    let q = q.f64()?;

    let mut map: AHashMap<TrajId, Population> = AHashMap::default();
    for i in 0..df.height() {
        let Some(t) = traj.get(i) else { continue };
        let key = TrajId::Int(t);
        if map.contains_key(&key) {
            continue;
        }
        let a_v = a.get(i).unwrap_or(f64::NAN);
        let q_v = q.get(i).unwrap_or(f64::NAN);
        map.insert(key, Population::classify(a_v, q_v));
    }
    Ok(map)
}
