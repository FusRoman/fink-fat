pub mod generate_candidate;
pub mod io;
pub mod track_storage;
pub mod tracklet_data;
pub mod update;

use ahash::{HashMap, HashMapExt};
use kiddo::SquaredEuclidean;
use outfit::{
    DifferentialCorrectionConfig, IODParams, JPLEphem, cache::OutfitCache,
    constants::FitOrbitResult, differential_orbit_correction::differential_correction,
};
use photom::observation_dataset::{ObsDataset, ObsId, observation::Observation};

use crate::{
    ecliptic_state::{EclipticState, SingerParams},
    error::EngineError,
    propagator::{
        PredictionGeometry, Propagator,
        index::{NightIndex, ecl_to_unit_sphere},
    },
    seeding::{error::SeedingError, pairs::Pair, triplets::Triplet},
    tracklet::{track_storage::TrackId, tracklet_data::TrackletData},
};

// ── ScoredObservation ────────────────────────────────────────────────────────

/// An observation paired with its squared Mahalanobis distance to a tracklet's
/// predicted position.
///
/// This type is produced by [`Tracklet::associate_tracklet_to_night`] and
/// carries the $d^2$ value already computed during the $\chi^2$ gate, so that
/// downstream steps (e.g. exposure-time deduplication) can rank candidates
/// without recomputing the distance.
///
/// $$d^2 = (\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1}
///          (\mathbf{x} - \boldsymbol{\mu})$$
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScoredObservation<'a> {
    pub(crate) obs: &'a Observation,
    /// Squared Mahalanobis distance to the tracklet's predicted position at
    /// the observation's exposure time.
    pub(crate) mahalanobis_sq: f64,
}

impl<'a> ScoredObservation<'a> {
    fn new(obs: &'a Observation, mahalanobis_sq: f64) -> Self {
        Self {
            obs,
            mahalanobis_sq,
        }
    }
}

// ── Tracklet ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Tracklet {
    Seed(TrackletData<EclipticState>),
    Filter(TrackletData<EclipticState>),
    Orbit(TrackletData<FitOrbitResult>),
}

impl Tracklet {
    pub fn epoch(&self) -> f64 {
        match self {
            Tracklet::Seed(d) | Tracklet::Filter(d) => d.state.epoch,
            Tracklet::Orbit(orbit) => orbit.state.epoch(),
        }
    }

    pub fn state(&self) -> Option<&EclipticState> {
        self.ecliptic_data().map(|d| &d.state)
    }

    pub fn key(&self) -> TrackId {
        match self {
            Tracklet::Seed(d) | Tracklet::Filter(d) => d.key,
            Tracklet::Orbit(d) => d.key,
        }
    }

    pub fn obs_keys(&self) -> &[ObsId] {
        match self {
            Tracklet::Seed(d) | Tracklet::Filter(d) => &d.obs_keys,
            Tracklet::Orbit(d) => &d.obs_keys,
        }
    }

    pub fn get_ref_mag(&self) -> (f64, f64) {
        match self {
            Tracklet::Seed(d) | Tracklet::Filter(d) => (d.ref_mag, d.ref_mag_err),
            Tracklet::Orbit(d) => (d.ref_mag, d.ref_mag_err),
        }
    }

    pub fn is_orbit(&self) -> bool {
        match self {
            Tracklet::Orbit(_) => true,
            Tracklet::Seed(_) | Tracklet::Filter(_) => false,
        }
    }

    /// Returns the ecliptic state if the tracklet is in `Seed` or `Filter` state.
    pub fn ecliptic_data(&self) -> Option<&TrackletData<EclipticState>> {
        match self {
            Tracklet::Seed(d) | Tracklet::Filter(d) => Some(d),
            Tracklet::Orbit(_) => None,
        }
    }

    pub fn orbital_data(&self) -> Option<&TrackletData<FitOrbitResult>> {
        match self {
            Tracklet::Orbit(d) => Some(d),
            _ => None,
        }
    }

    pub fn is_seed(&self) -> bool {
        matches!(self, Tracklet::Seed(_))
    }

    pub fn is_filter(&self) -> bool {
        matches!(self, Tracklet::Filter(_))
    }

    pub(crate) fn ecliptic_data_mut(&mut self) -> Option<&mut TrackletData<EclipticState>> {
        match self {
            Tracklet::Seed(d) | Tracklet::Filter(d) => Some(d),
            Tracklet::Orbit(_) => None,
        }
    }

    pub fn resolve_members<'obs>(
        &self,
        obs_dataset: &'obs ObsDataset,
    ) -> Result<Vec<&'obs Observation>, SeedingError> {
        self.obs_keys()
            .iter()
            .map(|obs_idx| {
                obs_dataset
                    .get_observation(*obs_idx)
                    .ok_or_else(|| SeedingError::ObservationIndexNotFound(*obs_idx))
            })
            .collect()
    }

    pub fn resolve_member_owned(
        &self,
        obs_dataset: &ObsDataset,
    ) -> Result<Vec<Observation>, SeedingError> {
        self.obs_keys()
            .iter()
            .map(|obs_idx| {
                obs_dataset
                    .get_observation(*obs_idx)
                    .ok_or_else(|| SeedingError::ObservationIndexNotFound(*obs_idx))
                    .cloned()
            })
            .collect()
    }

    /// Build a [`Tracklet::Seed`] from a pair of observations.
    ///
    /// Delegates to [`TrackletData::from_pair`].
    pub(crate) fn seed_from_pairs(
        id: TrackId,
        pair: &Pair<'_>,
        acc_prior_var: f64,
        max_speed_rad_per_day: f64,
        process_noise_q: f64,
        singer_params: Option<SingerParams>,
    ) -> Option<Self> {
        TrackletData::from_pair(
            id,
            pair.a,
            pair.b,
            acc_prior_var,
            max_speed_rad_per_day,
            process_noise_q,
            singer_params,
        )
        .map(Tracklet::Seed)
    }

    /// Build a [`Tracklet::Filter`] from a triplet of observations.
    ///
    /// Delegates to [`TrackletData::from_triplet`].
    pub(crate) fn seed_from_triplet(
        id: TrackId,
        triplet: &Triplet<'_>,
        max_speed_rad_per_day: f64,
        process_noise_q: f64,
        singer_params: Option<SingerParams>,
    ) -> Option<Self> {
        TrackletData::from_triplet(
            id,
            triplet.a,
            triplet.b,
            triplet.c,
            max_speed_rad_per_day,
            process_noise_q,
            singer_params,
        )
        .map(Tracklet::Seed)
    }

    /// Find all observations in `index` that pass the coarse spatial gate and
    /// the Mahalanobis $\chi^2$ gate for this tracklet.
    ///
    /// Each surviving observation is returned as a [`ScoredObservation`]
    /// carrying the squared Mahalanobis distance $d^2$ already computed during
    /// the gate evaluation. This avoids recomputing the distance in downstream
    /// steps such as exposure-time deduplication.
    ///
    /// Two successive gates are applied per candidate:
    ///
    /// 1. **Coarse spatial gate** — bounding-circle query on the unit-sphere
    ///    KD-tree using a conservative angular radius derived from the
    ///    prediction covariance bounding box.
    ///
    /// 2. **Mahalanobis $\chi^2$ gate** — exact test:
    ///    $$d^2 = (\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1}
    ///             (\mathbf{x} - \boldsymbol{\mu}) < \chi^2_{\text{threshold}}$$
    ///
    /// Arguments
    /// ---------
    /// * `index`             – Spatial and temporal index of the night's alerts.
    /// * `association_sigma` – Angular search radius in units of $\sigma$,
    ///   used to size the coarse bounding circle.
    /// * `chi2_threshold`    – Maximum allowed $\chi^2$ residual for the fine
    ///   gate.
    ///
    /// Return
    /// ------
    /// * `Vec<ScoredObservation>` – Observations that passed both gates, each
    ///   annotated with its $d^2$ value. The list may contain duplicates across
    ///   exposure times; deduplication is handled by the caller.
    pub(crate) fn associate_tracklet_to_night<'a>(
        &self,
        index: &NightIndex<'a>,
        association_sigma: f64,
        chi2_threshold: f64,
    ) -> Vec<ScoredObservation<'a>> {
        // Use an index-keyed map to deduplicate observations that might appear
        // in multiple time-slot queries, keeping the lowest d² for each.
        let mut matched: HashMap<usize, ScoredObservation<'a>> = HashMap::new();

        for &predict_time in &index.unique_times {
            let Some((_, coord_cov)) = self.predict(predict_time) else {
                continue;
            };

            let Some(tree) = index.trees_by_time.get(&predict_time.to_bits()) else {
                continue;
            };

            let theta = coord_cov.bounding_box(association_sigma).max_half_width();
            let coarse_radius_sq = 2.0 * (1.0 - theta.cos());
            let query = ecl_to_unit_sphere(&coord_cov);

            for neighbour in tree.within::<SquaredEuclidean>(&query, coarse_radius_sq) {
                let obs_idx = neighbour.item as usize;

                // Fine Mahalanobis gate — reuse d² instead of discarding it.
                let Some(d2) = coord_cov.mahalanobis_sq(&index.ecl_coords[obs_idx]) else {
                    continue;
                };

                if d2 < chi2_threshold {
                    // Keep the entry with the lowest d² in case of duplicates.
                    matched
                        .entry(obs_idx)
                        .and_modify(|s| {
                            if d2 < s.mahalanobis_sq {
                                s.mahalanobis_sq = d2;
                            }
                        })
                        .or_insert_with(|| {
                            ScoredObservation::new(&index.observations[obs_idx], d2)
                        });
                }
            }
        }

        matched.into_values().collect()
    }

    fn fit_initial_orbit(
        &self,
        obs_dataset: &ObsDataset,
        cache: &OutfitCache,
        jpl: &JPLEphem,
        iod_params: &IODParams,
        diff_cor_config: &DifferentialCorrectionConfig,
        rng: &mut impl rand::Rng,
    ) -> Result<Self, EngineError> {
        let obs_members = self.resolve_member_owned(obs_dataset)?;
        let orbit = differential_correction(
            obs_members.as_slice(),
            cache,
            jpl,
            iod_params,
            diff_cor_config,
            None,
            rng,
        )?;

        let (ref_mag, ref_mag_err) = self.get_ref_mag();
        let track_data = TrackletData::new(
            self.key(),
            self.obs_keys().to_vec(),
            orbit,
            ref_mag,
            ref_mag_err,
        );
        Ok(Tracklet::Orbit(track_data))
    }
}
