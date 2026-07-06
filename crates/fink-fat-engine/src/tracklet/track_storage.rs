use ahash::{HashMap, HashMapExt, HashSet};
use camino::Utf8Path;
use hifitime::ut1::Ut1Provider;
use outfit::{DifferentialCorrectionConfig, IODParams, JPLEphem, cache::OutfitCache};
use photom::{
    observation_dataset::{ObsDataset, observation::Observation},
    observer::error_model::{ModelCorrection, ObsErrorModel},
};
use polars::error::PolarsResult;
use rand::SeedableRng;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

use crate::{
    engine_config::EngineConfig,
    error::EngineError,
    seeding_step::process_one_night,
    spacetime_bucket::healpix_binner::HealpixBinner,
    tracklet::{
        Tracklet,
        generate_candidate::{Associations, generate_next_night_candidate},
        io::write_tracklets_parquet,
        update::branch_tracklet,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TrackId(pub u32);

pub struct TrackStorage {
    tracklets: HashMap<TrackId, Tracklet>,
    free_ids: HashSet<TrackId>,
    next_fresh_id: u32,
}

impl TrackStorage {
    pub fn new() -> Self {
        Self {
            tracklets: HashMap::new(),
            free_ids: HashSet::default(),
            next_fresh_id: 0,
        }
    }

    fn next_id(&mut self) -> TrackId {
        if let Some(&id) = self.free_ids.iter().next() {
            self.free_ids.remove(&id);
            id
        } else {
            let id = TrackId(self.next_fresh_id);
            self.next_fresh_id += 1;
            id
        }
    }

    pub fn remove(&mut self, id: TrackId) -> Option<Tracklet> {
        let tracklet = self.tracklets.remove(&id)?;
        self.free_ids.insert(id);
        Some(tracklet)
    }

    pub fn get(&self, id: TrackId) -> Option<&Tracklet> {
        self.tracklets.get(&id)
    }

    pub fn get_mut(&mut self, id: TrackId) -> Option<&mut Tracklet> {
        self.tracklets.get_mut(&id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (TrackId, &Tracklet)> {
        self.tracklets.iter().map(|(&id, t)| (id, t))
    }

    pub fn len(&self) -> usize {
        self.tracklets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tracklets.is_empty()
    }

    /// Return an iterator over all tracklets held in storage.
    pub fn iter_tracklets(&self) -> impl Iterator<Item = &Tracklet> {
        self.tracklets.values()
    }

    /// Generate seed tracklets from a slice of observations and insert them
    /// into the storage with globally-unique [`TrackId`]s.
    ///
    /// Internally calls [`process_one_night`] which produces tracklets with
    /// provisional placeholder ids. Those ids are overwritten here with real
    /// ids drawn from the storage allocator before insertion.
    ///
    /// Arguments
    /// ---------
    /// * `config`       – Engine configuration (healpix depth, time binner
    ///                    width, pair/triplet parameters).
    /// * `observations` – Raw alert slice for the night to process. All observations should belong to the same night
    ///
    /// Return
    /// ------
    /// * `Ok(usize)` – Number of new tracklets inserted.
    /// * `Err(EngineError)` – If the observation slice is empty or seeding fails.
    pub fn generate_seeds(
        mut self,
        config: &EngineConfig,
        observations: &[Observation],
    ) -> Result<(Self, usize), EngineError> {
        let spatial_binner = HealpixBinner::new(config.healpix_depth);

        let new_tracklets = process_one_night(
            observations,
            &spatial_binner,
            config,
            config.time_binner_width,
        )?;

        let n = new_tracklets.len();
        self.tracklets.reserve(n);

        for mut tracklet in new_tracklets {
            let track_key = tracklet.key();
            let id = self.next_id();

            // Overwrite the provisional id assigned by process_one_night.
            let ecliptic_data_mut = tracklet
                .ecliptic_data_mut()
                .ok_or(EngineError::NotTrackletVariant(track_key))?;
            ecliptic_data_mut.key = id;
            self.tracklets.insert(id, tracklet);
        }

        Ok((self, n))
    }

    /// Build candidate associations between all active tracklets and the
    /// observations of the next night.
    ///
    /// This is a thin delegation to [`generate_next_night_candidate`] that
    /// passes the full tracklet collection held by the storage. Tracklets are
    /// collected into a temporary `Vec` so that a contiguous slice can be
    /// forwarded to the underlying function.
    ///
    /// See [`generate_next_night_candidate`] for the full pipeline description
    /// (HEALPix prefilter → spatial association → magnitude filter →
    /// exposure-time deduplication).
    ///
    /// Arguments
    /// ---------
    /// * `obs_dataset`   – Full observation dataset (all nights).
    /// * `next_night`    – Identifier of the target night.
    /// * `engine_config` – Association parameters (`association_sigma`,
    ///                     `chi2_threshold`, `max_magnitude_diff`).
    ///
    /// Return
    /// ------
    /// * `Ok(Associations)` – One entry per tracklet that survived all
    ///   filters, paired with its candidate observations.
    /// * `Err(EngineError::StageFailed)` – If the target night is absent
    ///   from the dataset or uses a non-contiguous memory layout.
    pub fn generate_candidates<'o>(
        &self,
        observations: &'o [Observation],
        engine_config: &EngineConfig,
    ) -> Result<Associations<'o>, EngineError> {
        generate_next_night_candidate(observations, engine_config, self.tracklets.values())
    }

    /// Update all active tracklets from the inter-night associations.
    ///
    /// For each associated tracklet:
    /// - the first successful Kalman update is applied **in-place**,
    /// - each additional successful update is inserted as a new branch
    ///   with a fresh [`TrackId`].
    ///
    /// Tracklets absent from `associations` or for which all Kalman updates
    /// fail are left untouched.
    ///
    /// Arguments
    /// ---------
    /// * `associations` – Associations produced by [`TrackStorage::generate_candidates`].
    ///
    /// Return
    /// ------
    /// `Self` with updated tracklet entries.
    pub fn update_from_associations(
        mut self,
        associations: Associations<'_>,
    ) -> Result<Self, EngineError> {
        let updates: Vec<_> = associations
            .into_iter()
            .map(|(id, c)| {
                let tracklet = self
                    .tracklets
                    .get(&id)
                    .ok_or(EngineError::TrackIdNotFound(id))?;
                Ok((id, branch_tracklet(tracklet, c)))
            })
            .collect::<Result<Vec<_>, EngineError>>()?;

        for (parent_id, branches) in updates {
            let mut iter = branches.into_iter();

            // First branch replaces the parent in-place.
            let Some(first) = iter.next() else { continue };
            if let Some(existing) = self.tracklets.get_mut(&parent_id) {
                *existing = first;
            }

            // Remaining branches are new entries.
            for mut branch in iter {
                let track_key = branch.key();
                let id = self.next_id();

                let ecliptic_data_mut = branch
                    .ecliptic_data_mut()
                    .ok_or(EngineError::NotTrackletVariant(track_key))?;
                ecliptic_data_mut.key = id;
                self.tracklets.insert(id, branch);
            }
        }

        Ok(self)
    }

    /// Map tracklet filter variant into orbit variant if orbit fitting converge and no error return.
    /// Compute orbit using all variant filter tracklets and do it in parallel with rayon.
    pub fn promote_to_orbit(
        mut self,
        obs_dataset: &ObsDataset,
        error_model: ObsErrorModel,
        jpl: &JPLEphem,
        ut1_provider: &Ut1Provider,
        iod_params: &IODParams,
        diff_cor_config: &DifferentialCorrectionConfig,
        rng: &mut impl rand::Rng,
    ) -> Result<Self, EngineError> {
        let corrected_dataset = obs_dataset
            .clone()
            .with_error_model(error_model)
            .apply_model_errors()
            .apply_batch_rms_correction(iod_params.gap_max);
        let cache = OutfitCache::build(&corrected_dataset, jpl, ut1_provider, true)?;

        let seed: u64 = rng.random();

        self.tracklets.par_iter_mut().for_each(|(_key, tracklet)| {
            if let Tracklet::Filter(_) = tracklet {
                let mut local_rng = rand::rngs::SmallRng::seed_from_u64(seed);

                let orbit_result = tracklet.fit_initial_orbit(
                    &corrected_dataset,
                    &cache,
                    jpl,
                    iod_params,
                    diff_cor_config,
                    &mut local_rng,
                );
                if let Ok(track_with_orbit) = orbit_result {
                    *tracklet = track_with_orbit;
                }
            }
        });

        Ok(self)
    }

    /// remove every tracklet variant filter and seed if the tracklet epoch exceed the keep time limit.
    pub fn clear_lost_tracklet(mut self, current_time: f64, time_limit: f64) -> Self {
        self.tracklets.retain(|_, tracklet| match tracklet {
            Tracklet::Filter(_) | Tracklet::Seed(_) => current_time - tracklet.epoch() < time_limit,
            Tracklet::Orbit(_) => true,
        });
        self
    }

    /// Serialize the track storage to a Parquet file.
    ///
    /// Each tracklet is written as a single row. Orbital elements are
    /// converted to Keplerian form for a uniform representation. Tracklets
    /// of type `Seed` and `Filter` have `null` orbital columns.
    ///
    /// Arguments
    /// ---------
    /// * `path` – Destination path for the Parquet file.
    ///
    /// Return
    /// ------
    /// * `Ok(())` – File written successfully.
    /// * `Err(PolarsError)` – If DataFrame construction or file I/O fails.
    pub fn write_parquet(&self, path: impl AsRef<Utf8Path>) -> PolarsResult<()> {
        write_tracklets_parquet(self.tracklets.values(), path)
    }

    pub fn filter_orbit_out(mut self) -> Self {
        self.tracklets = self
            .tracklets
            .iter()
            .filter_map(|(track_id, track)| {
                if !track.is_orbit() {
                    Some((*track_id, track.clone()))
                } else {
                    self.free_ids.insert(track.key());
                    None
                }
            })
            .collect::<HashMap<TrackId, Tracklet>>();
        self
    }
}
