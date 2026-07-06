use hifitime::ut1::Ut1Provider;
use nalgebra::Vector3;
use outfit::{
    JPLEphem, OutfitError,
    cache::{
        observer_centric_cache::ObserverCentricCache, observer_fixed_cache::ObserverFixedCache,
    },
    constants::ROT_EQUMJ2000_TO_ECLMJ2000,
};
use photom::{
    MJDTT,
    observation_dataset::{ObsDataset, observation::Observation},
    observer::Observer,
};

pub fn get_observer<'a>(
    obs_dataset: &'a ObsDataset,
    obs: &Observation,
) -> Result<&'a Observer, OutfitError> {
    obs_dataset
        .get_observer(*obs.id())
        .ok_or_else(|| OutfitError::ObserverIdIsNone(*obs.id()))
}

pub fn new_fixed_cache(observer: &Observer) -> Result<ObserverFixedCache, OutfitError> {
    ObserverFixedCache::new(observer)
}

pub fn new_centric_cache(
    observer: &Observer,
    jpl: &JPLEphem,
    ut1_provider: &Ut1Provider,
    obs_time: MJDTT,
) -> Result<ObserverCentricCache, OutfitError> {
    let observer_fixed_cache = new_fixed_cache(observer)?;
    ObserverCentricCache::new(jpl, ut1_provider, obs_time, &observer_fixed_cache, true)
}

#[derive(Debug)]
pub struct EphemState {
    pub jpl: JPLEphem,
    pub ut1_provider: Ut1Provider,
}

impl EphemState {
    pub fn new(ephem_file_name: &str, ut1_file_version: Option<&str>) -> Self {
        let jpl: JPLEphem = ephem_file_name
            .try_into()
            .expect("Failed to load JPL ephemeris");

        let ut1_provider =
            Ut1Provider::download_from_jpl(ut1_file_version.unwrap_or("latest_eop2.long"))
                .expect("Download of the JPL short time scale UT1 data failed");
        Self { jpl, ut1_provider }
    }

    pub fn helio_observer_state(
        &self,
        observer: &Observer,
        obs_time: MJDTT,
    ) -> Result<HelioObsState, OutfitError> {
        let cache = new_centric_cache(observer, &self.jpl, &self.ut1_provider, obs_time)?;

        // unwrap is safe because we always compute the velocity in new_centric_cache
        // (cache_velocity: bool is set to true in the code)
        let vel = cache.helio_velocity.unwrap();

        Ok(HelioObsState {
            helio_cart_pos: ROT_EQUMJ2000_TO_ECLMJ2000
                * cache.helio_position.map(|v| v.into_inner()),
            helio_cart_vel: ROT_EQUMJ2000_TO_ECLMJ2000 * vel.map(|v| v.into_inner()),
        })
    }
}

pub struct HelioObsState {
    /// Observer heliocentric position at `obs_time` (AU), ecliptic J2000.
    pub helio_cart_pos: Vector3<f64>,
    /// Observer heliocentric velocity at `obs_time` (AU/day), ecliptic J2000.
    pub helio_cart_vel: Vector3<f64>,
}

impl HelioObsState {
    pub fn mid_state(&self, other: &HelioObsState) -> HelioObsState {
        HelioObsState {
            helio_cart_pos: (self.helio_cart_pos + other.helio_cart_pos) * 0.5,
            helio_cart_vel: (self.helio_cart_vel + other.helio_cart_vel) * 0.5,
        }
    }
}
