use ahash::{HashMap, HashMapExt};
use kiddo::KdTree;
use photom::{
    MJDTT, coordinates::ecliptic::EclipticCoordCov, observation_dataset::observation::Observation,
};

/// Precomputed ecliptic representation of a night's alerts,
/// indexed by exposure time for efficient per-time KD-tree lookup.
pub(crate) struct NightIndex<'a> {
    pub(crate) observations: &'a [Observation],
    pub(crate) ecl_coords: Vec<EclipticCoordCov>,
    pub(crate) trees_by_time: HashMap<u64, KdTree<f64, 3>>,
    pub(crate) unique_times: Vec<MJDTT>,
}

/// Convert ecliptic coordinates to a unit-sphere Cartesian point `[x, y, z]`.
#[inline]
pub(crate) fn ecl_to_unit_sphere(ecl: &EclipticCoordCov) -> [f64; 3] {
    let (sin_lat, cos_lat) = ecl.coord.lat.sin_cos();
    let (sin_lon, cos_lon) = ecl.coord.lon.sin_cos();
    [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
}

impl<'a> NightIndex<'a> {
    /// Build ecliptic coordinates and one KD-tree per unique exposure time in a
    /// single pass over the observation slice.
    pub(crate) fn build_night_index(observations: &'a [Observation]) -> Self {
        let mut ecl_coords = Vec::with_capacity(observations.len());
        let mut alerts_by_time: HashMap<u64, Vec<usize>> = HashMap::new();

        for (idx, obs) in observations.iter().enumerate() {
            ecl_coords.push(EclipticCoordCov::from(*obs.equ_coord()));
            alerts_by_time
                .entry(obs.mjd_tt().to_bits())
                .or_default()
                .push(idx);
        }

        let mut unique_times = Vec::with_capacity(alerts_by_time.len());
        let trees_by_time = alerts_by_time
            .iter()
            .map(|(&time_bits, indices)| {
                unique_times.push(MJDTT::from_bits(time_bits));
                let mut tree = KdTree::with_capacity(indices.len());
                for &idx in indices {
                    tree.add(&ecl_to_unit_sphere(&ecl_coords[idx]), idx as u64);
                }
                (time_bits, tree)
            })
            .collect();

        NightIndex {
            observations,
            ecl_coords,
            trees_by_time,
            unique_times,
        }
    }
}
