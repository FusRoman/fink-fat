//! Local alert density estimation, used as the clutter background `λ_clutter`
//! in branch log-likelihood-ratio scoring
//! (see [`llr_score`](crate::topocentric_kf::branching::llr_score)).
//!
//! Lives here rather than in `topocentric_kf` because it is a property of
//! the alert index itself ([`BucketIndex`]/[`HealpixBinner`]), not of the
//! Kalman machinery — the same estimator is reusable wherever a "how busy is
//! this patch of sky tonight?" question comes up.

use photom::coordinates::equatorial::EquCoord;

use crate::spacetime_bucket::{
    bucket::{BucketIndex, SpacetimeBucketEvent},
    healpix_binner::HealpixBinner,
    spatial_binner::SpatialBinner,
    spatial_binner::SpatialKey,
};

/// Local alert density (alerts per steradian) around a sky position.
///
/// # Arguments
/// * `bucket_index` – Spatial(+time) index of the current night's alerts.
/// * `spatial_binner` – The same binner used to build `bucket_index`.
/// * `center` – Sky position to estimate the density around (typically a
///   [`SearchRegion`](crate::topocentric_kf::kalman_bank::ellipse_region_finder::SearchRegion)'s
///   center).
///
/// # Returns
/// `alerts_in_cell / cell_area_steradian`, or `0.0` if the cell is empty.
/// Callers taking a logarithm of this value (branch LLR scoring) must floor
/// it first — see
/// [`observation_llr_delta`](crate::topocentric_kf::branching::llr_score::observation_llr_delta).
pub fn local_clutter_density<Object>(
    bucket_index: &BucketIndex<Object>,
    spatial_binner: &HealpixBinner,
    center: &EquCoord,
) -> f64 {
    let center_key = spatial_binner.key_for(center);
    let alerts_in_cell = count_alerts_in_cell(bucket_index, center_key);
    let density = if alerts_in_cell == 0 {
        0.0
    } else {
        alerts_in_cell as f64 / healpix_pixel_area_sr(spatial_binner.depth())
    };
    SpacetimeBucketEvent::ClutterDensityEstimate {
        alerts_in_cell,
        density_per_sr: density,
    }
    .emit();
    density
}

/// Total number of alerts across every time bin of a single HEALPix cell.
///
/// The clutter estimate is a per-night, purely spatial density: it sums
/// alerts over all time bins sharing `space_key` rather than restricting to
/// one bin, since a night's `BucketIndex` may already be split into several
/// narrow time bins upstream (see `build_kf_bank_collection`).
fn count_alerts_in_cell<Object>(
    bucket_index: &BucketIndex<Object>,
    space_key: SpatialKey,
) -> usize {
    bucket_index
        .buckets
        .iter()
        .filter(|(key, _)| key.space_key == space_key)
        .map(|(_, bucket)| bucket.members.len())
        .sum()
}

/// HEALPix pixel area in steradians at a given order (depth).
///
/// $$\Omega = \frac{4\pi}{12 \cdot 4^{depth}}$$
pub fn healpix_pixel_area_sr(depth: u8) -> f64 {
    (4.0 * std::f64::consts::PI) / (12.0 * 4f64.powi(depth as i32))
}

#[cfg(test)]
mod clutter_density_tests {
    use super::*;
    use crate::spacetime_bucket::{
        bucket::build_alert_bucket_index, uniform_time_binner::UniformTimeBinner,
    };
    use photom::{
        observation_dataset::{ObsDataset, observation::ObservationInput},
        photometry::{Filter, Photometry},
    };

    fn mk_observation(id: u64, ra: f64, dec: f64, mjd_tt: f64) -> Observation {
        let obs_dataset = ObsDataset::empty();
        let equ = EquCoord::new(ra, 0.0, dec, 0.0);
        let photometry = Photometry {
            magnitude: 20.0,
            error: 0.1,
            filter: Filter::String("r".to_string()),
        };
        let input = ObservationInput::new(id, equ, photometry, mjd_tt, None);
        let (obs_dataset, obs_id) = obs_dataset.push_observation(vec![input]).unwrap();
        obs_dataset
            .get_obs_by_index(*obs_id.get(0).unwrap())
            .unwrap()
            .clone()
    }

    use photom::observation_dataset::observation::Observation;

    #[test]
    fn healpix_pixel_area_matches_closed_form_at_a_few_depths() {
        for depth in [0u8, 4, 8, 12] {
            let expected = (4.0 * std::f64::consts::PI) / (12.0 * 4f64.powi(depth as i32));
            assert!((healpix_pixel_area_sr(depth) - expected).abs() < 1e-15);
        }
    }

    #[test]
    fn local_clutter_density_is_zero_for_empty_cell() {
        let spatial_binner = HealpixBinner::new(8);
        let time_binner = UniformTimeBinner::new(60000.0, 1.0);
        let bucket_index =
            build_alert_bucket_index(std::iter::empty(), &spatial_binner, &time_binner);

        let center = EquCoord::new(1.0, 0.0, 0.2, 0.0);
        assert_eq!(
            local_clutter_density(&bucket_index, &spatial_binner, &center),
            0.0
        );
    }

    #[test]
    fn local_clutter_density_scales_with_alert_count_in_cell() {
        let spatial_binner = HealpixBinner::new(8);
        let time_binner = UniformTimeBinner::new(60000.0, 1.0);

        let center_ra = 1.0;
        let center_dec = 0.2;
        // Several alerts landing in the same cell as `center`.
        let alerts: Vec<Observation> = (0..5)
            .map(|id| mk_observation(id, center_ra, center_dec, 60000.1))
            .collect();
        let alert_refs: Vec<&Observation> = alerts.iter().collect();

        let bucket_index =
            build_alert_bucket_index(alert_refs.iter().copied(), &spatial_binner, &time_binner);

        let center = EquCoord::new(center_ra, 0.0, center_dec, 0.0);
        let density = local_clutter_density(&bucket_index, &spatial_binner, &center);

        let expected = 5.0 / healpix_pixel_area_sr(spatial_binner.depth());
        assert!((density - expected).abs() < 1e-9);
    }
}
