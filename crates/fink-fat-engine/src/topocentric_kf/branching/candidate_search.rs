//! Match next-night observations against a bank's predicted search ellipse.
//!
//! Salvaged from a stale draft (`bank_collection/night_candidate_search.rs`,
//! which referenced a `KFPairBank` type that never existed in the crate and
//! was never wired into the build) and retyped against the real
//! [`KFBank`](crate::topocentric_kf::kalman_bank::KFBank)/[`SearchRegion`].
//! The core two-stage gate (cone query + Mahalanobis/likelihood filter) is
//! unchanged.
//!
//! # Strategy
//! 1. Build a spatial index over the next night's observations once
//!    ([`build_alert_bucket_index`](crate::spacetime_bucket::bucket::build_alert_bucket_index)
//!    + [`HealpixBinner`]) — O(N log N).
//! 2. For a bank's predicted [`SearchRegion`], cone-query the spatial index
//!    around `(center_ra, center_dec, radius_rad)`, then for each candidate
//!    apply a cheap per-component Mahalanobis gate
//!    ([`SearchRegion::any_component_contains`]) before paying for the full
//!    mixture likelihood ([`SearchRegion::mixture_likelihood`]).

use nalgebra::Vector2;
use photom::{
    MJDTT,
    coordinates::equatorial::EquCoord,
    observation_dataset::{ObsId, observation::Observation},
};

use crate::{
    spacetime_bucket::{
        bucket::{BucketIndex, BucketKey},
        healpix_binner::HealpixBinner,
        spatial_binner::SpatialBinner,
        time_binner::{TimeBin, TimeBinner},
    },
    topocentric_kf::kalman_bank::ellipse_region_finder::SearchRegion,
};

/// A next-night observation accepted inside at least one Kalman hypothesis'
/// error ellipse, with its full mixture-likelihood score.
#[derive(Debug, Clone, Copy)]
pub struct CandidateMatch<'obs> {
    pub observation: &'obs Observation,
    pub likelihood: f64,
}

/// Candidate matches found for a single bank.
#[derive(Debug, Clone)]
pub struct BankCandidates<'obs> {
    /// Association history of the bank these candidates were matched
    /// against, i.e. [`KFBank::track_ids`](crate::topocentric_kf::kalman_bank::KFBank::track_ids) —
    /// identifies the bank without needing a fictional pair-id type.
    pub track_ids: Vec<ObsId>,
    /// Predicted search region the candidates were evaluated against.
    pub search_region: SearchRegion,
    pub matches: Vec<CandidateMatch<'obs>>,
}

/// Time binner mapping every epoch to the same bin.
///
/// [`BucketIndex`] is keyed by `(SpatialKey, TimeBin)`, but a next-night cone
/// search at a single target epoch has no temporal dimension to discretize.
/// This binner lets us reuse
/// [`build_alert_bucket_index`](crate::spacetime_bucket::bucket::build_alert_bucket_index)
/// unchanged instead of introducing a parallel spatial-only index type.
pub struct SingleBinTimeBinner;

impl TimeBinner for SingleBinTimeBinner {
    fn bin_for(&self, _mjd_tt: MJDTT) -> TimeBin {
        TimeBin(0)
    }

    fn bins_in_range(&self, _t0: MJDTT, _t1: MJDTT) -> Vec<TimeBin> {
        vec![TimeBin(0)]
    }

    fn bin_width(&self) -> MJDTT {
        f64::INFINITY
    }

    fn bin_start(&self, _k: i64) -> MJDTT {
        f64::NEG_INFINITY
    }
}

const SINGLE_TIME_BIN: TimeBin = TimeBin(0);

/// Cone-query the spatial index around `region`'s center and radius.
///
/// This is a coarse over-approximation (whole HEALPix cells, not an exact
/// disk): candidates still need the Mahalanobis gate applied by
/// [`filter_candidates`].
fn query_region_candidates<'obs>(
    bucket_index: &BucketIndex<&'obs Observation>,
    spatial_binner: &HealpixBinner,
    region: &SearchRegion,
) -> Vec<&'obs Observation> {
    let center = EquCoord::new(region.center_ra, 0.0, region.center_dec, 0.0);
    let center_key = spatial_binner.key_for(&center);

    let mut neighbor_keys = spatial_binner.neighbors(center_key, region.radius_rad);
    neighbor_keys.sort_unstable();
    neighbor_keys.dedup();

    neighbor_keys
        .into_iter()
        .filter_map(|space_key| {
            bucket_index.buckets.get(&BucketKey {
                space_key,
                time_bin: SINGLE_TIME_BIN,
            })
        })
        .flat_map(|bucket| bucket.members.iter().copied())
        .collect()
}

/// Apply the two-stage gate described in the module docs: a cheap
/// per-component Mahalanobis test, then the full mixture likelihood for
/// survivors.
fn filter_candidates<'obs>(
    region: &SearchRegion,
    candidates: impl IntoIterator<Item = &'obs Observation>,
    gate_chi2: f64,
    likelihood_threshold: f64,
) -> Vec<CandidateMatch<'obs>> {
    candidates
        .into_iter()
        .filter_map(|observation| {
            let coord = observation.equ_coord();
            if !region.any_component_contains(coord.ra, coord.dec, gate_chi2) {
                return None;
            }
            let likelihood = region.mixture_likelihood(coord.ra, coord.dec);
            (likelihood >= likelihood_threshold).then_some(CandidateMatch {
                observation,
                likelihood,
            })
        })
        .collect()
}

/// Find the next-night observations falling inside one bank's predicted
/// search ellipse.
///
/// # Arguments
/// * `search_region` – Bank's predicted region, e.g. from
///   [`KFBank::predict_search_region`](crate::topocentric_kf::kalman_bank::KFBank::predict_search_region).
/// * `track_ids` – Association history of the bank, used to identify it in
///   the returned [`BankCandidates`] (see
///   [`KFBank::track_ids`](crate::topocentric_kf::kalman_bank::KFBank::track_ids)).
/// * `bucket_index` – Spatial index of the next night's observations (built
///   once per night, shared across every bank — see [`SingleBinTimeBinner`]).
/// * `spatial_binner` – Same binner used to build `bucket_index`.
/// * `gate_chi2`, `likelihood_threshold` – Two-stage gate parameters (cheap
///   Mahalanobis pre-filter, then mixture-likelihood threshold).
///
/// # Returns
/// The bank's [`BankCandidates`] — possibly with an empty `matches` list,
/// which is the common case at LSST cadence and simply means the following
/// branching step will spawn only the null branch.
pub fn find_candidates_for_bank<'obs>(
    search_region: &SearchRegion,
    track_ids: Vec<ObsId>,
    bucket_index: &BucketIndex<&'obs Observation>,
    spatial_binner: &HealpixBinner,
    gate_chi2: f64,
    likelihood_threshold: f64,
) -> BankCandidates<'obs> {
    let candidates = query_region_candidates(bucket_index, spatial_binner, search_region);
    let matches = filter_candidates(search_region, candidates, gate_chi2, likelihood_threshold);

    BankCandidates {
        track_ids,
        search_region: search_region.clone(),
        matches,
    }
}

/// Diagonal observation-noise matrix `[σ_RA², σ_Dec²]`, as consumed by
/// [`KFBank::predict_search_region`](crate::topocentric_kf::kalman_bank::KFBank::predict_search_region).
///
/// Small helper kept here (rather than duplicated at every call site) since
/// candidate search is the first place in the per-night flow that needs it.
pub fn observation_noise_diagonal(ra_error: f64, dec_error: f64) -> Vector2<f64> {
    Vector2::new(ra_error * ra_error, dec_error * dec_error)
}

#[cfg(test)]
mod candidate_search_tests {
    use super::*;
    use nalgebra::Matrix2;
    use photom::{
        observation_dataset::{ObsDataset, observation::ObservationInput},
        photometry::{Filter, Photometry},
    };

    use crate::{
        astro_math::arcsec_to_rad, spacetime_bucket::bucket::build_alert_bucket_index,
        topocentric_kf::kalman_bank::ellipse_region_finder::SearchComponent,
    };

    fn mk_observation(id: u64, ra: f64, dec: f64, mjd_tt: f64) -> Observation {
        let obs_dataset = ObsDataset::empty();
        let pos_err = arcsec_to_rad(0.5);
        let equ = EquCoord::new(ra, pos_err, dec, pos_err);
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

    /// Build a single-component region: an isotropic Gaussian of `sigma_rad`
    /// standard deviation centered at `(center_ra, center_dec)`, mirroring the
    /// shape `KFBank::predict_search_region` would produce for one hypothesis.
    fn mk_region(center_ra: f64, center_dec: f64, sigma_rad: f64, gate_chi2: f64) -> SearchRegion {
        let s = Matrix2::identity() * (sigma_rad * sigma_rad);
        let component = SearchComponent::new(1.0, center_ra, center_dec, s, gate_chi2)
            .expect("s is positive-definite by construction");
        SearchRegion {
            center_ra,
            center_dec,
            radius_rad: gate_chi2.sqrt() * sigma_rad,
            components: vec![component],
        }
    }

    #[test]
    fn query_region_candidates_only_returns_neighbor_cells() {
        let spatial_binner = HealpixBinner::new(10);

        let center_ra = 1.0;
        let center_dec = 0.2;
        let radius = arcsec_to_rad(20.0);

        let near = mk_observation(0, center_ra + arcsec_to_rad(2.0), center_dec, 60000.0);
        // Far enough that it cannot land in a HEALPix cell reachable from the
        // 20" cone query.
        let far = mk_observation(1, center_ra + 1.0, center_dec, 60000.0);

        let obs = vec![near, far];
        let obs_refs: Vec<&Observation> = obs.iter().collect();

        let bucket_index = build_alert_bucket_index(
            obs_refs.iter().copied(),
            &spatial_binner,
            &SingleBinTimeBinner,
        );

        let region = SearchRegion {
            center_ra,
            center_dec,
            radius_rad: radius,
            components: Vec::new(),
        };

        let candidates = query_region_candidates(&bucket_index, &spatial_binner, &region);
        assert_eq!(candidates.len(), 1);
        assert_eq!(*candidates[0].id(), 0);
    }

    #[test]
    fn filter_candidates_applies_gate_then_likelihood_threshold() {
        let center_ra = 1.0;
        let center_dec = 0.2;
        let sigma = arcsec_to_rad(5.0);
        let gate_chi2 = 23.0; // ~99.999% gate, matches KFBankConfig::default().gate_chi2

        let region = mk_region(center_ra, center_dec, sigma, gate_chi2);

        // At the center: passes the gate and has the highest likelihood.
        let center_obs = mk_observation(0, center_ra, center_dec, 60000.0);
        // ~3 sigma away: still inside the gate, but with lower likelihood.
        let near_obs = mk_observation(1, center_ra + 3.0 * sigma, center_dec, 60000.0);
        // Far outside the gate entirely.
        let far_obs = mk_observation(2, center_ra + 1.0, center_dec, 60000.0);

        let candidates = [&center_obs, &near_obs, &far_obs];

        // Threshold above the near-center likelihood: only the exact center
        // point survives.
        let strict_threshold = region.mixture_likelihood(center_ra + 1.5 * sigma, center_dec);
        let strict_matches = filter_candidates(
            &region,
            candidates.iter().copied(),
            gate_chi2,
            strict_threshold,
        );
        let strict_ids: Vec<u64> = strict_matches.iter().map(|m| *m.observation.id()).collect();
        assert_eq!(strict_ids, vec![0]);

        // Threshold at zero: everything that passes the gate is kept, i.e.
        // the far-outside-the-gate point is still rejected regardless.
        let lenient_matches =
            filter_candidates(&region, candidates.iter().copied(), gate_chi2, 0.0);
        let mut lenient_ids: Vec<u64> = lenient_matches
            .iter()
            .map(|m| *m.observation.id())
            .collect();
        lenient_ids.sort_unstable();
        assert_eq!(lenient_ids, vec![0, 1]);
    }

    #[test]
    fn find_candidates_for_bank_wires_track_ids_through_to_the_result() {
        let center_ra = 1.0;
        let center_dec = 0.2;
        let sigma = arcsec_to_rad(5.0);
        let gate_chi2 = 23.0;

        let region = mk_region(center_ra, center_dec, sigma, gate_chi2);
        let center_obs = mk_observation(0, center_ra, center_dec, 60000.0);
        let obs = vec![center_obs];
        let obs_refs: Vec<&Observation> = obs.iter().collect();

        let spatial_binner = HealpixBinner::new(10);
        let bucket_index = build_alert_bucket_index(
            obs_refs.iter().copied(),
            &spatial_binner,
            &SingleBinTimeBinner,
        );

        let track_ids = vec![10u64, 11u64];
        let result = find_candidates_for_bank(
            &region,
            track_ids.clone(),
            &bucket_index,
            &spatial_binner,
            gate_chi2,
            0.0,
        );

        assert_eq!(result.track_ids, track_ids);
        assert_eq!(result.matches.len(), 1);
    }
}
