//! Incremental linear-motion tracklet linker.
//!
//! [`pairs::generate_pairs`](super::pairs::generate_pairs) enumerates every
//! `(a, b)` observation pair consistent with its gates — including every
//! combination between observations of the *same* object on a night with
//! many intra-night revisits, which is an `O(n²)` blowup carrying no extra
//! seeding value (a single tracklet already suffices to seed a lineage for
//! that object). [`link_tracklets`] replaces that enumeration with a
//! time-ordered sweep that links each object's detections into one growing
//! **track**, and emits a single [`Pair`] per track spanning its first and
//! last observation (maximizing the baseline, which minimizes the
//! astrometric-noise error on the derived angular rate fed into
//! `admissible_region_grid`).
//!
//! # Algorithm
//!
//! Observations are processed in time order. A pool of active tracks is
//! maintained, each holding an incremental linear `(ra, dec)` vs. time fit
//! (`track_fit::TrackFit`). For every new observation, the best-matching
//! nearby active track is found (`evaluate_candidate`) and extended; if
//! none matches, a new single-observation track is started. Candidate
//! lookup is restricted to tracks in the observation's own sky neighborhood
//! (`active_tracks::ActiveTracks`), keeping each step close to `O(1)`
//! amortized rather than scanning every active track.
//!
//! Two different gates apply depending on how much history a candidate
//! track already has:
//! - **No fit yet (1 observation)**: falls back to the same pairwise
//!   `max_angular_speed` dot-product gate `generate_pairs` uses today. This
//!   is the *only* path exercised on a "normal cadence" night (≤2
//!   detections/object), so behavior there is unchanged.
//! - **Fit established (≥2 observations)**: gates on the angular distance
//!   between the observation and the track's own linear extrapolation,
//!   normalized by a statistically-derived tolerance (see
//!   `track_fit::TrackFit::prediction_variance`) rather than the generic
//!   population-wide speed cone — this is what lets a dense-revisit object
//!   collapse to one track instead of many redundant pairs, without also
//!   bridging together two different nearby objects.

mod active_tracks;
mod track_fit;

use photom::observation_dataset::observation::Observation;

use crate::{
    engine_config::pair_config::PairConfig,
    seeding::pairs::{Pair, Pairs},
    spacetime_bucket::spatial_binner::{SpatialBinner, SpatialKey},
};

use self::active_tracks::{ActiveTracks, Track};
use self::track_fit::{TrackFit, observation_variance};

/// Link a night's observations into per-object tracklets and emit one
/// maximal-baseline [`Pair`] per multi-detection object.
///
/// See the module docs for the algorithm. `night_obs` need not be
/// pre-sorted by time; this function sorts its own working copy.
pub fn link_tracklets<'alert_lf, Bs: SpatialBinner>(
    night_obs: &[&'alert_lf Observation],
    spatial_binner: &Bs,
    config: &PairConfig,
) -> Pairs<'alert_lf> {
    let mut sorted_obs: Vec<&'alert_lf Observation> = night_obs.to_vec();
    sorted_obs.sort_unstable_by(|a, b| a.mjd_tt().total_cmp(&b.mjd_tt()));

    let mut active_tracks = ActiveTracks::default();

    // Same conservative bound `generate_pairs` uses for its own bucket
    // neighborhood search: the furthest a linked candidate could plausibly
    // be, given the population-wide speed cap, plus one cell radius so a
    // cell-boundary split never hides a real candidate.
    let search_radius = config.sep_cap() + spatial_binner.cell_radius();
    let mut neighbor_cells: Vec<SpatialKey> = Vec::new();

    for (observation_index, &observation) in sorted_obs.iter().enumerate() {
        let epoch = observation.mjd_tt();
        let cell = spatial_binner.key_for(observation.equ_coord());
        spatial_binner.neighbors_into(cell, search_radius, &mut neighbor_cells);

        let candidate_track_ids =
            active_tracks.candidates_near(&neighbor_cells, epoch, config.max_dt);
        let best_track_id =
            select_best_candidate(&active_tracks, &candidate_track_ids, observation, config);

        match best_track_id {
            Some(track_id) => {
                active_tracks.extend_track(track_id, observation_index, observation, cell);
            }
            None => {
                active_tracks.start_track(observation_index, TrackFit::new(observation), cell);
            }
        }
    }

    extract_endpoint_pairs(&sorted_obs, active_tracks.into_tracks())
}

/// How well a candidate observation matches an existing track, used to rank
/// competing candidate tracks for the same observation.
///
/// A fit-based match always outranks a fallback match: an object with an
/// established trajectory is a far more specific (and therefore more
/// trustworthy) signal than the generic population-wide speed cone used
/// before any fit exists.
#[derive(Clone, Copy, PartialEq)]
enum MatchScore {
    /// Matched a single-observation track via the plain pairwise
    /// angular-speed gate (no fit to extrapolate from yet).
    Fallback { dt: f64 },
    /// Matched an established fit; `normalized_residual` is the
    /// extrapolation residual in units of its own gating tolerance (smaller
    /// is a tighter match).
    Fitted { normalized_residual: f64 },
}

impl MatchScore {
    fn is_better_than(self, other: Self) -> bool {
        match (self, other) {
            (Self::Fitted { .. }, Self::Fallback { .. }) => true,
            (Self::Fallback { .. }, Self::Fitted { .. }) => false,
            (
                Self::Fitted {
                    normalized_residual: a,
                },
                Self::Fitted {
                    normalized_residual: b,
                },
            ) => a < b,
            (Self::Fallback { dt: a }, Self::Fallback { dt: b }) => a < b,
        }
    }
}

/// Score every candidate track against `observation` and return the id of
/// the best-matching one, if any survives its gate.
fn select_best_candidate(
    active_tracks: &ActiveTracks,
    candidate_track_ids: &[usize],
    observation: &Observation,
    config: &PairConfig,
) -> Option<usize> {
    let mut best: Option<(usize, MatchScore)> = None;

    for &track_id in candidate_track_ids {
        let track = active_tracks.track(track_id);
        let Some(score) = evaluate_candidate(track, observation, config) else {
            continue;
        };
        let is_new_best = match best {
            None => true,
            Some((_, best_score)) => score.is_better_than(best_score),
        };
        if is_new_best {
            best = Some((track_id, score));
        }
    }

    best.map(|(track_id, _)| track_id)
}

/// Decide whether `observation` may extend `track`, and how good a match it
/// is (see [`MatchScore`]).
///
/// Both gate paths share two checks: chronological order (defensive — the
/// sweep is already chronological, so `dt <= 0` should not occur) and the
/// magnitude-similarity gate from [`PairConfig::max_mag_difference`].
fn evaluate_candidate(
    track: &Track,
    observation: &Observation,
    config: &PairConfig,
) -> Option<MatchScore> {
    let dt = observation.mjd_tt() - track.fit.last_epoch();
    if dt <= 0.0 {
        return None;
    }
    if (observation.photometry().magnitude - track.fit.last_magnitude()).abs()
        > config.max_mag_difference
    {
        return None;
    }

    if track.fit.n_points() >= 2 {
        fitted_match_score(track, observation, config)
    } else {
        fallback_match_score(track, observation, config, dt)
    }
}

/// Gate a candidate against an established linear fit: accept only if the
/// angular distance from the fit's extrapolation is within
/// `config.tracklet_residual_sigma` standard deviations of the combined
/// observation/extrapolation uncertainty.
fn fitted_match_score(
    track: &Track,
    observation: &Observation,
    config: &PairConfig,
) -> Option<MatchScore> {
    let epoch = observation.mjd_tt();
    let predicted_position = track.fit.predict(epoch);
    let residual = predicted_position.angular_separation(observation.equ_coord());

    let tolerance_variance =
        track.fit.prediction_variance(epoch) + observation_variance(observation.equ_coord());
    let tolerance = tolerance_variance.max(f64::EPSILON).sqrt();
    let normalized_residual = residual / tolerance;

    if normalized_residual > config.tracklet_residual_sigma {
        return None;
    }
    Some(MatchScore::Fitted {
        normalized_residual,
    })
}

/// Gate a candidate against a single-observation track with no fit yet,
/// using the same pairwise angular-speed dot-product test
/// `generate_pairs` applies today — this is the "normal cadence" fallback
/// path, deliberately left unchanged from existing behavior.
fn fallback_match_score(
    track: &Track,
    observation: &Observation,
    config: &PairConfig,
    dt: f64,
) -> Option<MatchScore> {
    let last_position = track.fit.first_position();
    let angular_speed = last_position.angular_separation(observation.equ_coord()) / dt;

    if angular_speed > config.max_angular_speed {
        return None;
    }
    Some(MatchScore::Fallback { dt })
}

/// Reduce every finished track with ≥2 members to the single [`Pair`]
/// spanning its first and last linked observation.
///
/// Single-observation tracks (never extended) are dropped — exactly like
/// today's `generate_pairs`, a lone detection can't seed a Kalman bank on
/// its own.
fn extract_endpoint_pairs<'alert_lf>(
    sorted_obs: &[&'alert_lf Observation],
    tracks: Vec<Track>,
) -> Pairs<'alert_lf> {
    let mut pairs: Pairs<'alert_lf> = tracks
        .into_iter()
        .filter(|track| track.member_indices.len() >= 2)
        .map(|track| {
            let first_index = *track
                .member_indices
                .first()
                .expect("filtered to len >= 2 above");
            let last_index = *track
                .member_indices
                .last()
                .expect("filtered to len >= 2 above");
            Pair {
                a: sorted_obs[first_index],
                b: sorted_obs[last_index],
            }
        })
        .collect();

    // Deterministic ordering, matching `generate_pairs`'s output contract.
    pairs.sort_unstable_by(|p1, p2| p1.a.cmp(p2.a).then_with(|| p1.b.cmp(p2.b)));
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spacetime_bucket::healpix_binner::HealpixBinner;
    use photom::{
        coordinates::equatorial::EquCoord,
        observation_dataset::{ObsDataset, observation::ObservationInput},
        photometry::{Filter, Photometry},
    };

    fn mk_obs(id: u64, ra: f64, dec: f64, mjd_tt: f64, mag: f64) -> Observation {
        let obs_dataset = ObsDataset::empty();
        let equ = EquCoord::new(ra, 1e-6, dec, 1e-6);
        let phot = Photometry {
            magnitude: mag,
            error: 0.1,
            filter: Filter::Int(1),
        };
        let input = ObservationInput::new(id, equ, phot, mjd_tt, None);
        let (obs_dataset, obs_id) = obs_dataset.push_observation(vec![input]).unwrap();
        obs_dataset
            .get_obs_by_index(*obs_id.get(0).unwrap())
            .unwrap()
            .clone()
    }

    fn default_test_config() -> PairConfig {
        PairConfig {
            max_dt: 15.0 / 1440.0,
            max_angular_speed: 0.02,
            allow_same_timebin: true,
            max_mag_difference: 1.0,
            ..Default::default()
        }
    }

    /// The "normal cadence" case (exactly 2 detections of one object):
    /// output must match a single pairwise link, exercising only the
    /// fallback (no-fit-yet) gate — no behavior change vs. plain pairing.
    #[test]
    fn two_observations_collapse_to_one_pair_via_fallback_gate() {
        let sb = HealpixBinner::new(8);
        let config = default_test_config();

        let a = mk_obs(0, 1.0, 0.2, 60000.0, 20.0);
        let b = mk_obs(1, 1.0 + 1e-5, 0.2 + 1e-5, 60000.0 + 5.0 / 1440.0, 20.05);
        let alerts = vec![a, b];
        let refs: Vec<&Observation> = alerts.iter().collect();

        let pairs = link_tracklets(&refs, &sb, &config);

        assert_eq!(pairs.len(), 1);
        assert_eq!(*pairs[0].a.id(), 0);
        assert_eq!(*pairs[0].b.id(), 1);
    }

    /// A dense multi-revisit object (linear track over several visits,
    /// interleaved with an unrelated second object) must collapse to
    /// exactly one pair per object, spanning first-to-last observation.
    #[test]
    fn dense_revisits_of_one_object_collapse_to_single_endpoint_pair() {
        let sb = HealpixBinner::new(8);
        let config = default_test_config();

        let dec0 = 0.3;
        let ra_rate = 1e-4; // rad/day, well inside max_angular_speed
        let t0 = 60000.0;
        let dt_step = 0.003; // days between revisits, well inside max_dt (15 min)

        let mut alerts = Vec::new();
        // Object A: 6 revisits on a straight line.
        for k in 0..6u64 {
            let t = t0 + k as f64 * dt_step;
            alerts.push(mk_obs(k, 1.0 + ra_rate * (t - t0), dec0, t, 20.0));
        }
        // Object B: unrelated, far away on the sky, interleaved in time.
        for k in 0..3u64 {
            let t = t0 + k as f64 * dt_step * 2.0 + 0.001;
            alerts.push(mk_obs(100 + k, 4.0, -0.5, t, 19.0));
        }

        let refs: Vec<&Observation> = alerts.iter().collect();
        let pairs = link_tracklets(&refs, &sb, &config);

        // One pair per object with >=2 detections.
        assert_eq!(pairs.len(), 2, "expected one endpoint pair per object");

        let object_a_pair = pairs
            .iter()
            .find(|p| *p.a.id() == 0)
            .expect("object A should produce a pair starting at its first observation");
        assert_eq!(
            *object_a_pair.b.id(),
            5,
            "object A's pair should span its first to its LAST observation, not an intermediate one"
        );
    }

    /// The fallback (no-fit) gate must still reject a candidate whose
    /// implied angular speed exceeds `max_angular_speed`, exactly like
    /// `generate_pairs`.
    #[test]
    fn fallback_gate_rejects_excessive_angular_speed() {
        let sb = HealpixBinner::new(8);
        let config = default_test_config();

        let a = mk_obs(0, 1.0, 0.2, 60000.0, 20.0);
        // Implied speed far above max_angular_speed over a short dt.
        let b = mk_obs(1, 1.5, 0.2, 60000.0 + 1.0 / 1440.0, 20.0);
        let alerts = vec![a, b];
        let refs: Vec<&Observation> = alerts.iter().collect();

        let pairs = link_tracklets(&refs, &sb, &config);
        assert!(pairs.is_empty());
    }
}
