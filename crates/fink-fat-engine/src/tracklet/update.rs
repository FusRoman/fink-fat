//! # Tracklet update after inter-night association
//!
//! This module applies Kalman filter updates to active tracklets after the
//! inter-night association step has produced a set of candidate observations
//! for each tracklet.
//!
//! ## Update pipeline
//!
//! For each association produced by [`generate_next_night_candidate`], the
//! following steps are applied in order:
//!
//! 1. **Chronological sorting** — candidate observations are sorted by
//!    exposure time (MJD TT). This ensures that the Kalman filter assimilates
//!    measurements in temporal order, which is required for a correct
//!    predict–update cycle.
//!
//! 2. **Sequential Kalman update** — for each observation:
//!    - the state is propagated to the observation epoch via
//!      [`EclipticState::propagate`] or [`EclipticState::propagate_singer`],
//!      depending on whether Singer parameters are configured,
//!    - the Kalman correction is applied via [`EclipticState::kalman_update`].
//!    - If the innovation covariance is singular, the observation is skipped.
//!
//! 3. **Observation key registration** — the [`ObsId`] of each successfully
//!    assimilated observation is appended to the tracklet's `obs_keys`.
//!
//! 4. **Variant transition** — the updated tracklet is always emitted as
//!    [`Tracklet::Filter`], regardless of whether the input was a
//!    [`Tracklet::Seed`] or a [`Tracklet::Filter`].
//!
//! Tracklets for which all Kalman updates fail (singular innovation covariance
//! for every candidate) are dropped from the output.

use photom::{
    coordinates::ecliptic::EclipticCoordCov, observation_dataset::observation::Observation,
};

use crate::{
    ecliptic_state::EclipticState,
    tracklet::{Tracklet, tracklet_data::TrackletData},
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Propagate an [`EclipticState`] to the epoch of `obs` using the model
/// configured in `data`.
///
/// Selects Singer propagation if `data.singer` is `Some`, otherwise falls
/// back to the constant-acceleration model with spectral density
/// `data.process_noise_q`.
///
/// Arguments
/// ---------
/// * `state` – State to propagate (borrowed, not mutated).
/// * `data`  – Tracklet data carrying propagation parameters.
/// * `obs`   – Target observation (epoch taken from `obs.mjd_tt()`).
///
/// Return
/// ------
/// A new [`EclipticState`] valid at `obs.mjd_tt()`.
#[inline]
fn propagate_to_obs(
    state: &EclipticState,
    data: &TrackletData<EclipticState>,
    obs: &Observation,
) -> EclipticState {
    match &data.state.singer {
        Some(singer) => state.propagate_singer(obs.mjd_tt(), singer),
        None => state.propagate(obs.mjd_tt(), data.state.process_noise_q),
    }
}

// ---------------------------------------------------------------------------
// Single-tracklet update
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Apply a single Kalman predict–update cycle to a tracklet for one observation.
///
/// The state is propagated from the tracklet's last known epoch to
/// `obs.mjd_tt()`, then corrected by the Kalman filter using the ecliptic
/// projection of `obs`. This method promote every tracklet seed variant
/// into a tracklet filter variant if the update is successful.
///
/// Arguments
/// ---------
/// * `tracklet` – Source tracklet (state is read, not mutated).
/// * `obs`      – Single candidate observation to assimilate.
///
/// Return
/// ------
/// * `Some(Tracklet::Filter)` – Updated tracklet with `obs` appended to
///   `obs_keys`.
/// * `None` – Innovation covariance is singular; observation cannot be
///   assimilated.
fn update_single_obs(tracklet: &Tracklet, obs: &Observation) -> Option<Tracklet> {
    let data = tracklet.ecliptic_data()?;

    let (old_ref_mag, old_ref_mag_err) = tracklet.get_ref_mag();

    let propagated = propagate_to_obs(&data.state, data, obs);
    let ecl_obs: EclipticCoordCov = (*obs.equ_coord()).into();

    let updated_state = propagated.kalman_update(&ecl_obs)?;

    let mut new_obs_keys = data.obs_keys.clone();
    new_obs_keys.push(obs.id().clone());

    let phot = obs.photometry();
    let w_new_obs = 1.0 / (phot.error * phot.error);

    let w_old = 1.0 / (old_ref_mag_err * old_ref_mag_err);
    let w_total = w_old + w_new_obs;

    let new_ref_mag = (w_old * old_ref_mag + w_new_obs * phot.magnitude) / w_total;
    let new_ref_mag_err = 1.0 / w_total.sqrt();

    let updated_data = TrackletData {
        key: data.key,
        state: updated_state,
        obs_keys: new_obs_keys,
        ref_mag: new_ref_mag,
        ref_mag_err: new_ref_mag_err,
    };

    Some(Tracklet::Filter(updated_data))
}

/// Produce one updated branch per candidate observation.
///
/// Each branch is an independent copy of the parent tracklet updated with
/// exactly one candidate. The `key` field in each branch is left identical
/// to the parent's key; the caller is responsible for assigning a fresh
/// [`TrackId`] before insertion into storage.
///
/// Candidates that produce a singular innovation covariance are silently
/// dropped.
///
/// Arguments
/// ---------
/// * `tracklet`   – Parent tracklet.
/// * `candidates` – Candidate observations (order is irrelevant here).
///
/// Return
/// ------
/// * `Vec<Tracklet>` – One [`Tracklet::Filter`] per successfully assimilated
///   candidate. Empty if all updates fail.
pub fn branch_tracklet<'o>(tracklet: &Tracklet, candidates: Vec<&'o Observation>) -> Vec<Tracklet> {
    candidates
        .into_iter()
        .filter_map(|obs| update_single_obs(tracklet, obs))
        .collect()
}
