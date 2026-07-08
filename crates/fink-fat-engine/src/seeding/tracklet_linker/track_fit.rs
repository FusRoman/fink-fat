//! Incremental linear on-sky motion fit for a single tracklet candidate.
//!
//! A tracklet's angular position is modeled as linear in time over a single
//! night's short baseline: `ra(t) ≈ ra_ref + slope_ra · (t − t_ref)`,
//! `dec(t) ≈ dec_ref + slope_dec · (t − t_ref)`. The fit is built
//! incrementally, one observation at a time, using running least-squares
//! sums so that folding in a new observation is `O(1)` regardless of how
//! many points the track already holds — this matters because
//! [`super::link_tracklets`] evaluates every incoming observation against
//! every nearby active track, so per-track state must stay cheap to update.

use photom::{coordinates::equatorial::EquCoord, observation_dataset::observation::Observation};

use crate::topocentric_kf::single_kalman::update::wrap_angle;

/// Combined 2-D angular variance of a single observation, in the flat
/// tangent-plane projection used by [`TrackFit`].
///
/// Right ascension error is scaled by `cos(dec)` to convert it from an
/// angle-on-a-circle-of-latitude to a true angular distance, matching the
/// same projection [`TrackFit`] fits against. Declination error needs no
/// scaling.
pub(crate) fn observation_variance(coord: &EquCoord) -> f64 {
    let ra_error_angular = coord.ra_error * coord.dec.cos();
    ra_error_angular * ra_error_angular + coord.dec_error * coord.dec_error
}

/// Running least-squares fit of `(ra, dec)` as a linear function of time,
/// updated one observation at a time.
///
/// # Coordinate handling
///
/// Right ascension is fit as a small offset from the track's first
/// observation (`ra_ref`), wrapped into `(−π, π]` with [`wrap_angle`] before
/// being scaled by `cos(dec_ref)` into a locally-flat tangent coordinate.
/// This sidesteps the RA `0`/`2π` discontinuity without a full spherical
/// regression — an acceptable approximation over a single night's short
/// angular baseline, consistent with the small-motion assumption already
/// used elsewhere in intra-night seeding (e.g. `tracklet_geometry`).
///
/// Declination is fit around a fixed `dec_ref` (the first observation's
/// declination): re-projecting at every point would cost more than the
/// accuracy it buys back over a single night's small declination drift.
#[derive(Clone, Debug)]
pub(crate) struct TrackFit {
    /// Reference epoch (the first observation's `mjd_tt`). Times are fit
    /// relative to this epoch to keep the regression numerically well
    /// conditioned.
    epoch_ref: f64,
    /// Reference right ascension (first observation's `ra`), radians.
    ra_ref: f64,
    /// Reference declination (first observation's `dec`), radians. Held
    /// fixed for the life of the track (see module docs).
    dec_ref: f64,

    /// Number of observations folded into the fit so far, including the
    /// first one used to construct it.
    n_points: usize,
    /// Σ(t − epoch_ref).
    sum_t: f64,
    /// Σ(t − epoch_ref)².
    sum_t2: f64,
    /// Σx, where x = wrap_angle(ra − ra_ref) · cos(dec_ref) is the projected
    /// right-ascension offset.
    sum_x: f64,
    /// Σ(t − epoch_ref)·x.
    sum_tx: f64,
    /// Σy, where y = dec − dec_ref.
    sum_y: f64,
    /// Σ(t − epoch_ref)·y.
    sum_ty: f64,

    /// Epoch of the most recently folded-in observation — bounds how far
    /// forward the track may still be extended (see `max_dt` gating in
    /// `super::link_tracklets`).
    last_epoch: f64,
    /// Apparent magnitude of the most recently folded-in observation, kept
    /// alongside the fit purely so the magnitude gate in
    /// `super::evaluate_candidate` doesn't need to re-resolve the track's
    /// last observation from its index.
    last_magnitude: f64,
    /// Running mean of each observation's own [`observation_variance`],
    /// used as the fit's assumed measurement-noise level when sizing the
    /// extrapolation-residual gate (see [`Self::prediction_variance`]).
    mean_obs_variance: f64,
}

impl TrackFit {
    /// Start a new track from its first observation.
    pub(crate) fn new(first: &Observation) -> Self {
        let coord = first.equ_coord();
        Self {
            epoch_ref: first.mjd_tt(),
            ra_ref: coord.ra,
            dec_ref: coord.dec,
            n_points: 1,
            sum_t: 0.0,
            sum_t2: 0.0,
            sum_x: 0.0,
            sum_tx: 0.0,
            sum_y: 0.0,
            sum_ty: 0.0,
            last_epoch: first.mjd_tt(),
            last_magnitude: first.photometry().magnitude,
            mean_obs_variance: observation_variance(coord),
        }
    }

    /// Number of observations folded into this fit so far.
    pub(crate) fn n_points(&self) -> usize {
        self.n_points
    }

    /// Epoch of the most recently folded-in observation.
    pub(crate) fn last_epoch(&self) -> f64 {
        self.last_epoch
    }

    /// Apparent magnitude of the most recently folded-in observation.
    pub(crate) fn last_magnitude(&self) -> f64 {
        self.last_magnitude
    }

    /// The track's first (and, while `n_points() == 1`, only) observed sky
    /// position — the position to gate a candidate against when no linear
    /// fit exists yet.
    pub(crate) fn first_position(&self) -> EquCoord {
        EquCoord::new(self.ra_ref, 0.0, self.dec_ref, 0.0)
    }

    /// Fold a new observation into the fit.
    ///
    /// The first point (folded in at construction, in [`Self::new`])
    /// contributes exactly `(t, x, y) = (0, 0, 0)` to the running sums by
    /// construction, since `epoch_ref`/`ra_ref`/`dec_ref` are that point's
    /// own coordinates — so no special-casing is needed here for the
    /// second observation onward.
    pub(crate) fn push(&mut self, obs: &Observation) {
        let coord = obs.equ_coord();
        let t = obs.mjd_tt() - self.epoch_ref;
        let x = wrap_angle(coord.ra - self.ra_ref) * self.dec_ref.cos();
        let y = coord.dec - self.dec_ref;

        self.sum_t += t;
        self.sum_t2 += t * t;
        self.sum_x += x;
        self.sum_tx += t * x;
        self.sum_y += y;
        self.sum_ty += t * y;
        self.n_points += 1;
        self.last_epoch = obs.mjd_tt();
        self.last_magnitude = obs.photometry().magnitude;

        // Incremental running mean: avoids re-summing every past
        // observation's variance on each push.
        let variance = observation_variance(coord);
        self.mean_obs_variance += (variance - self.mean_obs_variance) / self.n_points as f64;
    }

    /// Ordinary-least-squares slope/intercept for the projected RA (`x`)
    /// and Dec (`y`) axes, derived from the running sums.
    ///
    /// Only meaningful once `n_points() >= 2` (a single point has zero
    /// variance in `t`, so the fit is degenerate — callers must check
    /// [`Self::n_points`] before using this).
    fn linear_coefficients(&self) -> (f64, f64, f64, f64) {
        let n = self.n_points as f64;
        let denom = (n * self.sum_t2 - self.sum_t * self.sum_t).max(f64::EPSILON);

        let slope_x = (n * self.sum_tx - self.sum_t * self.sum_x) / denom;
        let intercept_x = (self.sum_x - slope_x * self.sum_t) / n;
        let slope_y = (n * self.sum_ty - self.sum_t * self.sum_y) / denom;
        let intercept_y = (self.sum_y - slope_y * self.sum_t) / n;

        (slope_x, intercept_x, slope_y, intercept_y)
    }

    /// Predict the sky position at `epoch`, extrapolating the current
    /// linear fit. Only meaningful once `n_points() >= 2`.
    pub(crate) fn predict(&self, epoch: f64) -> EquCoord {
        let (slope_x, intercept_x, slope_y, intercept_y) = self.linear_coefficients();
        let t = epoch - self.epoch_ref;

        let x = intercept_x + slope_x * t;
        let y = intercept_y + slope_y * t;

        let ra = self.ra_ref + x / self.dec_ref.cos();
        let dec = self.dec_ref + y;
        EquCoord::new(ra, 0.0, dec, 0.0)
    }

    /// Variance of [`Self::predict`]'s extrapolation at `epoch`, using the
    /// standard ordinary-least-squares prediction-variance formula
    /// `σ² · (1/n + (t − t̄)² / Sxx)`, where `σ²` is approximated by
    /// [`Self::mean_obs_variance`] (the fit treats every folded-in
    /// observation as having that same combined positional variance) and
    /// `Sxx` is the sum of squared deviations of `t` from its mean.
    ///
    /// The `(t − t̄)² / Sxx` "leverage" term grows the further `epoch` is
    /// extrapolated beyond the track's own observed time span — this is
    /// what lets the gate in `super::evaluate_candidate` stay tight near
    /// the track's recent observations while still tolerating a longer
    /// reach forward for a track with many, well-spread points.
    ///
    /// Only meaningful once `n_points() >= 2`.
    pub(crate) fn prediction_variance(&self, epoch: f64) -> f64 {
        let n = self.n_points as f64;
        let t = epoch - self.epoch_ref;
        let t_mean = self.sum_t / n;
        let sum_sq_deviation_t = (self.sum_t2 - self.sum_t * self.sum_t / n).max(f64::EPSILON);

        let leverage = 1.0 / n + (t - t_mean) * (t - t_mean) / sum_sq_deviation_t;
        self.mean_obs_variance * leverage
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use photom::{
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

    /// A track fit through exactly two points must reproduce the simple
    /// two-point finite-difference rate exactly (no bias from the OLS
    /// machinery when there is nothing to average over).
    #[test]
    fn two_point_fit_matches_finite_difference() {
        let a = mk_obs(0, 1.0, 0.2, 60000.0, 20.0);
        let b = mk_obs(1, 1.0 + 1e-4, 0.2 + 2e-4, 60000.05, 20.1);

        let mut fit = TrackFit::new(&a);
        fit.push(&b);

        let predicted_at_b = fit.predict(b.mjd_tt());
        let residual = predicted_at_b.angular_separation(b.equ_coord());
        assert!(
            residual < 1e-12,
            "fit through 2 points should pass through both exactly, residual={residual}"
        );
    }

    /// A track fit through several points lying exactly on a line should
    /// extrapolate forward with ~zero residual.
    #[test]
    fn linear_track_extrapolates_accurately() {
        let dec0 = 0.3;
        let ra_rate = 2e-4; // rad/day
        let dec_rate = -1e-4; // rad/day

        let mut fit: Option<TrackFit> = None;
        for k in 0..5 {
            let t = 60000.0 + k as f64 * 0.02;
            let obs = mk_obs(
                k,
                1.0 + ra_rate * (t - 60000.0),
                dec0 + dec_rate * (t - 60000.0),
                t,
                20.0,
            );
            match &mut fit {
                None => fit = Some(TrackFit::new(&obs)),
                Some(f) => f.push(&obs),
            }
        }
        let fit = fit.unwrap();

        let t_future = 60000.0 + 0.5;
        let predicted = fit.predict(t_future);
        let expected = EquCoord::new(1.0 + ra_rate * 0.5, 0.0, dec0 + dec_rate * 0.5, 0.0);
        let residual = predicted.angular_separation(&expected);
        assert!(
            residual < 1e-10,
            "extrapolation off a perfectly linear track should be near-exact, residual={residual}"
        );
    }

    /// Prediction variance should shrink as more points are folded in, and
    /// grow the further the query epoch is extrapolated beyond the track's
    /// observed span.
    #[test]
    fn prediction_variance_grows_with_extrapolation_distance() {
        let mut fit = TrackFit::new(&mk_obs(0, 1.0, 0.2, 60000.0, 20.0));
        fit.push(&mk_obs(1, 1.0001, 0.2001, 60000.02, 20.0));
        fit.push(&mk_obs(2, 1.0002, 0.2002, 60000.04, 20.0));

        let near = fit.prediction_variance(60000.04);
        let far = fit.prediction_variance(60000.5);
        assert!(
            far > near,
            "extrapolating far beyond the observed span should carry more variance"
        );
    }
}
