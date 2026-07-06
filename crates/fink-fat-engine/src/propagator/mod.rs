pub mod index;

use nalgebra::{Matrix2, Vector2};
use photom::coordinates::ecliptic::{EclipticCoord, EclipticCoordCov};

use crate::tracklet::Tracklet;

/// Predicts the position of a tracklet at a given epoch.
///
/// Both seeds and Kalman filters implement this trait, providing a uniform
/// interface for observation association regardless of the internal state
/// representation.
pub trait Propagator {
    /// Predicted position type (e.g. ecliptic coordinates).
    type Position;

    /// Uncertainty associated with the predicted position.
    type Uncertainty;

    /// Predict the position at epoch `t` (MJD).
    ///
    /// Arguments
    /// ---------
    /// * `t` – Target epoch in MJD.
    ///
    /// Return
    /// ------
    /// * `Some((position, uncertainty))` – If prediction is valid at `t`.
    /// * `None` – If `t` is outside the valid extrapolation range.
    fn predict(&self, t: f64) -> Option<(Self::Position, Self::Uncertainty)>;

    /// Last epoch at which this propagator was updated (MJD).
    fn last_epoch(&self) -> f64;

    /// Number of observations assimilated so far.
    fn obs_count(&self) -> usize;
}

/// Maximum extrapolation window allowed for a seed (days).
///
/// Predictions requested beyond this horizon are considered unreliable
/// and [`Propagator::predict`] returns `None` for [`Tracklet::Seed`].
const SEED_MAX_EXTRAPOLATION_DAYS: f64 = 30.0;

impl Propagator for Tracklet {
    /// Predicted ecliptic position  $ (\lambda, \beta) $ .
    type Position = EclipticCoord;
    /// Full  $ 2 \times 2 $  position covariance on  $ (\lambda, \beta) $ .
    type Uncertainty = EclipticCoordCov;

    /// Predict the ecliptic position and its uncertainty at epoch `t` (MJD).
    ///
    /// The state is propagated via [`EclipticState::propagate`] and the
    /// resulting  $ (\lambda, \beta) $  covariance is extracted with
    /// [`EclipticState::to_ecliptic_coord_cov`].
    ///
    /// For [`Tracklet::Seed`], returns `None` when
    ///  $ |t - t_\text{epoch}| > \text{SEED\_MAX\_EXTRAPOLATION\_DAYS} $ .
    ///
    /// For [`Tracklet::Filter`], the prediction is always attempted.
    fn predict(&self, t: f64) -> Option<(EclipticCoord, EclipticCoordCov)> {
        let data = self.ecliptic_data()?;

        if let Tracklet::Seed(_) = self {
            if (t - data.state.epoch).abs() > SEED_MAX_EXTRAPOLATION_DAYS {
                return None;
            }
        }

        let propagated = match &data.state.singer {
            Some(singer) => data.state.propagate_singer(t, singer),
            None => data.state.propagate(t, data.state.process_noise_q),
        };

        let coord_cov = propagated.to_ecliptic_coord_cov();
        Some((coord_cov.coord, coord_cov))
    }

    fn last_epoch(&self) -> f64 {
        self.epoch()
    }

    fn obs_count(&self) -> usize {
        self.obs_keys().len()
    }
}

/// A 2-D axis-aligned bounding box in ecliptic coordinates (radians).
#[derive(Debug, Clone, Copy)]
pub struct EclBBox {
    pub lon_min: f64,
    pub lon_max: f64,
    pub lat_min: f64,
    pub lat_max: f64,
}

impl EclBBox {
    /// Returns `true` if `coord` falls inside the bounding box.
    #[inline]
    pub fn contains(&self, coord: &EclipticCoord) -> bool {
        coord.lon >= self.lon_min
            && coord.lon <= self.lon_max
            && coord.lat >= self.lat_min
            && coord.lat <= self.lat_max
    }

    /// Returns the angular half-widths in longitude and latitude (radians).
    #[inline]
    pub fn half_widths(&self) -> (f64, f64) {
        (
            (self.lon_max - self.lon_min) * 0.5,
            (self.lat_max - self.lat_min) * 0.5,
        )
    }

    /// Returns the maximum angular half-width across both axes (radians).
    #[inline]
    pub fn max_half_width(&self) -> f64 {
        let (dlon, dlat) = self.half_widths();
        dlon.max(dlat)
    }
}

/// Spatial and statistical utilities for a predicted ecliptic position.
///
/// Implemented for types that carry both a predicted position and its
/// associated covariance (e.g. [`EclipticCoordCov`]).
pub trait PredictionGeometry {
    /// Compute an axis-aligned bounding box in **ecliptic coordinates**
    /// around the predicted position.
    ///
    /// The prediction covariance $P_\text{pred}$ is expressed in the ecliptic
    /// frame $(\lambda, \beta)$. To build a bbox in the equatorial frame
    /// $(\alpha, \delta)$, the uncertainty is propagated through the ecliptic
    /// → equatorial Jacobian $J$:
    ///
    /// $$P_\text{equ} = J \, P_\text{pred} \, J^\top$$
    ///
    /// The bbox is then constructed from the marginal standard deviations of
    /// $P_\text{equ}$, scaled by $k$:
    ///
    /// $$\text{bbox} = \bigl[\alpha \pm k\,\sigma_\alpha,\; \delta \pm k\,\sigma_\delta\bigr]$$
    ///
    /// This is a **conservative** approximation: the true $k$-sigma ellipse in
    /// equatorial coordinates is always contained within this box.
    ///
    /// Arguments
    /// ---------
    /// * `k` – Number of standard deviations. Typical values: `3.0`–`5.0`.
    ///
    /// Return
    /// ------
    /// * [`EclBBox`] centered on the predicted position in ecliptic coordinates.
    fn bounding_box(&self, k: f64) -> EclBBox;

    /// Compute the squared Mahalanobis distance between the prediction and an
    /// observation.
    ///
    /// The innovation covariance $S$ combines the prediction uncertainty
    /// $P_\text{pred}$ and the observation noise $R$:
    ///
    /// $$S = P_\text{pred} + R$$
    ///
    /// The squared Mahalanobis distance is then:
    ///
    /// $$d^2 = \Delta\mathbf{x}^\top S^{-1} \Delta\mathbf{x}$$
    ///
    /// where $\Delta\mathbf{x} = [\lambda_\text{obs} - \lambda_\text{pred},\;
    /// \beta_\text{obs} - \beta_\text{pred}]^\top$.
    ///
    /// Under the null hypothesis that the observation belongs to the predicted
    /// tracklet, $d^2$ follows a $\chi^2$ distribution with 2 degrees of
    /// freedom. A common acceptance threshold is $\chi^2_2(0.99) \approx 9.21$.
    ///
    /// Arguments
    /// ---------
    /// * `obs` – Observed ecliptic position with its measurement covariance.
    ///
    /// Return
    /// ------
    /// * `Some(d²)` – Squared Mahalanobis distance.
    /// * `None` – If the innovation covariance $S$ is not invertible.
    fn mahalanobis_sq(&self, obs: &EclipticCoordCov) -> Option<f64>;
}

impl PredictionGeometry for EclipticCoordCov {
    fn bounding_box(&self, k: f64) -> EclBBox {
        let sigma_lon = self.cov.xx.sqrt();
        let sigma_lat = self.cov.yy.sqrt();

        EclBBox {
            lon_min: self.coord.lon - k * sigma_lon,
            lon_max: self.coord.lon + k * sigma_lon,
            lat_min: self.coord.lat - k * sigma_lat,
            lat_max: self.coord.lat + k * sigma_lat,
        }
    }

    fn mahalanobis_sq(&self, obs: &EclipticCoordCov) -> Option<f64> {
        let dx = Vector2::new(
            obs.coord.lon - self.coord.lon,
            obs.coord.lat - self.coord.lat,
        );

        let s = Matrix2::new(
            self.cov.xx + obs.cov.xx,
            self.cov.xy + obs.cov.xy,
            self.cov.xy + obs.cov.xy,
            self.cov.yy + obs.cov.yy,
        );

        let chol = s.cholesky()?;
        let s_inv_dx = chol.solve(&dx);

        Some(dx.dot(&s_inv_dx))
    }
}
