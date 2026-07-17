use crate::engine_config::units::de_angle_arcsec;
use nalgebra::Matrix2;
use serde::{Deserialize, Serialize};

use crate::topocentric_kf::kalman_bank::ellipse_region_finder::SearchComponent;

/// Inner strategy choice for [`RadiusStrategy::Clamped`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MixOrMax {
    MaxEllipse,
    MixtureCovariance,
}

/// Controls how the bounding radius of a [`SearchRegion`](super::SearchRegion) is computed.
///
/// Variants
/// --------
/// - [`RadiusStrategy::MaxEllipse`] – conservative: radius is the maximum
///   over selected hypotheses of
///   $$r_i = \sqrt{\chi^2_{gate} \cdot \lambda_{max}(S_i)} + \|\mu_i - \bar\mu\|$$
///   Correct but can be very large when hypotheses are spatially dispersed.
/// - [`RadiusStrategy::MixtureCovariance`] – computes the full mixture
///   covariance
///   $$S_{mix} = \sum_i w_i \bigl(S_i + (\mu_i - \bar\mu)(\mu_i -
///   \bar\mu)^\top\bigr)$$
///   and sets $r = \sqrt{\chi^2_{gate} \cdot \lambda_{max}(S_{mix})}$.
///   Tighter in practice; still conservative because both the within-component
///   spread ($S_i$) and the between-component spread are included.
/// - [`RadiusStrategy::Clamped`] – applies an inner strategy then clamps the
///   result to a hard maximum expressed in arcseconds.  Used as a safety net
///   when the bank has not yet converged and the mixture covariance can still
///   be large.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum RadiusStrategy {
    /// Original conservative behaviour.
    MaxEllipse,
    /// Tighter mixture-covariance bound.
    #[default]
    MixtureCovariance,
    /// Hard clamp applied on top of another strategy.
    Clamped {
        /// Strategy to apply before clamping.
        inner: MixOrMax,
        /// Maximum allowed radius (arcseconds).
        #[serde(deserialize_with = "de_angle_arcsec")]
        max_arcsec: f64,
    },
}

impl RadiusStrategy {
    /// Dispatch radius computation over `components` according to `self`.
    pub fn radius(self, components: &[SearchComponent], center_ra: f64, center_dec: f64) -> f64 {
        match self {
            RadiusStrategy::MaxEllipse => {
                Self::max_ellipse_radius(components, center_ra, center_dec)
            }
            RadiusStrategy::MixtureCovariance => {
                Self::mixture_covariance_radius(components, center_ra, center_dec)
            }
            RadiusStrategy::Clamped { inner, max_arcsec } => {
                let inner_strategy = match inner {
                    MixOrMax::MaxEllipse => RadiusStrategy::MaxEllipse,
                    MixOrMax::MixtureCovariance => RadiusStrategy::MixtureCovariance,
                };
                let r = inner_strategy.radius(components, center_ra, center_dec);
                r.min((max_arcsec / 3600.0_f64).to_radians())
            }
        }
    }

    /// $$r = \max_i \left( \sqrt{\chi^2 \cdot \lambda_{max}(S_i)} + \|\mu_i - \bar\mu\| \right)$$
    fn max_ellipse_radius(components: &[SearchComponent], center_ra: f64, center_dec: f64) -> f64 {
        components
            .iter()
            .map(|c| c.per_hypothesis_radius(center_ra, center_dec))
            .fold(0.0_f64, f64::max)
    }

    /// Compute the bounding radius from the full mixture covariance.
    ///
    /// $$S_{mix} = \sum_i w_i \bigl(S_i + (\mu_i - \bar\mu)(\mu_i - \bar\mu)^\top\bigr)$$
    /// $$r = \sqrt{\chi^2 \cdot \lambda_{max}(S_{mix})}$$
    ///
    /// Both the within-component uncertainty ($S_i$) and the between-component
    /// spatial spread contribute to $S_{mix}$, so the radius is a valid
    /// conservative bound on the mixture support.
    fn mixture_covariance_radius(
        components: &[SearchComponent],
        center_ra: f64,
        center_dec: f64,
    ) -> f64 {
        let mut s_mix = Matrix2::zeros();
        let mut gate_chi2 = 0.0;
        for c in components {
            let delta = c.offset_from(center_ra, center_dec);
            s_mix += c.weight * (c.s + delta * delta.transpose());
            gate_chi2 = c.gate_chi2;
        }
        gate_chi2.sqrt() * largest_eigenvalue_2x2(&s_mix).sqrt()
    }
}

/// Largest eigenvalue of a symmetric $2 \times 2$ matrix via the analytic
/// formula.
///
/// For $S = \begin{pmatrix} a & b \\ b & d \end{pmatrix}$:
///
/// $$\lambda_{max} = \frac{a+d}{2} + \sqrt{\left(\frac{a-d}{2}\right)^2 + b^2}$$
pub fn largest_eigenvalue_2x2(s: &Matrix2<f64>) -> f64 {
    let a = s[(0, 0)];
    let d = s[(1, 1)];
    let b = s[(0, 1)];
    let mid = (a + d) / 2.0;
    let half_diff = (a - d) / 2.0;
    mid + (half_diff * half_diff + b * b).sqrt()
}
