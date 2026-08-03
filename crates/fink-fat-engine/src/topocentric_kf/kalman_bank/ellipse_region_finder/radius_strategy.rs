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
    /// Bascule entre deux stratégies selon la taille du mélange sélectionné
    /// — `MaxEllipse` (insensible aux poids, donc robuste à l'ambiguïté
    /// ρ/ρ̇ post-bootstrap où les poids ne sont pas encore informatifs) tant
    /// que la banque est grande, `MixtureCovariance` clampé une fois
    /// qu'elle a convergé vers un petit nombre d'hypothèses. Voir
    /// l'investigation `not_matched%` aux steps 1/2 dans `mot_analysis`.
    AdaptiveConvergence {
        /// Si `components.len()` dépasse ce seuil, la banque est considérée
        /// non convergée : `MaxEllipse`, clampé à
        /// `unconverged_clamp_arcsec` (généreux — c'est justement le clamp
        /// serré actuel qui coupe les modes corrects mais éloignés).
        max_hypotheses_for_mixture: usize,
        #[serde(deserialize_with = "de_angle_arcsec")]
        unconverged_clamp_arcsec: f64,
        /// Sinon (banque élaguée, poids informatifs) : comportement
        /// actuel, `MixtureCovariance` clampé à `converged_clamp_arcsec`.
        #[serde(deserialize_with = "de_angle_arcsec")]
        converged_clamp_arcsec: f64,
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
            RadiusStrategy::AdaptiveConvergence {
                max_hypotheses_for_mixture,
                unconverged_clamp_arcsec,
                converged_clamp_arcsec,
            } => {
                let (strategy, clamp_arcsec) = if components.len() > max_hypotheses_for_mixture {
                    (RadiusStrategy::MaxEllipse, unconverged_clamp_arcsec)
                } else {
                    (RadiusStrategy::MixtureCovariance, converged_clamp_arcsec)
                };
                let r = strategy.radius(components, center_ra, center_dec);
                r.min((clamp_arcsec / 3600.0_f64).to_radians())
            }
        }
    }

    /// Hard cap this strategy applies to any radius it returns, in radians,
    /// for a mixture of `n_components` hypotheses.
    ///
    /// `None` for the unclamped variants — those return whatever radius they
    /// compute, so they can never truncate the mixture.
    ///
    /// Used for two things, both needing the clamp that *actually applies*:
    /// [`SearchRegion::radius_pinned_at_clamp`](super::SearchRegion::radius_pinned_at_clamp)
    /// tests whether the radius came back pinned at it (the signal that the
    /// coarse cone may be hiding modes), and
    /// [`sky_cover_regions`](super::sky_cover_regions) uses it as the tiling
    /// granularity — a single cone can never usefully exceed this radius, so
    /// cones of exactly this half-size are the coarsest tiling that loses
    /// nothing to the clamp.
    ///
    /// `n_components` matters because [`RadiusStrategy::AdaptiveConvergence`]
    /// switches between its converged and unconverged clamps on exactly that
    /// count; reporting the wrong one would compare the radius against a clamp
    /// that was never applied.
    pub fn clamp_rad(self, n_components: usize) -> Option<f64> {
        let arcsec = match self {
            RadiusStrategy::MaxEllipse | RadiusStrategy::MixtureCovariance => return None,
            RadiusStrategy::Clamped { max_arcsec, .. } => max_arcsec,
            RadiusStrategy::AdaptiveConvergence {
                max_hypotheses_for_mixture,
                unconverged_clamp_arcsec,
                converged_clamp_arcsec,
            } => {
                if n_components > max_hypotheses_for_mixture {
                    unconverged_clamp_arcsec
                } else {
                    converged_clamp_arcsec
                }
            }
        };
        Some((arcsec / 3600.0_f64).to_radians())
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
