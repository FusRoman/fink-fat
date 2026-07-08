pub mod radius_strategy;
pub mod top_k;

use nalgebra::{Matrix2, Vector2};

use crate::topocentric_kf::{
    kalman_bank::{
        KFBank,
        ellipse_region_finder::{
            radius_strategy::{RadiusStrategy, largest_eigenvalue_2x2},
            top_k::TopK,
        },
    },
    single_kalman::{KFState, propagate::PropagateError, update::wrap_angle},
};

/// A single Gaussian component of the search-region mixture, corresponding to
/// one selected Kalman-filter hypothesis propagated to the target epoch.
///
/// The inverse covariance and normalization constant are precomputed once at
/// construction time, since they are reused for every candidate observation
/// tested against this component (`mixture_likelihood`, `mahalanobis2`).
#[derive(Debug, Clone)]
pub struct SearchComponent {
    /// Renormalized mixture weight $w_i$.
    pub weight: f64,
    /// Predicted right ascension $\mu_{ra}$ (rad).
    pub center_ra: f64,
    /// Predicted declination $\mu_{dec}$ (rad).
    pub center_dec: f64,
    /// Innovation covariance $S_i = H P_i H^\top + R$.
    pub s: Matrix2<f64>,
    /// Cached $S_i^{-1}$.
    s_inv: Matrix2<f64>,
    /// Cached normalization constant $1 / (2\pi\sqrt{|S_i|})$.
    norm_const: f64,
    /// $\chi^2$ threshold this component was built with (for gating).
    gate_chi2: f64,
}

impl SearchComponent {
    /// Build a component from a propagated hypothesis, caching $S_i^{-1}$ and
    /// the Gaussian normalization constant.
    ///
    /// Returns `None` if `s` is singular or not positive-definite.
    ///
    /// `pub(crate)` so tests elsewhere in the crate (e.g.
    /// `seeding::night_candidate_search`) can build a [`SearchRegion`] by hand
    /// without going through a full `KFBank` propagation.
    pub(crate) fn new(
        weight: f64,
        center_ra: f64,
        center_dec: f64,
        s: Matrix2<f64>,
        gate_chi2: f64,
    ) -> Option<Self> {
        let det = s.determinant();
        if det <= 0.0 {
            return None;
        }
        let s_inv = s.try_inverse()?;
        let norm_const = 1.0 / (std::f64::consts::TAU * det.sqrt());
        Some(Self {
            weight,
            center_ra,
            center_dec,
            s,
            s_inv,
            norm_const,
            gate_chi2,
        })
    }

    /// Angle-aware offset $(z - \mu_i)$ from an arbitrary sky position.
    fn offset(&self, ra: f64, dec: f64) -> Vector2<f64> {
        Vector2::new(wrap_angle(ra - self.center_ra), dec - self.center_dec)
    }

    /// Angle-aware offset of this component's center from an arbitrary point
    /// (typically the mixture centroid).
    fn offset_from(&self, ra: f64, dec: f64) -> Vector2<f64> {
        Vector2::new(wrap_angle(self.center_ra - ra), self.center_dec - dec)
    }

    /// Squared Mahalanobis distance $(z-\mu_i)^\top S_i^{-1} (z-\mu_i)$.
    ///
    /// Cheap gating primitive: reuses the cached inverse, no exponential.
    pub fn mahalanobis2(&self, ra: f64, dec: f64) -> f64 {
        let nu = self.offset(ra, dec);
        (nu.transpose() * self.s_inv * nu)[(0, 0)]
    }

    /// Whether `(ra, dec)` falls within this component's `n_sigma2` gate
    /// (in squared Mahalanobis distance, i.e. a $\chi^2$ threshold).
    pub fn contains(&self, ra: f64, dec: f64, chi2_gate: f64) -> bool {
        self.mahalanobis2(ra, dec) <= chi2_gate
    }

    /// Weighted Gaussian density $w_i \, \mathcal{N}(z; \mu_i, S_i)$ at `(ra, dec)`.
    pub fn weighted_density(&self, ra: f64, dec: f64) -> f64 {
        let exponent = -0.5 * self.mahalanobis2(ra, dec);
        self.weight * self.norm_const * exponent.exp()
    }

    /// Per-hypothesis bounding radius contribution used by
    /// [`RadiusStrategy::MaxEllipse`]:
    /// $$r_i = \sqrt{\chi^2 \cdot \lambda_{max}(S_i)} + \|\mu_i - \bar\mu\|$$
    fn per_hypothesis_radius(&self, center_ra: f64, center_dec: f64) -> f64 {
        let offset = self.offset_from(center_ra, center_dec).norm();
        self.gate_chi2.sqrt() * largest_eigenvalue_2x2(&self.s).sqrt() + offset
    }
}

/// A conservative bounding region on the sky enclosing the selected hypotheses
/// at a predicted epoch.
///
/// Used to query an observation catalogue for association candidates before
/// committing to a [`KFBank::step`] update.
///
/// The set of hypotheses contributing to this region is controlled by
/// [`TopK`] at construction time.
#[derive(Debug, Clone)]
pub struct SearchRegion {
    /// Weighted-mean predicted RA (rad).
    pub center_ra: f64,
    /// Weighted-mean predicted Dec (rad).
    pub center_dec: f64,
    /// Conservative bounding radius (rad).
    pub radius_rad: f64,
    /// Per-hypothesis sky ellipses, for fine-grained mixture likelihood
    /// scoring after the coarse cone search.
    pub components: Vec<SearchComponent>,
}

impl SearchRegion {
    /// Evaluate the mixture predictive likelihood at a sky position.
    ///
    /// $$\ell(z) = \sum_i w_i \, \mathcal{N}(z;\, \mu_i,\, S_i)$$
    pub fn mixture_likelihood(&self, ra: f64, dec: f64) -> f64 {
        self.components
            .iter()
            .map(|c| c.weighted_density(ra, dec))
            .sum()
    }

    /// Whether `(ra, dec)` falls inside at least one component's `chi2_gate`
    /// (squared Mahalanobis distance).
    ///
    /// This is a cheap pre-filter to apply to candidates already selected by
    /// the coarse cone search (`center_ra`, `center_dec`, `radius_rad`),
    /// before paying for a full [`Self::mixture_likelihood`] evaluation.
    pub fn any_component_contains(&self, ra: f64, dec: f64, chi2_gate: f64) -> bool {
        self.components
            .iter()
            .any(|c| c.contains(ra, dec, chi2_gate))
    }
}

impl<'state_lf, 'bank_config> KFBank<'state_lf, 'bank_config> {
    /// Predict a sky search region at a future epoch.
    ///
    /// Each hypothesis selected by `top_k` is propagated read-only to
    /// `t_prop`. The bounding radius is computed according to `radius_strategy`.
    ///
    /// With [`RadiusStrategy::MixtureCovariance`] (default):
    ///
    /// $$S_{mix} = \sum_i w_i \bigl(S_i + (\mu_i - \bar\mu)(\mu_i - \bar\mu)^\top\bigr)$$
    ///
    /// $$r = \sqrt{\chi^2_{region} \cdot \lambda_{max}(S_{mix})}$$
    ///
    /// With [`RadiusStrategy::MaxEllipse`] (original conservative behaviour):
    ///
    /// $$r = \max_i \left( \sqrt{\chi^2_{region} \cdot \lambda_{max}(S_i)}
    ///       + \| \mu_i - \bar{\mu} \| \right)$$
    ///
    /// $\chi^2_{region}$ comes from `self.config.search_region_chi2`, which is
    /// **independent of `gate_chi2`**.  This decoupling lets the gate stay
    /// tight (small `gate_chi2`) while the search region remains generous
    /// enough to reliably contain the next observation.
    ///
    /// Hypothesis selection (`top_k`)
    /// --------------------------------
    /// - [`TopK::All`]                  – all live hypotheses.
    /// - [`TopK::Map`]                  – single highest-weight hypothesis.
    /// - [`TopK::Best(k)`]              – top-`k` by descending weight, renormalized.
    /// - [`TopK::WeightThreshold(t)`]   – minimal set covering cumulative weight `t`.
    ///
    /// In all cases the selected weights are renormalized to sum to 1 before
    /// computing the centroid and mixture components.
    ///
    /// Arguments
    /// ---------
    /// * `t_prop`          – Target epoch (MJD TT).
    /// * `r_obs_new`       – Observer heliocentric position at `t_prop` (AU).
    /// * `v_obs_new`       – Observer heliocentric velocity at `t_prop` (AU/day).
    /// * `obs_noise`       – Diagonal $[\sigma_{RA}^2, \sigma_{Dec}^2]$ (rad²).
    ///   Added to each $H P H^\top$ to form $S_i$.
    /// * `top_k`           – Hypothesis selection policy (see [`TopK`]).
    /// * `radius_strategy` – Radius computation policy (see [`RadiusStrategy`]).
    ///
    /// Return
    /// ------
    /// * `Ok(SearchRegion)` – Bounding region and per-hypothesis components.
    /// * `Err(PropagateError)` – If all selected hypotheses fail to propagate.
    pub fn predict_search_region(
        &self,
        t_prop: f64,
        r_obs_new: nalgebra::Vector3<f64>,
        v_obs_new: nalgebra::Vector3<f64>,
        obs_noise: Vector2<f64>,
        top_k: TopK,
        radius_strategy: RadiusStrategy,
    ) -> Result<SearchRegion, PropagateError> {
        let span = tracing::trace_span!(
            "predict_search_region",
            t_prop = t_prop,
            n_hypotheses = self.hypotheses.len(),
            top_k = ?top_k,
        );
        let _enter = span.enter();

        let mut predicted = self.propagate_hypotheses(t_prop, r_obs_new, v_obs_new)?;
        top_k.apply(&mut predicted);
        if predicted.is_empty() {
            return Err(PropagateError::SingularJacobian);
        }
        tracing::trace!(n_selected = predicted.len(), "Hypothesis selection applied");

        let (center_ra, center_dec) = weighted_sky_centroid(&predicted);
        let r_noise = Matrix2::from_diagonal(&obs_noise);

        // NOTE: we use `search_region_chi2` here, NOT `gate_chi2`.
        // These two parameters serve different purposes:
        //
        //   gate_chi2          — tight chi-square threshold for discarding
        //                        implausible hypotheses during the update step
        //                        (e.g. 23.0 ≈ 99.999 %).
        //
        //   search_region_chi2 — determines how large the predicted sky region
        //                        is.  It should be generous enough to reliably
        //                        contain the next observation even when the
        //                        filter is slightly overconfident.  Typical
        //                        values: 100–500 (10–22σ).
        //
        // Coupling them caused the search radius to shrink whenever gate_chi2
        // was reduced to a physically meaningful value, making `in_r` coverage
        // drop to ~25 % even when the filter was tracking correctly.
        let chi2 = self.config.search_region_chi2;
        let components = build_components(&predicted, r_noise, chi2);
        let radius_rad = radius_strategy.radius(&components, center_ra, center_dec);

        tracing::trace!(
            center_ra_deg = center_ra.to_degrees(),
            center_dec_deg = center_dec.to_degrees(),
            radius_arcsec = radius_rad.to_degrees() * 3600.0,
            n_components = components.len(),
            "Search region computed"
        );

        Ok(SearchRegion {
            center_ra,
            center_dec,
            radius_rad,
            components,
        })
    }

    /// Propagate every live hypothesis read-only to `t_prop`, skipping and
    /// logging failures.
    ///
    /// Thin wrapper around [`Self::predict_hypotheses`] (shared with
    /// [`Self::predict_to`]) that projects each predicted hypothesis down to
    /// the bare `(weight, KFState)` pair this module's mixture bookkeeping
    /// needs.
    ///
    /// Returns `Err(PropagateError::SingularJacobian)` if no hypothesis
    /// propagates successfully.
    fn propagate_hypotheses(
        &'_ self,
        t_prop: f64,
        r_obs_new: nalgebra::Vector3<f64>,
        v_obs_new: nalgebra::Vector3<f64>,
    ) -> Result<Vec<(f64, KFState<'_>)>, PropagateError> {
        let predicted: Vec<(f64, KFState)> = self
            .predict_hypotheses(t_prop, r_obs_new, v_obs_new)
            .into_iter()
            .map(|hyp| (hyp.weight(), hyp.kf))
            .collect();

        if predicted.is_empty() {
            tracing::trace!("All hypotheses failed to predict; search region unavailable");
            return Err(PropagateError::SingularJacobian);
        }
        tracing::trace!(
            n_predicted = predicted.len(),
            n_live = self.hypotheses.len(),
            "Predicted live hypotheses for search-region construction"
        );
        Ok(predicted)
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Compute the weighted-mean sky position.
///
/// Assumes weights are already normalized (sum to 1).
fn weighted_sky_centroid(predicted: &[(f64, KFState)]) -> (f64, f64) {
    predicted.iter().fold((0.0, 0.0), |(ra, dec), (w, kf)| {
        (ra + w * kf.state[0], dec + w * kf.state[1])
    })
}

/// Build one [`SearchComponent`] per predicted hypothesis, skipping those
/// whose sky covariance is unavailable or degenerate.
fn build_components(
    predicted: &[(f64, KFState)],
    r_noise: Matrix2<f64>,
    gate_chi2: f64,
) -> Vec<SearchComponent> {
    predicted
        .iter()
        .filter_map(|(w, kf)| {
            let s = match kf.sky_covariance() {
                Ok(cov) => cov + r_noise,
                Err(e) => {
                    tracing::trace!(error = ?e, "Sky covariance unavailable, skipping");
                    return None;
                }
            };
            SearchComponent::new(*w, kf.state[0], kf.state[1], s, gate_chi2)
        })
        .collect()
}
