use nalgebra::{Matrix2, Vector2};

use crate::topocentric_kf::{
    KFState, kalman_bank::KFBank, propagate::PropagateError, update::wrap_angle,
};

/// Controls which hypotheses from the bank are used to build a [`SearchRegion`].
///
/// When the bank holds many low-weight hypotheses, restricting the region to
/// the most probable ones yields a tighter, more actionable search area while
/// preserving the probabilistic guarantees that matter.
///
/// Variants
/// --------
/// - [`TopK::All`] – conservative fallback: every live hypothesis contributes.
/// - [`TopK::Map`] – only the single highest-weight hypothesis (Maximum A
///   Posteriori). Equivalent to `TopK::Best(1)`.
/// - [`TopK::Best(k)`] – the `k` hypotheses with the highest weights,
///   renormalized to sum to 1.
/// - [`TopK::WeightThreshold(theta)`] – retains the minimal set of hypotheses
///   (sorted by descending weight) whose cumulative weight reaches `theta`.
///   For example, `WeightThreshold(0.99)` discards all hypotheses beyond the
///   99 % credible set, eliminating low-weight spatial outliers that would
///   otherwise inflate the search region.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TopK {
    /// Use all live hypotheses (original conservative behaviour).
    All,
    /// Use only the single best hypothesis (MAP estimate).
    Map,
    /// Use the `k` best hypotheses by descending weight, renormalized.
    Best(usize),
    /// Keep the minimal prefix of hypotheses (sorted by descending weight)
    /// whose cumulative weight reaches `theta ∈ (0, 1]`, then renormalize.
    WeightThreshold(f64),
}

impl Default for TopK {
    /// Defaults to [`TopK::All`] to preserve backward-compatible behaviour.
    fn default() -> Self {
        TopK::All
    }
}

impl TopK {
    /// Select and renormalize hypotheses in-place according to `self`.
    fn apply(self, predicted: &mut Vec<(f64, KFState)>) {
        match self {
            TopK::All => {}
            TopK::Map => Self::keep_map(predicted),
            TopK::Best(k) => Self::keep_best(predicted, k),
            TopK::WeightThreshold(theta) => Self::keep_threshold(predicted, theta),
        }
    }

    /// Keep only the highest-weight hypothesis, weight forced to 1.0.
    fn keep_map(predicted: &mut Vec<(f64, KFState)>) {
        let best_idx = predicted
            .iter()
            .enumerate()
            .max_by(|(_, (wa, _)), (_, (wb, _))| wa.total_cmp(wb))
            .map(|(i, _)| i)
            .unwrap_or(0);
        predicted.swap(0, best_idx);
        predicted.truncate(1);
        predicted[0].0 = 1.0;
    }

    /// Keep the `k` highest-weight hypotheses, renormalized.
    fn keep_best(predicted: &mut Vec<(f64, KFState)>, k: usize) {
        let k = k.max(1).min(predicted.len());
        predicted.select_nth_unstable_by(k - 1, |(wa, _), (wb, _)| wb.total_cmp(wa));
        predicted.truncate(k);
        renormalize(predicted);
    }

    /// Keep the minimal prefix (by descending weight) reaching cumulative
    /// weight `theta`, renormalized.
    fn keep_threshold(predicted: &mut Vec<(f64, KFState)>, theta: f64) {
        let theta = theta.clamp(0.0, 1.0);
        predicted.sort_unstable_by(|(wa, _), (wb, _)| wb.total_cmp(wa));
        let mut cumul = 0.0;
        let mut keep = 0;
        for (w, _) in predicted.iter() {
            cumul += w;
            keep += 1;
            if cumul >= theta {
                break;
            }
        }
        predicted.truncate(keep);
        renormalize(predicted);
    }
}

/// Renormalize weights in-place so they sum to 1.
fn renormalize(predicted: &mut [(f64, KFState)]) {
    let total: f64 = predicted.iter().map(|(w, _)| w).sum();
    if total > 0.0 {
        predicted.iter_mut().for_each(|(w, _)| *w /= total);
    }
}

/// Inner strategy choice for [`RadiusStrategy::Clamped`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixOrMax {
    MaxEllipse,
    MixtureCovariance,
}

/// Controls how the bounding radius of a [`SearchRegion`] is computed.
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RadiusStrategy {
    /// Original conservative behaviour.
    MaxEllipse,
    /// Tighter mixture-covariance bound.
    MixtureCovariance,
    /// Hard clamp applied on top of another strategy.
    Clamped {
        /// Strategy to apply before clamping.
        inner: MixOrMax,
        /// Maximum allowed radius (arcseconds).
        max_arcsec: f64,
    },
}

impl Default for RadiusStrategy {
    fn default() -> Self {
        RadiusStrategy::MixtureCovariance
    }
}

impl RadiusStrategy {
    /// Dispatch radius computation over `components` according to `self`.
    fn radius(self, components: &[SearchComponent], center_ra: f64, center_dec: f64) -> f64 {
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
    fn new(
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

impl<'state_lf> KFBank<'state_lf> {
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
    /// Returns `Err(PropagateError::SingularJacobian)` if no hypothesis
    /// propagates successfully.
    fn propagate_hypotheses(
        &'_ self,
        t_prop: f64,
        r_obs_new: nalgebra::Vector3<f64>,
        v_obs_new: nalgebra::Vector3<f64>,
    ) -> Result<Vec<(f64, KFState<'_>)>, PropagateError> {
        let predicted: Vec<(f64, KFState)> = self
            .hypotheses
            .iter()
            .filter_map(|h| match h.kf.predict(t_prop, r_obs_new, v_obs_new) {
                Ok(kf) => Some((h.weight(), kf)),
                Err(e) => {
                    tracing::trace!(
                        hyp_id = h.id,
                        error = ?e,
                        "Hypothesis prediction failed, excluding from search region"
                    );
                    None
                }
            })
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

/// Largest eigenvalue of a symmetric $2 \times 2$ matrix via the analytic
/// formula.
///
/// For $S = \begin{pmatrix} a & b \\ b & d \end{pmatrix}$:
///
/// $$\lambda_{max} = \frac{a+d}{2} + \sqrt{\left(\frac{a-d}{2}\right)^2 + b^2}$$
fn largest_eigenvalue_2x2(s: &Matrix2<f64>) -> f64 {
    let a = s[(0, 0)];
    let d = s[(1, 1)];
    let b = s[(0, 1)];
    let mid = (a + d) / 2.0;
    let half_diff = (a - d) / 2.0;
    mid + (half_diff * half_diff + b * b).sqrt()
}
