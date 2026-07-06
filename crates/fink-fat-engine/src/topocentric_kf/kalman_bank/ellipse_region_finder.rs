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
    /// Per-hypothesis sky ellipses with their (renormalized) weights, for
    /// fine-grained mixture likelihood scoring after the coarse cone search.
    ///
    /// Each entry is `(weight, center_ra, center_dec, S)` where `S` is the
    /// $2 \times 2$ innovation covariance.
    pub components: Vec<(f64, f64, f64, Matrix2<f64>)>,
}

impl SearchRegion {
    /// Evaluate the mixture predictive likelihood at a sky position.
    ///
    /// $$\ell(z) = \sum_i w_i \, \mathcal{N}(z;\, \mu_i,\, S_i)$$
    ///
    /// Arguments
    /// ---------
    /// * `ra`  – Right ascension of the candidate (rad).
    /// * `dec` – Declination of the candidate (rad).
    ///
    /// Return
    /// ------
    /// Mixture likelihood (linear scale, not log).
    pub fn mixture_likelihood(&self, ra: f64, dec: f64) -> f64 {
        self.components
            .iter()
            .filter_map(|(w, mu_ra, mu_dec, s)| {
                let nu = Vector2::new(wrap_angle(ra - mu_ra), dec - mu_dec);
                let s_inv = s.try_inverse()?;
                let det = s.determinant();
                if det <= 0.0 {
                    return None;
                }
                let exponent = -0.5 * (nu.transpose() * s_inv * nu)[(0, 0)];
                let norm = 1.0 / (std::f64::consts::TAU * det.sqrt());
                Some(w * norm * exponent.exp())
            })
            .sum()
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

        // Propagate every live hypothesis read-only; skip failures.
        let mut predicted: Vec<(f64, KFState)> = self
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

        // Apply top-k / weight-threshold selection.
        select_top_k(&mut predicted, top_k);

        if predicted.is_empty() {
            return Err(PropagateError::SingularJacobian);
        }

        tracing::trace!(n_selected = predicted.len(), "Hypothesis selection applied");

        // Weighted centroid on the sky.
        let (center_ra, center_dec) = weighted_sky_centroid(&predicted);

        // Per-hypothesis innovation covariances + bounding radius.
        //
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
        let r_noise = Matrix2::from_diagonal(&obs_noise);
        let (components, radius_rad) = hypothesis_components(
            &predicted,
            center_ra,
            center_dec,
            r_noise,
            self.config.search_region_chi2,
            radius_strategy,
        );

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
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Select and renormalize hypotheses according to the [`TopK`] policy.
///
/// The vector is modified in-place:
///
/// - [`TopK::All`] – no-op.
/// - [`TopK::Map`] – swaps the best hypothesis to index 0, truncates to
///   length 1, sets weight to 1.0.
/// - [`TopK::Best(k)`] – partially sorts by descending weight, truncates to
///   `k`, renormalizes.
/// - [`TopK::WeightThreshold(theta)`] – sorts by descending weight, retains
///   the minimal prefix whose cumulative weight reaches `theta`, renormalizes.
fn select_top_k(predicted: &mut Vec<(f64, KFState)>, top_k: TopK) {
    match top_k {
        TopK::All => {}

        TopK::Map => {
            let best_idx = predicted
                .iter()
                .enumerate()
                .max_by(|(_, (wa, _)), (_, (wb, _))| wa.partial_cmp(wb).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            predicted.swap(0, best_idx);
            predicted.truncate(1);
            predicted[0].0 = 1.0;
        }

        TopK::Best(k) => {
            let k = k.max(1).min(predicted.len());
            predicted.select_nth_unstable_by(k - 1, |(wa, _), (wb, _)| wb.partial_cmp(wa).unwrap());
            predicted.truncate(k);
            renormalize(predicted);
        }

        TopK::WeightThreshold(theta) => {
            let theta = theta.clamp(0.0, 1.0);
            // Full sort so we can walk the cumulative weight prefix.
            predicted.sort_unstable_by(|(wa, _), (wb, _)| wb.partial_cmp(wa).unwrap());
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
}

/// Renormalize weights in-place so they sum to 1.
fn renormalize(predicted: &mut Vec<(f64, KFState)>) {
    let total: f64 = predicted.iter().map(|(w, _)| w).sum();
    if total > 0.0 {
        predicted.iter_mut().for_each(|(w, _)| *w /= total);
    }
}

/// Compute the weighted-mean sky position.
///
/// Assumes weights are already normalized (sum to 1).
fn weighted_sky_centroid(predicted: &[(f64, KFState)]) -> (f64, f64) {
    predicted.iter().fold((0.0, 0.0), |(ra, dec), (w, kf)| {
        (ra + w * kf.state[0], dec + w * kf.state[1])
    })
}

fn hypothesis_components(
    predicted: &[(f64, KFState)],
    center_ra: f64,
    center_dec: f64,
    r_noise: Matrix2<f64>,
    gate_chi2: f64,
    radius_strategy: RadiusStrategy,
) -> (Vec<(f64, f64, f64, Matrix2<f64>)>, f64) {
    let nsigma = gate_chi2.sqrt();

    let components_raw: Vec<(f64, f64, f64, Matrix2<f64>, f64)> = predicted
        .iter()
        .filter_map(|(w, kf)| {
            let s = match kf.sky_covariance() {
                Ok(cov) => cov + r_noise,
                Err(e) => {
                    tracing::trace!(error = ?e, "Sky covariance unavailable, skipping");
                    return None;
                }
            };
            let mu_ra = kf.state[0];
            let mu_dec = kf.state[1];
            let d_ra = wrap_angle(mu_ra - center_ra);
            let d_dec = mu_dec - center_dec;
            let offset = (d_ra * d_ra + d_dec * d_dec).sqrt();
            let lambda_max = largest_eigenvalue_2x2(&s);
            let per_hyp_radius = nsigma * lambda_max.sqrt() + offset;
            Some((*w, mu_ra, mu_dec, s, per_hyp_radius))
        })
        .collect();

    let radius_rad = compute_radius(
        &components_raw,
        center_ra,
        center_dec,
        gate_chi2,
        radius_strategy,
    );

    let components = components_raw
        .into_iter()
        .map(|(w, ra, dec, s, _)| (w, ra, dec, s))
        .collect();

    (components, radius_rad)
}

/// Dispatch radius computation to the chosen [`RadiusStrategy`].
fn compute_radius(
    components: &[(f64, f64, f64, Matrix2<f64>, f64)],
    center_ra: f64,
    center_dec: f64,
    gate_chi2: f64,
    strategy: RadiusStrategy,
) -> f64 {
    match strategy {
        RadiusStrategy::MaxEllipse => components
            .iter()
            .map(|(_, _, _, _, r)| *r)
            .fold(0.0_f64, f64::max),

        RadiusStrategy::MixtureCovariance => {
            mixture_covariance_radius(components, center_ra, center_dec, gate_chi2)
        }

        RadiusStrategy::Clamped { inner, max_arcsec } => {
            let inner_strategy = match inner {
                MixOrMax::MaxEllipse => RadiusStrategy::MaxEllipse,
                MixOrMax::MixtureCovariance => RadiusStrategy::MixtureCovariance,
            };
            let r = compute_radius(components, center_ra, center_dec, gate_chi2, inner_strategy);
            let max_rad = (max_arcsec / 3600.0_f64).to_radians();
            r.min(max_rad)
        }
    }
}

/// Compute the bounding radius from the full mixture covariance.
///
/// $$S_{mix} = \sum_i w_i \bigl(S_i + (\mu_i - \bar\mu)(\mu_i - \bar\mu)^\top\bigr)$$
///
/// $$r = \sqrt{\chi^2_{gate} \cdot \lambda_{max}(S_{mix})}$$
///
/// Both the within-component uncertainty ($S_i$) and the between-component
/// spatial spread contribute to $S_{mix}$, so the radius is a valid
/// conservative bound on the mixture support.
fn mixture_covariance_radius(
    components: &[(f64, f64, f64, Matrix2<f64>, f64)],
    center_ra: f64,
    center_dec: f64,
    gate_chi2: f64,
) -> f64 {
    let mut s_mix = Matrix2::zeros();
    for (w, mu_ra, mu_dec, s_i, _) in components {
        let d_ra = wrap_angle(mu_ra - center_ra);
        let d_dec = mu_dec - center_dec;
        let delta = Vector2::new(d_ra, d_dec);
        s_mix += *w * (s_i + delta * delta.transpose());
    }
    let lambda_max = largest_eigenvalue_2x2(&s_mix);
    gate_chi2.sqrt() * lambda_max.sqrt()
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
