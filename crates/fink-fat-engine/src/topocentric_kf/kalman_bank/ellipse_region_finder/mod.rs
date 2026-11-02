pub mod radius_strategy;
pub mod top_k;

use nalgebra::{Matrix2, Vector2, Vector3};

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

use crate::logging::LogTarget;

/// Structured log events for search-region construction (hypothesis
/// selection, radius strategy). See [`crate::logging`] for the `.emit()`
/// pattern.
pub enum EllipseRegionEvent {
    HypothesisSelectionApplied {
        n_selected: usize,
    },
    SkyCovarianceUnavailable {
        error: String,
    },
    SearchRegionComputed {
        center_ra_deg: f64,
        center_dec_deg: f64,
        radius_arcsec: f64,
        n_components: usize,
    },
    SkyCoverComputed {
        n_cones: usize,
        n_components: usize,
        /// Hypotheses left outside every cone because `max_search_cones` was
        /// reached — non-zero means the cap is binding and recall may suffer.
        n_uncovered: usize,
    },
}

crate::impl_log_target!(
    EllipseRegionEvent,
    "ellipse_region",
    "Search-region construction from bank hypotheses (radius strategy, top-K selection)",
    [tracing::Level::TRACE]
);

impl EllipseRegionEvent {
    pub fn emit(&self) {
        use EllipseRegionEvent::*;
        match self {
            HypothesisSelectionApplied { n_selected } => tracing::trace!(
                target: EllipseRegionEvent::TARGET, n_selected, "Hypothesis selection applied"
            ),
            SkyCovarianceUnavailable { error } => tracing::trace!(
                target: EllipseRegionEvent::TARGET, error, "Sky covariance unavailable, skipping"
            ),
            SearchRegionComputed {
                center_ra_deg,
                center_dec_deg,
                radius_arcsec,
                n_components,
            } => tracing::trace!(
                target: EllipseRegionEvent::TARGET, center_ra_deg, center_dec_deg, radius_arcsec, n_components,
                "Search region computed"
            ),
            SkyCoverComputed {
                n_cones,
                n_components,
                n_uncovered,
            } => tracing::trace!(
                target: EllipseRegionEvent::TARGET, n_cones, n_components, n_uncovered,
                "Sky cover computed"
            ),
        }
    }

    pub fn span(t_prop: f64, n_hypotheses: usize, top_k: &TopK) -> tracing::Span {
        tracing::trace_span!(
            target: EllipseRegionEvent::TARGET,
            "predict_search_region",
            t_prop,
            n_hypotheses,
            top_k = ?top_k,
        )
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

    /// Whether `radius_strategy`'s hard clamp truncated this region's radius.
    ///
    /// The clamp is the *only* thing that can silently shrink a region below
    /// what the configured strategy asked for: every other path returns the
    /// radius the strategy computed, which covers the mixture by construction.
    /// So a pinned radius is exactly the signal that the coarse cone may be
    /// hiding modes, and hence the trigger for [`sky_cover_regions`] — see its
    /// doc for why this happens routinely at steps 1–2.
    ///
    /// Cheap by design: one scalar comparison against a value already in hand.
    /// The earlier geometric test ("does one cone enclose every component's
    /// ellipse?") fired on 55/60 of the dumped cases versus 32/60 for this one
    /// **at identical recall** — it was flagging banks whose region was never
    /// truncated in the first place, and each false trigger cost a full cover
    /// plus a multi-cone catalogue query.
    ///
    /// Returns `false` for an unclamped strategy, which can never be truncated.
    pub fn radius_pinned_at_clamp(&self, radius_strategy: RadiusStrategy) -> bool {
        radius_strategy
            .clamp_rad(self.components.len())
            .is_some_and(|clamp| self.radius_rad >= clamp * (1.0 - 1e-9))
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
    /// - [`TopK::Best`]`(k)`              – top-`k` by descending weight, renormalized.
    /// - [`TopK::WeightThreshold`]`(t)`   – minimal set covering cumulative weight `t`.
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
        let _enter = EllipseRegionEvent::span(t_prop, self.len(), &top_k).entered();

        self.predict_to(t_prop, r_obs_new, v_obs_new).search_region(
            obs_noise,
            top_k,
            radius_strategy,
        )
    }

    /// Like [`Self::predict_search_region`], but assumes `self` is already
    /// propagated to the target epoch (e.g. via [`Self::predict_to`]).
    ///
    /// Extracted so that callers who already need a [`Self::predict_to`]'d
    /// bank for other purposes (branching: `Branch::from_observation`,
    /// `Branch::from_null`) don't pay for the two-body Kepler propagation
    /// twice — `predict_search_region` used to re-propagate internally,
    /// duplicating work already done by a sibling `predict_to` call at the
    /// orchestration layer.
    pub fn search_region(
        &self,
        obs_noise: Vector2<f64>,
        top_k: TopK,
        radius_strategy: RadiusStrategy,
    ) -> Result<SearchRegion, PropagateError> {
        let predicted = self.selected_mixture(top_k);

        // NOTE: `search_region_chi2`, NOT `gate_chi2` — see
        // `search_region_from_mixture`'s doc for why these are deliberately
        // decoupled.
        search_region_from_mixture(
            &predicted,
            obs_noise,
            radius_strategy,
            self.config.search_region_chi2,
        )
    }

    /// `self`'s live hypotheses as `(weight, state)` pairs, after applying
    /// `top_k` — the exact input [`Self::search_region`] feeds to
    /// [`search_region_from_mixture`]. Exposed so a caller that already has
    /// an already-propagated bank (e.g. via [`Self::predict_to`]) can also
    /// build a [`sky_cover_regions`] call from the *same* selected mixture,
    /// without re-propagating or duplicating the selection logic.
    pub fn selected_mixture(&self, top_k: TopK) -> Vec<(f64, KFState<'state_lf>)> {
        let mut predicted: Vec<(f64, KFState)> = self
            .hypotheses()
            .iter()
            .map(|hyp| (hyp.weight(), hyp.kf.clone()))
            .collect();
        top_k.apply(&mut predicted);
        predicted
    }

    /// Every live hypothesis's predicted `(weight, state)` at `t_prop`,
    /// **before** any [`TopK`] selection — the raw input
    /// [`search_region_from_mixture`] (via [`Self::search_region`]) filters
    /// down.
    ///
    /// Exposed so a caller that needs to try several [`TopK`]/
    /// `search_region_chi2`/[`RadiusStrategy`] combinations against the
    /// *same* propagated epoch — e.g. re-deriving a [`SearchRegion`] for
    /// several calibration candidates without re-running the two-body
    /// propagation each time — can propagate once here and call
    /// [`search_region_from_mixture`] directly per combination, instead of
    /// paying for [`Self::predict_to`]'s Kepler solve again through
    /// [`Self::search_region`]/[`Self::predict_search_region`] each time.
    pub fn predicted_mixture(
        &self,
        t_prop: f64,
        r_obs_new: Vector3<f64>,
        v_obs_new: Vector3<f64>,
    ) -> Vec<(f64, KFState<'state_lf>)> {
        self.predict_to(t_prop, r_obs_new, v_obs_new)
            .hypotheses()
            .iter()
            .map(|hyp| (hyp.weight(), hyp.kf.clone()))
            .collect()
    }
}

/// Reduce an already-propagated mixture of `(weight, state)` hypotheses
/// (see [`KFBank::predicted_mixture`]) into a [`SearchRegion`] — the part of
/// [`KFBank::search_region`] that doesn't need the bank itself, only the
/// propagated hypotheses and the four inputs that shape the region
/// (`obs_noise`, `radius_strategy`, and `search_region_chi2` below).
///
/// Pulled out as a free function (rather than kept as a private step inside
/// [`KFBank::search_region`]) so a caller that already has a propagated
/// mixture — from [`KFBank::predicted_mixture`], recorded once — can
/// recompute a [`SearchRegion`] for as many `(obs_noise, radius_strategy,
/// search_region_chi2)` combinations as needed, without repeating the
/// two-body Kepler propagation each time. `top_k` selection is expected to
/// already have been applied to `predicted` by the caller (see
/// [`TopK::apply`]) — kept out of this function so it stays purely
/// geometric/statistical, no [`TopK`] dependency.
///
/// # `search_region_chi2` vs. `gate_chi2`
///
/// These two parameters serve different purposes:
///
///   `gate_chi2`          — tight chi-square threshold for discarding
///                          implausible hypotheses during the update step
///                          (e.g. 23.0 ≈ 99.999 %).
///
///   `search_region_chi2` — determines how large the predicted sky region
///                          is.  It should be generous enough to reliably
///                          contain the next observation even when the
///                          filter is slightly overconfident.  Typical
///                          values: 100–500 (10–22σ).
///
/// Coupling them caused the search radius to shrink whenever `gate_chi2`
/// was reduced to a physically meaningful value, making `in_r` coverage
/// drop to ~25 % even when the filter was tracking correctly.
pub fn search_region_from_mixture(
    predicted: &[(f64, KFState)],
    obs_noise: Vector2<f64>,
    radius_strategy: RadiusStrategy,
    search_region_chi2: f64,
) -> Result<SearchRegion, PropagateError> {
    if predicted.is_empty() {
        return Err(PropagateError::SingularJacobian);
    }
    EllipseRegionEvent::HypothesisSelectionApplied {
        n_selected: predicted.len(),
    }
    .emit();

    let (center_ra, center_dec) = map_center(predicted);
    let r_noise = Matrix2::from_diagonal(&obs_noise);

    let components = build_components(predicted, r_noise, search_region_chi2);
    let radius_rad = radius_strategy.radius(&components, center_ra, center_dec);

    EllipseRegionEvent::SearchRegionComputed {
        center_ra_deg: center_ra.to_degrees(),
        center_dec_deg: center_dec.to_degrees(),
        radius_arcsec: radius_rad.to_degrees() * 3600.0,
        n_components: components.len(),
    }
    .emit();

    Ok(SearchRegion {
        center_ra,
        center_dec,
        radius_rad,
        components,
    })
}

/// Tile the predicted mixture's sky footprint with a small set of cones, so
/// the coarse catalogue query sees *every* plausible mode instead of only the
/// one the MAP hypothesis happens to sit on.
///
/// Returns `(center_ra, center_dec, radius_rad)` triples for
/// [`find_candidates_for_bank_multi_region`](crate::topocentric_kf::branching::candidate_search::find_candidates_for_bank_multi_region).
/// The fine per-hypothesis gate still runs against the full-mixture
/// [`SearchRegion`], so this changes coarse *recall* only — matching precision
/// is untouched.
///
/// # Why this exists
///
/// A single [`SearchRegion`] is one cone centered on the MAP hypothesis whose
/// radius is clamped (`RadiusStrategy::Clamped`, 30′ in practice). Right after
/// a 2-point bootstrap the bank holds several hundred (ρ, ρ̇) grid nodes whose
/// predicted sky positions span **degrees** along-track — median 0.6°, up to
/// 178° in the `mot_analysis` dumps — while the MAP node carries ~1 % of the
/// weight and is therefore statistically arbitrary. The clamp binds in
/// essentially every such case, so the coarse pool degenerates to a 30′ disc
/// around a random node of a degrees-wide mixture and the true observation is
/// never even offered to the gate.
///
/// Measured on the 60 `SearchedButNotMatched` step-1/2 cases dumped by
/// `mot_analysis`, the truth entered the coarse pool for 42/60 with the single
/// cone, 54/60 with 8 ρ-clusters, and **58/60** with this cover — 58 being the
/// ceiling, the other 2 being genuine χ² outliers. ρ-clustering saturates
/// because ρ̇ also drives along-track spread, so a ρ-chunk is not angularly
/// compact; clustering on the predicted sky position is exact by construction.
///
/// # Algorithm
///
/// Greedy set cover in descending weight order: the heaviest not-yet-covered
/// hypothesis seeds a cone, every uncovered hypothesis within `cone_half_rad`
/// of that seed joins it, and the cone's radius grows to
/// `max_j (sep(seed, μ_j) + r_j)` with `r_j = sqrt(chi2 · λmax(S_j))` so each
/// member's own error ellipse is enclosed. Repeat until every hypothesis is
/// covered or `max_cones` cones have been emitted.
///
/// Seeding in weight order is what makes the `max_cones` truncation safe: the
/// cones that survive are the ones covering the most probable hypotheses, so
/// exceeding the cap degrades recall gracefully from the tail inward rather
/// than dropping an arbitrary mode. In the dumps the median cover is 2 cones
/// (fewer than the 8 ρ-clusters it replaces), so the cap rarely binds at all.
///
/// Takes an already-built `components` slice — in practice
/// [`SearchRegion::components`], the `top_k`-selected set the single-cone
/// region already paid to build. Covering that subset rather than the full
/// bank was measured to give **identical** recall (58/60 either way): at
/// step 1 the weights are near-uniform, so a `WeightThreshold(0.99)` cut keeps
/// essentially the whole bank anyway. Reusing it makes the cover's marginal
/// component-construction cost exactly zero.
///
/// # Complexity
///
/// `max_cones` bounds the work: the greedy loop stops as soon as it has
/// emitted that many cones, so the pairwise scan runs at most
/// `max_cones × n` times, and only over hypotheses still uncovered (the
/// candidate list is compacted with `swap_remove` as they are absorbed). At
/// the measured operating point — `max_cones = 4`, n ≈ 450 — that is ~900
/// iterations, versus ~147 000 for a full O(n²) pass. Spatial indexing here
/// is counterproductive at these n: a previous revision bucketed through
/// `HealpixBinner` and was markedly *slower*, since a hash map plus a
/// per-bucket `Vec` allocation and a cone query per seed cost far more than a
/// few hundred dot products.
pub fn sky_cover_regions(
    components: &[SearchComponent],
    cone_half_rad: f64,
    max_cones: usize,
) -> Vec<(f64, f64, f64)> {
    if components.is_empty() || max_cones == 0 {
        return Vec::new();
    }

    // Precompute unit vectors and per-hypothesis ellipse radii. Membership is
    // tested as `u_seed · u_j >= cos(cone_half_rad)`, which avoids an `acos`
    // per pair; the actual angle is only needed once per hypothesis, when it
    // joins a cone.
    let unit: Vec<Vector3<f64>> = components
        .iter()
        .map(|c| {
            let (sin_dec, cos_dec) = c.center_dec.sin_cos();
            let (sin_ra, cos_ra) = c.center_ra.sin_cos();
            Vector3::new(cos_dec * cos_ra, cos_dec * sin_ra, sin_dec)
        })
        .collect();
    let ellipse_rad: Vec<f64> = components
        .iter()
        .map(|c| (c.gate_chi2 * largest_eigenvalue_2x2(&c.s)).sqrt())
        .collect();

    let mut order: Vec<usize> = (0..components.len()).collect();
    order.sort_unstable_by(|&a, &b| components[b].weight.total_cmp(&components[a].weight));

    // Still-uncovered hypotheses, compacted as they are absorbed so later
    // seeds never re-scan what earlier cones already took. `covered` mirrors
    // it only to make the seed-skip test O(1).
    let mut uncovered: Vec<usize> = order.clone();
    let mut covered = vec![false; components.len()];
    let cos_cone = cone_half_rad.cos();
    let mut cones = Vec::with_capacity(max_cones.min(components.len()));

    for &seed in &order {
        if covered[seed] {
            continue;
        }
        if cones.len() == max_cones {
            break;
        }

        // The seed absorbs itself on the first hit (dot == 1, so the term is
        // just its own ellipse radius) — no need to special-case it.
        let mut radius = 0.0_f64;
        let mut j = 0;
        while j < uncovered.len() {
            let cand = uncovered[j];
            let dot = unit[seed].dot(&unit[cand]);
            if dot < cos_cone {
                j += 1;
                continue;
            }
            covered[cand] = true;
            radius = radius.max(dot.clamp(-1.0, 1.0).acos() + ellipse_rad[cand]);
            uncovered.swap_remove(j);
        }

        cones.push((
            components[seed].center_ra,
            components[seed].center_dec,
            radius,
        ));
    }

    EllipseRegionEvent::SkyCoverComputed {
        n_cones: cones.len(),
        n_components: components.len(),
        n_uncovered: uncovered.len(),
    }
    .emit();

    cones
}

/// The coarse-search decision in one place: return a multi-cone cover for
/// `region` if its radius was truncated by `radius_strategy`'s clamp,
/// otherwise `None` (meaning "the single cone is fine, use the fast path").
///
/// Both the engine (`orchestrate::spawn_branches_for_lineage`) and the
/// `mot_analysis` evaluator must make this decision identically — if they
/// drift, `not_matched%` stops measuring the engine — so the trigger and the
/// cover call live here together rather than being spelled out at each site.
///
/// Returning `None` is the common case and costs a single float comparison:
/// see [`SearchRegion::radius_pinned_at_clamp`].
pub fn cover_if_clamped(
    region: &SearchRegion,
    radius_strategy: RadiusStrategy,
    cone_half_rad: f64,
    max_cones: usize,
) -> Option<Vec<(f64, f64, f64)>> {
    if max_cones == 0 || !region.radius_pinned_at_clamp(radius_strategy) {
        return None;
    }
    let cover = sky_cover_regions(&region.components, cone_half_rad, max_cones);
    (!cover.is_empty()).then_some(cover)
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Sky position of the MAP (highest-weight) hypothesis.
///
/// Used instead of a weighted mean because after a 2-point bootstrap the
/// (ρ, ρ̇) grid still contains hypotheses with wildly implausible implied
/// angular rates that no data has yet penalized (a 2-point fit has zero
/// residual for every grid node) — a *weighted* mean gets dragged by these
/// outliers toward a sky position that can be 100+ degrees from every
/// plausible mode, including the correct one. The MAP hypothesis, being a
/// single mode rather than a blend, isn't subject to this — see the
/// `mot_analysis` step-1/step-2 `not_matched%` investigation.
fn map_center(predicted: &[(f64, KFState)]) -> (f64, f64) {
    predicted
        .iter()
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, kf)| (kf.state[0], kf.state[1]))
        .unwrap_or((0.0, 0.0))
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
                    EllipseRegionEvent::SkyCovarianceUnavailable {
                        error: format!("{e:?}"),
                    }
                    .emit();
                    return None;
                }
            };
            SearchComponent::new(*w, kf.state[0], kf.state[1], s, gate_chi2)
        })
        .collect()
}
