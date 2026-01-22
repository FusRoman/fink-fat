// src/seeding/seed_node.rs

//! Compact intra-night seed representation.
//!
//! A [`SeedNode`] encodes the minimal, self-contained information required for
//! **persistence**, **spatial indexing**, and **inter-night linkage** within the
//! Fink-FAT engine.
//!
//! It is intentionally *data-only*: all modelling, projection logic, prediction
//! or geometric filters are delegated to [`TangentPlaneModel`] or higher-level
//! components. This keeps the struct easy to serialize (via `serde` or
//! `bincode`), cheap to move across threads, and lightweight when stored in
//! per-night indices such as [`SeedSpatialIndex`].
//!
//! ## What a `SeedNode` contains
//! - A unique [`SeedId`] and the associated [`NightId`].
//! - A local tangent-plane kinematic model ([`TangentPlaneModel`]) fitted from
//!   2 points (pair) or 3 points (triplet).
//! - Aggregated photometry (mean/dispersion/band).
//! - The ordered list of constituent detection identifiers (`members`).
//!
//! ## Typical usage
//! 1. Construct seeds from pairs or triplets of alerts.
//! 2. Serialize them to disk or insert them into a [`SeedSpatialIndex`].
//! 3. At prediction time, call [`SeedNode::predict_cone`] or
//!    [`SeedNode::cone_candidates`] to obtain candidate neighbours for
//!    inter-night linking.
//!
//! ## Notes
//! - `cos_dec0` and `sin_dec0` fields inside `TangentCenter` are cached
//!   trigonometric values for fast projection; they can be recomputed
//!   if the model is manually rebuilt.
//! - Photometry is minimalistic by design—only what is required for scoring
//!   or band-matching at linkage time.

use ahash::AHashMap;
use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

use std::fmt::{self, Display, Formatter};

use crate::{
    Alert, AlertId, MjdTt, Radians,
    alerts::AlertStore,
    astro_math::{fit_quad_1d, radec_to_tangent, spherical_midpoint, tangent_to_radec},
    display_format::indent_block,
    engine_config::{edge_config::EdgeConfig, propagator_config::PredictorParams},
    graph::score::ScoredEdge,
    night_id::NightId,
    seeding::{
        photometry::Photometry,
        seed_id::SeedId,
        seed_spatial_index::SeedSpatialIndex,
        tangent_plane::{TangentCenter, TangentPlaneModel},
    },
    spacetime_bucket::{
        spatial_binner::{SpatialBinner, SpatialKey},
        time_binner::TimeBinner,
    },
};

/// Compact intra-night seed object used in the inter-night graph.
///
/// This struct intentionally contains **no geometric logic**; it only stores:
///
/// - the seed identity and night information,
/// - a local tangent-plane dynamical model,
/// - aggregated photometric metadata,
/// - the ordered list of member alert identifiers,
/// - basic covariance matrices for position and velocity.
///
/// This makes `SeedNode` cheap to serialize, hash, index, or store in memory.
/// All prediction logic is delegated to `TangentPlaneModel`.
#[derive(Clone, Debug, Serialize, Deserialize, Encode, Decode, PartialEq)]
pub struct SeedNode {
    /// Globally unique seed identifier.
    pub seed_id: SeedId,

    /// Night identifier (intra-night seeds cannot mix nights).
    pub night_id: NightId,

    /// Local tangent-plane model describing kinematics.
    pub plane: TangentPlaneModel,

    /// Aggregated photometry for scoring / filtering.
    pub photom: Photometry,

    /// Number of detections used to form the seed (2 = pair, 3 = triplet).
    pub n_obs: u16,

    /// Alert identifiers forming the seed, sorted by observation time.
    pub members: Vec<AlertId>,
}

impl Display for SeedNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "SeedNode {{")?;

        writeln!(f, "  id        : {}", self.seed_id)?;
        writeln!(f, "  night     : {}", self.night_id)?;
        writeln!(f, "  n_obs     : {}", self.n_obs)?;
        writeln!(f)?;

        writeln!(
            f,
            "  plane     : {}",
            indent_block(&self.plane.to_string(), 14)
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "  photom   : {}",
            indent_block(&self.photom.to_string(), 14)
        )?;
        writeln!(f)?;

        write!(f, "  members  : [")?;
        for (i, id) in self.members.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{id}")?;
        }
        writeln!(f, "]")?;

        writeln!(f, "}}")
    }
}

impl SeedNode {
    /// Resolve the concrete member alerts for this seed from an [`AlertStore`].
    ///
    /// For each `AlertId` in [`SeedNode::members`], this method looks up the
    /// corresponding [`Alert`] in the provided store and returns a vector of
    /// shared references.
    ///
    /// If **any** member cannot be found, the whole operation fails and
    /// returns `None`. This makes it safer for downstream consumers that
    /// expect the seed to be fully materialisable.
    ///
    /// Arguments
    /// ---------
    /// * `store` – Global alert store, expected to contain all `Alert` entries
    ///   referenced by this seed. The invariant `alert.id.idx() == index` must
    ///   hold for the underlying `alerts` container.
    ///
    /// Return
    /// ------
    /// * `Some(Vec<&Alert>)` if all member alerts were successfully resolved.
    /// * `None` if at least one `AlertId` could not be found in `store`.
    ///
    /// Notes
    /// -----
    /// * This is primarily intended for:
    ///   - debugging or inspection in higher-level pipelines,
    ///   - detailed scoring after a coarse graph pass.
    /// * For pure geometric or linkage operations, you should prefer working
    ///   with `SeedNode` fields directly (e.g. [`SeedNode::plane`]) instead
    ///   of materialising alerts.
    #[inline]
    pub fn resolve_seed_members<'a>(&self, store: &'a AlertStore) -> Option<Vec<&'a Alert>> {
        self.members
            .iter()
            .map(|&id| store.alerts.get(id.idx()))
            .collect()
    }

    /// Predict the sky position `(RA, Dec)` at a target epoch using the
    /// underlying tangent-plane model.
    ///
    /// This is a thin convenience wrapper around
    /// [`TangentPlaneModel::predict_radec`]. It returns the **deterministic**
    /// best-fit position given the kinematic parameters stored in
    /// [`SeedNode::plane`].
    ///
    /// Arguments
    /// ---------
    /// * `t_target` – Target epoch (MJD TT) at which to evaluate the model.
    ///
    /// Return
    /// ------
    /// * `(ra, dec)` – Predicted right ascension and declination in radians
    ///   (J2000, same frame as stored alerts).
    ///
    /// Notes
    /// -----
    /// * No uncertainty, padding or cone geometry is returned here. If you
    ///   need an uncertainty-aware region for candidate search, use
    ///   [`SeedNode::predict_cone`] or [`SeedNode::cone_candidates`] instead.
    /// * The prediction is valid only in the local neighbourhood where the
    ///   tangent-plane approximation and the underlying fit are reliable.
    #[inline]
    pub fn predict_radec(&self, t_target: MjdTt) -> (Radians, Radians) {
        self.plane.predict_radec(t_target)
    }

    /// Predict a sky **cone** `(RA, Dec, radius)` covering the possible
    /// position of this seed at a target epoch.
    ///
    /// The prediction proceeds in two stages:
    ///
    /// 1. Use the tangent-plane model to compute a base prediction:
    ///    - propagate the kinematics to `t_target`,
    ///    - inflate the radius according to the noise model and `k_sigma`,
    ///      via [`TangentPlaneModel::predict_cone_base`].
    /// 2. Optionally add a **cell padding** term:
    ///    - if `predictor_params.pad_cell_radius == true`,
    ///      add [`SpatialBinner::cell_radius`] so the cone safely covers
    ///      neighbouring spatial cells during bucket-based queries.
    ///
    /// Arguments
    /// ---------
    /// * `t_target` – Target epoch (MJD TT) at which to predict the cone.
    /// * `binner` – Spatial binner used for bucket construction (e.g. HEALPix).
    ///   Only `cell_radius()` is used here.
    /// * `predictor_params` – Predictor configuration containing:
    ///   - `noise` – noise model used to inflate the cone radius,
    ///   - `k_sigma` – multiplicative factor for the uncertainty radius,
    ///   - `pad_cell_radius` – whether to add an extra cell-radius padding.
    ///
    /// Return
    /// ------
    /// * `(ra_center, dec_center, radius)` – Centre and angular radius of the
    ///   predicted search cone, all in radians.
    ///
    /// Notes
    /// -----
    /// * This routine does **not** perform any index lookup; it only produces
    ///   a geometric region. Use [`SeedNode::cone_candidates`] to directly
    ///   query a [`SeedSpatialIndex`].
    /// * The radius is meant to be conservative: it should cover the joint
    ///   effect of:
    ///   - the fitted motion model uncertainty,
    ///   - the error model in `predictor_params.noise`,
    ///   - an optional spatial-cell padding.
    #[inline]
    pub fn predict_cone<Bs: SpatialBinner>(
        &self,
        t_target: MjdTt,
        binner: &Bs,
        predictor_params: &PredictorParams,
    ) -> (Radians, Radians, f64) {
        // Predict the cone via the tangent-plane model (centre + radius),
        // then optionally add a padding term based on the spatial cell radius.
        let (ra, dec, mut radius) = self.plane.predict_cone_base(
            t_target,
            &predictor_params.noise,
            predictor_params.k_sigma,
        );
        if predictor_params.pad_cell_radius {
            radius += binner.cell_radius();
        }
        (ra, dec, radius)
    }

    /// Score inter-night edge candidates from this seed to a **time-sorted** set of
    /// right-hand seeds using **spatio-temporal binning** followed by **exact
    /// kinematic scoring**.
    ///
    /// This method targets the common inter-night linking case where:
    /// - all `right` seeds belong to a **single night**,
    /// - `right` is **sorted by `SeedNode::plane.epoch_mid` (ascending)**,
    /// - spatial candidate fan-out must be reduced by **partitioning in time**
    ///   and building **one spatial index per time bin**,
    /// - the temporal partitioning is provided by a generic [`TimeBinner`]
    ///   (uniform bins, cadence-aware bins, etc.).
    ///
    /// ## Algorithm overview
    ///
    /// The algorithm proceeds as follows:
    ///
    /// 1. **Time binning**
    ///    The time span covered by `right` is partitioned into bins using
    ///    `time_binner.bins_in_range(t_min_r, t_max_r)`. The concrete binning
    ///    strategy (uniform, adaptive, cadence-aware) is entirely delegated
    ///    to the [`TimeBinner`] implementation.
    ///
    /// 2. **Monotone bin slicing**
    ///    Because `right` is sorted by `epoch_mid`, each time bin `[t0, t1)`
    ///    corresponds to a contiguous slice `right[lo..hi)`. These bounds are
    ///    maintained using a **monotone scan** over `right`:
    ///    - `lo` and `hi` only move forward,
    ///    - no binary search is performed inside the bin loop.
    ///
    /// 3. **Spatial index per time bin**
    ///    For each non-empty slice `right[lo..hi)`, a dedicated
    ///    [`SeedSpatialIndex`] is built. This ensures that the coarse spatial
    ///    search is **time-consistent by construction**: all returned candidates
    ///    belong to the queried time bin.
    ///
    /// 4. **Per-bin coarse spatial search**
    ///    For each bin:
    ///    - a reference epoch `t_center` (the bin midpoint) is chosen,
    ///    - this seed is propagated to `t_center` using [`SeedNode::predict_cone`],
    ///    - the cone radius is conservatively inflated by
    ///      `(|v| + v_slack) · (Δt / 2)`, where
    ///      `Δt = time_binner.bin_width()`,
    ///      to cover any target epoch within the bin.
    ///    The inflated cone is then queried against the bin-local spatial index.
    ///
    /// 5. **Exact scoring**
    ///    Each spatial candidate is evaluated with [`ScoredEdge::score`] at the
    ///    **true epoch** of the right-hand seed. The returned [`ScoredEdge`]
    ///    encodes the final kinematic and photometric consistency.
    ///
    /// The result is a flat list of scored edges suitable for subsequent Top-K
    /// selection or global graph construction.
    ///
    /// ## Parameters
    ///
    /// * `right` – Slice of candidate right-hand seeds, **sorted by
    ///   `plane.epoch_mid`** (ascending).
    /// * `spatial_binner` – Spatial binner used for cone queries (e.g. HEALPix).
    /// * `time_binner` – Time partitioner defining the binning scheme. Must provide
    ///   `bins_in_range`, `bin_start`, `bin_end`, and `bin_width`.
    /// * `edge_config` – Edge and predictor configuration, including:
    ///   - predictor noise and padding parameters,
    ///   - optional velocity slack `v_slack`.
    /// * `delta_revisit` – Revisit separation between `left` and `right` nights,
    ///   expressed as an integer ≥ 1.
    ///
    /// ## Returns
    ///
    /// * `Vec<ScoredEdge>` – All scored edge candidates from this seed to `right`
    ///   seeds that pass the per-bin spatial prefilter and the exact scorer. The
    ///   vector is **not sorted** and may contain more entries than the final Top-K;
    ///   downstream code is expected to perform selection or truncation.
    ///
    /// ## Invariants
    ///
    /// - `right` **must** be sorted by `SeedNode::plane.epoch_mid`.
    /// - `time_binner` must be consistent with the timestamps in `right`
    ///   (i.e. `bin_start`, `bin_end`, and `bin_width` define a coherent partition).
    ///
    /// ## Complexity
    ///
    /// ### Notation
    /// - `N = right.len()` – total number of right-hand seeds (one night),
    /// - `B` – number of time bins,
    /// - `n_b` – number of seeds in bin `b` (`Σ_b n_b = N`),
    /// - `C_b` – number of candidates returned by the bin-local cone query,
    /// - `score_cost` – cost of one [`ScoredEdge::score`] call.
    ///
    /// ### This method (spatio-temporal, index per bin)
    ///
    /// - **Time slicing**: `O(N + B)` via a monotone scan of `right`,
    /// - **Index construction**: `Σ_b O(n_b) = O(N)` (each seed is indexed once),
    /// - **Scoring**: `Σ_b (C_b · score_cost)`.
    ///
    /// Overall complexity:
    /// ```
    /// O(N + (Σ_b C_b) · score_cost)
    /// ```
    ///
    /// ## Notes
    ///
    /// - Building a spatial index per bin increases preprocessing work
    ///   (many small indices) but can drastically reduce candidate fan-out when
    ///   nightly spatial density is high.
    /// - The coarse cone is intentionally conservative; false positives are
    ///   expected and filtered out by the exact scorer.
    ///
    /// ## See also
    ///
    /// * [`TimeBinner`] – Time partitioning interface (uniform or custom).
    /// * [`UniformTimeBinner`] – Uniform partitioning of the MJD(TT) axis.
    /// * [`SeedNode::predict_cone`] – Coarse kinematic prediction.
    /// * [`SeedSpatialIndex`] – Bucket-based spatial index used per time bin.
    /// * [`ScoredEdge::score`] – Exact inter-night edge scoring routine.
    pub fn score_edge_candidates<B: SpatialBinner, T: TimeBinner>(
        &self,
        right: &[SeedNode], // sorted by epoch_mid
        spatial_binner: &B,
        time_binner: &T,
        edge_config: &EdgeConfig,
        delta_revisit: u32,
    ) -> Vec<ScoredEdge> {
        let mut scored: Vec<ScoredEdge> = Vec::with_capacity(32);

        if right.is_empty() {
            return scored;
        }

        let pred_cfg = &edge_config.predictor_config;
        let score_cfg = &edge_config.score_config;

        // Build once: SeedId -> &SeedNode for the whole right slice.
        // (Assumes SeedId uniqueness within the night slice, which is your invariant.)
        let right_by_id: AHashMap<SeedId, &SeedNode> =
            right.iter().map(|s| (s.seed_id, s)).collect();

        // Time span in O(1) because `right` is time-sorted.
        let t_min_r = right[0].plane.epoch_mid;
        let t_max_r = right[right.len() - 1].plane.epoch_mid;

        let bins = time_binner.bins_in_range(t_min_r, t_max_r);
        if bins.is_empty() {
            return scored;
        }

        // Left seed speed on tangent plane (rad/day), with optional slack.
        let v_xy = self.plane.vel_xy;
        let speed = (v_xy[0].mul_add(v_xy[0], v_xy[1] * v_xy[1])).sqrt();
        let effective_speed = (speed + pred_cfg.v_slack).max(0.0);

        // Half-bin width used for conservative time padding.
        let half_bin_width_days = 0.5 * time_binner.bin_width().max(1e-12);

        // Monotone scan pointers for the bin slices.
        let mut lo: usize = 0;
        let mut hi: usize = 0;

        // Reused buffers.
        let mut cover_keys_buf: Vec<SpatialKey> = Vec::with_capacity(256);
        let mut candidate_ids_buf: Vec<SeedId> = Vec::with_capacity(1024);

        for bin in bins {
            let bin_start = time_binner.bin_start(bin.0);
            let bin_end = time_binner.bin_end(bin.0);
            let bin_center = 0.5 * (bin_start + bin_end);

            // Advance `lo` to first index with epoch_mid >= bin_start.
            while lo < right.len() && right[lo].plane.epoch_mid < bin_start {
                lo += 1;
            }
            // Ensure `hi >= lo`, then advance `hi` to first index with epoch_mid >= bin_end.
            if hi < lo {
                hi = lo;
            }
            while hi < right.len() && right[hi].plane.epoch_mid < bin_end {
                hi += 1;
            }

            if lo == hi {
                continue;
            }

            // Build a spatial index for this time bin only (borrows right[lo..hi]).
            let index_bin = SeedSpatialIndex::build(&right[lo..hi], spatial_binner);

            // Coarse cone at bin center + time padding to cover the full bin.
            let (ra_center, dec_center, mut cone_radius) =
                self.predict_cone(bin_center, spatial_binner, pred_cfg);

            cone_radius += effective_speed * half_bin_width_days;

            // Fill candidate_ids_buf with SeedIds only (no borrowed refs).
            index_bin.cone_query_ids_into(
                spatial_binner,
                ra_center,
                dec_center,
                cone_radius,
                &mut cover_keys_buf,
                &mut candidate_ids_buf,
            );

            // Exact scoring at each candidate's true epoch.
            scored.extend(candidate_ids_buf.iter().filter_map(|&sid| {
                let &target_seed = right_by_id.get(&sid)?;
                ScoredEdge::score(self, target_seed, score_cfg, delta_revisit)
            }));
        }

        scored
    }

    /// Retrieve **candidate neighbour seeds** from a [`SeedSpatialIndex`]
    /// using this seed’s predicted cone.
    ///
    /// This is the high-level entry point for inter-night candidate search:
    ///
    /// 1. Compute the search cone `(ra, dec, radius)` at `t_target` using
    ///    [`SeedNode::predict_cone`].
    /// 2. Invoke [`SeedSpatialIndex::cone_query`] with that cone to retrieve
    ///    all `SeedId`s falling in the approximate spatial cover.
    ///
    /// Arguments
    /// ---------
    /// * `t_target` – Target epoch (MJD TT) at which to predict the cone.
    /// * `index` – Per-night spatial index for seeds, typically built from
    ///   all seeds of the same night as this node.
    /// * `binner` – Spatial binner used both at index construction time and
    ///   for neighbour lookup (e.g. HEALPix).
    /// * `params` – Predictor configuration; see
    ///   [`SeedNode::predict_cone`] for details.
    ///
    /// Return
    /// ------
    /// * `Vec<SeedId>` – List of candidate neighbour seeds whose spatial
    ///   cells intersect the predicted cone.
    ///
    /// Notes
    /// -----
    /// * The result is **approximate by design**:
    ///   - some candidates might lie slightly outside the strict cone,
    ///   - some very marginal matches could be missed depending on the
    ///     behaviour of [`SpatialBinner::neighbors`].
    /// * Downstream code should always apply a more precise filter
    ///   (e.g. exact angular separation or orbit-fitting residuals) on the
    ///   returned candidates.
    #[inline]
    pub fn cone_candidates<'a, Bs: SpatialBinner>(
        &self,
        t_target: MjdTt,
        index: &'a SeedSpatialIndex,
        binner: &'a Bs,
        params: &PredictorParams,
    ) -> Vec<&'a SeedNode> {
        // Use the tangent-plane cone prediction to query the spatial index.
        let (ra, dec, radius) = self.predict_cone(t_target, binner, params);
        index.cone_query(binner, ra, dec, radius).collect()
    }

    /// Build a [`SeedNode`] from a **pair** of alerts.
    ///
    /// This constructor fits a **linear tangent-plane model** from two
    /// detections `(a, b)`:
    ///
    /// 1. Define the tangent-plane centre as the spherical midpoint of
    ///    `a` and `b`.
    /// 2. Project both alerts to tangent coordinates via [`radec_to_tangent`].
    /// 3. Use their midpoint as the reference position `pₘ`.
    /// 4. Estimate velocity by finite difference in tangent coordinates.
    /// 5. Build diagonal covariance matrices for:
    ///    - position, from the RA/Dec uncertainties of `a` and `b`,
    ///    - velocity, from the position errors and `Δt⁻²`.
    /// 6. Aggregate photometry (mean + dispersion of fluxes).
    ///
    /// An optional **physical realism filter** can be applied via
    /// `max_speed_rad_per_day`: if the fitted speed exceeds this threshold,
    /// the seed is discarded and `None` is returned.
    ///
    /// Arguments
    /// ---------
    /// * `seed_id` – Identifier to assign to the newly built seed.
    /// * `night_id` – Night to which both alerts belong.
    /// * `alert_a` – First alert (earlier or arbitrary order, but consistent with `b`).
    /// * `alert_b` – Second alert.
    /// * `max_speed_rad_per_day` – Optional maximum allowed angular speed in
    ///   radians per day. If `Some(vmax)` and the fitted speed satisfies
    ///   `‖v‖ > vmax`, the function returns `None`.
    ///
    /// Return
    /// ------
    /// * `Some(SeedNode)` if a valid linear model could be built and passes
    ///   the speed filter.
    /// * `None` if the fitted speed exceeds `max_speed_rad_per_day`.
    ///
    /// Notes
    /// -----
    /// * The resulting seed always has:
    ///   - `n_obs == 2`,
    ///   - `members == [a.id, b.id]` in that order.
    /// * Covariances are approximated as **isotropic** in the tangent plane,
    ///   using the maximum of RA/Dec errors as a scalar proxy per alert.
    /// * This is intended as a cheap, robust intra-night model; it is not a
    ///   substitute for a full orbit fit.
    pub fn from_pair(
        seed_id: SeedId,
        night_id: NightId,
        alert_a: &Alert,
        alert_b: &Alert,
        max_speed_rad_per_day: Option<f64>,
    ) -> Option<Self> {
        let ta = alert_a.mjd_tt;
        let tb = alert_b.mjd_tt;
        let tm = 0.5 * (ta + tb);
        let dt = tb - ta;
        let inv_dt = 1.0 / dt;
        let inv_dt2 = inv_dt * inv_dt;

        // Tangent-plane centre = spherical midpoint of the two endpoints.
        let (ra0, dec0) = spherical_midpoint(alert_a.ra, alert_a.dec, alert_b.ra, alert_b.dec);
        let center = TangentCenter::new(ra0, dec0);

        // Tangent-plane coordinates of the two detections.
        let pa = radec_to_tangent(alert_a.ra, alert_a.dec, ra0, dec0);
        let pb = radec_to_tangent(alert_b.ra, alert_b.dec, ra0, dec0);

        // Midpoint position in tangent coordinates.
        let pm = [(pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5];

        // Convert the midpoint back to sky coordinates for convenience.
        let (ra_mid, dec_mid) = tangent_to_radec(pm[0], pm[1], ra0, dec0);

        // Linear tangent-plane velocity estimate.
        let vx = (pb[0] - pa[0]) * inv_dt;
        let vy = (pb[1] - pa[1]) * inv_dt;

        // Optional speed sanity check.
        if let Some(vmax) = max_speed_rad_per_day {
            let speed2 = vx.mul_add(vx, vy * vy);
            if speed2 > vmax * vmax {
                return None;
            }
        }

        // Position and velocity covariance estimates (isotropic).
        let sa = alert_a.ra_err.max(alert_a.dec_err);
        let sb = alert_b.ra_err.max(alert_b.dec_err);
        let s2 = 0.5 * (sa * sa + sb * sb);
        let cov_pos = [[s2, 0.0], [0.0, s2]];
        let vel_var = 2.0 * s2 * inv_dt2;
        let cov_vel = [[vel_var, 0.0], [0.0, vel_var]];

        // Simple two-point flux statistics.
        let flux_mean = (alert_a.flux + alert_b.flux) * 0.5;
        let flux_std = ((alert_a.flux - flux_mean).abs() + (alert_b.flux - flux_mean).abs()) * 0.5;
        let photom = Photometry::from_pair(
            flux_mean as f32,
            flux_std as f32,
            alert_a.band,
            alert_b.band,
        );

        let plane = TangentPlaneModel::new(
            center,
            tm,
            pm,
            [vx, vy],
            None,
            cov_pos,
            cov_vel,
            ra_mid,
            dec_mid,
        );

        Some(SeedNode {
            seed_id,
            night_id,
            plane,
            photom,
            n_obs: 2,
            members: vec![alert_a.id, alert_b.id],
        })
    }

    /// Build a [`SeedNode`] from a **triplet** of alerts.
    ///
    /// Compared to [`SeedNode::from_pair`], this constructor fits a
    /// **quadratic** tangent-plane model that includes:
    ///
    /// - position at the mean epoch,
    /// - velocity,
    /// - acceleration (second-order term) in both tangent coordinates.
    ///
    /// The procedure is:
    ///
    /// 1. Define the tangent-plane centre as the spherical midpoint of `a`
    ///    and `c` (endpoints of the triplet).
    /// 2. Project `a`, `b`, `c` to tangent coordinates.
    /// 3. Shift observation times to `Δt = t_i − t̄` with `t̄ = (t_a + t_b + t_c)/3`.
    /// 4. Fit a quadratic polynomial independently in `x` and `y` using
    ///    [`fit_quad_1d`] to obtain `(p0, v, a)` for each axis.
    /// 5. Convert the reference position `(p0x, p0y)` back to RA/Dec for
    ///    convenience.
    /// 6. Derive position and velocity covariances from RA/Dec uncertainties
    ///    and a characteristic time span `Δt_char = max(t_c − t_a, 1e-6)`.
    /// 7. Aggregate photometry from the three flux measurements.
    ///
    /// Arguments
    /// ---------
    /// * `seed_id` – Identifier to assign to the newly built seed.
    /// * `night_id` – Night to which the three alerts belong.
    /// * `alert_a` – First alert in the triplet.
    /// * `alert_b` – Second alert.
    /// * `alert_c` – Third alert.
    ///
    /// Return
    /// ------
    /// * `SeedNode` – A quadratic tangent-plane model with:
    ///   - `n_obs == 3`,
    ///   - `members == [a.id, b.id, c.id]`,
    ///   - non-zero acceleration components stored in `plane`.
    ///
    /// Notes
    /// -----
    /// * The quadratic fit is performed independently in each coordinate,
    ///   assuming small-angle behaviour in the tangent plane.
    /// * The acceleration is particularly useful for fast-moving or
    ///   curved tracks (e.g. near opposition or for close encounters),
    ///   but is still an approximation of the true orbit.
    /// * The characteristic time `dt_char` is clamped to `1e-6` to avoid
    ///   numerical blow-up for nearly simultaneous observations.
    pub fn from_triplet(
        seed_id: SeedId,
        night_id: NightId,
        alert_a: &Alert,
        alert_b: &Alert,
        alert_c: &Alert,
    ) -> Self {
        let (ta, tb, tc) = (alert_a.mjd_tt, alert_b.mjd_tt, alert_c.mjd_tt);
        let tm = (ta + tb + tc) / 3.0;

        let (ra0, dec0) = spherical_midpoint(alert_a.ra, alert_a.dec, alert_c.ra, alert_c.dec);
        let center = TangentCenter::new(ra0, dec0);

        let pa = radec_to_tangent(alert_a.ra, alert_a.dec, ra0, dec0);
        let pb = radec_to_tangent(alert_b.ra, alert_b.dec, ra0, dec0);
        let pc = radec_to_tangent(alert_c.ra, alert_c.dec, ra0, dec0);

        // Quadratic fits in x and y around the mean epoch.
        let (p0x, vx, ax) = fit_quad_1d([ta - tm, tb - tm, tc - tm], [pa[0], pb[0], pc[0]]);
        let (p0y, vy, ay) = fit_quad_1d([ta - tm, tb - tm, tc - tm], [pa[1], pb[1], pc[1]]);

        let (ra_mid, dec_mid) = tangent_to_radec(p0x, p0y, ra0, dec0);

        // Aggregate uncertainty estimates.
        let sa = alert_a.ra_err.max(alert_a.dec_err);
        let sb = alert_b.ra_err.max(alert_b.dec_err);
        let sc = alert_c.ra_err.max(alert_c.dec_err);
        let s2 = (sa * sa + sb * sb + sc * sc) / 3.0;

        let dt_char = (tc - ta).max(1e-6);
        let inv_dt2 = 1.0 / (dt_char * dt_char);

        let cov_pos = [[s2 / 3.0, 0.0], [0.0, s2 / 3.0]];
        let vel_var = s2 * inv_dt2;
        let cov_vel = [[vel_var, 0.0], [0.0, vel_var]];

        // Three-point flux statistics.
        let flux_mean = (alert_a.flux + alert_b.flux + alert_c.flux) / 3.0;
        let flux_std = ((alert_a.flux - flux_mean).abs()
            + (alert_b.flux - flux_mean).abs()
            + (alert_c.flux - flux_mean).abs())
            / 3.0;

        let photom = Photometry::from_triplet(
            flux_mean as f32,
            flux_std as f32,
            alert_a.band,
            alert_b.band,
            alert_c.band,
        );

        let plane = TangentPlaneModel::new(
            center,
            tm,
            [p0x, p0y],
            [vx, vy],
            Some([ax, ay]),
            cov_pos,
            cov_vel,
            ra_mid,
            dec_mid,
        );

        SeedNode {
            seed_id,
            night_id,
            plane,
            photom,
            n_obs: 3,
            members: vec![alert_a.id, alert_b.id, alert_c.id],
        }
    }
}

#[cfg(test)]
mod seed_node_tests {
    use super::*;
    use proptest::prelude::*;

    use crate::{
        alerts::AlertStore,
        astro_math::{ang_sep, arcsec_to_rad},
        engine_config::propagator_config::{ModelNoise, PredictorParams},
        seeding::seed_spatial_index::SeedSpatialIndex,
        spacetime_bucket::{
            bucket::build_bucket_index, healpix_binner::HealpixBinner,
            uniform_time_binner::UniformTimeBinner,
        },
    };

    const LAT_EPS: f64 = 1e-6;

    /* ------------------------- helpers ------------------------- */

    fn mk_alert(id: AlertId, ra: f64, dec: f64, mjd_tt: f64, band: u8, flux: f32) -> Alert {
        Alert {
            id,
            dia_source_id: id.idx() as u64,
            ra,
            ra_err: arcsec_to_rad(0.5),
            dec,
            dec_err: arcsec_to_rad(0.5),
            mjd_tt,
            flux,
            flux_err: 0.0,
            band,
        }
    }

    fn default_predictor_params() -> PredictorParams {
        PredictorParams {
            noise: ModelNoise {
                variance_floor: 0.0,
                drift_per_day: 0.0,
                curvature_per_day2: 0.0,
            },
            k_sigma: 3.0,
            pad_cell_radius: true,
            time_bin_dt: 1.0,
            v_slack: 0.0,
        }
    }

    /* ------------------------- unit tests ------------------------- */

    #[test]
    fn from_pair_builds_expected_members_and_nobs() {
        let t0 = 60000.0;
        let dec: f64 = 0.25;
        let dr = arcsec_to_rad(6.0) / dec.cos();

        let a = mk_alert(AlertId::new(0), 1.0, dec, t0, 1, 1000.0);
        let b = mk_alert(
            AlertId::new(1),
            1.0 + dr,
            dec,
            t0 + 10.0 / 1440.0,
            1,
            1002.0,
        );

        let sn = SeedNode::from_pair(SeedId::new(7), NightId::new(42), &a, &b, None)
            .expect("pair should produce a seed");

        assert_eq!(sn.seed_id, SeedId::new(7));
        assert_eq!(sn.night_id, NightId::new(42));
        assert_eq!(sn.n_obs, 2);
        assert_eq!(sn.members, vec![a.id, b.id]);

        // Velocity is roughly dr / dt on the tangent plane.
        let dt = (b.mjd_tt - a.mjd_tt).max(1e-12);
        let (ra0, dec0) = spherical_midpoint(a.ra, a.dec, b.ra, b.dec);
        let pa = radec_to_tangent(a.ra, a.dec, ra0, dec0);
        let pb = radec_to_tangent(b.ra, b.dec, ra0, dec0);
        let vx = (pb[0] - pa[0]) / dt;
        let vy = (pb[1] - pa[1]) / dt;
        assert!((sn.plane.vel_xy[0] - vx).abs() < 1e-9);
        assert!((sn.plane.vel_xy[1] - vy).abs() < 1e-9);
    }

    #[test]
    fn from_pair_speed_filter_rejects_fast_pairs() {
        let t0 = 60000.0;
        let dec: f64 = 0.2;
        let slow_sep = arcsec_to_rad(5.0) / dec.cos();
        let fast_sep = arcsec_to_rad(200.0) / dec.cos();

        let a = mk_alert(AlertId::new(0), 2.0, dec, t0, 1, 1000.0);
        let b_slow = mk_alert(
            AlertId::new(1),
            2.0 + slow_sep,
            dec,
            t0 + 5.0 / 1440.0,
            1,
            1000.0,
        );
        let b_fast = mk_alert(
            AlertId::new(2),
            2.0 + fast_sep,
            dec,
            t0 + 5.0 / 1440.0,
            1,
            1000.0,
        );

        let dt = 5.0 / 1440.0;
        let speed_slow = slow_sep / dt;
        let speed_fast = fast_sep / dt;
        assert!(speed_fast > speed_slow);

        let vmax = (speed_slow + speed_fast) * 0.5;

        let keep = SeedNode::from_pair(SeedId::new(0), NightId::new(1), &a, &b_slow, Some(vmax));
        let drop = SeedNode::from_pair(SeedId::new(1), NightId::new(1), &a, &b_fast, Some(vmax));

        assert!(keep.is_some());
        assert!(drop.is_none());
    }

    #[test]
    fn from_triplet_builds_expected_members_and_nobs() {
        let t0 = 60000.0;
        let dec: f64 = 0.3;
        let dr = arcsec_to_rad(6.0) / dec.cos();

        let a = mk_alert(AlertId::new(0), 1.0, dec, t0, 1, 1000.0);
        let b = mk_alert(
            AlertId::new(1),
            1.0 + dr,
            dec,
            t0 + 10.0 / 1440.0,
            1,
            1001.0,
        );
        let c = mk_alert(
            AlertId::new(2),
            1.0 + 2.0 * dr,
            dec,
            t0 + 20.0 / 1440.0,
            1,
            1002.0,
        );

        let sn = SeedNode::from_triplet(SeedId::new(3), NightId::new(99), &a, &b, &c);

        assert_eq!(sn.seed_id, SeedId::new(3));
        assert_eq!(sn.night_id, NightId::new(99));
        assert_eq!(sn.n_obs, 3);
        assert_eq!(sn.members, vec![a.id, b.id, c.id]);

        // Midpoint time close to average.
        let tm = (a.mjd_tt + b.mjd_tt + c.mjd_tt) / 3.0;
        assert!((sn.plane.epoch_mid - tm).abs() < 1e-12);
    }

    #[test]
    fn resolve_seed_members_returns_alert_refs() {
        let t0 = 60000.0;
        let dec: f64 = 0.25;
        let dr = arcsec_to_rad(4.0) / dec.cos();

        let a = mk_alert(AlertId::new(0), 1.0, dec, t0, 1, 1000.0);
        let b = mk_alert(
            AlertId::new(1),
            1.0 + dr,
            dec,
            t0 + 10.0 / 1440.0,
            1,
            1001.0,
        );

        let store = AlertStore::new(t0.floor(), vec![a.clone(), b.clone()]);

        let sn = SeedNode::from_pair(SeedId::new(0), NightId::new(1), &a, &b, None).unwrap();
        let refs = sn.resolve_seed_members(&store).expect("valid ids");

        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].id, a.id);
        assert_eq!(refs[1].id, b.id);
    }

    #[test]
    fn predict_radec_and_cone_are_consistent() {
        let t0 = 60000.0;
        let dec: f64 = 0.25;
        let dr = arcsec_to_rad(8.0) / dec.cos();

        let a = mk_alert(AlertId::new(0), 2.0, dec, t0, 1, 1000.0);
        let b = mk_alert(
            AlertId::new(1),
            2.0 + dr,
            dec,
            t0 + 10.0 / 1440.0,
            1,
            1000.0,
        );

        let sn = SeedNode::from_pair(SeedId::new(1), NightId::new(5), &a, &b, None).unwrap();

        let predict_params = default_predictor_params();
        let tb = b.mjd_tt;

        let (ra_pred, dec_pred) = sn.predict_radec(tb);
        let (ra_cone, dec_cone, radius) =
            sn.predict_cone(tb, &HealpixBinner::new(8), &predict_params);

        let d = ang_sep(ra_pred, dec_pred, ra_cone, dec_cone);
        assert!(d <= radius + 1e-12);
    }

    #[test]
    fn cone_candidates_returns_seed_ids_in_cover_cells() {
        let spatial_binner = HealpixBinner::new(8);
        let time_binner = UniformTimeBinner::new(60000.0, 5.0 / 1440.0);
        let params = default_predictor_params();

        let t0 = 60010.0;
        let dec: f64 = 0.3;
        let dr = arcsec_to_rad(6.0) / dec.cos();

        // Two seeds on a small track: s1 from pair a→b, s2 from pair b→c.
        let a = mk_alert(AlertId::new(0), 1.0, dec, t0, 1, 1000.0);
        let b = mk_alert(AlertId::new(1), 1.0 + dr, dec, t0 + 5.0 / 1440.0, 1, 1001.0);
        let c = mk_alert(
            AlertId::new(2),
            1.0 + 2.0 * dr,
            dec,
            t0 + 10.0 / 1440.0,
            1,
            1002.0,
        );

        let s1 = SeedNode::from_pair(SeedId::new(0), NightId::new(9), &a, &b, None).unwrap();
        let s2 = SeedNode::from_pair(SeedId::new(1), NightId::new(9), &b, &c, None).unwrap();

        let vec_seed = vec![s1.clone(), s2.clone()];
        // Build a spatial index and bucket index (for completeness).
        let index = SeedSpatialIndex::build(&vec_seed, &spatial_binner);
        let _bucket_index = build_bucket_index(
            &vec![a.clone(), b.clone(), c.clone()],
            &spatial_binner,
            &time_binner,
        );

        // Query around s1 prediction near time of c, expect to find s2 (future position).
        let (ra, dec, radius) = s1.predict_cone(c.mjd_tt, &spatial_binner, &params);
        let candidates: Vec<&SeedNode> =
            index.cone_query(&spatial_binner, ra, dec, radius).collect();

        assert!(candidates.contains(&&s2));
    }

    /* ------------------------- property-based tests ------------------------- */

    fn ra_strategy() -> impl Strategy<Value = f64> {
        0.0f64..(2.0 * std::f64::consts::PI)
    }

    fn dec_strategy() -> impl Strategy<Value = f64> {
        (-(std::f64::consts::PI / 2.0 - LAT_EPS))..(std::f64::consts::PI / 2.0 - LAT_EPS)
    }

    fn t_strategy() -> impl Strategy<Value = f64> {
        60000.0f64..60000.1667f64 // ~4h window
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 32,
            .. ProptestConfig::default()
        })]

        /// For random pairs with increasing time, `from_pair` must:
        /// - produce a seed with 2 members and n_obs=2,
        /// - assign a consistent midpoint epoch,
        /// - yield finite velocities.
        #[test]
        fn prop_from_pair_basic_invariants(
            samples in proptest::collection::vec((ra_strategy(), dec_strategy(), t_strategy()), 2..60)
        ) {
            let mut alerts: Vec<Alert> = samples.iter().enumerate().map(|(i, (ra, dec, t))| {
                mk_alert(AlertId::new(i as u32), *ra, *dec, *t, 1, 1000.0)
            }).collect();

            alerts.sort_by(|a,b| a.mjd_tt.partial_cmp(&b.mjd_tt).unwrap());

            let mut count = 0usize;
            for i in 0..alerts.len().saturating_sub(1) {
                let a = &alerts[i];
                let b = &alerts[i+1];
                if b.mjd_tt <= a.mjd_tt { continue; }
                if let Some(sn) = SeedNode::from_pair(SeedId::new(i as u64), NightId::new(1), a, b, None) {
                    count += 1;
                    prop_assert_eq!(sn.n_obs, 2);
                    prop_assert_eq!(sn.members, vec![a.id, b.id]);
                    let tm = 0.5 * (a.mjd_tt + b.mjd_tt);
                    prop_assert!((sn.plane.epoch_mid - tm).abs() < 1e-9);
                    prop_assert!(sn.plane.vel_xy[0].is_finite() && sn.plane.vel_xy[1].is_finite());
                }
            }
            prop_assert!(count > 0);
        }

        /// For random triplets with strictly increasing times, `from_triplet` must:
        /// - produce a seed with 3 members and n_obs=3,
        /// - have a midpoint epoch within the convex hull of times,
        /// - return finite kinematic parameters.
        #[test]
        fn prop_from_triplet_basic_invariants(
            samples in proptest::collection::vec((ra_strategy(), dec_strategy(), t_strategy()), 3..60)
        ) {
            let mut alerts: Vec<Alert> = samples.iter().enumerate().map(|(i, (ra, dec, t))| {
                mk_alert(AlertId::new(i as u32), *ra, *dec, *t, 1, 1000.0)
            }).collect();

            alerts.sort_by(|a,b| a.mjd_tt.partial_cmp(&b.mjd_tt).unwrap());

            let mut built = 0usize;
            for i in 0..alerts.len().saturating_sub(2) {
                let (a, b, c) = (&alerts[i], &alerts[i+1], &alerts[i+2]);
                if !(a.mjd_tt < b.mjd_tt && b.mjd_tt < c.mjd_tt) { continue; }

                let sn = SeedNode::from_triplet(SeedId::new(i as u64), NightId::new(2), a, b, c);
                built += 1;

                prop_assert_eq!(sn.n_obs, 3);
                prop_assert_eq!(sn.members, vec![a.id, b.id, c.id]);

                let tmin = a.mjd_tt.min(b.mjd_tt).min(c.mjd_tt);
                let tmax = a.mjd_tt.max(b.mjd_tt).max(c.mjd_tt);
                prop_assert!(sn.plane.epoch_mid >= tmin && sn.plane.epoch_mid <= tmax);

                prop_assert!(sn.plane.vel_xy[0].is_finite() && sn.plane.vel_xy[1].is_finite());
                // acc_xy is Some; check finiteness.
                let acc = sn.plane.acc_xy.expect("triplet fits a quadratic");
                prop_assert!(acc[0].is_finite() && acc[1].is_finite());
            }
            prop_assert!(built > 0);
        }

        /// Predict round-trip: `predict_cone` centre should be near `predict_radec`
        /// at the same epoch, within the returned cone radius.
        #[test]
        fn prop_predict_cone_covers_predict_radec(
            samples in proptest::collection::vec((ra_strategy(), dec_strategy(), t_strategy()), 2..40)
        ) {
            let alerts: Vec<Alert> = samples.iter().enumerate().map(|(i, (ra, dec, t))| {
                mk_alert(AlertId::new(i as u32), *ra, *dec, *t, 1, 1000.0)
            }).collect();

            if alerts.len() < 2 { return Ok(()); }

            let a = &alerts[0];
            let b = &alerts[1];
            if b.mjd_tt <= a.mjd_tt { return Ok(()); }

            let sn = match SeedNode::from_pair(SeedId::new(0), NightId::new(3), a, b, None) {
                Some(s) => s,
                None => return Ok(()),
            };

            let params = default_predictor_params();
            let t = b.mjd_tt;
            let (rp, dp) = sn.predict_radec(t);
            let (rc, dc, rad) = sn.predict_cone(t, &HealpixBinner::new(8), &params);

            let d = ang_sep(rp, dp, rc, dc);
            prop_assert!(d <= rad + 1e-12);
        }
    }

    mod score_edge_candidates_test {
        use super::super::*;
        use super::*;
        /* ------------------------- helpers: score_edge_candidates ------------------------- */

        fn mk_edge_config_for_tests(predictor: PredictorParams) -> EdgeConfig {
            // Assumption: EdgeConfig has a predictor_config + score_config and implements Default.
            // If your EdgeConfig is different, adapt this helper accordingly.
            EdgeConfig {
                predictor_config: predictor,
                ..Default::default()
            }
        }

        /// Brute-force reference: score every right seed with the exact scorer,
        /// without any spatial/time prefilter.
        fn brute_force_scores(
            left: &SeedNode,
            right: &[SeedNode],
            edge_config: &EdgeConfig,
            delta_revisit: u32,
        ) -> Vec<ScoredEdge> {
            right
                .iter()
                .filter_map(|r| {
                    ScoredEdge::score(left, r, &edge_config.score_config, delta_revisit)
                })
                .collect()
        }

        fn dedup_pairs(mut v: Vec<(SeedId, SeedId)>) -> Vec<(SeedId, SeedId)> {
            v.sort_unstable();
            v.dedup();
            v
        }

        /// A TimeBinner that returns no bins (to test early return path).
        #[derive(Clone, Copy, Debug)]
        struct EmptyTimeBinner;

        impl TimeBinner for EmptyTimeBinner {
            fn bins_in_range(
                &self,
                _t_min: f64,
                _t_max: f64,
            ) -> Vec<crate::spacetime_bucket::time_binner::TimeBin> {
                Vec::new()
            }
            fn bin_start(&self, _bin: i64) -> f64 {
                0.0
            }
            fn bin_end(&self, _bin: i64) -> f64 {
                0.0
            }
            fn bin_width(&self) -> f64 {
                1.0
            }

            fn bin_for(&self, _: MjdTt) -> crate::spacetime_bucket::time_binner::TimeBin {
                todo!()
            }
        }

        /* ------------------------- unit tests: score_edge_candidates ------------------------- */

        #[test]
        fn score_edge_candidates_empty_right_returns_empty() {
            let spatial_binner = HealpixBinner::new(8);
            let params = default_predictor_params();
            let edge_config = mk_edge_config_for_tests(params);
            let delta = 1_u32;

            let t0 = 60000.0;
            let a = mk_alert(AlertId::new(0), 1.0, 0.2, t0, 1, 1000.0);
            let b = mk_alert(
                AlertId::new(1),
                1.0 + arcsec_to_rad(5.0) / 0.2f64.cos(),
                0.2,
                t0 + 5.0 / 1440.0,
                1,
                1000.0,
            );
            let left = SeedNode::from_pair(SeedId::new(0), NightId::new(1), &a, &b, None).unwrap();

            let out = left.score_edge_candidates(
                &[],
                &spatial_binner,
                &UniformTimeBinner::new(t0, 5.0 / 1440.0),
                &edge_config,
                delta,
            );

            assert!(out.is_empty());
        }

        #[test]
        fn score_edge_candidates_empty_bins_returns_empty() {
            let spatial_binner = HealpixBinner::new(8);
            let params = default_predictor_params();
            let edge_config = mk_edge_config_for_tests(params);

            let t0 = 60000.0;
            let dec: f64 = 0.25;
            let dr = arcsec_to_rad(6.0) / dec.cos();

            let a = mk_alert(AlertId::new(0), 1.0, dec, t0, 1, 1000.0);
            let b = mk_alert(
                AlertId::new(1),
                1.0 + dr,
                dec,
                t0 + 10.0 / 1440.0,
                1,
                1001.0,
            );
            let left = SeedNode::from_pair(SeedId::new(0), NightId::new(1), &a, &b, None).unwrap();

            // Right contains something, but time_binner returns no bins => empty output.
            let right = vec![left.clone()];

            let out = left.score_edge_candidates(
                &right,
                &spatial_binner,
                &EmptyTimeBinner,
                &edge_config,
                1,
            );

            assert!(out.is_empty());
        }

        #[test]
        fn score_edge_candidates_single_bin_huge_cone_matches_bruteforce() {
            let spatial_binner = HealpixBinner::new(4); // coarse cells => easier to cover
            let mut params = default_predictor_params();
            // Make the cone very generous so spatial prefilter should not reject anything.
            params.k_sigma = 1e6;
            params.pad_cell_radius = true;
            params.v_slack = 1e6;

            let edge_config = mk_edge_config_for_tests(params);
            let delta = 1_u32;

            let t0 = 60000.0;
            let dec: f64 = 0.2;
            let dr = arcsec_to_rad(10.0) / dec.cos();

            let a = mk_alert(AlertId::new(0), 2.0, dec, t0, 1, 1000.0);
            let b = mk_alert(AlertId::new(1), 2.0 + dr, dec, t0 + 5.0 / 1440.0, 1, 1000.0);
            let left = SeedNode::from_pair(SeedId::new(10), NightId::new(1), &a, &b, None).unwrap();

            // Build a "right night" with multiple seeds, strictly increasing epoch_mid.
            let mut right: Vec<SeedNode> = Vec::new();
            for i in 0..12 {
                let dt = (i as f64 + 1.0) * (5.0 / 1440.0);
                let a_i = mk_alert(
                    AlertId::new(100 + 2 * i),
                    2.0 + (i as f64) * dr,
                    dec,
                    t0 + dt,
                    1,
                    1000.0,
                );
                let b_i = mk_alert(
                    AlertId::new(100 + 2 * i + 1),
                    2.0 + (i as f64 + 1.0) * dr,
                    dec,
                    t0 + dt + 1.0 / 1440.0,
                    1,
                    1000.0,
                );
                let sn = SeedNode::from_pair(
                    SeedId::new(1000 + i as u64),
                    NightId::new(2),
                    &a_i,
                    &b_i,
                    None,
                )
                .unwrap();
                right.push(sn);
            }
            right.sort_by(|x, y| x.plane.epoch_mid.partial_cmp(&y.plane.epoch_mid).unwrap());

            // One bin covering everything (big width).
            let time_binner = UniformTimeBinner::new(t0, 10.0); // 10 days => 1 bin

            let out = left.score_edge_candidates(
                &right,
                &spatial_binner,
                &time_binner,
                &edge_config,
                delta,
            );
            let brute = brute_force_scores(&left, &right, &edge_config, delta);

            let out_pairs = dedup_pairs(out.iter().map(|e| (e.from, e.to)).collect());
            let brute_pairs = dedup_pairs(brute.iter().map(|e| (e.from, e.to)).collect());

            assert_eq!(
                out_pairs, brute_pairs,
                "With extremely large cone, spatio-temporal prefilter should not drop any accepted edges"
            );
        }

        #[test]
        fn score_edge_candidates_multi_bins_huge_cone_matches_bruteforce() {
            let spatial_binner = HealpixBinner::new(5);
            let mut params = default_predictor_params();
            params.k_sigma = 1e6;
            params.pad_cell_radius = true;
            params.v_slack = 1e6;
            params.time_bin_dt = 1.0 / 24.0; // 1 hour, but note: score_edge_candidates uses TimeBinner::bin_width()

            let edge_config = mk_edge_config_for_tests(params);
            let delta = 2_u32;

            let t0 = 60000.0;
            let dec: f64 = 0.15;
            let dr = arcsec_to_rad(12.0) / dec.cos();

            let a = mk_alert(AlertId::new(0), 1.5, dec, t0, 1, 1000.0);
            let b = mk_alert(AlertId::new(1), 1.5 + dr, dec, t0 + 3.0 / 1440.0, 1, 1000.0);
            let left = SeedNode::from_pair(SeedId::new(1), NightId::new(1), &a, &b, None).unwrap();

            // Right seeds spread across a few hours => multiple uniform bins.
            let mut right: Vec<SeedNode> = Vec::new();
            for i in 0..40 {
                let t = t0 + (i as f64) * (6.0 / 1440.0); // every 6 minutes
                let a_i = mk_alert(
                    AlertId::new(200 + 2 * i),
                    1.5 + (i as f64) * dr,
                    dec,
                    t,
                    1,
                    1000.0,
                );
                let b_i = mk_alert(
                    AlertId::new(200 + 2 * i + 1),
                    1.5 + (i as f64 + 1.0) * dr,
                    dec,
                    t + 1.0 / 1440.0,
                    1,
                    1000.0,
                );
                right.push(
                    SeedNode::from_pair(
                        SeedId::new(10_000 + i as u64),
                        NightId::new(2),
                        &a_i,
                        &b_i,
                        None,
                    )
                    .unwrap(),
                );
            }
            right.sort_by(|x, y| x.plane.epoch_mid.partial_cmp(&y.plane.epoch_mid).unwrap());

            let time_binner = UniformTimeBinner::new(t0, 30.0 / 1440.0); // 30 min bins

            let out = left.score_edge_candidates(
                &right,
                &spatial_binner,
                &time_binner,
                &edge_config,
                delta,
            );
            let brute = brute_force_scores(&left, &right, &edge_config, delta);

            let out_pairs = dedup_pairs(out.iter().map(|e| (e.from, e.to)).collect());
            let brute_pairs = dedup_pairs(brute.iter().map(|e| (e.from, e.to)).collect());

            assert_eq!(out_pairs, brute_pairs);
        }

        #[test]
        fn score_edge_candidates_result_is_subset_of_bruteforce_for_normal_cones() {
            let spatial_binner = HealpixBinner::new(8);
            let params = default_predictor_params(); // normal-sized cone
            let edge_config = mk_edge_config_for_tests(params);
            let delta = 1_u32;

            let t0 = 60000.0;
            let dec: f64 = 0.25;
            let dr = arcsec_to_rad(5.0) / dec.cos();

            let a = mk_alert(AlertId::new(0), 1.0, dec, t0, 1, 1000.0);
            let b = mk_alert(AlertId::new(1), 1.0 + dr, dec, t0 + 5.0 / 1440.0, 1, 1000.0);
            let left = SeedNode::from_pair(SeedId::new(0), NightId::new(1), &a, &b, None).unwrap();

            let mut right: Vec<SeedNode> = Vec::new();
            for i in 0..30 {
                let t = t0 + (i as f64) * (10.0 / 1440.0);
                let ra_i = 1.0 + (i as f64) * dr;
                let a_i = mk_alert(AlertId::new(300 + 2 * i), ra_i, dec, t, 1, 1000.0);
                let b_i = mk_alert(
                    AlertId::new(300 + 2 * i + 1),
                    ra_i + dr,
                    dec,
                    t + 1.0 / 1440.0,
                    1,
                    1000.0,
                );
                right.push(
                    SeedNode::from_pair(
                        SeedId::new(5000 + i as u64),
                        NightId::new(2),
                        &a_i,
                        &b_i,
                        None,
                    )
                    .unwrap(),
                );
            }
            right.sort_by(|x, y| x.plane.epoch_mid.partial_cmp(&y.plane.epoch_mid).unwrap());

            let time_binner = UniformTimeBinner::new(t0, 1.0 / 24.0); // 1 hour bins

            let out = left.score_edge_candidates(
                &right,
                &spatial_binner,
                &time_binner,
                &edge_config,
                delta,
            );
            let brute = brute_force_scores(&left, &right, &edge_config, delta);

            let out_pairs: std::collections::HashSet<(SeedId, SeedId)> =
                out.iter().map(|e| (e.from, e.to)).collect();
            let brute_pairs: std::collections::HashSet<(SeedId, SeedId)> =
                brute.iter().map(|e| (e.from, e.to)).collect();

            assert!(
                out_pairs.is_subset(&brute_pairs),
                "Prefilter must never invent edges; it can only drop brute-force accepted edges"
            );
        }

        #[test]
        fn score_edge_candidates_no_duplicate_pairs_in_output() {
            let spatial_binner = HealpixBinner::new(8);
            let mut params = default_predictor_params();
            params.k_sigma = 1e6;
            params.v_slack = 1e6;
            let edge_config = mk_edge_config_for_tests(params);

            let t0 = 60000.0;
            let dec: f64 = 0.25;
            let dr = arcsec_to_rad(5.0) / dec.cos();

            let a = mk_alert(AlertId::new(0), 1.0, dec, t0, 1, 1000.0);
            let b = mk_alert(AlertId::new(1), 1.0 + dr, dec, t0 + 5.0 / 1440.0, 1, 1000.0);
            let left = SeedNode::from_pair(SeedId::new(0), NightId::new(1), &a, &b, None).unwrap();

            let mut right: Vec<SeedNode> = Vec::new();
            for i in 0..20 {
                let t = t0 + (i as f64) * (6.0 / 1440.0);
                let a_i = mk_alert(
                    AlertId::new(400 + 2 * i),
                    1.0 + (i as f64) * dr,
                    dec,
                    t,
                    1,
                    1000.0,
                );
                let b_i = mk_alert(
                    AlertId::new(400 + 2 * i + 1),
                    1.0 + (i as f64 + 1.0) * dr,
                    dec,
                    t + 1.0 / 1440.0,
                    1,
                    1000.0,
                );
                right.push(
                    SeedNode::from_pair(
                        SeedId::new(6000 + i as u64),
                        NightId::new(2),
                        &a_i,
                        &b_i,
                        None,
                    )
                    .unwrap(),
                );
            }
            right.sort_by(|x, y| x.plane.epoch_mid.partial_cmp(&y.plane.epoch_mid).unwrap());

            let out = left.score_edge_candidates(
                &right,
                &spatial_binner,
                &UniformTimeBinner::new(t0, 30.0 / 1440.0),
                &edge_config,
                1,
            );

            let pairs: Vec<(SeedId, SeedId)> = out.iter().map(|e| (e.from, e.to)).collect();
            let pairs_uniq = dedup_pairs(pairs.clone());
            assert_eq!(
                pairs_uniq.len(),
                pairs.len(),
                "output must not contain duplicate (from,to) edges"
            );
        }

        #[test]
        fn score_edge_candidates_duplicate_seed_ids_in_right_does_not_panic() {
            // This tests the internal `right_by_id: AHashMap<SeedId, &SeedNode>`
            // overwrite behavior. It’s an invariant violation in production,
            // but the function should remain robust (no panic).
            let spatial_binner = HealpixBinner::new(8);
            let mut params = default_predictor_params();
            params.k_sigma = 1e6;
            params.v_slack = 1e6;
            let edge_config = mk_edge_config_for_tests(params);

            let t0 = 60000.0;
            let dec: f64 = 0.2;
            let dr = arcsec_to_rad(5.0) / dec.cos();

            let a = mk_alert(AlertId::new(0), 1.0, dec, t0, 1, 1000.0);
            let b = mk_alert(AlertId::new(1), 1.0 + dr, dec, t0 + 5.0 / 1440.0, 1, 1000.0);
            let left = SeedNode::from_pair(SeedId::new(0), NightId::new(1), &a, &b, None).unwrap();

            let sid = SeedId::new(999);

            let a1 = mk_alert(AlertId::new(10), 1.1, dec, t0 + 20.0 / 1440.0, 1, 1000.0);
            let b1 = mk_alert(
                AlertId::new(11),
                1.1 + dr,
                dec,
                t0 + 21.0 / 1440.0,
                1,
                1000.0,
            );
            let r1 = SeedNode::from_pair(sid, NightId::new(2), &a1, &b1, None).unwrap();

            let a2 = mk_alert(AlertId::new(12), 1.2, dec, t0 + 40.0 / 1440.0, 1, 1000.0);
            let b2 = mk_alert(
                AlertId::new(13),
                1.2 + dr,
                dec,
                t0 + 41.0 / 1440.0,
                1,
                1000.0,
            );
            let r2 = SeedNode::from_pair(sid, NightId::new(2), &a2, &b2, None).unwrap();

            let mut right = vec![r1, r2];
            right.sort_by(|x, y| x.plane.epoch_mid.partial_cmp(&y.plane.epoch_mid).unwrap());

            let _out = left.score_edge_candidates(
                &right,
                &spatial_binner,
                &UniformTimeBinner::new(t0, 1.0),
                &edge_config,
                1,
            );

            // No assert needed: "does not panic" is the test.
            // If you want: assert output edges refer to sid only when present.
        }

        /* ------------------------- proptest: score_edge_candidates ------------------------- */

        proptest! {
            #![proptest_config(ProptestConfig {
                cases: 48,
                .. ProptestConfig::default()
            })]

            /// With an extremely large cone (huge k_sigma + v_slack), the spatio-temporal
            /// prefilter should not remove any edge that the exact scorer accepts.
            #[test]
            fn prop_score_edge_candidates_matches_bruteforce_when_cone_is_huge(
                // Generate a strictly increasing list of times (sorted) and mild motion.
                n in 1usize..40
            ) {
                let spatial_binner = HealpixBinner::new(5);

                let mut params = default_predictor_params();
                params.k_sigma = 1e6;
                params.pad_cell_radius = true;
                params.v_slack = 1e6;

                let edge_config = mk_edge_config_for_tests(params);
                let delta = 1u32;

                let t0 = 60000.0;
                let dec: f64 = 0.25;
                let dr = arcsec_to_rad(5.0) / dec.cos();

                // Left seed
                let a = mk_alert(AlertId::new(0), 1.0, dec, t0, 1, 1000.0);
                let b = mk_alert(AlertId::new(1), 1.0 + dr, dec, t0 + 5.0/1440.0, 1, 1000.0);
                let left = SeedNode::from_pair(SeedId::new(0), NightId::new(1), &a, &b, None).unwrap();

                // Right seeds
                let mut right: Vec<SeedNode> = Vec::with_capacity(n);
                for i in 0..n {
                    let t = t0 + (i as f64) * (7.0/1440.0); // 7 minutes step => sorted by construction
                    let ra_i = 1.0 + (i as f64)*dr;
                    let a_i = mk_alert(AlertId::new(500 + 2*i as u32), ra_i, dec, t, 1, 1000.0);
                    let b_i = mk_alert(AlertId::new(500 + 2*i as u32 + 1), ra_i + dr, dec, t + 1.0/1440.0, 1, 1000.0);
                    right.push(SeedNode::from_pair(SeedId::new(10_000 + i as u64), NightId::new(2), &a_i, &b_i, None).unwrap());
                }
                right.sort_by(|x,y| x.plane.epoch_mid.partial_cmp(&y.plane.epoch_mid).unwrap());

                let time_binner = UniformTimeBinner::new(t0, 30.0/1440.0); // 30 min bins (multi-bin typically)

                let out = left.score_edge_candidates(&right, &spatial_binner, &time_binner, &edge_config, delta);
                let brute = brute_force_scores(&left, &right, &edge_config, delta);

                let out_pairs = dedup_pairs(out.iter().map(|e| (e.from, e.to)).collect());
                let brute_pairs = dedup_pairs(brute.iter().map(|e| (e.from, e.to)).collect());

                prop_assert_eq!(out_pairs, brute_pairs);
            }

            /// For normal predictor parameters, score_edge_candidates must never invent edges:
            /// output ⊆ brute-force accepted edges.
            #[test]
            fn prop_score_edge_candidates_is_subset_of_bruteforce(
                n in 1usize..60
            ) {
                let spatial_binner = HealpixBinner::new(8);
                let params = default_predictor_params();
                let edge_config = mk_edge_config_for_tests(params);
                let delta = 1u32;

                let t0 = 60000.0;
                let dec: f64 = 0.3;
                let dr = arcsec_to_rad(4.0) / dec.cos();

                let a = mk_alert(AlertId::new(0), 2.0, dec, t0, 1, 1000.0);
                let b = mk_alert(AlertId::new(1), 2.0 + dr, dec, t0 + 5.0/1440.0, 1, 1000.0);
                let left = SeedNode::from_pair(SeedId::new(0), NightId::new(1), &a, &b, None).unwrap();

                let mut right: Vec<SeedNode> = Vec::with_capacity(n);
                for i in 0..n {
                    let t = t0 + (i as f64) * (9.0/1440.0);
                    let ra_i = 2.0 + (i as f64)*dr;
                    let a_i = mk_alert(AlertId::new(800 + 2*i as u32), ra_i, dec, t, 1, 1000.0);
                    let b_i = mk_alert(AlertId::new(800 + 2*i as u32 + 1), ra_i + dr, dec, t + 1.0/1440.0, 1, 1000.0);
                    right.push(SeedNode::from_pair(SeedId::new(20_000 + i as u64), NightId::new(2), &a_i, &b_i, None).unwrap());
                }
                right.sort_by(|x,y| x.plane.epoch_mid.partial_cmp(&y.plane.epoch_mid).unwrap());

                let time_binner = UniformTimeBinner::new(t0, 45.0/1440.0);

                let out = left.score_edge_candidates(&right, &spatial_binner, &time_binner, &edge_config, delta);
                let brute = brute_force_scores(&left, &right, &edge_config, delta);

                let out_set: std::collections::HashSet<(SeedId, SeedId)> = out.iter().map(|e| (e.from, e.to)).collect();
                let brute_set: std::collections::HashSet<(SeedId, SeedId)> = brute.iter().map(|e| (e.from, e.to)).collect();

                prop_assert!(out_set.is_subset(&brute_set));
            }
        }
    }
}
