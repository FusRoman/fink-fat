//! Compact intra-night seed representation.
//!
//! A [`SeedNode`] stores the minimal information required to:
//! - persist intra-night “seeds” (pairs or triplets of detections),
//! - index them in spatio-temporal buckets (`SeedSpatialIndex`),
//! - and perform fast inter-night candidate retrieval for graph construction.
//!
//! The design is intentionally **data-centric**:
//! - geometric projection and kinematic prediction live in [`TangentPlaneModel`],
//! - query acceleration lives in [`SeedSpatialIndex`],
//! - scoring / ML ranking happen at higher levels (edges / features / models).
//!
//! This keeps `SeedNode` lightweight (cloneable, cache-friendly) and easy to
//! pass across threads.
//!
//! Data model overview
//! -------------------
//! A seed is built from either:
//! - a **pair** of alerts (linear motion on a tangent plane), or
//! - a **triplet** of alerts (quadratic motion, i.e. includes acceleration).
//!
//! The seed stores:
//! - its [`NightId`] (seeds do not mix nights),
//! - a local tangent-plane kinematic model ([`TangentPlaneModel`]),
//! - minimal photometric aggregates ([`Photometry`]),
//! - the ordered list of member detections (`members`), as `&Alert` references.
//!
//! Typical workflow
//! ----------------
//! 1. Build seeds for each night (`from_pair` / `from_triplet`).
//! 2. Build a [`SeedSpatialIndex`] for a “right-hand” night.
//! 3. For each left seed, call [`SeedNode::seed_edge_candidates`] to enumerate
//!    plausible right-hand neighbour seeds (coarse prediction + cone query).
//! 4. Compute exact features / scoring / ML ranking upstream.
//!
//! Notes on lifetimes
//! ------------------
//! `SeedNode<'alert_lf>` stores references to alerts (`&'alert_lf Alert`).
//! This avoids copying alert fields into the seed, but means the underlying
//! alerts must outlive the seeds (e.g. alerts owned by an `AlertStore`).
//!
//! Units & conventions
//! -------------------
//! - Angles (`ra`, `dec`, tangent-plane coordinates) are in **radians**.
//! - Epochs are **MJD TT** (`MJDTT`).
//! - Tangent-plane velocities are **rad/day**.
//!
//! See also
//! --------
//! - [`TangentPlaneModel`] – local kinematic model + prediction utilities.
//! - [`SeedSpatialIndex`] – spatio-temporal bucket index used for fast queries.
//! - [`EdgeFeatures::compute_features`](crate::graph::edge::edge_features::EdgeFeatures::compute_features) – exact feature extraction for edges.

pub mod error;
pub mod pairs;
pub mod photometry;
pub mod seed_spatial_index;
pub mod store;
pub mod tangent_plane;
pub mod triplets;

use std::{
    cmp::Ordering,
    fmt::{self, Display, Formatter},
};

use camino::Utf8PathBuf;
use serde::{Deserialize, Serialize};

use photom::{
    coordinates::{
        cov2::Cov2,
        equatorial::EquCoord,
        gnomonic_projection::{TangentPlane, TangentPoint, TangentVec},
    },
    observation_dataset::{observation::Observation, ObsDataset, ObsId},
    MJDTT,
};

use crate::{
    astro_math::fit_quad_1d,
    display_format::indent_block,
    engine_config::{edge_config::EdgeConfig, propagator_config::PredictorParams},
    night_id::NightId,
    persistence::{
        compression::Compression, envelope::DiskEnvelope, error::PersistenceIoError,
        layout::PersistenceLayout, manifest::Manifest, SEED_STORE_SCHEMA_VERSION,
    },
    seeding::{
        error::SeedingError,
        photometry::Photometry,
        seed_spatial_index::SeedSpatialIndex,
        store::{SeedId, SeedStore},
        tangent_plane::{Acceleration, PosWithCov, TangentPlaneModel, VelWithCov},
    },
    spacetime_bucket::spatial_binner::SpatialBinner,
};

#[derive(
    Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize, Default,
)]
pub struct SeedKey {
    pub night_id: NightId,
    pub unique_id: SeedId,
}

impl Display for SeedKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "SeedKey(nid={} sid={})", self.night_id, self.unique_id)
    }
}

/// Seed node with borrowed alert references.
/// This is the main struct used for seeding and graph construction.
///
/// The `core` field contains the cloneable seed data, while `members` holds references to the original alerts.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SeedNode {
    /// Seed identifier: night ID + unique global ID. This is used for persistence and indexing.
    key: SeedKey,

    /// Local tangent-plane kinematic model (position/velocity/(optional) acceleration).
    pub plane_model: TangentPlaneModel,

    /// Aggregated photometry for scoring / filtering.
    pub photom: Photometry,

    /// Number of detections used to form the seed (2 = pair, 3 = triplet).
    pub n_obs: u16,

    /// Member detections forming the seed, sorted by observation time.
    ///
    /// The ordering is meaningful: constructors keep members in time order and
    /// upstream logic may assume it for display/debugging.
    pub members: Vec<ObsId>,
}

impl PartialEq for SeedNode {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
            && self.plane_model == other.plane_model
            && self.photom == other.photom
            && self.n_obs == other.n_obs
            && self.members == other.members
    }
}
impl Eq for SeedNode {}

impl PartialOrd for SeedNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SeedNode {
    /// Order seeds primarily by epoch, with the seed key as a stable tie-breaker.
    ///
    /// Rationale
    /// ---------
    /// - Primary key: `plane_model.epoch_mid` — seeds are almost always consumed
    ///   in chronological order (edge building, time binning, etc.).
    /// - Tie-breaker: `key` (night_id, unique_id) — guarantees a **total**
    ///   order consistent with `Eq`, since `SeedKey` is unique by construction.
    fn cmp(&self, other: &Self) -> Ordering {
        self.plane_model
            .epoch_mid
            .total_cmp(&other.plane_model.epoch_mid)
            .then_with(|| self.key.cmp(&other.key))
    }
}

impl Display for SeedNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "SeedNode {{")?;

        writeln!(f, "  night     : {}", self.night_id())?;
        writeln!(f, "  n_obs     : {}", self.n_obs)?;
        writeln!(f)?;

        writeln!(
            f,
            "  plane     : {}",
            indent_block(&self.plane_model.to_string(), 14)
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "  photom   : {}",
            indent_block(&self.photom.to_string(), 14)
        )?;
        writeln!(f)?;

        writeln!(f, "]")?;

        writeln!(f, "}}")
    }
}

impl SeedNode {
    /// Return the stable seed identifier (night ID + unique global ID).
    ///
    /// Return
    /// ------
    /// SeedKey uniquely identifying this seed (used for persistence and indexing).
    pub fn key(&self) -> SeedKey {
        self.key
    }

    /// Return the night ID of this seed.
    ///
    /// Return
    /// ------
    /// NightId of this seed (same as `self.key.night_id`).
    pub fn night_id(&self) -> NightId {
        self.key.night_id
    }

    /// Overwrite the seed key.
    ///
    /// Arguments
    /// ---------
    /// * `key` – New key to assign.
    ///
    /// Notes
    /// -----
    /// Intended exclusively for re-keying seeds that were produced by a parallel
    /// worker using a temporary local [`crate::seeding::store::SeedStore`], before
    /// their final insertion into the pipeline seed store.
    pub(crate) fn set_key(&mut self, key: SeedKey) {
        self.key = key;
    }

    pub fn resolve_members<'obs>(
        &self,
        obs_dataset: &'obs ObsDataset,
    ) -> Result<Vec<&'obs Observation>, SeedingError> {
        self.members
            .iter()
            .map(|obs_idx| {
                obs_dataset
                    .get_observation(*obs_idx)
                    .ok_or_else(|| SeedingError::ObservationIndexNotFound(*obs_idx))
            })
            .collect()
    }

    /// Deterministically propagate this seed model by `dt` on its tangent plane.
    ///
    /// This is a low-level helper used by scoring and candidate search logic.
    /// It produces a **deterministic** kinematic prediction on the tangent plane
    /// (no noise model, no uncertainty inflation).
    ///
    /// Motion model
    /// ------------
    /// - If the seed has no acceleration term: constant velocity
    ///   `p(t) = p0 + v0 · dt`.
    /// - If the seed includes acceleration: constant acceleration
    ///   `p(t) = p0 + v0 · dt + 0.5 · a · dt²`,
    ///   `v(t) = v0 + a · dt`.
    ///
    /// Parameters
    /// ----------
    /// dt : f64
    ///     Time offset in **days**.
    /// dt_sq : f64
    ///     Precomputed `dt²` (micro-optimization for tight loops).
    ///
    /// Returns
    /// -------
    /// (TangentPoint, TangentVec, f64)
    ///     `(p_pred, v_pred, has_acc)` where:
    ///     - `p_pred` is the predicted tangent-plane position (radians),
    ///     - `v_pred` is the predicted tangent-plane velocity (rad/day),
    ///     - `has_acc` is `1.0` if acceleration is present, else `0.0`.
    ///
    /// Notes
    /// -----
    /// This assumes the seed tangent plane remains a valid local linearization
    /// over the time gap considered (typical for inter-night asteroid linking).
    #[inline]
    pub(crate) fn propagate_from(&self, dt: f64) -> (TangentPoint, TangentVec, f64) {
        let p = self.plane_model.predict_position(dt);
        let v = self.plane_model.predict_velocity(dt);
        let has_acc = if self.plane_model.acc.is_some() {
            1.0
        } else {
            0.0
        };
        (p, v, has_acc)
    }

    /// Predict the sky position `(ra, dec)` at `t_target` from the fitted model.
    ///
    /// This is a thin wrapper around [`TangentPlaneModel::predict_radec`].
    /// It returns the deterministic best-fit position (no uncertainty cone).
    ///
    /// Parameters
    /// ----------
    /// t_target : MJDTT
    ///     Target epoch (MJD TT).
    ///
    /// Returns
    /// -------
    /// EquCoord
    ///     Predicted sky position `(ra, dec)` in radians (same frame as alerts stored in the seed).
    ///
    /// See also
    /// --------
    /// - [`SeedNode::predict_cone`] – uncertainty-aware cone for candidate search.
    #[inline]
    pub fn predict_radec(&self, t_target: MJDTT) -> EquCoord {
        self.plane_model.predict_radec(t_target)
    }

    /// Predict a conservative sky cone `(ra, dec, radius)` for candidate search.
    ///
    /// This builds an uncertainty-aware search region at `t_target`:
    /// 1. The tangent-plane model predicts a base centre and radius using the
    ///    configured noise model and `k_sigma`.
    /// 2. Optionally, an extra padding of one spatial cell radius is added
    ///    (`pad_cell_radius`) so bucket-based queries do not miss neighbours on
    ///    cell boundaries.
    ///
    /// Parameters
    /// ----------
    /// t_target : MJDTT
    ///     Target epoch (MJD TT).
    /// binner : &impl SpatialBinner
    ///     Spatial binner used by the index; only `cell_radius()` is used here.
    /// predictor_params : &PredictorParams
    ///     Predictor configuration (noise model, `k_sigma`, and padding flags).
    ///
    /// Returns
    /// -------
    /// (EquCoord, f64)
    ///     `(center, radius)` where `center` is the predicted sky position in radians
    ///     and `radius` is the search cone radius in radians.
    ///
    /// Notes
    /// -----
    /// This function **does not** query any index; it only returns a geometric
    /// region. Use [`SeedNode::seed_edge_candidates`] or [`SeedNode::cone_candidates`]
    /// to actually retrieve neighbour seeds.
    #[inline]
    pub fn predict_cone<Bs: SpatialBinner + ?Sized>(
        &self,
        t_target: MJDTT,
        binner: &Bs,
        predictor_params: &PredictorParams,
    ) -> (EquCoord, f64) {
        let (center, mut radius) = self.plane_model.predict_cone_base(
            t_target,
            &predictor_params.noise,
            predictor_params.k_sigma,
        );

        if predictor_params.pad_cell_radius {
            radius += binner.cell_radius();
        }
        (center, radius)
    }

    /// Absolute time separation (days) between this seed and another seed.
    ///
    /// Parameters
    /// ----------
    /// other : &SeedNode
    ///     The other seed node.
    ///
    /// Returns
    /// -------
    /// f64
    ///     `|other.epoch_mid - self.epoch_mid|` in days.
    #[inline]
    pub fn delta_days(&self, other: &SeedNode) -> f64 {
        let t_self = self.plane_model.epoch_mid;
        let t_other = other.plane_model.epoch_mid;
        (t_other - t_self).abs()
    }

    /// Sky position of the seed at its reference epoch (`epoch_mid`).
    #[inline]
    pub fn predicted_center_equ(&self) -> EquCoord {
        self.plane_model.pos.tangent_point.unproject()
    }

    /// Enumerate candidate right-hand seeds for inter-night linking.
    ///
    /// This is the *coarse* candidate-generation stage used by the edge builder.
    /// It relies on the spatio-temporal preindexing provided by [`SeedSpatialIndex`]:
    /// - the right-hand seeds are partitioned into time bins,
    /// - each bin has an associated spatial bucket index,
    /// - this method predicts one cone per bin and queries the corresponding index.
    ///
    /// Compared to a naive “single global cone query”, the per-bin approach gives
    /// time-consistent candidate sets and allows conservative time padding without
    /// exploding the search radius.
    ///
    /// Parameters
    /// ----------
    /// right_seed_index : &SeedSpatialIndex
    ///     Pre-built spatio-temporal index for the right-hand night.
    ///     The index provides:
    ///     - `time_bins`: the list of bins to consider,
    ///     - `time_binner`: bin geometry (`bin_start`, `bin_end`, `bin_width`),
    ///     - `spatial_binner`: cell geometry (`cell_radius`),
    ///     - `cone_query(...)`: iterator over seeds inside the cone for that bin.
    /// edge_config : &EdgeConfig
    ///     Configuration controlling candidate search. This method uses the
    ///     predictor configuration (`edge_config.predictor_config`), including:
    ///     - noise model + `k_sigma` (cone inflation),
    ///     - `pad_cell_radius` (optional cell padding),
    ///     - `v_slack` (extra velocity slack, rad/day).
    ///
    /// Returns
    /// -------
    /// impl Iterator<Item = &SeedNode>
    ///     Iterator over candidate right-hand seeds. The iterator is lazy and
    ///     yields seeds across all time bins (flat-mapped).
    ///
    /// Notes
    /// -----
    /// - The cone radius is additionally inflated by a conservative time padding:
    ///   `(|v| + v_slack) * (bin_width / 2)`, where `|v|` is the seed speed on
    ///   the tangent plane (rad/day).
    /// - This stage is intentionally permissive: it returns many false positives
    ///   that must be filtered by exact scoring / ML ranking upstream.
    pub fn seed_edge_candidates<'iter, 'seed_lf>(
        &'iter self,
        right_seed_index: &'iter SeedSpatialIndex<'seed_lf, '_>,
        edge_config: &EdgeConfig,
    ) -> impl Iterator<Item = &'seed_lf SeedNode> + 'iter {
        let pred_cfg = edge_config.predictor_config;

        // Left seed speed on the tangent plane (rad/day), with optional slack.
        let speed = self.plane_model.vel.v.norm();
        let effective_speed = (speed + pred_cfg.v_slack).max(0.0);

        // Half-bin width used for conservative time padding.
        let half_bin_width_days = 0.5 * right_seed_index.time_binner.bin_width().max(1e-12);

        right_seed_index.time_bins.iter().flat_map(move |bin| {
            let bin_start = right_seed_index.time_binner.bin_start(bin.0);
            let bin_end = right_seed_index.time_binner.bin_end(bin.0);
            let bin_center = 0.5 * (bin_start + bin_end);

            // Base (k_sigma-inflated, no cell / time padding) and full query radius.
            let (center, base_r) =
                self.plane_model
                    .predict_cone_base(bin_center, &pred_cfg.noise, pred_cfg.k_sigma);

            let mut cone_radius = base_r;
            if pred_cfg.pad_cell_radius {
                cone_radius += right_seed_index.spatial_binner.cell_radius();
            }

            // Conservative padding: ensure the cone covers any epoch within the bin.
            if pred_cfg.pad_time_bin_radius {
                cone_radius += effective_speed * half_bin_width_days;
            }

            // Hard cap: clamp to max_cone_radius when set.
            // Seeds whose predicted uncertainty is very large (e.g. pairs over a long
            // gap) would otherwise generate enormous cones with many FP candidates.
            if let Some(max_r) = pred_cfg.max_cone_radius {
                cone_radius = cone_radius.min(max_r);
            }

            let max_norm = pred_cfg.max_norm_offset;

            // Collect per-bin so that `center` (a local) is not borrowed by
            // the returned iterator — required under Rust 2024 lifetime capture
            // rules.  The collected Vec holds &SeedNode references whose
            // lifetime ('seed_lf) is independent of `center`.
            let candidates: Vec<&'seed_lf SeedNode> = right_seed_index
                .cone_query(&center, cone_radius, bin_center)
                .filter(|to| {
                    // Normalised-offset cut: reject candidates whose angular
                    // separation from the predicted centre exceeds max_norm · base_r.
                    max_norm.is_none_or(|mn| {
                        let to_center = to.predicted_center_equ();
                        let sep = center.angular_separation(&to_center);
                        sep / base_r <= mn
                    })
                })
                .collect();
            candidates.into_iter()
        })
    }

    /// Query an index for candidates around the predicted cone at `t_target`.
    ///
    /// This is a convenience wrapper around [`SeedNode::predict_cone`] +
    /// [`SeedSpatialIndex::cone_query`]. It is best suited for one-off queries
    /// at a specific epoch.
    ///
    /// Parameters
    /// ----------
    /// t_target : MJDTT
    ///     Target epoch (MJD TT).
    /// index : &SeedSpatialIndex
    ///     Seed index to query.
    /// binner : &impl SpatialBinner
    ///     Spatial binner used for cone geometry.
    /// params : &PredictorParams
    ///     Predictor configuration (noise, `k_sigma`, padding).
    ///
    /// Returns
    /// -------
    /// Vec<&SeedNode>
    ///     Collected candidates returned by the index query.
    ///
    /// Notes
    /// -----
    /// `seed_edge_candidates` is usually preferred for the inter-night pipeline,
    /// because it aligns with the index time-bin structure and adds the
    /// conservative half-bin time padding.
    #[inline]
    pub fn cone_candidates<'seed_lf, Bs: SpatialBinner>(
        &self,
        t_target: MJDTT,
        index: &SeedSpatialIndex<'seed_lf, '_>,
        binner: &Bs,
        params: &PredictorParams,
    ) -> Vec<&'seed_lf SeedNode> {
        let (center, radius) = self.predict_cone(t_target, binner, params);
        index.cone_query(&center, radius, t_target).collect()
    }

    /// Build a [`SeedNode`] from a **pair** of alerts (linear tangent-plane model).
    ///
    /// This constructor:
    /// - defines a tangent-plane centre as the spherical midpoint of the two detections,
    /// - projects both detections onto the tangent plane,
    /// - fits a linear motion model (position at mid-epoch + velocity),
    /// - builds simple isotropic covariance estimates for position and velocity,
    /// - aggregates minimal photometry from the two fluxes.
    ///
    /// Arguments
    /// ---------
    /// * seed_store : &mut SeedStore
    ///   Seed store used to generate a unique seed key for this night.
    /// * night_id : NightId
    ///   Night identifier shared by the two alerts (seeds do not mix nights).
    /// * alert_a : &Alert
    ///   First detection.
    /// * alert_b : &Alert
    ///   Second detection.
    /// * max_speed_rad_per_day : `Option<f64>`
    ///   Optional physical sanity check on the fitted speed (rad/day).
    ///   If set and `||v|| > vmax`, the seed is rejected.
    ///
    /// Returns
    /// -------
    /// * `Option<SeedNode>`
    ///   `Some(seed)` if the model is built and passes the optional speed filter,
    ///   `None` if rejected by the speed filter.
    ///
    /// Notes
    /// -----
    /// - Members are stored in time order: `[alert_a, alert_b]` as passed here.
    ///   (Callers should pass them in chronological order if that matters.)
    /// - Covariances are approximated as isotropic using `max(ra_err, dec_err)`.
    /// - The model is meant as a cheap, robust intra-night approximation.
    pub fn from_pair(
        seed_store: &mut SeedStore,
        night_id: NightId,
        alert_a: &Observation,
        alert_b: &Observation,
        max_speed_rad_per_day: Option<f64>,
    ) -> Option<Self> {
        let ta = alert_a.mjd_tt();
        let tb = alert_b.mjd_tt();
        let epoch_mid = 0.5 * (ta + tb);
        let dt = tb - ta;
        let inv_dt = 1.0 / dt;
        let inv_dt2 = inv_dt * inv_dt;

        let mid = alert_a.equ_coord().spherical_midpoint(alert_b.equ_coord());
        let plane = TangentPlane::new(mid);

        // Project both alerts onto the tangent plane.
        let tp_a = plane.project(alert_a.equ_coord());
        let tp_b = plane.project(alert_b.equ_coord());

        // Midpoint on the plane.
        let tp_mid = TangentPoint::midpoint(tp_a, tp_b);

        // Velocity vector on the plane (rad / day).
        let v = (tp_b - tp_a) * inv_dt;

        if let Some(vmax) = max_speed_rad_per_day {
            if v.norm_sq() > vmax * vmax {
                return None;
            }
        }

        // Per-alert covariances on the sky (RA/Dec).
        // NB: we use them directly in the tangent plane here, under the
        // small-angle assumption that the gnomonic Jacobian is ≈ identity
        // near the projection centre. For a rigorous treatment, propagate
        // through the projection Jacobian J: Σ_xy = J Σ_αδ Jᵀ.
        let sigma_a = Cov2::from_equ(&alert_a.equ_coord());
        let sigma_b = Cov2::from_equ(&alert_b.equ_coord());
        let sum_ab = sigma_a + sigma_b;

        // Two-point least-squares fit at the midpoint epoch:
        //   Var(p̄) = (Σa + Σb) / 4
        //   Var(v̂) = (Σa + Σb) / Δt²
        let cov_pos = sum_ab * 0.25;
        let cov_vel = sum_ab * inv_dt2;

        let model = TangentPlaneModel {
            epoch_mid,
            pos: PosWithCov {
                tangent_point: tp_mid,
                cov: cov_pos,
            },
            vel: VelWithCov { v, cov: cov_vel },
            acc: None,
        };

        let mag_mean = (alert_a.photometry().magnitude + alert_b.photometry().magnitude) * 0.5;
        let flux_std = ((alert_a.photometry().magnitude - mag_mean).abs()
            + (alert_b.photometry().magnitude - mag_mean).abs())
            * 0.5;
        let photom = Photometry::from_pair(
            mag_mean as f32,
            flux_std as f32,
            &alert_a.photometry().filter,
            &alert_b.photometry().filter,
        );

        Some(SeedNode {
            key: seed_store.next_key(night_id),
            plane_model: model,
            photom,
            n_obs: 2,
            members: vec![alert_a.id().clone(), alert_b.id().clone()],
        })
    }

    /// Build a [`SeedNode`] from a **triplet** of alerts (quadratic tangent-plane model).
    ///
    /// This constructor fits a quadratic model independently in tangent `x` and `y`:
    /// it yields position at mean epoch, velocity, and acceleration.
    ///
    /// Arguments
    /// ---------
    /// * seed_store : &mut SeedStore
    ///   Seed store used to generate a unique seed key for this night.
    /// * night_id : NightId
    ///   Night identifier shared by the three alerts (seeds do not mix nights).
    /// * alert_a : &Alert
    ///   First detection.
    /// * alert_b : &Alert
    ///   Second detection.
    /// * alert_c : &Alert
    ///   Third detection.
    ///
    /// Returns
    /// -------
    /// * SeedNode
    ///   A seed with `n_obs == 3` and `plane_model.acc.is_some() == true`.
    ///
    /// Notes
    /// -----
    /// - The tangent-plane centre is chosen as the spherical midpoint of endpoints
    ///   `(a, c)` to stabilize projection.
    /// - Uncertainty estimates are coarse and isotropic (similar philosophy as pairs).
    /// - This is still an approximation of true orbital motion.
    pub fn from_triplet(
        seed_store: &mut SeedStore,
        night_id: NightId,
        alert_a: &Observation,
        alert_b: &Observation,
        alert_c: &Observation,
    ) -> Self {
        let (ta, tb, tc) = (alert_a.mjd_tt(), alert_b.mjd_tt(), alert_c.mjd_tt());
        let dt_char = (tc - ta).max(1e-6);
        let inv_dt2 = 1.0 / (dt_char * dt_char);
        let epoch_mid = (ta + tb + tc) / 3.0;

        // Tangent plane centred on the spherical midpoint of the extremes.
        let mid = alert_a.equ_coord().spherical_midpoint(alert_c.equ_coord());
        let plane = TangentPlane::new(mid);

        // Project the three alerts onto the plane.
        let tp_a = plane.project(alert_a.equ_coord());
        let tp_b = plane.project(alert_b.equ_coord());
        let tp_c = plane.project(alert_c.equ_coord());

        let dts = [ta - epoch_mid, tb - epoch_mid, tc - epoch_mid];
        let (tp_fit, v, a) = fit_quad_tangent(dts, [tp_a, tp_b, tp_c]);

        // Per-alert anisotropic variances (RA, Dec independent).
        let var_of = |o: &Observation| {
            let e = o.equ_coord();
            (e.ra_error * e.ra_error, e.dec_error * e.dec_error)
        };
        let (vxa, vya) = var_of(alert_a);
        let (vxb, vyb) = var_of(alert_b);
        let (vxc, vyc) = var_of(alert_c);

        // Mean variance per tangent-plane axis.
        let var_x = (vxa + vxb + vxc) / 3.0;
        let var_y = (vya + vyb + vyc) / 3.0;

        // Propagated covariances: position averages over 3 samples,
        // velocity scales as σ² / Δt².
        let cov_pos = Cov2::diag(var_x / 3.0, var_y / 3.0);
        let cov_vel = Cov2::diag(var_x * inv_dt2, var_y * inv_dt2);

        let model = TangentPlaneModel {
            epoch_mid,
            pos: PosWithCov {
                tangent_point: tp_fit,
                cov: cov_pos,
            },
            vel: VelWithCov { v, cov: cov_vel },
            acc: Some(Acceleration(a)),
        };

        let flux_mean = (alert_a.photometry().magnitude
            + alert_b.photometry().magnitude
            + alert_c.photometry().magnitude)
            / 3.0;
        let flux_std = ((alert_a.photometry().magnitude - flux_mean).abs()
            + (alert_b.photometry().magnitude - flux_mean).abs()
            + (alert_c.photometry().magnitude - flux_mean).abs())
            / 3.0;

        let photom = Photometry::from_triplet(
            flux_mean as f32,
            flux_std as f32,
            &alert_a.photometry().filter,
            &alert_b.photometry().filter,
            &alert_c.photometry().filter,
        );

        SeedNode {
            key: seed_store.next_key(night_id),
            plane_model: model,
            photom,
            n_obs: 3,
            members: vec![
                alert_a.id().clone(),
                alert_b.id().clone(),
                alert_c.id().clone(),
            ],
        }
    }
}

/// Fit a quadratic motion in a tangent plane through three projected points.
///
/// Returns `(p0, v, a)` where:
/// - `p0` is a [`TangentPoint`] on the same plane as the inputs,
/// - `v`, `a` are [`TangentVec`] displacements per unit time / time².
fn fit_quad_tangent(
    dt: [f64; 3],
    pts: [TangentPoint; 3],
) -> (TangentPoint, TangentVec, TangentVec) {
    debug_assert_eq!(pts[0].plane, pts[1].plane);
    debug_assert_eq!(pts[1].plane, pts[2].plane);

    let (p0x, vx, ax) = fit_quad_1d(dt, [pts[0].x, pts[1].x, pts[2].x]);
    let (p0y, vy, ay) = fit_quad_1d(dt, [pts[0].y, pts[1].y, pts[2].y]);

    (
        TangentPoint::new(pts[0].plane, p0x, p0y),
        TangentVec { dx: vx, dy: vy },
        TangentVec { dx: ax, dy: ay },
    )
}

pub trait SeedNodeSlice {
    fn save_seeds_night(
        &self,
        layout: &PersistenceLayout,
        manifest: &Manifest,
        night_id: NightId,
        compression: Compression,
    ) -> Result<Utf8PathBuf, PersistenceIoError>;
}

impl SeedNodeSlice for &[SeedNode] {
    fn save_seeds_night(
        &self,
        layout: &PersistenceLayout,
        manifest: &Manifest,
        night_id: NightId,
        compression: Compression,
    ) -> Result<Utf8PathBuf, PersistenceIoError> {
        let abs_path = layout.seeds_night_path(night_id);

        // Write payload (enveloped).
        let env = DiskEnvelope::new(
            self.to_vec(),
            SEED_STORE_SCHEMA_VERSION,
            manifest.created_unix_s,
            compression,
        );
        env.save_enveloped(&abs_path)?;
        Ok(abs_path)
    }
}

#[cfg(test)]
mod seed_node_tests {
    use super::*;
    use proptest::prelude::*;

    use crate::{
        astro_math::arcsec_to_rad,
        engine_config::propagator_config::{ModelNoise, PredictorParams},
        spacetime_bucket::{healpix_binner::HealpixBinner, uniform_time_binner::UniformTimeBinner},
    };

    use photom::{
        coordinates::equatorial::EquCoord,
        observation_dataset::observation::Observation,
        photometry::{Filter, Photometry as PhotomPhotometry},
    };

    const LAT_EPS: f64 = 1e-6;

    /* ------------------------- helpers ------------------------- */

    fn mk_alert(
        source_id: u64,
        ra: f64,
        dec: f64,
        mjd_tt: f64,
        band: u8,
        flux: f64,
    ) -> Observation {
        let pos_err = arcsec_to_rad(0.5);
        let equ_coord = EquCoord::new(ra, pos_err, dec, pos_err);
        let photometry = PhotomPhotometry {
            magnitude: flux,
            error: 0.0,
            filter: Filter::Int(band as u32),
        };
        Observation::new(source_id, equ_coord, photometry, mjd_tt, None)
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
            pad_time_bin_radius: true,
            time_bin_dt: 1.0,
            v_slack: 0.0,
            max_cone_radius: None,
            max_norm_offset: None,
        }
    }

    /* ------------------------- unit tests ------------------------- */

    #[test]
    fn from_pair_builds_expected_members_and_nobs() {
        let t0 = 60000.0;
        let dec: f64 = 0.25;
        let dr = arcsec_to_rad(6.0) / dec.cos();

        // IMPORTANT: store alerts in a vec so their references live long enough.
        let alerts = vec![
            mk_alert(0, 1.0, dec, t0, 1, 1000.0),
            mk_alert(1, 1.0 + dr, dec, t0 + 10.0 / 1440.0, 1, 1002.0),
        ];
        let (a, b) = (&alerts[0], &alerts[1]);

        let mut seed_store = SeedStore::new();

        let sn = SeedNode::from_pair(&mut seed_store, NightId::new(42), a, b, None)
            .expect("pair should produce a seed");

        assert_eq!(sn.night_id(), NightId::new(42));
        assert_eq!(sn.n_obs, 2);

        // members are stored as ObsId (u64)
        assert_eq!(sn.members.len(), 2);
        assert_eq!(sn.members[0], *a.id());
        assert_eq!(sn.members[1], *b.id());

        // Velocity is roughly dr / dt on the tangent plane.
        let dt = (b.mjd_tt() - a.mjd_tt()).max(1e-12);
        let mid = a.equ_coord().spherical_midpoint(b.equ_coord());
        let plane = TangentPlane::new(mid);
        let pa = plane.project(a.equ_coord());
        let pb = plane.project(b.equ_coord());
        let vx = (pb.x - pa.x) / dt;
        let vy = (pb.y - pa.y) / dt;

        assert!((sn.plane_model.vel.v.dx - vx).abs() < 1e-9);
        assert!((sn.plane_model.vel.v.dy - vy).abs() < 1e-9);
    }

    #[test]
    fn from_pair_speed_filter_rejects_fast_pairs() {
        let t0 = 60000.0;
        let dec: f64 = 0.2;
        let slow_sep = arcsec_to_rad(5.0) / dec.cos();
        let fast_sep = arcsec_to_rad(200.0) / dec.cos();

        let alerts = vec![
            mk_alert(0, 2.0, dec, t0, 1, 1000.0),
            mk_alert(1, 2.0 + slow_sep, dec, t0 + 5.0 / 1440.0, 1, 1000.0),
            mk_alert(2, 2.0 + fast_sep, dec, t0 + 5.0 / 1440.0, 1, 1000.0),
        ];

        let mut seed_store = SeedStore::new();

        let a = &alerts[0];
        let b_slow = &alerts[1];
        let b_fast = &alerts[2];

        let dt = 5.0 / 1440.0;
        let speed_slow = slow_sep / dt;
        let speed_fast = fast_sep / dt;
        assert!(speed_fast > speed_slow);

        let vmax = (speed_slow + speed_fast) * 0.5;

        let keep = SeedNode::from_pair(&mut seed_store, NightId::new(1), a, b_slow, Some(vmax));
        let drop = SeedNode::from_pair(&mut seed_store, NightId::new(1), a, b_fast, Some(vmax));

        assert!(keep.is_some());
        assert!(drop.is_none());
    }

    #[test]
    fn from_triplet_builds_expected_members_and_nobs() {
        let t0 = 60000.0;
        let dec: f64 = 0.3;
        let dr = arcsec_to_rad(6.0) / dec.cos();

        let alerts = vec![
            mk_alert(0, 1.0, dec, t0, 1, 1000.0),
            mk_alert(1, 1.0 + dr, dec, t0 + 10.0 / 1440.0, 1, 1001.0),
            mk_alert(2, 1.0 + 2.0 * dr, dec, t0 + 20.0 / 1440.0, 1, 1002.0),
        ];
        let (a, b, c) = (&alerts[0], &alerts[1], &alerts[2]);

        let sn = SeedNode::from_triplet(&mut SeedStore::new(), NightId::new(99), a, b, c);

        assert_eq!(sn.night_id(), NightId::new(99));
        assert_eq!(sn.n_obs, 3);

        assert_eq!(sn.members.len(), 3);
        assert_eq!(sn.members[0], *a.id());
        assert_eq!(sn.members[1], *b.id());
        assert_eq!(sn.members[2], *c.id());

        // Midpoint time close to average.
        let tm = (a.mjd_tt() + b.mjd_tt() + c.mjd_tt()) / 3.0;
        assert!((sn.plane_model.epoch_mid - tm).abs() < 1e-12);
    }

    #[test]
    fn predict_radec_and_cone_are_consistent() {
        let t0 = 60000.0;
        let dec: f64 = 0.25;
        let dr = arcsec_to_rad(8.0) / dec.cos();

        let alerts = vec![
            mk_alert(0, 2.0, dec, t0, 1, 1000.0),
            mk_alert(1, 2.0 + dr, dec, t0 + 10.0 / 1440.0, 1, 1000.0),
        ];
        let (a, b) = (&alerts[0], &alerts[1]);

        let sn = SeedNode::from_pair(&mut SeedStore::new(), NightId::new(1), a, b, None).unwrap();

        let predict_params = default_predictor_params();
        let tb = b.mjd_tt();

        let pred = sn.predict_radec(tb);
        let (center, radius) = sn.predict_cone(tb, &HealpixBinner::new(8), &predict_params);

        let d = pred.angular_separation(&center);
        assert!(d <= radius + 1e-12);
    }

    #[test]
    fn seed_edge_candidates_basic_smoke() {
        // Goal: ensure method typechecks + returns something plausible.
        // We'll build 2 "right" seeds and query from 1 "left" seed.

        use crate::engine_config::edge_config::EdgeConfig;
        use crate::seeding::seed_spatial_index::SeedSpatialIndex;

        let spatial_binner = HealpixBinner::new(8);

        let t0 = 60010.0;
        let time_binner = UniformTimeBinner::new(t0, 5.0 / 1440.0);

        let dec: f64 = 0.3;
        let dr = arcsec_to_rad(6.0) / dec.cos();

        // alerts live in this vec
        let alerts = vec![
            mk_alert(0, 1.0, dec, t0, 1, 1000.0),
            mk_alert(1, 1.0 + dr, dec, t0 + 5.0 / 1440.0, 1, 1001.0),
            mk_alert(2, 1.0 + 2.0 * dr, dec, t0 + 10.0 / 1440.0, 1, 1002.0),
        ];

        let a = &alerts[0];
        let b = &alerts[1];
        let c = &alerts[2];

        let left =
            SeedNode::from_pair(&mut SeedStore::new(), NightId::new(10), a, b, None).unwrap();
        let right1 =
            SeedNode::from_pair(&mut SeedStore::new(), NightId::new(10), b, c, None).unwrap();
        let right2 =
            SeedNode::from_pair(&mut SeedStore::new(), NightId::new(10), a, c, None).unwrap();

        let rights = vec![right1, right2];

        // build index over right seeds
        let right_index = SeedSpatialIndex::build(&rights, &spatial_binner, &time_binner);

        // edge config (use whatever Default you have; otherwise construct minimal)
        let edge_cfg = EdgeConfig::default();

        let cand: Vec<&SeedNode> = left.seed_edge_candidates(&right_index, &edge_cfg).collect();

        // We don't assert exact count; just ensure no lifetime/borrow issue and deterministic content.
        assert!(cand.len() <= rights.len());
        for s in cand {
            assert_eq!(s.night_id(), NightId::new(10));
        }
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

        #[test]
        fn prop_from_pair_basic_invariants(
            samples in proptest::collection::vec((ra_strategy(), dec_strategy(), t_strategy()), 2..60)
        ) {
            let mut alerts: Vec<Observation> = samples.iter().enumerate().map(|(i, (ra, dec, t))| {
                mk_alert(i as u64, *ra, *dec, *t, 1, 1000.0)
            }).collect();

            alerts.sort_by(|a,b| a.mjd_tt().partial_cmp(&b.mjd_tt()).unwrap());

            let mut count = 0usize;
            for i in 0..alerts.len().saturating_sub(1) {
                let a = &alerts[i];
                let b = &alerts[i+1];
                if b.mjd_tt() <= a.mjd_tt() { continue; }

                if let Some(sn) = SeedNode::from_pair(
                    &mut SeedStore::new(),
                    NightId::new(1),
                    a,
                    b,
                    None,
                ) {
                    count += 1;
                    prop_assert_eq!(sn.n_obs, 2);
                    prop_assert_eq!(sn.members.len(), 2);
                    prop_assert_eq!(sn.members[0], *a.id());
                    prop_assert_eq!(sn.members[1], *b.id());

                    let tm = 0.5 * (a.mjd_tt() + b.mjd_tt());
                    prop_assert!((sn.plane_model.epoch_mid - tm).abs() < 1e-9);

                    prop_assert!(sn.plane_model.vel.v.dx.is_finite() && sn.plane_model.vel.v.dy.is_finite());
                }
            }
            prop_assert!(count > 0);
        }

        #[test]
        fn prop_from_triplet_basic_invariants(
            samples in proptest::collection::vec((ra_strategy(), dec_strategy(), t_strategy()), 3..60)
        ) {
            let mut alerts: Vec<Observation> = samples.iter().enumerate().map(|(i, (ra, dec, t))| {
                mk_alert(i as u64, *ra, *dec, *t, 1, 1000.0)
            }).collect();

            alerts.sort_by(|a,b| a.mjd_tt().partial_cmp(&b.mjd_tt()).unwrap());

            let mut built = 0usize;
            for i in 0..alerts.len().saturating_sub(2) {
                let (a, b, c) = (&alerts[i], &alerts[i+1], &alerts[i+2]);
                if !(a.mjd_tt() < b.mjd_tt() && b.mjd_tt() < c.mjd_tt()) { continue; }

                let sn = SeedNode::from_triplet(
                    &mut SeedStore::new(),
                    NightId::new(1),
                    a,
                    b,
                    c,
                );
                built += 1;

                prop_assert_eq!(sn.n_obs, 3);
                prop_assert_eq!(sn.members.len(), 3);
                prop_assert_eq!(sn.members[0], *a.id());
                prop_assert_eq!(sn.members[1], *b.id());
                prop_assert_eq!(sn.members[2], *c.id());

                let tmin = a.mjd_tt().min(b.mjd_tt()).min(c.mjd_tt());
                let tmax = a.mjd_tt().max(b.mjd_tt()).max(c.mjd_tt());
                prop_assert!(sn.plane_model.epoch_mid >= tmin && sn.plane_model.epoch_mid <= tmax);

                prop_assert!(sn.plane_model.vel.v.dx.is_finite() && sn.plane_model.vel.v.dy.is_finite());
                let acc = sn.plane_model.acc.expect("triplet fits a quadratic");
                prop_assert!(acc.0.dx.is_finite() && acc.0.dy.is_finite());
            }
            prop_assert!(built > 0);
        }

        #[test]
        fn prop_predict_cone_covers_predict_radec(
            samples in proptest::collection::vec((ra_strategy(), dec_strategy(), t_strategy()), 2..40)
        ) {
            let alerts: Vec<Observation> = samples.iter().enumerate().map(|(i, (ra, dec, t))| {
                mk_alert(i as u64, *ra, *dec, *t, 1, 1000.0)
            }).collect();

            if alerts.len() < 2 { return Ok(()); }

            let a = &alerts[0];
            let b = &alerts[1];
            if b.mjd_tt() <= a.mjd_tt() { return Ok(()); }

            let sn = match SeedNode::from_pair(
                &mut SeedStore::new(),
                NightId::new(3),
                a,
                b,
                None,
            ) {
                Some(s) => s,
                None => return Ok(()),
            };

            let params = default_predictor_params();
            let t = b.mjd_tt();

            let pred = sn.predict_radec(t);
            let (center, rad) = sn.predict_cone(t, &HealpixBinner::new(8), &params);

            let d = pred.angular_separation(&center);
            prop_assert!(d <= rad + 1e-12);
        }
    }
}
