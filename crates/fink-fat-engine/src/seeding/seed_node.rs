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

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

use crate::{
    Alert, AlertId, MjdTt, Radians,
    alerts::AlertStore,
    astro_math::{fit_quad_1d, radec_to_tangent, spherical_midpoint, tangent_to_radec},
    engine_config::propagator_config::PredictorParams,
    night_id::NightId,
    seeding::{
        photometry::Photometry,
        seed_id::SeedId,
        seed_spatial_index::SeedSpatialIndex,
        tangent_plane::{TangentCenter, TangentPlaneModel},
    },
    spacetime_bucket::spatial_binner::SpatialBinner,
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
#[derive(Clone, Debug, Serialize, Deserialize, Encode, Decode)]
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
    pub fn cone_candidates<Bs: SpatialBinner>(
        &self,
        t_target: MjdTt,
        index: &SeedSpatialIndex,
        binner: &Bs,
        params: &PredictorParams,
    ) -> Vec<SeedId> {
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
        let flux_std =
            ((alert_a.flux - flux_mean).abs() + (alert_b.flux - flux_mean).abs()) * 0.5;
        let photom = Photometry::new(flux_mean, flux_std, alert_a.band);

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
        let flux_std =
            ((alert_a.flux - flux_mean).abs() + (alert_b.flux - flux_mean).abs() + (alert_c.flux - flux_mean).abs())
                / 3.0;
        let photom = Photometry::new(flux_mean, flux_std, alert_a.band);

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

        // Build a spatial index and bucket index (for completeness).
        let index = SeedSpatialIndex::build(&[s1.clone(), s2.clone()], &spatial_binner);
        let _bucket_index = build_bucket_index(
            &vec![a.clone(), b.clone(), c.clone()],
            &spatial_binner,
            &time_binner,
        );

        // Query around s1 prediction near time of c, expect to find s2 (future position).
        let (ra, dec, radius) = s1.predict_cone(c.mjd_tt, &spatial_binner, &params);
        let candidates: Vec<SeedId> = index.cone_query(&spatial_binner, ra, dec, radius).collect();

        assert!(candidates.contains(&s2.seed_id));
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
}
