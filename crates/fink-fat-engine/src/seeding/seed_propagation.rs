use photom::{
    MJDTT,
    coordinates::{
        equatorial::EquCoord,
        gnomonic_projection::{TangentPoint, TangentVec},
    },
};

use crate::{
    engine_config::{edge_config::EdgeConfig, propagator_config::PredictorParams},
    seeding::{SeedNode, seed_spatial_index::SeedSpatialIndex},
    spacetime_bucket::spatial_binner::SpatialBinner,
};

impl SeedNode {
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

    /// Project this seed's central sky position onto another seed's tangent plane.
    ///
    /// The receiver (`self`) contributes only its cached central sky position
    /// (via [`SeedNode::predicted_center_equ`]); the `frame` seed provides the
    /// tangent-plane reference into which that position is projected.
    ///
    /// The underlying [`TangentPlane::project`] uses precomputed trigonometric
    /// values, so no rotation/projection matrices are rebuilt per call.
    ///
    /// Arguments
    /// ---------
    /// * `frame` – Seed whose tangent plane defines the target reference frame.
    ///
    /// Return
    /// ------
    /// A [`TangentPoint`] expressed in the tangent frame of `frame`.
    ///
    /// Notes
    /// -----
    /// - Purely geometric operation; no uncertainty propagation is performed.
    /// - Typical usage projects a *target* seed onto a *source* seed's frame:
    ///   ```ignore
    ///   let tp = target.project_onto(source);
    ///   ```
    #[inline]
    pub fn project_onto(&self, frame: &SeedNode) -> TangentPoint {
        frame
            .plane_model
            .pos
            .tangent_point
            .plane
            .project(&self.tangent_seed_center())
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
                        let to_center = to.tangent_seed_center();
                        let sep = center.angular_separation(&to_center);
                        sep / base_r <= mn
                    })
                })
                .collect();
            candidates.into_iter()
        })
    }
}

#[cfg(test)]
mod seed_node_propagation_tests {
    use proptest::prelude::*;

    use photom::{
        NightId,
        coordinates::equatorial::EquCoord,
        observation_dataset::observation::Observation,
        photometry::{Filter, Photometry},
    };
    use proptest::prelude::Strategy;

    use crate::{
        astro_math::arcsec_to_rad,
        engine_config::propagator_config::{ModelNoise, PredictorParams},
        seeding::{SeedNode, store::SeedStore},
        spacetime_bucket::{healpix_binner::HealpixBinner, uniform_time_binner::UniformTimeBinner},
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
        let photometry = Photometry {
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
