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
pub mod seed_propagation;
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
    NightId,
    coordinates::equatorial::EquCoord,
    observation_dataset::{ObsDataset, ObsId, observation::Observation},
};

use crate::{
    display_format::indent_block,
    persistence::{
        SEED_STORE_SCHEMA_VERSION, compression::Compression, envelope::DiskEnvelope,
        error::PersistenceIoError, layout::PersistenceLayout, manifest::Manifest,
    },
    seeding::{
        error::SeedingError, photometry::SeedPhotometry, store::SeedId,
        tangent_plane::TangentPlaneModel,
    },
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
    pub photom: SeedPhotometry,

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
    pub fn tangent_seed_center(&self) -> EquCoord {
        self.plane_model.pos.tangent_point.unproject()
    }
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
        seeding::store::SeedStore,
    };

    use photom::{
        coordinates::{equatorial::EquCoord, gnomonic_projection::TangentPlane},
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
    }
}
