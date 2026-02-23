use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray, UInt64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use outfit::{FullOrbitResult, ObjectNumber};

use crate::{
    alerts::store::AlertStore,
    graph::AlertLinkageDAG,
    night_id::{NightId, PairingMode},
    persistence::{
        envelope::save_parquet, error::PersistenceIoError, layout::PersistenceLayout,
        manifest::Manifest,
    },
    seeding::store::SeedStore,
    solver::{HypothesisId, HypothesisSet},
    trajectory::track_id::TrackId,
};

/// Loaded runtime state built from persisted artifacts.
///
/// This matches the runtime needs:
/// - `AlertStore` owns the alert vectors (per night).
/// - `SeedStoreOwned` owns `SeedNodeOwned`.
/// - `InterNightGraph<'seed,'alert>` owns `Edge<'seed,'alert>` that borrow seeds.
///
/// Notes
/// -----
/// This struct is meant to be created and then moved into your runtime engine
/// (or into a `FinkFat` instance).
pub struct RuntimeState {
    pub manifest: Manifest,
    pub window: Option<PairingMode>,
    pub alert_store: AlertStore,
    pub seed_store: SeedStore,
    pub graph: AlertLinkageDAG,
    pub track_hypotheses: HypothesisSet,
    pub orbit_results: FullOrbitResult,
}

// =============================================================================
// Parquet export helpers
// =============================================================================

/// Arrow schema for the track-members Parquet file.
///
/// Columns
/// -------
/// - `track_id`       : Utf8   — deterministic trajectory identifier (`TRK…`).
/// - `dia_source_id`  : UInt64 — unique alert identifier within the survey.
fn track_members_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("track_id", DataType::Utf8, false),
        Field::new("dia_source_id", DataType::UInt64, false),
    ]))
}

/// Arrow schema for the orbital-parameters Parquet file.
///
/// Columns
/// -------
/// - `track_id`                    : Utf8    — trajectory identifier (join key).
/// - `orbit_type`                  : Utf8    — `"Preliminary"` or `"Corrected"`.
/// - `reference_epoch`             : Float64 — MJD reference epoch of the orbit.
/// - `semi_major_axis`             : Float64 — semi-major axis (AU).
/// - `eccentricity`                : Float64 — orbital eccentricity.
/// - `inclination`                 : Float64 — inclination (radians).
/// - `ascending_node_longitude`    : Float64 — longitude of ascending node Ω (radians).
/// - `periapsis_argument`          : Float64 — argument of periapsis ω (radians).
/// - `mean_anomaly`                : Float64 — mean anomaly M (radians).
/// - `rms`                         : Float64 — RMS of normalized astrometric residuals.
fn orbital_params_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("track_id", DataType::Utf8, false),
        Field::new("orbit_type", DataType::Utf8, false),
        Field::new("reference_epoch", DataType::Float64, false),
        Field::new("semi_major_axis", DataType::Float64, false),
        Field::new("eccentricity", DataType::Float64, false),
        Field::new("inclination", DataType::Float64, false),
        Field::new("ascending_node_longitude", DataType::Float64, false),
        Field::new("periapsis_argument", DataType::Float64, false),
        Field::new("mean_anomaly", DataType::Float64, false),
        Field::new("rms", DataType::Float64, false),
    ]))
}

impl RuntimeState {
    /// Export track members and orbital parameters as two Parquet files.
    ///
    /// This writes:
    /// 1. A **track-members** Parquet mapping each `track_id` to its constituent
    ///    `dia_source_id` values.
    /// 2. An **orbital-parameters** Parquet mapping each `track_id` to its
    ///    Keplerian orbital elements, reference epoch, RMS, and orbit type.
    ///
    /// Both files are written atomically to the `orbits/` directory under
    /// the persistence layout, partitioned by the given `night_id`.
    ///
    /// Arguments
    /// ---------
    /// * `layout`   - Persistence layout providing path conventions.
    /// * `night_id` - Night during which the orbits were fitted.
    ///
    /// Return
    /// ------
    /// `Ok(())` on success, otherwise a [`PersistenceIoError`].
    ///
    /// Notes
    /// -----
    /// - Hypotheses that fail track-ID computation or alert resolution are
    ///   silently skipped (logged via the return counters of the pipeline stage).
    /// - Orbit results that fail Keplerian conversion are skipped.
    pub fn export_orbit_parquets(
        &self,
        layout: &PersistenceLayout,
        night_id: NightId,
    ) -> Result<(), PersistenceIoError> {
        // Early return: nothing to export.
        if self.track_hypotheses.is_empty() {
            return Ok(());
        }

        // -----------------------------------------------------------------
        // 1) Build a stable HypothesisId → TrackId mapping and the
        //    track-members rows.
        // -----------------------------------------------------------------

        // Deterministic iteration order (sorted by hypothesis id).
        let mut sorted_ids: Vec<HypothesisId> = self.track_hypotheses.keys().copied().collect();
        sorted_ids.sort_unstable();

        // Accumulators for track-members Parquet.
        let mut tm_track_ids: Vec<String> = Vec::new();
        let mut tm_dia_source_ids: Vec<u64> = Vec::new();

        // Map HypothesisId → TrackId (String) for the orbital-params step.
        let mut hyp_to_track: Vec<(HypothesisId, TrackId)> = Vec::new();

        for &hyp_id in &sorted_ids {
            let trk = &self.track_hypotheses[&hyp_id];

            // Compute the deterministic track identifier.
            let track_id = match trk.track_id(&self.alert_store, &self.seed_store) {
                Ok(tid) => tid,
                Err(_) => continue, // skip hypothesis if track_id cannot be computed
            };

            // Resolve all alerts for this hypothesis.
            let alerts = match trk.get_alerts(&self.alert_store, &self.seed_store) {
                Ok(a) => a,
                Err(_) => continue,
            };

            for alert in &alerts {
                tm_track_ids.push(track_id.as_str().to_owned());
                tm_dia_source_ids.push(alert.key.dia_source_id);
            }

            hyp_to_track.push((hyp_id, track_id));
        }

        // Write track-members Parquet (may be empty if all hypotheses failed).
        if !tm_track_ids.is_empty() {
            let schema = track_members_schema();
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StringArray::from(tm_track_ids)) as ArrayRef,
                    Arc::new(UInt64Array::from(tm_dia_source_ids)) as ArrayRef,
                ],
            )
            .map_err(|e| PersistenceIoError::Arrow(e.to_string()))?;

            let path = layout.track_members_night_path(night_id);
            save_parquet(&path, &batch)?;
        }

        // -----------------------------------------------------------------
        // 2) Build orbital-parameters rows from orbit_results.
        // -----------------------------------------------------------------
        let mut op_track_ids: Vec<String> = Vec::new();
        let mut op_orbit_types: Vec<String> = Vec::new();
        let mut op_ref_epochs: Vec<f64> = Vec::new();
        let mut op_sma: Vec<f64> = Vec::new();
        let mut op_ecc: Vec<f64> = Vec::new();
        let mut op_inc: Vec<f64> = Vec::new();
        let mut op_raan: Vec<f64> = Vec::new();
        let mut op_aop: Vec<f64> = Vec::new();
        let mut op_ma: Vec<f64> = Vec::new();
        let mut op_rms: Vec<f64> = Vec::new();

        for (hyp_id, track_id) in &hyp_to_track {
            let obj_key = ObjectNumber::Int(*hyp_id);
            let Some(result) = self.orbit_results.get(&obj_key) else {
                continue;
            };
            let Ok((gauss_result, rms)) = result else {
                continue;
            };

            let orbit_type = if gauss_result.is_corrected() {
                "Corrected"
            } else {
                "Preliminary"
            };

            // Convert to Keplerian elements; skip if conversion fails.
            let kep = match gauss_result.get_orbit().to_keplerian() {
                Ok(k) => k,
                Err(_) => continue,
            };

            op_track_ids.push(track_id.as_str().to_owned());
            op_orbit_types.push(orbit_type.to_owned());
            op_ref_epochs.push(kep.reference_epoch);
            op_sma.push(kep.semi_major_axis);
            op_ecc.push(kep.eccentricity);
            op_inc.push(kep.inclination);
            op_raan.push(kep.ascending_node_longitude);
            op_aop.push(kep.periapsis_argument);
            op_ma.push(kep.mean_anomaly);
            op_rms.push(*rms);
        }

        if !op_track_ids.is_empty() {
            let schema = orbital_params_schema();
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StringArray::from(op_track_ids)) as ArrayRef,
                    Arc::new(StringArray::from(op_orbit_types)) as ArrayRef,
                    Arc::new(Float64Array::from(op_ref_epochs)) as ArrayRef,
                    Arc::new(Float64Array::from(op_sma)) as ArrayRef,
                    Arc::new(Float64Array::from(op_ecc)) as ArrayRef,
                    Arc::new(Float64Array::from(op_inc)) as ArrayRef,
                    Arc::new(Float64Array::from(op_raan)) as ArrayRef,
                    Arc::new(Float64Array::from(op_aop)) as ArrayRef,
                    Arc::new(Float64Array::from(op_ma)) as ArrayRef,
                    Arc::new(Float64Array::from(op_rms)) as ArrayRef,
                ],
            )
            .map_err(|e| PersistenceIoError::Arrow(e.to_string()))?;

            let path = layout.orbital_params_night_path(night_id);
            save_parquet(&path, &batch)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ahash::AHashMap;
    use arrow_array::{Array, Float64Array, StringArray, UInt64Array};
    use camino::Utf8PathBuf;
    use outfit::{
        GaussResult, ObjectNumber, OrbitalElements,
        orbit_type::keplerian_element::KeplerianElements,
    };
    use tempfile::tempdir;

    use crate::{
        Alert, AlertKey, AlertStore,
        graph::{AlertLinkageDAG, edge::EdgeKey},
        night_id::NightId,
        persistence::{
            envelope::load_parquet, layout::PersistenceLayout, manifest::Manifest,
            runtime_state::RuntimeState,
        },
        seeding::{SeedNode, store::SeedStore},
        solver::HypothesisSet,
        trajectory::TrackHypothesis,
    };

    // -----------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------

    fn utf8(p: std::path::PathBuf) -> Utf8PathBuf {
        Utf8PathBuf::from_path_buf(p).expect("temp paths should be valid UTF-8")
    }

    /// Build a minimal alert with the given night, id, and epoch.
    fn make_alert(night: u32, dia: u64, mjd: f64) -> Alert {
        Alert {
            key: AlertKey {
                night_id: NightId(night),
                dia_source_id: dia,
            },
            ra: 0.1,
            ra_err: 1e-6,
            dec: 0.2,
            dec_err: 1e-6,
            mjd_tt: mjd,
            flux: 100.0,
            flux_err: 1.0,
            band: 1,
            observer_mpc_code: Arc::new("W84".to_string()),
        }
    }

    /// Build a minimal seed pointing at the given alert keys.
    fn make_seed(members: Vec<AlertKey>) -> SeedNode {
        let mut s = SeedNode::default();
        s.n_obs = 2;
        s.members = members;
        s
    }

    /// Build a `RuntimeState` with two hypotheses:
    ///   - hyp 0: night 1 seed → night 2 seed  (alerts 10,11 → 20,21)
    ///   - hyp 1: night 1 seed → night 2 seed  (alerts 12,13 → 22,23)
    ///
    /// Also populates `orbit_results` for both hypotheses with
    /// simple Keplerian elements.
    fn make_test_state() -> RuntimeState {
        let nid1 = NightId(1);
        let nid2 = NightId(2);

        // -- Alerts -------------------------------------------------------
        let a10 = make_alert(1, 10, 60000.0);
        let a11 = make_alert(1, 11, 60000.1);
        let a12 = make_alert(1, 12, 60000.2);
        let a13 = make_alert(1, 13, 60000.3);
        let a20 = make_alert(2, 20, 60001.0);
        let a21 = make_alert(2, 21, 60001.1);
        let a22 = make_alert(2, 22, 60001.2);
        let a23 = make_alert(2, 23, 60001.3);

        let mut alert_store = AlertStore::new();
        alert_store.insert(
            nid1,
            vec![a10.clone(), a11.clone(), a12.clone(), a13.clone()],
        );
        alert_store.insert(nid2, vec![a20, a21, a22, a23]);

        // -- Seeds --------------------------------------------------------
        let mut seed_store = SeedStore::new();

        let sk_a = seed_store.insert_seed(nid1, make_seed(vec![a10.key, a11.key]));
        let sk_b = seed_store.insert_seed(
            nid2,
            make_seed(vec![
                AlertKey {
                    night_id: nid2,
                    dia_source_id: 20,
                },
                AlertKey {
                    night_id: nid2,
                    dia_source_id: 21,
                },
            ]),
        );
        let sk_c = seed_store.insert_seed(nid1, make_seed(vec![a12.key, a13.key]));
        let sk_d = seed_store.insert_seed(
            nid2,
            make_seed(vec![
                AlertKey {
                    night_id: nid2,
                    dia_source_id: 22,
                },
                AlertKey {
                    night_id: nid2,
                    dia_source_id: 23,
                },
            ]),
        );

        // -- Track hypotheses ---------------------------------------------
        let mut hypotheses: HypothesisSet = AHashMap::new();
        hypotheses.insert(
            0,
            TrackHypothesis {
                nodes: vec![sk_a, sk_b],
                edges: vec![EdgeKey {
                    from: sk_a,
                    to: sk_b,
                }],
                cost: 1.0,
                night_span: 1,
            },
        );
        hypotheses.insert(
            1,
            TrackHypothesis {
                nodes: vec![sk_c, sk_d],
                edges: vec![EdgeKey {
                    from: sk_c,
                    to: sk_d,
                }],
                cost: 2.0,
                night_span: 1,
            },
        );

        // -- Orbit results ------------------------------------------------
        let kep0 = KeplerianElements {
            reference_epoch: 60000.0,
            semi_major_axis: 2.5,
            eccentricity: 0.08,
            inclination: 0.15,
            ascending_node_longitude: 1.2,
            periapsis_argument: 0.9,
            mean_anomaly: 0.3,
        };
        let kep1 = KeplerianElements {
            reference_epoch: 60000.0,
            semi_major_axis: 1.0,
            eccentricity: 0.017,
            inclination: 0.41,
            ascending_node_longitude: 0.5,
            periapsis_argument: 1.8,
            mean_anomaly: 2.1,
        };

        let mut orbit_results = outfit::FullOrbitResult::default();
        orbit_results.insert(
            ObjectNumber::Int(0),
            Ok((
                GaussResult::PrelimOrbit(OrbitalElements::Keplerian(kep0)),
                0.05,
            )),
        );
        orbit_results.insert(
            ObjectNumber::Int(1),
            Ok((
                GaussResult::CorrectedOrbit(OrbitalElements::Keplerian(kep1)),
                0.02,
            )),
        );

        RuntimeState {
            manifest: Manifest::new(0),
            window: None,
            alert_store,
            seed_store,
            graph: AlertLinkageDAG::new(),
            track_hypotheses: hypotheses,
            orbit_results,
        }
    }

    // -----------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------

    #[test]
    fn export_orbit_parquets_noop_on_empty_hypotheses() {
        let dir = tempdir().unwrap();
        let layout = PersistenceLayout::new(utf8(dir.path().to_path_buf()));

        let state = RuntimeState {
            manifest: Manifest::new(0),
            window: None,
            alert_store: AlertStore::new(),
            seed_store: SeedStore::new(),
            graph: AlertLinkageDAG::new(),
            track_hypotheses: HypothesisSet::new(),
            orbit_results: outfit::FullOrbitResult::default(),
        };

        state
            .export_orbit_parquets(&layout, NightId(1))
            .expect("should succeed on empty hypotheses");

        // No files should have been created.
        assert!(!layout.track_members_night_path(NightId(1)).exists());
        assert!(!layout.orbital_params_night_path(NightId(1)).exists());
    }

    #[test]
    fn export_orbit_parquets_creates_track_members_file() {
        let dir = tempdir().unwrap();
        let layout = PersistenceLayout::new(utf8(dir.path().to_path_buf()));
        let state = make_test_state();

        state
            .export_orbit_parquets(&layout, NightId(2))
            .expect("export should succeed");

        let path = layout.track_members_night_path(NightId(2));
        assert!(path.exists(), "track members parquet must be created");
    }

    #[test]
    fn export_orbit_parquets_creates_orbital_params_file() {
        let dir = tempdir().unwrap();
        let layout = PersistenceLayout::new(utf8(dir.path().to_path_buf()));
        let state = make_test_state();

        state
            .export_orbit_parquets(&layout, NightId(2))
            .expect("export should succeed");

        let path = layout.orbital_params_night_path(NightId(2));
        assert!(path.exists(), "orbital params parquet must be created");
    }

    #[test]
    fn track_members_parquet_has_correct_schema_and_rows() {
        let dir = tempdir().unwrap();
        let layout = PersistenceLayout::new(utf8(dir.path().to_path_buf()));
        let state = make_test_state();

        state.export_orbit_parquets(&layout, NightId(2)).unwrap();

        let path = layout.track_members_night_path(NightId(2));
        let (schema, batches) = load_parquet(&path).unwrap();

        // Schema: track_id (Utf8), dia_source_id (UInt64)
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(schema.field(0).name(), "track_id");
        assert_eq!(schema.field(1).name(), "dia_source_id");

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        // 2 hypotheses × (2 alerts per seed × 2 seeds per track) = 8 rows
        assert_eq!(total_rows, 8, "expected 8 rows (2 tracks × 4 alerts each)");

        // All track_id values should start with "TRK"
        for batch in &batches {
            let track_ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            for i in 0..track_ids.len() {
                assert!(
                    track_ids.value(i).starts_with("TRK"),
                    "track_id should start with TRK, got: {}",
                    track_ids.value(i)
                );
            }
        }

        // All dia_source_id values should be from our test data.
        let expected_dias: std::collections::HashSet<u64> =
            [10, 11, 12, 13, 20, 21, 22, 23].into_iter().collect();
        for batch in &batches {
            let dias = batch
                .column(1)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            for i in 0..dias.len() {
                assert!(
                    expected_dias.contains(&dias.value(i)),
                    "unexpected dia_source_id: {}",
                    dias.value(i)
                );
            }
        }
    }

    #[test]
    fn orbital_params_parquet_has_correct_schema_and_rows() {
        let dir = tempdir().unwrap();
        let layout = PersistenceLayout::new(utf8(dir.path().to_path_buf()));
        let state = make_test_state();

        state.export_orbit_parquets(&layout, NightId(2)).unwrap();

        let path = layout.orbital_params_night_path(NightId(2));
        let (schema, batches) = load_parquet(&path).unwrap();

        // 10 columns
        let expected_cols = [
            "track_id",
            "orbit_type",
            "reference_epoch",
            "semi_major_axis",
            "eccentricity",
            "inclination",
            "ascending_node_longitude",
            "periapsis_argument",
            "mean_anomaly",
            "rms",
        ];
        assert_eq!(schema.fields().len(), expected_cols.len());
        for (i, name) in expected_cols.iter().enumerate() {
            assert_eq!(schema.field(i).name(), *name);
        }

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2, "expected 2 orbit rows (one per hypothesis)");

        // Check orbit_type values: one "Preliminary", one "Corrected"
        let mut orbit_types: Vec<String> = Vec::new();
        for batch in &batches {
            let col = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            for i in 0..col.len() {
                orbit_types.push(col.value(i).to_string());
            }
        }
        orbit_types.sort();
        assert_eq!(orbit_types, vec!["Corrected", "Preliminary"]);
    }

    #[test]
    fn orbital_params_values_match_input() {
        let dir = tempdir().unwrap();
        let layout = PersistenceLayout::new(utf8(dir.path().to_path_buf()));
        let state = make_test_state();

        state.export_orbit_parquets(&layout, NightId(2)).unwrap();

        let path = layout.orbital_params_night_path(NightId(2));
        let (_schema, batches) = load_parquet(&path).unwrap();
        assert_eq!(batches.len(), 1);
        let batch = &batches[0];

        // Collect SMA and RMS values
        let sma = batch
            .column(3)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let rms = batch
            .column(9)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        let mut sma_values: Vec<f64> = (0..sma.len()).map(|i| sma.value(i)).collect();
        sma_values.sort_by(|a, b| a.total_cmp(b));

        // Our two hypotheses have SMA 1.0 and 2.5
        assert!((sma_values[0] - 1.0).abs() < 1e-10);
        assert!((sma_values[1] - 2.5).abs() < 1e-10);

        let mut rms_values: Vec<f64> = (0..rms.len()).map(|i| rms.value(i)).collect();
        rms_values.sort_by(|a, b| a.total_cmp(b));

        // RMS values: 0.02 and 0.05
        assert!((rms_values[0] - 0.02).abs() < 1e-10);
        assert!((rms_values[1] - 0.05).abs() < 1e-10);
    }

    #[test]
    fn export_skips_hypotheses_without_orbit_results() {
        let dir = tempdir().unwrap();
        let layout = PersistenceLayout::new(utf8(dir.path().to_path_buf()));

        let mut state = make_test_state();
        // Remove orbit result for hyp 1 → only hyp 0 should appear in orbital params.
        state.orbit_results.remove(&ObjectNumber::Int(1));

        state.export_orbit_parquets(&layout, NightId(2)).unwrap();

        // Track members should still have rows for both hypotheses (they have
        // valid seeds/alerts even without orbit results).
        let path_tm = layout.track_members_night_path(NightId(2));
        let (_, batches_tm) = load_parquet(&path_tm).unwrap();
        let total_tm: usize = batches_tm.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_tm, 8);

        // Orbital params should have only 1 row (hyp 0).
        let path_op = layout.orbital_params_night_path(NightId(2));
        let (_, batches_op) = load_parquet(&path_op).unwrap();
        let total_op: usize = batches_op.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_op, 1);
    }

    #[test]
    fn export_different_night_ids_produce_different_files() {
        let dir = tempdir().unwrap();
        let layout = PersistenceLayout::new(utf8(dir.path().to_path_buf()));
        let state = make_test_state();

        state.export_orbit_parquets(&layout, NightId(10)).unwrap();
        state.export_orbit_parquets(&layout, NightId(20)).unwrap();

        assert!(layout.track_members_night_path(NightId(10)).exists());
        assert!(layout.track_members_night_path(NightId(20)).exists());
        assert!(layout.orbital_params_night_path(NightId(10)).exists());
        assert!(layout.orbital_params_night_path(NightId(20)).exists());
    }
}
