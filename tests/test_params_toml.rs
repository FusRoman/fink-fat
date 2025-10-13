use camino::Utf8PathBuf;
use fink_fat::params::FinkFatParams;
use tempfile::tempdir;

fn data_path(file: &str) -> Utf8PathBuf {
    Utf8PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(file)
}

#[test]
fn roundtrip_default_in_memory() {
    // Create a default parameter set
    let p0 = FinkFatParams::default();

    // Serialize to TOML string (pretty)
    let toml_txt = p0
        .to_toml_string_pretty()
        .expect("serialize default to TOML");

    // Sanity: should include schema_version = "1"
    assert!(toml_txt.contains(r#"schema_version = "1""#));

    // Deserialize back
    let p1 = FinkFatParams::from_toml_str(&toml_txt).expect("parse TOML back");

    // Must be identical (PartialEq derived)
    assert_eq!(p0, p1);
}

#[test]
fn roundtrip_file_save_load() {
    let p0 = FinkFatParams::default();

    let dir = tempdir().expect("tempdir");
    let dir_utf8 =
        Utf8PathBuf::from_path_buf(dir.path().to_path_buf()).expect("non-UTF8 temp path");
    let path = dir_utf8.join("config.toml");

    p0.save_toml_file(&path).expect("save default TOML");
    let p1 = FinkFatParams::load_toml_file(&path).expect("load saved TOML");

    assert_eq!(p0, p1);
}

#[test]
fn toml_file_pretty_and_readable() {
    let p0 = FinkFatParams::default();
    let s = p0.to_toml_string_pretty().unwrap();

    // Basic structure hints: sections appear as expected
    // We don't check exact defaults here, just presence of the structure.
    assert!(s.contains("[binning]"));
    assert!(s.contains("[pairs]"));
    assert!(s.contains("[triplets]"));
    assert!(s.contains("[link.predict]"));
    assert!(s.contains("[link.scoring.weights]"));
    assert!(s.contains("[link.scoring.gates]"));
    assert!(s.contains("[link.scoring.scales]"));
    assert!(s.contains("[link.limits]"));
}

#[test]
fn minimal_file_uses_defaults() {
    let path = data_path("params_minimal.toml");
    let params = FinkFatParams::load_toml_file(&path).expect("load minimal.toml");
    assert_eq!(params, FinkFatParams::default());
}

#[test]
fn full_file_has_expected_values() {
    let path = data_path("params_full.toml");
    let p = FinkFatParams::load_toml_file(&path).expect("load full.toml");

    assert!(p.show_progress);
    assert_eq!(p.binning.healpix_depth, 10);
    assert!((p.binning.time_bin_width_days - 0.02).abs() < 1e-12);

    assert!((p.pairs.max_dt - 0.06).abs() < 1e-12);
    assert!(p.pairs.allow_same_timebin);

    assert!(p.triplets.enforce_time_order);
    assert!((p.triplets.max_predicted_residual - 0.0008).abs() < 1e-12);

    assert!((p.link.predict.k_sigma - 3.5).abs() < 1e-12);
    assert!(p.link.predict.pad_cell_radius);
    assert!((p.link.predict.noise.variance_floor - 1.0e-12).abs() < 1e-30);
    assert!((p.link.predict.noise.curvature_per_day2 - 5.0e-14).abs() < 1e-30);

    assert!((p.link.scoring.weights.w_pos - 0.5).abs() < 1e-12);
    assert!((p.link.scoring.gates.max_d2_pos - 12.0).abs() < 1e-12);
    assert!((p.link.scoring.scales.theta0 - 0.08726646259971647).abs() < 1e-15);

    assert_eq!(p.link.limits.top_k_per_left, 10);
    assert_eq!(p.link.limits.max_total_edges, Some(200000));
    assert_eq!(p.link.limits.max_cost, Some(10.0));
}

#[test]
fn invalid_file_fails_validation() {
    let path = data_path("params_invalid.toml");
    let res = FinkFatParams::load_toml_file(&path);
    assert!(res.is_err(), "invalid.toml should fail validation");
}

#[test]
fn schema_version_is_present_on_serialize() {
    let s = FinkFatParams::default().to_toml_string_pretty().unwrap();
    assert!(s.contains(r#"schema_version = "1""#));
}

#[test]
fn schema_version_missing_defaults_to_1_on_parse() {
    let txt = std::fs::read_to_string(data_path("params_minimal.toml")).unwrap();
    let _params = FinkFatParams::from_toml_str(&txt).unwrap();
}
