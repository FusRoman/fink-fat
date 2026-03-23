use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use assert_cmd::{Command, pkg_name};
use datafusion::arrow::array::{
    ArrayRef, Float64Array, StringArray, UInt8Array, UInt32Array, UInt64Array,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::parquet::arrow::ArrowWriter;
use tempfile::TempDir;

fn write_minimal_engine_config(path: &Path, storage_path: &Path) {
    let yaml = format!("version: 1\nstorage_path: \"{}\"\n", storage_path.display());
    fs::write(path, yaml).expect("write config file");
}

fn alert_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("night_id", DataType::UInt32, false),
        Field::new("dia_source_id", DataType::UInt64, false),
        Field::new("ra", DataType::Float64, false),
        Field::new("ra_err", DataType::Float64, false),
        Field::new("dec", DataType::Float64, false),
        Field::new("dec_err", DataType::Float64, false),
        Field::new("mjd_tt", DataType::Float64, false),
        Field::new("mag", DataType::Float64, false),
        Field::new("mag_err", DataType::Float64, false),
        Field::new("band", DataType::UInt8, false),
        Field::new("observer_mpc_code", DataType::Utf8, false),
    ]))
}

fn write_minimal_alert_parquet(path: &Path) {
    let schema = alert_schema();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![60000_u32])) as ArrayRef,
            Arc::new(UInt64Array::from(vec![1_u64])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0_f64])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0e-6_f64])) as ArrayRef,
            Arc::new(Float64Array::from(vec![0.1_f64])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1.0e-6_f64])) as ArrayRef,
            Arc::new(Float64Array::from(vec![60000.5_f64])) as ArrayRef,
            Arc::new(Float64Array::from(vec![1200.0_f64])) as ArrayRef,
            Arc::new(Float64Array::from(vec![50.0_f64])) as ArrayRef,
            Arc::new(UInt8Array::from(vec![2_u8])) as ArrayRef,
            Arc::new(StringArray::from(vec!["I41"])) as ArrayRef,
        ],
    )
    .expect("build alert batch");

    let file = fs::File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("create parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}

fn file_uri(path: &Path) -> String {
    format!("file://{}", path.display())
}

struct CliFixture {
    _tmp: TempDir,
    config_path: PathBuf,
    alerts_path: PathBuf,
    storage_path: PathBuf,
}

fn make_fixture() -> CliFixture {
    let tmp = TempDir::new().expect("create tempdir");
    let root = tmp.path();

    let config_path = root.join("config.yml");
    let alerts_path = root.join("alerts.parquet");
    let storage_path = root.join("storage");

    write_minimal_engine_config(&config_path, &storage_path);
    write_minimal_alert_parquet(&alerts_path);

    CliFixture {
        _tmp: tmp,
        config_path,
        alerts_path,
        storage_path,
    }
}

#[test]
fn help_lists_expected_cli_options() {
    let mut cmd = Command::cargo_bin(pkg_name!()).expect("binary should build");
    let assert = cmd.arg("--help").assert().success();

    let stdout = String::from_utf8_lossy(&assert.get_output().stdout);
    assert!(stdout.contains("--alerts"));
    assert!(stdout.contains("--config"));
    assert!(stdout.contains("--progress"));
    assert!(stdout.contains("--logs"));
}

#[test]
fn cli_runs_pipeline_successfully_with_minimal_inputs() {
    let fx = make_fixture();

    Command::cargo_bin(pkg_name!())
        .expect("binary should build")
        .args([
            "--alerts",
            &file_uri(&fx.alerts_path),
            "--config",
            fx.config_path.to_str().expect("utf-8 path"),
        ])
        .assert()
        .success();

    assert!(fx.storage_path.exists(), "storage path should be created");
}

#[test]
fn cli_with_logs_creates_log_file() {
    let fx = make_fixture();

    Command::cargo_bin(pkg_name!())
        .expect("binary should build")
        .args([
            "--alerts",
            &file_uri(&fx.alerts_path),
            "--config",
            fx.config_path.to_str().expect("utf-8 path"),
            "--logs",
        ])
        .assert()
        .success();

    let logs_dir = fx.storage_path.join("logs");
    assert!(logs_dir.is_dir(), "logs directory should exist");

    let entries: Vec<_> = fs::read_dir(&logs_dir)
        .expect("read logs dir")
        .filter_map(Result::ok)
        .collect();
    assert!(!entries.is_empty(), "at least one log file is expected");

    let has_run_log = entries.iter().any(|e| {
        let name = e.file_name();
        let s = name.to_string_lossy();
        s.starts_with("run-") && s.ends_with(".log")
    });
    assert!(has_run_log, "expected a run-*.log file");
}

#[test]
fn cli_with_progress_succeeds() {
    let fx = make_fixture();

    Command::cargo_bin(pkg_name!())
        .expect("binary should build")
        .args([
            "--alerts",
            &file_uri(&fx.alerts_path),
            "--config",
            fx.config_path.to_str().expect("utf-8 path"),
            "--progress",
        ])
        .assert()
        .success();
}

#[test]
fn cli_with_logs_and_progress_succeeds_and_logs() {
    let fx = make_fixture();

    Command::cargo_bin(pkg_name!())
        .expect("binary should build")
        .args([
            "--alerts",
            &file_uri(&fx.alerts_path),
            "--config",
            fx.config_path.to_str().expect("utf-8 path"),
            "--logs",
            "--progress",
        ])
        .assert()
        .success();

    let logs_dir = fx.storage_path.join("logs");
    assert!(logs_dir.is_dir(), "logs directory should exist");
}

#[test]
fn cli_fails_when_config_file_is_missing() {
    let fx = make_fixture();
    let missing_config = fx.storage_path.join("missing-config.yml");

    let assert = Command::cargo_bin(pkg_name!())
        .expect("binary should build")
        .args([
            "--alerts",
            &file_uri(&fx.alerts_path),
            "--config",
            missing_config.to_str().expect("utf-8 path"),
        ])
        .assert()
        .failure();

    let stderr = String::from_utf8_lossy(&assert.get_output().stderr).to_lowercase();
    assert!(
        stderr.contains("no such file")
            || stderr.contains("not found")
            || stderr.contains("failed"),
        "stderr should mention missing config file, got: {stderr}"
    );
}

#[test]
fn cli_fails_with_invalid_alerts_uri() {
    let fx = make_fixture();

    let assert = Command::cargo_bin(pkg_name!())
        .expect("binary should build")
        .args([
            "--alerts",
            "not-a-valid-uri",
            "--config",
            fx.config_path.to_str().expect("utf-8 path"),
        ])
        .assert()
        .failure();

    let stderr = String::from_utf8_lossy(&assert.get_output().stderr).to_lowercase();
    assert!(
        stderr.contains("invalid") || stderr.contains("uri"),
        "stderr should mention invalid URI parsing, got: {stderr}"
    );
}

#[test]
fn cli_fails_with_invalid_parquet_input() {
    let fx = make_fixture();
    let invalid_alerts = fx.storage_path.join("invalid-alerts.parquet");
    fs::create_dir_all(&fx.storage_path).expect("create storage path");
    fs::write(&invalid_alerts, b"this is not a parquet file").expect("write invalid parquet");

    let assert = Command::cargo_bin(pkg_name!())
        .expect("binary should build")
        .args([
            "--alerts",
            &file_uri(&invalid_alerts),
            "--config",
            fx.config_path.to_str().expect("utf-8 path"),
        ])
        .assert()
        .failure();

    let stderr = String::from_utf8_lossy(&assert.get_output().stderr).to_lowercase();
    assert!(
        stderr.contains("parquet") || stderr.contains("stage failed") || stderr.contains("error"),
        "stderr should indicate ingest/parquet failure, got: {stderr}"
    );
}
