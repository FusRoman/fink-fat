//! Config file generation for integration tests.
//!
//! Writes a minimal but realistic fink-fat YAML configuration
//! that mirrors the settings in `test_exp/test_config_file.yml`:
//! - `singer_cwna` cost function (σ_q=0.003) — matching the cost model used
//!   when the five ground-truth trajectories were evaluated.
//! - `max_gap_nights: 10` — allows linking across the wide gaps in the fixture
//!   data (e.g. nights 2926 → 2930 → 2934).
//! - Top-K ML ranking disabled (`top_k_per_left: ~`) — no ONNX model is required.
//! - `min_nodes: 3` in the bounded-beam solver.

use std::path::Path;

/// Write a fink-fat YAML config to `dest`, setting `storage_path` to
/// the provided directory.
///
/// All parameters match the production settings from `test_config_file.yml`.
/// Logging is disabled (`log_level: "off"`) to keep test output clean.
pub fn write_test_config(storage_path: &Path, dest: &Path) {
    let storage_str = storage_path
        .to_str()
        .expect("storage_path must be valid UTF-8");

    let yaml = format!(
        r#"version: 1
storage_path: "{storage_str}"
max_gap_nights: 10
compact_graph_every_delta: 4
binary_compression: "Zstd"
log_level: "error"

pairs:
  max_dt: "5 hours"
  max_angular_speed: "40 arcmin/day"
  max_flux_difference: 1.6
  allow_same_timebin: true

triplets:
  max_dt_between: "4 hours"
  max_pair_sep: "2.6 arcmin"
  max_predicted_residual: "0.3 arcmin"
  enforce_time_order: true
  max_flux_difference: 1.6

edges:
  use_ml_ranking: false
  max_cost_cut: 700.0
  top_k_per_left: ~
  onnx_batch_size: 128
  parallel_left_batches: true
  parallel_left_batch_size: 2048
  predictor_config:
    k_sigma: 3.0
    noise:
      variance_floor: 0.0
      drift_per_day: 0.0
      curvature_per_day2: 5.0e-13
    pad_cell_radius: true
    time_bin_dt: 0.021
    v_slack: 0.0
  cost:
    variant: singer_cwna
    sigma_q: 0.003
    cauchy_scale: 2.0
    student_nu: 3.0

solver_config:
  solver_policy:
    routing:
      Force: BoundedBeam
    trivial_max_nodes: 8
    trivial_max_active_edges: 16
    mcf_budget_s: 0.05
    k_mcf_s_per_edge_logn: 1.0e-8
    max_night_span_for_mcf: 4
  bounded_beam:
    max_tracks: 60000
    min_nodes: 3
    beam_width: 40000
    max_out_per_node: 2000
    max_tracks_per_source: 2000
    max_expansions: 50000
"#
    );

    std::fs::write(dest, yaml)
        .unwrap_or_else(|e| panic!("cannot write test config to {dest:?}: {e}"));
}
