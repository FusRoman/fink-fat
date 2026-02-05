//! ONNX edge-ranking inference utilities (ONNX Runtime via `ort`).
//!
//! This module provides a small, fast wrapper around an ONNX classification model
//! used to score candidate inter-night edges in the fink-fat graph. The typical
//! workflow is:
//! 1. Load an ONNX model once (process-wide ORT init + session build).
//! 2. Resolve the indices of the relevant outputs (e.g. `probabilities`, `label`).
//! 3. Run inference repeatedly on batches of [`EdgeFeatures`].
//!
//! The design is intentionally optimized for high call counts:
//! - output lookup by **index** (no string scan in the hot path),
//! - input preparation as a contiguous `[N, D]` matrix,
//! - minimal post-processing for `p(class=1)` extraction.
//!
//! Output conventions
//! ------------------
//! The module assumes an ONNX model exporting at least one output named
//! `probabilities` with shape `[N, 2]` and dtype `f32`, where columns are
//! `[p(class=0), p(class=1)]`.
//!
//! Some exporters also provide a `label` output with shape `[N]` and dtype `i64`.
//! This output is treated as optional.
//!
//! Thread-safety and mutability
//! ----------------------------
//! ONNX Runtime may mutate internal session state during inference (memory arenas,
//! cached allocations), so `Session::run` requires `&mut Session`. If you want to
//! run inference in parallel, use one [`EdgeRankingModel`] per worker thread or
//! guard a shared model with a mutex (the latter may reduce throughput).

use std::ops::Index;

use ndarray::Array2;
use once_cell::sync::OnceCell;
use ort::value::Tensor;
use thiserror::Error;

use camino::Utf8Path;

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

use crate::graph::edge::edge_features::EdgeFeatures;

/// Global guard to ensure `ort::init().commit()` is run at most once.
///
/// ONNX Runtime maintains global state (environment, allocators, logging).
/// Initializing it more than once is unnecessary and can be problematic.
static ORT_INIT: OnceCell<()> = OnceCell::new();

/// Errors that can occur while loading the model or running ONNX inference.
#[derive(Debug, Error)]
pub enum EdgeModelError {
    /// ONNX Runtime / `ort` error (session build, tensor extraction, run failure, etc.).
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),

    /// The provided model path does not exist.
    ///
    /// This is returned early to provide a clear user-facing message instead of
    /// a lower-level ORT error.
    #[error("ONNX model file not found: {0}")]
    ModelNotFound(String),

    /// ML ranking was requested but no EdgeRankingModel was provided.
    #[error("ML ranking requested but no EdgeRankingModel was provided")]
    MissingModel,
}

/// Initialize ONNX Runtime once for the whole process.
///
/// This function is idempotent: multiple calls are safe and will only initialize
/// ONNX Runtime a single time.
///
/// Notes
/// -----
/// * This calls `ort::init().commit()` the first time it is invoked.
/// * It is intentionally private; callers should rely on public constructors
///   (e.g. [`EdgeRankingModel::load_edge_ranking_model`]) which guarantee ORT is
///   initialized.
fn init_ort_once() {
    ORT_INIT.get_or_init(|| {
        ort::init().commit();
    });
}

/// High-level wrapper for an ONNX edge-ranking model.
///
/// This struct owns:
/// - an ONNX Runtime [`Session`],
/// - a resolved mapping from semantic output names to output indices
///   ([`EdgeModelOutputs`]).
///
/// It is intended to be constructed once and reused for many inference calls.
pub struct EdgeRankingModel {
    session: Session,
    outputs: EdgeModelOutputs,
}

impl EdgeRankingModel {
    /// Load an ONNX edge-ranking model from disk and resolve output indices.
    ///
    /// This is the main entry point to create an inference-ready model. It:
    /// 1. Initializes ONNX Runtime (process-wide) if needed,
    /// 2. Builds an optimized ORT session for the given `.onnx` file,
    /// 3. Resolves output indices for fast inference (e.g. finds `probabilities`).
    ///
    /// Arguments
    /// ---------
    /// * `model_path` – Path to the `.onnx` model file (UTF-8).
    ///
    /// Return
    /// ------
    /// * `Ok(EdgeRankingModel)` on success.
    /// * `Err(EdgeModelError)` if:
    ///   - the file does not exist,
    ///   - session creation fails,
    ///   - or required outputs cannot be resolved.
    ///
    /// Notes
    /// -----
    /// * Output resolution currently relies on output names:
    ///   - required: `"probabilities"`
    ///   - optional: `"label"`
    /// * If your exporter uses different output names, adapt
    ///   [`EdgeModelOutputs::resolve_output_indices`].
    pub fn load_edge_ranking_model(
        model_path: impl AsRef<Utf8Path>,
    ) -> Result<Self, EdgeModelError> {
        let session = load_edge_model_session(model_path)?;
        let outputs = EdgeModelOutputs::resolve_output_indices(&session)?;
        Ok(Self { session, outputs })
    }

    /// Get a reference to the internal ONNX Runtime session.
    ///
    /// This can be useful to inspect model inputs/outputs or metadata, e.g. for
    /// debugging or logging.
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Get a reference to resolved output indices.
    ///
    /// This exposes the indices of semantic outputs (probabilities/label) that
    /// were resolved at load time, allowing the caller to inspect the mapping.
    pub fn outputs(&self) -> &EdgeModelOutputs {
        &self.outputs
    }

    /// Predict class probabilities for a batch of edges.
    ///
    /// This function:
    /// 1. Converts `batch` into a dense `[N, D]` float tensor,
    /// 2. Runs the ONNX model once for the whole batch,
    /// 3. Extracts the `probabilities` output as a 2D matrix `[N, 2]`.
    ///
    /// Arguments
    /// ---------
    /// * `batch` – Slice of [`EdgeFeatures`] to score.
    ///
    /// Return
    /// ------
    /// * `Ok(Array2<f32>)` – Probabilities matrix of shape `[N, 2]` where columns
    ///   are `[p(class=0), p(class=1)]`.
    /// * `Err(EdgeModelError)` if:
    ///   - input tensor creation fails,
    ///   - ORT execution fails,
    ///   - output extraction fails or has an unexpected shape.
    ///
    /// Performance
    /// -----------
    /// * Output selection is by index (resolved once), avoiding repeated string
    ///   comparisons in the hot path.
    /// * For maximum throughput, prefer batching many edges per call.
    pub fn predict_proba(&mut self, batch: &[EdgeFeatures]) -> Result<Array2<f32>, EdgeModelError> {
        let input = build_input_tensor(batch)?;
        let outputs = self.session.run(ort::inputs![input])?;

        let v = outputs.index(self.outputs.probabilities);

        let (shape, data) = v.try_extract_tensor::<f32>()?;
        let (n, k) = expect_2d_usize(shape)?;
        array2_from_flat((n, k), data)
    }

    /// Predict the positive-class probability `p(class=1)` for a batch of edges.
    ///
    /// This is a convenience wrapper around [`Self::predict_proba`] that returns
    /// a single score per edge.
    ///
    /// Arguments
    /// ---------
    /// * `batch` – Slice of [`EdgeFeatures`] to score.
    ///
    /// Return
    /// ------
    /// * `Ok(Vec<f32>)` – Vector of length `N` with `p(class=1)` for each edge.
    /// * `Err(EdgeModelError)` – If probability prediction fails.
    ///
    /// Notes
    /// -----
    /// * Assumes the model exports `probabilities` with shape `[N, 2]` and uses
    ///   column 1 as the positive class. A debug assertion checks `ncols == 2`
    ///   in debug builds.
    pub fn predict_positive_proba(
        &mut self,
        batch: &[EdgeFeatures],
    ) -> Result<Vec<f32>, EdgeModelError> {
        let proba = self.predict_proba(batch)?;
        debug_assert_eq!(proba.ncols(), 2);
        Ok(proba.rows().into_iter().map(|r| r[1]).collect())
    }

    /// Predict class labels for a batch of edges, if the model exports them.
    ///
    /// Many `sklearn-onnx` exports include a `label` output (dtype `i64`,
    /// shape `[N]`). Some models only export probabilities. This method handles
    /// both cases.
    ///
    /// Arguments
    /// ---------
    /// * `batch` – Slice of [`EdgeFeatures`] to classify.
    ///
    /// Return
    /// ------
    /// * `Ok(Some(Vec<i64>))` if the model provides a `label` output:
    ///   - vector length is `N`,
    ///   - values are typically `0` or `1` for binary classification.
    /// * `Ok(None)` if the model does not export a `label` output.
    /// * `Err(EdgeModelError)` if inference or extraction fails, or if the
    ///   `label` output has an unexpected shape.
    pub fn predict_label(
        &mut self,
        batch: &[EdgeFeatures],
    ) -> Result<Option<Vec<i64>>, EdgeModelError> {
        let Some(label_idx) = self.outputs.label else {
            return Ok(None);
        };

        let input = build_input_tensor(batch)?;
        let outputs = self.session.run(ort::inputs![input])?;

        let v = outputs.index(label_idx);

        let (shape, data) = v.try_extract_tensor::<i64>()?;
        if shape.len() != 1 {
            return Err(EdgeModelError::Ort(ort::Error::new(
                "Label output is not 1D",
            )));
        }

        Ok(Some(data.to_vec()))
    }
}

/// Resolved ONNX output indices for fast inference.
///
/// This struct is computed once at model load time and reused for all subsequent
/// inference calls. It avoids repeated lookup by name (which can be expensive
/// when inference is called many times).
///
/// Conventions
/// -----------
/// * `probabilities` is required and must refer to a `Tensor<f32>` with shape
///   `[N, 2]` (binary classifier).
/// * `label` is optional and, if present, refers to a `Tensor<i64>` with shape
///   `[N]`.
#[derive(Debug, Clone)]
pub struct EdgeModelOutputs {
    /// Index of the `probabilities` output (Tensor<f32>, shape [N, 2]).
    pub probabilities: usize,

    /// Optional index of the `label` output (Tensor<i64>, shape [N]).
    pub label: Option<usize>,
}

impl EdgeModelOutputs {
    /// Resolve output indices from a session by matching output names.
    ///
    /// This scans `session.outputs()` once and records the indices of:
    /// - required: `"probabilities"`
    /// - optional: `"label"`
    ///
    /// Arguments
    /// ---------
    /// * `session` – Loaded ONNX Runtime session.
    ///
    /// Return
    /// ------
    /// * `Ok(EdgeModelOutputs)` on success.
    /// * `Err(EdgeModelError)` if `probabilities` cannot be found.
    ///
    /// Notes
    /// -----
    /// * Output naming depends on the export toolchain. If your model uses
    ///   different names, change the match strings here.
    fn resolve_output_indices(session: &Session) -> Result<Self, EdgeModelError> {
        let mut prob_idx = None;
        let mut label_idx = None;

        for (i, out) in session.outputs().iter().enumerate() {
            match out.name() {
                "probabilities" => prob_idx = Some(i),
                "label" => label_idx = Some(i),
                _ => {}
            }
        }

        let probabilities = prob_idx.ok_or_else(|| {
            EdgeModelError::Ort(ort::Error::new(
                "ONNX model has no output named 'probabilities'",
            ))
        })?;

        Ok(Self {
            probabilities,
            label: label_idx,
        })
    }
}

/// Load an ONNX model session from disk with ORT initialization.
///
/// This function:
/// 1. Ensures ONNX Runtime is initialized once per process,
/// 2. Validates that the model file exists,
/// 3. Creates an optimized ORT session from the file.
///
/// Arguments
/// ---------
/// * `model_path` – Path to the `.onnx` model file (UTF-8).
///
/// Return
/// ------
/// * `Ok(Session)` – Ready-to-run ONNX Runtime session.
/// * `Err(EdgeModelError)` if:
///   - the model file does not exist,
///   - session building fails.
///
/// Notes
/// -----
/// * The optimization level is currently set to `Level3`, which usually yields
///   best throughput for repeated inference, at the cost of longer session build.
fn load_edge_model_session(model_path: impl AsRef<Utf8Path>) -> Result<Session, EdgeModelError> {
    init_ort_once();

    let path = model_path.as_ref();

    if !path.exists() {
        return Err(EdgeModelError::ModelNotFound(path.as_str().to_string()));
    }

    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(path.as_std_path())?;
    Ok(session)
}

/// Convert a batch of [`EdgeFeatures`] to a dense `[N, D]` matrix of `f32`.
///
/// This is the canonical input preparation step for the ONNX model. Each
/// [`EdgeFeatures`] instance is flattened using [`EdgeFeatures::iter_flat`] and
/// written into one row of the output matrix.
///
/// Arguments
/// ---------
/// * `batch` – Feature vectors to convert.
///
/// Return
/// ------
/// * `Array2<f32>` – Matrix of shape `[N, D]` in row-major order.
///
/// Notes
/// -----
/// * The feature ordering must match training exactly.
/// * Values are cast from `f64` to `f32`.
fn features_to_array2_f32(batch: &[EdgeFeatures]) -> Array2<f32> {
    let n = batch.len();
    let d = EdgeFeatures::len_flat();

    let mut x = Array2::<f32>::zeros((n, d));
    for (i, feat) in batch.iter().enumerate() {
        for (j, v) in feat.iter_flat().enumerate() {
            x[(i, j)] = v as f32;
        }
    }
    x
}

/// Build an ONNX input tensor from a batch of [`EdgeFeatures`].
///
/// This wraps [`features_to_array2_f32`] and converts the resulting matrix into
/// an ONNX Runtime [`Tensor<f32>`].
///
/// Arguments
/// ---------
/// * `batch` – Slice of features to feed into the model.
///
/// Return
/// ------
/// * `Ok(Tensor<f32>)` – Input tensor of shape `[N, D]`.
/// * `Err(EdgeModelError)` – If tensor creation fails in ORT.
fn build_input_tensor(batch: &[EdgeFeatures]) -> Result<Tensor<f32>, EdgeModelError> {
    let x = features_to_array2_f32(batch);
    Ok(Tensor::from_array(x)?)
}

/// Validate a 2D ONNX tensor shape and convert to `(usize, usize)`.
///
/// ONNX Runtime reports dimensions as `i64`. This function verifies that the
/// shape is exactly 2D and converts both dimensions to `usize`.
///
/// Arguments
/// ---------
/// * `shape` – ONNX dimension slice (e.g. `[N, K]`).
///
/// Return
/// ------
/// * `Ok((n, k))` – Converted dimensions.
/// * `Err(EdgeModelError)` – If `shape.len() != 2` or any dimension is invalid.
fn expect_2d_usize(shape: &[i64]) -> Result<(usize, usize), EdgeModelError> {
    if shape.len() != 2 {
        return Err(EdgeModelError::Ort(ort::Error::new(format!(
            "Expected 2D output, got shape={shape:?}"
        ))));
    }

    let n = dim_to_usize(shape[0], "N", shape)?;
    let d = dim_to_usize(shape[1], "D", shape)?;
    Ok((n, d))
}

/// Convert a single ONNX dimension to `usize` with context for error messages.
///
/// Arguments
/// ---------
/// * `dim` – Dimension value (typically `i64`).
/// * `name` – Friendly dimension name (e.g. `"N"`, `"K"`).
/// * `shape` – The full shape slice, included in error messages.
///
/// Return
/// ------
/// * `Ok(usize)` – Converted dimension.
/// * `Err(EdgeModelError)` – If `dim` is negative or cannot fit into `usize`.
fn dim_to_usize(dim: i64, name: &'static str, shape: &[i64]) -> Result<usize, EdgeModelError> {
    usize::try_from(dim).map_err(|_| {
        EdgeModelError::Ort(ort::Error::new(format!(
            "Invalid {name}={dim} in shape={shape:?}"
        )))
    })
}

/// Build an `Array2<f32>` from a flat row-major buffer and an explicit shape.
///
/// This helper verifies that the data buffer length matches the requested shape.
/// It is primarily used to reconstruct a 2D output matrix from an ONNX output
/// tensor returned as a flat slice.
///
/// Arguments
/// ---------
/// * `shape` – Target `(n, k)` shape.
/// * `data` – Flat buffer, expected length `n * k`, in row-major order.
///
/// Return
/// ------
/// * `Ok(Array2<f32>)` – Owned `ndarray` matrix of shape `(n, k)`.
/// * `Err(EdgeModelError)` – If buffer length mismatches or ndarray rejects the shape.
fn array2_from_flat(shape: (usize, usize), data: &[f32]) -> Result<Array2<f32>, EdgeModelError> {
    let expected = shape.0 * shape.1;
    if data.len() != expected {
        return Err(EdgeModelError::Ort(ort::Error::new(format!(
            "Output buffer length mismatch: got {}, expected {} for shape={:?}",
            data.len(),
            expected,
            shape
        ))));
    }

    Array2::from_shape_vec(shape, data.to_vec()).map_err(|e| {
        EdgeModelError::Ort(ort::Error::new(format!(
            "ndarray shape error for shape={shape:?}: {e}"
        )))
    })
}

#[cfg(test)]
mod edge_prediction_test {

    use crate::graph::edge::{
        photometry_features::EdgePhotometryFeatures, position_features::EdgePositionFeatures,
        uncertainty_features::EdgeUncertaintyFeatures, velocity_features::EdgeVelocityFeatures,
    };

    use super::*;
    use camino::Utf8PathBuf;

    use super::{EdgeModelError, load_edge_model_session};

    /// Build a fully-populated `EdgeFeatures` instance for tests.
    ///
    /// We intentionally fill every scalar leaf feature with a deterministic
    /// pattern, so we can verify canonical ordering and casting.
    /// Build a fully-populated `EdgeFeatures` instance for tests.
    ///
    /// We intentionally fill every scalar leaf feature with a deterministic
    /// pattern, so we can verify canonical ordering and casting.
    fn dummy_edge_features(base: f64) -> EdgeFeatures {
        EdgeFeatures {
            position: EdgePositionFeatures {
                chi2_pos: base + 0.0,
                log_chi2_pos: base + 1.0,
                z_dx: base + 2.0,
                z_dy: base + 3.0,
                z_resid_norm: base + 4.0,
                z_along: base + 5.0,
                z_cross: base + 6.0,
                chol_z1: base + 7.0,
                chol_z2: base + 8.0,
                chol_z_norm: base + 9.0,
            },
            velocity: EdgeVelocityFeatures {
                cos_dtheta_v: base + 10.0,
                rel_speed_diff: base + 11.0,
                innov_speed_ratio: base + 12.0,
                chi2_vel: base + 13.0,
                log_chi2_vel: base + 14.0,
            },
            uncertainty: EdgeUncertaintyFeatures(base + 15.0),
            photometry: EdgePhotometryFeatures {
                z_flux: base + 16.0,
                flux_std_ratio: base + 17.0,
                band_shared: base + 18.0,
            },
        }
    }

    fn model_path() -> Result<Utf8PathBuf, String> {
        match std::env::var("FINK_FAT_ONNX_MODEL") {
            Ok(p) if !p.is_empty() => Ok(Utf8PathBuf::from(p)),
            Ok(_) => Err(
                "Environment variable FINK_FAT_ONNX_MODEL is set but empty.\n\
Please set it to the path of the ONNX model, e.g.:\n\
export FINK_FAT_ONNX_MODEL=/path/to/edge_classifier.onnx"
                    .to_string(),
            ),
            Err(_) => Err("Environment variable FINK_FAT_ONNX_MODEL is not set.\n\
Please set it to the path of the ONNX model, e.g.:\n\
export FINK_FAT_ONNX_MODEL=/path/to/edge_classifier.onnx"
                .to_string()),
        }
    }

    #[test]
    fn ort_can_initialize_and_load_model() {
        let path = model_path().expect("Missing FINK_FAT_ONNX_MODEL");

        assert!(
            path.exists(),
            "ONNX model not found at {:?}. Set FINK_FAT_ONNX_MODEL to override.",
            path
        );

        let session = load_edge_model_session(&path).expect("Failed to load ONNX model session");

        assert!(
            !session.inputs().is_empty(),
            "ONNX model should have at least one input"
        );
        assert!(
            !session.outputs().is_empty(),
            "ONNX model should have at least one output"
        );
    }

    #[test]
    fn loading_nonexistent_model_fails_cleanly() {
        let bad_path = Utf8PathBuf::from("this/path/does/not/exist.onnx");

        let err =
            load_edge_model_session(&bad_path).expect_err("Expected failure for missing ONNX file");

        match err {
            EdgeModelError::ModelNotFound(_) => {}
            other => panic!("Unexpected error variant: {:?}", other),
        }
    }

    #[test]
    fn dim_to_usize_accepts_positive() {
        let shape = &[3, 4];
        let n = dim_to_usize(3, "N", shape).unwrap();
        assert_eq!(n, 3);
    }

    #[test]
    fn dim_to_usize_rejects_negative() {
        let shape = &[-1, 4];
        let err = dim_to_usize(-1, "N", shape).unwrap_err();
        match err {
            EdgeModelError::Ort(_) => {}
            _ => panic!("Expected Ort error"),
        }
    }

    #[test]
    fn expect_2d_usize_accepts_valid_shape() {
        let shape = &[5, 7];
        let (n, d) = expect_2d_usize(shape).unwrap();
        assert_eq!((n, d), (5, 7));
    }

    #[test]
    fn expect_2d_usize_rejects_1d() {
        let shape = &[10];
        assert!(expect_2d_usize(shape).is_err());
    }

    #[test]
    fn expect_2d_usize_rejects_3d() {
        let shape = &[2, 3, 4];
        assert!(expect_2d_usize(shape).is_err());
    }

    #[test]
    fn array2_from_flat_builds_correct_array() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let arr = array2_from_flat((2, 2), &data).unwrap();

        assert_eq!(arr.shape(), &[2, 2]);
        assert_eq!(arr[[0, 0]], 1.0);
        assert_eq!(arr[[1, 1]], 4.0);
    }

    #[test]
    fn array2_from_flat_rejects_size_mismatch() {
        let data = vec![1.0, 2.0, 3.0];
        let err = array2_from_flat((2, 2), &data).unwrap_err();

        match err {
            EdgeModelError::Ort(_) => {}
            _ => panic!("Expected Ort error"),
        }
    }

    // -------------------------------------------------------------------------
    // Pure unit tests (no ONNX model needed)
    // -------------------------------------------------------------------------

    #[test]
    fn ort_init_is_idempotent() {
        init_ort_once();
        init_ort_once();
        init_ort_once();
    }

    #[test]
    fn expect_2d_usize_rejects_non_2d() {
        assert!(expect_2d_usize(&[10_i64]).is_err());
        assert!(expect_2d_usize(&[2_i64, 3_i64, 4_i64]).is_err());
    }

    #[test]
    fn features_to_array2_f32_has_expected_shape_and_order() {
        let f0 = dummy_edge_features(0.0);
        let f1 = dummy_edge_features(100.0);

        let x = features_to_array2_f32(&[f0.clone(), f1.clone()]);
        assert_eq!(x.nrows(), 2);
        assert_eq!(x.ncols(), EdgeFeatures::len_flat());

        // Row 0 matches iter_flat exactly
        for (j, v) in f0.iter_flat().enumerate() {
            assert_eq!(x[[0, j]], v as f32, "Mismatch at row0 col{j}");
        }

        // Row 1 matches iter_flat exactly
        for (j, v) in f1.iter_flat().enumerate() {
            assert_eq!(x[[1, j]], v as f32, "Mismatch at row1 col{j}");
        }
    }

    // -------------------------------------------------------------------------
    // Integration tests (require FINK_FAT_ONNX_MODEL)
    // -------------------------------------------------------------------------

    #[test]
    fn edge_ranking_model_loads_and_resolves_outputs() {
        let path = model_path().expect("Missing FINK_FAT_ONNX_MODEL");

        let model = EdgeRankingModel::load_edge_ranking_model(&path)
            .expect("Failed to load EdgeRankingModel");

        // Sanity: the model must have inputs/outputs
        assert!(!model.session().inputs().is_empty());
        assert!(!model.session().outputs().is_empty());

        // Critical: probabilities output must be resolved
        let outs = model.outputs();
        assert!(outs.probabilities < model.session().outputs().len());

        // label is optional; if present, index must be in bounds
        if let Some(label_idx) = outs.label {
            assert!(label_idx < model.session().outputs().len());
        }
    }

    #[test]
    fn edge_ranking_model_predict_proba_runs_and_shapes_match() {
        let path = model_path().expect("Missing FINK_FAT_ONNX_MODEL");

        let mut model =
            EdgeRankingModel::load_edge_ranking_model(&path).expect("Failed to load model");

        let batch = vec![dummy_edge_features(0.0), dummy_edge_features(10.0)];
        let proba = model.predict_proba(&batch).expect("predict_proba failed");

        assert_eq!(proba.nrows(), batch.len());
        assert_eq!(
            proba.ncols(),
            2,
            "Expected probabilities shape [N,2], got [N,{}]",
            proba.ncols()
        );
    }

    #[test]
    fn edge_ranking_model_predict_positive_proba_is_in_0_1() {
        let path = model_path().expect("Missing FINK_FAT_ONNX_MODEL");

        let mut model =
            EdgeRankingModel::load_edge_ranking_model(&path).expect("Failed to load model");

        let batch = vec![dummy_edge_features(0.0), dummy_edge_features(10.0)];
        let p1 = model
            .predict_positive_proba(&batch)
            .expect("predict_positive_proba failed");

        assert_eq!(p1.len(), batch.len());
        for (i, p) in p1.iter().enumerate() {
            assert!((0.0..=1.0).contains(p), "p1[{i}] out of [0,1] range: {p}");
        }
    }

    #[test]
    fn edge_ranking_model_predict_label_matches_batch_len_if_present() {
        let path = model_path().expect("Missing FINK_FAT_ONNX_MODEL");

        let mut model =
            EdgeRankingModel::load_edge_ranking_model(&path).expect("Failed to load model");

        let batch = vec![dummy_edge_features(0.0), dummy_edge_features(10.0)];
        let labels = model.predict_label(&batch).expect("predict_label failed");

        if let Some(labels) = labels {
            assert_eq!(labels.len(), batch.len());
        } else {
            // Allowed: some models do not export `label`.
            eprintln!("Model has no 'label' output (ok).");
        }
    }
}
