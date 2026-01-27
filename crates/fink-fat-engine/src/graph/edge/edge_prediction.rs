use once_cell::sync::OnceCell;
use thiserror::Error;

use camino::Utf8Path;

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

/// Global singleton holding the ONNX Runtime session for the whole process.
static EDGE_MODEL_SESSION: OnceCell<Session> = OnceCell::new();

/// Global guard to ensure `ort::init().commit()` is run at most once.
static ORT_INIT: OnceCell<()> = OnceCell::new();

#[derive(Debug, Error)]
pub enum EdgeModelError {
    /// ONNX Runtime / ort error (session build, run, etc.).
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),

    /// The provided model path does not exist (nice early error message).
    #[error("ONNX model file not found: {0}")]
    ModelNotFound(String),
}

/// Initialize ONNX Runtime environment once (process-wide).
fn init_ort_once() {
    ORT_INIT.get_or_init(|| {
        ort::init().commit();
    });
}

/// Load the ONNX model as a process-global singleton session.
///
/// This keeps the model in memory for the duration of the program and avoids
/// paying model load/initialization costs repeatedly.
///
/// Parameters
/// ----------
/// * `model_path` - Path to the `.onnx` file (UTF-8 path).
///
/// Returns
/// -------
/// * `Ok(&'static Session)` - A reference to the global session.
/// * `Err(EdgeModelError)` - If the file is missing or session creation fails.
pub fn load_edge_model_session(
    model_path: impl AsRef<Utf8Path>,
) -> Result<&'static Session, EdgeModelError> {
    init_ort_once();

    let path = model_path.as_ref();

    if !path.exists() {
        return Err(EdgeModelError::ModelNotFound(path.as_str().to_string()));
    }

    EDGE_MODEL_SESSION.get_or_try_init(|| {
        // ort expects std::path::Path internally.
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(path.as_std_path())?;
        Ok(session)
    })
}

#[cfg(test)]
mod edge_prediction_test {
    use camino::Utf8PathBuf;

    use ort::session::Session;

    use super::{EdgeModelError, load_edge_model_session};

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
    fn onnx_session_is_singleton() {
        let path = model_path().expect("Missing FINK_FAT_ONNX_MODEL");

        let s1 = load_edge_model_session(&path).expect("First load failed");
        let s2 = load_edge_model_session(&path).expect("Second load failed");

        let p1 = s1 as *const Session as usize;
        let p2 = s2 as *const Session as usize;

        assert_eq!(p1, p2, "ONNX session is not a singleton (loaded twice)");
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
}
