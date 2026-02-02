import numpy as np

# -----------------------------------------------------------------------------
# ONNX export + sanity check (unchanged)
# -----------------------------------------------------------------------------


def export_model_to_onnx(model, *, n_features: int, out_path) -> None:
    """
    Export a trained classifier to ONNX.

    - XGBoost: uses onnxmltools.convert.convert_xgboost (requires onnxmltools types)
    - scikit-learn: uses skl2onnx.convert_sklearn (uses onnxconverter_common types)

    Install:
        pip install onnx onnxconverter-common skl2onnx onnxmltools
    """
    is_xgb = model.__class__.__module__.startswith("xgboost")

    if is_xgb:
        try:
            from onnxmltools.convert import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType
        except Exception as e:
            raise RuntimeError(
                "XGBoost -> ONNX export requires onnxmltools.\n"
                "Install with:\n"
                "  pip install onnx onnxconverter-common skl2onnx onnxmltools\n"
                f"Original error: {e}"
            )

        booster = model.get_booster()
        initial_types = [("input", FloatTensorType([None, n_features]))]
        onnx_model = convert_xgboost(booster, initial_types=initial_types)
    else:
        try:
            from skl2onnx import convert_sklearn
            from onnxconverter_common.data_types import FloatTensorType
        except Exception as e:
            raise RuntimeError(
                "scikit-learn -> ONNX export requires skl2onnx + onnxconverter_common.\n"
                "Install with:\n"
                "  pip install onnx onnxconverter-common skl2onnx\n"
                f"Original error: {e}"
            )

        initial_types = [("input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_types)

    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())


def _extract_onnx_positive_proba(outputs) -> np.ndarray:
    """
    Extract the positive-class probability from ONNX Runtime outputs.
    Tries to handle the common output formats produced by tree converters.
    """
    float_arrays = [
        o for o in outputs if isinstance(o, np.ndarray) and o.dtype.kind == "f"
    ]
    if float_arrays:
        for arr in float_arrays:
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, 1]
        return float_arrays[-1].reshape(-1)

    raise RuntimeError(
        "Could not extract probabilities from ONNX outputs. "
        f"Got outputs types: {[type(o) for o in outputs]}"
    )


def sanity_check_onnx(
    model,
    onnx_path,
    X: np.ndarray,
    *,
    n_samples: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Validate ONNX export by comparing probabilities with the reference model.
    """
    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError(
            "onnxruntime is required for ONNX sanity check.\n"
            "Install with:\n"
            "  pip install onnxruntime\n"
            f"Original error: {e}"
        )

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    m = min(int(n_samples), n)

    idx = rng.choice(n, size=m, replace=False) if m < n else np.arange(n)
    Xs = X[idx].astype(np.float32, copy=False)

    p_ref = model.predict_proba(Xs)[:, 1].astype(np.float64, copy=False)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    print("ONNX inputs :", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print("ONNX outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: Xs})
    p_onnx = _extract_onnx_positive_proba(outputs).astype(np.float64, copy=False)

    if p_onnx.shape[0] != p_ref.shape[0]:
        raise RuntimeError(
            f"ONNX output shape mismatch: p_onnx={p_onnx.shape}, p_ref={p_ref.shape}"
        )

    diff = np.abs(p_ref - p_onnx)
    return {
        "n_checked": int(m),
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "p99_abs_diff": float(np.quantile(diff, 0.99)),
        "corr": float(np.corrcoef(p_ref, p_onnx)[0, 1]) if m > 1 else float("nan"),
    }
