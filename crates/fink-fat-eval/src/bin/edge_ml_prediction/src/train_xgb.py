# edge_ml/train_xgb.py — model construction, training, evaluation and persistence

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from xgboost import XGBClassifier

import numpy as np
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from skl2onnx import to_onnx as skl_to_onnx, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from xgboost import XGBClassifier as _XGBClassifier


def build_model(
    params: dict, n_negative: int, n_positive: int, early_stopping_rounds: int
) -> XGBClassifier:
    """
    Instantiate an XGBClassifier from a flat hyperparameter dict.

    If ``scale_pos_weight`` is absent or None in params, it is computed
    automatically as ``n_negative / n_positive`` to compensate class imbalance.

    ``early_stopping_rounds`` is a constructor parameter in XGBoost ≥ 3.0.
    """
    p = dict(params)
    if p.get("scale_pos_weight") is None:
        p["scale_pos_weight"] = n_negative / max(n_positive, 1)
    p["early_stopping_rounds"] = early_stopping_rounds
    return XGBClassifier(**p)


def train(
    model: XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> XGBClassifier:
    """
    Fit the model with early stopping monitored on the test set.

    ``early_stopping_rounds`` and ``feature_names`` must be set in the
    constructor (XGBoost ≥ 3.0). Returns the fitted model; the best iteration
    is preserved automatically.
    """
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    return model


def evaluate(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """
    Compute classification metrics on the test set.

    Returns a dict with ``roc_auc``, ``pr_auc``, and ``logloss``.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "logloss": log_loss(y_test, y_proba),
    }


def save_model(model: XGBClassifier, path: Path) -> None:
    """Save the model in XGBoost binary format (`.ubj`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)
    print(f"Model saved → {path}")


def export_onnx(model: XGBClassifier, path: Path) -> None:
    """Export the trained model to ONNX format via ``skl2onnx``.

    ``XGBClassifier.save_model()`` saves in XGBoost-native format regardless
    of the file extension — it does NOT produce a valid ONNX protobuf.
    ``skl2onnx.to_onnx`` is the correct conversion path for sklearn-compatible
    models.

    The resulting ONNX graph has two outputs:

    * ``label``         — int64, shape ``[N]``         — predicted class (0 or 1)
    * ``probabilities`` — float32, shape ``[N, 2]``    — P(class=0), P(class=1)

    ``zipmap=False`` is required so that ``probabilities`` is exported as a
    plain float32 tensor rather than a sequence-of-maps (ZipMap), which is
    what the Rust ``ort`` consumer expects.

    In Rust, extract ``probabilities[:, 1]`` to get P(true_edge).
    """
    update_registered_converter(
        _XGBClassifier,
        "XGBoostXGBClassifier",
        calculate_linear_classifier_output_shapes,
        convert_xgboost,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )

    path = Path(path).with_suffix(".onnx")
    path.parent.mkdir(parents=True, exist_ok=True)

    n_features = model.n_features_in_
    X_sample = np.zeros((1, n_features), dtype=np.float32)
    onnx_model = skl_to_onnx(
        model,
        X_sample,
        options={"zipmap": False},
        target_opset={"": 17, "ai.onnx.ml": 3},
    )

    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"ONNX model saved → {path}")
