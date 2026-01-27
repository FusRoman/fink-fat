# edge_ml/model.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def list_models() -> List[str]:
    """Return available model kinds."""
    out = ["gb"]  # always available
    try:
        import xgboost  # noqa: F401

        out.append("xgb")
    except Exception:
        pass
    return out


def _require_xgboost() -> None:
    try:
        import xgboost  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "XGBoost is not installed (or failed to import). "
            "Install with: pip install xgboost\n"
            f"Original error: {e}"
        )


@dataclass(frozen=True)
class ModelSpec:
    kind: str
    name: str
    supports_predict_proba: bool = True


def get_model_specs() -> Dict[str, ModelSpec]:
    specs = {
        "gb": ModelSpec(kind="gb", name="sklearn GradientBoosting (pipeline)"),
    }
    if "xgb" in list_models():
        specs["xgb"] = ModelSpec(kind="xgb", name="XGBoost XGBClassifier")
    return specs


# ---------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------


def _make_gb_pipeline(random_state: int = 42, **overrides: Any) -> Pipeline:
    """
    Classic sklearn pipeline with impute+scale and GradientBoostingClassifier.

    overrides: any GradientBoostingClassifier kw args (e.g. n_estimators=300)
    """
    clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=random_state,
        **overrides,
    )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", clf),
        ]
    )


def _make_xgb_classifier(random_state: int = 42, **overrides: Any):
    """
    XGBoost model (no sklearn pipeline by default).

    Notes:
    - XGBoost handles missing values natively, so imputer not mandatory.
    - Scaling is not needed for trees.
    - You can still wrap it in a sklearn Pipeline later if you want.
    """
    _require_xgboost()
    from xgboost import XGBClassifier

    # Reasonable defaults for your use-case (tabular, many features, imbalance ~10-12%)
    params = dict(
        objective="binary:logistic",
        n_estimators=600,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        eval_metric="auc",
        tree_method="hist",  # fast CPU
        random_state=random_state,
        n_jobs=-1,
    )
    params.update(overrides)
    return XGBClassifier(**params)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def make_model(
    *,
    kind: str = "gb",
    random_state: int = 42,
    model_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Build a model by kind.

    Parameters
    ----------
    kind:
        - "gb": sklearn GradientBoostingClassifier inside a pipeline
        - "xgb": xgboost.XGBClassifier
    random_state:
        RNG seed used where applicable.
    model_kwargs:
        Extra kwargs forwarded to the underlying estimator constructor:
        - for "gb": passed to GradientBoostingClassifier(...)
        - for "xgb": passed to XGBClassifier(...)

    Returns
    -------
    A fitted-able estimator with predict_proba.

    Example
    -------
    model = make_model(kind="xgb", random_state=42, model_kwargs={"max_depth": 4})
    """
    model_kwargs = model_kwargs or {}

    kind = kind.lower().strip()
    if kind == "gb":
        return _make_gb_pipeline(random_state=random_state, **model_kwargs)
    if kind == "xgb":
        return _make_xgb_classifier(random_state=random_state, **model_kwargs)

    raise ValueError(
        f"Unknown model kind={kind!r}. Available: {list_models()} "
        "(install xgboost to enable 'xgb')."
    )
