from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import logging

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, precision_recall_fscore_support, roc_auc_score

from onnx_helpers import export_model_to_onnx, sanity_check_onnx
from save_model import (
    EdgeMLResult,
    _make_run_dir,
    _setup_file_logger,
    save_model_artifacts,
)
from config import PARQUET_DIR, FEATURE_COLUMNS, TARGET_COLUMN, OUT_DIR
from plotting import save_all_standard_plots

logger = logging.getLogger(__name__)


def _pick_threshold_max_tpr_under_fpr(
    y_true: np.ndarray,
    p: np.ndarray,
    *,
    fpr_max: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Pick a probability threshold that maximizes TPR under a maximum FPR constraint.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (0/1).
    p : np.ndarray
        Predicted probabilities for the positive class.
    fpr_max : float
        Maximum allowed false positive rate.

    Returns
    -------
    (float, dict)
        Selected threshold and a small summary dict.
    """
    fpr, tpr, thr = roc_curve(y_true, p)

    # roc_curve returns thresholds descending, with thr[0]=inf sometimes.
    finite = np.isfinite(thr)
    fpr = fpr[finite]
    tpr = tpr[finite]
    thr = thr[finite]

    ok = fpr <= fpr_max
    if not np.any(ok):
        i = int(np.argmax(thr))
        return float(thr[i]), {"note": "no_threshold_met_fpr_max"}

    i = int(np.argmax(tpr[ok]))
    thr_ok = thr[ok]
    fpr_ok = fpr[ok]
    tpr_ok = tpr[ok]

    return float(thr_ok[i]), {
        "fpr_at_threshold": float(fpr_ok[i]),
        "tpr_at_threshold": float(tpr_ok[i]),
        "fpr_max": float(fpr_max),
        "note": "ok",
    }


def _metrics_at_threshold(
    y_true: np.ndarray, p: np.ndarray, thr: float
) -> Dict[str, float]:
    """
    Compute basic metrics given probabilities and a threshold.
    """
    y_pred = (p >= thr).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true, p)
    pos_rate = float(np.mean(y_true))

    # FPR/TPR at this threshold
    fpr, tpr, thresholds = roc_curve(y_true, p)
    idx = int(np.argmin(np.abs(thresholds - thr)))
    fpr_thr = float(fpr[idx])
    tpr_thr = float(tpr[idx])

    return {
        "auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pos_rate": float(pos_rate),
        "threshold": float(thr),
        "fpr": float(fpr_thr),
        "tpr": float(tpr_thr),
    }


def _log_split_summary(name: str, y: np.ndarray) -> None:
    """Log basic split summary (size, positives, rate)."""
    n = int(y.size)
    n_pos = int(np.sum(y == 1))
    rate = (n_pos / n) if n > 0 else float("nan")
    logger.info("%s: n=%d  pos=%d  pos_rate=%.4f", name, n, n_pos, rate)


def save_shap_plots_for_xgb(
    *,
    model: xgb.XGBClassifier,
    X: np.ndarray,
    feature_names: List[str],
    out_dir: Path,
    prefix: str = "",
    max_samples: int = 50_000,
    random_state: int = 42,
) -> None:
    """
    Save SHAP plots for an XGBoost model (TreeExplainer).

    Notes
    -----
    SHAP is computed on the *base* XGBoost model, not on the calibrated wrapper.
    This explains the tree model decision logic (ranking), which is what we want
    for sanity checks.
    """
    try:
        import shap  # type: ignore
    except Exception as e:
        logger.warning("SHAP is not installed, skipping SHAP plots. (%s)", e)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Subsample for speed/memory
    n = X.shape[0]
    if n > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=max_samples, replace=False)
        Xs = X[idx]
        logger.info("SHAP: subsampling %d/%d rows", Xs.shape[0], n)
    else:
        Xs = X
        logger.info("SHAP: using full set (%d rows)", n)

    logger.info("Computing SHAP values (TreeExplainer)...")
    t0 = time.perf_counter()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)

    logger.info("SHAP computed in %.2fs", time.perf_counter() - t0)

    # Summary bar plot (global importance)
    import matplotlib.pyplot as plt

    fig = plt.figure()
    shap.summary_plot(
        shap_values,
        Xs,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=min(25, len(feature_names)),
    )
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}shap_summary_bar.png", dpi=160)
    plt.close(fig)

    # Beeswarm plot (distribution of effects)
    fig = plt.figure()
    shap.summary_plot(
        shap_values,
        Xs,
        feature_names=feature_names,
        show=False,
        max_display=min(25, len(feature_names)),
    )
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}shap_summary_beeswarm.png", dpi=160)
    plt.close(fig)

    logger.info("SHAP plots saved to %s", out_dir)


def train_xgb_calibrated_thresholded(
    train_path: Path,
    val_path: Path,
    test_path: Optional[Path],
    feature_cols: List[str] = FEATURE_COLUMNS,
    target_col: str = TARGET_COLUMN,
    *,
    fpr_max: float = 0.01,
    random_state: int = 42,
    n_estimators: int = 400,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    calibrate_method: str = "isotonic",
    calibrate_cv: int = 3,
) -> EdgeMLResult:
    """
    Train a robust edge classifier pipeline: XGBoost + probability calibration + threshold selection.

    Notes
    -----
    Newer scikit-learn versions removed support for `cv="prefit"` in
    `CalibratedClassifierCV`. We therefore calibrate with CV on the validation
    set (the base estimator is refit on CV folds of `val`).
    """
    t0 = time.perf_counter()
    logger.info("Starting ML pipeline (xgboost + calibration + threshold)")
    logger.info("train=%s", train_path)
    logger.info("val  =%s", val_path)
    logger.info("test =%s", test_path if test_path else "<none>")
    logger.info("features (%d): %s", len(feature_cols), ", ".join(feature_cols))
    logger.info("target: %s", target_col)
    logger.info("threshold selection: maximize TPR with FPR <= %.4f", fpr_max)
    logger.info("calibration: method=%s cv=%s", calibrate_method, calibrate_cv)

    # Load data
    t_load = time.perf_counter()
    logger.info("Loading parquet splits...")
    df_tr = pd.read_parquet(train_path, columns=feature_cols + [target_col])
    df_va = pd.read_parquet(val_path, columns=feature_cols + [target_col])
    df_te = (
        pd.read_parquet(test_path, columns=feature_cols + [target_col])
        if test_path
        else None
    )
    logger.info(
        "Loaded dataframes: train=%s  val=%s  test=%s  (%.2fs)",
        df_tr.shape,
        df_va.shape,
        df_te.shape if df_te is not None else None,
        time.perf_counter() - t_load,
    )

    # Prepare arrays
    X_tr = df_tr[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_tr = df_tr[target_col].to_numpy(dtype=np.int8, copy=False)
    X_va = df_va[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_va = df_va[target_col].to_numpy(dtype=np.int8, copy=False)

    _log_split_summary("TRAIN", y_tr)
    _log_split_summary("VAL", y_va)

    # Imbalance
    n_pos = float(np.sum(y_tr == 1))
    n_neg = float(np.sum(y_tr == 0))
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    logger.info(
        "Class imbalance: n_pos=%.0f n_neg=%.0f scale_pos_weight=%.4f",
        n_pos,
        n_neg,
        scale_pos_weight,
    )

    # Train base model on TRAIN
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )

    logger.info("Fitting base XGBoost model on TRAIN...")
    t_fit = time.perf_counter()
    model.fit(X_tr, y_tr)
    logger.info("Base fit done (%.2fs)", time.perf_counter() - t_fit)

    # Calibration (CV on VAL)
    # Important: newer sklearn disallows cv="prefit"
    logger.info(
        "Calibrating on VAL with CV (method=%s, cv=%d)...",
        calibrate_method,
        calibrate_cv,
    )
    t_cal = time.perf_counter()
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=calibrate_method,
        cv=calibrate_cv,
        n_jobs=-1,
    )
    calibrated.fit(X_va, y_va)
    logger.info("Calibration done (%.2fs)", time.perf_counter() - t_cal)

    # Threshold selection on VAL (on calibrated proba)
    logger.info("Selecting threshold on VAL (FPR<=%.4f)...", fpr_max)
    p_va = calibrated.predict_proba(X_va)[:, 1]
    thr, thr_info = _pick_threshold_max_tpr_under_fpr(y_va, p_va, fpr_max=fpr_max)
    logger.info("Selected threshold: %.6f | info=%s", thr, thr_info)

    metrics_val = _metrics_at_threshold(y_va, p_va, thr)
    for k, v in thr_info.items():
        if isinstance(v, (int, float)):
            metrics_val[k] = float(v)

    logger.info(
        "VAL metrics: auc=%.5f precision=%.4f recall=%.4f f1=%.4f fpr=%.4f tpr=%.4f thr=%.6f",
        metrics_val["auc"],
        metrics_val["precision"],
        metrics_val["recall"],
        metrics_val["f1"],
        metrics_val["fpr"],
        metrics_val["tpr"],
        metrics_val["threshold"],
    )

    metrics_test = None
    if df_te is not None:
        logger.info("Computing test metrics...")
        X_te = df_te[feature_cols].to_numpy(dtype=np.float32, copy=False)
        y_te = df_te[target_col].to_numpy(dtype=np.int8, copy=False)
        p_te = calibrated.predict_proba(X_te)[:, 1]
        metrics_test = _metrics_at_threshold(y_te, p_te, thr)
        logger.info(
            "TEST metrics: auc=%.5f precision=%.4f recall=%.4f f1=%.4f fpr=%.4f tpr=%.4f thr=%.6f",
            metrics_test["auc"],
            metrics_test["precision"],
            metrics_test["recall"],
            metrics_test["f1"],
            metrics_test["fpr"],
            metrics_test["tpr"],
            metrics_test["threshold"],
        )

    logger.info("Pipeline finished in %.2fs", time.perf_counter() - t0)

    return EdgeMLResult(
        model=model,
        calibrated=calibrated,
        threshold=thr,
        metrics_val=metrics_val,
        metrics_test=metrics_test,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    train_path = PARQUET_DIR / "train.parquet"
    val_path = PARQUET_DIR / "val.parquet"
    test_path = PARQUET_DIR / "test.parquet"

    # Create a versioned run directory and log file
    run_dir = _make_run_dir(OUT_DIR, FEATURE_COLUMNS, tag="edge_xgb")
    _setup_file_logger(run_dir)
    logger.info("Run directory: %s", run_dir)

    # Train
    res = train_xgb_calibrated_thresholded(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        feature_cols=FEATURE_COLUMNS,
        target_col=TARGET_COLUMN,
        fpr_max=0.01,
        calibrate_method="isotonic",
        calibrate_cv=3,
    )

    logger.info("Threshold: %.16g", res.threshold)
    logger.info("VAL metrics: %s", res.metrics_val)
    logger.info("TEST metrics: %s", res.metrics_test)

    # -----------------------
    # Load splits once for plotting + metadata
    # -----------------------
    logger.info("Loading splits for plots + metadata...")
    df_tr = pd.read_parquet(train_path, columns=FEATURE_COLUMNS + [TARGET_COLUMN])
    df_va = pd.read_parquet(val_path, columns=FEATURE_COLUMNS + [TARGET_COLUMN])
    df_te = pd.read_parquet(test_path, columns=FEATURE_COLUMNS + [TARGET_COLUMN])

    # Basic stats (pos_rate)
    y_tr = df_tr[TARGET_COLUMN].to_numpy(dtype=np.int8, copy=False)
    y_va = df_va[TARGET_COLUMN].to_numpy(dtype=np.int8, copy=False)
    y_te = df_te[TARGET_COLUMN].to_numpy(dtype=np.int8, copy=False)

    train_pos_rate = float(np.mean(y_tr))
    val_pos_rate = float(np.mean(y_va))
    test_pos_rate = float(np.mean(y_te))

    # Prepare TEST arrays for plots
    X_te = df_te[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False)

    # -----------------------
    # Save plots to disk (versioned)
    # -----------------------
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Writing standard plots to %s", plots_dir)
    save_all_standard_plots(
        model=res.calibrated,
        X_test=X_te,
        y_test=y_te,
        feature_names=FEATURE_COLUMNS,
        out_dir=plots_dir,
        prefix="test_",
        threshold=res.threshold,
    )

    # SHAP plots (on base XGB model)
    save_shap_plots_for_xgb(
        model=res.model,
        X=X_te,
        feature_names=list(FEATURE_COLUMNS),
        out_dir=plots_dir,
        prefix="test_",
        max_samples=50_000,
        random_state=42,
    )

    # -----------------------
    # Export ONNX + sanity check (versioned)
    # -----------------------
    onnx_dir = run_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "model.onnx"

    logger.info("Exporting base XGBoost model to ONNX: %s", onnx_path)
    t_onnx = time.perf_counter()
    export_model_to_onnx(
        res.model,  # base XGB
        n_features=len(FEATURE_COLUMNS),
        out_path=onnx_path,
    )
    logger.info("ONNX export done (%.2fs)", time.perf_counter() - t_onnx)

    logger.info("Running ONNX sanity check...")
    t_chk = time.perf_counter()
    onnx_report = sanity_check_onnx(
        res.model,  # compare ONNX to base model
        onnx_path,
        X_te,  # test distribution
        n_samples=2000,  # tune if you want
        seed=42,
    )
    logger.info(
        "ONNX sanity check done (%.2fs): %s", time.perf_counter() - t_chk, onnx_report
    )

    # Save report next to ONNX model
    import json

    (onnx_dir / "onnx_sanity.json").write_text(
        json.dumps(onnx_report, indent=2) + "\n", encoding="utf-8"
    )

    # -----------------------
    # Save versioned artifacts (model + metadata)
    # -----------------------
    params = {
        "fpr_max": 0.01,
        "random_state": 42,
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "min_child_weight": 1.0,
        "calibrate_method": "isotonic",
        "calibrate_cv": 3,
        "xgb_tree_method": "hist",
        "xgb_objective": "binary:logistic",
        "xgb_eval_metric": "logloss",
    }

    save_model_artifacts(
        run_dir=run_dir,
        res=res,
        feature_cols=list(FEATURE_COLUMNS),
        target_col=TARGET_COLUMN,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        train_shape=df_tr.shape,
        val_shape=df_va.shape,
        test_shape=df_te.shape,
        train_pos_rate=train_pos_rate,
        val_pos_rate=val_pos_rate,
        test_pos_rate=test_pos_rate,
        params=params,
        extra_meta={
            "onnx": {
                "path": str(onnx_path),
                "sanity": onnx_report,
            }
        },
    )

    logger.info("Done. Run artifacts available in %s", run_dir)
