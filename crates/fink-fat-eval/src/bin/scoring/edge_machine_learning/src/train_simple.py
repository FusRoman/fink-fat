from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import time
import logging

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

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


# -----------------------------
# Utilities
# -----------------------------
def parquet_pos_rate(path: Path, target_col: str) -> float:
    """
    Compute positive rate by scanning a single Parquet column.

    This avoids loading the whole dataset in memory.

    Parameters
    ----------
    path : Path
        Path to a Parquet file.
    target_col : str
        Name of the binary target column (0/1).

    Returns
    -------
    float
        Mean of the target column (positive rate).
    """
    import pyarrow.dataset as ds
    import pyarrow.compute as pc

    dataset = ds.dataset(str(path), format="parquet")
    table = dataset.to_table(columns=[target_col])
    arr = table[target_col]
    s = pc.sum(arr).as_py()
    n = arr.length()
    return float(s) / float(n) if n else float("nan")


def _log_split_summary(name: str, y: np.ndarray) -> None:
    """Log basic split summary (size, positives, rate)."""
    n = int(y.size)
    n_pos = int(np.sum(y == 1))
    rate = (n_pos / n) if n > 0 else float("nan")
    logger.info("%s: n=%d  pos=%d  pos_rate=%.6f", name, n, n_pos, rate)


def _pick_threshold_max_tpr_under_fpr(
    y_true: np.ndarray,
    score: np.ndarray,
    *,
    fpr_max: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Pick a score threshold that maximizes TPR under a maximum FPR constraint.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (0/1).
    score : np.ndarray
        Ranking score (can be probability, margin, etc.) where higher means "more positive".
    fpr_max : float
        Maximum allowed false positive rate.

    Returns
    -------
    (float, dict)
        Selected threshold and a small summary dict.
    """
    fpr, tpr, thr = roc_curve(y_true, score)

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
    y_true: np.ndarray, score: np.ndarray, thr: float
) -> Dict[str, float]:
    """
    Compute basic metrics given a ranking score and a threshold.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (0/1).
    score : np.ndarray
        Ranking score (higher => more positive).
    thr : float
        Threshold applied to `score`.

    Returns
    -------
    dict
        Dict with AUC, precision, recall, f1, FPR/TPR, etc.
    """
    y_pred = (score >= thr).astype(np.int8)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    auc = roc_auc_score(y_true, score)
    ap = average_precision_score(y_true, score)
    pos_rate = float(np.mean(y_true))

    fpr, tpr, thresholds = roc_curve(y_true, score)
    idx = int(np.argmin(np.abs(thresholds - thr)))
    fpr_thr = float(fpr[idx])
    tpr_thr = float(tpr[idx])

    return {
        "auc": float(auc),
        "ap": float(ap),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pos_rate": float(pos_rate),
        "threshold": float(thr),
        "fpr": float(fpr_thr),
        "tpr": float(tpr_thr),
    }


def _score_model(
    model: Union[xgb.XGBClassifier, CalibratedClassifierCV],
    X: np.ndarray,
    *,
    score_kind: str = "proba",
) -> np.ndarray:
    """
    Compute a ranking score for binary classification.

    Parameters
    ----------
    model : XGBClassifier or CalibratedClassifierCV
        Trained model.
    X : np.ndarray
        Feature matrix.
    score_kind : {"proba", "margin"}
        - "proba": use predict_proba[:, 1]
        - "margin": use raw margin (logit) if available (XGBoost only)

    Returns
    -------
    np.ndarray
        Score vector (higher => more positive).
    """
    if score_kind not in ("proba", "margin"):
        raise ValueError(f"score_kind must be 'proba' or 'margin', got {score_kind}")

    # Calibrated models expose predict_proba, but not margin.
    if isinstance(model, CalibratedClassifierCV) or score_kind == "proba":
        return model.predict_proba(X)[:, 1].astype(np.float64, copy=False)

    # XGBoost margin (logit)
    # This is often a better *ranking score* than probability.
    return model.predict(X, output_margin=True).astype(np.float64, copy=False)


# -----------------------------
# Top-k / ranking metrics
# -----------------------------
def _topk_global_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
    *,
    k: int,
) -> Dict[str, float]:
    """
    Compute top-K metrics globally over a split.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (0/1).
    score : np.ndarray
        Ranking score (higher => more positive).
    k : int
        Number of edges to keep globally.

    Returns
    -------
    dict
        precision@k, recall@k, k, etc.
    """
    n = int(y_true.size)
    k_eff = int(min(max(k, 1), n))

    order = np.argsort(score)[::-1]
    top = order[:k_eff]

    tp = int(np.sum(y_true[top] == 1))
    total_pos = int(np.sum(y_true == 1))

    precision_k = tp / k_eff if k_eff > 0 else 0.0
    recall_k = tp / total_pos if total_pos > 0 else 0.0

    return {
        "topk_global_k": float(k_eff),
        "topk_global_tp": float(tp),
        "topk_global_precision": float(precision_k),
        "topk_global_recall": float(recall_k),
        "topk_global_total_pos": float(total_pos),
        "n": float(n),
    }


def _topk_per_from_seed_metrics(
    df: pd.DataFrame,
    *,
    label_col: str,
    score_col: str,
    from_id_col: str = "from_seed_id",
    k_per_from: int = 20,
) -> Dict[str, float]:
    """
    Compute top-k metrics per `from_seed_id` (keep k outgoing edges per seed).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `from_id_col`, `label_col`, `score_col`.
    label_col : str
        Name of binary target column.
    score_col : str
        Name of score column.
    from_id_col : str, default "from_seed_id"
        Seed identifier for outgoing edge grouping.
    k_per_from : int, default 20
        Keep at most k edges per from_seed_id.

    Returns
    -------
    dict
        Recall/precision for edges kept after per-seed top-k filtering, plus a hit-rate
        at the seed level.
    """
    if from_id_col not in df.columns:
        return {
            "topk_from_note": f"missing_column:{from_id_col}",
        }

    k = int(max(k_per_from, 1))

    # Sort within groups and take head(k)
    df_sorted = df.sort_values([from_id_col, score_col], ascending=[True, False])
    kept = df_sorted.groupby(from_id_col, sort=False).head(k)

    y_all = df[label_col].to_numpy(dtype=np.int8, copy=False)
    y_kept = kept[label_col].to_numpy(dtype=np.int8, copy=False)

    total_pos = int(np.sum(y_all == 1))
    tp = int(np.sum(y_kept == 1))
    n_kept = int(y_kept.size)

    precision = tp / n_kept if n_kept > 0 else 0.0
    recall = tp / total_pos if total_pos > 0 else 0.0

    # Seed-level hit-rate: fraction of from_seed_id for which we kept at least one true edge
    # among its top-k.
    # (Only meaningful if true edges exist per from_seed_id.)
    per_from_true = kept.groupby(from_id_col)[label_col].max()
    from_hit_rate = float(per_from_true.mean()) if per_from_true.size > 0 else 0.0

    return {
        "topk_from_k": float(k),
        "topk_from_kept_edges": float(n_kept),
        "topk_from_tp": float(tp),
        "topk_from_precision": float(precision),
        "topk_from_recall": float(recall),
        "topk_from_total_pos": float(total_pos),
        "topk_from_seed_hit_rate": float(from_hit_rate),
        "topk_from_n_from_seeds": float(per_from_true.size),
    }


# -----------------------------
# SHAP (unchanged)
# -----------------------------
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
    SHAP is computed on the *base* XGBoost model, not on a calibrated wrapper.
    """
    try:
        import shap  # type: ignore
    except Exception as e:
        logger.warning("SHAP is not installed, skipping SHAP plots. (%s)", e)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

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


# -----------------------------
# Training entry point
# -----------------------------
def train_xgb_ranking_pipeline(
    train_path: Path,
    val_path: Path,
    test_path: Optional[Path],
    feature_cols: List[str] = FEATURE_COLUMNS,
    target_col: str = TARGET_COLUMN,
    *,
    # Decision strategy params
    use_calibration: bool = False,
    calibrate_method: str = "sigmoid",
    calibrate_cv: int = 3,
    score_kind: str = "proba",  # "proba" or "margin" (margin only for base XGB, not calibrated)
    # Reference thresholding (optional, mostly for monitoring)
    compute_fpr_threshold: bool = True,
    fpr_max: float = 0.01,
    # Top-k metrics
    topk_global: int = 50_000,
    topk_per_from: int = 20,
    from_id_col: str = "from_seed_id",
    # XGB params
    random_state: int = 42,
    n_estimators: int = 400,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
) -> Tuple[EdgeMLResult, Tuple[int, int], Tuple[int, int]]:
    """
    Train a robust edge ranking model: XGBoost (+ optional calibration) + ranking metrics.

    Parameters
    ----------
    train_path, val_path, test_path : Path
        Parquet splits.
    feature_cols : list of str
        Feature columns.
    target_col : str
        Binary label.
    use_calibration : bool, default False
        If True, fit a CalibratedClassifierCV on VAL. If False, use base XGB scores directly.
    calibrate_method : str, default "sigmoid"
        Calibration method ("sigmoid" or "isotonic"). "sigmoid" is usually more robust under shift.
    calibrate_cv : int, default 3
        CV folds for calibration on VAL (sklearn will refit base estimator on folds).
    score_kind : {"proba","margin"}, default "proba"
        Ranking score. "margin" uses raw logit and is often great for ranking, but not available when calibrated.
    compute_fpr_threshold : bool, default True
        If True, compute and log a reference threshold on VAL under an FPR constraint.
        This is mainly for monitoring; decision should be top-k in production.
    fpr_max : float, default 0.01
        FPR constraint used for the reference threshold.
    topk_global : int, default 50_000
        Global top-K metrics computed on VAL/TEST.
    topk_per_from : int, default 20
        Per-seed top-k metrics computed on VAL/TEST (requires `from_id_col` in parquet).
    from_id_col : str, default "from_seed_id"
        Column used to group outgoing edges.

    Returns
    -------
    (EdgeMLResult, (train_rows, train_cols), (val_rows, val_cols))
        The result container and shapes for metadata.
    """
    t0 = time.perf_counter()

    logger.info("Starting ranking pipeline (xgboost + optional calibration)")
    logger.info("train=%s", train_path)
    logger.info("val  =%s", val_path)
    logger.info("test =%s", test_path if test_path else "<none>")
    logger.info("features (%d): %s", len(feature_cols), ", ".join(feature_cols))
    logger.info("target: %s", target_col)
    logger.info(
        "use_calibration=%s method=%s cv=%s",
        use_calibration,
        calibrate_method,
        calibrate_cv,
    )
    logger.info("score_kind=%s", score_kind)
    logger.info(
        "topk_global=%d topk_per_from=%d from_id_col=%s",
        topk_global,
        topk_per_from,
        from_id_col,
    )

    # Load splits: train needs only features+label; val/test need extra ids for top-k metrics (if available).
    t_load = time.perf_counter()
    logger.info("Loading parquet splits...")

    df_tr = pd.read_parquet(train_path, columns=feature_cols + [target_col])

    val_cols = feature_cols + [target_col]
    test_cols = feature_cols + [target_col]

    # for top-k per from_seed_id metrics, load the id column if present
    # (if it's absent, metrics function will emit a note)
    # NOTE: if your parquet does not contain this column, nothing breaks.
    df_va = pd.read_parquet(
        val_path, columns=val_cols + [from_id_col] if from_id_col else val_cols
    )

    df_te = None
    if test_path:
        df_te = pd.read_parquet(
            test_path, columns=test_cols + [from_id_col] if from_id_col else test_cols
        )

    logger.info(
        "Loaded dataframes: train=%s  val=%s  test=%s  (%.2fs)",
        df_tr.shape,
        df_va.shape,
        df_te.shape if df_te is not None else None,
        time.perf_counter() - t_load,
    )

    # Arrays
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

    # Fit base model
    base = xgb.XGBClassifier(
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
    base.fit(X_tr, y_tr)
    logger.info("Base fit done (%.2fs)", time.perf_counter() - t_fit)

    # Optional calibration on VAL
    if use_calibration:
        # Margin is not available on CalibratedClassifierCV
        if score_kind == "margin":
            logger.warning(
                "score_kind='margin' is not supported with calibration; switching to 'proba'."
            )
            score_kind = "proba"

        logger.info(
            "Calibrating on VAL with CV (method=%s, cv=%d)...",
            calibrate_method,
            calibrate_cv,
        )
        t_cal = time.perf_counter()
        calibrated = CalibratedClassifierCV(
            estimator=base,
            method=calibrate_method,
            cv=calibrate_cv,
            n_jobs=-1,
        )
        calibrated.fit(X_va, y_va)
        logger.info("Calibration done (%.2fs)", time.perf_counter() - t_cal)
        final_model: Union[xgb.XGBClassifier, CalibratedClassifierCV] = calibrated
    else:
        final_model = base

    # Scores on VAL (ranking)
    s_va = _score_model(final_model, X_va, score_kind=score_kind)

    # Core ranking metrics
    metrics_val: Dict[str, float] = {
        "auc": float(roc_auc_score(y_va, s_va)),
        "ap": float(average_precision_score(y_va, s_va)),
        "pos_rate": float(np.mean(y_va)),
    }

    # Reference threshold under FPR constraint (optional)
    thr = 0.5
    if compute_fpr_threshold:
        logger.info(
            "Computing reference threshold on VAL (maximize TPR with FPR<=%.4f)...",
            fpr_max,
        )
        thr, thr_info = _pick_threshold_max_tpr_under_fpr(y_va, s_va, fpr_max=fpr_max)
        metrics_thr = _metrics_at_threshold(y_va, s_va, thr)
        metrics_val.update({f"thr_{k}": float(v) for k, v in metrics_thr.items()})
        for k, v in thr_info.items():
            if isinstance(v, (int, float)):
                metrics_val[f"thr_{k}"] = float(v)
        logger.info("Reference threshold (VAL): %.6f | info=%s", thr, thr_info)
        logger.info(
            "VAL thr-metrics: auc=%.5f ap=%.5f precision=%.4f recall=%.4f fpr=%.4f tpr=%.4f thr=%.6f",
            metrics_thr["auc"],
            metrics_thr["ap"],
            metrics_thr["precision"],
            metrics_thr["recall"],
            metrics_thr["fpr"],
            metrics_thr["tpr"],
            metrics_thr["threshold"],
        )
    else:
        logger.info(
            "Skipping reference threshold computation (compute_fpr_threshold=False)."
        )

    # Top-k metrics on VAL
    metrics_val.update(_topk_global_metrics(y_va, s_va, k=topk_global))
    df_va_scored = df_va[
        [target_col] + ([from_id_col] if from_id_col in df_va.columns else [])
    ].copy()
    df_va_scored["_score"] = s_va
    metrics_val.update(
        _topk_per_from_seed_metrics(
            df_va_scored,
            label_col=target_col,
            score_col="_score",
            from_id_col=from_id_col,
            k_per_from=topk_per_from,
        )
    )

    logger.info(
        "VAL ranking: auc=%.5f ap=%.5f | topk_global_recall=%.4f topk_from_recall=%.4f",
        metrics_val["auc"],
        metrics_val["ap"],
        metrics_val.get("topk_global_recall", float("nan")),
        metrics_val.get("topk_from_recall", float("nan")),
    )

    # TEST metrics (ranking + top-k)
    metrics_test: Optional[Dict[str, float]] = None
    if df_te is not None:
        X_te = df_te[feature_cols].to_numpy(dtype=np.float32, copy=False)
        y_te = df_te[target_col].to_numpy(dtype=np.int8, copy=False)
        s_te = _score_model(final_model, X_te, score_kind=score_kind)

        metrics_test = {
            "auc": float(roc_auc_score(y_te, s_te)),
            "ap": float(average_precision_score(y_te, s_te)),
            "pos_rate": float(np.mean(y_te)),
        }
        # apply the reference threshold if computed (for monitoring only)
        if compute_fpr_threshold:
            metrics_thr_te = _metrics_at_threshold(y_te, s_te, thr)
            metrics_test.update(
                {f"thr_{k}": float(v) for k, v in metrics_thr_te.items()}
            )

        metrics_test.update(_topk_global_metrics(y_te, s_te, k=topk_global))
        df_te_scored = df_te[
            [target_col] + ([from_id_col] if from_id_col in df_te.columns else [])
        ].copy()
        df_te_scored["_score"] = s_te
        metrics_test.update(
            _topk_per_from_seed_metrics(
                df_te_scored,
                label_col=target_col,
                score_col="_score",
                from_id_col=from_id_col,
                k_per_from=topk_per_from,
            )
        )

        logger.info(
            "TEST ranking: auc=%.5f ap=%.5f | topk_global_recall=%.4f topk_from_recall=%.4f",
            metrics_test["auc"],
            metrics_test["ap"],
            metrics_test.get("topk_global_recall", float("nan")),
            metrics_test.get("topk_from_recall", float("nan")),
        )

    logger.info("Pipeline finished in %.2fs", time.perf_counter() - t0)

    # For downstream code compatibility:
    # - keep EdgeMLResult.calibrated as the "model used for scoring"
    # - keep EdgeMLResult.threshold as the reference threshold (monitoring), not the production decision
    calibrated_for_api = final_model  # could be base or calibrated
    return (
        EdgeMLResult(
            model=base,  # always keep base model for ONNX export + SHAP
            calibrated=calibrated_for_api,  # scoring model
            threshold=float(thr),
            metrics_val=metrics_val,
            metrics_test=metrics_test,
        ),
        df_tr.shape,
        df_va.shape,
    )


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    train_path = PARQUET_DIR / "train.parquet"
    val_path = PARQUET_DIR / "val.parquet"
    test_path = PARQUET_DIR / "test.parquet"

    # -----------------------
    # Top-level decision config
    # -----------------------
    USE_CALIBRATION = False  # <--- 1) top-level toggle
    CALIBRATE_METHOD = "sigmoid"  # "sigmoid" recommended under shift
    CALIBRATE_CV = 3

    SCORE_KIND = "proba"  # "proba" or "margin" (margin only if USE_CALIBRATION=False)

    TOPK_GLOBAL = 50_000  # global top-K metrics
    TOPK_PER_FROM = 20  # per from_seed_id top-k metrics
    FROM_ID_COL = "from_seed_id"

    COMPUTE_FPR_THRESHOLD = True  # keep as monitoring signal
    FPR_MAX = 0.01

    # Create a versioned run directory and log file
    run_dir = _make_run_dir(OUT_DIR, FEATURE_COLUMNS, tag="edge_xgb")
    _setup_file_logger(run_dir)
    logger.info("Run directory: %s", run_dir)

    # Train
    res, train_shape, val_shape = train_xgb_ranking_pipeline(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        feature_cols=FEATURE_COLUMNS,
        target_col=TARGET_COLUMN,
        use_calibration=USE_CALIBRATION,
        calibrate_method=CALIBRATE_METHOD,
        calibrate_cv=CALIBRATE_CV,
        score_kind=SCORE_KIND,
        compute_fpr_threshold=COMPUTE_FPR_THRESHOLD,
        fpr_max=FPR_MAX,
        topk_global=TOPK_GLOBAL,
        topk_per_from=TOPK_PER_FROM,
        from_id_col=FROM_ID_COL,
    )

    logger.info("Reference threshold (monitoring): %.16g", res.threshold)
    logger.info("VAL metrics: %s", res.metrics_val)
    logger.info("TEST metrics: %s", res.metrics_test)

    # -----------------------
    # Load TEST split for plots + metadata (avoid reloading train/val)
    # -----------------------
    logger.info("Loading TEST split for plots + metadata...")
    df_te = pd.read_parquet(test_path, columns=FEATURE_COLUMNS + [TARGET_COLUMN])

    y_te = df_te[TARGET_COLUMN].to_numpy(dtype=np.int8, copy=False)
    X_te = df_te[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False)

    train_pos_rate = parquet_pos_rate(train_path, TARGET_COLUMN)
    val_pos_rate = parquet_pos_rate(val_path, TARGET_COLUMN)
    test_pos_rate = parquet_pos_rate(test_path, TARGET_COLUMN)

    # -----------------------
    # Save plots to disk (versioned)
    # -----------------------
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # For plots, we pass the scoring model (base or calibrated).
    # Threshold is a "reference threshold" for monitoring only.
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

    # SHAP plots (always on base XGB model)
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
        n_samples=2000,
        seed=42,
    )
    logger.info(
        "ONNX sanity check done (%.2fs): %s", time.perf_counter() - t_chk, onnx_report
    )

    import json

    (onnx_dir / "onnx_sanity.json").write_text(
        json.dumps(onnx_report, indent=2) + "\n", encoding="utf-8"
    )

    # -----------------------
    # Save versioned artifacts (model + metadata)
    # -----------------------
    params = {
        "use_calibration": USE_CALIBRATION,
        "calibrate_method": CALIBRATE_METHOD,
        "calibrate_cv": CALIBRATE_CV,
        "score_kind": SCORE_KIND,
        "topk_global": TOPK_GLOBAL,
        "topk_per_from": TOPK_PER_FROM,
        "from_id_col": FROM_ID_COL,
        "compute_fpr_threshold": COMPUTE_FPR_THRESHOLD,
        "fpr_max": FPR_MAX,
        "random_state": 42,
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "min_child_weight": 1.0,
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
        train_shape=train_shape,
        val_shape=val_shape,
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
