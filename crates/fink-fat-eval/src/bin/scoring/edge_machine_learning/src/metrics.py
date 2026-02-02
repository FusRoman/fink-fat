# edge_ml/metrics.py

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
    log_loss,
)


def eval_classification(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict:
    """
    Evaluate classification quality for probabilistic binary predictions.

    Notes
    -----
    - Adds calibration-sensitive metrics: Brier score and log loss.
    - Clips probabilities to avoid infinite log loss when models output exact 0/1.
    """
    from sklearn.metrics import brier_score_loss, log_loss

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    # Avoid logloss inf on exact 0/1 (can happen with tree ensembles)
    eps = 1e-15
    y_proba_clip = np.clip(y_proba, eps, 1.0 - eps)

    y_pred = (y_proba >= threshold).astype(int)

    out: dict = {}

    # Calibration/probabilistic metrics (lower is better)
    out["brier"] = float(brier_score_loss(y_true, y_proba))
    out["logloss"] = float(
        log_loss(y_true, np.c_[1.0 - y_proba_clip, y_proba_clip], labels=[0, 1])
    )

    # Ranking metrics (higher is better); undefined if y_true is single-class
    out["roc_auc"] = (
        roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float("nan")
    )
    out["pr_auc"] = (
        average_precision_score(y_true, y_proba)
        if len(np.unique(y_true)) > 1
        else float("nan")
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})

    out["report"] = classification_report(y_true, y_pred, digits=4, zero_division=0)
    return out


def eval_hit_at_k(
    df: pd.DataFrame,
    *,
    group_col: str,
    label_col: str,
    score_col: str,
    ks=(1, 5, 10),
) -> dict:
    if group_col not in df.columns:
        return {"hit_at_k": None, "note": f"missing column: {group_col}"}

    g = df.groupby(group_col, sort=False)
    groups = []
    for _, sub in g:
        if sub[label_col].sum() > 0:
            groups.append(sub)

    if not groups:
        return {"hit_at_k": None, "note": "no groups with positive labels"}

    ks = tuple(int(k) for k in ks)
    hits = {k: 0 for k in ks}
    n_groups = 0

    for sub in groups:
        sub = sub.sort_values(score_col, ascending=False)
        y = sub[label_col].to_numpy()
        n_groups += 1

        n = len(y)
        for k in ks:
            kk = min(k, n)  # clamp, but keep original key k
            if kk > 0 and y[:kk].max() == 1:
                hits[k] += 1

    sizes = np.array([len(sub) for sub in groups], dtype=int)
    return {
        "hit_at_k": {k: hits[k] / n_groups for k in ks},
        "n_groups": n_groups,
        "group_size": {
            "min": int(sizes.min()),
            "p50": int(np.median(sizes)),
            "p90": int(np.quantile(sizes, 0.90)),
            "max": int(sizes.max()),
        },
    }
