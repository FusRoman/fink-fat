# edge_ml/model_evaluation/plotting.py — post-training model quality plots

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier


def _savefig(fig: plt.Figure, out_dir: Path, name: str, dpi: int = 140) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=dpi)
    plt.close(fig)


# ── Individual plots ──────────────────────────────────────────────────────────


def plot_roc_pr_curves(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    *,
    out_dir: Path,
) -> None:
    """ROC curve and Precision-Recall curve side by side."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(rec, prec)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(fpr, tpr, color="#1565C0", lw=2, label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(rec, prec, color="#2E7D32", lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax.axhline(
        y_test.mean(), color="k", linestyle="--", lw=1,
        label=f"Baseline (prevalence) = {y_test.mean():.3f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Model discrimination — ROC & PR", fontsize=13)
    _savefig(fig, out_dir, "roc_pr_curves.png")


def plot_score_distribution(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    *,
    out_dir: Path,
    bins: int = 80,
) -> None:
    """Histogram of predicted probabilities, split by true label."""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {0: "#F44336", 1: "#4CAF50"}
    labels = {0: "FP (0)", 1: "TP (1)"}
    for cls in [0, 1]:
        ax.hist(
            y_proba[y_test == cls], bins=bins, range=(0, 1),
            density=True, alpha=0.6, color=colors[cls], label=labels[cls],
        )
    ax.set_xlabel("Predicted probability P(TP)")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution — TP vs FP")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, out_dir, "score_distribution.png")


def plot_threshold_analysis(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    *,
    out_dir: Path,
) -> None:
    """Precision, Recall, and F1 as a function of the decision threshold."""
    prec, rec, thresholds = precision_recall_curve(y_test, y_proba)
    # precision_recall_curve appends a sentinel — align arrays to equal length
    thresholds_full = np.append(thresholds, 1.0)
    denom = prec + rec
    f1 = np.where(denom > 0, 2 * prec * rec / denom, 0.0)
    best_idx = int(np.argmax(f1))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds_full, prec, label="Precision", color="#1565C0", lw=1.5)
    ax.plot(thresholds_full, rec, label="Recall", color="#2E7D32", lw=1.5)
    ax.plot(thresholds_full, f1, label="F1", color="#E65100", lw=2)
    ax.axvline(
        thresholds_full[best_idx], color="k", linestyle="--", lw=1.2,
        label=f"Best F1 threshold = {thresholds_full[best_idx]:.3f}  (F1 = {f1[best_idx]:.4f})",
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    _savefig(fig, out_dir, "threshold_analysis.png")


def plot_confusion_matrices(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    *,
    out_dir: Path,
) -> None:
    """
    Side-by-side confusion matrices at threshold=0.5 and at the threshold
    that maximises F1 on the test set.
    """
    prec, rec, thresholds = precision_recall_curve(y_test, y_proba)
    denom = prec[:-1] + rec[:-1]
    f1 = np.where(denom > 0, 2 * prec[:-1] * rec[:-1] / denom, 0.0)
    best_thresh = float(thresholds[np.argmax(f1)])

    pairs = [("Default (0.50)", 0.5), (f"Best F1 ({best_thresh:.3f})", best_thresh)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (title, t) in zip(axes, pairs):
        y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        thresh_val = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=13, color="white" if cm[i, j] > thresh_val else "black",
                )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred FP", "Pred TP"])
        ax.set_yticklabels(["True FP", "True TP"])
        ax.set_title(title)
        ax.set_xlabel(f"F1 = {f1_score(y_test, y_pred):.4f}")

    fig.suptitle("Confusion matrices", fontsize=13)
    _savefig(fig, out_dir, "confusion_matrices.png")


def plot_feature_importance(
    model: XGBClassifier,
    feature_names: Sequence[str],
    *,
    top_n: int = 20,
    out_dir: Path,
) -> None:
    """
    Horizontal bar charts for the three XGBoost importance types:
    gain (avg. improvement per split), cover (avg. samples covered),
    and weight (number of times used in splits).
    """
    booster = model.get_booster()
    importance_types = [
        ("gain",   "Gain (avg. improvement per split)", "#1565C0"),
        ("cover",  "Cover (avg. samples per split)",    "#2E7D32"),
        ("weight", "Weight (# splits)",                 "#E65100"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19, max(5, top_n * 0.42)))

    for ax, (imp_type, xlabel, color) in zip(axes, importance_types):
        scores = booster.get_score(importance_type=imp_type)
        # Map f0/f1/… back to actual feature names when no names were set
        named: dict[str, float] = {}
        for k, v in scores.items():
            if k.startswith("f") and k[1:].isdigit():
                idx = int(k[1:])
                named[feature_names[idx] if idx < len(feature_names) else k] = v
            else:
                named[k] = v

        sorted_items = sorted(named.items(), key=lambda x: x[1], reverse=True)[:top_n]
        if not sorted_items:
            ax.set_visible(False)
            continue

        names, values = zip(*sorted_items)
        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, color=color, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(f"Feature importance — {imp_type}")
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("XGBoost feature importance", fontsize=13)
    _savefig(fig, out_dir, "feature_importance.png", dpi=120)


def plot_learning_curve(
    model: XGBClassifier,
    *,
    out_dir: Path,
) -> None:
    """
    Eval-set metric over boosting rounds with the best iteration marked.
    Detects overfitting and validates early stopping.
    """
    results = model.evals_result()
    if not results:
        return
    eval_key = next(iter(results))
    metric_key = next(iter(results[eval_key]))
    scores = np.array(results[eval_key][metric_key])
    rounds = np.arange(1, len(scores) + 1)
    best_iter = int(np.argmin(scores))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rounds, scores, color="#1565C0", lw=1.5, label=f"Eval {metric_key}")
    ax.axvline(
        best_iter + 1, color="#E65100", linestyle="--", lw=1.5,
        label=f"Best iteration = {best_iter + 1}  ({metric_key} = {scores[best_iter]:.5f})",
    )
    ax.set_xlabel("Boosting round")
    ax.set_ylabel(metric_key)
    ax.set_title(f"Learning curve — eval {metric_key}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, out_dir, "learning_curve.png")


def plot_calibration(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    *,
    out_dir: Path,
    n_bins: int = 20,
) -> None:
    """
    Reliability diagram: mean predicted probability per bin vs actual fraction
    of positives. A perfectly calibrated model lies on the diagonal.
    """
    frac_pos, mean_pred = calibration_curve(
        y_test, y_proba, n_bins=n_bins, strategy="uniform"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(mean_pred, frac_pos, "o-", color="#1565C0", lw=2, label="Model")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability diagram")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(y_proba, bins=50, range=(0, 1), color="steelblue", alpha=0.8)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of predicted scores")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, alpha=0.3)

    fig.suptitle("Model calibration", fontsize=13)
    _savefig(fig, out_dir, "calibration.png")


# ── Entry point ───────────────────────────────────────────────────────────────


def plot_all(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Sequence[str],
    out_dir: str | Path,
) -> None:
    """
    Generate the full model evaluation plot suite and save to ``out_dir``.

    Plots produced
    --------------
    roc_pr_curves.png      — ROC and PR curves
    score_distribution.png — Predicted P(TP) split by true label
    threshold_analysis.png — Precision / Recall / F1 vs decision threshold
    confusion_matrices.png — At default 0.5 and best-F1 thresholds
    feature_importance.png — Gain / cover / weight (top-20)
    learning_curve.png     — Eval logloss per boosting round
    calibration.png        — Reliability diagram + score histogram
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    y_proba = model.predict_proba(X_test)[:, 1]

    print("  roc_pr_curves …")
    plot_roc_pr_curves(y_test, y_proba, out_dir=out)

    print("  score_distribution …")
    plot_score_distribution(y_test, y_proba, out_dir=out)

    print("  threshold_analysis …")
    plot_threshold_analysis(y_test, y_proba, out_dir=out)

    print("  confusion_matrices …")
    plot_confusion_matrices(y_test, y_proba, out_dir=out)

    print("  feature_importance …")
    plot_feature_importance(model, feature_names, out_dir=out)

    print("  learning_curve …")
    plot_learning_curve(model, out_dir=out)

    print("  calibration …")
    plot_calibration(y_test, y_proba, out_dir=out)

    print(f"\nEvaluation plots saved → {out}")
