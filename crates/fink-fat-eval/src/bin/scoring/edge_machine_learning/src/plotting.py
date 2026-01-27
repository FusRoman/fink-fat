# edge_ml/plotting.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:  # seaborn is optional
    sns = None  # type: ignore

from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve
import time

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _ensure_dir(out_dir: Optional[Union[str, Path]]) -> Optional[Path]:
    if out_dir is None:
        return None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _savefig(
    fig: plt.Figure, out_dir: Optional[Path], name: str, dpi: int = 140
) -> None:
    if out_dir is None:
        return
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=dpi)
    plt.close(fig)


def _maybe_title(ax: plt.Axes, title: Optional[str]) -> None:
    if title:
        ax.set_title(title)


def _get_proba(model: ClassifierMixin, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return p[:, 1]
    if hasattr(model, "decision_function"):
        # map decision scores to [0,1] via a logistic squashing for plotting
        s = model.decision_function(X)
        s = np.asarray(s)
        return 1.0 / (1.0 + np.exp(-s))
    raise TypeError("Model has neither predict_proba nor decision_function.")


def _get_feature_names(
    df_or_names: Union[pd.DataFrame, Sequence[str]],
) -> Sequence[str]:
    if isinstance(df_or_names, pd.DataFrame):
        return list(df_or_names.columns)
    return list(df_or_names)


# ---------------------------------------------------------------------
# Core plots
# ---------------------------------------------------------------------


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    out_dir: Optional[Union[str, Path]] = None,
    prefix: str = "",
) -> None:
    """
    ROC + PR curves (scikit-learn display helpers).
    """
    out = _ensure_dir(out_dir)

    # ROC
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name="classifier")
    try:
        roc = (
            roc_auc_score(y_true, y_proba)
            if len(np.unique(y_true)) > 1
            else float("nan")
        )
        _maybe_title(ax, f"ROC curve (AUC={roc:.4f})")
    except Exception:
        _maybe_title(ax, "ROC curve")
    _savefig(fig, out, f"{prefix}roc_curve.png")

    # PR
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax, name="classifier")
    try:
        ap = (
            average_precision_score(y_true, y_proba)
            if len(np.unique(y_true)) > 1
            else float("nan")
        )
        _maybe_title(ax, f"Precision-Recall (AP={ap:.4f})")
    except Exception:
        _maybe_title(ax, "Precision-Recall curve")
    _savefig(fig, out, f"{prefix}pr_curve.png")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    normalize: Optional[str] = "true",
    out_dir: Optional[Union[str, Path]] = None,
    prefix: str = "",
    title: str = "Confusion matrix",
) -> None:
    """
    Confusion matrix (normalize can be: 'true', 'pred', 'all' or None).
    """
    out = _ensure_dir(out_dir)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        normalize=normalize,
        values_format=".2f" if normalize else "d",
        ax=ax,
    )
    _maybe_title(ax, title)
    _savefig(fig, out, f"{prefix}confusion_matrix.png")


def plot_probability_histograms(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    bins: int = 60,
    out_dir: Optional[Union[str, Path]] = None,
    prefix: str = "",
    title: str = "Predicted probability distributions",
) -> None:
    """
    Histogrammes des probas prédites, séparés par classe vraie.
    Super utile pour voir le recouvrement vrai/faux.
    """
    out = _ensure_dir(out_dir)
    fig, ax = plt.subplots()

    y_true = np.asarray(y_true).astype(int)
    p0 = y_proba[y_true == 0]
    p1 = y_proba[y_true == 1]

    if sns is not None:
        sns.histplot(
            p0,
            bins=bins,
            stat="density",
            kde=False,
            ax=ax,
            label="false (0)",
            alpha=0.5,
        )
        sns.histplot(
            p1, bins=bins, stat="density", kde=False, ax=ax, label="true (1)", alpha=0.5
        )
    else:
        ax.hist(p0, bins=bins, density=True, alpha=0.5, label="false (0)")
        ax.hist(p1, bins=bins, density=True, alpha=0.5, label="true (1)")

    ax.set_xlabel("p(y=1)")
    ax.set_ylabel("density")
    ax.legend()
    _maybe_title(ax, title)
    _savefig(fig, out, f"{prefix}proba_hist.png")


def plot_calibration(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_bins: int = 15,
    out_dir: Optional[Union[str, Path]] = None,
    prefix: str = "",
) -> None:
    """
    Courbe de calibration + Brier score.
    Si tu veux passer ensuite à CalibratedClassifierCV, c'est LE plot à regarder.
    """
    out = _ensure_dir(out_dir)

    fig, ax = plt.subplots()
    CalibrationDisplay.from_predictions(
        y_true, y_proba, n_bins=n_bins, ax=ax, name="classifier"
    )
    try:
        bs = brier_score_loss(y_true, y_proba)
        _maybe_title(ax, f"Calibration curve (Brier={bs:.4f})")
    except Exception:
        _maybe_title(ax, "Calibration curve")
    _savefig(fig, out, f"{prefix}calibration.png")


# ---------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------


def plot_feature_importance(
    model: ClassifierMixin,
    feature_names: Sequence[str],
    *,
    top_n: int = 25,
    out_dir: Optional[Union[str, Path]] = None,
    prefix: str = "",
) -> bool:
    """
    Importance "native" du modèle si disponible (ex: tree/GB/forest: feature_importances_).
    Retourne True si plot fait, False si indisponible.
    """
    out = _ensure_dir(out_dir)

    # pipeline-safe: on récupère le dernier step si possible
    clf = getattr(model, "named_steps", {}).get("clf", model)

    if not hasattr(clf, "feature_importances_"):
        return False

    imp = np.asarray(clf.feature_importances_, dtype=float)
    if imp.shape[0] != len(feature_names):
        # si tu changes l'espace de features (one-hot etc.), ça peut diverger
        return False

    order = np.argsort(imp)[::-1][:top_n]
    names = [feature_names[i] for i in order]
    vals = imp[order]

    fig, ax = plt.subplots(figsize=(7, max(4, 0.25 * len(names) + 1)))
    ax.barh(range(len(names))[::-1], vals, align="center")
    ax.set_yticks(range(len(names))[::-1])
    ax.set_yticklabels(names)
    ax.set_xlabel("importance")
    ax.set_title(f"Feature importances (top {len(names)})")
    _savefig(fig, out, f"{prefix}feature_importances.png")
    return True


def plot_permutation_importance(
    model: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    scoring: str = "roc_auc",
    n_repeats: int = 10,
    random_state: int = 42,
    top_n: int = 25,
    out_dir: Optional[Union[str, Path]] = None,
    prefix: str = "",
) -> None:
    """
    Importance par permutation (agnostique du modèle).
    Très utile pour vérifier que 'resid_*' / 'd2_pos' dominent bien, etc.
    """
    out = _ensure_dir(out_dir)

    r = permutation_importance(
        model,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    imp = r.importances_mean
    std = r.importances_std

    order = np.argsort(imp)[::-1][:top_n]
    names = [feature_names[i] for i in order]
    vals = imp[order]
    errs = std[order]

    fig, ax = plt.subplots(figsize=(7, max(4, 0.25 * len(names) + 1)))
    ax.barh(range(len(names))[::-1], vals, xerr=errs, align="center")
    ax.set_yticks(range(len(names))[::-1])
    ax.set_yticklabels(names)
    ax.set_xlabel(f"Δ score (permutation, {scoring})")
    ax.set_title(f"Permutation importance (top {len(names)})")
    _savefig(fig, out, f"{prefix}permutation_importance_{scoring}.png")


# ---------------------------------------------------------------------
# Training diagnostics
# ---------------------------------------------------------------------


def plot_learning_curve(
    model: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    *,
    scoring: str = "roc_auc",
    cv: int = 5,
    train_sizes: Sequence[float] = (0.05, 0.1, 0.2, 0.4, 0.7, 1.0),
    random_state: int = 42,
    out_dir: Optional[Union[str, Path]] = None,
    prefix: str = "",
) -> None:
    """
    Learning curve: aide à savoir si tu es en underfit / overfit / manque de data.
    """
    out = _ensure_dir(out_dir)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        train_sizes=train_sizes,
        shuffle=True,
        random_state=random_state,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots()
    ax.plot(train_sizes_abs, train_mean, marker="o", label="train")
    ax.fill_between(
        train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.2
    )
    ax.plot(train_sizes_abs, val_mean, marker="o", label="cv")
    ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.2)
    ax.set_xlabel("Training set size")
    ax.set_ylabel(scoring)
    ax.set_title(f"Learning curve ({scoring}, cv={cv})")
    ax.legend()
    _savefig(fig, out, f"{prefix}learning_curve_{scoring}.png")


# ---------------------------------------------------------------------
# Edge-specific plot: "ranking-like" diagnostics from classification scores
# ---------------------------------------------------------------------


def plot_hit_at_k_curve(
    df_scored: pd.DataFrame,
    *,
    group_col: str,
    label_col: str,
    score_col: str,
    ks: Sequence[int] = (1, 2, 3, 5, 10, 20, 50),
    out_dir: Optional[Union[str, Path]] = None,
    prefix: str = "",
) -> Optional[dict]:
    """
    Trace Hit@K vs K si tu as une colonne de group (ex: from_seed_id).
    - Pour chaque groupe, on trie par score décroissant.
    - Hit@K = fraction des groupes où au moins un vrai est dans le top-K.

    Renvoie un dict avec hit@k, sinon None si impossible.
    """
    if group_col not in df_scored.columns:
        return None
    out = _ensure_dir(out_dir)

    # ne garder que les groupes ayant au moins un positif
    g = df_scored.groupby(group_col, sort=False)
    subs = []
    for _, sub in g:
        if sub[label_col].sum() > 0:
            subs.append(sub)

    if not subs:
        return None

    hits = {k: 0 for k in ks}
    n_groups = 0

    for sub in subs:
        sub = sub.sort_values(score_col, ascending=False)
        y = sub[label_col].to_numpy()
        n_groups += 1
        for k in ks:
            kk = min(int(k), len(y))
            if kk > 0 and y[:kk].max() == 1:
                hits[k] += 1

    hit_rate = {k: hits[k] / n_groups for k in hits}

    fig, ax = plt.subplots()
    ax.plot(list(hit_rate.keys()), list(hit_rate.values()), marker="o")
    ax.set_xlabel("K")
    ax.set_ylabel("Hit@K")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"Hit@K curve (n_groups={n_groups})")
    _savefig(fig, out, f"{prefix}hit_at_k.png")

    return {"hit_at_k": hit_rate, "n_groups": n_groups}


# ---------------------------------------------------------------------
# One-stop function
# ---------------------------------------------------------------------


def save_all_standard_plots(
    *,
    model: ClassifierMixin,
    df_test: pd.DataFrame,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Sequence[str],
    out_dir: Union[str, Path],
    prefix: str = "",
    group_col: str = "from_seed_id",
    label_col: str = "is_true_edge",
) -> None:
    """
    Génère un pack standard de plots dans out_dir.

    Attend df_test (avec colonnes debug optionnelles) + X_test/y_test.
    """
    out = _ensure_dir(out_dir)
    assert out is not None

    y_proba = _get_proba(model, X_test)
    y_pred = (y_proba >= 0.5).astype(int)

    t_before = time.time()
    plot_roc_pr_curves(y_test, y_proba, out_dir=out, prefix=prefix)
    print(f"ROC + PR curves plotted in {time.time() - t_before:.2f} seconds.")

    t_before = time.time()
    plot_confusion_matrix(y_test, y_pred, normalize="true", out_dir=out, prefix=prefix)
    print(f"Confusion matrix plotted in {time.time() - t_before:.2f} seconds.")

    t_before = time.time()
    plot_probability_histograms(y_test, y_proba, out_dir=out, prefix=prefix)
    print(f"Probability histograms plotted in {time.time() - t_before:.2f} seconds.")

    t_before = time.time()
    plot_calibration(y_test, y_proba, out_dir=out, prefix=prefix)
    print(f"Calibration plot plotted in {time.time() - t_before:.2f} seconds.")
    print("All basic plots done.\n")

    t_before = time.time()
    # feature importance (native if possible)
    _ = plot_feature_importance(model, feature_names, out_dir=out, prefix=prefix)
    print(f"Feature importance plotted in {time.time() - t_before:.2f} seconds.")

    t_before = time.time()
    # permutation importance (always possible but potentially expensive)
    plot_permutation_importance(
        model,
        X_test,
        y_test,
        feature_names,
        scoring="roc_auc",
        n_repeats=8,
        out_dir=out,
        prefix=prefix,
    )
    print(f"Permutation importance plotted in {time.time() - t_before:.2f} seconds.")
    print("All feature importance plots done.\n")

    t_before = time.time()
    # ranking-like eval / curve si group_col existe
    df_scored = df_test.copy()
    df_scored["y_proba"] = y_proba
    _ = plot_hit_at_k_curve(
        df_scored,
        group_col=group_col,
        label_col=label_col,
        score_col="y_proba",
        out_dir=out,
        prefix=prefix,
    )
    print(f"Hit@K curve plotted in {time.time() - t_before:.2f} seconds.")