# edge_ml/plotting.py — feature inspection plots

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    import seaborn as sns
except ImportError:
    sns = None  # type: ignore


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ensure_dir(out_dir: Optional[Union[str, Path]]) -> Optional[Path]:
    if out_dir is None:
        return None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _savefig(fig: plt.Figure, out_dir: Optional[Path], name: str, dpi: int = 140) -> None:
    if out_dir is None:
        return
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=dpi)
    plt.close(fig)


# ── EDA plots ─────────────────────────────────────────────────────────────────


def plot_label_balance(
    y: pd.Series,
    *,
    out_dir: Optional[Union[str, Path]] = None,
    label_name: str = "is_true_edge",
) -> None:
    """Bar chart of class counts and proportions."""
    counts = y.value_counts().sort_index()
    labels = ["FP (0)", "TP (1)"]
    colors = ["#F44336", "#4CAF50"]
    out = _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white")
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.01,
            f"{count:,}\n({count / counts.sum() * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("Count")
    ax.set_title(f"Label balance — {label_name}")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    _savefig(fig, out, "label_balance.png")


def plot_feature_distributions(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    *,
    bins: int = 80,
    clip_quantile: float = 0.995,
    out_dir: Optional[Union[str, Path]] = None,
) -> None:
    """
    One subplot per feature: overlaid density histograms for TP vs FP.
    Values above clip_quantile (and below 1 - clip_quantile) are clipped
    for readability.
    """
    out = _ensure_dir(out_dir)
    n = len(feature_cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()
    colors = {0: "#F44336", 1: "#4CAF50"}
    class_labels = {0: "FP (0)", 1: "TP (1)"}

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        for cls in [0, 1]:
            vals = df.loc[df[label_col] == cls, col].dropna()
            hi = vals.quantile(clip_quantile)
            lo = vals.quantile(1 - clip_quantile)
            vals = vals.clip(lo, hi)
            ax.hist(vals, bins=bins, density=True, alpha=0.55, color=colors[cls], label=class_labels[cls])
        ax.set_title(col, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature distributions — TP vs FP", fontsize=12, y=1.01)
    _savefig(fig, out, "feature_distributions.png", dpi=120)


def plot_feature_boxplots(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    *,
    clip_quantile: float = 0.99,
    out_dir: Optional[Union[str, Path]] = None,
) -> None:
    """
    One subplot per feature: side-by-side box plots for FP / TP.
    Useful to judge median shift and spread between classes.
    """
    out = _ensure_dir(out_dir)
    n = len(feature_cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()
    colors = ["#F44336", "#4CAF50"]

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        groups = []
        for cls in [0, 1]:
            vals = df.loc[df[label_col] == cls, col].dropna()
            hi = vals.quantile(clip_quantile)
            lo = vals.quantile(1 - clip_quantile)
            groups.append(vals.clip(lo, hi).values)
        bp = ax.boxplot(groups, labels=["FP", "TP"], patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(col, fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature box plots — TP vs FP (clipped at 1%/99%)", fontsize=12, y=1.01)
    _savefig(fig, out, "feature_boxplots.png", dpi=120)


def plot_correlation_matrix(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    out_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Pearson correlation heatmap of all features."""
    out = _ensure_dir(out_dir)
    corr = df[list(feature_cols)].corr()
    n = len(feature_cols)
    fig, ax = plt.subplots(figsize=(n * 0.7 + 1, n * 0.7 + 1))
    if sns is not None:
        sns.heatmap(
            corr,
            ax=ax,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            center=0,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 7},
            square=True,
            linewidths=0.4,
        )
    else:
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(feature_cols, fontsize=8)
        plt.colorbar(im, ax=ax)
    ax.set_title("Pearson correlation matrix")
    _savefig(fig, out, "correlation_matrix.png", dpi=120)


def plot_gap_distribution(
    df: pd.DataFrame,
    *,
    gap_col: str = "gap_nights",
    label_col: str = "is_true_edge",
    out_dir: Optional[Union[str, Path]] = None,
) -> None:
    """
    Distribution of gap_nights (edge span in nights) overall and split by label.
    Useful to check whether long-gap edges are disproportionately FP.
    """
    if gap_col not in df.columns:
        return
    out = _ensure_dir(out_dir)
    max_gap = int(df[gap_col].quantile(0.995))
    bins = np.arange(-0.5, max_gap + 1.5, 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(df[gap_col].clip(0, max_gap), bins=bins, color="steelblue", edgecolor="white", lw=0.3)
    ax.set_xlabel("gap_nights")
    ax.set_ylabel("Count")
    ax.set_title("Overall gap distribution")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    ax = axes[1]
    colors = {0: "#F44336", 1: "#4CAF50"}
    for cls, label in [(0, "FP"), (1, "TP")]:
        vals = df.loc[df[label_col] == cls, gap_col].clip(0, max_gap)
        ax.hist(vals, bins=bins, density=True, alpha=0.55, color=colors[cls], label=label)
    ax.set_xlabel("gap_nights")
    ax.set_ylabel("Density")
    ax.set_title("Gap distribution — TP vs FP")
    ax.legend()

    fig.suptitle("Edge gap_nights distribution", fontsize=12)
    _savefig(fig, out, "gap_distribution.png")


def plot_pairwise_scatter(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    *,
    sample_n: int = 5_000,
    out_dir: Optional[Union[str, Path]] = None,
) -> None:
    """
    Seaborn pairplot on a random sample — reveals non-linear structure
    and inter-feature correlations that the heatmap misses.
    Only produced when seaborn is available.
    """
    if sns is None:
        return
    out = _ensure_dir(out_dir)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(df), size=min(sample_n, len(df)), replace=False)
    sample = df.iloc[idx][list(feature_cols) + [label_col]].copy()
    sample[label_col] = sample[label_col].map({0: "FP", 1: "TP"})
    g = sns.pairplot(sample, hue=label_col, palette={"FP": "#F44336", "TP": "#4CAF50"}, plot_kws={"alpha": 0.3, "s": 8}, diag_kind="kde")
    g.figure.suptitle(f"Pairwise scatter (n={len(sample):,} sample)", y=1.01, fontsize=11)
    if out is not None:
        g.figure.savefig(out / "pairwise_scatter.png", dpi=100, bbox_inches="tight")
        plt.close(g.figure)
