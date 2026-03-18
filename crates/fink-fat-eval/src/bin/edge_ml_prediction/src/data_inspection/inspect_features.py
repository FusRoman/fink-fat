"""
Feature inspection script for the edge_ml_prediction dataset.

Loads ``edge_features.parquet`` and produces a standard EDA plot pack
in ``OUT_DIR`` (configured in ``config.py``):

- ``label_balance.png``        — class counts and proportions
- ``feature_distributions.png``— per-feature density histograms (TP vs FP)
- ``feature_boxplots.png``     — per-feature box plots (TP vs FP)
- ``correlation_matrix.png``   — Pearson correlation heatmap
- ``gap_distribution.png``     — gap_nights distribution (TP vs FP)
- ``pairwise_scatter.png``     — pairwise scatter on a random sample (seaborn)

Usage::

    cd crates/fink-fat-eval/src/bin/edge_ml_prediction
    pdm run python src/data_inspection/inspect_features.py
"""

import sys
from pathlib import Path

# Allow running from the project root (edge_ml_prediction/) as well as src/.
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

import config as C
import plotting as P

# ── Load ──────────────────────────────────────────────────────────────────────

print(f"Loading {C.PARQUET_PATH} …")
df = pd.read_parquet(C.PARQUET_PATH)
n_tp = int(df[C.TARGET_COLUMN].sum())
n_fp = int((df[C.TARGET_COLUMN] == 0).sum())
print(f"  {len(df):,} edges  |  {n_tp:,} TP  |  {n_fp:,} FP  |  ratio {n_tp/len(df)*100:.1f}% TP")

C.OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output → {C.OUT_DIR}\n")

# ── Plots ─────────────────────────────────────────────────────────────────────

print("1/6  Label balance …")
P.plot_label_balance(df[C.TARGET_COLUMN], out_dir=C.OUT_DIR, label_name=C.TARGET_COLUMN)

print("2/6  Feature distributions …")
P.plot_feature_distributions(df, C.FEATURE_COLUMNS, C.TARGET_COLUMN, out_dir=C.OUT_DIR)

print("3/6  Feature box plots …")
P.plot_feature_boxplots(df, C.FEATURE_COLUMNS, C.TARGET_COLUMN, out_dir=C.OUT_DIR)

print("4/6  Correlation matrix …")
P.plot_correlation_matrix(df, C.FEATURE_COLUMNS, out_dir=C.OUT_DIR)

print("5/6  Gap distribution …")
P.plot_gap_distribution(df, out_dir=C.OUT_DIR)

print("6/6  Pairwise scatter (subsample) …")
P.plot_pairwise_scatter(df, C.FEATURE_COLUMNS, C.TARGET_COLUMN, sample_n=8_000, out_dir=C.OUT_DIR)

print(f"\nDone. All plots saved to {C.OUT_DIR}")
