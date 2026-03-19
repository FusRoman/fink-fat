# edge_ml/data.py — dataset loading and train/test split

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_xy(
    parquet_path: Path,
    feature_cols: Sequence[str],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load the parquet and return (X, y, feature_names).

    Only the requested feature and target columns are read from disk.
    """
    cols = list(feature_cols) + [target_col]
    df = pd.read_parquet(parquet_path, columns=cols)

    X = df[list(feature_cols)].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.int8)
    return X, y, list(feature_cols)


def split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train / test split."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
