# edge_ml/data.py

from pathlib import Path
import pandas as pd
import numpy as np

from config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    GROUP_COLUMN,
)


def load_parquet_dataset(parquet_dir: Path, limit: int = None) -> pd.DataFrame:
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No parquet files found in {parquet_dir}")

    print(f"Detected {len(files)} parquet files from {parquet_dir} ...")

    if limit is not None:
        files = files[:limit]
        print(f"Limiting to first {limit} files for testing ...")

    dfs = [pd.read_parquet(p) for p in files]
    print(f"Loaded {len(dfs)} parquet files.")
    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataframe shape: {df.shape}")
    return df


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    # Nettoyage minimal pour sklearn
    df = df.replace([np.inf, -np.inf], np.nan)

    needed = list(FEATURE_COLUMNS) + [TARGET_COLUMN]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in dataset: {missing}")

    df = df.dropna(subset=needed).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # Booleans -> int si jamais
    for c in FEATURE_COLUMNS:
        if df[c].dtype == "bool":
            df[c] = df[c].astype(int)

    return df


def make_xy(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df[TARGET_COLUMN].to_numpy()
    return X, y


def train_test_split_indices(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
):
    """
    Retourne (train_idx, test_idx).

    - Si GROUP_COLUMN est présent, split "group-aware" : un même from_seed_id
      ne peut pas être à la fois dans train et test.
    - Sinon: split aléatoire standard.
    """
    rng = np.random.default_rng(random_state)

    if GROUP_COLUMN in df.columns:
        groups = df[GROUP_COLUMN].to_numpy()
        uniq = np.unique(groups)
        rng.shuffle(uniq)

        n_test_groups = max(1, int(len(uniq) * test_size))
        test_groups = set(uniq[:n_test_groups])

        is_test = np.array([g in test_groups for g in groups], dtype=bool)
        test_idx = np.where(is_test)[0]
        train_idx = np.where(~is_test)[0]

        print(
            f"Group split on {GROUP_COLUMN}: "
            f"{len(train_idx)} train rows, {len(test_idx)} test rows "
            f"({len(uniq)-n_test_groups} train groups, {n_test_groups} test groups)"
        )
        return train_idx, test_idx

    # fallback random split
    n = len(df)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = max(1, int(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    print(f"Random split: {len(train_idx)} train rows, {len(test_idx)} test rows")
    return train_idx, test_idx
