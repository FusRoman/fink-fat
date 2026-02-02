# edge_ml/data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import logging
import time
from collections import Counter

from config import FEATURE_COLUMNS, TARGET_COLUMN, GROUP_COLUMN, DEBUG_COLUMNS

logger = logging.getLogger(__name__)


def split_mask_from_group(
    groups: np.ndarray,
    *,
    test_size: float,
    val_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic split per group id (fast C-based hashing).

    Notes
    -----
    `test_size` and `val_size` are *fractions of the full dataset*.
    Therefore, you must have: test_size + val_size < 1.0
    (the remainder is assigned to train).

    Returns
    -------
    is_train, is_test, is_val : boolean masks (same length as groups)
    """
    t_test = float(test_size)
    t_val = float(val_size)

    if not (0.0 <= t_test <= 1.0 and 0.0 <= t_val <= 1.0):
        raise ValueError(
            f"Invalid split fractions: test_size={t_test}, val_size={t_val}. "
            "Both must be in [0, 1]."
        )

    if (t_test + t_val) >= 1.0:
        raise ValueError(
            f"Invalid split fractions: test_size + val_size = {t_test + t_val:.6f} >= 1.0. "
            "This leaves 0 rows for TRAIN. "
            "Use e.g. test_size=0.2, val_size=0.2 (train=0.6)."
        )

    t_val_end = t_test + t_val

    # Fast, deterministic hashing in C (pandas)
    s = pd.Series(groups, copy=False)
    h = pd.util.hash_pandas_object(s, index=False).to_numpy(dtype=np.uint64, copy=False)

    # Map to [0, 1)
    r = (h % np.uint64(10_000_000)).astype(np.float64) / 10_000_000.0

    is_test = r < t_test
    is_val = (r >= t_test) & (r < t_val_end)
    is_train = ~(is_test | is_val)
    return is_train, is_test, is_val


def split_mask_rowwise(
    n: int,
    *,
    random_state: int,
    test_size: float,
    val_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Row-wise deterministic split (fallback if group column absent).

    Notes
    -----
    `test_size` and `val_size` are *fractions of the full dataset*.
    Therefore, you must have: test_size + val_size < 1.0
    (the remainder is assigned to train).
    """
    t_test = float(test_size)
    t_val = float(val_size)

    if not (0.0 <= t_test <= 1.0 and 0.0 <= t_val <= 1.0):
        raise ValueError(
            f"Invalid split fractions: test_size={t_test}, val_size={t_val}. "
            "Both must be in [0, 1]."
        )

    if (t_test + t_val) >= 1.0:
        raise ValueError(
            f"Invalid split fractions: test_size + val_size = {t_test + t_val:.6f} >= 1.0. "
            "This leaves 0 rows for TRAIN. "
            "Use e.g. test_size=0.2, val_size=0.2 (train=0.6)."
        )

    rng = np.random.default_rng(int(random_state))
    r = rng.random(int(n))

    t_val_end = t_test + t_val

    is_test = r < t_test
    is_val = (r >= t_test) & (r < t_val_end)
    is_train = ~(is_test | is_val)
    return is_train, is_test, is_val


# -----------------------------------------------------------------------------
# Reservoir sampling (bounded memory evaluation sets)
# -----------------------------------------------------------------------------
@dataclass
class Reservoir:
    max_rows: int
    rng: np.random.Generator
    X: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None

    # Optional row storage for diagnostics/plots (kept aligned with X/y)
    df_rows: Optional[List[Optional[dict]]] = None

    n_seen: int = 0  # total seen so far
    filled: int = 0  # how many rows are currently stored (<= max_rows)

    def add(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        df_new: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Add a batch to the reservoir using reservoir sampling.

        Optimizations
        -------------
        - Bulk copy while reservoir not full (vectorized).
        - Vectorized replacement sampling for the remainder of the batch.
        - If df_new is provided, store columns in pre-allocated arrays instead of
        per-row dicts (avoids slow df.iloc[i].to_dict() loops).

        Notes
        -----
        The replacement phase is vectorized. If the RNG generates duplicate target
        indices, later rows in the same call overwrite earlier ones (same as any
        "last write wins" behavior). This does not affect the correctness of having
        a uniform reservoir sample in practice, and is dramatically faster.
        """
        if self.max_rows <= 0:
            return
        n = int(len(y_new))
        if n == 0:
            return

        # Lazy allocate arrays
        if self.X is None:
            n_features = int(X_new.shape[1])
            self.X = np.empty((self.max_rows, n_features), dtype=X_new.dtype)
            self.y = np.empty((self.max_rows,), dtype=y_new.dtype)
            self.filled = 0
            self.n_seen = 0

            # Optional: store df columns in arrays (fast) instead of list-of-dicts (slow)
            if df_new is not None:
                # We'll create one array per column, same length as reservoir
                self._df_cols = {
                    c: np.empty((self.max_rows,), dtype=object) for c in df_new.columns
                }
            else:
                self._df_cols = None

        # Safety
        assert self.X is not None and self.y is not None

        # ---------------------------------------------------------------------
        # 1) Bulk fill while not full
        # ---------------------------------------------------------------------
        if self.filled < self.max_rows:
            take = min(self.max_rows - self.filled, n)
            a = self.filled
            b = self.filled + take

            self.X[a:b] = X_new[:take]
            self.y[a:b] = y_new[:take]

            if df_new is not None and getattr(self, "_df_cols", None) is not None:
                # Copy each column in bulk (fast)
                for c, arr in self._df_cols.items():
                    arr[a:b] = df_new[c].to_numpy(dtype=object, copy=False)[:take]

            self.filled = b
            self.n_seen += take

            if take == n:
                return

            # Remaining part to process
            X_new = X_new[take:]
            y_new = y_new[take:]
            if df_new is not None:
                df_new = df_new.iloc[take:]
            n = int(len(y_new))

        # ---------------------------------------------------------------------
        # 2) Vectorized reservoir replacement for the remaining rows
        # ---------------------------------------------------------------------
        # For incoming item k (1..n), valid range is [0, n_seen + k)
        # We generate u in [0,1), then floor(u * (n_seen + k))
        base = int(self.n_seen)
        ks = np.arange(1, n + 1, dtype=np.int64)
        denom = base + ks

        u = self.rng.random(n, dtype=np.float64)
        j = (u * denom).astype(np.int64)

        accept = j < int(self.max_rows)
        if not np.any(accept):
            self.n_seen += n
            return

        j_acc = j[accept].astype(np.int64, copy=False)
        idx_acc = np.nonzero(accept)[0].astype(np.int64, copy=False)

        self.X[j_acc] = X_new[idx_acc]
        self.y[j_acc] = y_new[idx_acc]

        if df_new is not None and getattr(self, "_df_cols", None) is not None:
            # Column-wise assignment using fancy indexing (fast)
            for c, arr in self._df_cols.items():
                col = df_new[c].to_numpy(dtype=object, copy=False)
                arr[j_acc] = col[idx_acc]

        self.n_seen += n

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the filled part of the reservoir.
        """
        if self.X is None or self.y is None:
            return np.empty((0, 0)), np.empty((0,))
        return self.X[: self.filled].copy(), self.y[: self.filled].copy()

    def finalize_df(self) -> Optional[pd.DataFrame]:
        """
        Return a DataFrame aligned with the filled part of the reservoir.
        """
        df_cols = getattr(self, "_df_cols", None)
        if df_cols is None:
            return None
        if self.filled <= 0:
            return None
        data = {c: arr[: self.filled].copy() for c, arr in df_cols.items()}
        return pd.DataFrame(data).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Streaming parquet reader
# -----------------------------------------------------------------------------
def iter_parquet_batches(
    parquet_dir: Path,
    *,
    columns: Sequence[str],
    batch_rows: int,
) -> Iterator[pd.DataFrame]:
    """
    Stream parquet dataset in row batches using pyarrow.dataset.

    Logging
    -------
    - Logs a compact summary at the end (batches, empty skipped, rows, throughput).
    - Optional "status line" updated every `log_every` batches (single-line refresh).
    """
    try:
        import pyarrow.dataset as ds
    except Exception as e:
        raise RuntimeError(
            "Streaming parquet requires pyarrow.\n"
            "Install with:\n"
            "  pip install pyarrow\n"
            f"Original error: {e}"
        )

    parquet_dir = Path(parquet_dir)
    if not parquet_dir.exists():
        raise FileNotFoundError(str(parquet_dir))

    # Fast visibility: don't rglob the whole dataset
    try:
        import itertools

        it = parquet_dir.rglob("*.parquet")
        first = list(itertools.islice(it, 5))
        if first:
            logger.info("Parquet dataset dir: %s", parquet_dir)
            logger.info("First 5 parquet file(s): %s", [str(p) for p in first])
        else:
            logger.warning("No parquet files found under: %s", parquet_dir)
    except Exception:
        logger.info("Parquet dataset dir: %s", parquet_dir)

    dataset = ds.dataset(str(parquet_dir), format="parquet")
    scanner = dataset.scanner(columns=list(columns), batch_size=int(batch_rows))

    # ---- logging knobs ----
    log_every = 200  # update status every N batches
    status_line = logger.isEnabledFor(logging.INFO)

    t0 = time.time()
    n_batches = 0
    n_rows_total = 0
    n_empty_skipped = 0
    n_nonempty = 0

    # only for summary (no per-batch debug spam)
    max_batch_rows = 0
    min_batch_rows = None

    def _status(msg: str) -> None:
        # Single-line refresh on stdout (not logger) to avoid huge logs
        if not status_line:
            return
        print("\r" + msg, end="", flush=True)

    for batch in scanner.to_batches():
        n_batches += 1
        n_rows = int(batch.num_rows)
        n_rows_total += n_rows

        if n_rows == 0:
            n_empty_skipped += 1
            if (n_batches % log_every) == 0:
                dt = max(1e-9, time.time() - t0)
                _status(
                    f"[scan] batches={n_batches:,} nonempty={n_nonempty:,} empty={n_empty_skipped:,} "
                    f"rows={n_rows_total:,} ({n_rows_total/dt:,.0f} rows/s)"
                )
            continue

        n_nonempty += 1
        max_batch_rows = max(max_batch_rows, n_rows)
        min_batch_rows = (
            n_rows if min_batch_rows is None else min(min_batch_rows, n_rows)
        )

        if (n_batches == 1) or ((n_batches % log_every) == 0):
            dt = max(1e-9, time.time() - t0)
            _status(
                f"[scan] batches={n_batches:,} nonempty={n_nonempty:,} empty={n_empty_skipped:,} "
                f"rows={n_rows_total:,} ({n_rows_total/dt:,.0f} rows/s)"
            )

        df = batch.to_pandas(self_destruct=True)
        yield df

    # finish status line cleanly
    if status_line:
        print("", flush=True)

    dt = max(1e-9, time.time() - t0)
    logger.info(
        "Finished scanning: batches=%d nonempty=%d empty_skipped=%d total_rows=%d elapsed=%.2fs (%.1f rows/s) "
        "batch_rows(min=%s max=%d) selected_columns=%d",
        n_batches,
        n_nonempty,
        n_empty_skipped,
        n_rows_total,
        dt,
        (n_rows_total / dt),
        str(min_batch_rows) if min_batch_rows is not None else "NA",
        int(max_batch_rows),
        len(columns),
    )


def prepare_df_batch(df: pd.DataFrame) -> pd.DataFrame:
    n_raw = len(df)
    if n_raw == 0:
        return df

    needed = list(FEATURE_COLUMNS) + [TARGET_COLUMN]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        logger.error("Missing required columns: %s", missing)
        raise KeyError(f"Missing columns in dataset batch: {missing}")

    # Only touch required columns (fastest win)
    df[needed] = df[needed].replace([np.inf, -np.inf], np.nan)

    n_before = len(df)
    df = df.dropna(subset=needed)
    n_after = len(df)
    if n_after == 0:
        return df

    # Ensure target int (this may force a copy; that's OK)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(np.int8, copy=False)

    # Optional: convert bool -> int only for those columns that are bool
    # (still fine; usually few)
    for c in FEATURE_COLUMNS:
        if df[c].dtype == "bool":
            df[c] = df[c].astype(np.int8, copy=False)

    logger.debug(
        "Prepared batch: raw=%d cleaned=%d dropped=%d",
        n_raw,
        n_after,
        n_before - n_after,
    )
    return df


def make_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df[TARGET_COLUMN].to_numpy()
    return X, y


def stream_splits(
    parquet_dir: Path,
    *,
    batch_rows: int,
    test_size: float,
    val_size: float,
    random_state: int,
    keep_df: bool,
    max_total_rows: Optional[int] = None,
) -> Iterator[Tuple[str, np.ndarray, np.ndarray, Optional[pd.DataFrame]]]:
    """
    Yield streaming batches assigned to train/test/val with deterministic split.

    Logging
    -------
    - No per-batch logger spam.
    - Aggregated logs every `log_every` non-empty batches + one final summary.
    - Optional single-line status refresh on stdout.
    """
    cols = list(
        dict.fromkeys(FEATURE_COLUMNS + [TARGET_COLUMN, GROUP_COLUMN] + DEBUG_COLUMNS)
    )

    logger.info(
        "Starting stream_splits: dir=%s batch_rows=%d test_size=%.3f val_size=%.3f keep_df=%s max_total_rows=%s",
        str(parquet_dir),
        int(batch_rows),
        float(test_size),
        float(val_size),
        bool(keep_df),
        str(max_total_rows),
    )

    # ---- logging knobs ----
    log_every = 100  # aggregated log window
    status_line = logger.isEnabledFor(logging.INFO)

    def _status(msg: str) -> None:
        if not status_line:
            return
        print("\r" + msg, end="", flush=True)

    t0_all = time.time()

    seen_total = 0
    batch_idx = 0
    nonempty_batches = 0

    # Global totals
    cum = Counter()

    # Window totals (reset every log_every)
    win = Counter()
    t0_win = time.time()

    for df in iter_parquet_batches(parquet_dir, columns=cols, batch_rows=batch_rows):
        batch_idx += 1
        n_in = int(len(df))
        if n_in == 0:
            continue

        df = prepare_df_batch(df)
        n_clean = int(len(df))
        if n_clean == 0:
            continue

        nonempty_batches += 1

        # Enforce max_total_rows cap
        if max_total_rows is not None:
            remaining = int(max_total_rows) - seen_total
            if remaining <= 0:
                break
            if n_clean > remaining:
                df = df.iloc[:remaining]
                n_clean = int(len(df))

        # Split masks
        if GROUP_COLUMN in df.columns:
            groups = df[GROUP_COLUMN].to_numpy(copy=False)
            is_train, is_test, is_val = split_mask_from_group(
                groups, test_size=test_size, val_size=val_size
            )
        else:
            is_train, is_test, is_val = split_mask_rowwise(
                n_clean,
                random_state=random_state,
                test_size=test_size,
                val_size=val_size,
            )
            # warn once (not in a loop spam)
            if cum["warn_rowwise"] == 0:
                logger.warning(
                    "Group column '%s' missing -> falling back to rowwise split.",
                    GROUP_COLUMN,
                )
            cum["warn_rowwise"] += 1

        n_train = int(is_train.sum())
        n_test = int(is_test.sum())
        n_val = int(is_val.sum())

        seen_total += n_clean

        # Update totals
        cum["rows_raw"] += n_in
        cum["rows_clean"] += n_clean
        cum["train"] += n_train
        cum["test"] += n_test
        cum["val"] += n_val
        cum["batches_nonempty"] += 1

        # Window stats
        win["rows_raw"] += n_in
        win["rows_clean"] += n_clean
        win["train"] += n_train
        win["test"] += n_test
        win["val"] += n_val
        win["batches"] += 1

        # Status line (single line)
        dt_all = max(1e-9, time.time() - t0_all)
        _status(
            f"[split] batches={nonempty_batches:,} rows={seen_total:,}/{max_total_rows if max_total_rows else '∞'} "
            f"train={cum['train']:,} test={cum['test']:,} val={cum['val']:,} "
            f"({seen_total/dt_all:,.0f} rows/s)"
        )

        # Aggregated log every N batches
        if (win["batches"] % log_every) == 0:
            dt = max(1e-9, time.time() - t0_win)
            logger.info(
                "Progress: +%d batches | +%d rows (clean) in %.2fs (%.0f rows/s) | "
                "train=%d test=%d val=%d | total_clean=%d",
                int(win["batches"]),
                int(win["rows_clean"]),
                dt,
                (int(win["rows_clean"]) / dt),
                int(win["train"]),
                int(win["test"]),
                int(win["val"]),
                int(cum["rows_clean"]),
            )
            win = Counter()
            t0_win = time.time()

        # Build arrays once
        X_all = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False)
        y_all = df[TARGET_COLUMN].to_numpy(dtype=np.int64, copy=False)

        # Emit splits
        for name, mask in (("train", is_train), ("test", is_test), ("val", is_val)):
            if not np.any(mask):
                continue

            X = X_all[mask]
            y = y_all[mask]

            sub_df = None
            if keep_df and name != "train":
                sub_df = df.loc[mask]

            yield name, X, y, sub_df

        if max_total_rows is not None and seen_total >= int(max_total_rows):
            break

    # finish status line cleanly
    if status_line:
        print("", flush=True)

    dt_all = max(1e-9, time.time() - t0_all)
    logger.info(
        "stream_splits finished: nonempty_batches=%d raw_rows=%d clean_rows=%d "
        "train=%d test=%d val=%d elapsed=%.2fs (%.0f clean_rows/s)",
        int(cum["batches_nonempty"]),
        int(cum["rows_raw"]),
        int(cum["rows_clean"]),
        int(cum["train"]),
        int(cum["test"]),
        int(cum["val"]),
        dt_all,
        (int(cum["rows_clean"]) / dt_all),
    )
