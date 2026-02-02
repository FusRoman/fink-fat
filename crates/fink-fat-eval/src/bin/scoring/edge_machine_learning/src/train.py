# edge_ml/train.py
from __future__ import annotations

import copy
import time
from typing import List, Optional, Tuple

import joblib
import numpy as np
import logging

from onnx_helpers import export_model_to_onnx, sanity_check_onnx
from config import (
    ONNX_MODEL_PATH,
    PARQUET_DIR,
    OUT_DIR,
    MODEL_PATH,
    RANDOM_SEED,
    TEST_SIZE,
    VALID_SIZE,
    EARLY_STOP_PATIENCE,
    MONITOR_METRIC,
    TARGET_COLUMN,
    GROUP_COLUMN,
    FEATURE_COLUMNS,
    STREAM_BATCH_ROWS,
    EVAL_TEST_MAX_ROWS,
    EVAL_VALID_MAX_ROWS,
    KEEP_EVAL_DATAFRAMES,
    TREES_PER_CHUNK,
    MAX_EPOCHS,
    MAX_TOTAL_ROWS,
    TRAIN_BUFFER_FORCE_FLUSH_ROWS,
    TRAIN_BUFFER_MAX_ROWS,
    TRAIN_BUFFER_MIN_ROWS,
)
from data import Reservoir, stream_splits
from model import make_model, list_models
from metrics import eval_classification, eval_hit_at_k
from plotting import save_all_standard_plots


# -----------------------------------------------------------------------------
# Streaming eval sample build
# -----------------------------------------------------------------------------
def build_eval_reservoirs() -> Tuple[Reservoir, Reservoir]:
    """
    Single streaming pass to build bounded-memory test/val evaluation sets.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    test_res = Reservoir(max_rows=int(EVAL_TEST_MAX_ROWS), rng=rng)
    val_res = Reservoir(max_rows=int(EVAL_VALID_MAX_ROWS), rng=rng)

    counts = {"train": 0, "test": 0, "val": 0}
    pos = {"train": 0, "test": 0, "val": 0}

    t0 = time.time()
    for split, X, y, df in stream_splits(
        PARQUET_DIR,
        batch_rows=STREAM_BATCH_ROWS,
        test_size=TEST_SIZE,
        val_size=VALID_SIZE,
        random_state=RANDOM_SEED,
        keep_df=KEEP_EVAL_DATAFRAMES,
        max_total_rows=MAX_TOTAL_ROWS,
    ):
        counts[split] += int(len(y))
        pos[split] += int(y.sum())
        if split == "test":
            test_res.add(X, y, df)
        elif split == "val":
            val_res.add(X, y, df)

    print("\n=== Dataset scan summary (streaming) ===")
    for k in ("train", "test", "val"):
        n = counts[k]
        p = pos[k]
        frac = (100.0 * p / n) if n > 0 else 0.0
        print(f"{k:>5}: rows={n:,} positives={p:,} ({frac:.2f}%)")

    print(f"Streaming scan took {time.time() - t0:.2f}s.")
    return test_res, val_res


# -----------------------------------------------------------------------------
# Streaming training loop (XGBoost incremental fit per chunk)
# -----------------------------------------------------------------------------
def train_streaming_xgb(
    model,
    *,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[object, List[dict]]:
    """
    Train incrementally by reading the dataset in streaming batches and adding
    TREES_PER_CHUNK trees per training *buffer* flush.

    Early stopping and best-model selection are done on the VALIDATION set
    (X_val, y_val). The test set must remain untouched until the final eval.

    Parameters
    ----------
    model
        XGBoost classifier (sklearn API).
    X_val : np.ndarray
        Fixed validation features (bounded reservoir).
    y_val : np.ndarray
        Fixed validation labels (bounded reservoir).

    Returns
    -------
    (object, List[dict])
        Best model snapshot (by MONITOR_METRIC on validation set) and history list.
    """
    history: List[dict] = []

    minimize_metrics = {"brier", "logloss"}
    monitor = str(MONITOR_METRIC)

    best_metric = float("inf") if monitor in minimize_metrics else -np.inf
    best_model = None
    patience = 0
    step = 0

    # -------------------------------------------------------------------------
    # Buffer state + helpers
    # -------------------------------------------------------------------------
    X_buf: Optional[np.ndarray] = None
    y_buf: Optional[np.ndarray] = None
    buf_rows = 0
    buf_seen_since_flush = 0

    def _buffer_has_both_classes(y: np.ndarray) -> bool:
        u = np.unique(y)
        return u.size >= 2

    def _append_to_buffer(X_new: np.ndarray, y_new: np.ndarray) -> None:
        nonlocal X_buf, y_buf, buf_rows, buf_seen_since_flush
        if len(y_new) == 0:
            return

        buf_seen_since_flush += int(len(y_new))

        if X_buf is None:
            X_buf = X_new.copy()
            y_buf = y_new.copy()
        else:
            X_buf = np.concatenate([X_buf, X_new], axis=0)
            y_buf = np.concatenate([y_buf, y_new], axis=0)

        buf_rows = int(y_buf.shape[0])

    def _reset_buffer() -> None:
        nonlocal X_buf, y_buf, buf_rows, buf_seen_since_flush
        X_buf, y_buf = None, None
        buf_rows = 0
        buf_seen_since_flush = 0

    def _is_better(metric_val: float, best_val: float) -> bool:
        if monitor in minimize_metrics:
            return metric_val < best_val
        return metric_val > best_val

    def _flush_buffer_if_ready(
        *,
        epoch_idx: int,
        force: bool = False,
    ) -> bool:
        """
        Flush the current training buffer into a training step if ready.

        A flush happens when:
        - buffer has both classes AND buf_rows >= TRAIN_BUFFER_MIN_ROWS, OR
        - force=True, OR
        - buf_rows >= TRAIN_BUFFER_MAX_ROWS, OR
        - buf_seen_since_flush >= TRAIN_BUFFER_FORCE_FLUSH_ROWS.

        If flush is triggered but buffer is still single-class, we skip training
        and reset the buffer (prevents infinite growth).

        Returns True if a training step happened.
        """
        nonlocal step, patience, best_metric, best_model, history
        nonlocal X_buf, y_buf, buf_rows, buf_seen_since_flush

        if X_buf is None or y_buf is None or buf_rows == 0:
            return False

        has_two = _buffer_has_both_classes(y_buf)
        ready = has_two and (buf_rows >= int(TRAIN_BUFFER_MIN_ROWS))
        too_big = buf_rows >= int(TRAIN_BUFFER_MAX_ROWS)
        forced = (
            force
            or too_big
            or (buf_seen_since_flush >= int(TRAIN_BUFFER_FORCE_FLUSH_ROWS))
        )

        if not (ready or forced):
            return False

        if not has_two:
            u = np.unique(y_buf)
            print(
                "buffer_flush skipped "
                f"(single-class y={u.tolist()}, buf_rows={buf_rows}, "
                f"seen_since_flush={buf_seen_since_flush})"
            )
            _reset_buffer()
            return False

        # One incremental training step on buffered data.
        step += 1
        t0 = time.time()

        if step == 1:
            model.fit(X_buf, y_buf, eval_set=[(X_val, y_val)], verbose=False)
        else:
            prev = model.get_booster()
            model.n_estimators = int(TREES_PER_CHUNK)
            model.fit(
                X_buf,
                y_buf,
                eval_set=[(X_val, y_val)],
                verbose=False,
                xgb_model=prev,
            )

        proba_val = model.predict_proba(X_val)[:, 1]
        report = eval_classification(y_val, proba_val, threshold=0.5)

        metric_val = report.get(monitor)
        if metric_val is None or (
            isinstance(metric_val, float) and np.isnan(metric_val)
        ):
            # If metric is missing/NaN, do not consider it an improvement.
            metric_val = float("inf") if monitor in minimize_metrics else float("-inf")

        entry = {
            "epoch": int(epoch_idx) + 1,
            "step": int(step),
            "train_buf_rows": int(buf_rows),
        }
        entry.update(report)
        history.append(entry)

        print(
            f"step={step:>5}  "
            f"train_buf_rows={buf_rows:>9}  "
            f"{monitor}={metric_val:.6f}  "
            f"pr_auc={entry.get('pr_auc', float('nan')):.6f}  "
            f"roc_auc={entry.get('roc_auc', float('nan')):.6f}  "
            f"brier={entry.get('brier', float('nan')):.6f}  "
            f"logloss={entry.get('logloss', float('nan')):.6f}  "
            f"dt={time.time() - t0:.2f}s"
        )

        if _is_better(metric_val, best_metric):
            best_metric = metric_val
            best_model = copy.deepcopy(model)
            patience = 0
        else:
            patience += 1

        _reset_buffer()
        return True

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    for epoch in range(int(MAX_EPOCHS)):
        print(f"\n=== Epoch {epoch + 1}/{MAX_EPOCHS} ===")
        t_epoch = time.time()

        for split, X, y, _df in stream_splits(
            PARQUET_DIR,
            batch_rows=STREAM_BATCH_ROWS,
            test_size=TEST_SIZE,
            val_size=VALID_SIZE,
            random_state=RANDOM_SEED,
            keep_df=False,
            max_total_rows=MAX_TOTAL_ROWS,
        ):
            if split != "train":
                continue

            _append_to_buffer(X, y)

            did_step = _flush_buffer_if_ready(epoch_idx=epoch, force=False)
            if did_step and patience >= EARLY_STOP_PATIENCE:
                print(
                    f"\nEarly stopping: no improvement in {EARLY_STOP_PATIENCE} consecutive steps."
                )
                print(f"Epoch time: {time.time() - t_epoch:.2f}s")
                return (best_model or model), history

        # End of epoch: force a flush for leftovers (train if possible)
        _flush_buffer_if_ready(epoch_idx=epoch, force=True)

        if patience >= EARLY_STOP_PATIENCE:
            print(
                f"\nEarly stopping: no improvement in {EARLY_STOP_PATIENCE} consecutive steps."
            )
            print(f"Epoch time: {time.time() - t_epoch:.2f}s")
            return (best_model or model), history

        print(f"Epoch time: {time.time() - t_epoch:.2f}s")

    return (best_model or model), history


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,  # switch to DEBUG for more details
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    print("\nAvailable models:\n", list_models())

    # Ensure output directory exists early
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Build bounded evaluation sets (test + validation)
    test_res, val_res = build_eval_reservoirs()

    X_test, y_test = test_res.finalize()
    if len(y_test) == 0:
        raise RuntimeError(
            "Empty test evaluation set. Check split parameters / dataset."
        )

    X_val, y_val = val_res.finalize()
    if len(y_val) == 0:
        raise RuntimeError(
            "Empty validation evaluation set. Check split parameters / dataset."
        )

    # 2) Build model (streaming incremental xgb)
    # Recommendation: use logloss if you care about probability calibration.
    model = make_model(
        kind="xgb",
        random_state=RANDOM_SEED,
        model_kwargs={
            "n_estimators": int(TREES_PER_CHUNK),
            "eval_metric": "logloss",  # was "auc"
            "tree_method": "hist",
        },
    )

    # 3) Train streaming (monitor on VALIDATION, not test)
    t0 = time.time()
    best_model, history = train_streaming_xgb(model, X_val=X_val, y_val=y_val)
    print(f"\nTraining complete in {time.time() - t0:.2f} seconds.")

    # 4) Final evaluation on test + validation reservoirs
    datasets = {
        "test": (test_res.finalize_df(), X_test, y_test),
        "validation": (val_res.finalize_df(), X_val, y_val),
    }

    print("\n=== Final Metrics (best streaming model) ===")
    for name, (df_subset, X_subset, y_subset) in datasets.items():
        proba = best_model.predict_proba(X_subset)[:, 1]
        report = eval_classification(y_subset, proba, threshold=0.5)

        print(f"\n-- {name.upper()} set --")
        print(f"ROC AUC : {report.get('roc_auc', float('nan')):.6f}")
        print(f"PR  AUC : {report.get('pr_auc', float('nan')):.6f}")
        print(f"Brier   : {report.get('brier', float('nan')):.6f}")
        print(f"LogLoss : {report.get('logloss', float('nan')):.6f}")
        print(
            f"Confusion: tn={report['tn']} fp={report['fp']} fn={report['fn']} tp={report['tp']}"
        )
        print(report["report"])

        if df_subset is not None:
            df_scored = df_subset.copy()
            df_scored["y_proba"] = proba
            hitk = eval_hit_at_k(
                df_scored,
                group_col=GROUP_COLUMN,
                label_col=TARGET_COLUMN,
                score_col="y_proba",
                ks=(1, 5, 10),
            )
            if hitk.get("hit_at_k") is not None:
                print("Hit@K (ranking-like eval)")
                print(f"n_groups={hitk['n_groups']}")
                for k, v in hitk["hit_at_k"].items():
                    print(f"Hit@{k}: {v:.4f}")
            else:
                print(f"[hit@k skipped] {hitk.get('note')}")
        else:
            print(
                "[hit@k skipped] no eval dataframe retained (KEEP_EVAL_DATAFRAMES=False)"
            )

    # 5) Plots (on the reservoirs)
    out_plots = OUT_DIR / "plots"

    for prefix, (df_subset, X_subset, y_subset) in datasets.items():
        if df_subset is None:
            print(f"Skipping plots for {prefix}: no dataframe sample retained.")
            continue
        print(f"Generating plots for {prefix} set...")
        save_all_standard_plots(
            model=best_model,
            df_test=df_subset,
            X_test=X_subset,
            y_test=y_subset,
            feature_names=FEATURE_COLUMNS,
            out_dir=out_plots,
            prefix=f"{prefix}_",
            group_col=GROUP_COLUMN,
            label_col=TARGET_COLUMN,
        )
        print(f"Plots for {prefix} set done.\n")

    print(f"\nPlots saved to: {out_plots}")

    # 6) Save model
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best model to: {MODEL_PATH}")

    # 7) ONNX export + sanity check (on test reservoir)
    export_model_to_onnx(
        best_model, n_features=X_test.shape[1], out_path=ONNX_MODEL_PATH
    )
    print(f"Saved ONNX model to: {ONNX_MODEL_PATH}")

    try:
        stats = sanity_check_onnx(
            best_model,
            ONNX_MODEL_PATH,
            X_test,
            n_samples=min(50_000, len(y_test)),
            seed=RANDOM_SEED,
        )
        print("\n=== ONNX sanity check ===")
        print(f"n_checked     : {stats['n_checked']}")
        print(f"max_abs_diff  : {stats['max_abs_diff']:.6e}")
        print(f"mean_abs_diff : {stats['mean_abs_diff']:.6e}")
        print(f"p99_abs_diff  : {stats['p99_abs_diff']:.6e}")
        print(f"corr          : {stats['corr']:.8f}")
    except Exception as e:
        print(f"[ONNX sanity check skipped] {e}")

    # 8) Training history summary (light)
    if history:
        print("\n=== Training history (last 10 steps) ===")
        for entry in history[-10:]:
            mv = entry.get(MONITOR_METRIC)
            mv_str = (
                f"{mv:.6f}"
                if isinstance(mv, (int, float)) and not np.isnan(mv)
                else str(mv)
            )
            print(
                f"epoch={entry.get('epoch')} step={entry.get('step')} "
                f"{MONITOR_METRIC}={mv_str} "
                f"pr_auc={entry.get('pr_auc', float('nan')):.6f} "
                f"brier={entry.get('brier', float('nan')):.6f}"
            )


if __name__ == "__main__":
    main()
