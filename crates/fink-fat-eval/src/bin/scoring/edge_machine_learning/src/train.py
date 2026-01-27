# edge_ml/train.py
from __future__ import annotations

import copy
import time
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

from config import (
    ONNX_MODEL_PATH,
    PARQUET_DIR,
    OUT_DIR,
    MODEL_PATH,
    RANDOM_SEED,
    TEST_SIZE,
    VALID_SIZE,
    TREES_PER_BATCH,
    MAX_BATCHES,
    EARLY_STOP_PATIENCE,
    MONITOR_METRIC,
    TARGET_COLUMN,
    GROUP_COLUMN,
)
from data import load_parquet_dataset, prepare_df, make_xy
from model import make_model, list_models
from metrics import eval_classification, eval_hit_at_k
from plotting import save_all_standard_plots
from config import FEATURE_COLUMNS

# -----------------------------------------------------------------------------
# ONNX export + sanity check
# -----------------------------------------------------------------------------


def export_model_to_onnx(model, *, n_features: int, out_path) -> None:
    """
    Export a trained classifier to ONNX.

    - XGBoost: uses onnxmltools.convert.convert_xgboost (requires onnxmltools types)
    - scikit-learn: uses skl2onnx.convert_sklearn (uses onnxconverter_common types)

    Install:
        pip install onnx onnxconverter-common skl2onnx onnxmltools
    """
    is_xgb = model.__class__.__module__.startswith("xgboost")

    if is_xgb:
        try:
            from onnxmltools.convert import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType
        except Exception as e:
            raise RuntimeError(
                "XGBoost -> ONNX export requires onnxmltools.\n"
                "Install with:\n"
                "  pip install onnx onnxconverter-common skl2onnx onnxmltools\n"
                f"Original error: {e}"
            )

        initial_types = [("input", FloatTensorType([None, n_features]))]
        onnx_model = convert_xgboost(model, initial_types=initial_types)
    else:
        try:
            from skl2onnx import convert_sklearn
            from onnxconverter_common.data_types import FloatTensorType
        except Exception as e:
            raise RuntimeError(
                "scikit-learn -> ONNX export requires skl2onnx + onnxconverter_common.\n"
                "Install with:\n"
                "  pip install onnx onnxconverter-common skl2onnx\n"
                f"Original error: {e}"
            )

        initial_types = [("input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_types)

    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())


def _extract_onnx_positive_proba(outputs) -> np.ndarray:
    """
    Extract the positive-class probability from ONNX Runtime outputs.
    Tries to handle the common output formats produced by tree converters.
    """
    float_arrays = [
        o for o in outputs if isinstance(o, np.ndarray) and o.dtype.kind == "f"
    ]
    if float_arrays:
        for arr in float_arrays:
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, 1]
        return float_arrays[-1].reshape(-1)

    raise RuntimeError(
        "Could not extract probabilities from ONNX outputs. "
        f"Got outputs types: {[type(o) for o in outputs]}"
    )


def sanity_check_onnx(
    model,
    onnx_path,
    X: np.ndarray,
    *,
    n_samples: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Validate ONNX export by comparing probabilities with the reference model.
    """
    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError(
            "onnxruntime is required for ONNX sanity check.\n"
            "Install with:\n"
            "  pip install onnxruntime\n"
            f"Original error: {e}"
        )

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    m = min(int(n_samples), n)

    idx = rng.choice(n, size=m, replace=False) if m < n else np.arange(n)
    Xs = X[idx].astype(np.float32, copy=False)

    p_ref = model.predict_proba(Xs)[:, 1].astype(np.float64, copy=False)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: Xs})
    p_onnx = _extract_onnx_positive_proba(outputs).astype(np.float64, copy=False)

    if p_onnx.shape[0] != p_ref.shape[0]:
        raise RuntimeError(
            f"ONNX output shape mismatch: p_onnx={p_onnx.shape}, p_ref={p_ref.shape}"
        )

    diff = np.abs(p_ref - p_onnx)
    return {
        "n_checked": int(m),
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "p99_abs_diff": float(np.quantile(diff, 0.99)),
        "corr": float(np.corrcoef(p_ref, p_onnx)[0, 1]) if m > 1 else float("nan"),
    }


# -----------------------------------------------------------------------------
# Splitting
# -----------------------------------------------------------------------------


def split_train_test_val_indices(
    df, *, test_size: float, val_size: float, random_state: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataframe indices into train, test and validation.

    - Group-aware split if GROUP_COLUMN exists (groups stay in one split).
    - Otherwise, random row split.

    Returns
    -------
    train_idx, test_idx, val_idx
    """
    rng = np.random.default_rng(random_state)
    n = len(df)

    if GROUP_COLUMN in df.columns:
        groups = df[GROUP_COLUMN].to_numpy()
        uniq = np.unique(groups)
        rng.shuffle(uniq)
        n_groups = len(uniq)

        n_test_groups = max(1, int(n_groups * test_size))
        n_val_groups = max(1, int(n_groups * val_size)) if val_size > 0 else 0
        if n_test_groups + n_val_groups > n_groups:
            n_val_groups = max(0, n_groups - n_test_groups)

        test_groups = set(uniq[:n_test_groups])
        val_groups = set(uniq[n_test_groups : n_test_groups + n_val_groups])

        is_test = np.array([g in test_groups for g in groups], dtype=bool)
        is_val = np.array([g in val_groups for g in groups], dtype=bool)
        is_train = ~(is_test | is_val)

        test_idx = np.where(is_test)[0]
        val_idx = np.where(is_val)[0]
        train_idx = np.where(is_train)[0]
        return train_idx, test_idx, val_idx

    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = max(1, int(n * test_size))
    n_val = max(1, int(n * val_size)) if val_size > 0 else 0
    if n_test + n_val > n:
        n_val = max(0, n - n_test)

    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return train_idx, test_idx, val_idx


# -----------------------------------------------------------------------------
# Training loop (incremental by tree batches)
# -----------------------------------------------------------------------------


def train_with_batches(model, X_train, y_train, X_test, y_test):
    """
    Train an XGBoost model in batches of trees, monitoring MONITOR_METRIC on X_test.

    Returns
    -------
    best_model : XGBClassifier | None
        Best model snapshot encountered during training (deep copy).
    history : list[dict]
        Per-batch metrics on the test set.
    best_n_trees : int | None
        Total number of trees corresponding to best_model.
    """
    history: List[dict] = []
    best_metric = -np.inf
    best_model = None
    best_batch: Optional[int] = None
    patience_counter = 0

    for batch in range(int(MAX_BATCHES)):
        t0 = time.time()
        print(f"Training batch {batch + 1}...", end=" ")

        if batch == 0:
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        else:
            model.n_estimators = TREES_PER_BATCH
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
                xgb_model=model,
            )

        proba_test = model.predict_proba(X_test)[:, 1]
        report = eval_classification(y_test, proba_test, threshold=0.5)

        entry = {"batch": batch + 1}
        entry.update(report)
        history.append(entry)

        metric_val = report.get(MONITOR_METRIC, float("-inf"))
        print(f"done in {time.time() - t0:.2f}s. {MONITOR_METRIC}: {metric_val:.6f}")

        if metric_val > best_metric:
            best_metric = metric_val
            best_model = copy.deepcopy(model)
            best_batch = batch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(
                    f"Early stopping: no improvement in {EARLY_STOP_PATIENCE} consecutive batches."
                )
                break

    best_n_trees = None if best_batch is None else int(best_batch * TREES_PER_BATCH)
    return best_model, history, best_n_trees


def _refit_clean_xgb_for_onnx(
    *,
    random_state: int,
    n_estimators: int,
    X_refit: np.ndarray,
    y_refit: np.ndarray,
    base_model_params: Optional[dict] = None,
):
    """
    Refit a fresh XGBClassifier in a single .fit() call.

    Why: models trained incrementally with repeated fit(..., xgb_model=...) can
    export to ONNX but crash at runtime in onnxruntime for TreeEnsembleClassifier.
    A one-shot refit avoids that.
    """
    base_model_params = dict(base_model_params or {})
    base_model_params.update(
        {
            "n_estimators": int(n_estimators),
            "eval_metric": "auc",
        }
    )

    m = make_model(
        kind="xgb",
        random_state=random_state,
        model_kwargs=base_model_params,
    )
    m.fit(X_refit, y_refit, verbose=False)
    return m


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    print("\nAvailable models:\n", list_models())

    # limit loading for quick testing; set to None to load the full dataset
    limit = 130

    df = load_parquet_dataset(PARQUET_DIR, limit=limit)
    df = prepare_df(df)

    train_idx, test_idx, val_idx = split_train_test_val_indices(
        df,
        test_size=TEST_SIZE,
        val_size=VALID_SIZE,
        random_state=RANDOM_SEED,
    )

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    X_train, y_train = make_xy(df_train)
    X_test, y_test = make_xy(df_test)
    X_val, y_val = make_xy(df_val)

    print(
        f"Train positives: {y_train.sum()} / {len(y_train)} ({100 * y_train.mean():.2f}%)"
    )
    print(
        f"Test  positives: {y_test.sum()} / {len(y_test)} ({100 * y_test.mean():.2f}%)"
    )
    if len(y_val) > 0:
        print(
            f"Validation positives: {y_val.sum()} / {len(y_val)} ({100 * y_val.mean():.2f}%)"
        )

    # Batch-training model
    model = make_model(
        kind="xgb",
        random_state=RANDOM_SEED,
        model_kwargs={"n_estimators": TREES_PER_BATCH, "eval_metric": "auc"},
    )

    t0 = time.time()
    best_model, history, best_n_trees = train_with_batches(
        model, X_train, y_train, X_test, y_test
    )
    print(f"\nTraining complete in {time.time() - t0:.2f} seconds.\n")

    if best_model is None:
        best_model = model
    if best_n_trees is None:
        best_n_trees = int(TREES_PER_BATCH)

    # Evaluate best incremental model (the one you actually selected)
    datasets = {
        "train": (df_train, X_train, y_train),
        "test": (df_test, X_test, y_test),
    }
    if len(y_val) > 0:
        datasets["validation"] = (df_val, X_val, y_val)

    print("\n=== Final Metrics (best incremental model) ===")
    for name, (df_subset, X_subset, y_subset) in datasets.items():
        proba = best_model.predict_proba(X_subset)[:, 1]
        report = eval_classification(y_subset, proba, threshold=0.5)
        print(f"\n-- {name.upper()} set --")
        print(f"ROC AUC: {report['roc_auc']:.6f}")
        print(f"PR  AUC: {report['pr_auc']:.6f}")
        print(
            f"Confusion: tn={report['tn']} fp={report['fp']} fn={report['fn']} tp={report['tp']}"
        )
        print(report["report"])

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

    # Plots for incremental best model
    t0 = time.time()
    out_plots = OUT_DIR / "plots"
    for prefix, (df_subset, X_subset, y_subset) in datasets.items():
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
        print("=============================\n")
    print(f"\nPlots saved to: {out_plots}")
    print(f"Plotting took {time.time() - t0:.2f} seconds.\n")

    # Save the selected (incremental) best model for Python use/debug
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best (incremental) model to: {MODEL_PATH}")

    # -------------------------------------------------------------------------
    # ONNX FIX: refit a clean one-shot XGB model for runtime compatibility
    # -------------------------------------------------------------------------
    # Use train+test for refit (since test is used for monitoring/selection).
    X_refit = np.vstack([X_train, X_test])
    y_refit = np.concatenate([y_train, y_test])

    print(
        f"\nRefitting a clean XGB model for ONNX export "
        f"(one-shot fit, n_estimators={best_n_trees})..."
    )
    t0 = time.time()

    # NOTE: if you tweak XGB hyperparams in make_model above, replicate them here.
    # For now we keep only n_estimators/eval_metric consistent.
    onnx_ready_model = _refit_clean_xgb_for_onnx(
        random_state=RANDOM_SEED,
        n_estimators=best_n_trees,
        X_refit=X_refit,
        y_refit=y_refit,
        base_model_params={},  # add any fixed hyperparams if needed
    )

    print(f"Refit done in {time.time() - t0:.2f}s.")

    export_model_to_onnx(
        onnx_ready_model, n_features=X_train.shape[1], out_path=ONNX_MODEL_PATH
    )
    print(f"Saved ONNX model to: {ONNX_MODEL_PATH}")

    # Optional sanity-check (compares probabilities vs the ONNX-ready refit model)
    try:
        stats = sanity_check_onnx(
            onnx_ready_model,
            ONNX_MODEL_PATH,
            X_test,
            n_samples=10000,
            seed=RANDOM_SEED,
        )
        print("\n=== ONNX sanity check (vs ONNX-refit model) ===")
        print(f"n_checked     : {stats['n_checked']}")
        print(f"max_abs_diff  : {stats['max_abs_diff']:.6e}")
        print(f"mean_abs_diff : {stats['mean_abs_diff']:.6e}")
        print(f"p99_abs_diff  : {stats['p99_abs_diff']:.6e}")
        print(f"corr          : {stats['corr']:.8f}")
    except Exception as e:
        print(f"[ONNX sanity check skipped] {e}")

    # Print training history summary
    if history:
        print("\n=== Training history (per batch) ===")
        for entry in history:
            batch_no = entry.get("batch", "?")
            m = entry.get(MONITOR_METRIC, float("nan"))
            print(f"Batch {batch_no:>2} -> {MONITOR_METRIC}: {m:.6f}")


if __name__ == "__main__":
    main()
