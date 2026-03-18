"""
Hyperparameter optimisation for the XGBoost edge classifier using Optuna.

Usage::

    cd crates/fink-fat-eval/src/bin/edge_ml_prediction
    pdm run python src/optimize.py [--n-trials N] [--write-best]

Flags
-----
--n-trials N         Number of Optuna trials (default: 100).
--write-best         Overwrite ``xgb_params.yml`` with the best found parameters.
--study-name S       Name of the Optuna study (default: "edge_xgb").
--storage URL        Optuna storage URL for persistence across runs, e.g.
                     ``sqlite:///optuna.db``.  Defaults to in-memory (no persistence).
--dashboard          Launch the Optuna Dashboard web UI after optimisation.
--port N             Port for the dashboard server (default: 8080).
--hpo-sample-size N  Number of training samples used per trial (default: 300_000).
                     A stratified subsample is drawn once and reused for all
                     trials, keeping the full validation set for fair scoring.
                     Set to 0 to use the full training set (slow).

Dashboard notes
---------------
With ``--dashboard`` the script starts a blocking web server after the
optimisation loop finishes.  Open http://localhost:<port>/ in your browser.

If no ``--storage`` URL is provided, an explicit ``InMemoryStorage`` is used
so the dashboard can read the study.  Pass ``--storage sqlite:///optuna.db``
to make the study resumable *and* launch the dashboard simultaneously.

You can also inspect a previous study without re-running optimisation::

    optuna-dashboard sqlite:///optuna.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import optuna
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import config as C
import data

# ── Search-space definition ───────────────────────────────────────────────────

# Fixed parameters that are not part of the search space (infrastructure /
# determinism settings) are loaded from xgb_params.yml.
_FIXED_KEYS = {"tree_method", "device", "random_state", "n_jobs", "eval_metric"}


def _suggest_params(trial: optuna.trial.Trial) -> dict:
    """Sample one point in the hyperparameter search space."""
    return {
        # Cap at 600 for HPO speed; early stopping will find the right depth.
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }


# ── Objective ─────────────────────────────────────────────────────────────────


def _make_objective(
    X_hpo_train: np.ndarray,
    y_hpo_train: np.ndarray,
    X_hpo_eval: np.ndarray,
    y_hpo_eval: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fixed_params: dict,
    early_stopping_rounds: int,
) -> Callable[[optuna.trial.Trial], float]:
    # class balance computed from the HPO train slice
    n_negative = int((y_hpo_train == 0).sum())
    n_positive = int(y_hpo_train.sum())
    scale_pos_weight = n_negative / max(n_positive, 1)

    def objective(trial: optuna.trial.Trial) -> float:
        params = _suggest_params(trial)
        params.update(fixed_params)
        params["scale_pos_weight"] = scale_pos_weight
        params["early_stopping_rounds"] = early_stopping_rounds
        params["verbosity"] = 0

        model = XGBClassifier(**params)
        # Early stopping monitored on the small HPO eval slice (fast per-round
        # scoring).  The full val set is only touched once after fit() to get
        # unbiased PR-AUC / ROC-AUC metrics.
        model.fit(
            X_hpo_train,
            y_hpo_train,
            eval_set=[(X_hpo_eval, y_hpo_eval)],
            verbose=False,
        )

        y_proba = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, y_proba)
        roc_auc = roc_auc_score(y_val, y_proba)

        trial.set_user_attr("roc_auc", roc_auc)
        trial.set_user_attr("pr_auc", pr_auc)
        trial.set_user_attr("best_iteration", model.best_iteration)

        return pr_auc

    return objective


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna HPO for the XGBoost edge classifier."
    )
    p.add_argument("--n-trials", type=int, default=100, metavar="N")
    p.add_argument(
        "--write-best",
        action="store_true",
        help="Overwrite xgb_params.yml with the best found parameters.",
    )
    p.add_argument("--study-name", default="edge_xgb")
    p.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL (e.g. sqlite:///optuna.db). Enables resumable studies.",
    )
    p.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch the Optuna Dashboard web UI after optimisation (blocking).",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8080,
        metavar="N",
        help="Port for the dashboard server (default: 8080).",
    )
    p.add_argument(
        "--hpo-sample-size",
        type=int,
        default=150_000,
        metavar="N",
        help="Total HPO samples per trial (default: 150_000, split 80/20 for"
        " train/early-stop). Use 0 for full dataset.",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).parent.parent

    # Load base config for fixed params and training settings.
    params_path = project_root / "xgb_params.yml"
    with open(params_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg: dict = cfg["model"]
    train_cfg: dict = cfg["training"]

    fixed_params = {k: v for k, v in model_cfg.items() if k in _FIXED_KEYS}
    early_stopping_rounds: int = train_cfg["early_stopping_rounds"]

    # Load data.
    print(f"Loading {C.PARQUET_PATH} …")
    X, y, _ = data.load_xy(C.PARQUET_PATH, C.FEATURE_COLUMNS, C.TARGET_COLUMN)
    print(
        f"  {len(X):,} samples  |  {int(y.sum()):,} TP  |  {int((y == 0).sum()):,} FP"
    )

    X_train, X_val, y_train, y_val = data.split(
        X,
        y,
        test_size=train_cfg["test_size"],
        random_state=model_cfg["random_state"],
    )
    print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}")

    # Subsample the training set for HPO to keep each trial fast.
    # The subsample is split 80/20: train slice (fit) + eval slice (early stopping).
    # The full validation set is only used once per trial for unbiased scoring.
    hpo_n = args.hpo_sample_size
    if hpo_n > 0 and hpo_n < len(X_train):
        _, X_hpo, _, y_hpo = train_test_split(
            X_train,
            y_train,
            test_size=hpo_n,
            stratify=y_train,
            random_state=model_cfg["random_state"],
        )
        X_hpo_train, X_hpo_eval, y_hpo_train, y_hpo_eval = train_test_split(
            X_hpo,
            y_hpo,
            test_size=0.2,
            stratify=y_hpo,
            random_state=model_cfg["random_state"],
        )
        print(
            f"  HPO subsample: {len(X_hpo):,} total"
            f" ({len(X_hpo_train):,} train / {len(X_hpo_eval):,} early-stop eval)"
        )
    else:
        X_hpo_train, X_hpo_eval, y_hpo_train, y_hpo_eval = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            stratify=y_train,
            random_state=model_cfg["random_state"],
        )
        print("  HPO subsample: full train set (--hpo-sample-size 0 or >= train size)")
    print()

    # Silence Optuna's per-trial logs; only show study-level progress.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=model_cfg["random_state"])
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)

    # Use an explicit storage object when the dashboard is requested without a
    # persistent URL, so the dashboard can read the in-memory study.
    if args.storage is not None:
        storage: optuna.storages.BaseStorage | str = args.storage
    elif args.dashboard:
        storage = optuna.storages.InMemoryStorage()
    else:
        storage = None  # type: ignore[assignment]

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    objective = _make_objective(
        X_hpo_train, y_hpo_train, X_hpo_eval, y_hpo_eval,
        X_val, y_val, fixed_params, early_stopping_rounds,
    )

    print(f"Running {args.n_trials} Optuna trials (study: '{args.study_name}') …")
    study.optimize(
        objective,
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\nBest trial #{best.number}")
    print(f"  PR-AUC  : {best.value:.4f}")
    print(f"  ROC-AUC : {best.user_attrs.get('roc_auc', float('nan')):.4f}")
    print(f"  best_iteration: {best.user_attrs.get('best_iteration', 'n/a')}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    if args.write_best:
        # Merge best params back into the model section of the config, keeping
        # fixed keys and overriding everything else with the optimal values.
        new_model_cfg = dict(fixed_params)
        new_model_cfg.update(best.params)
        # Preserve keys that are not in the search space but were in the original.
        for k, v in model_cfg.items():
            if k not in new_model_cfg:
                new_model_cfg[k] = v
        cfg["model"] = new_model_cfg

        with open(params_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"\nxgb_params.yml updated with best parameters → {params_path}")

    if args.dashboard:
        from optuna_dashboard import run_server

        host = "127.0.0.1"
        print(f"\nStarting Optuna Dashboard on http://{host}:{args.port}/")
        print("Hit Ctrl-C to quit.")
        run_server(storage, host=host, port=args.port)


if __name__ == "__main__":
    main()
