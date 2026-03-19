"""
Train an XGBoost classifier on the edge features dataset.

Usage::

    cd crates/fink-fat-eval/src/bin/edge_ml_prediction
    pdm run python src/train.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml

import config as C
import data
import train_xgb
from model_evaluation import plotting as EP

# ── Load hyperparameters ──────────────────────────────────────────────────────

params_path = Path(__file__).parent.parent / "xgb_params.yml"
with open(params_path) as f:
    cfg = yaml.safe_load(f)

model_params = cfg["model"]
train_cfg = cfg["training"]

# ── Load and split data ───────────────────────────────────────────────────────

print(f"Loading {C.PARQUET_PATH} …")
X, y, feature_names = data.load_xy(C.PARQUET_PATH, C.FEATURE_COLUMNS, C.TARGET_COLUMN)
print(f"  {len(X):,} samples  |  {int(y.sum()):,} TP  |  {int((y == 0).sum()):,} FP")

X_train, X_test, y_train, y_test = data.split(
    X, y,
    test_size=train_cfg["test_size"],
    random_state=model_params["random_state"],
)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

# ── Build and train ───────────────────────────────────────────────────────────

model = train_xgb.build_model(
    model_params,
    n_negative=int((y_train == 0).sum()),
    n_positive=int(y_train.sum()),
    early_stopping_rounds=train_cfg["early_stopping_rounds"],
)
model = train_xgb.train(model, X_train, y_train, X_test, y_test)

# ── Evaluate ──────────────────────────────────────────────────────────────────

metrics = train_xgb.evaluate(model, X_test, y_test)
print("\nTest-set metrics:")
for name, value in metrics.items():
    print(f"  {name}: {value:.4f}")

# ── Save model ────────────────────────────────────────────────────────────────

model_path = Path(__file__).parent.parent / train_cfg["model_output"]
train_xgb.save_model(model, model_path)

# ── Export to ONNX ────────────────────────────────────────────────────────────

onnx_path = Path(__file__).parent.parent / train_cfg["onnx_output"]
print(f"\nExporting ONNX model → {onnx_path}")
train_xgb.export_onnx(model, onnx_path)

# ── Evaluation plots ──────────────────────────────────────────────────────────

eval_plots_dir = Path(__file__).parent.parent / train_cfg["eval_plots_dir"]
print(f"\nGenerating evaluation plots → {eval_plots_dir}")
EP.plot_all(model, X_test, y_test, feature_names, eval_plots_dir)
