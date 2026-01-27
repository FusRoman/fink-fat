# edge_ml/config.py

from pathlib import Path

root = Path(__file__).parent.parent.parent.parent.parent.parent

# Dossier contenant les parquet générés par Rust
PARQUET_DIR = root / "edge_features_oracle_dataset"

# Colonnes utilisées comme features
FEATURE_COLUMNS = [
    "dt_days",
    # "dt_days_sq",
    # "inv_dt_days",
    "d2_pos",
    "log_d2_pos",
    "resid_norm",
    "resid_dx",
    "resid_dy",
    "speed_from",
    "speed_to",
    "speed_diff",
    "distance_travelled",
    # "n_obs_from",
    # "n_obs_to",
    # "trace_cov_pos_from",
    # "trace_cov_pos_to",
    # "trace_cov_vel_from",
    # "trace_cov_vel_to",
    "flux_abs_diff",
    "z_flux",
    "flux_std_ratio",
    "band_shared",
    # "has_acc",
]

TARGET_COLUMN = "is_true_edge"

# Colonnes de debug (non utilisées pour l'entraînement)
DEBUG_COLUMNS = [
    "left_nid",
    "right_nid",
    "gap_nights"
]

# Pour éviter les fuites entre train/val, on peut splitter par "from_seed_id"
# (si la colonne existe dans tes parquets). Sinon split aléatoire.
GROUP_COLUMN = "from_seed_id"

RANDOM_SEED = 42
TEST_SIZE = 0.2

# -----------------------------------------------------------------------------
# Additional training settings
# -----------------------------------------------------------------------------
# Fraction of the data to reserve for the final validation set.  This set is used
# for a final unbiased evaluation after the model has been trained using the
# train and test (early‑stopping) splits.  The remaining portion of the data
# (after removing the test and validation fractions) will be used for
# training.  Set to 0 to disable the validation split.
VALID_SIZE = 0.1

# Number of boosting rounds (trees) to add to the XGBoost model at each
# training batch.  During training we repeatedly add TREES_PER_BATCH new
# trees and evaluate the model on the test set; if the monitored metric
# stops improving we can stop early.  See train.py for details.
TREES_PER_BATCH = 50

# Maximum number of training batches.  The total number of trees trained can
# be up to TREES_PER_BATCH * MAX_BATCHES.  Training will stop earlier if
# early stopping criteria are met.
MAX_BATCHES = 20

# Number of consecutive batches with no improvement on the monitored metric
# before stopping training early.  A larger patience allows the model to
# continue training even if metrics temporarily plateau.
EARLY_STOP_PATIENCE = 5

# Name of the metric to monitor during training for early stopping.  This
# should correspond to a key in the dict returned by metrics.eval_classification
# (e.g., "roc_auc" or "pr_auc").  The model will be retained whenever this
# metric improves on the test set.
MONITOR_METRIC = "pr_auc"

# Sorties
OUT_DIR = root / "ml_out"
MODEL_PATH = OUT_DIR / "edge_classifier.joblib"

# Optional: export the trained model to ONNX for deployment.
# Requires installing ONNX conversion dependencies (see train.py).
ONNX_MODEL_PATH = OUT_DIR / "edge_classifier.onnx"