# edge_ml/config.py
from pathlib import Path

root = Path(__file__).parent.parent.parent.parent.parent.parent

# Dossier contenant les parquet générés par Rust
PARQUET_DIR = root / "edge_features_dataset"

# Colonnes utilisées comme features
FEATURE_COLUMNS = [
    "position.chi2_pos",
    "position.log_chi2_pos",
    "position.z_dx",
    "position.z_dy",
    "position.z_resid_norm",
    "position.z_along",
    "position.z_cross",
    "position.chol_z1",
    "position.chol_z2",
    "position.chol_z_norm",
    "velocity.cos_dtheta_v",
    "velocity.rel_speed_diff",
    "velocity.innov_speed_ratio",
    # "uncertainty.cov_pos_ratio",
    "uncertainty.cov_vel_ratio",
    # "uncertainty.anisotropy_pos_from",
    # "uncertainty.anisotropy_pos_to",
    # "photometry.flux_abs_diff",
    "photometry.z_flux",
    "photometry.flux_std_ratio",
    "photometry.band_shared",
    # "model.has_acc",
]

TARGET_COLUMN = "is_true_edge"

# Colonnes de debug (non utilisées pour l'entraînement)
DEBUG_COLUMNS = ["left_nid", "right_nid", "gap_nights"]

# Split “group-aware” si présent
GROUP_COLUMN = "from_seed_id"

RANDOM_SEED = 42
TEST_SIZE = 0.2
VALID_SIZE = 0.2

# Max total rows to read from the whole dataset (train+test+val combined).
# None means "no limit".
MAX_TOTAL_ROWS: int | None = 100_000_000

# -----------------------------------------------------------------------------
# Streaming / memory settings
# -----------------------------------------------------------------------------
# Nombre de lignes par batch lu depuis les parquets (scan streaming).
STREAM_BATCH_ROWS = 30_000_000

# Taille max des échantillons en RAM pour l'évaluation (early stopping + plots).
# Ajuste selon ta RAM (ex: 300k ~ confortable, 1M ~ plus lourd).
EVAL_TEST_MAX_ROWS = 1_000_000
EVAL_VALID_MAX_ROWS = 1_000_000

# Conserver aussi un df (features+labels+group+debug) pour hit@k/plots
# (si False, on ne fait que X/y en numpy et on skip hit@k).
KEEP_EVAL_DATAFRAMES = True

# -----------------------------------------------------------------------------
# XGBoost incremental training settings
# -----------------------------------------------------------------------------
# Nombre d'arbres ajoutés par "step" (un step = un batch train lu).
TREES_PER_CHUNK = 5

# Nombre max de passes complètes sur le dataset (epochs).
MAX_EPOCHS = 3

# -----------------------------------------------------------------------------
# Streaming buffer for training (avoids single-class chunks)
# -----------------------------------------------------------------------------
# We accumulate training batches until we have enough rows AND both classes.
TRAIN_BUFFER_MIN_ROWS = 50_000

# Hard cap to avoid large memory spikes; buffer is flushed when exceeded.
TRAIN_BUFFER_MAX_ROWS = 400_000

# If we only see one class for too long, flush anyway after this many incoming rows.
# (will be skipped if still single-class, but prevents infinite buffering)
TRAIN_BUFFER_FORCE_FLUSH_ROWS = 800_000

EARLY_STOP_PATIENCE = 5

# Name of the metric to monitor during training for early stopping.  This
# should correspond to a key in the dict returned by metrics.eval_classification
# (e.g., "roc_auc" or "pr_auc").  The model will be retained whenever this
# metric improves on the test set.
MONITOR_METRIC = "logloss"

# Sorties
OUT_DIR = root / "ml_out"
MODEL_PATH = OUT_DIR / "edge_classifier.joblib"

# Optional: export the trained model to ONNX for deployment.
# Requires installing ONNX conversion dependencies (see train.py).
ONNX_MODEL_PATH = OUT_DIR / "edge_classifier.onnx"
