# edge_ml/config.py
from pathlib import Path

# Root of the edge_ml_prediction project (one level above src/).
_PROJECT_ROOT = Path(__file__).parent.parent

# Single parquet file produced by fink-fat-engine.
PARQUET_PATH = _PROJECT_ROOT / "edge_features.parquet"

# Feature columns used for ML training and inspection.
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
    "uncertainty.cov_vel_ratio",
    "photometry.z_mag",
    "photometry.mag_std_ratio",
    "photometry.band_shared",
]

TARGET_COLUMN = "is_true_edge"

# Debug/context columns (not used as features).
DEBUG_COLUMNS = ["left_nid", "right_nid", "gap_nights"]

# Grouping column (one group = one source seed).
GROUP_COLUMN = "from_seed_id"

# Output directory for feature inspection plots.
OUT_DIR = _PROJECT_ROOT / "feature_inspection_plots"
