import pandas as pd
import time as t
import numpy as np
from alert_association.inter_night_associations import night_to_night_association
import astropy.units as u
from pandas.testing import assert_frame_equal
from alert_association.utils import load_data

if __name__ == "__main__":

    df_sso = load_data("Solar System MPC", 0)

    traj_count = df_sso.groupby(["ssnamenr"]).count().reset_index()

    traj_name = traj_count[traj_count["ra"].isin([20])]["ssnamenr"][:5]
    df_sso = df_sso[df_sso["ssnamenr"].isin(traj_name)].sort_values(["ssnamenr"])

    tr_orb_columns = [
        "provisional designation",
        "ref_epoch",
        "a",
        "e",
        "i",
        "long. node",
        "arg. peric",
        "mean anomaly",
        "rms_a",
        "rms_e",
        "rms_i",
        "rms_long. node",
        "rms_arg. peric",
        "rms_mean anomaly",
        "not_updated",
        "trajectory_id",
    ]

    required_columns = ["ra", "dec", "jd", "nid", "fid", "dcmag", "candid"]

    trajectory_df = pd.DataFrame(columns=tr_orb_columns)
    old_observation = pd.DataFrame(columns=["nid"])

    last_nid = np.min(df_sso["nid"])

    for tr_nid in np.unique(df_sso["nid"]):

        new_observation = df_sso[df_sso["nid"] == tr_nid]
        with pd.option_context("mode.chained_assignment", None):
            new_observation[tr_orb_columns] = -1.0
            new_observation["not_updated"] = np.ones(
                len(new_observation), dtype=np.bool_
            )

        next_nid = new_observation["nid"].values[0]

        t_before = t.time()
        trajectory_df, old_observation, report = night_to_night_association(
            trajectory_df,
            old_observation,
            new_observation,
            last_nid,
            next_nid,
            traj_time_window=200,
            obs_time_window=200,
            intra_night_sep_criterion=500 * u.arcsecond,
            sep_criterion=0.5 * u.degree,
            acceleration_criteria=1000,
            mag_criterion_same_fid=5,
            mag_criterion_diff_fid=5,
            orbfit_limit=5,
            angle_criterion=200
        )

        last_nid = next_nid

    path_ci = "alert_association/CI_expected_output.parquet"

    ci_df = pd.read_parquet(path_ci)
    trajectory_df = trajectory_df.drop(["provisional designation"], axis=1)
    assert_frame_equal(trajectory_df, ci_df, check_dtype=False)
