import pandas as pd
import numpy as np
from fink_fat.associations.inter_night_associations import night_to_night_association
import astropy.units as u
from pandas.testing import assert_frame_equal
import fink_fat.test.test_sample as ts
import sys


def ci_function(
    df_sso,
    path_ci,
    traj_time_window=200,
    obs_time_window=200,
    traj_2_points_time_window=200,
    intra_night_sep_criterion=500 * u.arcsecond,
    sep_criterion=0.5 * u.degree,
    mag_criterion_same_fid=5,
    mag_criterion_diff_fid=5,
    orbfit_limit=5,
    angle_criterion=200,
):
    tr_orb_columns = [
        "not_updated",
        "trajectory_id",
    ]

    trajectory_df = pd.DataFrame(columns=tr_orb_columns)
    old_observation = pd.DataFrame(columns=["nid"])

    last_nid = np.min(df_sso["nid"])

    tr_id = 0

    for tr_nid in np.unique(df_sso["nid"]):

        new_observation = df_sso[df_sso["nid"] == tr_nid]
        with pd.option_context("mode.chained_assignment", None):
            new_observation[tr_orb_columns] = -1.0
            new_observation["not_updated"] = np.ones(
                len(new_observation), dtype=np.bool_
            )

        next_nid = new_observation["nid"].values[0]

        trajectory_df, old_observation = night_to_night_association(
            trajectory_df,
            old_observation,
            new_observation,
            tr_id,
            last_nid,
            next_nid,
            traj_time_window=traj_time_window,
            obs_time_window=obs_time_window,
            traj_2_points_time_window=traj_2_points_time_window,
            intra_night_sep_criterion=intra_night_sep_criterion,
            sep_criterion=sep_criterion,
            mag_criterion_same_fid=mag_criterion_same_fid,
            mag_criterion_diff_fid=mag_criterion_diff_fid,
            orbfit_limit=orbfit_limit,
            angle_criterion=angle_criterion,
        )

        if len(trajectory_df) > 0:
            tr_id = np.max(trajectory_df["trajectory_id"]) + 1

        last_nid = next_nid

    ci_df = pd.read_parquet(path_ci)

    assert_frame_equal(
        trajectory_df.sort_values(["trajectory_id", "jd"]).reset_index(drop=True),
        ci_df.sort_values(["trajectory_id", "jd"]).reset_index(drop=True),
        check_dtype=False,
    )


if __name__ == "__main__":

    mpc = ts.trajectory_sample_2

    ci_function(mpc, "fink_fat/test/CI_expected_output_2.parquet", orbfit_limit=40)

    df_sso = ts.trajectory_sample
    ci_function(df_sso, "fink_fat/test/CI_expected_output.parquet")

    sys.exit(0)
