import pandas as pd
import time as t
import numpy as np
from alert_association.intra_night_association import intra_night_association
from alert_association.intra_night_association import new_trajectory_id_assignation
from alert_association.intra_night_association import get_n_last_observations_from_trajectories
from alert_association.night_to_night_association import night_to_night_association
import astropy.units as u
from pandas.testing import assert_frame_equal
import sys
from collections import Counter
import pyarrow.parquet as pq
import glob
import os


def load_data(object_class):
    all_df = []

    all_path = sorted(glob.glob(os.path.join('data', 'month=*')))[:-1]

    # load all data
    for path in all_path:
        df_sso = pd.read_pickle(path)
        all_df.append(df_sso)
    
    df_sso = pd.concat(all_df).sort_values(["jd"]).drop_duplicates()

    df_sso = df_sso.drop_duplicates(["candid"])
    df_sso = df_sso[df_sso["fink_class"] == object_class]

    return df_sso


def time_window_management(
    trajectory_df, old_observation, last_nid, nid_next_night, time_window, remove_small_track=True
):
    """
    Management of the old observation and trajectories. Remove the old observation with a nid difference with
    the nid of the next night greater than the time window. Perform the same process for trajectories but take
    the most recent trajectory extremity.

    If a number of night without observation exceeds the time window parameters, keep the observations and trajectories
    from the night before the non observation gap.

    Parameters
    ----------
    trajectory_df : dataframe
        all recorded trajectories
    old_observation : dataframe
        old observation from previous night
    last_nid : integer
        nid of the previous observation night
    nid_next_night : integer
        nid of the incoming night
    time_window : integer
        limit to keep old observation and trajectories
    remove_small_track : boolean
        remove the tracks with less than 2 points if the last associated observation night exceed the time window.

    Return
    ------
    oldest_traj : dataframe
        the trajectories older than the time window
    most_recent_traj : dataframe
        the trajectories with an nid from an extremity observation smaller than the time window
    old_observation : dataframe
        all the old observation with a difference of nid with the next incoming nid smaller than
        the time window

    Examples
    --------
    >>> test_traj = pd.DataFrame({
    ... "candid" : [10, 11, 12, 13, 14, 15],
    ... "nid" : [1, 2, 3, 10, 11, 12],
    ... "jd" : [1, 2, 3, 10, 11, 12],
    ... "trajectory_id" : [1, 1, 1, 2, 2, 2]
    ... })

    >>> test_obs = pd.DataFrame({
    ... "nid" : [1, 4, 11, 12],
    ... "candid" : [16, 17, 18, 19]
    ... })

    >>> (test_old_traj, test_most_recent_traj), test_old_obs = time_window_management(test_traj, test_obs, 12, 17, 3)

    >>> expected_old_traj = pd.DataFrame({
    ... "candid" : [10, 11, 12],
    ... "nid" : [1, 2, 3],
    ... "jd" : [1, 2, 3],
    ... "trajectory_id" : [1, 1, 1]
    ... })

    >>> expected_most_recent_traj = pd.DataFrame({
    ... "candid" : [13, 14, 15],
    ... "nid" : [10, 11, 12],
    ... "jd" : [10, 11, 12],
    ... "trajectory_id" : [2, 2, 2]
    ... })

    >>> expected_old_obs = pd.DataFrame({
    ... "nid" : [12],
    ... "candid" : [19]
    ... })

    >>> assert_frame_equal(expected_old_traj.reset_index(drop=True), test_old_traj.reset_index(drop=True))
    >>> assert_frame_equal(expected_most_recent_traj.reset_index(drop=True), test_most_recent_traj.reset_index(drop=True))
    >>> assert_frame_equal(expected_old_obs.reset_index(drop=True), test_old_obs.reset_index(drop=True))
    """
    most_recent_traj = pd.DataFrame()
    oldest_traj = pd.DataFrame()

    if len(trajectory_df) > 0:
        last_obs_of_all_traj = get_n_last_observations_from_trajectories(
            trajectory_df, 1
        )

        if nid_next_night - last_nid > time_window:
            last_obs_of_all_traj = last_obs_of_all_traj[
                last_obs_of_all_traj["nid"] == last_nid
            ]
            mask_traj = trajectory_df["trajectory_id"].isin(
                last_obs_of_all_traj["trajectory_id"]
            )
            most_recent_traj = trajectory_df[mask_traj]
            oldest_traj = trajectory_df[~mask_traj]
            old_observation = old_observation[old_observation["nid"] == last_nid]

            return (oldest_traj, most_recent_traj), old_observation

        last_obs_of_all_traj["diff_nid"] = nid_next_night - last_obs_of_all_traj["nid"]

        most_recent_last_obs = last_obs_of_all_traj[
            last_obs_of_all_traj["diff_nid"] <= time_window
        ]

        mask_traj = trajectory_df["trajectory_id"].isin(
            most_recent_last_obs["trajectory_id"]
        )

        most_recent_traj = trajectory_df[mask_traj]
        oldest_traj = trajectory_df[~mask_traj]

        if remove_small_track:
            old_test = (
                oldest_traj.groupby(["trajectory_id"])
                .agg({"trajectory_id": list, "ra": list, "candid": len})
                .explode(["trajectory_id", "ra"])
                .reset_index(drop=True)
            )
            old_test = old_test[old_test["candid"] > 2]

            oldest_traj = oldest_traj.reset_index(drop=True)
            oldest_traj = oldest_traj[
                oldest_traj["trajectory_id"].isin(old_test["trajectory_id"])
            ]

    diff_nid_old_observation = nid_next_night - old_observation["nid"]
    old_observation = old_observation[diff_nid_old_observation < time_window]

    return (oldest_traj, most_recent_traj), old_observation


if __name__ == "__main__":
    import doctest

    data_path = "data/month=0"

    df_sso = load_data("Solar System MPC")

    specific_mpc = df_sso[
        (
            df_sso["ssnamenr"].isin(
                [
                    "232351",
                    "73972",
                    "75653",
                    "53317",
                    "1951",
                    "80343",
                    "1196",
                    "23101",
                    "1758",
                ]
            )
        )
    ]

    mpc_plot = (
        specific_mpc.groupby(["ssnamenr"])
        .agg({"ra": list, "dec": list, "candid": len})
        .reset_index()
    )

    all_night = np.unique(specific_mpc["nid"])

    last_nid = all_night[0]
    df_night1 = specific_mpc[specific_mpc["nid"] == last_nid]

    left, right, _ = intra_night_association(df_night1)

    last_trajectory_id = 0
    traj_df = new_trajectory_id_assignation(left, right, last_trajectory_id)
    traj_df = traj_df.reset_index(drop=True)

    if len(traj_df) > 0:  # pragma: no cover
        old_observation = pd.concat(
            [df_night1[~df_night1["candid"].isin(traj_df["candid"])]]
        )

        last_trajectory_id = np.max(traj_df["trajectory_id"].values) + 1
    else:
        old_observation = df_night1

    time_window_limit = 16

    for i in range(1, len(all_night)):
        t_before = t.time()
        new_night = all_night[i]

        df_next_night = specific_mpc[specific_mpc["nid"] == new_night]
        current_night_id = df_next_night["nid"].values[0]

        if len(traj_df) > 0:
            last_trajectory_id = np.max(traj_df["trajectory_id"].values) + 1

        (oldest_traj, most_recent_traj), old_observation = time_window_management(
            traj_df, old_observation, last_nid, current_night_id, time_window_limit
        )

        traj_df, old_observation, report = night_to_night_association(
            most_recent_traj,
            old_observation,
            df_next_night,
            last_trajectory_id,
            intra_night_sep_criterion=10 * u.arcminute,
            sep_criterion=1 * u.degree,
            mag_criterion_same_fid=0.6,
            mag_criterion_diff_fid=0.85,
            angle_criterion=30,
            run_metrics=True,
        )

        traj_df = pd.concat([traj_df, oldest_traj])
        last_nid = current_night_id

    all_alert_not_associated = specific_mpc[
        ~specific_mpc["candid"].isin(traj_df["candid"])
    ]

    def trajectory_metrics(rows):
        ssnamenr = rows["ssnamenr"]
        count = Counter(ssnamenr)

        nb_traj_change = len(count)
        most_commons = count.most_common()

        most_common = most_commons[0][1]
        sum_other = np.sum([el[1] for el in most_commons[1:]])

        traj_precision = sum_other / most_common

        return [nb_traj_change, traj_precision]

    gb_traj = (
        traj_df.groupby(["trajectory_id"])
        .agg(
            {
                "ra": list,
                "dec": list,
                "dcmag": list,
                "fid": list,
                "nid": list,
                "ssnamenr": list,
                "candid": len,
            }
        )
        .reset_index()
    )
    all_counter = np.array(gb_traj.apply(trajectory_metrics, axis=1).tolist())
    nb_traj_change = all_counter[:, 0]
    traj_precision = all_counter[:, 1]

    prec_occur = Counter(traj_precision)
    change_counter = Counter(nb_traj_change)

    try:
        from unittest import TestCase

        expected_output = pq.read_table(
            "alert_association/CI_expected_output.parquet"
        ).to_pandas()

        assert_frame_equal(
            traj_df.reset_index(drop=True), expected_output, check_dtype=False
        )
        assert_frame_equal(
            all_alert_not_associated,
            pd.DataFrame(
                columns=[
                    "ra",
                    "dec",
                    "ssnamenr",
                    "jd",
                    "fid",
                    "nid",
                    "fink_class",
                    "objectId",
                    "candid",
                    "magpsf",
                    "sigmapsf",
                    "magnr",
                    "sigmagnr",
                    "magzpsci",
                    "isdiffpos",
                    "dcmag",
                    "dcmagerr",
                ]
            ),
            check_index_type=False,
            check_dtype=False,
        )
        TestCase().assertEqual(
            len(specific_mpc), len(traj_df), "dataframes size are not equal"
        )

        TestCase().assertEqual(
            prec_occur[0],
            9,
            "some trajectories diverge from real trajectories: {}".format(prec_occur),
        )

        TestCase().assertEqual(
            change_counter[1], 9, "some trajectories overlap: {}".format(change_counter)
        )

        sys.exit(doctest.testmod()[0])
    except AssertionError as e:  # pragma: no cover
        print(e)
        sys.exit(-1)
